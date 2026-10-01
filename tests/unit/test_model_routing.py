from dataclasses import replace
from types import SimpleNamespace
import copy
import pytest
from llm_local_rl.config import TrainRunConfig, RolloutConfig
from llm_local_rl.driver import TrainingDriver
from llm_local_rl.model_routing import RoutedSampler, RoutedTrainer
from llm_local_rl.types import TrainExample, SamplingRequest, SamplingResult
from test_shadow_judge import label_batch


def hetero_config(**changes):
    values = dict(model_path="actor", output_dir="out", judge_model_path="judge-model",
                  adapter_layout="split", debate_judge_adapter="judge", train_judge=True,
                  rollout=RolloutConfig(mode="debate"), gradient_checkpointing=False,
                  lora_rank=2, learning_rate=.01, trace_model_io=False,
                  debate_judge_harness="constitution_single_token_v1",
                  debate_judge_bidirectional=True, debate_judge_constrain_single_token=True,
                  debate_judge_score_mode="order_sym_soft_logit",
                  judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1")
    values.update(changes)
    return TrainRunConfig(**values)


def test_config_roundtrip_and_old_fingerprint():
    from llm_local_rl.checkpointing import config_fingerprint
    cfg = hetero_config()
    assert TrainRunConfig.from_dict(cfg.to_dict()) == cfg
    old = dict(model_path="actor", output_dir="out")
    neutral = dict(old, judge_model_path=None, judge_tokenizer_path=None,
                   judge_sampler_gpu_memory_utilization=.25)
    assert config_fingerprint(old) == config_fingerprint(neutral)
    assert config_fingerprint(dict(neutral, judge_model_path="other")) != config_fingerprint(old)


@pytest.mark.parametrize("change", [dict(sampler_backend="sglang"), dict(trace_model_io=True),
    dict(sampler_sleep_level=2), dict(sampler_teardown_before_training=True),
    dict(debate_mock_judge_seed=2), dict(debate_judge_adapter="debate"),
    dict(judge_sampler_gpu_memory_utilization=0)])
def test_unsupported_routes_fail_before_model_construction(change):
    with pytest.raises(ValueError):
        hetero_config(**change)


@pytest.fixture
def native_driver(tmp_path, monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from llm_local_rl.trainer import MultiAdapterTrainer
    torch.set_num_threads(1)
    paths = {}
    for role, hidden, vocab in (("actor",16,64),("judge",8,16)):
        path = tmp_path/role
        torch.manual_seed(7)
        LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=hidden,
            intermediate_size=hidden*2, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, attention_dropout=0.)).save_pretrained(path)
        paths[role] = str(path)
    seen = []
    def tokenizer(**kwargs):
        seen.append(kwargs["base_model_path"])
        return SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(MultiAdapterTrainer, "_load_tokenizer", staticmethod(tokenizer))
    driver = object.__new__(TrainingDriver)
    driver.output_dir = tmp_path/"run"; driver.output_dir.mkdir()
    driver.config = hetero_config(model_path=paths["actor"], judge_model_path=paths["judge"])
    driver.trainer = driver._make_fresh_trainer()
    return driver, seen


def weights(trainer):
    return {n:p.detach().clone() for n,p in trainer.model.named_parameters()}


def unchanged(trainer, before):
    import torch
    return all(torch.equal(before[n],p) for n,p in trainer.model.named_parameters())


@pytest.mark.parametrize("objective", ["supervised_label_ce_js", "unsupervised_js", "grpo"])
def test_same_driver_routes_warmup_and_coherence_without_actor_mutation(native_driver, objective):
    import torch
    driver, seen = native_driver
    routed = driver.trainer
    assert isinstance(routed, RoutedTrainer)
    assert routed.actor.model.config.hidden_size == 16
    assert routed.judge.model.config.hidden_size == 8
    assert seen == [driver.config.model_path, driver.config.judge_model_path]
    actor_before = weights(routed.actor); judge_before = weights(routed.judge)
    batch = label_batch()
    if objective == "grpo":
        # Normal GRPO is the existing PPO trainer with grouped advantages.
        batch = [replace(x, behavior_logprob_mask=x.loss_mask,
                         advantages=[0., 1. if i==0 else -1.], metadata={})
                 for i,x in enumerate(batch)]
        old = routed.compute_logprobs(adapter_name="judge", batch=batch)
        batch = [replace(x, old_logprobs=lp) for x,lp in zip(batch,old,strict=True)]
    driver.config = replace(driver.config, judge_training_objective=objective)
    metric = driver._train_adapter_batch(adapter_name="judge",batch=batch,step_num=1)
    assert metric["completed_optimizer_updates"] == 1
    assert unchanged(routed.actor,actor_before)
    assert not unchanged(routed.judge,judge_before)
    if objective != "grpo":
        assert metric["training_objective"] == objective
    # Actor PPO uses its own vocabulary and cannot change judge weights.
    judge_after=weights(routed.judge)
    actor_batch=[TrainExample(adapter_name="debate",input_ids=[40,41],target_ids=[41,42],
                 loss_mask=[0,1],old_logprobs=[0.,0.],advantages=[0.,1.])]
    lp=routed.compute_logprobs(adapter_name="debate",batch=actor_batch)
    actor_batch=[replace(actor_batch[0],old_logprobs=lp[0])]
    routed.train_batch(adapter_name="debate",batch=actor_batch)
    assert not unchanged(routed.actor,actor_before)
    assert unchanged(routed.judge,judge_after)


def assert_tree_equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: assert_tree_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right,strict=True): assert_tree_equal(a,b)
    else: assert left==right


def test_both_optimizers_and_next_update_resume_exactly(native_driver):
    import torch
    from llm_local_rl.checkpointing import save_exact_resume_checkpoint,load_exact_resume_checkpoint
    driver,_=native_driver; routed=driver.trainer
    routed.train_batch(adapter_name="judge",batch=label_batch(),objective="supervised_label_ce_js")
    actor=[TrainExample(adapter_name="debate",input_ids=[40,41],target_ids=[41,42],
        loss_mask=[0,1],old_logprobs=[0.,0.],advantages=[0.,1.])]
    actor=[replace(actor[0],old_logprobs=routed.compute_logprobs(adapter_name="debate",batch=actor)[0])]
    routed.train_batch(adapter_name="debate",batch=actor)
    routed.actor._microbatch_limits["debate"]=1
    routed.judge._microbatch_limits["judge"]=2
    paths={name:routed.save_adapter(adapter_name=name,output_dir=driver.output_dir/"adapters")
           for name in driver._adapter_names()}
    checkpoint=save_exact_resume_checkpoint(root=driver.output_dir/"exact",step=1,
        run_config=driver.config.to_dict(),adapter_dirs=paths,trainer=routed)
    saved=copy.deepcopy(routed.training_state_dict())
    expected_rng=torch.rand(5)
    expected_actor=weights(routed.actor)
    routed.train_batch(adapter_name="judge",batch=label_batch(),objective="supervised_label_ce_js")
    expected_judge=weights(routed.judge)
    driver.current_adapter_dirs=paths
    restored=driver._make_trainer_from_current_adapters()
    load_exact_resume_checkpoint(path=checkpoint,trainer=restored,run_config=driver.config.to_dict())
    assert_tree_equal(saved,restored.training_state_dict())
    assert torch.equal(torch.rand(5),expected_rng)
    restored.train_batch(adapter_name="judge",batch=label_batch(),objective="supervised_label_ce_js")
    assert unchanged(restored.actor,expected_actor)
    assert unchanged(restored.judge,expected_judge)
    bad=copy.deepcopy(saved);bad["judge_model_path"]="wrong-model"
    with pytest.raises(ValueError,match="backbone mismatch"):
        restored.load_training_state_dict(bad)


class Engine:
    def __init__(self): self.calls=[]
    def wake_up(self): self.calls.append("wake")
    def sleep(self,level=1): self.calls.append(("sleep",level))
    def sample_many(self,requests): self.calls.append(requests); return requests
    def set_adapter_paths(self,**kw): self.calls.append(kw)
    def unload_adapters(self,**kw): self.calls.append(kw)
    def close(self): self.calls.append("close")


def request(name):
    return SamplingRequest(adapter_name=name,prompt_token_ids=[2],stop_token_ids=[1],max_tokens=1,temperature=1.)


def test_sampler_switches_sleeping_backbones_and_rejects_mixed_batch():
    actor,judge=Engine(),Engine(); sampler=RoutedSampler(actor=actor,judge=judge)
    sampler.sample_many([request("debate")]);sampler.sample_many([request("judge")])
    assert actor.calls[-1]==("sleep",1)
    assert judge.calls[0]=="wake"
    with pytest.raises(ValueError,match="one backbone"):
        sampler.sample_many([request("judge"),request("debate")])
    sampler.set_adapter_paths(adapter_paths={"debate":"d","judge":"j"})
    assert actor.calls[-1]=={"adapter_paths":{"debate":"d"}}
    assert judge.calls[-1]=={"adapter_paths":{"judge":"j"}}
    sampler.sleep();sampler.close()
    assert judge.calls[-2:]==[("sleep",1),"close"]
