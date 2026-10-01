from dataclasses import replace
from types import SimpleNamespace
import copy
from pathlib import Path
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
                  judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1",
                  judge_training_objective="supervised_label_ce_js",
                  debate_r1_reward="none", debate_r23_reward="soft_judge_raw")
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
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    paths = {}
    for role, hidden, vocab in (("actor",16,64),("judge",8,16)):
        path = tmp_path/role
        torch.manual_seed(7)
        LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=hidden,
            intermediate_size=hidden*2, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, attention_dropout=0.)).save_pretrained(path)
        mapping={f"t{i}":i for i in range(vocab)}
        eos=2 if role=="actor" else 3
        specials={0:"<pad>",1:"<unk>",eos:"<eos>",40 if role=="actor" else 5:"A",41 if role=="actor" else 6:"B"}
        for index,text in specials.items():
            del mapping[f"t{index}"]; mapping[text]=index
        backend=BackendTokenizer(WordLevel(mapping,unk_token="<unk>"))
        backend.pre_tokenizer=Whitespace()
        token=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token="<pad>",
                                     unk_token="<unk>",eos_token="<eos>")
        token.chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}"
        token.save_pretrained(path)
        paths[role] = str(path)
    seen = []
    real_load = MultiAdapterTrainer._load_tokenizer
    def tokenizer(**kwargs):
        seen.append(kwargs["base_model_path"])
        return real_load(**kwargs)
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
    changes = dict(judge_training_objective=objective)
    if objective == "grpo":
        changes["debate_judge_score_mode"] = "hard_verdict"
        changes["debate_r23_reward"] = "constant"
        changes["judge_label_token_contract"] = "none"
    driver.config = replace(driver.config, **changes)
    metric = driver._train_adapter_batch(adapter_name="judge",batch=batch,step_num=1)
    assert metric["num_optimizer_steps"] == 1
    assert unchanged(routed.actor,actor_before)
    assert not unchanged(routed.judge,judge_before)
    if objective != "grpo":
        assert metric["training_objective"] == objective
    # Actor PPO uses its own vocabulary and cannot change judge weights.
    judge_after=weights(routed.judge)
    actor_batch=[TrainExample(adapter_name="debate",input_ids=[40,41],target_ids=[41,42],
                 loss_mask=[0,1],behavior_logprob_mask=[0,1],old_logprobs=[0.,0.],advantages=[0.,1.])]
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
        loss_mask=[0,1],behavior_logprob_mask=[0,1],old_logprobs=[0.,0.],advantages=[0.,1.])]
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
    actor,judge=Engine(),Engine(); sampler=RoutedSampler(actor=actor,judge=judge,trainable_adapter_names={"debate","judge"})
    sampler.sample_many([request("debate")]);sampler.sample_many([request("judge")])
    assert actor.calls[-1]==("sleep",1)
    assert judge.calls[0]=="wake"
    with pytest.raises(ValueError,match="one backbone"):
        sampler.sample_many([request("judge"),request("debate")])
    sampler.set_adapter_paths(adapter_paths={"debate":"d","judge":"j"})
    assert actor.calls[-1]=={"adapter_paths":{"debate":"d"}}
    assert judge.calls[-1]=={"adapter_paths":{"judge":"j"}}
    sampler.sleep();sampler.close()
    assert judge.calls[-3:]==[{"adapter_names":{"judge"}},("sleep",1),"close"]


class Tokenizer:
    def __init__(self,offset):
        self.offset=offset;self.eos_token="EOS";self.all_special_tokens=[]
        self.name_or_path=str(offset)
    def encode(self,text,add_special_tokens=False):
        return [self.offset+{"A":1," A":1,"B":2," B":2,"EOS":3}.get(text,4)]
    def decode(self,tokens,**kw):
        return {self.offset+1:"A",self.offset+2:"B"}.get(tokens[0],"prompt")
    def apply_chat_template(self,messages,**kwargs):
        return "messages"


def test_judge_prompt_label_and_eos_ids_use_native_vocabulary(monkeypatch):
    from llm_local_rl.debate_runtime import DebateRuntime,DebateRuntimeConfig
    from llm_local_rl.judge_harness import JudgeTranscript,AgentDebateText
    actor,judge=Tokenizer(10),Tokenizer(100)
    runtime=object.__new__(DebateRuntime)
    runtime.tokenizer=actor;runtime.judge_tokenizer=judge
    runtime.runtime_config=DebateRuntimeConfig(judge_adapter="judge", judge_harness_id="constitution_single_token_v1",
        judge_constrain_single_token=True,judge_score_mode="hard_verdict")
    runtime.debate_config=SimpleNamespace(system_judge="unused")
    transcript=JudgeTranscript("question","rule",AgentDebateText("a","b","c"),AgentDebateText("d","e","f"))
    prompt=runtime._encode_judge_transcript(transcript)
    assert prompt==[104]
    assert set(runtime._judge_allowed_token_ids())=={101,102}
    class Recorder:
        def sample_many(self,requests):
            assert requests[0].prompt_token_ids==[104]
            assert requests[0].stop_token_ids==[103]
            assert set(requests[0].allowed_token_ids)=={101,102}
            return [SamplingResult(adapter_name="judge",prompt_token_ids=[104],
                completion_token_ids=[101],completion_logprobs=[-.5],text="A")]
    runtime.judge_sampler=Recorder()
    runtime._sample_judge_many(prompt_tokens_list=[prompt],round_num=99,step_seed=1,
        stop_token_ids=[103],max_tokens=1,temperature=1.)


def test_failed_judge_engine_start_closes_actor(native_driver,monkeypatch):
    driver,_=native_driver
    driver.current_adapter_dirs={name:name for name in driver._adapter_names()}
    actor=Engine();calls=[]
    def make(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:return actor
        raise RuntimeError("judge creation failed")
    monkeypatch.setattr(driver,"_make_vllm_sampler",make)
    with pytest.raises(RuntimeError,match="judge creation failed"):
        driver._make_sampler()
    assert actor.calls==[("sleep",1),"close"]
    assert calls[0]["model_path"]==driver.config.model_path
    assert calls[1]["model_path"]==driver.config.judge_model_path


class StrictEngine(Engine):
    def __init__(self,trainable):
        super().__init__();self.asleep=True;self.loaded=set();self.trainable=set(trainable)
    def wake_up(self):
        super().wake_up();self.asleep=False
    def sleep(self,level=1):
        assert not (self.loaded & self.trainable), "Mutable LoRA buffers must be evicted before sleep"
        super().sleep(level);self.asleep=True
    def unload_adapters(self,*,adapter_names):
        assert not self.asleep, "Never mutate unmapped buffers of a sleeping engine"
        super().unload_adapters(adapter_names=adapter_names)
        self.loaded.difference_update(adapter_names)
    def sample_many(self,requests):
        assert not self.asleep
        self.loaded.update(x.adapter_name for x in requests)
        return super().sample_many(requests)


def test_mutable_loras_evict_before_sleep_and_remain_evicted_across_path_changes():
    actor,judge=StrictEngine({"debate"}),StrictEngine(set())
    sampler=RoutedSampler(actor=actor,judge=judge,trainable_adapter_names={"debate"})
    sampler.sample_many([request("debate")]);sampler.sample_many([request("judge")])
    assert actor.asleep and not actor.loaded
    sampler.unload_adapters(adapter_names={"debate"})
    sampler.sleep()
    assert judge.loaded=={"judge"} # A frozen judge survives sleep.
    sampler.set_adapter_paths(adapter_paths={"debate":"new-actor","judge":"frozen-judge"})
    sampler.sample_many([request("debate")]);sampler.sample_many([request("judge")])
    sampler.sleep()
    assert actor.asleep and judge.asleep
    with pytest.raises(ValueError,match="frozen adapter"):
        sampler.unload_adapters(adapter_names={"judge"})


def test_real_fast_tokenizers_load_and_judge_samples_native_ids(native_driver):
    from llm_local_rl.debate_runtime import DebateRuntime,DebateRuntimeConfig
    from llm_local_rl.judge_harness import JudgeTranscript,AgentDebateText
    driver,_=native_driver
    driver.tokenizer=driver._load_tokenizer()
    judge=driver._load_judge_tokenizer()
    assert driver.tokenizer.encode("A")==[40] and judge.encode("A")==[5]
    assert len(driver.tokenizer)==64 and len(judge)==16
    class Recorder:
        def sample_many(self,requests):
            req=requests[0]
            assert max(req.prompt_token_ids)<16
            assert req.stop_token_ids==[3]
            assert set(req.allowed_token_ids)=={5,6}
            return [SamplingResult(adapter_name="judge",prompt_token_ids=req.prompt_token_ids,
                completion_token_ids=[5],completion_logprobs=[-.5],text="A")]
    runtime=DebateRuntime(task=SimpleNamespace(stop_token_ids=lambda tokenizer:[tokenizer.eos_token_id]),
        tokenizer=driver.tokenizer,judge_tokenizer=judge,sampler=Recorder(),
        debate_config=SimpleNamespace(system_judge="rule"),adapter_layout="split",
        runtime_config=DebateRuntimeConfig(judge_adapter="judge",judge_harness_id="constitution_single_token_v1",
            judge_constrain_single_token=True,judge_score_mode="hard_verdict"))
    transcript=JudgeTranscript("question","rule",AgentDebateText("a","b","c"),AgentDebateText("d","e","f"))
    result=runtime._run_llm_judge_transcripts(transcripts=[transcript],step_seed=1)
    assert result[0]==["A"] and result[3]==[[5]]


def test_dashboard_retains_both_backbone_identities():
    from llm_local_rl.dashboard_sync import default_name,metadata
    cfg=hetero_config().to_dict()
    assert "judge=judge-model" in default_name(cfg,"full")
    assert '"judge": "judge-model"' in metadata(cfg,"full")["observability_backbones"]


def test_inprocess_vllm_workers_rejected_before_construction(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING","0")
    with pytest.raises(ValueError,match="separate V1 worker processes"):
        hetero_config()


def test_single_backbone_tokenizer_override_preserves_historical_dispatch(tmp_path,monkeypatch):
    driver=object.__new__(TrainingDriver);driver.output_dir=tmp_path
    driver.config=TrainRunConfig(model_path="actor",output_dir=str(tmp_path),tokenizer_path="legacy-override")
    assert driver._trainer_config(device="cpu").tokenizer_path is None
    driver.current_adapter_dirs={name:name for name in driver._adapter_names()}
    seen=[]
    monkeypatch.setattr(driver,"_make_vllm_sampler",lambda **kw:seen.append(kw))
    driver._make_sampler()
    assert seen[0].get("tokenizer_path") is None


def test_shadow_judge_initialization_is_on_the_smaller_backbone(native_driver):
    import torch
    driver,_=native_driver
    driver.config=replace(driver.config,train_shadow_judge=True,judge_coherence_js_weight=0.,
        shadow_judge_init_seed=17,shadow_judge_init_std=.02)
    routed=driver._make_fresh_trainer()
    assert "judge_shadow" not in routed.actor.config.adapter_names
    assert "judge_shadow" in routed.judge.config.adapter_names
    params=dict(routed.judge.model.named_parameters())
    for name,parameter in params.items():
        if ".lora_B.judge_shadow." in name:
            assert torch.count_nonzero(parameter)>0
    assert (driver.output_dir/"shadow_judge_initialization.json").exists()


@pytest.mark.parametrize("frozen_judge",[False,True])
def test_successful_driver_sampler_factory_defaults_and_frozen_judge(native_driver,monkeypatch,frozen_judge):
    driver,_=native_driver
    if frozen_judge:
        driver.config=replace(driver.config,train_judge=False,judge_training_objective="grpo",
                              debate_judge_temperature=0.)
    driver.current_adapter_dirs={name:name for name in driver._adapter_names()}
    engines=[]
    def make(**kw):
        engine=Engine();engines.append(engine);return engine
    monkeypatch.setattr(driver,"_make_vllm_sampler",make)
    sampler=driver._make_sampler()
    assert len(engines)==2 and sampler.actor is engines[0] and sampler.judge is engines[1]
    assert ("judge" in sampler.trainable_adapter_names) is (not frozen_judge)
    assert "debate" in sampler.trainable_adapter_names
    sampler.sample_many([request("debate")]);sampler.sample_many([request("judge")])
    sampler.sleep();sampler.close()


def test_actual_driver_init_and_resume_keep_native_tokenizers_and_rng(native_driver,monkeypatch):
    import torch
    from llm_local_rl.checkpointing import restore_rng_state,capture_rng_state
    prepared,_=native_driver
    config=replace(prepared.config,output_dir=str(prepared.output_dir/"driver"),
                   resource_logging=False,wandb_enabled=False,reference_kl_every=0)
    def make(self,**kw):
        torch.rand(9) # Engine initialization must not change training RNG.
        return Engine()
    monkeypatch.setattr(TrainingDriver,"_make_vllm_sampler",make)
    fresh=TrainingDriver(config=config)
    try:
        assert fresh.tokenizer.encode("A")==[40] and fresh.judge_tokenizer.encode("A")==[5]
        rng=torch.load(Path(fresh.latest_exact_resume_checkpoint)/"rng_state.pt",weights_only=False)
        state=capture_rng_state()
        assert torch.equal(rng["torch_cpu"],state["torch_cpu"])
        fresh.sampler.close()
        resumed=TrainingDriver.resume(output_dir=config.output_dir)
        try:
            assert isinstance(resumed.trainer,RoutedTrainer)
            assert resumed.start_step==0
            assert resumed.judge_tokenizer.encode("A")==[5]
            assert torch.equal(torch.get_rng_state(),rng["torch_cpu"])
        finally:
            resumed.sampler.close();resumed.resource_monitor.stop();resumed.observability.finish()
    finally:
        fresh.resource_monitor.stop();fresh.observability.finish()
