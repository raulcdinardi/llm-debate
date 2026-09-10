from dataclasses import replace
from types import SimpleNamespace
import os
import pytest
from llm_local_rl.qwen35_instruct_format import INSTRUCTION, EMPTY_THINK, continuation_parts, audit_three_points
from llm_local_rl.soft_judge import QWEN35_INSTRUCT_AB_V1, resolve_judge_label_token_contract
from llm_local_rl.judge_harness import QWEN35_CHAT_SINGLE_TOKEN_V1, JudgeTranscript, AgentDebateText
from llm_local_rl.debate_runtime import DebateRuntime
from test_shadow_judge import paired_config


def test_qwen_harness_contract_pair_is_explicit():
    good = paired_config(debate_judge_harness=QWEN35_CHAT_SINGLE_TOKEN_V1,
                         judge_label_token_contract=QWEN35_INSTRUCT_AB_V1,
                         debate_prompt_format='qwen35_instruct_three_points', thinking_mode='no_think')
    for change in [dict(debate_judge_harness='constitution_single_token_v1'),
                   dict(judge_label_token_contract='lfm25_openbookqa_spaced_ab_v1')]:
        with pytest.raises(ValueError):
            replace(good, **change)


def test_word_cap_checks_actual_generated_lines_without_truncation():
    valid = '1) ' + ' '.join(['word'] * 30) + '\n2) Claim two.\n3) Claim three.\nCONCLUDED'
    assert audit_three_points(text=valid, round_num=2)['strict_ok']
    assert not audit_three_points(text=valid.replace('word\n', 'word extra\n'), round_num=2)['strict_ok']
    for malformed in [valid+'\nIntroduction', 'Introduction\n'+valid,
                      valid.replace('2)', '3)'), valid.replace('CONCLUDED', ''),
                      valid.replace('Claim two.', 'Claim\ntwo.')]:
        assert not audit_three_points(text=malformed, round_num=3)['strict_ok']


@pytest.fixture
def qwen_tokenizer():
    path = os.environ.get('QWEN_TOKENIZER_PATH')
    if not path:
        pytest.skip('Set QWEN_TOKENIZER_PATH to pinned local tokenizer for integration checks')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_official_qwen_continuation_and_label_boundary(qwen_tokenizer):
    tok = qwen_tokenizer
    contract = resolve_judge_label_token_contract(tokenizer=tok, contract_name=QWEN35_INSTRUCT_AB_V1)
    assert contract.allowed_token_ids == (32, 33)
    for round_num in (2, 3):
        left, right = continuation_parts(tok, round_num=round_num)
        text = tok.decode(left+tok.encode('Opponent payload', add_special_tokens=False)+right)
        assert text.startswith('<|im_end|>\n<|im_start|>user\n'+INSTRUCTION)
        assert text.endswith('<|im_start|>assistant\n'+EMPTY_THINK)
        assert 'Opponent payload' in text
        assert text.count(INSTRUCTION)==1
    runner = object.__new__(DebateRuntime)
    runner.tokenizer=tok
    runner.runtime_config=SimpleNamespace(judge_harness_id=QWEN35_CHAT_SINGLE_TOKEN_V1,
                                         judge_label_token_contract=QWEN35_INSTRUCT_AB_V1)
    runner.debate_config=SimpleNamespace(system_judge='unused')
    transcript=JudgeTranscript('Which answer?', 'Prefer correctness.', AgentDebateText('one','arg','reb'), AgentDebateText('two','arg2','reb2'))
    ids=runner._encode_judge_transcript(transcript)
    assert tok.decode(ids).endswith(EMPTY_THINK)
    for label, label_id in [('A',32),('B',33)]:
        assert tok.encode(tok.decode(ids)+label,add_special_tokens=False)==ids+[label_id]


def test_tiny_native_qwen_loading_paired_ce_and_selective_head(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from llm_local_rl.trainer import MultiAdapterTrainer, TrainerConfig
    from llm_local_rl.driver import TrainingDriver
    from llm_local_rl.shadow_judge import shadow_label_examples
    from test_shadow_judge import label_batch
    torch.set_num_threads(1)
    config=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=16,
        layer_types=['linear_attention','full_attention'],linear_key_head_dim=16,
        linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,3,3]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                           out_hidden_size=32,num_position_embeddings=16))
    model=Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(MultiAdapterTrainer,'_load_tokenizer',staticmethod(lambda **kw:SimpleNamespace(pad_token_id=0)))
    trainer=MultiAdapterTrainer(config=TrainerConfig(base_model_path=str(tmp_path),
        adapter_names=('solution','debate','judge','judge_shadow'),device='cpu',torch_dtype='float32',
        lora_rank=2,target_modules=('q_proj','v_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj'),
        train_logprob_backend='selective_lm_head',gradient_checkpointing=False,on_policy_logprob_check=True))
    trainer.initialize_shadow_judge(seed=17,std=.02)
    trainer.set_adapter('judge');trainer.model.eval()
    x=dict(input_ids=torch.tensor([[2,3,4]]),attention_mask=torch.ones((1,3),dtype=torch.long))
    with torch.no_grad():
        logits=trainer.model(**x).logits
        with trainer.model.disable_adapter():assert torch.equal(logits,trainer.model(**x).logits)
        hidden=trainer._selective_lm_head_hidden_states(tensors=x)
        projected=trainer._causal_lm_for_selective_lm_head().get_output_embeddings()(hidden)
        torch.testing.assert_close(logits,projected)
    driver=object.__new__(TrainingDriver);driver.config=paired_config();driver.trainer=trainer
    for name,batch in [('judge',label_batch()),('judge_shadow',shadow_label_examples(label_batch()))]:
        before={n:p.detach().clone() for n,p in trainer.model.named_parameters()}
        metrics=driver._train_adapter_batch(adapter_name=name,batch=batch,step_num=1)
        assert metrics['supervised_label_nll'] > 0
        assert any(not torch.equal(before[n],p) for n,p in trainer.model.named_parameters() if f'.{name}.' in n)
        assert all(torch.equal(before[n],p) for n,p in trainer.model.named_parameters() if f'.{name}.' not in n)
