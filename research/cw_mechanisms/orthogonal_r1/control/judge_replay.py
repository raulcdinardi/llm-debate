from pathlib import Path
import sys,json,time,hashlib,math,fcntl,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source/src'))
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors.torch import load_file
from llm_local_rl.judge_harness import get_judge_harness,harness_fingerprint,JudgeTranscript,AgentDebateText
OUT=R/'outputs'
def save(p,x):p.write_text(json.dumps(x,indent=2))
lease=(OUT/'judge.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.time();torch.set_num_threads(4)
h='constitution_single_token_v1';mf=json.loads((R/'inputs/judge/judge_harness.json').read_text());assert mf['harness_id']==h and harness_fingerprint(h)==mf['harness_fingerprint']
assert hashlib.sha256((R/'inputs/judge/adapter_model.safetensors').read_bytes()).hexdigest()=='9aa0acc28a687a15cb36fde1ec638432bdd087ad3cf168a3b8ac519f196058f9'
tok=AutoTokenizer.from_pretrained(R/'inputs/base_exact');model=AutoModelForCausalLM.from_pretrained(R/'inputs/base_exact',dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
state=load_file(str(R/'inputs/judge/adapter_model.safetensors'));cfg=json.loads((R/'inputs/judge/adapter_config.json').read_text());params=dict(model.named_parameters());assert not cfg.get('use_dora') and not cfg.get('use_rslora');scale=cfg['lora_alpha']/cfg['r']
with torch.no_grad():
 for k,a in state.items():
  if not k.endswith('.lora_A.weight'):continue
  b=state[k.replace('.lora_A.weight','.lora_B.weight')];name=k.removeprefix('base_model.model.').replace('.lora_A.weight','.weight');w=params[name];v=w.float();v.addmm_(b.to('cuda',dtype=torch.float32),a.to('cuda',dtype=torch.float32),alpha=scale);w.copy_(v.to(w.dtype))
del state
rows=[json.loads(l) for l in (R/'inputs/judge_pairs.jsonl').open()];ids=json.loads((R/'inputs/judge_ids.json').read_text());rows=[x for x in rows if x['id'] in ids][:8];results=[]
for x in rows:
 if time.time()-start>1700:raise RuntimeError('GPU_TIME_CAP')
 s=x['sample'];tr=JudgeTranscript(question=s['question'],constitution='Prefer the agent whose story best satisfies the user.',agent_a=AgentDebateText(**{k:s['trajectory_a'][k] for k in ['r1','r2','r3']}),agent_b=AgentDebateText(**{k:s['trajectory_b'][k] for k in ['r1','r2','r3']}));zs=[];details=[]
 for t in [tr,tr.swapped()]:
  text=get_judge_harness(h).render_checked(transcript=t,base_system_text='').raw_text;inputs=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
  with torch.inference_mode():
   logits=model(**inputs,use_cache=False).logits[0,-1].float();z=(torch.logsumexp(logits[[41,334]],0)-torch.logsumexp(logits[[42,378]],0)).item()
  zs.append(z);details.append({'z':z,'prompt_tokens':inputs.input_ids.shape[1],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'label_logits':{str(i):logits[i].item() for i in [41,334,42,378]}})
 score=math.tanh((zs[0]-zs[1])/4);old=s['judge']['raw_response']['soft_score']['score'];res={'id':x['id'],'score':score,'archived_score':old,'difference':score-old,'orders':details};results.append(res)
 with (OUT/'original_replay.jsonl').open('a') as f:f.write(json.dumps(res)+'\n')
 print(x['id'],score,old,flush=True)
save(OUT/'judge_replay_terminal.json',{'status':'DONE_BASELINE_ONLY','n':len(results),'mean_abs_s_difference':sum(abs(x['difference']) for x in results)/len(results),'max_abs_s_difference':max(abs(x['difference']) for x in results),'seconds':time.time()-start,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,'note':'Transformers BF16 merged LoRA versus archived vLLM; not bitwise replay. One manually reviewed original/edited pilot pair; no population inference.'})
