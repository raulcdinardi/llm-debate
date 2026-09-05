from pathlib import Path
import os,sys,json,time,hashlib,copy,argparse,signal
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/src'))
from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask
from llm_local_rl.task_types import TaskInstance
import llm_local_rl.prompts as promptlib
promptlib._PROMPT_DIR=ROOT/'source/prompts'
TASK=ConstrainedWritingDebateTask.from_args(rules_per_speaker=2,reward_scope='both',sides='both',rule_family='generic',reward_mode='additive',letter_temperature=1.,anchors='on')

def seed(*parts):return int(hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()[:8],16)%2147483647

def writej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)

def plain(system,user,prefill=''):
 return (f'System:\n{system}\n\n' if system else '')+f'User:\n{user}\nAssistant:\n{prefill}'

def extension(case,n,opponent):
 e=TASK.build_base_text_debate_extension(inst=TaskInstance(instance_id=case['case_id'],payload=case['payload']),opponent_round=n,opponent_answer=opponent)
 return '\n\n'+plain(e.system_text,e.user_text,e.assistant_prefill),e.assistant_prefill

def r2prompt(c,side):
 other='b' if side=='a' else 'a';ext,pre=extension(c,1,c['r1_'+other])
 return plain(None,c['question'])+c['r1_'+side]+ext,pre

def r3prompt(c,side,r2):
 other='b' if side=='a' else 'a';p,pre=r2prompt(c,side);ext,pre3=extension(c,2,r2[other])
 assert r2[side].startswith(pre)
 return p+r2[side][len(pre):]+ext,pre3

def insert(original,claim):
 index=original.rfind('CONCLUDED')
 if index<0:index=len(original)
 added='\n'+claim+'\n';out=original[:index]+added+original[index:]
 assert out[:index]+out[index+len(added):]==original
 return out,dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),modified_sha256=hashlib.sha256(out.encode()).hexdigest(),span=[index,index+len(added)],claim=claim)

class Engine:
 def __init__(self):
  torch.set_num_threads(4)
  self.tok=AutoTokenizer.from_pretrained(ROOT/'inputs/base_exact',padding_side='left')
  if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
  self.model=AutoModelForCausalLM.from_pretrained(ROOT/'inputs/base_exact',dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
  self.params=dict(self.model.named_parameters());self.original={};self.current=None;self.tokens=0;self.seconds=0
 def state(self,step,role):return load_file(str(ROOT/f'inputs/ckpt_{step}/{role}/adapter_model.safetensors'),device='cpu')
 @torch.no_grad()
 def policy(self,role,step=0,window=None,alpha=0):
  key=(role,step,tuple(window) if window else None,alpha)
  if self.current==key:return
  terms=[(step,1.)] if window is None else [(0,1.),(window[1],alpha),(window[0],-alpha)]
  states=[]
  for s,coef in terms:
   if coef==0:continue
   config=json.loads((ROOT/f'inputs/ckpt_{s}/{role}/adapter_config.json').read_text())
   assert not config.get('use_dora') and not config.get('use_rslora') and not config.get('fan_in_fan_out')
   assert config.get('bias')=='none' and not config.get('rank_pattern') and not config.get('alpha_pattern')
   states.append((self.state(s,role),coef*config['lora_alpha']/config['r']))
  first=states[0][0]
  for k in first:
   if not k.endswith('.lora_A.weight'):continue
   bkey=k.replace('.lora_A.weight','.lora_B.weight')
   name=k.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
   if name not in self.params:raise ValueError('unmatched adapter target '+name)
   target=self.params[name]
   if name not in self.original:self.original[name]=target.detach().cpu().clone()
   value=self.original[name].to(device='cuda',dtype=torch.float32)
   for state,coef in states:
    aa=state[k].to(device='cuda',dtype=torch.float32);bb=state[bkey].to(device='cuda',dtype=torch.float32)
    value.addmm_(bb,aa,alpha=coef)
   target.copy_(value.to(target.dtype))
  self.current=key
 @torch.inference_mode()
 def generate(self,requests,label):
  prompts=[x[0] for x in requests];prefix=[x[1] for x in requests];ids=[x[2] for x in requests]
  rng=seed(ids,label);torch.manual_seed(rng);torch.cuda.manual_seed_all(rng)
  inputs=self.tok(prompts,return_tensors='pt',padding=True).to('cuda')
  if inputs.input_ids.shape[1]>6000:raise RuntimeError('unexpected overlength prompt')
  t=time.time();out=self.model.generate(**inputs,max_new_tokens=250,do_sample=True,temperature=1.,top_p=1.,top_k=0,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id,use_cache=True)
  tokens=out[:,inputs.input_ids.shape[1]:].cpu();elapsed=time.time()-t;rows=[]
  for i,row in enumerate(tokens.tolist()):
   if self.tok.eos_token_id in row:row=row[:row.index(self.tok.eos_token_id)+1]
   text=self.tok.decode(row,skip_special_tokens=True)
   rows.append(dict(text=prefix[i]+text,raw=text,token_ids=row,prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),seed=rng,cap_hit=len(row)==250,eos=bool(row and row[-1]==self.tok.eos_token_id)))
  self.tokens+=sum(len(x['token_ids']) for x in rows);self.seconds+=elapsed
  return rows

def clean_batch(engine,cases,pair):
 rows=[dict(case_id=c['case_id'],kind='debate',cell=list(pair),variant='clean',question=c['question'],topic=c['topic'],r1_a=c['r1_a'],r1_b=c['r1_b'],source_step=c['source_step'],meta={}) for c in cases]
 for side,step in zip(['a','b'],pair):
  engine.policy('debate',step)
  req=[(*r2prompt(c,side),c['case_id']) for c in cases]
  for r,z in zip(rows,engine.generate(req,'r2_'+side)):r['r2_'+side]=z['text'];r['meta']['r2_'+side]=z
 for side,step in zip(['a','b'],pair):
  engine.policy('debate',step)
  req=[(*r3prompt(c,side,{s:r['r2_'+s] for s in ['a','b']}),c['case_id']) for c,r in zip(cases,rows)]
  for r,z in zip(rows,engine.generate(req,'r3_'+side)):r['r3_'+side]=z['text'];r['meta']['r3_'+side]=z
 return rows

def causal_batch(engine,cases,clean,step):
 out=[]
 # Group by focal side, same case lists and seeds for truthful/false propagated variants.
 for side in ['a','b']:
  subset=[(c,r) for c,r in zip(cases,clean) if c['focal_side']==side]
  if not subset:continue
  other='b' if side=='a' else 'a';engine.policy('debate',step)
  for treatment in ['true','false']:
   changed=[]
   for c,r in subset:
    q=copy.deepcopy(r);q['variant']=treatment+'_frozen';q['focal_side']=side;q['claim_word']=c['claim_word'];q['true_count']=c['true_count']
    q['r2_'+other],q['edit']=insert(q['r2_'+other],c[treatment+'_claim']);changed.append(q);out.append(q)
   req=[(*r3prompt(c,side,{s:r['r2_'+s] for s in ['a','b']}),c['case_id']) for (c,_),r in zip(subset,changed)]
   for r,z in zip(changed,engine.generate(req,'propagated_r3_'+side)):
    q=copy.deepcopy(r);q['variant']=treatment+'_propagated';q['r3_'+side]=z['text'];q['meta']['r3_'+side]=z;out.append(q)
 return out

def amplification(engine,cases,role,window,alpha,step=None):
 engine.policy(role,step or 0,window,alpha)
 if role=='solution':
  out=engine.generate([(plain(None,c['question']),'',c['case_id']) for c in cases],'amp_solution')
  return [dict(kind='amplification',role=role,window=window,alpha=alpha,step=step,case_id=c['case_id'],question=c['question'],text=z['text'],meta=z) for c,z in zip(cases,out)]
 req=[(*r2prompt(c,'a'),c['case_id']) for c in cases];opening=engine.generate(req,'amp_r2')
 req=[(*r3prompt(c,'a',{'a':z['text'],'b':c['archive_r2_b']}),c['case_id']) for c,z in zip(cases,opening)]
 closing=engine.generate(req,'amp_r3')
 return [dict(kind='amplification',role=role,window=window,alpha=alpha,step=step,case_id=c['case_id'],question=c['question'],r1_a=c['r1_a'],r1_b=c['r1_b'],r2_a=z['text'],r2_b=c['archive_r2_b'],r3_a=w['text'],meta={'r2':z,'r3':w}) for c,z,w in zip(cases,opening,closing)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 spec=json.loads((ROOT/'spec.json').read_text());cases=[json.loads(l) for l in (ROOT/'inputs/panel.jsonl').read_text().splitlines()];cases=[c for c in cases if c['shard']==args.shard]
 start=time.time();deadline=start+spec['execution']['hard_hours']*3600
 engine=Engine();batch=4
 if args.pilot:
  sample=cases[:4];t=time.time();clean=clean_batch(engine,sample,(100,100));variants=causal_batch(engine,sample,clean,100)
  amp=amplification(engine,sample,'debate',[120,140],2);sol=amplification(engine,sample,'solution',[160,180],-1)
  rows=clean+variants+amp+sol
  (ROOT/'outputs/pilot_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  writej(ROOT/'outputs/pilot.json',dict(status='MEASURED',seconds=time.time()-t,generated_tokens=engine.tokens,generation_seconds=engine.seconds,tokens_per_second=engine.tokens/engine.seconds,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,rows=len(rows),cap_rate=sum(z.get('cap_hit',False) for r in clean for z in r['meta'].values())/16,nonempty=all(r.get('r3_a',r.get('text','')).strip() for r in rows)))
  return
 gate=json.loads((ROOT/'execution/release.json').read_text());assert gate['status']=='PROCEED'
 cells=[(100,100),(140,140),(160,160),(200,200),(100,200),(200,100)]
 outdir=ROOT/'outputs/blocks';outdir.mkdir(exist_ok=True)
 # Complete every cell for a four-prompt block before advancing.
 for i in range(0,len(cases),batch):
  if time.time()>deadline:break
  path=outdir/f'causal_{i:03d}.jsonl'
  if path.exists():continue
  rows=[]
  for pair in cells:
   clean=clean_batch(engine,cases[i:i+batch],pair);rows.extend(clean)
   if pair[0]==pair[1]:rows.extend(causal_batch(engine,cases[i:i+batch],clean,pair[0]))
  tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(path)
  writej(ROOT/'outputs/heartbeat.json',dict(stage='causal',completed_prompts=i+batch,total_prompts=len(cases),time=time.time(),generated_tokens=engine.tokens,elapsed=time.time()-start))
 # The first 64 globally shuffled IDs form the amplification panel, 32 on each host.
 amp_cases=cases[:32]
 variants=[(None,0,0)]+[(w,a,None) for w in [[120,140],[160,180]] for a in [-1,1,2]]+[(None,0,s) for s in [120,140,160,180]]
 for role in ['solution','debate']:
  for vi,(window,alpha,step) in enumerate(variants):
   for i in range(0,len(amp_cases),batch):
    if time.time()>deadline:break
    path=outdir/f'amp_{role}_{vi:02d}_{i:03d}.jsonl'
    if path.exists():continue
    rows=amplification(engine,amp_cases[i:i+batch],role,window,alpha,step)
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(path)
    writej(ROOT/'outputs/heartbeat.json',dict(stage='amplification',role=role,variant=vi,prompts=i+batch,time=time.time(),generated_tokens=engine.tokens,elapsed=time.time()-start))
 expected=16+2*len(variants)*8
 actual=len(list(outdir.glob('*.jsonl')))
 writej(ROOT/'outputs/terminal.json',dict(status='DONE' if actual==expected else 'PARTIAL_DEADLINE',blocks=actual,expected_blocks=expected,elapsed=time.time()-start,generated_tokens=engine.tokens))
if __name__=='__main__':
 try:main()
 except Exception as e:
  writej(ROOT/'outputs/FAILED.json',dict(error=type(e).__name__,message=str(e),time=time.time()));raise
