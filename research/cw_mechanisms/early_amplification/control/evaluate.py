from pathlib import Path
import argparse,json,time,fcntl,hashlib,traceback
import runner as old
import torch
import os
R=Path(__file__).resolve().parents[1];host=os.environ.get('CW_HOST',R.parts[2]);spec=json.loads((R/'spec.json').read_text())
assert hashlib.sha256((R/'control/runner.py').read_bytes()).hexdigest()==spec['archived_runner_sha256']
for name,digest in spec['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest

def saveblock(path,rows):
 assert not path.exists(),'No duplicate/resume without explicit review'
 t=path.with_suffix('.tmp');t.write_text(''.join(json.dumps(r)+'\n' for r in rows));t.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 lock=(R/'execution/gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 shard=0 if host=='vm02' else 1
 cases=[c for c in map(json.loads,(R/'inputs/panel.jsonl').read_text().splitlines()) if c['shard']==shard]
 amps=list(map(json.loads,(R/'inputs/amplification_panel.jsonl').read_text().splitlines()))
 engine=old.Engine();start=time.time()
 if args.pilot:
  clean=old.clean_batch(engine,cases[:4],(20,20));rows=clean+old.causal_batch(engine,cases[:4],clean,20)
  restored=[]
  for role in spec['amplification']['roles']:
   engine.policy(role,0);original={n:engine.params[n].detach().cpu().clone() for n in engine.original}
   for w in spec['amplification']['windows_by_host'][host]:
    for alpha in [32,-32]:
     engine.policy(role,window=w,alpha=alpha);assert all(torch.isfinite(p).all() for p in engine.params.values())
   rows.extend(old.amplification(engine,amps[:2],role,spec['amplification']['windows_by_host'][host][0],32))
   engine.policy(role,0);restored.append(all(torch.equal(v,engine.params[n].detach().cpu()) for n,v in original.items()))
  (R/'outputs/pilot_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  old.writej(R/'outputs/preflight.json',dict(status='PASS' if all(restored) and len(rows)==24 else 'FAIL',rows=len(rows),finite_weights=True,alpha0_restoration=all(restored),elapsed=time.time()-start,tokens=engine.tokens,tokens_per_second=engine.tokens/engine.seconds,gpu_gib=torch.cuda.max_memory_allocated()/2**30))
  return
 assert json.loads((R/'execution/release.json').read_text())['status']=='PROCEED'
 out=R/'outputs/blocks';out.mkdir(exist_ok=True)
 def heartbeat(stage,**more):old.writej(R/'outputs/heartbeat.json',dict(stage=stage,blocks=len(list(out.glob('*.jsonl'))),expected=304,elapsed=time.time()-start,tokens=engine.tokens,**more))
 for cp in spec['causal']['checkpoints']:
  for i in range(0,len(cases),4):
   clean=old.clean_batch(engine,cases[i:i+4],(cp,cp));rows=clean+old.causal_batch(engine,cases[i:i+4],clean,cp)
   saveblock(out/f'causal_cp{cp:03d}_{i:03d}.jsonl',rows);heartbeat('causal',checkpoint=cp)
 for w in spec['amplification']['windows_by_host'][host]:
  variants=[(None,0,0)]+[(w,a,None) for a in spec['amplification']['alphas'] if a]+[(None,0,t) for t in w]
  for role in spec['amplification']['roles']:
   for vi,(window,a,t) in enumerate(variants):
    engine.policy(role,step=t or 0,window=window,alpha=a);assert all(torch.isfinite(p).all() for p in engine.params.values())
    for i in range(0,len(amps),4):
     rows=old.amplification(engine,amps[i:i+4],role,window,a,t)
     for r in rows:r['control_for_window']=w
     saveblock(out/f'behavior_w{w[0]:03d}_{role}_{vi:02d}_{i:02d}.jsonl',rows);heartbeat('amplification',window=w,alpha=a,role=role)
 old.writej(R/'outputs/terminal.json',dict(status='DONE',blocks=len(list(out.glob('*.jsonl'))),expected_blocks=304,elapsed=time.time()-start,tokens=engine.tokens))
if __name__=='__main__':
 try:main()
 except Exception as e:old.writej(R/'outputs/FAILED.json',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
