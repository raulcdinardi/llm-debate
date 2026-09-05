from pathlib import Path
import os,sys,json,hashlib,time,random,threading,concurrent.futures,urllib.request,urllib.error,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source/src'))
from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask
from llm_local_rl.task_types import TaskInstance
TASK=ConstrainedWritingDebateTask.from_args(rules_per_speaker=2,reward_scope='both',sides='both',rule_family='generic',reward_mode='additive',letter_temperature=1.,anchors='on')
OUT=R/'outputs';(OUT/'raw').mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True);LOCK=threading.Lock()
MODEL='deepseek/deepseek-v4-flash-0731';CAP=3.;RESERVE=.03
COST=json.loads((OUT/'cost.json').read_text()) if (OUT/'cost.json').exists() else {'spent':0.,'reserved':0.,'cap':CAP}
COST['spent']+=COST['reserved'];COST['reserved']=0
KEY=os.environ.get('OPENROUTER_API_KEY')
assert KEY

def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def call(label,prompt,limit=1800):
 key=hashlib.sha256((MODEL+'baidu'+prompt).encode()).hexdigest();p=OUT/'cache'/f'{key}.json'
 if p.exists():return json.loads(p.read_text())['parsed']
 for attempt in range(3):
  with LOCK:
   if COST['spent']+COST['reserved']+RESERVE>CAP:raise RuntimeError('API_CAP')
   COST['reserved']+=RESERVE;save(OUT/'cost.json',COST)
  body={'model':MODEL,'provider':{'only':['baidu'],'allow_fallbacks':False,'require_parameters':True},'reasoning':{'effort':'none','exclude':True},'temperature':0,'max_tokens':limit,'response_format':{'type':'json_object'},'messages':[{'role':'user','content':prompt}]}
  rec={'label':label,'key':key,'attempt':attempt,'request':body,'time':time.time()};cost=RESERVE;rawpath=OUT/'raw'/f'{key}_{time.time_ns()}.json'
  try:
   req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as f:txt=f.read().decode();rec['status']=f.status
   rec['response_text']=txt;save(rawpath,rec)
   j=json.loads(txt);cost=j.get('usage',{}).get('cost')
   if cost is None:cost=RESERVE;raise ValueError('missing cost')
   assert j['model']==MODEL and j['provider'].lower()=='baidu' and not j.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0),'IDENTITY'
   choice=j['choices'][0];rec['finish_reason']=choice['finish_reason']
   if choice['finish_reason']=='length':limit=min(limit*2,4000);raise ValueError('truncation')
   text=choice['message']['content'].strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
   parsed=json.loads(text);assert isinstance(parsed,dict)
   save(p,{'parsed':parsed,'raw':str(rawpath.relative_to(R))});return parsed
  except Exception as e:
   if isinstance(e,urllib.error.HTTPError):
    rec['http_status']=e.code;rec['response_text']=e.read().decode(errors='replace')
   rec['error']=type(e).__name__+':'+str(e)[:150]
   if 'IDENTITY' in str(e) or isinstance(e,urllib.error.HTTPError) and e.code in [400,401,402,403,404]:raise
   if attempt==2:raise
  finally:
   save(rawpath,rec)
   with LOCK:COST['reserved']-=RESERVE;COST['spent']+=float(cost);save(OUT/'cost.json',COST)
class Tok:
 def decode(self,x,**kw):return x

def metrics(text,payload):
 z=TASK.compute_reward(inst=TaskInstance(instance_id='audit',payload=payload),completion_tokens=text,tokenizer=Tok())
 return {'reward':z.reward,**{k:v for k,v in z.metrics.items() if k in ['parse_success','rule_scores','anchor_penalty','anchor_sentence_count','anchor_sentence_count_penalty','anchor_word_window_penalty','anchor_word_window_violations','anchor_repetition_violated','anchor_duplicate_sentences','anchor_words_per_sentence']}}
RUBRIC='Rate only writing craft. Ignore constraint satisfaction, formatting quirks and stray leading letters. overall integer1-10: holistic coherence, imagery and flow. Score each independently with the same rubric; equal scores are allowed. Return JSON {"quality_X":integer,"quality_Y":integer}.\n'
def quality(label,a,b,reverse=False):
 z=call(label,RUBRIC+'STORY X:\n'+(b if reverse else a)+'\nSTORY Y:\n'+(a if reverse else b),160)
 assert all(type(z[k]) is int and 1<=z[k]<=10 for k in ['quality_X','quality_Y'])
 return [z['quality_Y'],z['quality_X']] if reverse else [z['quality_X'],z['quality_Y']]
def process(x):
 result={'id':x['id'],'candidate':x,'status':'REJECTED','edits':[]}
 try:
  s=x['sample'];a=s['trajectory_a']['r1'];b=s['trajectory_b']['r1']
  result['tie_selection']='existing archived overall scores; no new baseline tie scores'
  for side in ['a','b']:
   original=s['trajectory_'+side]['r1'];debates={f'{s2.upper()}:{rnd}':s['trajectory_'+s2][rnd] for s2 in ['a','b'] for rnd in ['r2','r3']}
   prompt='''Create ONE minimally edited worse version of TARGET STORY for a controlled experiment. Reduce prose craft (imagery or flow) ONLY in a dimension not addressed by ANY of the four fixed debate turns. Preserve all factual events, entities, relations, quoted spans, word counts mentioned, sentence count and words per sentence, sentence initial/final words, colors, ed-suffix verbs and all rule satisfaction/penalties. Replace unmentioned expressive words with bland words where feasible. Do not add/delete sentences or append garbage. Do not change the truth status, relevance or applicability of ANY debate claim, including already false ones and qualitative/comparative claims. If any such preservation prevents a real quality drop, return feasible:false. Return JSON {"feasible":boolean,"edited_story":string,"quality_dimension":string,"changes":[{"original":string,"replacement":string,"why_unargued":string}],"preservation_explanation":string}.\n'''+json.dumps({'target_side':side.upper(),'target_story':original,'other_story':b if side=='a' else a,'fixed_debate':debates,'rules_and_metrics':s['trajectory_'+side]['task_reward_metrics']},ensure_ascii=False)
   edit=call(x['id']+' edit '+side,prompt,2400);e={'side':side,'proposal':edit,'accepted':False};result['edits'].append(e)
   if edit.get('feasible') is not True:continue
   new=edit['edited_story'];assert isinstance(new,str)
   before=metrics(original,s['trajectory_'+side]['task_reward_metrics']);after=metrics(new,s['trajectory_'+side]['task_reward_metrics']);e['metrics_before']=before;e['metrics_after']=after
   if before!=after:e['reason']='mechanical_invariance_failed';continue
   q=[quality(x['id']+' degradation '+side,original,new,rev) for rev in [False,True]];e['quality']=q
   if not all(z[0]>z[1] for z in q):e['reason']='quality_drop_not_confirmed_both_orders';continue
   audit=call(x['id']+' invariance '+side,'''Audit a proposed fixed-debate counterfactual. Independently check EVERY claim in all four debate turns against the original and modified stories. Originally false claims may remain false: the requirement is no change in truth status, relevance, referents or applicability. Include qualitative, comparative, count, quotation and format claims. Be strict: a claim of vividness/coherence is invalidated if the edit worsens the praised dimension. Return JSON {"all_claims_preserved":boolean,"claims":[{"round":string,"claim":string,"before":string,"after":string,"unchanged":boolean}],"concerns":string}. Do not assume the editor was correct.\n'''+json.dumps({'target_side':side.upper(),'original':original,'modified':new,'other_story':b if side=='a' else a,'debate':debates},ensure_ascii=False),3000)
   e['audit']=audit;e['accepted']=audit.get('all_claims_preserved') is True and bool(audit.get('claims')) and all(z.get('unchanged') is True for z in audit['claims'])
   if not e['accepted']:e['reason']='semantic_invariance_failed'
  result['status']='CANDIDATE_PASS' if any(e['accepted'] for e in result['edits']) else 'REJECTED'
 except Exception as e:result['status']='ERROR';result['error']=type(e).__name__+':'+str(e)[:150]
 return result

def main():
 lease=(OUT/'pilot.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 xs=[json.loads(l) for l in (R/'inputs/tied_candidates.jsonl').open()];random.Random(20260905).shuffle(xs);xs=xs[:24]
 save(R/'inputs/pilot_ids.json',[x['id'] for x in xs])
 seen={json.loads(l)['id'] for l in (OUT/'pilot_results.jsonl').open()} if (OUT/'pilot_results.jsonl').exists() else set()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for res in pool.map(process,[x for x in xs if x['id'] not in seen]):
   with (OUT/'pilot_results.jsonl').open('a') as f:f.write(json.dumps(res,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
   print(res['id'],res['status'],res.get('reason',''),flush=True)
   save(OUT/'heartbeat.json',{'stage':'pilot_manipulation','time':time.time(),'completed':sum(1 for _ in (OUT/'pilot_results.jsonl').open()),'cost':COST})
 rows=[json.loads(l) for l in (OUT/'pilot_results.jsonl').open()];save(OUT/'manipulation_terminal.json',{'status':'COLLECTED_FOR_MANUAL_REVIEW','candidates':len(rows),'candidate_passes':sum(x['status']=='CANDIDATE_PASS' for x in rows),'cost':COST})
if __name__=='__main__':main()
