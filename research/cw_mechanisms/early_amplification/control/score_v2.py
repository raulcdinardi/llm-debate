"""Transport-only repair: immutable job keys/prompts, durable raw responses, per-item failure isolation."""
from pathlib import Path
import os,json,time,hashlib,argparse,concurrent.futures,threading,uuid,fcntl
import requests
from score import ROOT,MODEL,SYSTEM,prompt
LOCK=threading.Lock();CAP=3.;RESERVE=.03
OUT=ROOT/'outputs';RAW=OUT/'scoring_v2_raw';RAW.mkdir(exist_ok=True)

def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x));q.replace(p)

def validate(p,row):
 if row['kind']=='debate':
  assert p['winner'] in ['A','B']
  for f in ['claim_response_A','claim_response_B']:assert p[f] in ['rejects','accepts','ignored','unclear','not_applicable']
  for f in ['grounding_A','grounding_B']:assert type(p[f]) is int and 0<=p[f]<=4
  for f in ['fabricated_evidence_A','fabricated_evidence_B','engages_opponent_A','engages_opponent_B']:assert type(p[f]) is bool
  for f in ['evidence_A','evidence_B']:assert isinstance(p[f],str)
 else:
  for f in ['quality','grounding']:assert type(p[f]) is int and 0<=p[f]<=4
  for f in ['fabricated_evidence','repetitive_template','coherent']:assert type(p[f]) is bool
  assert isinstance(p['evidence'],str)

def evaluate(job):
 global spent,reserved
 row,reverse,key,origin=job;limit=384;tool_mode=False
 for attempt in range(4):
  with LOCK:
   if spent+reserved+RESERVE>CAP:return {'key':key,'failed':True,'error':'API_CAP','origin':origin}
   reserved+=RESERVE;atomic(OUT/'scoring_v2_cost.json',{'spent':spent,'reserved':reserved,'cap':CAP})
  cost=RESERVE;uid=uuid.uuid4().hex;record={'key':key,'attempt':attempt,'max_tokens':limit,'origin':origin,'time':time.time()}
  body=dict(model=MODEL,provider={'order':['relace'],'allow_fallbacks':False},temperature=0,reasoning={'effort':'none','exclude':True},max_tokens=limit,messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt(row,reverse)}])
  if tool_mode:
   if row['kind']=='debate':
    properties={f:{'type':'string','enum':['rejects','accepts','ignored','unclear','not_applicable']} for f in ['claim_response_A','claim_response_B']}
    properties['winner']={'type':'string','enum':['A','B']}
    properties.update({f:{'type':'integer','minimum':0,'maximum':4} for f in ['grounding_A','grounding_B']})
    properties.update({f:{'type':'boolean'} for f in ['fabricated_evidence_A','fabricated_evidence_B','engages_opponent_A','engages_opponent_B']})
    properties.update({f:{'type':'string'} for f in ['evidence_A','evidence_B']})
   else:
    properties={f:{'type':'integer','minimum':0,'maximum':4} for f in ['quality','grounding']}
    properties.update({f:{'type':'boolean'} for f in ['fabricated_evidence','repetitive_template','coherent']});properties['evidence']={'type':'string'}
   body['tools']=[{'type':'function','function':{'name':'submit_grade','description':'Return the requested evaluation using exactly the supplied rubric.','parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}]
   body['tool_choice']={'type':'function','function':{'name':'submit_grade'}}
   body['provider']['require_parameters']=True
  record['tool_mode']=tool_mode
  record['request']=body;failure=None
  try:
   resp=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},json=body,timeout=90)
   record.update(http_status=resp.status_code,response_text=resp.text)
   # Persist exact response BEFORE decoding either envelope or content.
   atomic(RAW/(key+'_'+uid+'.json'),record)
   resp.raise_for_status();j=resp.json();usage=j.get('usage',{});cost=usage.get('cost',RESERVE)
   if cost is None:cost=RESERVE;raise ValueError('missing cost')
   provider=j.get('provider');served=j.get('model');reason=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
   if str(provider).lower()!='relace' or served!=MODEL or reason:raise ValueError('IDENTITY_OR_REASONING_VIOLATION')
   choice=j['choices'][0];finish=choice.get('finish_reason');record['finish_reason']=finish
   if tool_mode:
    calls=choice['message'].get('tool_calls',[])
    assert len(calls)==1 and calls[0]['function']['name']=='submit_grade'
    text=calls[0]['function']['arguments'].strip()
   else:text=choice['message']['content'].strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
   parsed=json.loads(text);validate(parsed,row)
   if finish=='length':raise ValueError('truncated completion')
   result=dict(key=key,case_id=row['case_id'],kind=row['kind'],cell=row.get('cell'),variant=row.get('variant'),role=row.get('role'),window=row.get('window'),alpha=row.get('alpha'),step=row.get('step'),reverse=reverse,parsed=parsed,provider=provider,model=served,usage=usage,response_id=j.get('id'),prompt_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),transport_revision="2.1",tool_mode=tool_mode,max_tokens=limit,origin=origin,raw_file=str((RAW/(key+'_'+uid+'.json')).relative_to(ROOT)))
   record['accepted']=True
   return result
  except Exception as e:
   failure=type(e).__name__+': '+str(e)[:300];record['error']=failure
   # Increase ONLY after an observed length finish; never change scientific prompts/schema.
   if record.get('finish_reason')=='length' and limit<1536:
    record['next_attempt_reason']='observed length truncation';limit*=2
   elif record.get('finish_reason') and isinstance(e,(json.JSONDecodeError,AssertionError,KeyError)):
    record['next_attempt_reason']='observed output syntax/schema failure; same rubric via supported tool transport';tool_mode=True
   if 'IDENTITY_OR_REASONING_VIOLATION' in failure:return {'key':key,'failed':True,'error':failure,'origin':origin}
  finally:
   atomic(RAW/(key+'_'+uid+'.json'),record)
   with LOCK:
    reserved-=RESERVE;spent+=float(cost);atomic(OUT/'scoring_v2_cost.json',{'spent':spent,'reserved':reserved,'cap':CAP})
  if attempt<3:time.sleep(2**attempt)
 return {'key':key,'failed':True,'error':failure,'origin':origin}

def main():
 global spent,reserved
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 lease=(OUT/'scoring_v2.lock').open('w');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'scoring_v2_cost.json').exists():
  c=json.loads((OUT/'scoring_v2_cost.json').read_text());spent=c['spent']+c.get('reserved',0)
 else:spent=json.loads((OUT/'scoring_cost.json').read_text())['billed_or_conservatively_reserved']
 reserved=0
 scores=OUT/'scores.jsonl';seen={json.loads(l)['key'] for l in scores.read_text().splitlines()}
 files=sorted((OUT/'blocks').glob('causal_*.jsonl'))+sorted((OUT/'blocks').glob('amp_*.jsonl'))
 jobs=[]
 for f in files:
  for i,line in enumerate(f.read_text().splitlines()):
   row=json.loads(line)
   for reverse in ([False,True] if row['kind']=='debate' else [False]):
    key=hashlib.sha256((f.name+':'+str(i)+':'+str(reverse)).encode()).hexdigest()
    if key not in seen:jobs.append((row,reverse,key,{'file':f.name,'row':i}))
 if args.pilot:jobs=jobs[:8]
 atomic(OUT/'scoring_v2_process.json',{'pid':os.getpid(),'pilot':args.pilot,'start':time.time(),'jobs':len(jobs)})
 accepted=0;failed=0;start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(evaluate,j):j[2] for j in jobs}
  for future in concurrent.futures.as_completed(futures):
   try:r=future.result()
   except Exception as e:r={'key':futures[future],'failed':True,'error':type(e).__name__+': '+str(e)[:300]}
   target=OUT/'scoring_v2_errors.jsonl' if r.get('failed') else scores
   with target.open('a') as h:h.write(json.dumps(r)+'\n');h.flush();os.fsync(h.fileno())
   if r.get('failed'):failed+=1
   else:accepted+=1
   atomic(OUT/'scoring_v2_heartbeat.json',{'accepted_new':accepted,'failed':failed,'jobs':len(jobs),'elapsed':time.time()-start,'cost':spent,'time':time.time()})
 result={'status':'PASS' if args.pilot and failed==0 else ('DONE' if failed==0 else 'PARTIAL_FAILURE'),'accepted_new':accepted,'failed':failed,'jobs':len(jobs),'cost':spent,'elapsed':time.time()-start}
 atomic(OUT/('scoring_v2_pilot.json' if args.pilot else 'scoring_v2_terminal.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
