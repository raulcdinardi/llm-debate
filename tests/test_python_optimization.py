import json
import math
from pathlib import Path
from llm_local_rl.python_optimization import PythonOptimizationEnv,INSTRUCTION,PREFILL
from llm_local_rl.task_types import TaskInstance

class Tokenizer:
    eos_token_id=9
    def encode(self,text,add_special_tokens=False):return list(text.encode())
    def decode(self,ids,skip_special_tokens=True):return bytes(ids).decode()

def test_exact_prompt():
    env=object.__new__(PythonOptimizationEnv);env.settings={}
    task={'text':'Sum values.','code':'def f(x): return sum(x)','test_list':['assert f([1]) == 1']}
    got=env.build_initial_prompt(instance=TaskInstance('x',task))
    assert got=='User:\nSum values.\n\nCanonical Python solution:\n```python\ndef f(x): return sum(x)\n```\n\nTests:\nassert f([1]) == 1\n\n'+INSTRUCTION+'\nAssistant:\n'+PREFILL

def test_log_speedup_does_not_gate_incorrect_output():
    env=object.__new__(PythonOptimizationEnv);env.settings={};env.reference={1:(10,600)}
    class FakeSandbox:
        def run(self,*a,**kw):return dict(status='ok',elapsed_ns=200,correct=False)
    env.sandbox=FakeSandbox()
    reward,metrics=env._score(TaskInstance('x',{'task_id':1}),'def f(x): return None\n```')
    assert reward==math.log(3.0) and metrics['correct'] is False
    assert metrics['speedup']==3.0
    assert metrics['parse_success']==1.0

def test_invalid_penalty_and_no_repair():
    env=object.__new__(PythonOptimizationEnv);env.settings={}
    assert env._score(TaskInstance('x',{'task_id':1}),'def :')[0]==-10.0

def test_slower_completion_has_negative_log_reward():
    env=object.__new__(PythonOptimizationEnv);env.settings={};env.reference={1:(10,200)}
    class FakeSandbox:
        def run(self,*a,**kw):return dict(status='ok',elapsed_ns=400,correct=True)
    env.sandbox=FakeSandbox()
    reward,metrics=env._score(TaskInstance('x',{'task_id':1}),'def f(x): return x')
    assert reward==math.log(0.5) and metrics['speedup']==0.5

def test_timeout_has_negative_penalty():
    env=object.__new__(PythonOptimizationEnv);env.settings={};env.reference={1:(10,200)}
    class FakeSandbox:
        def run(self,*a,**kw):return dict(status='timeout',wall_ns=5000000000)
    env.sandbox=FakeSandbox()
    reward,metrics=env._score(TaskInstance('x',{'task_id':1}),'while True: pass')
    assert reward==-10.0 and metrics['status']=='timeout'

def test_batch_order_and_stable_cpu(monkeypatch):
    env=object.__new__(PythonOptimizationEnv);env.settings={};env.settings={'cpus':[2,4]}
    seen=[];monkeypatch.setattr('os.sched_setaffinity',lambda pid,cpus:seen.append(cpus))
    env._score=lambda inst,text:(float(inst.payload['task_id']),{'text':text})
    instances=[TaskInstance(str(i),{'task_id':i}) for i in [5,2,1]]
    got=env.score_completions(instances=instances,tokenizer=Tokenizer(),completion_token_ids=[[97],[98],[99]])
    assert [r for r,m in got]==[5.,2.,1.]
    assert [m['text'] for r,m in got]==['a','b','c']
    assert {tuple(x) for x in seen}=={(2,),(4,)}


def test_configured_invalid_reward():
    env=object.__new__(PythonOptimizationEnv);env.settings={'invalid_reward':-0.2}
    reward, metrics=env._score(TaskInstance(instance_id='x',payload={'task_id':1}), 'def broken(')
    assert reward == -0.2
    assert metrics['status'] == 'syntax_error'
