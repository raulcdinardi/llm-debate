"""Single-turn MBPP speed reward; task correctness is diagnostic, never a gate."""
from __future__ import annotations
import ast
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import statistics
from llm_local_rl.task_types import TaskInstance
from llm_local_rl.python_sandbox import PythonSandbox

INSTRUCTION = 'Optimize the canonical function to run faster while preserving its behavior and exact function names. Return only the complete replacement Python code, with no explanation. If no safe optimization is apparent, keep the original implementation.'
PREFILL = 'Here is the optimized implementation:\n```python\n'
INVALID_REWARD = -10.0

class PythonOptimizationEnv:
    name = 'python_optimization'
    def __init__(self, *, config_path):
        self.settings=json.loads(Path(config_path).read_text())
        self.tasks=[json.loads(line) for line in Path(self.settings['dataset_path']).read_text().splitlines()]
        self.sandbox=PythonSandbox(self.settings)
        self.reference={}
    def sample_instances(self, *, n, seed):
        rng=random.Random(seed)
        return [TaskInstance(instance_id=f'mbpp:{seed}:{i}:{t["task_id"]}',payload=t) for i,t in enumerate(rng.choices(self.tasks,k=n))]
    def build_initial_prompt(self, *, instance):
        t=instance.payload
        return ('User:\n'+t['text']+'\n\nCanonical Python solution:\n```python\n'+t['code']+'\n```\n\nTests:\n'+'\n'.join(t.get('prompt_test_list',t['test_list']))+'\n\n'+INSTRUCTION+'\nAssistant:\n'+PREFILL)
    def build_initial_prompt_token_ids(self, *, instance, tokenizer, enable_thinking=None):
        return tokenizer.encode(self.build_initial_prompt(instance=instance),add_special_tokens=False)
    def stop_token_ids(self, *, tokenizer):
        ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        if self.settings.get('stop_at_code_fence', False):
            ids.extend(token_id for text, token_id in tokenizer.get_vocab().items() if '```' in text)
        return sorted(set(ids))
    def _score(self, instance, text):
        code=text.split('```',1)[0]
        metrics={'task_id':instance.payload['task_id'],'code':code,'closed_fence':'```' in text,'reward_definition':'ln(reference_time/candidate_time)','correct':False,'parse_success':0.0}
        if self.settings.get('require_closing_fence', False) and not metrics['closed_fence']:
            return float(self.settings.get('invalid_reward', INVALID_REWARD)),dict(metrics,status='missing_fence',speedup=0.0)
        try:ast.parse(code)
        except SyntaxError:return float(self.settings.get("invalid_reward", INVALID_REWARD)),dict(metrics,status='syntax_error',speedup=0.0)
        if not code.strip():return float(self.settings.get("invalid_reward", INVALID_REWARD)),dict(metrics,status='empty',speedup=0.0)
        metrics['parse_success']=1.0
        t=instance.payload;key=t['task_id']
        if key not in self.reference:
            calibration=self.sandbox.run(t,code=t['code'],loops=0)
            if calibration['status']!='ok':raise RuntimeError(f'Reference calibration failed: {key}: {calibration}')
            loops=calibration['calibration_loops']
            times=[]
            for _ in range(3):
                out=self.sandbox.run(t,code=t['code'],loops=loops)
                if out['status']!='ok' or not out['correct']:raise RuntimeError(f'Reference validation failed: {key}: {out}')
                times.append(out['elapsed_ns'])
            self.reference[key]=(loops,statistics.median(times))
        loops,reference_ns=self.reference[key]
        out=self.sandbox.run(t,code=code,loops=loops)
        if out['status']!='ok':return float(self.settings.get("invalid_reward", INVALID_REWARD)),dict(metrics,**out,speedup=0.0,reference_ns=reference_ns)
        speedup=reference_ns/out['elapsed_ns']
        reward=math.log(speedup)
        assert math.isfinite(reward)
        return reward,dict(metrics,**out,speedup=speedup,reference_ns=reference_ns,benchmark_loops=loops)
    def score_completion(self, *, instance, tokenizer, completion_token_ids):
        return self.score_completions(instances=[instance],tokenizer=tokenizer,completion_token_ids=[completion_token_ids])[0]
    def score_completions(self, *, instances, tokenizer, completion_token_ids):
        items=[(i,tokenizer.decode(t,skip_special_tokens=True)) for i,t in zip(instances,completion_token_ids,strict=True)]
        cpus=self.settings['cpus'];lanes=[[] for _ in cpus]
        # Same task stays on the same physical core in every step and replay.
        for position,(instance,text) in enumerate(items):lanes[instance.payload['task_id']%len(cpus)].append((position,instance,text))
        def work(lane):
            os.sched_setaffinity(0,{cpus[lane]})
            return [(pos,self._score(inst,text)) for pos,inst,text in lanes[lane]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:results=[r for lane in pool.map(work,range(len(cpus))) for r in lane]
        return [score for _,score in sorted(results)]
