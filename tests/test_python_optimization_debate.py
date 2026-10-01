import pytest
from llm_local_rl.python_optimization import PREFILL
from llm_local_rl.python_optimization_debate import PythonOptimizationDebateTask
from llm_local_rl.debate_runtime import DebateRuntime,_base_text_prompt
from llm_local_rl.task_types import TaskInstance,TaskReward

TASK=TaskInstance('mbpp:1',{'text':'Sum values.','code':'def f(x): return sum(x)','test_list':['assert f([1]) == 1']})
def test_debate_r1_is_byte_identical_to_baseline():
 task=object.__new__(PythonOptimizationDebateTask)
 assert _base_text_prompt(system_text=None,user_text=task.r1_context_text(inst=TASK),assistant_prefill=PREFILL)==task.build_initial_prompt(instance=TASK)

def test_each_debate_round_has_thirty_word_point_instruction():
 task=object.__new__(PythonOptimizationDebateTask)
 for opponent_round in [1,2]:
  extension=task.build_base_text_debate_extension(inst=TASK,opponent_round=opponent_round,opponent_answer='def f(x): return 0')
  assert 'exactly 3 numbered points, each at most 30 words' in extension.user_text
  assert 'fixed and cannot change' in extension.system_text
  assert 'CONCLUDED' in extension.user_text

