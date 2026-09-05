# Archived training-tensor regression fixtures

Expected values were produced in separate Python processes importing each
unmodified source checkout, not by the consolidation implementation:

- `cw_original.json`: original `0ecfe57` CW soft-delta baseline.
- `cw.json`: `10ede7c` at q=0.5, q=1 and four-round q=1.
- `mmlu.json`: `f7b7975` frozen-judge prompt-GRPO.

Inputs are the source test helpers' two debates for one prompt, task rewards
(1,3)/(4,2), soft scores 0.25/-0.5 and token offset 100 on the second debate.
For sources that expose JS, divergences are 0.2/0.9. Later rounds share one
adapter. Fixtures contain every training-example field except diagnostic
metadata: tokens, logprobs, masks and advantages must match exactly.

The original baseline calls `soft_judge`, which meant raw +/-s in that source;
the recorded test invocation uses its explicit current name `soft_judge_raw`.
No scoring API or model generation was used to create these synthetic
regression inputs. These fixtures establish projection compatibility for the
recorded cases, not GPU numerical parity for an entire run.
