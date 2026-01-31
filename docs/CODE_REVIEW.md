# Code Review Notes

Issues found during code review that need attention.

---

## Duplicate `debate_types.py` File

**Status:** Needs cleanup

**Files:**
- `debate_types.py` (root) - **OUTDATED, NOT IMPORTED**
- `src/tinker_debate/debate_types.py` - **CURRENT, ACTIVELY USED**

### Problem

There are two `debate_types.py` files. The root version is outdated and not imported by any script, but its presence can cause confusion.

All imports use the package version:
```python
from tinker_debate.debate_types import DebateConfig  # imports from src/tinker_debate/
```

### Differences

| Feature | Root (outdated) | Package (current) |
|---------|-----------------|-------------------|
| `Verdict` type | `Literal["A", "B"]` | `Literal["A", "B", "INVALID"]` |
| `Transition.raw_response` | Missing | Present |
| `DebateResult` judge fields | Missing | Has `judge_prompt_tokens`, `judge_completion_tokens`, `judge_completion_logprobs`, `judge_raw_response` |
| `system_propose` prompt | Has `<THINKING>` block format | Simpler format |
| `compute_training_stats` | Divides by `total` | Divides by `valid_total` (handles INVALID correctly) |

### Recommendation

Delete the root `debate_types.py` file since:
1. It is not imported anywhere
2. It contains outdated code that doesn't match the current implementation
3. It could cause confusion if someone edits the wrong file
4. IDE autocompletion might suggest imports from the wrong file

### How to Fix

```bash
# After confirming nothing imports from root:
rm debate_types.py
```

Or if you want to keep it for reference:
```bash
mv debate_types.py debate_types.py.old
```

---

## Possible Issues (Need Investigation)

These issues were identified during code review but require further investigation to confirm.

### Continuation Token Slicing (85% confidence - Likely Real Bug)

**File:** `src/tinker_debate/debate_types.py:195-199`

**Code:**
```python
r1_full_len = len(t1.prompt_tokens) + len(t1.completion_tokens)
r2_continuation_tokens = t2.prompt_tokens[r1_full_len:]
```

**Investigation Results:**

The token flow is:
```
R1:
  r1_prompt (string) → tokenize → r1_prompt_tokens (client-side)
  API generates → r1_completion_tokens (from API)
  r1_a = decode(r1_completion_tokens) → string

R2:
  r2_prompt = r1_prompt + r1_a + continuation (string concatenation)
  r2_prompt → tokenize → r2_prompt_tokens (NEW tokenization)
```

**The Problem:**
- `r2_prompt_tokens` = tokenize(r1_prompt + decode(r1_completion_tokens) + continuation)
- This is NOT the same as `r1_prompt_tokens + r1_completion_tokens + continuation_tokens`
- BPE tokenizers produce different tokens at string boundaries

**Example:**
```
API returns: [100, 200] → decode → "hello world"
tokenize("prefix" + "hello world") ≠ tokenize("prefix") + [100, 200]
```

**Impact:** Training data could have misaligned tokens. The merged sequence in `assemble_training_data` may be corrupted.

**Severity:** Medium - continuation tokens get advantage=0, so gradient impact is limited, but data integrity is compromised.

**To verify:** Run a test comparing token counts before/after to see if mismatch occurs in practice.

---

### Solution Extraction Failures Not Tracked (90% confidence)

**File:** `src/tinker_debate/debate_env.py:248-249`

**Code:**
```python
solution_a = extract_solution(r1_a)  # Could return None
solution_b = extract_solution(r1_b)  # No tracking of failure rate
```

**Issue:** When `extract_solution()` returns `None` (model didn't output `<SOLUTION>` tag), there's no metric tracking how often this happens.

**Impact:** Unknown failure rate - could be 0.1% or 20% of debates.

**Note:** May be intentional design choice. Decide if tracking is needed for research.

---

### Hardcoded Chat Template (70% confidence - Limitation)

**File:** `src/tinker_debate/debate_env.py:49-54`

**Code:**
```python
def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"  # Qwen/ChatML format

def _im_end() -> str:
    return "<|im_end|>\n"
```

**Issue:** Chat template tokens are hardcoded for Qwen/ChatML format. Would break with other models.

**Note:** This is a limitation, not a bug - the codebase is designed specifically for Qwen models (see `Qwen3InstructRenderer` in tinker_client.py). Only a problem if you want to support other models.

---

### Empty Solution Returns Empty String (Minor)

**File:** `src/tinker_debate/debate_env.py:24-26`

**Code:**
```python
def extract_solution(text: str) -> str | None:
    match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL)
    return match.group(1).strip() if match else None
```

**Issue:** If model outputs `<SOLUTION>   </SOLUTION>` (whitespace only), returns `""` not `None`.

**Impact:** Minimal - empty string is falsy like None, so checks like `if solution:` still work.

---

### Ground Truth Exact String Match (Minor)

**File:** `src/tinker_debate/debate_types.py:261`

**Code:**
```python
if winner.frozen_solution == d.ground_truth:
```

**Issue:** No normalization. `"56"` ≠ `"56."` ≠ `" 56"` ≠ `"fifty-six"`.

**Impact:** Accuracy metric may be unreliable for non-exact matches.

---

### INVALID Skip Not Logged in assemble_training_data (Minor)

**File:** `src/tinker_debate/debate_types.py:179-180`

**Code:**
```python
if debate.verdict not in ("A", "B"):
    continue  # No logging here
```

**Note:** Count IS tracked in `compute_training_stats`, but `assemble_training_data` doesn't log how many were skipped.

---

*Last updated: 2024-12*
