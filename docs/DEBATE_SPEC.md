# Symmetric Binary Debate: Mathematical Specification

## Overview

Two agents (A, B) debate over a question Q. Each agent proposes a solution in Round 1, then argues for their solution in Rounds 2-3. A separate **Judge LLM** evaluates the full transcript and declares a winner.

**Training signal:** Rejection sampling - we train on the **winner only**. The loser's trajectory is discarded.

**Key design principle:** Solutions are **FROZEN** after Round 1. The debate is about argumentation quality, not solution updates.

---

## Notation

| Symbol | Description |
|--------|-------------|
| $Q$ | Question (input) |
| $G$ | Ground truth answer (optional, for metrics) |
| $\pi_\theta$ | Policy (shared by both agents) |
| $\pi_{\text{ref}}$ | Reference policy (frozen, for KL penalty) |
| $R^{(r)}_a$ | Response from agent $a \in \{A, B\}$ in round $r \in \{1, 2, 3\}$ |
| $s(R)$ | Solution extracted from response $R$ (frozen after R1) |
| $J$ | Judge verdict: $J \in \{A, B, \text{tie}\}$ |

---

## Three-Round Debate Structure

### Round 1: Propose (Solutions FROZEN here)
Each agent independently proposes a solution with reasoning.
- Agent A generates $R^{(1)}_A$ with solution $s_A$
- Agent B generates $R^{(1)}_B$ with solution $s_B$
- **Solutions $s_A$ and $s_B$ are FROZEN** - no changes allowed in later rounds

### Round 2: Argue
Each agent sees the opponent's R1 and argues for their own frozen solution.
- Agent A generates $R^{(2)}_A$: defends $s_A$, critiques $s_B$
- Agent B generates $R^{(2)}_B$: defends $s_B$, critiques $s_A$

### Round 3: Respond to Criticism
Each agent sees the opponent's R2 argument and responds.
- Agent A generates $R^{(3)}_A$: rebuts B's critique, reinforces case for $s_A$
- Agent B generates $R^{(3)}_B$: rebuts A's critique, reinforces case for $s_B$

### Judge: Declare Winner
A separate LLM call sees the full transcript and declares a winner.
- Input: Full debate transcript (Q, R1_A, R1_B, R2_A, R2_B, R3_A, R3_B)
- Output: Winner $J \in \{A, B, \text{tie}\}$
- Optional: Include ground truth G for metrics (but judge may not use it)

---

## Trajectory Structure

### Agent A's Full Trajectory

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [System: P_propose]                                            no loss  │
│ [User: Q]                                                      no loss  │
│ [Assistant: R¹_A]  (proposes solution s_A)                  ← LOSS HERE │
│ [System: P_argue]                                              no loss  │
│ [User: "Opponent proposed:\n" + R¹_B]                          no loss  │
│ [Assistant: R²_A]  (defends s_A, critiques s_B)             ← LOSS HERE │
│ [User: "Opponent's argument:\n" + R²_B]                        no loss  │
│ [Assistant: R³_A]  (responds to criticism)                  ← LOSS HERE │
└─────────────────────────────────────────────────────────────────────────┘
```

### Token-Level View

For agent A, the full sequence with loss mask:

| Segment | Role | Loss Mask |
|---------|------|-----------|
| System prompt (propose) | system | 0 |
| Question $Q$ | user | 0 |
| $R^{(1)}_A$ (propose) | assistant | **1** |
| System prompt (argue) | system | 0 |
| Inject $R^{(1)}_B$ | user | 0 |
| $R^{(2)}_A$ (argue) | assistant | **1** |
| Inject $R^{(2)}_B$ | user | 0 |
| $R^{(3)}_A$ (respond) | assistant | **1** |

---

## Generation Protocol

### Step 1: R1 Generation (Parallel)
Both agents generate proposals independently:
$$R^{(1)}_A \sim \pi_\theta(\cdot \mid P_{\text{propose}}, Q)$$
$$R^{(1)}_B \sim \pi_\theta(\cdot \mid P_{\text{propose}}, Q)$$

Extract and freeze solutions: $s_A = s(R^{(1)}_A)$, $s_B = s(R^{(1)}_B)$

### Step 2: R2 Generation (Cross-inject R1)
Each agent sees opponent's R1:
$$R^{(2)}_A \sim \pi_\theta(\cdot \mid \text{ctx}_A, R^{(1)}_B)$$
$$R^{(2)}_B \sim \pi_\theta(\cdot \mid \text{ctx}_B, R^{(1)}_A)$$

### Step 3: R3 Generation (Cross-inject R2)
Each agent sees opponent's R2 argument:
$$R^{(3)}_A \sim \pi_\theta(\cdot \mid \text{ctx}_A, R^{(2)}_B)$$
$$R^{(3)}_B \sim \pi_\theta(\cdot \mid \text{ctx}_B, R^{(2)}_A)$$

### Step 4: Judge Evaluation
Separate LLM call evaluates full transcript:
$$J = \text{Judge}(Q, R^{(1)}_A, R^{(1)}_B, R^{(2)}_A, R^{(2)}_B, R^{(3)}_A, R^{(3)}_B)$$

---

## Rejection Sampling (Training Signal)

### Core Idea

**We only train on winners.** The loser's trajectory is rejected (discarded).

$$\text{Training set} = \{\tau \mid J(\tau) = \text{winner}\}$$

### Why Rejection Sampling?

1. **Simple gradient signal**: Train on good examples, ignore bad ones
2. **No negative reward issues**: Avoids problems with REINFORCE variance
3. **Implicit preference**: The winner is "preferred" over the loser
4. **Sample efficient**: Each debate produces exactly 1 training trajectory

### Handling Edge Cases

| Judge Verdict | Agent A | Agent B |
|---------------|---------|---------|
| A wins | **TRAIN** | reject |
| B wins | reject | **TRAIN** |
| Tie | reject both | reject both |

**Ties:** Both trajectories rejected. No training signal from tied debates.

### Training Loop

```python
for debate in debates:
    winner = judge(debate)
    if winner == "A":
        train_on(debate.trajectory_a)
    elif winner == "B":
        train_on(debate.trajectory_b)
    else:  # tie
        pass  # reject both
```

### Advantage in Rejection Sampling

Since we only train on winners:
- All trained trajectories have advantage = 1 (or constant positive value)
- GRPO-style z-score normalization still applies across the batch
- Length normalization: divide by token count

Within a batch of winning trajectories:
$$\hat{A}(\tau) = \frac{1}{|o|} \quad \text{(uniform positive advantage, length-normalized)}$$

Or if using z-score across batch:
$$\hat{A}(\tau) = \frac{r(\tau) - \mu_{\text{batch}}}{\sigma_{\text{batch}}} \cdot \frac{1}{|o|}$$

where all $r(\tau) = 1$ for winners, so variance is 0 and we fall back to uniform.

---

## Prompts

### P_propose (Round 1: Propose Solution)

```
You are solving a problem. Think step by step, then provide your answer.

Format:
<THINKING>
[Your reasoning process]
</THINKING>
<SOLUTION>[Your final answer]</SOLUTION>
```

### P_argue (Round 2-3: Debate)

```
You are in a debate. Your solution is FIXED and cannot change.
Your goal: convince the judge your solution is correct.

- Defend your reasoning
- Point out flaws in opponent's approach
- Be persuasive but accurate

Do NOT change your solution. Only argue for it.
```

### P_judge (Judge Evaluation)

```
You are a judge evaluating a debate between two solvers.

Read the full transcript:
- Round 1: Both solvers propose solutions
- Round 2: Both argue for their solutions
- Round 3: Both respond to criticism

Evaluate:
1. Which solution is more likely correct?
2. Which debater made stronger arguments?
3. Who addressed criticisms better?

Output your verdict:
<VERDICT>[A or B or TIE]</VERDICT>
<REASONING>[Brief explanation]</REASONING>
```

---

## Judge Implementation

### API Call

The judge is a separate LLM call (can be same or different model):

```python
def call_judge(
    question: str,
    r1_a: str, r1_b: str,
    r2_a: str, r2_b: str,
    r3_a: str, r3_b: str,
    ground_truth: str | None = None,
) -> tuple[str, str]:  # (verdict, reasoning)
    """
    Call judge LLM to evaluate debate.
    Returns: ('A', 'B', or 'TIE'), reasoning string
    """
```

### Judge Prompt Construction

```
Question: {Q}

=== AGENT A ===
Round 1 (Proposal):
{R1_A}

Round 2 (Argument):
{R2_A}

Round 3 (Response):
{R3_A}

=== AGENT B ===
Round 1 (Proposal):
{R1_B}

Round 2 (Argument):
{R2_B}

Round 3 (Response):
{R3_B}

Based on the debate above, which agent made a more convincing case?
Consider: solution correctness, argument quality, rebuttal effectiveness.

<VERDICT>[A or B or TIE]</VERDICT>
<REASONING>[Your explanation]</REASONING>
```

---

## Metrics

### Primary Metrics
1. **Win Rate**: $\mathbb{E}[\mathbb{1}[J = a]]$ per agent (should be ~50% with shared policy)
2. **Tie Rate**: $\mathbb{E}[\mathbb{1}[J = \text{tie}]]$ (debates with no training signal)
3. **Accuracy** (if ground truth available): $\mathbb{E}[\mathbb{1}[s_a = G]]$

### Debate Quality Metrics
1. **Judge Confidence**: How often judge gives clear verdict (not tie)
2. **Solution Agreement**: How often $s_A = s_B$ before debate
3. **Correct Solution Wins**: $P(\text{win} \mid s_a = G)$
4. **Wrong Solution Wins**: $P(\text{win} \mid s_a \neq G)$ (concerning if high)

### Training Efficiency
1. **Rejection Rate**: % of debates that are ties (no training signal)
2. **Data Efficiency**: Effective training examples per API call

### Expected Learning Trajectory

| Step | Win Rate | Accuracy | Tie Rate | Notes |
|------|----------|----------|----------|-------|
| 0 | ~50% | X% | ~Y% | Random |
| 50 | ~50% | ~X% | ~Y% | Learning to argue |
| 100 | ~50% | X±5% | ~Y% | Better persuasion |
| 200 | ~50% | ? | ? | Does accuracy improve? |

Key question: Does training on debate winners improve **solution accuracy**, or just **argumentation skill**?

---

## Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Rounds | - | 3 | R1 (propose) + R2 (argue) + R3 (respond) |
| Rollouts per question | $N$ | 4 | More = more winners to train on |
| Max tokens per turn | $T$ | 512 | Per round |
| Temperature | $\tau$ | 0.8 | Higher = more exploration |
| Learning rate | $\eta$ | $10^{-5}$ | |
| KL coefficient | $\beta_{\text{KL}}$ | 0.01 | |

---

## Flow Diagram

```
                    ┌──────────────────┐
                    │  Question Q      │
                    └────────┬─────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  ┌─────────────┐                         ┌─────────────┐
  │ Agent A     │                         │ Agent B     │
  │ R1: Propose │                         │ R1: Propose │
  │ → s_A FIXED │                         │ → s_B FIXED │
  └──────┬──────┘                         └──────┬──────┘
         │                                       │
         │◄──────────── CROSS-INJECT ───────────►│
         │                                       │
  ┌──────▼──────┐                         ┌──────▼──────┐
  │ Agent A     │                         │ Agent B     │
  │ R2: Argue   │                         │ R2: Argue   │
  │ for s_A     │                         │ for s_B     │
  └──────┬──────┘                         └──────┬──────┘
         │                                       │
         │◄──────────── CROSS-INJECT ───────────►│
         │                                       │
  ┌──────▼──────┐                         ┌──────▼──────┐
  │ Agent A     │                         │ Agent B     │
  │ R3: Respond │                         │ R3: Respond │
  │ to critique │                         │ to critique │
  └──────┬──────┘                         └──────┬──────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             ▼
                    ┌──────────────────┐
                    │      JUDGE       │
                    │ (separate LLM)   │
                    │                  │
                    │ Verdict: A/B/TIE │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │ A wins:     │     │ B wins:     │     │ Tie:        │
  │ Train on A  │     │ Train on B  │     │ Reject both │
  │ Reject B    │     │ Reject A    │     │             │
  └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Comparison: Rejection Sampling vs Binary Reward

| Aspect | Rejection Sampling | Binary Reward (+1/-1) |
|--------|-------------------|----------------------|
| Training data | Winners only | All trajectories |
| Gradient | Positive only | Positive + negative |
| Variance | Lower (filtered) | Higher |
| Sample efficiency | 50% (lose half) | 100% |
| Mode collapse risk | Lower | Higher (pushing away from losers) |
| Implementation | Simpler | More complex |

We use rejection sampling for simplicity and stability.
