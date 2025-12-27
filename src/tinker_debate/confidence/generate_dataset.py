"""
Binomial probability problems.

Flip a p-biased coin n times, ask about the count of heads.
Ground truth = the probability.
"""
import json
import random
from math import comb
from pathlib import Path

SEED = 42


def binomial_prob(n: int, p: float, k: int, op: str) -> float:
    """P(heads op k) for n flips of p-biased coin."""
    def term(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    if op == ">=":
        return sum(term(i) for i in range(k, n + 1))
    if op == "<=":
        return sum(term(i) for i in range(k + 1))
    if op == "=":
        return term(k)
    if op == ">":
        return sum(term(i) for i in range(k + 1, n + 1))
    if op == "<":
        return sum(term(i) for i in range(k))


def generate_problem(rng: random.Random) -> dict:
    n = rng.randint(5, 20)
    p = rng.randint(15, 85) / 100
    k = rng.randint(1, n - 1)
    op = rng.choice([">=", "<=", "=", ">", "<"])

    prob = binomial_prob(n, p, k, op)
    answer = "yes" if prob >= 0.5 else "no"
    confidence = prob if prob >= 0.5 else 1 - prob

    op_text = {">=": "at least", "<=": "at most", "=": "exactly", ">": "more than", "<": "fewer than"}[op]
    question = (
        f"A coin with {int(p*100)}% chance of heads was flipped {n} times. "
        f"Were there {op_text} {k} heads?"
    )
    return {"question": question, "answer": answer, "probability": round(confidence, 4)}


def generate_dataset(n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    return [generate_problem(rng) for _ in range(n)]


def main():
    problems = generate_dataset(100)
    out_path = Path(__file__).parent / "questions.json"
    with open(out_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Generated {len(problems)} -> {out_path}")


if __name__ == "__main__":
    main()
