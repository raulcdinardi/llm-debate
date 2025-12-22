"""Dataset loading and formatting for debate training.

Each loader returns a list of dicts with:
  - question: str (formatted question text)
  - ground_truth: str (correct answer)
"""

from __future__ import annotations

import random
from typing import Literal


GPQASplit = Literal["gpqa_diamond", "gpqa_extended", "gpqa_main"]
DatasetName = Literal["gpqa_diamond", "gpqa_extended", "gpqa_main", "test"]


# Simple test dataset - no API calls needed
TEST_QUESTIONS = [
    {"question": "What is 7 * 8?", "ground_truth": "56"},
    {"question": "What is 12 + 15?", "ground_truth": "27"},
    {"question": "What is 100 / 4?", "ground_truth": "25"},
    {"question": "What is 9 * 9?", "ground_truth": "81"},
    {"question": "What is 144 / 12?", "ground_truth": "12"},
    {"question": "What is 17 + 28?", "ground_truth": "45"},
    {"question": "What is 6 * 7?", "ground_truth": "42"},
    {"question": "What is 50 - 23?", "ground_truth": "27"},
]


def load_test_dataset() -> list[dict]:
    """Load simple arithmetic test dataset (no API calls)."""
    return TEST_QUESTIONS.copy()


def format_gpqa_question(row: dict) -> dict:
    """Format a single GPQA row into question + ground_truth."""
    question_text = row["Question"]

    # GPQA has randomized answer order; correct answer indicated by "Correct Answer" field
    choices = [
        ("A", row["Correct Answer"]),
        ("B", row["Incorrect Answer 1"]),
        ("C", row["Incorrect Answer 2"]),
        ("D", row["Incorrect Answer 3"]),
    ]

    # Shuffle choices so correct answer isn't always A
    random.shuffle(choices)

    # Find which letter the correct answer ended up at
    correct_letter = None
    for letter, text in choices:
        if text == row["Correct Answer"]:
            correct_letter = letter
            break

    # Build formatted question
    formatted = f"{question_text}\n\n"
    for letter, text in choices:
        formatted += f"({letter}) {text}\n"

    return {
        "question": formatted.strip(),
        "ground_truth": correct_letter,
    }


def load_gpqa(split: GPQASplit = "gpqa_diamond", seed: int | None = None) -> list[dict]:
    """Load GPQA dataset and format questions.

    Args:
        split: Which GPQA split to use (diamond=198, extended=546, main=448)
        seed: Random seed for shuffling answer choices

    Returns:
        List of {"question": str, "ground_truth": str} dicts
    """
    from datasets import load_dataset

    if seed is not None:
        random.seed(seed)

    ds = load_dataset("Idavidrein/gpqa", split, split="train")

    formatted = [format_gpqa_question(row) for row in ds]
    return formatted


def sample_questions(
    dataset: list[dict],
    n: int,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """Sample n questions from dataset.

    Returns list of (question, ground_truth) tuples.
    """
    if seed is not None:
        random.seed(seed)

    sampled = random.sample(dataset, min(n, len(dataset)))
    return [(d["question"], d["ground_truth"]) for d in sampled]
