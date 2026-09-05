from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Iterable


_OFFICIAL_ZIP_URL = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"


@dataclass(frozen=True)
class QualityQuestion:
    question_id: str
    article_id: str
    title: str
    source: str
    topic: str
    article: str
    question: str
    correct_answer: str
    distractor_answer: str
    original_gold_label: int
    original_distractor_label: int
    split: str


def default_quality_data_dir() -> Path:
    env_path = os.environ.get("QUALITY_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser()
    candidates = [
        Path.cwd() / "data" / "quality",
        Path.cwd() / "quality_data",
        Path("/workspace/quality_data"),
        Path("/mnt/c/Users/raulc/Desktop/quality_data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_quality_data(data_dir: str | Path | None = None) -> Path:
    root = Path(data_dir).expanduser() if data_dir is not None else default_quality_data_dir()
    train_file = root / "QuALITY.v1.0.1.htmlstripped.train"
    if train_file.exists():
        return root

    import urllib.request
    import zipfile

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "QuALITY.v1.0.1.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(_OFFICIAL_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    if not train_file.exists():
        raise FileNotFoundError(f"Downloaded QuALITY but did not find {train_file}")
    return root


def _quality_split_path(root: Path, split: str) -> Path:
    path = root / f"QuALITY.v1.0.1.htmlstripped.{split}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing QuALITY split {split!r} at {path}. "
            "Set QUALITY_DATA_DIR or pass --quality-data-dir."
        )
    return path


def _best_distractor_label(question: dict, gold_label: int) -> int:
    candidates = [
        int(row["untimed_best_distractor"])
        for row in question.get("validation", [])
        if row.get("untimed_best_distractor") is not None
    ]
    candidates = [label for label in candidates if label != gold_label]
    if candidates:
        return Counter(candidates).most_common(1)[0][0]
    for label in range(1, len(question["options"]) + 1):
        if label != gold_label:
            return label
    raise ValueError("QuALITY question must have at least one distractor.")


def iter_quality_questions(
    *,
    data_dir: str | Path | None = None,
    split: str = "train",
    hard_only: bool = True,
    source: str | None = "Gutenberg",
    topic_contains: str | None = "Science fiction",
    download: bool = False,
) -> Iterable[QualityQuestion]:
    root = ensure_quality_data(data_dir) if download else (Path(data_dir).expanduser() if data_dir is not None else default_quality_data_dir())
    path = _quality_split_path(root, split)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            article_row = json.loads(line)
            if source is not None and article_row.get("source") != source:
                continue
            topic = str(article_row.get("topic", ""))
            if topic_contains is not None and topic_contains not in topic:
                continue
            for question in article_row.get("questions", []):
                if hard_only and int(question.get("difficult", 0)) != 1:
                    continue
                gold_label = int(question["gold_label"])
                distractor_label = _best_distractor_label(question, gold_label)
                options = question["options"]
                yield QualityQuestion(
                    question_id=str(question["question_unique_id"]),
                    article_id=str(article_row["article_id"]),
                    title=str(article_row.get("title", "")),
                    source=str(article_row.get("source", "")),
                    topic=topic,
                    article=str(article_row["article"]),
                    question=str(question["question"]).strip(),
                    correct_answer=str(options[gold_label - 1]).strip(),
                    distractor_answer=str(options[distractor_label - 1]).strip(),
                    original_gold_label=gold_label,
                    original_distractor_label=distractor_label,
                    split=split,
                )


def load_quality_questions(
    *,
    data_dir: str | Path | None = None,
    split: str = "train",
    hard_only: bool = True,
    source: str | None = "Gutenberg",
    topic_contains: str | None = "Science fiction",
    download: bool = False,
) -> list[QualityQuestion]:
    return list(
        iter_quality_questions(
            data_dir=data_dir,
            split=split,
            hard_only=hard_only,
            source=source,
            topic_contains=topic_contains,
            download=download,
        )
    )


def sample_quality_questions(
    questions: list[QualityQuestion],
    *,
    n: int,
    seed: int | None,
) -> list[QualityQuestion]:
    if n <= 0:
        return []
    if not questions:
        raise ValueError("Cannot sample from an empty QuALITY question set.")
    rng = random.Random(seed)
    if n <= len(questions):
        return rng.sample(questions, n)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    out: list[QualityQuestion] = []
    while len(out) < n:
        out.extend(shuffled)
    return out[:n]
