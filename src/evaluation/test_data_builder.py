"""
Utilities for preparing DeepEval-style test cases from KG-RAG datasets.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

DIFFICULTY_HARD = "hard"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_EASY = "easy"


@dataclass
class TestCase:
    question: str
    answer: str
    sources: List[str]
    dataset: str
    question_type: str
    difficulty: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def determine_difficulty(question_type: str) -> str:
    """Map question_type text to a difficulty bucket."""
    normalized = (question_type or "").lower()
    if "multi-doc" in normalized:
        return DIFFICULTY_HARD
    if "multi-chunk" in normalized:
        return DIFFICULTY_MEDIUM
    return DIFFICULTY_EASY


def parse_source_docs(raw_value: str) -> List[str]:
    """Split `Source Docs` column into clean identifiers."""
    if not raw_value:
        return []
    tokens = re.split(r"[;,]", raw_value)
    cleaned = [token.strip().strip("*") for token in tokens]
    return [token for token in cleaned if token]


def load_qna_csv(csv_path: Path, dataset_name: str) -> List[TestCase]:
    """Parse a qna_data CSV into TestCase objects."""
    cases: List[TestCase] = []
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                continue
            q_type = (row.get("Question Type") or "").strip()
            case = TestCase(
                question=question,
                answer=answer,
                sources=parse_source_docs(row.get("Source Docs", "")),
                dataset=dataset_name,
                question_type=q_type,
                difficulty=determine_difficulty(q_type),
            )
            cases.append(case)
    return cases


def select_test_cases(
    cases: Iterable[TestCase],
    target_per_difficulty: int = 8,
    max_total: int = 24,
) -> List[TestCase]:
    """Pick a balanced subset of cases per difficulty."""
    buckets: Dict[str, List[TestCase]] = {
        DIFFICULTY_EASY: [],
        DIFFICULTY_MEDIUM: [],
        DIFFICULTY_HARD: [],
    }
    seen_questions = set()
    for case in cases:
        if case.question in seen_questions:
            continue
        seen_questions.add(case.question)
        buckets.setdefault(case.difficulty, []).append(case)

    selected: List[TestCase] = []
    for difficulty in (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD):
        if len(selected) >= max_total:
            break
        bucket = buckets.get(difficulty, [])
        count = min(target_per_difficulty, len(bucket), max_total - len(selected))
        selected.extend(bucket[:count])

    return selected


def export_test_cases(cases: List[TestCase], output_path: Path) -> None:
    """Write selected cases to JSON."""
    payload = [case.to_dict() for case in cases]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
