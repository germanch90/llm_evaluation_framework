import json
from pathlib import Path

import pytest

from src.evaluation.test_data_builder import (
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MEDIUM,
    TestCase,
    determine_difficulty,
    load_qna_csv,
    parse_source_docs,
    select_test_cases,
    export_test_cases,
)


def test_determine_difficulty_mapping():
    assert determine_difficulty("Single-Doc RAG") == DIFFICULTY_EASY
    assert determine_difficulty("Single-Doc Multi-Chunk") == DIFFICULTY_MEDIUM
    assert determine_difficulty("Multi-Doc RAG") == DIFFICULTY_HARD


def test_parse_source_docs_splits_tokens():
    value = "*AAPL*; MSFT ; item"
    assert parse_source_docs(value) == ["AAPL", "MSFT", "item"]


def test_load_qna_csv(tmp_path: Path):
    csv_content = (
        "Question,Source Docs,Question Type,Source Chunk Type,Answer\n"
        '"What is X?","*Doc1*","Single-Doc RAG","Single Chunk","Answer text"\n'
    )
    csv_path = tmp_path / "qna_data.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    cases = load_qna_csv(csv_path, "demo")
    assert len(cases) == 1
    case = cases[0]
    assert case.dataset == "demo"
    assert case.sources == ["Doc1"]
    assert case.difficulty == DIFFICULTY_EASY


def test_select_test_cases_balances_buckets():
    cases = [
        TestCase("q1", "a", [], "d", "Single", DIFFICULTY_EASY),
        TestCase("q2", "a", [], "d", "Multi-Chunk", DIFFICULTY_MEDIUM),
        TestCase("q3", "a", [], "d", "Multi-Doc", DIFFICULTY_HARD),
        TestCase("q4", "a", [], "d", "Single", DIFFICULTY_EASY),
    ]
    selected = select_test_cases(cases, target_per_difficulty=1, max_total=2)
    assert len(selected) == 2
    assert {case.difficulty for case in selected} == {
        DIFFICULTY_EASY,
        DIFFICULTY_MEDIUM,
    }


def test_export_test_cases(tmp_path: Path):
    cases = [
        TestCase("q1", "a1", ["Doc1"], "dataset", "Single", DIFFICULTY_EASY),
    ]
    output = tmp_path / "cases.json"
    export_test_cases(cases, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data[0]["question"] == "q1"
