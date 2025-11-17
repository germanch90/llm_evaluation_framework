#!/usr/bin/env python3
"""
Generate DeepEval-ready test cases from KG-RAG datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.evaluation.test_data_builder import (
    export_test_cases,
    load_qna_csv,
    select_test_cases,
)  # pylint: disable=wrong-import-position


def find_qna_files(dataset_root: Path) -> List[Path]:
    """Locate qna_data*.csv files across datasets."""
    return sorted(dataset_root.rglob("qna_data*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DeepEval test data from KG-RAG datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / "kg-rag-dataset",
        help="Root directory where KG-RAG datasets are stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "test_cases.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--per-difficulty",
        type=int,
        default=8,
        help="Target number of test cases per difficulty bucket.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=24,
        help="Maximum number of test cases overall.",
    )
    args = parser.parse_args()

    qna_files = find_qna_files(args.dataset_root)
    if not qna_files:
        raise SystemExit(
            f"No qna_data*.csv files found under {args.dataset_root}. "
            "Run scripts/download_dataset.py first."
        )

    all_cases = []
    for csv_path in qna_files:
        dataset_name = csv_path.parents[3].name  # dataset_root/<dataset>/data/...
        all_cases.extend(load_qna_csv(csv_path, dataset_name))

    selected = select_test_cases(
        all_cases, target_per_difficulty=args.per_difficulty, max_total=args.max_total
    )
    export_test_cases(selected, args.output)

    print(
        f"Prepared {len(selected)} test cases across {len(qna_files)} files → {args.output}"
    )


if __name__ == "__main__":
    main()
