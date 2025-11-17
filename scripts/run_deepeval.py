#!/usr/bin/env python3
"""
Execute DeepEval metrics over prepared test cases.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from src.evaluation.deepeval_runner import (
    DeepevalRunner,
    MetricThresholds,
    load_test_cases,
)
from src.pipeline.rag_pipeline import build_pipeline


def load_thresholds(config_path: Path) -> MetricThresholds:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = config.get("evaluation", {}).get("metrics", {})
    return MetricThresholds(
        answer_relevancy=metrics.get("answer_relevancy", {}).get("threshold", 0.7),
        faithfulness=metrics.get("faithfulness", {}).get("threshold", 0.8),
        contextual_relevancy=metrics.get("contextual_relevancy", {}).get("threshold", 0.6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval metrics.")
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path("data") / "test_cases.json",
        help="Path to JSON produced by scripts/prepare_test_data.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config") / "config.yaml",
        help="Path to config file for threshold defaults",
    )
    parser.add_argument(
        "--eval-model",
        default="gpt-4o-mini",
        help="LLM used by DeepEval metrics (requires provider API key)",
    )
    parser.add_argument(
        "--eval-provider",
        choices=["openai", "ollama", "google", "groq"],
        default="openai",
        help="Which backend to score metrics with.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://ollama:11434",
        help="Ollama host to use when --eval-provider=ollama.",
    )
    args = parser.parse_args()

    cases = load_test_cases(args.test_cases)
    if not cases:
        raise SystemExit(f"No test cases found in {args.test_cases}")

    thresholds = load_thresholds(args.config)
    pipeline = build_pipeline()

    if args.eval_provider == "ollama":
        from deepeval.models import OllamaModel

        eval_model = OllamaModel(model=args.eval_model, base_url=args.ollama_host)
    elif args.eval_provider == "google":
        from deepeval.models import GeminiModel

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY is required for google eval provider")
        eval_model = GeminiModel(model_name=args.eval_model, api_key=api_key)
    elif args.eval_provider == "groq":
        from src.evaluation.groq_model import GroqEvalModel

        eval_model = GroqEvalModel(model_name=args.eval_model)
    else:
        eval_model = args.eval_model

    runner = DeepevalRunner(pipeline, thresholds, eval_model=eval_model)
    runner.run(cases)


if __name__ == "__main__":
    main()
