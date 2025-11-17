"""
Integration with DeepEval metrics for RAG evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.evaluation.test_data_builder import TestCase
from src.pipeline.rag_pipeline import RAGPipeline


@dataclass
class MetricThresholds:
    answer_relevancy: float = 0.7
    faithfulness: float = 0.8
    contextual_relevancy: float = 0.6


class DeepevalRunner:
    """Wrapper that runs DeepEval metrics over pipeline results."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        thresholds: MetricThresholds,
        eval_model: Union[str, DeepEvalBaseLLM],
    ):
        self.pipeline = pipeline
        self.thresholds = thresholds
        self.eval_model = eval_model

    def build_metrics(self):
        """Instantiate DeepEval metrics with configured thresholds."""
        return [
            AnswerRelevancyMetric(
                threshold=self.thresholds.answer_relevancy,
                model=self.eval_model,
            ),
            FaithfulnessMetric(
                threshold=self.thresholds.faithfulness,
                model=self.eval_model,
            ),
            ContextualRelevancyMetric(
                threshold=self.thresholds.contextual_relevancy,
                model=self.eval_model,
            ),
        ]

    def build_test_case(self, test_case: TestCase) -> LLMTestCase:
        """Run the pipeline and adjust data into LLMTestCase."""
        result = self.pipeline.query(test_case.question)
        retrieval_context = [chunk.content for chunk in result.contexts]
        return LLMTestCase(
            input=test_case.question,
            actual_output=result.answer,
            expected_output=test_case.answer,
            retrieval_context=retrieval_context,
        )

    def run(self, cases: Sequence[TestCase]) -> None:
        """Execute metrics evaluation via DeepEval."""
        llm_test_cases = [self.build_test_case(case) for case in cases]
        metrics = self.build_metrics()
        evaluate(metrics=metrics, test_cases=llm_test_cases)


def load_test_cases(path: Path) -> List[TestCase]:
    """Load TestCase objects from JSON produced by prepare_test_data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        TestCase(
            question=item["question"],
            answer=item["answer"],
            sources=item.get("sources", []),
            dataset=item.get("dataset", "unknown"),
            question_type=item.get("question_type", ""),
            difficulty=item.get("difficulty", ""),
        )
        for item in payload
    ]
    return cases
