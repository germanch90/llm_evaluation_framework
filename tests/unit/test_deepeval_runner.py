from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.deepeval_runner import (
    DeepevalRunner,
    MetricThresholds,
    TestCase,
    load_test_cases,
)
from src.pipeline.rag_pipeline import RAGResult, RetrievedChunk


def dummy_result():
    chunk = RetrievedChunk(
        id="c1",
        score=0.9,
        content="context text",
        metadata={},
    )
    return RAGResult(
        question="q",
        answer="answer",
        contexts=[chunk],
        llm_response=None,
        metrics_ms={},
    )


def test_runner_builds_llm_test_cases(tmp_path):
    pipeline = MagicMock()
    pipeline.query.return_value = dummy_result()
    runner = DeepevalRunner(pipeline, MetricThresholds(), eval_model="mock-model")

    case = TestCase("What?", "Ground truth", [], "dataset", "type", "easy")

    with patch("src.evaluation.deepeval_runner.AnswerRelevancyMetric") as mock_ans, \
        patch("src.evaluation.deepeval_runner.FaithfulnessMetric") as mock_faith, \
        patch("src.evaluation.deepeval_runner.ContextualRelevancyMetric") as mock_ctx, \
        patch("src.evaluation.deepeval_runner.evaluate") as mock_eval:
        mock_ans.return_value = MagicMock()
        mock_faith.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        runner.run([case])

    mock_ans.assert_called_with(
        threshold=runner.thresholds.answer_relevancy, model="mock-model"
    )
    mock_faith.assert_called_with(
        threshold=runner.thresholds.faithfulness, model="mock-model"
    )
    mock_ctx.assert_called_with(
        threshold=runner.thresholds.contextual_relevancy, model="mock-model"
    )

    args, kwargs = mock_eval.call_args
    llm_case = kwargs["test_cases"][0]
    assert llm_case.input == "What?"
    assert llm_case.expected_output == "Ground truth"


def test_load_test_cases_json(tmp_path):
    data = [
        {
            "question": "q1",
            "answer": "a1",
            "sources": ["s1"],
            "dataset": "demo",
            "question_type": "type",
            "difficulty": "easy",
        }
    ]
    path = tmp_path / "cases.json"
    path.write_text(__import__("json").dumps(data), encoding="utf-8")
    cases = load_test_cases(path)
    assert cases[0].question == "q1"
