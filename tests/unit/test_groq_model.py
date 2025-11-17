import json

import pytest

from pydantic import BaseModel

from src.evaluation.groq_model import GroqEvalModel


def test_groq_eval_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GroqEvalModel(api_key=None)


def test_groq_eval_model_calls_api(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class DummyResponse:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"content": " hi "}},
                ]
            }

        def raise_for_status(self):
            return None

    class DummySession:
        def __init__(self):
            self.last_request = None

        def post(self, url, json=None, headers=None, timeout=None):
            self.last_request = (url, json, headers, timeout)
            return DummyResponse()

    session = DummySession()
    monkeypatch.setattr(
        "src.evaluation.groq_model.requests.Session",
        lambda: session,
    )

    model = GroqEvalModel(model_name="llama3")
    output = model.generate("hello")
    assert output == "hi"
    url, payload, headers, timeout = session.last_request
    assert "chat/completions" in url
    assert payload["model"] == "llama3"
    assert payload["messages"][0]["content"] == "hello"
    assert headers["Authorization"] == "Bearer test-key"


def test_groq_eval_model_parses_schema(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class DummyResponse:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"value": 42}'}},
                ]
            }

        def raise_for_status(self):
            return None

    class DummySession:
        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(
        "src.evaluation.groq_model.requests.Session",
        lambda: DummySession(),
    )

    class Schema(BaseModel):
        value: int

    model = GroqEvalModel(model_name="llama3")
    result = model.generate("hi", schema=Schema)
    assert isinstance(result, Schema)
    assert result.value == 42
