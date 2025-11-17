"""
DeepEval-compatible wrapper for Groq's OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import requests
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqEvalModel")
        self.base_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()
        super().__init__(model_name=model_name)

    def load_model(self, *args, **kwargs) -> Any:
        """No client object needed since we call the HTTP API directly."""
        return None

    def _generate(self, prompt: str, schema: Any = None) -> Any:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # enrich message
            try:
                detail = response.json()
            except Exception:  # pragma: no cover - best effort
                detail = response.text
            raise requests.HTTPError(f"{exc} | detail={detail}") from exc

        data = response.json()
        message = (data["choices"][0]["message"]["content"] or "").strip()

        if schema is None:
            return message

        # Attempt to parse model output into the requested schema
        try:
            return schema.parse_raw(message)
        except Exception:
            # Try to interpret as JSON dict
            import json

            parsed = json.loads(message)
            return schema.parse_obj(parsed)

    def generate(self, prompt: str, **kwargs) -> Any:
        schema = kwargs.get("schema")
        return self._generate(prompt, schema=schema)

    async def a_generate(self, prompt: str, **kwargs) -> Any:
        schema = kwargs.get("schema")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, prompt, schema)

    def get_model_name(self, *args, **kwargs) -> str:
        return self.model_name
