"""
Async Ollama LLM client for RAG generation pipeline.
"""
import asyncio
import logging
import os
import time
from typing import Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Token usage statistics from Ollama generation."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total tokens used")


class LLMResponse(BaseModel):
    """Response from LLM generation."""

    answer: str = Field(..., description="Generated text response")
    token_usage: TokenUsage = Field(..., description="Token usage statistics")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    model: str = Field(..., description="Model name used for generation")


class AsyncOllamaClient:
    """
    Async Ollama client for generating text using local Ollama instance.
    
    Provides async interface for text generation with configurable parameters,
    retry logic, and token tracking.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        ollama_host: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 30,
        retry_attempts: int = 3,
    ):
        """
        Initialize async Ollama client.

        Args:
            model_name: Name of the Ollama model (e.g., "llama3.1:8b")
            ollama_host: Ollama server URL (default: http://localhost:11434)
                        Can override with OLLAMA_HOST environment variable
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            timeout: HTTP request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests (1-10)

        Raises:
            ValueError: If parameters are invalid
        """
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if retry_attempts < 1 or retry_attempts > 10:
            raise ValueError("retry_attempts must be between 1 and 10")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Determine Ollama host
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Ensure no trailing slash
        self.ollama_host = self.ollama_host.rstrip("/")

        # API endpoint
        self.api_url = f"{self.ollama_host}/api/generate"

        logger.info(
            f"Initialized AsyncOllamaClient - "
            f"model: {model_name}, "
            f"host: {self.ollama_host}, "
            f"temperature: {temperature}, "
            f"max_tokens: {max_tokens}, "
            f"timeout: {timeout}s, "
            f"retry_attempts: {retry_attempts}"
        )

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Ollama asynchronously.

        Args:
            prompt: Input prompt for generation
            temperature: Optional override for sampling temperature
            max_tokens: Optional override for max tokens

        Returns:
            LLMResponse with generated text and token usage

        Raises:
            ValueError: If prompt is empty
            httpx.RequestError: If request fails after retries
            Exception: If response parsing fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Use provided overrides or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        logger.info(f"Generating text (model: {self.model_name}, prompt_len: {len(prompt)})")

        return await self._generate_with_retry(prompt, temp, tokens)

    def _extract_token_usage(self, response: dict) -> TokenUsage:
        """
        Extract token usage from Ollama response.

        Args:
            response: Raw Ollama API response

        Returns:
            TokenUsage object with token counts

        Raises:
            KeyError: If required fields are missing or malformed
        """
        # Extract token counts (Ollama v0.5.x format)
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        return TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )

    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        Generate text with exponential backoff retry logic.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generation results

        Raises:
            httpx.RequestError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                return await self._call_ollama_api(prompt, temperature, max_tokens)

            except (httpx.TimeoutException, httpx.ConnectError, httpx.PoolTimeout) as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Ollama request failed (attempt {attempt + 1}/{self.retry_attempts}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Ollama request failed after {self.retry_attempts} attempts: {e}"
                    )

            except Exception as e:
                logger.error(f"Unexpected error in Ollama generation: {e}")
                raise

        # All retries exhausted
        raise httpx.RequestError(
            f"Failed to get response from Ollama after {self.retry_attempts} attempts",
            request=None,
        ) from last_error

    async def _call_ollama_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        Call Ollama API asynchronously.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with parsed Ollama response

        Raises:
            httpx.RequestError: If HTTP request fails
            Exception: If response parsing fails
        """
        start_time = time.time()

        request_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=request_payload,
                )
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama request failed: {e}")
            raise

        # Parse response
        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise

        # Extract generation
        answer = result.get("response", "").strip()
        if not answer:
            logger.warning("Ollama returned empty response")
            answer = ""

        # Extract token counts using helper method
        token_usage = self._extract_token_usage(result)
        prompt_tokens = token_usage.prompt_tokens
        completion_tokens = token_usage.completion_tokens
        total_tokens = token_usage.total_tokens

        generation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Generation complete - "
            f"tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}), "
            f"time: {generation_time_ms:.1f}ms"
        )

        return LLMResponse(
            answer=answer,
            token_usage=token_usage,
            generation_time_ms=generation_time_ms,
            model=self.model_name,
        )

    async def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text with provided context (RAG scenario).

        Args:
            query: User query
            context: Retrieved context for grounding
            system_prompt: System prompt for formatting
            temperature: Optional temperature override
            max_tokens: Optional token limit override

        Returns:
            LLMResponse with generated answer

        Raises:
            ValueError: If query or context is empty
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not context or not context.strip():
            logger.warning("Context is empty for RAG generation")

        # Build prompt with context
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            full_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        return await self.generate(full_prompt, temperature, max_tokens)
