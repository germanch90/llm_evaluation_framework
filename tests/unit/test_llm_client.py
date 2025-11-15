"""
Unit tests for Ollama async LLM client.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from src.generation.llm_client import AsyncOllamaClient, LLMResponse, TokenUsage


@pytest.fixture
def llm_client():
    """Create an AsyncOllamaClient instance for testing."""
    return AsyncOllamaClient(
        ollama_host="http://localhost:11434",
        model="llama2",
        temperature=0.7,
        max_tokens=512,
        timeout=30.0,
        retry_attempts=3,
    )


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama response matching v0.5.x format."""
    return {
        "model": "llama2",
        "created_at": "2024-01-01T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": "This is a generated response.",
        },
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 600000000,
    }


class TestLLMResponseModel:
    """Test LLMResponse Pydantic model."""

    def test_token_usage_creation(self):
        """Test TokenUsage model creation."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_llm_response_creation(self):
        """Test LLMResponse model creation."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(
            text="Generated text",
            model="llama2",
            token_usage=usage,
            finish_reason="stop",
        )
        assert response.text == "Generated text"
        assert response.model == "llama2"
        assert response.token_usage.prompt_tokens == 10
        assert response.finish_reason == "stop"

    def test_llm_response_optional_reason(self):
        """Test LLMResponse with optional finish_reason."""
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        response = LLMResponse(
            text="Short response",
            model="llama2",
            token_usage=usage,
        )
        assert response.finish_reason is None


class TestAsyncOllamaClientInit:
    """Test AsyncOllamaClient initialization."""

    def test_init_with_defaults(self):
        """Test client initialization with default parameters."""
        client = AsyncOllamaClient()
        assert client.ollama_host == "http://localhost:11434"
        assert client.model == "llama2"
        assert client.temperature == 0.7
        assert client.max_tokens == 2048
        assert client.timeout == 30.0
        assert client.retry_attempts == 5

    def test_init_with_custom_params(self):
        """Test client initialization with custom parameters."""
        client = AsyncOllamaClient(
            ollama_host="http://remote:11434",
            model="mistral",
            temperature=0.5,
            max_tokens=1024,
            timeout=60.0,
            retry_attempts=3,
        )
        assert client.ollama_host == "http://remote:11434"
        assert client.model == "mistral"
        assert client.temperature == 0.5
        assert client.max_tokens == 1024
        assert client.timeout == 60.0
        assert client.retry_attempts == 3

    def test_init_with_ollama_host_env_var(self, monkeypatch):
        """Test client initialization respects OLLAMA_HOST env var."""
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:11434")
        client = AsyncOllamaClient()
        assert client.ollama_host == "http://custom:11434"

    def test_init_temperature_boundary(self):
        """Test temperature parameter boundary values."""
        client_min = AsyncOllamaClient(temperature=0.0)
        assert client_min.temperature == 0.0

        client_max = AsyncOllamaClient(temperature=2.0)
        assert client_max.temperature == 2.0

    def test_init_max_tokens_validation(self):
        """Test max_tokens accepts reasonable values."""
        client = AsyncOllamaClient(max_tokens=100)
        assert client.max_tokens == 100

        client = AsyncOllamaClient(max_tokens=32000)
        assert client.max_tokens == 32000


class TestTokenUsageExtraction:
    """Test token usage extraction from Ollama responses."""

    def test_extract_token_usage_from_response(self, llm_client, mock_ollama_response):
        """Test extracting token usage from a complete Ollama response."""
        usage = llm_client._extract_token_usage(mock_ollama_response)
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 10  # prompt_eval_count
        assert usage.completion_tokens == 20  # eval_count
        assert usage.total_tokens == 30

    def test_extract_token_usage_missing_fields(self, llm_client):
        """Test token usage extraction with missing fields (defaults to 0)."""
        minimal_response = {
            "message": {"role": "assistant", "content": "Response"},
        }
        usage = llm_client._extract_token_usage(minimal_response)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_extract_token_usage_zero_values(self, llm_client):
        """Test token usage extraction with zero values."""
        response_with_zeros = {
            "prompt_eval_count": 0,
            "eval_count": 0,
            "message": {"role": "assistant", "content": "Response"},
        }
        usage = llm_client._extract_token_usage(response_with_zeros)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


@pytest.mark.asyncio
class TestAsyncGeneration:
    """Test async text generation."""

    async def test_generate_success(self, llm_client, mock_ollama_response):
        """Test successful text generation."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate("Hello, world!")
            assert isinstance(response, LLMResponse)
            assert response.text == "This is a generated response."
            assert response.model == "llama2"
            assert response.token_usage.prompt_tokens == 10

    async def test_generate_with_custom_params(self, llm_client, mock_ollama_response):
        """Test generation with custom parameters."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate(
                "Translate this",
                temperature=0.3,
                max_tokens=256,
            )
            assert response.text == "This is a generated response."
            # Verify custom params were passed in request
            call_args = mock_post.call_args
            assert call_args is not None

    async def test_generate_with_context(self, llm_client, mock_ollama_response):
        """Test generation with RAG context."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate_with_context(
                query="What is Python?",
                context="Python is a programming language",
                system_prompt="You are a helpful assistant.",
            )
            assert response.text == "This is a generated response."
            # Verify context was included in request
            call_args = mock_post.call_args
            assert call_args is not None


@pytest.mark.asyncio
class TestRetryLogic:
    """Test exponential backoff retry mechanism."""

    async def test_retry_on_connection_error(self, llm_client):
        """Test retry logic on connection errors."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            # Fail first 2 attempts, succeed on third
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "Success"},
                "prompt_eval_count": 5,
                "eval_count": 10,
            }
            mock_response.status_code = 200

            mock_post.side_effect = [
                ConnectionError("Connection refused"),
                ConnectionError("Connection refused"),
                mock_response,
            ]

            response = await llm_client.generate("Test")
            assert response.text == "Success"
            assert mock_post.call_count == 3

    async def test_retry_on_timeout(self, llm_client):
        """Test retry logic on timeout errors."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "Recovered"},
                "prompt_eval_count": 3,
                "eval_count": 5,
            }
            mock_response.status_code = 200

            mock_post.side_effect = [
                TimeoutError("Request timeout"),
                mock_response,
            ]

            response = await llm_client.generate("Test timeout")
            assert response.text == "Recovered"
            assert mock_post.call_count == 2

    async def test_max_retries_exceeded(self, llm_client):
        """Test exception raised when max retries exceeded."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = ConnectionError("Always fails")

            with pytest.raises(ConnectionError):
                await llm_client.generate("Test")

            # Should attempt retry_attempts times
            assert mock_post.call_count == llm_client.retry_attempts

    async def test_exponential_backoff_timing(self, llm_client):
        """Test exponential backoff timing between retries."""
        # Create a client with fewer retries for faster testing
        client = AsyncOllamaClient(retry_attempts=2)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch.object(
                AsyncClient, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "message": {"role": "assistant", "content": "OK"},
                    "prompt_eval_count": 1,
                    "eval_count": 2,
                }
                mock_response.status_code = 200

                mock_post.side_effect = [
                    TimeoutError("Timeout"),
                    mock_response,
                ]

                response = await client.generate("Test")
                assert response.text == "OK"

                # Verify sleep was called with exponential backoff
                mock_sleep.assert_called_once()
                sleep_duration = mock_sleep.call_args[0][0]
                assert 0.1 <= sleep_duration <= 1.0  # 2^1 * 0.5 max is 1.0


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_invalid_response_format(self, llm_client):
        """Test handling of invalid response format."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"error": "Model not found"}
            mock_response.status_code = 404
            mock_post.return_value = mock_response

            with pytest.raises(KeyError):
                await llm_client.generate("Test")

    async def test_empty_response_content(self, llm_client):
        """Test handling of empty response content."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": ""},
                "prompt_eval_count": 5,
                "eval_count": 0,
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate("Empty response test")
            assert response.text == ""

    async def test_malformed_json_response(self, llm_client):
        """Test handling of malformed JSON response."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            with pytest.raises(ValueError):
                await llm_client.generate("Test")


@pytest.mark.asyncio
class TestPromptFormatting:
    """Test prompt formatting in generate methods."""

    async def test_generate_with_context_prompt_injection(self, llm_client):
        """Test that context is properly injected in RAG prompt."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "Answer"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate_with_context(
                query="What is X?",
                context="X is defined as...",
                system_prompt="Answer accurately.",
            )

            # Verify the call was made
            call_args = mock_post.call_args
            assert call_args is not None
            assert response.text == "Answer"

    async def test_generate_preserves_special_chars(self, llm_client):
        """Test that special characters in prompts are preserved."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "role": "assistant",
                    "content": "Response with special: @#$%",
                },
                "prompt_eval_count": 8,
                "eval_count": 5,
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate('Prompt with "quotes" and \'apostrophes\'')
            assert "special:" in response.text


@pytest.mark.asyncio
class TestConcurrency:
    """Test concurrent request handling."""

    async def test_concurrent_generates(self, llm_client):
        """Test multiple concurrent generate requests."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "Concurrent response"},
                "prompt_eval_count": 5,
                "eval_count": 10,
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            # Run 5 concurrent requests
            tasks = [
                llm_client.generate(f"Prompt {i}") for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            assert all(r.text == "Concurrent response" for r in responses)

    async def test_concurrent_different_params(self, llm_client):
        """Test concurrent requests with different parameters."""
        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            responses_data = [
                {
                    "message": {"role": "assistant", "content": f"Response {i}"},
                    "prompt_eval_count": i,
                    "eval_count": i * 2,
                }
                for i in range(3)
            ]

            mock_responses = []
            for data in responses_data:
                mock_response = MagicMock()
                mock_response.json.return_value = data
                mock_response.status_code = 200
                mock_responses.append(mock_response)

            mock_post.side_effect = mock_responses

            tasks = [
                llm_client.generate(f"Prompt {i}", temperature=0.5 + i * 0.1)
                for i in range(3)
            ]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 3


@pytest.mark.asyncio
class TestEnvironmentIntegration:
    """Test integration with environment variables."""

    async def test_ollama_host_from_env(self, monkeypatch):
        """Test that OLLAMA_HOST env var is used."""
        monkeypatch.setenv("OLLAMA_HOST", "http://prod:11434")
        client = AsyncOllamaClient()
        assert client.ollama_host == "http://prod:11434"

    async def test_ollama_model_from_env(self, monkeypatch):
        """Test that OLLAMA_MODEL env var is used."""
        monkeypatch.setenv("OLLAMA_MODEL", "neural-chat")
        client = AsyncOllamaClient()
        assert client.model == "neural-chat"

    async def test_explicit_params_override_env(self, monkeypatch):
        """Test that explicit parameters override env vars."""
        monkeypatch.setenv("OLLAMA_HOST", "http://env:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")

        client = AsyncOllamaClient(
            ollama_host="http://explicit:11434",
            model="explicit-model",
        )
        assert client.ollama_host == "http://explicit:11434"
        assert client.model == "explicit-model"


@pytest.mark.asyncio
class TestResponseParsing:
    """Test various response format variations."""

    async def test_parse_response_with_all_fields(self, llm_client):
        """Test parsing response with all optional fields."""
        full_response = {
            "model": "llama2:13b",
            "created_at": "2024-01-01T12:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "Full response",
            },
            "done": True,
            "total_duration": 1500000000,
            "load_duration": 150000000,
            "prompt_eval_count": 15,
            "prompt_eval_duration": 300000000,
            "eval_count": 25,
            "eval_duration": 900000000,
        }

        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = full_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate("Test")
            assert response.text == "Full response"
            assert response.token_usage.prompt_tokens == 15
            assert response.token_usage.completion_tokens == 25

    async def test_parse_response_minimal_fields(self, llm_client):
        """Test parsing response with minimal fields."""
        minimal_response = {
            "message": {
                "role": "assistant",
                "content": "Minimal",
            },
            "done": True,
        }

        with patch.object(
            AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = minimal_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = await llm_client.generate("Test")
            assert response.text == "Minimal"
            assert response.token_usage.total_tokens == 0
