"""Unit tests for OllamaClient using mocked HTTP responses (zero live LLM dependency)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from mavenmagnet.llm_client import (
    GenerationResult,
    OllamaClient,
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


@pytest.fixture
def client() -> OllamaClient:
    """Fixture providing OllamaClient with minimal retry delay for fast test execution."""
    return OllamaClient(
        base_url="http://localhost:11434",
        model="qwen3:14b",
        timeout_seconds=5.0,
        max_retries=2,
        retry_backoff_seconds=0.01,
    )


def test_check_health_mocked_success(client: OllamaClient) -> None:
    """Verify check_health returns True when daemon responds with HTTP 200."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_ctx.get.return_value = MagicMock(status_code=200)

        assert client.check_health() is True


def test_check_health_mocked_failure(client: OllamaClient) -> None:
    """Verify check_health returns False when connection fails."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_ctx.get.side_effect = httpx.ConnectError("Connection refused")

        assert client.check_health() is False


def test_list_installed_models_mocked(client: OllamaClient) -> None:
    """Verify list_installed_models parses tags API response correctly."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen3:14b"},
                {"name": "qwen3.5:9b"},
                {"name": "gemma4:31b"},
            ]
        }
        mock_ctx.get.return_value = mock_resp

        models = client.list_installed_models()
        assert models == ["qwen3:14b", "qwen3.5:9b", "gemma4:31b"]
        assert client.is_model_installed("qwen3:14b") is True
        assert client.is_model_installed("non_existent_model") is False


def test_successful_generation_mocked(client: OllamaClient) -> None:
    """Verify successful inference response parsing and timing metrics."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "qwen3:14b",
            "response": '{"predictions": [{"level_1": "Acne", "level_2": "Acne Control"}]}',
            "eval_count": 25,
            "eval_duration": 1_000_000_000,  # 1 second
            "total_duration": 1_200_000_000,
        }
        mock_ctx.post.return_value = mock_resp

        result = client.generate("Classify this review")
        assert isinstance(result, GenerationResult)
        assert "predictions" in result.raw_response
        assert result.model == "qwen3:14b"
        assert result.eval_count == 25
        assert result.tokens_per_second == 25.0


def test_connection_error_raises_ollama_connection_error(client: OllamaClient) -> None:
    """Verify connection errors are distinguished and raise OllamaConnectionError."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_ctx.post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama daemon"):
            client.generate("Test prompt")


def test_timeout_error_raises_ollama_timeout_error(client: OllamaClient) -> None:
    """Verify timeouts are distinguished and raise OllamaTimeoutError."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_ctx.post.side_effect = httpx.ReadTimeout("Read timed out")

        with pytest.raises(OllamaTimeoutError, match="timed out"):
            client.generate("Test prompt")


def test_model_not_found_raises_informative_error_without_retry(client: OllamaClient) -> None:
    """Verify HTTP 404 immediately raises OllamaModelNotFoundError instructing user on pull command."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_resp = MagicMock(status_code=404)
        mock_ctx.post.return_value = mock_resp

        with pytest.raises(OllamaModelNotFoundError, match="ollama pull qwen3:14b"):
            client.generate("Test prompt")

        # Must not retry when model does not exist
        assert mock_ctx.post.call_count == 1


def test_server_error_retry_and_recovery(client: OllamaClient) -> None:
    """Verify transient server errors trigger retries and recover on subsequent success."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx

        # First attempt: 500 Internal Server Error
        err_resp = MagicMock(status_code=500, text="Internal server error")
        req = MagicMock()
        err_exc = httpx.HTTPStatusError("Server error", request=req, response=err_resp)

        # Second attempt: 200 OK
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"model": "qwen3:14b", "response": '{"predictions": []}'}

        mock_ctx.post.side_effect = [err_exc, ok_resp]

        result = client.generate("Test prompt")
        assert result.raw_response == '{"predictions": []}'
        assert mock_ctx.post.call_count == 2


def test_never_silently_return_fake_prediction(client: OllamaClient) -> None:
    """CRITICAL: Verify client never returns fabricated or empty predictions on failure."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_ctx.post.side_effect = httpx.ConnectError("Daemon down")

        with pytest.raises(OllamaConnectionError):
            client.generate("Classify review")


def test_structured_json_formatting_payload(client: OllamaClient) -> None:
    """Verify format_json flag passes correct JSON formatting constraint to Ollama."""
    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_ctx
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"model": "qwen3:14b", "response": "{}"}
        mock_ctx.post.return_value = mock_resp

        client.generate("Prompt text", format_json=True)
        call_payload = mock_ctx.post.call_args[1]["json"]
        assert call_payload["format"] == "json"

        # Explicit schema passing
        schema = {"type": "object", "properties": {"predictions": {"type": "array"}}}
        client.generate("Prompt text", json_schema=schema)
        call_payload2 = mock_ctx.post.call_args[1]["json"]
        assert call_payload2["format"] == schema
