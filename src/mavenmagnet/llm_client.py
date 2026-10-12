"""Local Ollama LLM client supporting robust retries, structured JSON, and timing diagnostics.

Never silently fabricates predictions on error and strictly distinguishes
connection errors, timeouts, and generation failures.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger("mavenmagnet.llm_client")


class OllamaClientError(Exception):
    """Base exception for all Ollama client failures."""


class OllamaConnectionError(OllamaClientError):
    """Raised when connection to Ollama daemon cannot be established."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when an inference request to Ollama times out."""


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the specified model is not installed or available."""


class OllamaGenerationError(OllamaClientError):
    """Raised when Ollama returns an error payload or HTTP failure."""


@dataclass(frozen=True)
class GenerationResult:
    """Output metadata and text from Ollama inference."""

    raw_response: str
    model: str
    duration_seconds: float
    total_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration_ns: Optional[int] = None

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation tokens per second if available."""
        if self.eval_count and self.eval_duration_ns and self.eval_duration_ns > 0:
            return float(self.eval_count) / (self.eval_duration_ns / 1e9)
        if self.eval_count and self.duration_seconds > 0:
            return float(self.eval_count) / self.duration_seconds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        d = asdict(self)
        d["tokens_per_second"] = round(self.tokens_per_second, 2)
        return d


class OllamaClient:
    """Client for local instruction-tuned LLM inference via Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Local Ollama HTTP daemon endpoint (default: http://localhost:11434).
            model: Model tag (e.g. 'qwen3:14b', 'qwen3.5:9b').
            temperature: Sampling temperature (0.0 for deterministic classification).
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum attempts for transient network or server errors.
            retry_backoff_seconds: Multiplier for backoff delays between retries.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff_seconds

    def check_health(self) -> bool:
        """Check if local Ollama daemon is reachable.

        Returns:
            True if daemon responds with HTTP 200, False otherwise.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """Query Ollama daemon for all currently installed models.

        Returns:
            List of model tag names.

        Raises:
            OllamaConnectionError: If connection cannot be established.
            OllamaTimeoutError: If request times out.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                res.raise_for_status()
                data = res.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Timed out connecting to Ollama at {self.base_url}") from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Could not connect to Ollama daemon at {self.base_url}: {e}") from e
        except Exception as e:
            raise OllamaClientError(f"Failed to query installed models: {e}") from e

    def is_model_installed(self, model_name: Optional[str] = None) -> bool:
        """Check if target model is present in installed models.

        Args:
            model_name: Model name to check; defaults to self.model.

        Returns:
            True if model is installed, False otherwise.
        """
        target = model_name or self.model
        try:
            installed = self.list_installed_models()
            # Match either exact tag or base name
            return target in installed or any(m.startswith(f"{target}:") for m in installed)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        format_json: bool = True,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Send prompt to local Ollama instance and receive response text.

        Never silently returns fake predictions on failure.

        Args:
            prompt: User/query prompt text.
            system_prompt: Optional system prompt constraining behavior.
            temperature: Sampling temperature; defaults to self.temperature.
            format_json: If True, enforce JSON structured output.
            json_schema: Optional schema dictionary for structured decoding.

        Returns:
            GenerationResult instance.

        Raises:
            OllamaModelNotFoundError: If model is not installed.
            OllamaTimeoutError: If inference times out.
            OllamaConnectionError: If connection to daemon is lost.
            OllamaGenerationError: If server responds with generation error.
        """
        temp = self.temperature if temperature is None else temperature

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if json_schema is not None:
            payload["format"] = json_schema
        elif format_json:
            payload["format"] = "json"

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/api/generate", json=payload)

                    if resp.status_code == 404:
                        # Model not found: do not retry
                        raise OllamaModelNotFoundError(
                            f"Model '{self.model}' is not installed in Ollama. "
                            f"Run 'ollama pull {self.model}' to make it available."
                        )

                    resp.raise_for_status()
                    data = resp.json()
                    elapsed = time.perf_counter() - start_time

                    if "error" in data:
                        raise OllamaGenerationError(f"Ollama returned error: {data['error']}")

                    raw_text = data.get("response", "")
                    res = GenerationResult(
                        raw_response=raw_text,
                        model=data.get("model", self.model),
                        duration_seconds=elapsed,
                        total_duration_ns=data.get("total_duration"),
                        load_duration_ns=data.get("load_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count"),
                        eval_duration_ns=data.get("eval_duration"),
                    )

                    logger.info(
                        "Ollama inference completed for model=%s in %.2fs (eval_tokens=%s, speed=%.1f tok/s)",
                        res.model,
                        res.duration_seconds,
                        res.eval_count,
                        res.tokens_per_second,
                    )
                    return res

            except (OllamaModelNotFoundError, OllamaGenerationError):
                raise
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout}s on attempt {attempt}/{self.max_retries}"
                )
                logger.warning(str(last_error))
            except httpx.ConnectError as e:
                last_error = OllamaConnectionError(
                    f"Failed to connect to Ollama daemon at {self.base_url} on attempt {attempt}/{self.max_retries}: {e}"
                )
                logger.warning(str(last_error))
            except httpx.HTTPStatusError as e:
                last_error = OllamaGenerationError(
                    f"Ollama returned HTTP {e.response.status_code} error: {e.response.text}"
                )
                logger.warning(str(last_error))
            except Exception as e:
                last_error = OllamaClientError(f"Unexpected error during Ollama inference: {e}")
                logger.warning(str(last_error))

            if attempt < self.max_retries:
                sleep_time = self.retry_backoff * attempt
                logger.info("Retrying Ollama inference in %.1fs...", sleep_time)
                time.sleep(sleep_time)

        # Retries exhausted: never fabricate a fake prediction
        raise last_error or OllamaClientError(
            f"Ollama generation failed after {self.max_retries} attempts."
        )

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Convenience method returning raw text string from generate()."""
        return self.generate(prompt, **kwargs).raw_response
