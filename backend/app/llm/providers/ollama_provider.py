"""
Ollama/Local LLM Provider for Razorpay CloseLoop Phase 12A.

Implements the LLMProvider interface for Ollama and local LLM APIs.
Uses Ollama's OpenAI-compatible endpoint for maximum compatibility.

Configuration comes from environment variables via OllamaConfig.
No secrets are hardcoded.

IMPORTANT:
This provider generates text only.
It does NOT authorize financial actions.
"""

import time
from typing import Any, Dict, Optional

import httpx

from app.llm.config import OllamaConfig
from app.llm.providers.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMHealthStatus,
    LLMProvider,
    LLMProviderError,
    LLMProviderType,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
)


class OllamaProvider(LLMProvider):
    """Ollama/local LLM provider using httpx.

    Supports:
    - Ollama local server
    - Any OpenAI-compatible local endpoint

    Does NOT:
    - Authorize financial actions
    - Bypass guardrails
    - Access the database
    """

    def __init__(self, config: OllamaConfig):
        """Initialize the Ollama provider.

        Args:
            config: Ollama configuration (from env vars).

        Raises:
            LLMConfigError: If configuration is invalid.
        """
        if not config.base_url:
            raise LLMConfigError("Ollama base URL is required", provider="ollama")
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OLLAMA

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def config(self) -> OllamaConfig:
        return self._config

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion using the Ollama-compatible API.

        Uses Ollama's /v1/chat/completions endpoint (OpenAI-compatible).

        Args:
            request: Structured LLM request.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMTimeoutError: On timeout.
            LLMConnectionError: On connection failure.
            LLMResponseError: On invalid response.
        """
        model = request.model or self._config.model
        temperature = (
            request.temperature
            if request.temperature is not None
            else self._config.temperature
        )
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self._config.max_tokens
        )

        # Build request payload (OpenAI-compatible format)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        start_time = time.monotonic()

        try:
            client = await self._get_client()
            timeout_val = request.timeout or self._config.timeout

            response = await client.post(
                "/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(timeout_val),
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama request timed out after {timeout_val}s",
                provider="ollama",
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self._config.base_url}. "
                "Is the Ollama server running?",
                provider="ollama",
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                f"Ollama API returned error {e.response.status_code}: {e.response.text}",
                provider="ollama",
                details={"status_code": e.response.status_code},
            ) from e
        except httpx.RequestError as e:
            raise LLMConnectionError(
                f"Ollama request failed: {str(e)}",
                provider="ollama",
            ) from e

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Parse response
        try:
            data = response.json()
        except Exception as e:
            raise LLMResponseError(
                "Failed to parse Ollama response as JSON",
                provider="ollama",
            ) from e

        if "choices" not in data or len(data["choices"]) == 0:
            raise LLMResponseError(
                "Ollama response contains no choices",
                provider="ollama",
                details={"raw_keys": list(data.keys())},
            )

        choice = data["choices"][0]
        message = choice.get("message", {})

        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            provider="ollama",
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            metadata={
                "elapsed_ms": elapsed_ms,
                "request_id": data.get("id", ""),
                **request.metadata,
            },
        )

    async def health_check(self) -> LLMHealthStatus:
        """Check if the Ollama server is reachable.

        Uses the /api/tags endpoint to verify Ollama is running.

        Returns:
            LLMHealthStatus with health information.
        """
        start_time = time.monotonic()

        try:
            client = await self._get_client()

            response = await client.get(
                "/api/tags",
                timeout=httpx.Timeout(5.0),
            )
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return LLMHealthStatus(
                provider="ollama",
                healthy=True,
                model=self._config.model,
                latency_ms=elapsed_ms,
                details={
                    "endpoint": self._config.base_url,
                    "available_models": model_names,
                    "model_configured": self._config.model in model_names,
                },
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return LLMHealthStatus(
                provider="ollama",
                healthy=False,
                model=self._config.model,
                latency_ms=elapsed_ms,
                error=str(e),
                details={"endpoint": self._config.base_url},
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
