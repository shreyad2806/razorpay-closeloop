"""
OpenAI-Compatible LLM Provider for Razorpay CloseLoop Phase 12A.

Implements the LLMProvider interface for OpenAI-compatible APIs.
Supports any OpenAI-compatible endpoint (OpenAI, Azure, etc.)

Configuration comes from environment variables via OpenAIConfig.
No secrets are hardcoded.

IMPORTANT:
This provider generates text only.
It does NOT authorize financial actions.
"""

import time
from typing import Any, Dict, Optional

import httpx

from app.llm.config import OpenAIConfig
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


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider using httpx.

    Supports:
    - OpenAI API
    - Azure OpenAI
    - Any OpenAI-compatible endpoint

    Does NOT:
    - Authorize financial actions
    - Bypass guardrails
    - Access the database
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI provider.

        Args:
            config: OpenAI configuration (from env vars).

        Raises:
            LLMConfigError: If configuration is invalid.
        """
        if not config.api_base_url:
            raise LLMConfigError("OpenAI API base URL is required", provider="openai")
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def config(self) -> OpenAIConfig:
        return self._config

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._config.api_base_url,
                headers=headers,
                timeout=httpx.Timeout(self._config.timeout),
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion using the OpenAI-compatible API.

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

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.monotonic()

        try:
            client = await self._get_client()
            timeout_val = request.timeout or self._config.timeout

            response = await client.post(
                "/chat/completions",
                json=payload,
                timeout=httpx.Timeout(timeout_val),
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"OpenAI request timed out after {timeout_val}s",
                provider="openai",
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to OpenAI API at {self._config.api_base_url}",
                provider="openai",
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                f"OpenAI API returned error {e.response.status_code}: {e.response.text}",
                provider="openai",
                details={"status_code": e.response.status_code},
            ) from e
        except httpx.RequestError as e:
            raise LLMConnectionError(
                f"OpenAI request failed: {str(e)}",
                provider="openai",
            ) from e

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Parse response
        try:
            data = response.json()
        except Exception as e:
            raise LLMResponseError(
                "Failed to parse OpenAI response as JSON",
                provider="openai",
            ) from e

        if "choices" not in data or len(data["choices"]) == 0:
            raise LLMResponseError(
                "OpenAI response contains no choices",
                provider="openai",
                details={"raw_keys": list(data.keys())},
            )

        choice = data["choices"][0]
        message = choice.get("message", {})

        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            provider="openai",
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            metadata={
                "elapsed_ms": elapsed_ms,
                "request_id": data.get("id", ""),
                **request.metadata,
            },
        )

    async def health_check(self) -> LLMHealthStatus:
        """Check if the OpenAI API is reachable.

        Sends a minimal request to verify connectivity.

        Returns:
            LLMHealthStatus with health information.
        """
        start_time = time.monotonic()

        try:
            client = await self._get_client()

            # Use a models endpoint if available, otherwise try a minimal completion
            response = await client.get(
                "/models",
                timeout=httpx.Timeout(10.0),
            )
            response.raise_for_status()

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return LLMHealthStatus(
                provider="openai",
                healthy=True,
                model=self._config.model,
                latency_ms=elapsed_ms,
                details={"endpoint": self._config.api_base_url},
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return LLMHealthStatus(
                provider="openai",
                healthy=False,
                model=self._config.model,
                latency_ms=elapsed_ms,
                error=str(e),
                details={"endpoint": self._config.api_base_url},
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
