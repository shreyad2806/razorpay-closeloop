"""
LLM Retry Logic for Razorpay CloseLoop Phase 12B.

Provides controlled retry with exponential backoff for transient failures.

Safety rules:
- Only retry on transient errors (timeout, connection, 429, 500, 502, 503)
- Never retry on permanent errors (400, 401, 403, 404)
- Never retry indefinitely — bounded by max_retries
- Never retry financial execution because LLM failed
- LLM failure does not reduce safety requirements
"""

import asyncio
import random
from typing import Optional, Type

from app.llm.providers.base import (
    LLMConnectionError,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Retry Configuration
# ─────────────────────────────────────────────────────────────────────────────


class RetryConfig:
    """Configuration for LLM retry behavior."""

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry config.

        Args:
            max_retries: Maximum number of retry attempts (0 = no retries).
            base_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            backoff_factor: Multiplier for exponential backoff.
            jitter: Whether to add random jitter to delay.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given retry attempt.

        Uses exponential backoff with optional jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


# ─────────────────────────────────────────────────────────────────────────────
# Retryable Error Classification
# ─────────────────────────────────────────────────────────────────────────────


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is safe to retry.

    Retryable errors:
    - LLMTimeoutError (transient)
    - LLMConnectionError (transient)
    - LLMProviderError with status 429, 500, 502, 503 (transient)

    Non-retryable errors:
    - LLMConfigError (permanent)
    - LLMResponseError (permanent — bad output)
    - LLMProviderError with 400, 401, 403, 404 (permanent)
    """
    if isinstance(error, (LLMTimeoutError, LLMConnectionError)):
        return True

    if isinstance(error, LLMResponseError):
        return False

    if isinstance(error, LLMProviderError):
        status_code = error.details.get("status_code")
        if status_code is None:
            # Unknown provider error — don't retry
            return False
        # Retry on server errors and rate limits
        return status_code in (429, 500, 502, 503)

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Retry Executor
# ─────────────────────────────────────────────────────────────────────────────


class LLMRetryExecutor:
    """Executes LLM operations with controlled retry.

    Wraps a provider's generate() method with retry logic.
    Only retries on transient failures.
    Never retries indefinitely.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[RetryConfig] = None,
        on_retry=None,
    ):
        """Initialize the retry executor.

        Args:
            provider: The LLM provider to wrap.
            config: Retry configuration. Uses defaults if not provided.
            on_retry: Optional callback(attempt, max_retries, reason) for logging.
        """
        self._provider = provider
        self._config = config or RetryConfig()
        self._on_retry = on_retry

    @property
    def provider(self) -> LLMProvider:
        """The underlying provider."""
        return self._provider

    @property
    def config(self) -> RetryConfig:
        """The retry configuration."""
        return self._config

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute generate with retry.

        Args:
            request: LLM request.

        Returns:
            LLMResponse from the provider.

        Raises:
            LLMProviderError: If all retries exhausted or non-retryable error.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return await self._provider.generate(request)
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt >= self._config.max_retries:
                    break

                if not is_retryable_error(e):
                    break

                # Calculate delay
                delay = self._config.get_delay(attempt)

                # Notify callback
                if self._on_retry:
                    self._on_retry(
                        attempt + 1,
                        self._config.max_retries,
                        str(e),
                    )

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_error

    async def health_check(self):
        """Health check delegates directly to provider (no retry)."""
        return await self._provider.health_check()

    async def close(self):
        """Close the underlying provider."""
        await self._provider.close()
