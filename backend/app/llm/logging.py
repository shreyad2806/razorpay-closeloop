"""
LLM Observability Logger for Razorpay CloseLoop Phase 12B.

Provides safe structured logging for LLM operations.

Security rules:
- Never log API keys, tokens, or credentials
- Never log full financial amounts or merchant data
- Log safe metadata: provider, model, latency, success/failure
- Mask sensitive fields before storage
"""

import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Sensitive field masking
# ─────────────────────────────────────────────────────────────────────────────

SENSITIVE_FIELD_NAMES = frozenset({
    "api_key", "apikey", "api-key", "secret", "password",
    "token", "auth_token", "access_token", "refresh_token",
    "private_key", "encryption_key", "signing_key",
    "secret_key", "session_token",
})

SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"^sk_live_"),     # Stripe live keys
    re.compile(r"^sk_test_"),     # Stripe test keys
    re.compile(r"^AKIA"),         # AWS access keys
    re.compile(r"^ghp_"),         # GitHub personal tokens
    re.compile(r"^gho_"),         # GitHub OAuth tokens
]


def mask_sensitive_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a copy with sensitive values masked.

    Original data is never modified.
    """
    masked: Dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in SENSITIVE_FIELD_NAMES:
            masked[key] = "***MASKED***"
        elif isinstance(value, str) and any(p.match(value) for p in SENSITIVE_VALUE_PATTERNS):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_dict(value)
        else:
            masked[key] = value
    return masked


# ─────────────────────────────────────────────────────────────────────────────
# Log Event Types
# ─────────────────────────────────────────────────────────────────────────────


class LLMEventType(str, Enum):
    """Types of LLM log events."""

    REQUEST_START = "request_start"
    REQUEST_SUCCESS = "request_success"
    REQUEST_ERROR = "request_error"
    REQUEST_TIMEOUT = "request_timeout"
    RETRY_ATTEMPT = "retry_attempt"
    HEALTH_CHECK = "health_check"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CONFIG_VALIDATION = "config_validation"


# ─────────────────────────────────────────────────────────────────────────────
# Log Entry
# ─────────────────────────────────────────────────────────────────────────────


class LLMLogEntry(BaseModel):
    """Structured log entry for LLM operations."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event timestamp (UTC ISO 8601)",
    )
    event_type: LLMEventType = Field(
        ..., description="Type of log event"
    )
    provider: str = Field(default="", description="Provider name")
    model: str = Field(default="", description="Model name")
    request_id: Optional[str] = Field(
        default=None, description="Request correlation ID"
    )
    workflow_id: Optional[str] = Field(
        default=None, description="Calling workflow ID"
    )
    exception_id: Optional[str] = Field(
        default=None, description="Exception being processed"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Operation duration in milliseconds"
    )
    success: Optional[bool] = Field(
        default=None, description="Whether operation succeeded"
    )
    error_type: Optional[str] = Field(
        default=None, description="Error type if failed"
    )
    error_message: Optional[str] = Field(
        default=None, description="Sanitized error message (no stack traces)"
    )
    retry_attempt: Optional[int] = Field(
        default=None, description="Current retry attempt number"
    )
    max_retries: Optional[int] = Field(
        default=None, description="Maximum retry attempts"
    )
    tokens_used: Optional[int] = Field(
        default=None, description="Total tokens used"
    )
    finish_reason: Optional[str] = Field(
        default=None, description="Why generation stopped"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional safe metadata (sensitive fields masked)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Logger
# ─────────────────────────────────────────────────────────────────────────────


class LLMLogger:
    """Structured logger for LLM operations.

    Records safe metadata for every LLM call.
    Never logs API keys, tokens, or sensitive financial data.
    """

    def __init__(self, name: str = "llm"):
        self._logger = logging.getLogger(name)
        self._entries: List[LLMLogEntry] = []

    def _log(self, entry: LLMLogEntry) -> None:
        """Record a log entry."""
        self._entries.append(entry)

        # Also emit to Python logging
        level = logging.INFO
        if entry.event_type in (
            LLMEventType.REQUEST_ERROR,
            LLMEventType.REQUEST_TIMEOUT,
            LLMEventType.PROVIDER_UNAVAILABLE,
        ):
            level = logging.WARNING
        elif entry.event_type == LLMEventType.REQUEST_SUCCESS:
            level = logging.DEBUG

        self._logger.log(
            level,
            "[%s] %s provider=%s model=%s duration_ms=%.1f success=%s%s",
            entry.event_type.value,
            entry.request_id or "-",
            entry.provider or "-",
            entry.model or "-",
            entry.duration_ms or 0.0,
            entry.success if entry.success is not None else "-",
            f" error={entry.error_message}" if entry.error_message else "",
        )

    def log_request_start(
        self,
        provider: str,
        model: str,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the start of an LLM request."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.REQUEST_START,
            provider=provider,
            model=model,
            request_id=request_id,
            workflow_id=workflow_id,
            exception_id=exception_id,
            metadata=mask_sensitive_dict(metadata or {}),
        ))

    def log_request_success(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        tokens_used: Optional[int] = None,
        finish_reason: Optional[str] = None,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a successful LLM request."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.REQUEST_SUCCESS,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            success=True,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            request_id=request_id,
            workflow_id=workflow_id,
            exception_id=exception_id,
            metadata=mask_sensitive_dict(metadata or {}),
        ))

    def log_request_error(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        error_type: str,
        error_message: str,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> None:
        """Log a failed LLM request."""
        # Sanitize error message — remove stack traces, credentials
        safe_message = _sanitize_error(error_message)
        self._log(LLMLogEntry(
            event_type=LLMEventType.REQUEST_ERROR,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            success=False,
            error_type=error_type,
            error_message=safe_message,
            request_id=request_id,
            workflow_id=workflow_id,
            exception_id=exception_id,
        ))

    def log_request_timeout(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        request_id: Optional[str] = None,
    ) -> None:
        """Log an LLM request timeout."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.REQUEST_TIMEOUT,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            success=False,
            error_type="timeout",
            error_message=f"Request timed out after {duration_ms:.0f}ms",
            request_id=request_id,
        ))

    def log_retry(
        self,
        provider: str,
        model: str,
        attempt: int,
        max_retries: int,
        reason: str,
        request_id: Optional[str] = None,
    ) -> None:
        """Log a retry attempt."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.RETRY_ATTEMPT,
            provider=provider,
            model=model,
            retry_attempt=attempt,
            max_retries=max_retries,
            error_message=_sanitize_error(reason),
            request_id=request_id,
        ))

    def log_health_check(
        self,
        provider: str,
        model: str,
        healthy: bool,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a health check result."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.HEALTH_CHECK,
            provider=provider,
            model=model,
            success=healthy,
            latency_ms=latency_ms,
            error_message=_sanitize_error(error) if error else None,
        ))

    def log_provider_unavailable(
        self,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        """Log that the provider is unavailable."""
        self._log(LLMLogEntry(
            event_type=LLMEventType.PROVIDER_UNAVAILABLE,
            provider=provider,
            model=model,
            success=False,
            error_message=_sanitize_error(reason),
        ))

    def get_entries(
        self,
        event_type: Optional[LLMEventType] = None,
        provider: Optional[str] = None,
        limit: int = 100,
    ) -> List[LLMLogEntry]:
        """Query log entries with optional filters."""
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if provider:
            entries = [e for e in entries if e.provider == provider]
        return entries[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged LLM operations."""
        total = len(self._entries)
        successes = sum(1 for e in self._entries if e.success is True)
        failures = sum(1 for e in self._entries if e.success is False)
        retries = sum(1 for e in self._entries if e.event_type == LLMEventType.RETRY_ATTEMPT)
        avg_duration = 0.0
        durations = [e.duration_ms for e in self._entries if e.duration_ms is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)

        providers = list(set(e.provider for e in self._entries if e.provider))

        return {
            "total_entries": total,
            "successes": successes,
            "failures": failures,
            "retries": retries,
            "avg_duration_ms": round(avg_duration, 2),
            "providers_seen": providers,
        }

    def clear(self) -> None:
        """Clear all logged entries."""
        self._entries.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Error Sanitization
# ─────────────────────────────────────────────────────────────────────────────

# Patterns to strip from error messages
_STRIP_PATTERNS = [
    (re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE), "api_key=***"),
    (re.compile(r"token[=:]\s*\S+", re.IGNORECASE), "token=***"),
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"sk_(?:live|test)_[A-Za-z0-9]+"), "sk_***"),
    # Remove file paths that might reveal infrastructure
    (re.compile(r"File \"[^\"]+\""), "File \"***\""),
    # Remove line numbers from tracebacks
    (re.compile(r"line \d+"), "line ***"),
]


def _sanitize_error(message: str) -> str:
    """Sanitize an error message to remove sensitive information.

    Removes:
    - API keys and tokens
    - Stack trace details
    - File paths
    - Line numbers
    """
    result = message
    for pattern, replacement in _STRIP_PATTERNS:
        result = pattern.sub(replacement, result)
    # Truncate to prevent log flooding
    if len(result) > 500:
        result = result[:497] + "..."
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Timer Utility
# ─────────────────────────────────────────────────────────────────────────────


class LLMTimer:
    """Context manager for timing LLM operations."""

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        self._end = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._start is None:
            return 0.0
        end = self._end or time.monotonic()
        return (end - self._start) * 1000.0
