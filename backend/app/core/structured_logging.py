"""
Structured Logging for Razorpay CloseLoop.

Provides consistent, correlation-aware structured logging for the complete
financial workflow lifecycle.

Architecture:
  Every log entry is a structured JSON-like record with:
  - timestamp (UTC ISO 8601)
  - level (INFO/WARNING/ERROR/DEBUG)
  - event (lifecycle event name)
  - Correlation IDs (batch_id, exception_id, workflow_id, request_id)
  - Stage and status information
  - Timing data where available
  - Safe metadata (no secrets, no credentials)

Security:
  - Never log API keys, passwords, tokens, or credentials
  - Never log full financial records unnecessarily
  - Mask sensitive fields before storage
  - Sanitize error messages (no stack traces in production)
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================================
# Correlation ID Context
# ============================================================================

_batch_id_var: ContextVar[str] = ContextVar("batch_id", default="")
_exception_id_var: ContextVar[str] = ContextVar("exception_id", default="")
_workflow_id_var: ContextVar[str] = ContextVar("workflow_id", default="")
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def set_correlation_ids(
    batch_id: str = "",
    exception_id: str = "",
    workflow_id: str = "",
    request_id: str = "",
) -> None:
    """Set correlation IDs for the current context."""
    if batch_id:
        _batch_id_var.set(batch_id)
    if exception_id:
        _exception_id_var.set(exception_id)
    if workflow_id:
        _workflow_id_var.set(workflow_id)
    if request_id:
        _request_id_var.set(request_id)


def get_correlation_ids() -> Dict[str, str]:
    """Get current correlation IDs."""
    return {
        "batch_id": _batch_id_var.get(""),
        "exception_id": _exception_id_var.get(""),
        "workflow_id": _workflow_id_var.get(""),
        "request_id": _request_id_var.get(""),
    }


def generate_run_id() -> str:
    """Generate a unique run/correlation ID."""
    return f"run-{uuid.uuid4().hex[:12].upper()}"


# ============================================================================
# Event Types
# ============================================================================

class WorkflowEvent(str, Enum):
    """Lifecycle events for the financial workflow."""

    # Batch lifecycle
    BATCH_STARTED = "BATCH_STARTED"
    BATCH_COMPLETED = "BATCH_COMPLETED"
    BATCH_FAILED = "BATCH_FAILED"

    # Record processing
    RECORDS_RECEIVED = "RECORDS_RECEIVED"

    # Reconciliation
    RECONCILIATION_STARTED = "RECONCILIATION_STARTED"
    RECONCILIATION_COMPLETED = "RECONCILIATION_COMPLETED"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"

    # Exception management
    EXCEPTIONS_CREATED = "EXCEPTIONS_CREATED"
    EXCEPTION_LOADED = "EXCEPTION_LOADED"
    EXCEPTION_NOT_FOUND = "EXCEPTION_NOT_FOUND"

    # Classification
    CLASSIFICATION_STARTED = "CLASSIFICATION_STARTED"
    CLASSIFICATION_COMPLETED = "CLASSIFICATION_COMPLETED"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"

    # Evidence
    EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
    EVIDENCE_FAILED = "EVIDENCE_FAILED"

    # Similarity
    SIMILARITY_COMPLETED = "SIMILARITY_COMPLETED"
    SIMILARITY_FAILED = "SIMILARITY_FAILED"

    # Candidates
    CANDIDATES_GENERATED = "CANDIDATES_GENERATED"
    CANDIDATES_FAILED = "CANDIDATES_FAILED"

    # Guardrails
    GUARDRAILS_CHECKED = "GUARDRAILS_CHECKED"
    GUARDRAILS_FAILED = "GUARDRAILS_FAILED"

    # Agent workflow
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_DECISION = "AGENT_DECISION"
    AGENT_FAILED = "AGENT_FAILED"

    # Resolution
    RESOLUTION_STARTED = "RESOLUTION_STARTED"
    RESOLUTION_COMPLETED = "RESOLUTION_COMPLETED"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"
    RESOLUTION_PROPOSAL = "RESOLUTION_PROPOSAL"

    # Execution
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"

    # Verification
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

    # Outcomes
    AUTO_RESOLVED = "AUTO_RESOLVED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"

    # Human review actions
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

    # Feedback / Learning
    FEEDBACK_RECORDED = "FEEDBACK_RECORDED"
    REWARD_CALCULATED = "REWARD_CALCULATED"

    # System
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    HEALTH_CHECK = "HEALTH_CHECK"

    # LangGraph workflow lifecycle
    GRAPH_STARTED = "GRAPH_STARTED"
    GRAPH_COMPLETED = "GRAPH_COMPLETED"
    GRAPH_FAILED = "GRAPH_FAILED"
    NODE_STARTED = "NODE_STARTED"
    NODE_COMPLETED = "NODE_COMPLETED"
    NODE_FAILED = "NODE_FAILED"
    NODE_TIMING = "NODE_TIMING"
    ROUTING_DECISION = "ROUTING_DECISION"

    # MCP lifecycle
    MCP_SERVER_STARTED = "MCP_SERVER_STARTED"
    MCP_TOOL_CALLED = "MCP_TOOL_CALLED"
    MCP_TOOL_COMPLETED = "MCP_TOOL_COMPLETED"
    MCP_TOOL_FAILED = "MCP_TOOL_FAILED"

    # Errors
    SERVICE_ERROR = "SERVICE_ERROR"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


# ============================================================================
# Sensitive Data Masking
# ============================================================================

SENSITIVE_FIELD_NAMES = frozenset({
    "api_key", "apikey", "api-key", "secret", "password",
    "token", "auth_token", "access_token", "refresh_token",
    "private_key", "encryption_key", "signing_key",
    "secret_key", "session_token", "database_url",
})

SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"^sk_live_"),
    re.compile(r"^sk_test_"),
    re.compile(r"^AKIA"),
    re.compile(r"^ghp_"),
    re.compile(r"^gho_"),
    re.compile(r"^postgres(ql)?://"),
    re.compile(r"^mysql://"),
    re.compile(r"^mongodb://"),
]

_STRIP_ERROR_PATTERNS = [
    (re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE), "api_key=***"),
    (re.compile(r"token[=:]\s*\S+", re.IGNORECASE), "token=***"),
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"sk_(?:live|test)_[A-Za-z0-9]+"), "sk_***"),
    (re.compile(r"File \"[^\"]+\""), "File \"***\""),
    (re.compile(r"line \d+"), "line ***"),
    (re.compile(r"password[=:]\s*\S+", re.IGNORECASE), "password=***"),
]


def mask_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a copy with sensitive values masked. Original never modified."""
    masked: Dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in SENSITIVE_FIELD_NAMES:
            masked[key] = "***MASKED***"
        elif isinstance(value, str) and any(p.match(value) for p in SENSITIVE_VALUE_PATTERNS):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive(value)
        else:
            masked[key] = value
    return masked


def sanitize_error(message: str) -> str:
    """Sanitize an error message to remove sensitive information."""
    result = message
    for pattern, replacement in _STRIP_ERROR_PATTERNS:
        result = pattern.sub(replacement, result)
    if len(result) > 500:
        result = result[:497] + "..."
    return result


# ============================================================================
# Structured Logger
# ============================================================================

class StructuredLogger:
    """Production structured logger for the CloseLoop financial workflow.

    Emits structured log entries with correlation IDs, timing, and safe metadata.
    All entries are emitted to Python's logging system with consistent formatting.
    """

    def __init__(self, name: str, component: str = ""):
        """Initialize the structured logger.

        Args:
            name: Logger name (typically module path).
            component: Component name for log grouping (e.g., "batch", "guardrails").
        """
        self._logger = logging.getLogger(name)
        self._component = component

    def _emit(
        self,
        level: int,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        status: str = "ok",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        **extra_fields: Any,
    ) -> None:
        """Emit a structured log entry."""
        # Build the structured fields
        fields: Dict[str, Any] = {
            "event": event,
            "component": self._component,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add correlation IDs
        corr = get_correlation_ids()
        for k, v in corr.items():
            if v:
                fields[k] = v

        # Add optional fields
        if duration_ms is not None:
            fields["duration_ms"] = round(duration_ms, 2)
        if error_type:
            fields["error_type"] = error_type
        if error_message:
            fields["error_message"] = sanitize_error(error_message)

        # Add extra fields (sanitized)
        for k, v in extra_fields.items():
            if v is not None:
                if isinstance(v, dict):
                    fields[k] = mask_sensitive(v)
                else:
                    fields[k] = v

        # Format as structured log line
        # Format: [COMPONENT] EVENT status=ok duration_ms=123.45 | key=value key=value ...
        field_str = " ".join(f"{k}={v}" for k, v in fields.items() if k not in ("timestamp",))
        log_line = f"[{self._component.upper()}] {event} {field_str}"

        self._logger.log(level, log_line)

    # ── Convenience methods ──

    def info(self, event: str, message: str = "", **kwargs: Any) -> None:
        """Log an INFO-level workflow event."""
        kwargs.pop("status", None)
        self._emit(logging.INFO, event, message, **kwargs)

    def warning(self, event: str, message: str = "", **kwargs: Any) -> None:
        """Log a WARNING-level workflow event."""
        kwargs.pop("status", None)
        self._emit(logging.WARNING, event, message, status="warning", **kwargs)

    def error(self, event: str, message: str = "", **kwargs: Any) -> None:
        """Log an ERROR-level workflow event."""
        kwargs.pop("status", None)
        self._emit(logging.ERROR, event, message, status="error", **kwargs)

    def debug(self, event: str, message: str = "", **kwargs: Any) -> None:
        """Log a DEBUG-level workflow event."""
        kwargs.pop("status", None)
        self._emit(logging.DEBUG, event, message, **kwargs)

    def success(self, event: str, message: str = "", duration_ms: Optional[float] = None, **kwargs: Any) -> None:
        """Log a successful workflow event (INFO level)."""
        # Pop params that are passed explicitly to _emit to avoid duplicate keyword arguments
        for key in ("status", "duration_ms"):
            kwargs.pop(key, None)
        self._emit(logging.INFO, event, message, duration_ms=duration_ms, status="ok", **kwargs)

    def failure(self, event: str, message: str = "", duration_ms: Optional[float] = None,
                error_type: str = "", error_message: str = "", **kwargs: Any) -> None:
        """Log a failed workflow event (ERROR level)."""
        # Pop params that are passed explicitly to _emit to avoid duplicate keyword arguments
        for key in ("status", "duration_ms", "error_type", "error_message"):
            kwargs.pop(key, None)
        self._emit(logging.ERROR, event, message, duration_ms=duration_ms, status="failed",
                   error_type=error_type, error_message=error_message, **kwargs)

    # ── Timer utility ──

    def start_timer(self) -> float:
        """Start a timer. Returns the start time for computing duration."""
        return time.perf_counter()

    def elapsed_ms(self, start: float) -> float:
        """Compute elapsed milliseconds from a start time."""
        return round((time.perf_counter() - start) * 1000, 2)


# ============================================================================
# Component Loggers (pre-configured instances)
# ============================================================================

# Batch processing
batch_logger = StructuredLogger("closeloop.batch", component="batch")

# Exception management
exception_logger = StructuredLogger("closeloop.exception", component="exception")

# Intelligence pipeline (classification, similarity, candidates)
intelligence_logger = StructuredLogger("closeloop.intelligence", component="intelligence")

# Evidence retrieval
evidence_logger = StructuredLogger("closeloop.evidence", component="evidence")

# Guardrails
guardrail_logger = StructuredLogger("closeloop.guardrails", component="guardrails")

# Resolution and execution
resolution_logger = StructuredLogger("closeloop.resolution", component="resolution")

# Verification
verification_logger = StructuredLogger("closeloop.verification", component="verification")

# Human review
review_logger = StructuredLogger("closeloop.review", component="review")

# Feedback and learning
feedback_logger = StructuredLogger("closeloop.feedback", component="feedback")

# Agent workflow
agent_logger = StructuredLogger("closeloop.agent", component="agent")

# MCP
mcp_logger = StructuredLogger("closeloop.mcp", component="mcp")

# LLM
llm_logger = StructuredLogger("closeloop.llm", component="llm")

# System / startup
system_logger = StructuredLogger("closeloop.system", component="system")

# API
api_logger = StructuredLogger("closeloop.api", component="api")
