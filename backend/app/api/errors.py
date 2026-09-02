"""
Centralized API Error Handling for Razorpay CloseLoop Phase 13.8.

Provides:
- Consistent JSON error responses with request_id for correlation
- Structured error categories with proper HTTP status codes
- Safe logging (no secrets, no stack traces, no credentials)
- Error message sanitization
- Request ID middleware for correlation
- Guardrail rejection and business rule exceptions

Error categories:
- Validation error (422)
- Not found (404)
- Conflict (409)
- Invalid state transition (409)
- Business rule rejection (422)
- Guardrail rejection (403)
- Dependency unavailable (503)
- Database failure (503)
- Internal error (500)

Security rules:
- Never log stack traces, database credentials, API keys, or internal paths
- Never expose sensitive information in error responses
- Mask sensitive fields before storage
- Sanitize error messages
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request ID Context
# ─────────────────────────────────────────────────────────────────────────────

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get the current request ID."""
    return _request_id_var.get("")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# API Logger
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("api.errors")

# Sensitive field names to mask
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

# Patterns to strip from error messages
_STRIP_PATTERNS = [
    (re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE), "api_key=***"),
    (re.compile(r"token[=:]\s*\S+", re.IGNORECASE), "token=***"),
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"sk_(?:live|test)_[A-Za-z0-9]+"), "sk_***"),
    (re.compile(r"File \"[^\"]+\""), "File \"***\""),
    (re.compile(r"line \d+"), "line ***"),
    (re.compile(r"(?:password|secret)[=:]\s*\S+", re.IGNORECASE), "***MASKED***"),
]


def mask_sensitive_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a copy with sensitive values masked. Original never modified."""
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


def sanitize_error_message(message: str) -> str:
    """Sanitize an error message to remove sensitive information."""
    result = message
    for pattern, replacement in _STRIP_PATTERNS:
        result = pattern.sub(replacement, result)
    if len(result) > 500:
        result = result[:497] + "..."
    return result


def safe_log_error(
    error_code: str,
    status_code: int,
    request_id: str,
    message: str,
    path: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an API error safely. Never exposes secrets, stack traces, or credentials."""
    safe_details = mask_sensitive_dict(details) if details else {}
    safe_path = sanitize_error_message(path)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "error_code": error_code,
        "status_code": status_code,
        "message": sanitize_error_message(message),
        "path": safe_path,
        "details": safe_details,
    }

    if status_code >= 500:
        logger.error("[API_ERROR] %s", log_entry)
    elif status_code >= 400:
        logger.warning("[API_ERROR] %s", log_entry)
    else:
        logger.info("[API_ERROR] %s", log_entry)


# ─────────────────────────────────────────────────────────────────────────────
# Error Response Schema
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Machine-readable error categories."""

    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFLICT = "CONFLICT"
    INVALID_STATE = "INVALID_STATE"
    BUSINESS_RULE = "BUSINESS_RULE"
    GUARDRAIL_REJECTION = "GUARDRAIL_REJECTION"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Standard error response format.

    Every error response includes:
    - success: always false
    - error: human-readable message
    - error_code: machine-readable category
    - request_id: correlation ID for tracing
    - details: optional additional context
    """

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(default="", description="Machine-readable error code")
    request_id: str = Field(default="", description="Request correlation ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class NotFoundException(HTTPException):
    """Resource not found. Status: 404."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            status_code=404,
            detail=f"{resource} '{resource_id}' not found",
        )
        self.error_code = ErrorCategory.NOT_FOUND.value
        self.resource = resource
        self.resource_id = resource_id


class ValidationException(HTTPException):
    """Input validation failed. Status: 422."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=422,
            detail=message,
        )
        self.error_code = ErrorCategory.VALIDATION_ERROR.value
        self.details = details or {}


class ConflictException(HTTPException):
    """Resource conflict (e.g., duplicate operation, invalid state transition). Status: 409."""

    def __init__(self, message: str):
        super().__init__(
            status_code=409,
            detail=message,
        )
        self.error_code = ErrorCategory.CONFLICT.value


class InvalidStateException(HTTPException):
    """Invalid state transition. Status: 409."""

    def __init__(self, message: str, current_state: str = "", requested_action: str = ""):
        super().__init__(
            status_code=409,
            detail=message,
        )
        self.error_code = ErrorCategory.INVALID_STATE.value
        self.current_state = current_state
        self.requested_action = requested_action


class BusinessRuleException(HTTPException):
    """Business rule rejection. Status: 422."""

    def __init__(self, message: str, rule: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=422,
            detail=message,
        )
        self.error_code = ErrorCategory.BUSINESS_RULE.value
        self.rule = rule
        self.details = details or {}


class GuardrailRejectionException(HTTPException):
    """Guardrail rejected an operation. Status: 403.

    This is NOT a 422 — guardrail rejections are security-critical and
    should be clearly distinguished from validation errors.
    """

    def __init__(
        self,
        message: str,
        guardrail_reasons: Optional[List[str]] = None,
        risk_category: str = "",
        exposure_paise: int = 0,
    ):
        super().__init__(
            status_code=403,
            detail=message,
        )
        self.error_code = ErrorCategory.GUARDRAIL_REJECTION.value
        self.guardrail_reasons = guardrail_reasons or []
        self.risk_category = risk_category
        self.exposure_paise = exposure_paise


class ServiceUnavailableException(HTTPException):
    """Service temporarily unavailable. Status: 503."""

    def __init__(self, service: str, reason: str = ""):
        detail = f"Service '{service}' is unavailable"
        if reason:
            detail += f": {reason}"
        super().__init__(
            status_code=503,
            detail=detail,
        )
        self.error_code = ErrorCategory.DEPENDENCY_UNAVAILABLE.value
        self.service = service


class DatabaseFailureException(HTTPException):
    """Database operation failed. Status: 503."""

    def __init__(self, operation: str = ""):
        detail = "Database operation failed"
        if operation:
            detail += f": {operation}"
        super().__init__(
            status_code=503,
            detail=detail,
        )
        self.error_code = ErrorCategory.DATABASE_FAILURE.value


class InternalServerException(HTTPException):
    """Internal server error. Status: 500."""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            status_code=500,
            detail=message,
        )
        self.error_code = ErrorCategory.INTERNAL_ERROR.value


# ─────────────────────────────────────────────────────────────────────────────
# Request ID Middleware
# ─────────────────────────────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique request_id to every request.

    The request_id is:
    - Generated if not provided in X-Request-ID header
    - Stored in context variable for access in handlers
    - Included in response header X-Request-ID
    - Included in all error responses
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        _request_id_var.set(request_id)

        start_time = time.monotonic()

        try:
            response = await call_next(request)
        except Exception:
            # Let the exception handlers deal with it
            raise

        duration_ms = (time.monotonic() - start_time) * 1000.0

        # Add request ID to response header
        response.headers["X-Request-ID"] = request_id

        # Log successful requests (with duration, no sensitive data)
        if request.url.path not in ("/health", "/openapi.json", "/docs", "/redoc"):
            safe_path = sanitize_error_message(request.url.path)
            logger.debug(
                "[REQUEST] request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
                request_id,
                request.method,
                safe_path,
                response.status_code,
                duration_ms,
            )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# Exception Handlers
# ─────────────────────────────────────────────────────────────────────────────


def _get_request_id_from_exc(exc: Exception) -> str:
    """Extract request ID from context or generate one."""
    rid = get_request_id()
    if not rid:
        rid = generate_request_id()
    return rid


def _get_request_path(request: Request) -> str:
    """Get safe request path."""
    return sanitize_error_message(str(request.url.path))


async def not_found_handler(request: Request, exc: NotFoundException):
    """Handle NotFoundException. Status: 404."""
    request_id = _get_request_id_from_exc(exc)
    details = {"resource": exc.resource, "id": exc.resource_id}

    safe_log_error(
        ErrorCategory.NOT_FOUND.value, 404, request_id,
        str(exc.detail), _get_request_path(request), details,
    )

    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def validation_handler(request: Request, exc: ValidationException):
    """Handle ValidationException. Status: 422."""
    request_id = _get_request_id_from_exc(exc)

    safe_log_error(
        ErrorCategory.VALIDATION_ERROR.value, 422, request_id,
        str(exc.detail), _get_request_path(request), exc.details,
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details=exc.details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def conflict_handler(request: Request, exc: ConflictException):
    """Handle ConflictException. Status: 409."""
    request_id = _get_request_id_from_exc(exc)

    safe_log_error(
        ErrorCategory.CONFLICT.value, 409, request_id,
        str(exc.detail), _get_request_path(request),
    )

    return JSONResponse(
        status_code=409,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def invalid_state_handler(request: Request, exc: InvalidStateException):
    """Handle InvalidStateException. Status: 409."""
    request_id = _get_request_id_from_exc(exc)
    details = {}
    if exc.current_state:
        details["current_state"] = exc.current_state
    if exc.requested_action:
        details["requested_action"] = exc.requested_action

    safe_log_error(
        ErrorCategory.INVALID_STATE.value, 409, request_id,
        str(exc.detail), _get_request_path(request), details,
    )

    return JSONResponse(
        status_code=409,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def business_rule_handler(request: Request, exc: BusinessRuleException):
    """Handle BusinessRuleException. Status: 422."""
    request_id = _get_request_id_from_exc(exc)
    details = dict(exc.details)
    if exc.rule:
        details["rule"] = exc.rule

    safe_log_error(
        ErrorCategory.BUSINESS_RULE.value, 422, request_id,
        str(exc.detail), _get_request_path(request), details,
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def guardrail_rejection_handler(request: Request, exc: GuardrailRejectionException):
    """Handle GuardrailRejectionException. Status: 403."""
    request_id = _get_request_id_from_exc(exc)
    details: Dict[str, Any] = {}
    if exc.guardrail_reasons:
        details["guardrail_reasons"] = exc.guardrail_reasons
    if exc.risk_category:
        details["risk_category"] = exc.risk_category
    if exc.exposure_paise:
        details["exposure_paise"] = exc.exposure_paise

    safe_log_error(
        ErrorCategory.GUARDRAIL_REJECTION.value, 403, request_id,
        str(exc.detail), _get_request_path(request), details,
    )

    return JSONResponse(
        status_code=403,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def service_unavailable_handler(request: Request, exc: ServiceUnavailableException):
    """Handle ServiceUnavailableException. Status: 503."""
    request_id = _get_request_id_from_exc(exc)

    safe_log_error(
        ErrorCategory.DEPENDENCY_UNAVAILABLE.value, 503, request_id,
        str(exc.detail), _get_request_path(request),
        {"service": exc.service},
    )

    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
            details={"service": exc.service},
        ).model_dump(),
        headers={"X-Request-ID": request_id, "Retry-After": "30"},
    )


async def database_failure_handler(request: Request, exc: DatabaseFailureException):
    """Handle DatabaseFailureException. Status: 503."""
    request_id = _get_request_id_from_exc(exc)

    safe_log_error(
        ErrorCategory.DATABASE_FAILURE.value, 503, request_id,
        str(exc.detail), _get_request_path(request),
    )

    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id, "Retry-After": "10"},
    )


async def internal_error_handler(request: Request, exc: InternalServerException):
    """Handle InternalServerException. Status: 500."""
    request_id = _get_request_id_from_exc(exc)

    safe_log_error(
        ErrorCategory.INTERNAL_ERROR.value, 500, request_id,
        str(exc.detail), _get_request_path(request),
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc.detail),
            error_code=exc.error_code,
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def generic_error_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions. Status: 500.

    Never exposes stack traces, internal paths, or sensitive details.
    """
    request_id = _get_request_id_from_exc(exc)
    safe_error_type = type(exc).__name__

    safe_log_error(
        ErrorCategory.INTERNAL_ERROR.value, 500, request_id,
        f"Unexpected error: {safe_error_type}",
        _get_request_path(request),
        {"error_type": safe_error_type},
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code=ErrorCategory.INTERNAL_ERROR.value,
            request_id=request_id,
            details={"error_type": safe_error_type},
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request Validation Error Handler (Pydantic/FastAPI built-in 422)
# ─────────────────────────────────────────────────────────────────────────────


async def request_validation_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI's built-in RequestValidationError (422).

    Wraps Pydantic validation errors into our ErrorResponse format
    so clients always get a consistent error structure.
    """
    request_id = get_request_id() or generate_request_id()

    # Extract field-level errors
    errors_list = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        errors_list.append({
            "field": loc,
            "type": err.get("type", "unknown"),
            "message": err.get("msg", "Invalid input"),
        })

    details: Dict[str, Any] = {
        "errors": errors_list,
        "count": len(errors_list),
    }

    safe_log_error(
        ErrorCategory.VALIDATION_ERROR.value, 422, request_id,
        f"Validation failed: {len(errors_list)} error(s)",
        _get_request_path(request),
        {"error_count": len(errors_list)},
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=f"Validation failed: {len(errors_list)} error(s)",
            error_code=ErrorCategory.VALIDATION_ERROR.value,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request Validation Error Handler (Pydantic/FastAPI built-in 422)
# ─────────────────────────────────────────────────────────────────────────────


async def request_validation_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI's built-in RequestValidationError (422).

    Wraps Pydantic validation errors into our ErrorResponse format
    so clients always get a consistent error structure.
    """
    request_id = get_request_id() or generate_request_id()

    # Extract field-level errors
    errors_list = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        errors_list.append({
            "field": loc,
            "type": err.get("type", "unknown"),
            "message": err.get("msg", "Invalid input"),
        })

    details: Dict[str, Any] = {
        "errors": errors_list,
        "count": len(errors_list),
    }

    safe_log_error(
        ErrorCategory.VALIDATION_ERROR.value, 422, request_id,
        f"Validation failed: {len(errors_list)} error(s)",
        _get_request_path(request),
        {"error_count": len(errors_list)},
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=f"Validation failed: {len(errors_list)} error(s)",
            error_code=ErrorCategory.VALIDATION_ERROR.value,
            request_id=request_id,
            details=details,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation Helpers
# ─────────────────────────────────────────────────────────────────────────────


def validate_id(value: str, field_name: str = "ID") -> str:
    """Validate that an ID is non-empty and contains safe characters only.

    Raises ValidationException if invalid.
    """
    if not value or not value.strip():
        raise ValidationException(f"{field_name} must not be empty")

    # Allow alphanumeric, hyphens, underscores, dots
    if not re.match(r"^[a-zA-Z0-9._\-]+$", value):
        raise ValidationException(
            f"{field_name} contains invalid characters",
            details={"field": field_name, "value": value[:50]},
        )

    if len(value) > 200:
        raise ValidationException(
            f"{field_name} exceeds maximum length",
            details={"field": field_name, "max_length": 200},
        )

    return value.strip()


def validate_amount(amount: int, field_name: str = "amount") -> int:
    """Validate a financial amount in paise.

    Raises ValidationException if invalid.
    """
    if amount < 0:
        raise ValidationException(
            f"{field_name} must not be negative",
            details={"field": field_name, "value": amount},
        )

    if amount > 100_000_000:  # ₹10,00,000
        raise ValidationException(
            f"{field_name} exceeds maximum allowed amount",
            details={"field": field_name, "value": amount, "max": 100_000_000},
        )

    return amount


def validate_pagination(limit: int = 50, offset: int = 0) -> tuple:
    """Validate and normalize pagination parameters.

    Returns (limit, offset) tuple.
    """
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    return limit, offset
