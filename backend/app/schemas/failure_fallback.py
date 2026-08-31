"""
System Failure Fallback schema for Razorpay CloseLoop Phase 6D.

Defines structured error categories and fail-closed behavior for
dependent system failures.

The system must fail CLOSED rather than accidentally auto-resolve
when a critical dependency is unavailable.

It does NOT:
- execute financial actions
- modify financial records
- generate resolutions
- override reconciliation
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Error Categories
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Structured error categories for dependency failures."""

    ML_UNAVAILABLE = "ML_UNAVAILABLE"
    ML_LOADING_FAILURE = "ML_LOADING_FAILURE"
    ML_PREDICTION_FAILURE = "ML_PREDICTION_FAILURE"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    DATABASE_READ_FAILURE = "DATABASE_READ_FAILURE"
    DATABASE_WRITE_FAILURE = "DATABASE_WRITE_FAILURE"
    DATABASE_TIMEOUT = "DATABASE_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
    EVIDENCE_RETRIEVAL_FAILURE = "EVIDENCE_RETRIEVAL_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class FailureSeverity(str, Enum):
    """Severity of a dependency failure."""

    CRITICAL = "CRITICAL"
    DEGRADED = "DEGRADED"
    INFO = "INFO"


class FallbackAction(str, Enum):
    """Possible fallback actions when a dependency fails."""

    FAIL_CLOSED = "FAIL_CLOSED"
    CONTINUE_WITHOUT = "CONTINUE_WITHOUT"
    USE_DETERMINISTIC_ONLY = "USE_DETERMINISTIC_ONLY"


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Policy
# ─────────────────────────────────────────────────────────────────────────────


class DependencyPolicy(BaseModel):
    """Policy for how a dependency failure is handled.

    Defines whether the system can continue without a dependency
    or must fail closed.
    """

    dependency_name: str = Field(
        ..., description="Name of the dependency"
    )
    is_required: bool = Field(
        ...,
        description=(
            "If True, failure of this dependency always results in "
            "FAIL_CLOSED (HUMAN_REVIEW or UNRESOLVED). "
            "If False, the system can continue without it."
        ),
    )
    fallback_action: FallbackAction = Field(
        ...,
        description="What to do when this dependency fails",
    )
    fallback_status: str = Field(
        ...,
        description=(
            "What status to assign when this dependency fails. "
            "Typically HUMAN_REVIEW or UNRESOLVED."
        ),
    )
    error_category: ErrorCategory = Field(
        ..., description="Error category for this dependency"
    )
    description: str = Field(
        default="", description="Human-readable description"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Failure Event
# ─────────────────────────────────────────────────────────────────────────────


class DependencyFailure(BaseModel):
    """Record of a single dependency failure."""

    dependency_name: str = Field(..., description="Which dependency failed")
    error_category: ErrorCategory = Field(
        ..., description="Structured error category"
    )
    severity: FailureSeverity = Field(
        ..., description="Failure severity"
    )
    error_message: str = Field(
        default="", description="Structured error message (no stack traces)"
    )
    fallback_action: FallbackAction = Field(
        ..., description="Action taken"
    )
    fallback_status: str = Field(
        ..., description="Status assigned"
    )
    is_recoverable: bool = Field(
        default=False, description="Whether this failure is transient"
    )

    def summary(self) -> str:
        return (
            f"{self.dependency_name}: {self.error_category.value} "
            f"({self.severity.value}) → {self.fallback_action.value}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Result
# ─────────────────────────────────────────────────────────────────────────────


class FailureFallbackResult(BaseModel):
    """Result of the failure fallback evaluation.

    Determines whether the system can continue processing given
    the current state of dependencies.

    This is a FAIL-CLOSED system.
    Uncertainty always results in HUMAN_REVIEW.
    """

    # Core decision
    can_proceed: bool = Field(
        ...,
        description=(
            "Whether the system can proceed with resolution. "
            "False means FAIL_CLOSED — human review required."
        ),
    )
    action: FallbackAction = Field(
        ...,
        description="Action to take"
    )

    # Fail-closed status
    fallback_status: str = Field(
        ...,
        description="Status to assign (HUMAN_REVIEW, UNRESOLVED, or empty if proceeding)"
    )

    # Failures
    failures: List[DependencyFailure] = Field(
        default_factory=list,
        description="All dependency failures detected",
    )
    critical_failures: List[DependencyFailure] = Field(
        default_factory=list,
        description="Only critical failures",
    )

    # Category summary
    failed_categories: List[ErrorCategory] = Field(
        default_factory=list,
        description="All error categories present",
    )

    # Recovery
    has_critical_failure: bool = Field(
        default=False,
        description="Whether any critical failure exists",
    )
    can_use_deterministic_only: bool = Field(
        default=True,
        description=(
            "Whether the deterministic pipeline can continue "
            "without ML/optional dependencies"
        ),
    )

    # Reason
    reason: str = Field(
        ...,
        description="Primary reason for the fallback decision",
    )

    # Metadata
    fallback_version: str = Field(
        default="1.0.0",
        description="Version of the fallback system",
    )
    exception_id: Optional[str] = Field(
        default=None,
        description="Exception ID being evaluated",
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Case ID being evaluated",
    )

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PROCEED" if self.can_proceed else "FAIL_CLOSED"
        parts = [f"Fallback: {status}"]
        if self.fallback_status:
            parts.append(f"Status: {self.fallback_status}")
        if self.failures:
            parts.append(f"Failures: {len(self.failures)}")
        if self.has_critical_failure:
            parts.append("CRITICAL FAILURE")
        parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)
