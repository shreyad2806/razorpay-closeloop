"""
Financial Exposure Guard schema for Razorpay CloseLoop Phase 6B.

Defines configurable exposure thresholds and the guard result structure.

The exposure guard is a HARD SAFETY GATE.
It prevents high-value or financially risky resolutions from being
automatically recommended for execution.

It does NOT:
- execute financial actions
- modify financial records
- generate resolutions
- override reconciliation

Exposure is calculated from the ACTUAL proposed financial adjustment.
ML confidence does NOT determine exposure.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Guard Outcome
# ─────────────────────────────────────────────────────────────────────────────


class ExposureAction(str, Enum):
    """Possible outcomes from the exposure guard."""

    PASS = "PASS"
    BLOCK = "BLOCK"


class ExposureBlockReason(str, Enum):
    """Reasons the exposure guard may block a resolution."""

    ABOVE_MAX_AMOUNT = "ABOVE_MAX_AMOUNT"
    HIGH_VALUE_THRESHOLD = "HIGH_VALUE_THRESHOLD"
    HIGH_RISK_CATEGORY = "HIGH_RISK_CATEGORY"
    CONFLICTING_CASE = "CONFLICTING_CASE"
    NO_EXPOSURE_DATA = "NO_EXPOSURE_DATA"


# ─────────────────────────────────────────────────────────────────────────────
# Guard Configuration
# ─────────────────────────────────────────────────────────────────────────────


class ExposureGuardConfig(BaseModel):
    """Configurable financial exposure thresholds.

    The guard determines whether a proposed financial adjustment
    is too large or too risky for automated processing.

    All thresholds are configurable. Defaults are designed for
    the synthetic dataset and should be calibrated for production.
    """

    # Maximum amount that may qualify for auto-resolution
    max_auto_resolution_paise: int = Field(
        default=50000,
        description=(
            "Maximum financial adjustment in paise that may qualify "
            "for automatic resolution. Above this → BLOCK. "
            "Default ₹500 (50,000 paise) is conservative."
        ),
        ge=0,
    )

    # High-value threshold (informational + escalation)
    high_value_threshold_paise: int = Field(
        default=100000,
        description=(
            "Threshold in paise above which the case is flagged as "
            "high-value. High-value cases receive additional scrutiny."
        ),
        ge=0,
    )

    # Cumulative exposure limit per case
    cumulative_exposure_limit_paise: int = Field(
        default=200000,
        description=(
            "Maximum cumulative financial exposure across all candidates "
            "for a single exception. Protects against multi-adjustment cases "
            "with large aggregate exposure."
        ),
        ge=0,
    )

    # High-risk exception types that are always blocked
    blocked_exception_types: List[str] = Field(
        default=[
            "UNKNOWN",
            "COMPLEX_MULTI_ADJUSTMENT",
            "MISSING_RECORD",
        ],
        description=(
            "Exception types that are always blocked from auto-resolution "
            "regardless of adjustment amount. These represent inherently "
            "uncertain financial situations."
        ),
    )

    # High-risk resolution types that are always blocked
    blocked_resolution_types: List[str] = Field(
        default=[
            "UNKNOWN_UNRESOLVED",
            "MISSING_RECORD_ESCALATION",
            "MULTI_ADJUSTMENT",
        ],
        description=(
            "Resolution types that are always blocked from auto-resolution."
        ),
    )

    # Minimum evidence required for auto-resolution
    min_evidence_for_auto: int = Field(
        default=1,
        description="Minimum supporting evidence records required",
        ge=0,
    )

    # Maximum allowed conflict penalty
    max_conflict_for_auto: float = Field(
        default=0.15,
        description=(
            "Maximum conflict penalty allowed for auto-resolution. "
            "Cases with higher conflict are blocked."
        ),
        ge=0.0,
        le=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard Check
# ─────────────────────────────────────────────────────────────────────────────


class ExposureCheck(BaseModel):
    """Result of a single exposure check."""

    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether this check passed")
    value: Optional[float] = Field(
        default=None, description="Actual value checked"
    )
    threshold: Optional[float] = Field(
        default=None, description="Threshold compared against"
    )
    reason: str = Field(
        default="", description="Explanation of the check result"
    )
    block_reason: Optional[ExposureBlockReason] = Field(
        default=None,
        description="Structured block reason if check failed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard Result
# ─────────────────────────────────────────────────────────────────────────────


class ExposureGuardResult(BaseModel):
    """Result of the financial exposure guard evaluation.

    Determines whether a proposed financial adjustment is safe enough
    for automated processing based on financial exposure alone.

    This is a HARD SAFETY GATE.
    It cannot be overridden by confidence or ML scores.
    It does NOT execute financial actions.
    """

    # Core decision
    passed: bool = Field(
        ...,
        description="Whether the adjustment passes the exposure guard",
    )
    action: ExposureAction = Field(
        ...,
        description="PASS if allowed, BLOCK if blocked",
    )

    # Exposure calculation
    adjustment_amount_paise: int = Field(
        ...,
        description="Absolute proposed adjustment amount in paise",
    )
    max_auto_resolution_paise: int = Field(
        ...,
        description="The maximum auto-resolution threshold that was applied",
    )

    # Cumulative exposure
    cumulative_exposure_paise: int = Field(
        default=0,
        description="Cumulative exposure across all candidates for this exception",
    )

    # Reason
    reason: str = Field(
        ...,
        description="Primary reason for the guard decision",
    )

    # Detailed checks
    checks: List[ExposureCheck] = Field(
        default_factory=list,
        description="Individual exposure checks performed",
    )

    # Block tracking
    block_reasons: List[ExposureBlockReason] = Field(
        default_factory=list,
        description="All reasons for blocking (may be multiple)",
    )

    # High-value flag (informational, not blocking by itself)
    is_high_value: bool = Field(
        default=False,
        description="Whether the adjustment exceeds the high-value threshold",
    )

    # Metadata
    guard_version: str = Field(
        default="1.0.0",
        description="Version of the exposure guard",
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
        status = "PASS" if self.passed else "BLOCKED"
        parts = [
            f"Exposure Guard: {status}",
            f"Adjustment: {self.adjustment_amount_paise} paise",
            f"Max allowed: {self.max_auto_resolution_paise} paise",
        ]
        if self.is_high_value:
            parts.append("HIGH VALUE")
        if self.block_reasons:
            parts.append(f"Block reasons: {[r.value for r in self.block_reasons]}")
        parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)
