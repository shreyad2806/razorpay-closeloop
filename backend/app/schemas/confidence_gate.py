"""
Confidence Gate schema for Razorpay CloseLoop Phase 6A.

Defines configurable confidence thresholds and the gate result structure.

The confidence gate is a SAFETY GATE.
It decides whether a Phase 5 recommendation is safe enough for automation.

It does NOT:
- generate financial resolutions
- modify financial records
- execute financial actions
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Gate Action
# ─────────────────────────────────────────────────────────────────────────────


class GateAction(str, Enum):
    """Possible outcomes from the confidence gate."""

    CONTINUE = "CONTINUE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class RiskOverrideLevel(str, Enum):
    """Levels at which risk overrides the confidence gate."""

    NONE = "NONE"
    MEDIUM_BLOCK = "MEDIUM_BLOCK"
    HIGH_BLOCK = "HIGH_BLOCK"


# ─────────────────────────────────────────────────────────────────────────────
# Gate Configuration
# ─────────────────────────────────────────────────────────────────────────────


class ConfidenceGateConfig(BaseModel):
    """Configurable confidence gate thresholds.

    The gate determines whether a resolution candidate is safe enough
    to pass through for automated processing.

    All thresholds are configurable. Defaults are documented for the
    synthetic dataset. Production values should be calibrated against
    real validation data.
    """

    # Core confidence threshold
    min_confidence: float = Field(
        default=0.70,
        description=(
            "Minimum confidence for auto-resolution. Below this → HUMAN_REVIEW. "
            "Default 0.70 is conservative — designed for high precision."
        ),
        ge=0.0,
        le=1.0,
    )

    # Financial consistency threshold (from CandidateScore)
    min_financial_consistency: float = Field(
        default=0.60,
        description="Minimum financial consistency for auto-resolution",
        ge=0.0,
        le=1.0,
    )

    # Evidence coverage threshold
    min_evidence_coverage: float = Field(
        default=0.40,
        description="Minimum evidence coverage for auto-resolution",
        ge=0.0,
        le=1.0,
    )

    # Maximum risk that passes the gate
    allowed_risk_levels: List[str] = Field(
        default=["LOW", "MEDIUM"],
        description="Risk levels that are allowed through the gate. HIGH is blocked.",
    )

    # Conflict thresholds
    max_conflict_penalty: float = Field(
        default=0.10,
        description="Maximum conflict penalty allowed for auto-resolution",
        ge=0.0,
        le=1.0,
    )

    # Novelty thresholds
    max_novelty_penalty: float = Field(
        default=0.10,
        description="Maximum novelty penalty allowed for auto-resolution",
        ge=0.0,
        le=1.0,
    )

    # High-value financial adjustment override
    high_value_threshold_paise: int = Field(
        default=100000,
        description="Adjustment amount in paise above which HUMAN_REVIEW is forced",
        ge=0,
    )

    # Minimum evidence count
    min_supporting_evidence: int = Field(
        default=1,
        description="Minimum number of supporting evidence records",
        ge=0,
    )

    # Blocked resolution types — always require human review
    blocked_resolution_types: List[str] = Field(
        default=["UNKNOWN_UNRESOLVED", "MISSING_RECORD_ESCALATION"],
        description="Resolution types that always require human review regardless of confidence",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate Result
# ─────────────────────────────────────────────────────────────────────────────


class GateCheck(BaseModel):
    """Result of a single gate check."""

    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether this check passed")
    value: Optional[float] = Field(default=None, description="Actual value checked")
    threshold: Optional[float] = Field(default=None, description="Threshold compared against")
    reason: str = Field(default="", description="Explanation of the check result")


class ConfidenceGateResult(BaseModel):
    """Result of the confidence gate evaluation.

    Determines whether a Phase 5 recommendation is safe enough
    to pass through for automated processing.

    This is a SAFETY GATE.
    It does NOT execute financial actions.
    """

    # Core decision
    passed: bool = Field(
        ...,
        description="Whether the recommendation passes through the gate",
    )
    action: GateAction = Field(
        ...,
        description="CONTINUE if passed, HUMAN_REVIEW if blocked",
    )

    # Confidence
    confidence: float = Field(
        ...,
        description="The confidence value that was evaluated",
        ge=0.0,
        le=1.0,
    )
    threshold: float = Field(
        ...,
        description="The minimum confidence threshold applied",
        ge=0.0,
        le=1.0,
    )

    # Reason
    reason: str = Field(
        ...,
        description="Primary reason for the gate decision",
    )

    # Detailed checks
    checks: List[GateCheck] = Field(
        default_factory=list,
        description="Individual gate checks performed",
    )

    # Financial adjustment context
    adjustment_amount_paise: Optional[int] = Field(
        default=None,
        description="Financial adjustment amount if a candidate was selected",
    )
    blocked_by_high_value: bool = Field(
        default=False,
        description="Whether the gate was blocked due to high-value adjustment",
    )
    blocked_by_risk: bool = Field(
        default=False,
        description="Whether the gate was blocked due to risk level",
    )
    blocked_by_blocked_type: bool = Field(
        default=False,
        description="Whether the gate was blocked due to a blocked resolution type",
    )

    # Metadata
    gate_version: str = Field(
        default="1.0.0",
        description="Version of the confidence gate",
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
        status = "PASSED" if self.passed else "BLOCKED"
        parts = [
            f"Gate: {status}",
            f"Confidence: {self.confidence:.1%} (threshold: {self.threshold:.1%})",
            f"Reason: {self.reason}",
        ]
        if self.blocked_by_high_value:
            parts.append(f"High-value block: {self.adjustment_amount_paise} paise")
        if self.blocked_by_risk:
            parts.append("Risk level blocked")
        if self.blocked_by_blocked_type:
            parts.append("Resolution type blocked")
        return " | ".join(parts)
