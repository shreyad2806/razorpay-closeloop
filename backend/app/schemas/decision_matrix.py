"""
Automation Decision Matrix schema for Razorpay CloseLoop Phase 6E.

Defines the final automation decision: AUTO, HUMAN_REVIEW, or UNRESOLVED.

The decision matrix evaluates all safety signals and determines
whether a resolution can be automatically applied.

Safety priority order:
1. CRITICAL BLOCK → UNRESOLVED
2. HARD SAFETY FAILURE → HUMAN_REVIEW
3. ALL AUTO CONDITIONS PASS → AUTO

No hidden overrides allowed.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────


class AutomationDecision(str, Enum):
    """Final automation decision."""

    AUTO = "AUTO"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    UNRESOLVED = "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# Reason Codes
# ─────────────────────────────────────────────────────────────────────────────


class ReasonCode(str, Enum):
    """Structured reason codes for the decision."""

    # AUTO reasons
    ALL_GATES_PASSED = "ALL_GATES_PASSED"

    # HUMAN_REVIEW reasons
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    MODERATE_EXPOSURE = "MODERATE_EXPOSURE"
    EVIDENCE_AMBIGUITY = "EVIDENCE_AMBIGUITY"
    CANDIDATES_CLOSE = "CANDIDATES_CLOSE"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    ELEVATED_RISK = "ELEVATED_RISK"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NOVEL_PATTERN = "NOVEL_PATTERN"
    LOW_COVERAGE = "LOW_COVERAGE"
    LOW_CONSISTENCY = "LOW_CONSISTENCY"
    ML_UNAVAILABLE = "ML_UNAVAILABLE"
    OPTIONAL_DEP_UNAVAILABLE = "OPTIONAL_DEP_UNAVAILABLE"
    FINANCIAL_INCONSISTENCY = "FINANCIAL_INCONSISTENCY"

    # UNRESOLVED reasons
    UNKNOWN_PATTERN = "UNKNOWN_PATTERN"
    HIGH_EXPOSURE = "HIGH_EXPOSURE"
    VERY_LOW_CONFIDENCE = "VERY_LOW_CONFIDENCE"
    MISSING_CRITICAL_EVIDENCE = "MISSING_CRITICAL_EVIDENCE"
    CRITICAL_DEP_FAILURE = "CRITICAL_DEP_FAILURE"
    UNSAFE_SYSTEM_STATE = "UNSAFE_SYSTEM_STATE"
    ENGINE_DEFERRED = "ENGINE_DEFERRED"
    BLOCKED_RESOLUTION_TYPE = "BLOCKED_RESOLUTION_TYPE"
    BLOCKED_EXCEPTION_TYPE = "BLOCKED_EXCEPTION_TYPE"


# ─────────────────────────────────────────────────────────────────────────────
# Gate Status
# ─────────────────────────────────────────────────────────────────────────────


class GateStatus(str, Enum):
    """Status of an individual gate check."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class GateResult(BaseModel):
    """Result of a single gate evaluation in the decision matrix."""

    gate_name: str = Field(..., description="Name of the gate")
    status: GateStatus = Field(..., description="PASSED, FAILED, or SKIPPED")
    value: Optional[str] = Field(
        default=None, description="Actual value evaluated"
    )
    threshold: Optional[str] = Field(
        default=None, description="Threshold compared against"
    )
    reason_code: Optional[ReasonCode] = Field(
        default=None, description="Reason code if gate failed"
    )
    description: str = Field(
        default="", description="Human-readable gate description"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision Configuration
# ─────────────────────────────────────────────────────────────────────────────


class DecisionConfig(BaseModel):
    """Configurable thresholds for the decision matrix."""

    # AUTO confidence thresholds
    min_confidence_for_auto: float = Field(
        default=0.75,
        description="Minimum confidence for AUTO decision",
        ge=0.0,
        le=1.0,
    )
    min_confidence_for_human: float = Field(
        default=0.40,
        description="Minimum confidence to consider (below → UNRESOLVED)",
        ge=0.0,
        le=1.0,
    )

    # Exposure thresholds
    max_exposure_for_auto: int = Field(
        default=25000,
        description="Maximum exposure in paise for AUTO",
        ge=0,
    )
    max_exposure_for_human: int = Field(
        default=100000,
        description="Maximum exposure in paise for HUMAN_REVIEW (above → UNRESOLVED)",
        ge=0,
    )

    # Evidence thresholds
    min_evidence_coverage_for_auto: float = Field(
        default=0.60,
        description="Minimum evidence coverage for AUTO",
        ge=0.0,
        le=1.0,
    )
    min_evidence_consistency_for_auto: float = Field(
        default=0.60,
        description="Minimum evidence consistency for AUTO",
        ge=0.0,
        le=1.0,
    )

    # Candidate margin
    min_margin_for_auto: float = Field(
        default=0.15,
        description="Minimum margin over second candidate for AUTO",
        ge=0.0,
        le=1.0,
    )

    # Risk
    allowed_risk_for_auto: List[str] = Field(
        default=["LOW"],
        description="Risk levels allowed for AUTO",
    )
    allowed_risk_for_human: List[str] = Field(
        default=["LOW", "MEDIUM"],
        description="Risk levels allowed for HUMAN_REVIEW",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision Result
# ─────────────────────────────────────────────────────────────────────────────


class AutomationDecisionResult(BaseModel):
    """Complete decision matrix output.

    Determines whether a resolution can be automatically applied,
    requires human review, or should remain unresolved.

    This is the FINAL safety decision.
    No downstream component may override it.
    """

    # Core decision
    decision: AutomationDecision = Field(
        ...,
        description="AUTO, HUMAN_REVIEW, or UNRESOLVED",
    )

    # Reasons
    reason_codes: List[ReasonCode] = Field(
        default_factory=list,
        description="All reason codes contributing to the decision",
    )
    primary_reason: str = Field(
        ...,
        description="Primary human-readable reason for the decision",
    )

    # Confidence
    confidence: float = Field(
        ...,
        description="Confidence level evaluated",
        ge=0.0,
        le=1.0,
    )

    # Risk
    risk_category: str = Field(
        ...,
        description="Risk category evaluated",
    )

    # Financial exposure
    financial_exposure_paise: int = Field(
        default=0,
        description="Financial exposure in paise",
    )

    # Evidence status
    evidence_coverage: float = Field(
        default=0.0,
        description="Evidence coverage evaluated",
    )
    evidence_consistency: float = Field(
        default=0.0,
        description="Evidence consistency evaluated",
    )

    # Novelty and conflict
    # HIGH #8 FIX: None = unknown. False = verified safe. True = unsafe.
    is_novel: Optional[bool] = Field(
        default=None,
        description="Whether the case is novel (None=unknown)",
    )
    has_conflict: Optional[bool] = Field(
        default=None,
        description="Whether evidence conflicts exist (None=unknown)",
    )

    # Verification
    verification_possible: bool = Field(
        default=True,
        description="Whether the financial adjustment can be verified",
    )

    # Gate tracking
    passed_gates: List[GateResult] = Field(
        default_factory=list,
        description="Gates that passed",
    )
    failed_gates: List[GateResult] = Field(
        default_factory=list,
        description="Gates that failed",
    )

    # Dependency health
    system_healthy: bool = Field(
        default=True,
        description="Whether all critical dependencies are healthy",
    )
    critical_failures: List[str] = Field(
        default_factory=list,
        description="Critical dependency failures",
    )

    # Metadata
    decision_version: str = Field(
        default="1.0.0",
        description="Version of the decision matrix",
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
        parts = [
            f"Decision: {self.decision.value}",
            f"Confidence: {self.confidence:.1%}",
            f"Risk: {self.risk_category}",
            f"Exposure: {self.financial_exposure_paise} paise",
            f"Coverage: {self.evidence_coverage:.1%}",
        ]
        if self.is_novel:
            parts.append("NOVEL")
        if self.has_conflict:
            parts.append("CONFLICT")
        if not self.system_healthy:
            parts.append(f"FAILURES: {len(self.critical_failures)}")
        parts.append(f"Reason: {self.primary_reason}")
        return " | ".join(parts)
