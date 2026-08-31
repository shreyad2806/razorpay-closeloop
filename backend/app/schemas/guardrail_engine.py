"""
Unified Guardrail Engine schema for Razorpay CloseLoop Phase 6F.

Defines the combined output of all safety guards:
- Confidence Gate (6A)
- Financial Exposure Guard (6B)
- Evidence Safety Guard (6C)
- System Failure Fallbacks (6D)
- Decision Matrix (6E)

This is the FINAL safety decision layer.
It must NOT execute financial actions.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.confidence_gate import ConfidenceGateResult
from app.schemas.decision_matrix import (
    AutomationDecision,
    AutomationDecisionResult,
    GateResult,
    ReasonCode,
)
from app.schemas.evidence_guard import EvidenceGuardResult
from app.schemas.exposure_guard import ExposureGuardResult
from app.schemas.failure_fallback import FailureFallbackResult


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail Engine Result
# ─────────────────────────────────────────────────────────────────────────────


class GuardrailEngineResult(BaseModel):
    """
    Complete guardrail engine output for a single exception.

    Integrates all Phase 6 safety guards into one unified, auditable result.

    This is the FINAL safety decision layer.
    It must NOT execute financial actions.
    It must NOT alter financial amounts.
    It must NOT invent evidence.
    """

    # Identification
    exception_id: str = Field(..., description="Exception analyzed")
    case_id: str = Field(..., description="Case analyzed")
    payment_id: Optional[str] = Field(default=None, description="Payment analyzed")
    merchant_id: Optional[str] = Field(default=None, description="Merchant analyzed")

    # Candidate
    candidate_id: Optional[str] = Field(
        default=None, description="Selected candidate ID if any"
    )
    selected_resolution: Optional[str] = Field(
        default=None, description="Selected resolution type if any"
    )

    # Final decision
    decision: AutomationDecision = Field(
        ..., description="AUTO, HUMAN_REVIEW, or UNRESOLVED"
    )

    # Confidence and risk
    confidence: float = Field(
        ..., description="Confidence level evaluated", ge=0.0, le=1.0
    )
    risk_category: str = Field(..., description="Risk category")

    # Financial exposure
    financial_exposure_paise: int = Field(
        default=0, description="Financial exposure in paise"
    )

    # Evidence status
    evidence_coverage: float = Field(
        default=0.0, description="Evidence coverage"
    )
    evidence_consistency: float = Field(
        default=0.0, description="Evidence consistency"
    )

    # Novelty and conflict
    is_novel: bool = Field(default=False, description="Whether the case is novel")
    has_conflict: bool = Field(
        default=False, description="Whether evidence conflicts exist"
    )

    # Verification
    verification_possible: bool = Field(
        default=True, description="Whether verification is possible"
    )

    # Gate results
    passed_gates: List[GateResult] = Field(
        default_factory=list, description="Gates that passed"
    )
    failed_gates: List[GateResult] = Field(
        default_factory=list, description="Gates that failed"
    )

    # Reason codes
    reason_codes: List[ReasonCode] = Field(
        default_factory=list, description="All reason codes"
    )
    primary_reason: str = Field(
        default="", description="Primary human-readable reason"
    )

    # Individual guard results (for full audit trail)
    confidence_gate_result: Optional[ConfidenceGateResult] = Field(
        default=None, description="Phase 6A confidence gate result"
    )
    exposure_guard_result: Optional[ExposureGuardResult] = Field(
        default=None, description="Phase 6B exposure guard result"
    )
    evidence_guard_result: Optional[EvidenceGuardResult] = Field(
        default=None, description="Phase 6C evidence guard result"
    )
    fallback_result: Optional[FailureFallbackResult] = Field(
        default=None, description="Phase 6D fallback result"
    )
    decision_result: Optional[AutomationDecisionResult] = Field(
        default=None, description="Phase 6E decision matrix result"
    )

    # System health
    system_healthy: bool = Field(
        default=True, description="Whether all critical dependencies are healthy"
    )
    critical_failures: List[str] = Field(
        default_factory=list, description="Critical dependency failures"
    )

    # Safety
    is_recommendation_only: bool = Field(
        default=True,
        description="Always True — this must NOT execute financial actions",
    )

    # Metadata
    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the guardrail was evaluated",
    )
    guardrail_version: str = Field(
        default="1.0.0", description="Version of the guardrail engine"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Processing time in milliseconds"
    )

    def is_auto(self) -> bool:
        return self.decision == AutomationDecision.AUTO

    def is_human_review(self) -> bool:
        return self.decision == AutomationDecision.HUMAN_REVIEW

    def is_unresolved(self) -> bool:
        return self.decision == AutomationDecision.UNRESOLVED

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [
            f"Guardrail: {self.decision.value}",
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
