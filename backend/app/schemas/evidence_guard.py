"""
Evidence Safety Guard schema for Razorpay CloseLoop Phase 6C.

Defines configurable evidence safety thresholds and the guard result structure.

The evidence guard is a HARD SAFETY GATE.
It prevents automatic resolution when the financial explanation is
incomplete, conflicting, or novel.

It does NOT:
- execute financial actions
- modify financial records
- generate resolutions
- override reconciliation
- override confidence gate
- override exposure guard

Evidence safety is evaluated independently.
High ML confidence must NOT override conflicting or missing evidence.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Guard Outcome
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceAction(str, Enum):
    """Possible outcomes from the evidence safety guard."""

    PASS = "PASS"
    BLOCK = "BLOCK"


class EvidenceBlockReason(str, Enum):
    """Reasons the evidence guard may block a resolution."""

    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    LOW_COVERAGE = "LOW_COVERAGE"
    LOW_CONSISTENCY = "LOW_CONSISTENCY"
    NOVEL_PATTERN = "NOVEL_PATTERN"
    INSUFFICIENT_EXPLAINABILITY = "INSUFFICIENT_EXPLAINABILITY"
    UNEXPLAINED = "UNEXPLAINED"


# ─────────────────────────────────────────────────────────────────────────────
# Guard Configuration
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceGuardConfig(BaseModel):
    """Configurable evidence safety thresholds.

    The guard determines whether available evidence is sufficient
    for automated resolution.

    All thresholds are configurable. Defaults are designed for
    the synthetic dataset and should be calibrated for production.
    """

    # Minimum evidence coverage for auto-resolution
    min_evidence_coverage: float = Field(
        default=0.50,
        description=(
            "Minimum evidence coverage for auto-resolution. "
            "Below this → BLOCK. Default 0.50 requires at least "
            "half the discrepancy to be explained by evidence."
        ),
        ge=0.0,
        le=1.0,
    )

    # Minimum evidence consistency for auto-resolution
    min_evidence_consistency: float = Field(
        default=0.50,
        description=(
            "Minimum evidence consistency for auto-resolution. "
            "Below this → BLOCK."
        ),
        ge=0.0,
        le=1.0,
    )

    # Maximum allowed missing evidence count
    max_missing_evidence: int = Field(
        default=0,
        description=(
            "Maximum number of missing evidence records allowed. "
            "Above this → BLOCK. Default 0 means any missing evidence blocks."
        ),
        ge=0,
    )

    # Always block on conflicting evidence
    block_on_conflict: bool = Field(
        default=True,
        description=(
            "Always block auto-resolution when evidence conflicts. "
            "ML confidence must NOT override conflicting evidence."
        ),
    )

    # Always block on novel patterns
    block_on_novelty: bool = Field(
        default=True,
        description=(
            "Always block auto-resolution for novel patterns. "
            "Novel cases should not automatically inherit historical behavior."
        ),
    )

    # Minimum explanation status for auto-resolution
    allowed_explanation_statuses: List[str] = Field(
        default=["FULLY_EXPLAINED"],
        description=(
            "Explanation statuses that allow auto-resolution. "
            "PARTIALLY_EXPLAINED and UNEXPLAINED are blocked by default."
        ),
    )

    # Require explainability trace
    require_evidence_trace: bool = Field(
        default=True,
        description=(
            "Require that the candidate resolution traces to evidence. "
            "Candidates without evidence trace are blocked."
        ),
    )

    # Minimum supporting evidence count
    min_supporting_evidence: int = Field(
        default=1,
        description="Minimum number of supporting evidence records required",
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard Check
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceGuardCheck(BaseModel):
    """Result of a single evidence safety check."""

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
    block_reason: Optional[EvidenceBlockReason] = Field(
        default=None,
        description="Structured block reason if check failed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard Result
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceGuardResult(BaseModel):
    """Result of the evidence safety guard evaluation.

    Determines whether available evidence is sufficient for automated
    resolution based on evidence quality, completeness, and consistency.

    This is a HARD SAFETY GATE.
    High ML confidence must NOT override evidence safety failures.
    It does NOT execute financial actions.
    """

    # Core decision
    passed: bool = Field(
        ...,
        description="Whether the evidence passes the safety guard",
    )
    action: EvidenceAction = Field(
        ...,
        description="PASS if allowed, BLOCK if blocked",
    )

    # Evidence metrics
    evidence_coverage: float = Field(
        ...,
        description="Evidence coverage that was evaluated",
        ge=0.0,
        le=1.0,
    )
    evidence_consistency: float = Field(
        ...,
        description="Evidence consistency that was evaluated",
        ge=0.0,
        le=1.0,
    )

    # Conflict and missing
    has_conflict: bool = Field(
        default=False,
        description="Whether evidence conflicts were detected",
    )
    missing_evidence_count: int = Field(
        default=0,
        description="Number of missing evidence records",
    )

    # Novelty
    is_novel: bool = Field(
        default=False,
        description="Whether this is a novel pattern",
    )

    # Explanation
    explanation_status: Optional[str] = Field(
        default=None,
        description="Explanation status that was evaluated",
    )

    # Reason
    reason: str = Field(
        ...,
        description="Primary reason for the guard decision",
    )

    # Detailed checks
    checks: List[EvidenceGuardCheck] = Field(
        default_factory=list,
        description="Individual evidence safety checks performed",
    )

    # Block tracking
    block_reasons: List[EvidenceBlockReason] = Field(
        default_factory=list,
        description="All reasons for blocking (may be multiple)",
    )

    # Metadata
    guard_version: str = Field(
        default="1.0.0",
        description="Version of the evidence guard",
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
            f"Evidence Guard: {status}",
            f"Coverage: {self.evidence_coverage:.1%}",
            f"Consistency: {self.evidence_consistency:.1%}",
        ]
        if self.has_conflict:
            parts.append("CONFLICT")
        if self.missing_evidence_count > 0:
            parts.append(f"Missing: {self.missing_evidence_count}")
        if self.is_novel:
            parts.append("NOVEL")
        if self.block_reasons:
            parts.append(f"Blocks: {[r.value for r in self.block_reasons]}")
        parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)
