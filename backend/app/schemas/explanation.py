"""
Explanation result data contract for Razorpay CloseLoop.

Defines the deterministic output of the explanation engine.
No AI, no ML, no LLM — pure deterministic arithmetic.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ExplanationStatus(str, Enum):
    """Possible explanation outcomes."""

    FULLY_EXPLAINED = "FULLY_EXPLAINED"
    PARTIALLY_EXPLAINED = "PARTIALLY_EXPLAINED"
    UNEXPLAINED = "UNEXPLAINED"
    CONFLICTING = "CONFLICTING"


class ExplainingEvent(BaseModel):
    """A single evidence event that contributes to explaining the discrepancy."""

    record_id: str = Field(..., description="ID of the financial record")
    entity_type: str = Field(..., description="REFUND, FEE, TAX, ADJUSTMENT, SETTLEMENT")
    amount: int = Field(..., description="Original amount in paise")
    contribution: int = Field(..., description="Contribution to expected amount in paise")


class CandidateExplanation(BaseModel):
    """A candidate combination of events that may explain the discrepancy."""

    events: List[ExplainingEvent] = Field(
        ..., description="Events in this candidate explanation"
    )
    total_contribution: int = Field(
        ..., description="Sum of contributions — should equal discrepancy if this explains it"
    )
    is_exact_match: bool = Field(
        ..., description="True if total_contribution exactly equals the discrepancy"
    )


class ExplanationResult(BaseModel):
    """
    Deterministic explanation result for a single exception.

    Contains the explanation status, supporting evidence, candidate
    explanations, and any conflicts or missing evidence detected.

    Ground truth labels are NOT included here.
    """

    # Exception identification
    exception_id: str = Field(..., description="The exception being explained")
    case_id: str = Field(..., description="Reference to Case")
    payment_id: str = Field(..., description="Reference to Payment")

    # Financial context
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")

    # Explanation outcome
    explanation_status: ExplanationStatus = Field(
        ..., description="FULLY_EXPLAINED, PARTIALLY_EXPLAINED, UNEXPLAINED, or CONFLICTING"
    )
    explained_amount: int = Field(
        default=0, description="Portion of discrepancy explained by evidence in paise"
    )
    remaining_difference: int = Field(
        ..., description="Unexplained portion of discrepancy in paise"
    )

    # Supporting evidence
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Record IDs of evidence that explains the discrepancy",
    )

    # Candidate explanations (when multiple explanations are possible)
    candidate_explanations: List[CandidateExplanation] = Field(
        default_factory=list,
        description="All candidate explanation combinations found",
    )

    # Conflict detection
    conflict: bool = Field(
        default=False,
        description="True if multiple materially different explanations exist",
    )

    # Missing evidence
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Entity types of missing evidence that could explain the discrepancy",
    )

    # Human-readable reason
    explanation_reason: str = Field(
        default="",
        description="Deterministic template-generated explanation text",
    )

    def is_fully_explained(self) -> bool:
        """Check if the discrepancy is fully explained."""
        return self.explanation_status == ExplanationStatus.FULLY_EXPLAINED

    def has_conflict(self) -> bool:
        """Check if there are conflicting explanations."""
        return self.conflict
