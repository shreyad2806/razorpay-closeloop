"""
Resolution Engine result schema for Razorpay CloseLoop Phase 5E.

Defines the unified output of the resolution engine — combining
reconciliation, evidence, intelligence, candidate generation, scoring,
and selection into one result.

This is a RECOMMENDATION ONLY.
It must NOT execute financial actions.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.resolution_candidate import ResolutionProposal
from app.schemas.resolution_selection import (
    ExplainabilityDetail,
    SelectionStatus,
)


class ResolutionEngineResult(BaseModel):
    """
    Complete resolution engine output for a single exception.

    Combines all pipeline stages into one unified, auditable result.
    This is a RECOMMENDATION ONLY — it does NOT modify financial records.

    Pipeline:
    1. Deterministic reconciliation
    2. Evidence retrieval + graph
    3. Explanation
    4. Intelligence aggregation
    5. Candidate generation
    6. Candidate evidence
    7. Candidate scoring
    8. Candidate selection
    """

    # Identification
    exception_id: str = Field(..., description="Exception analyzed")
    case_id: str = Field(..., description="Case analyzed")
    payment_id: Optional[str] = Field(default=None, description="Payment analyzed")
    merchant_id: Optional[str] = Field(default=None, description="Merchant analyzed")

    # Financial context
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")

    # Selection result
    status: SelectionStatus = Field(
        ..., description="RECOMMENDED, HUMAN_REVIEW, or UNRESOLVED"
    )

    # Selected candidate (None if UNRESOLVED)
    selected_resolution: Optional[str] = Field(
        default=None, description="Selected resolution type"
    )
    selected_candidate: Optional[ResolutionProposal] = Field(
        default=None, description="Full selected candidate"
    )
    selected_score: Optional[CandidateScore] = Field(
        default=None, description="Score breakdown of selected candidate"
    )

    # Alternatives (always preserved)
    ranked_candidates: List[ResolutionProposal] = Field(
        default_factory=list, description="All candidates in rank order"
    )
    candidate_scores: List[CandidateScore] = Field(
        default_factory=list, description="Scores for all candidates"
    )

    # Confidence
    confidence: float = Field(
        ..., description="Overall recommendation confidence 0.0-1.0"
    )
    confidence_factors: Dict[str, float] = Field(
        default_factory=dict, description="Individual confidence factors"
    )

    # Risk
    risk_category: str = Field(
        ..., description="LOW, MEDIUM, HIGH"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Risk factor descriptions"
    )

    # Explainability
    explainability: Optional[ExplainabilityDetail] = Field(
        default=None, description="Explainability assessment"
    )

    # Rejection reasons (for UNRESOLVED/HUMAN_REVIEW)
    rejection_reasons: List[str] = Field(
        default_factory=list, description="Why no recommendation was made"
    )

    # Classification
    deterministic_exception_type: str = Field(
        ..., description="Exception type from deterministic reconciliation"
    )
    ml_exception_type: Optional[str] = Field(
        default=None, description="Exception type from ML classifier"
    )
    classification_agreement: bool = Field(
        default=True, description="Whether deterministic and ML agree"
    )

    # Evidence summary
    evidence_explanation_status: str = Field(
        default="", description="Evidence explanation status"
    )
    evidence_coverage: float = Field(
        default=0.0, description="Overall evidence coverage"
    )
    evidence_consistency: float = Field(
        default=0.0, description="Overall evidence consistency"
    )

    # Pipeline metadata
    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the engine processed this exception",
    )
    pipeline_version: str = Field(
        default="1.0.0", description="Version of the resolution engine"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Processing time in milliseconds"
    )

    # Safety
    is_recommendation_only: bool = Field(
        default=True,
        description="Always True — this result must NOT execute financial actions",
    )

    def is_recommended(self) -> bool:
        return self.status == SelectionStatus.RECOMMENDED

    def is_unresolved(self) -> bool:
        return self.status == SelectionStatus.UNRESOLVED

    def needs_human_review(self) -> bool:
        return self.status == SelectionStatus.HUMAN_REVIEW

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [
            f"Exception {self.exception_id}: {self.deterministic_exception_type}",
            f"Difference: {self.difference} paise",
            f"Status: {self.status.value}",
        ]
        if self.selected_resolution:
            parts.append(f"Selected: {self.selected_resolution}")
            if self.selected_candidate:
                adj = self.selected_candidate.financial_adjustment
                if adj.amount_paise > 0:
                    parts.append(f"Adjustment: {adj.amount_paise} paise {adj.direction}")
        parts.append(f"Confidence: {self.confidence:.1%}")
        parts.append(f"Risk: {self.risk_category}")
        parts.append(f"Candidates: {len(self.ranked_candidates)}")
        return " | ".join(parts)
