"""
Resolution candidate selection schema for Razorpay CloseLoop Phase 5D.

Defines the output of candidate selection — the final recommendation.

This is a RECOMMENDATION ONLY.
It is NOT financial truth.
It does NOT authorize execution.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.resolution_candidate import ResolutionProposal


class SelectionStatus(str, Enum):
    """Status of the resolution selection."""

    RECOMMENDED = "RECOMMENDED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    UNRESOLVED = "UNRESOLVED"


class ExplainabilityLevel(str, Enum):
    """How explainable the recommendation is."""

    FULLY_EXPLAINABLE = "FULLY_EXPLAINABLE"
    PARTIALLY_EXPLAINABLE = "PARTIALLY_EXPLAINABLE"
    NOT_EXPLAINABLE = "NOT_EXPLAINABLE"


class SelectionConfig(BaseModel):
    """Configurable thresholds for resolution selection.

    These thresholds determine when the system can confidently recommend
    a resolution vs when it must defer to human review.
    """

    # Minimum scores
    min_final_score: float = Field(
        default=0.4, description="Minimum final score to consider recommending"
    )
    min_evidence_coverage: float = Field(
        default=0.3, description="Minimum evidence coverage for recommendation"
    )
    min_financial_consistency: float = Field(
        default=0.5, description="Minimum financial consistency"
    )

    # Margin
    min_margin_over_second: float = Field(
        default=0.1, description="Minimum score margin over second-best candidate"
    )

    # Conflict thresholds
    max_conflict_penalty: float = Field(
        default=0.15, description="Maximum acceptable conflict penalty"
    )

    # Novelty
    max_novelty_penalty: float = Field(
        default=0.1, description="Maximum acceptable novelty penalty"
    )

    # Risk thresholds
    high_risk_adjustment_paise: int = Field(
        default=50000, description="Adjustment amount triggering HIGH risk"
    )
    medium_risk_adjustment_paise: int = Field(
        default=10000, description="Adjustment amount triggering MEDIUM risk"
    )

    # Explainability
    min_sources_for_explainable: int = Field(
        default=1, description="Minimum sources for PARTIALLY_EXPLAINABLE"
    )
    min_sources_for_fully_explainable: int = Field(
        default=2, description="Minimum sources for FULLY_EXPLAINABLE"
    )


class ExplainabilityDetail(BaseModel):
    """Detailed explainability assessment."""

    level: ExplainabilityLevel = Field(
        ..., description="Explainability level"
    )
    has_evidence_trace: bool = Field(
        default=False, description="Whether resolution traces to evidence"
    )
    has_financial_trace: bool = Field(
        default=False, description="Whether financial adjustment is traceable"
    )
    has_historical_basis: bool = Field(
        default=False, description="Whether historical cases support this"
    )
    has_ml_basis: bool = Field(
        default=False, description="Whether ML supports this"
    )
    source_count: int = Field(default=0, description="Number of supporting sources")
    explanation: str = Field(default="", description="Human-readable explanation")


class SelectionResult(BaseModel):
    """
    Final resolution selection result.

    Contains the selected candidate (if any), alternatives, confidence,
    risk, explainability, and full audit trail.

    This is a RECOMMENDATION ONLY.
    It must NOT execute financial actions.
    """

    # Status
    status: SelectionStatus = Field(
        ..., description="RECOMMENDED, HUMAN_REVIEW, or UNRESOLVED"
    )
    exception_id: str = Field(..., description="Exception analyzed")
    case_id: str = Field(..., description="Case analyzed")

    # Selected candidate (None if UNRESOLVED)
    selected_candidate: Optional[ResolutionProposal] = Field(
        default=None, description="The recommended resolution candidate"
    )
    selected_score: Optional[CandidateScore] = Field(
        default=None, description="Score breakdown of selected candidate"
    )

    # Alternatives (always preserved)
    alternatives: List[ResolutionProposal] = Field(
        default_factory=list, description="Other candidates in rank order"
    )
    alternative_scores: List[CandidateScore] = Field(
        default_factory=list, description="Scores of alternative candidates"
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
    explainability: ExplainabilityDetail = Field(
        ..., description="Explainability assessment"
    )

    # Rejection reasons (for UNRESOLVED/HUMAN_REVIEW)
    rejection_reasons: List[str] = Field(
        default_factory=list, description="Why no recommendation was made"
    )

    # Metadata
    selection_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When selection was made",
    )
    pipeline_version: str = Field(default="1.0.0")
    is_recommendation_only: bool = Field(
        default=True, description="Always True"
    )

    def is_recommended(self) -> bool:
        return self.status == SelectionStatus.RECOMMENDED

    def needs_human_review(self) -> bool:
        return self.status == SelectionStatus.HUMAN_REVIEW

    def is_unresolved(self) -> bool:
        return self.status == SelectionStatus.UNRESOLVED

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"Status: {self.status.value}"]
        if self.selected_candidate:
            parts.append(f"Selected: {self.selected_candidate.resolution_type}")
            adj = self.selected_candidate.financial_adjustment
            if adj.amount_paise > 0:
                parts.append(f"Adjustment: {adj.amount_paise} paise {adj.direction}")
        parts.append(f"Confidence: {self.confidence:.1%}")
        parts.append(f"Risk: {self.risk_category}")
        parts.append(f"Explainability: {self.explainability.level.value}")
        if self.rejection_reasons:
            parts.append(f"Reasons: {'; '.join(self.rejection_reasons[:3])}")
        return " | ".join(parts)
