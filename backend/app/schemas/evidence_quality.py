"""
Evidence quality scoring data contract for Razorpay CloseLoop.

Defines the deterministic output of evidence quality scoring.
These scores measure evidence quality, NOT ML prediction confidence
or resolution confidence.

No AI, no ML, no probabilistic reasoning.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NoveltyLevel(str, Enum):
    """Deterministic novelty classification."""

    KNOWN_PATTERN = "KNOWN_PATTERN"
    NOVEL_NO_HISTORICAL = "NOVEL_NO_HISTORICAL"


class EvidenceQualityResult(BaseModel):
    """
    Deterministic evidence quality scores for a single exception.

    These scores measure evidence quality, NOT:
    - ML prediction confidence
    - Resolution confidence
    - Authorization to auto-resolve

    Every score is traceable to observable evidence.
    """

    # Exception identification
    exception_id: str = Field(..., description="The exception being scored")
    case_id: str = Field(..., description="Reference to Case")

    # Consistency score (0.0 to 1.0)
    consistency_score: float = Field(
        ...,
        description="How consistently available evidence supports the explanation. "
        "1.0 = perfectly consistent, 0.0 = no consistency",
        ge=0.0,
        le=1.0,
    )

    # Coverage score (0.0 to 1.0)
    coverage_score: float = Field(
        ...,
        description="How much of the discrepancy is explained by evidence. "
        "1.0 = fully covered, 0.0 = no coverage",
        ge=0.0,
        le=1.0,
    )

    # Conflict indicator
    conflict: bool = Field(
        ..., description="True when materially competing explanations exist"
    )

    # Novelty indicator (deterministic only at this stage)
    novelty: NoveltyLevel = Field(
        ...,
        description="Deterministic novelty: KNOWN_PATTERN or NOVEL_NO_HISTORICAL. "
        "True semantic similarity/novelty added in Phase 4.",
    )

    # Missing evidence
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Entity types of missing evidence",
    )

    # Explanation status passthrough
    fully_explained: bool = Field(
        ..., description="Whether the discrepancy is fully explained"
    )
    partially_explained: bool = Field(
        ..., description="Whether the discrepancy is partially explained"
    )

    # Evidence count
    supporting_evidence_count: int = Field(
        ..., description="Number of evidence records supporting the explanation"
    )

    # Scoring breakdown (for explainability)
    consistency_breakdown: Optional[dict] = Field(
        default=None,
        description="Detailed breakdown of consistency score components",
    )

    def is_high_quality(self) -> bool:
        """Check if evidence quality is high (both scores >= 0.8, no conflict)."""
        return (
            self.consistency_score >= 0.8
            and self.coverage_score >= 0.8
            and not self.conflict
        )

    def needs_review(self) -> bool:
        """Check if evidence quality warrants human review."""
        return (
            self.consistency_score < 0.5
            or self.coverage_score < 0.5
            or self.conflict
            or len(self.missing_evidence) > 0
        )
