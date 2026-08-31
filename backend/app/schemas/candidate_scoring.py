"""
Resolution candidate scoring schema for Razorpay CloseLoop Phase 5C.

Defines the scoring components and configuration for ranking resolution candidates.

This is a RECOMMENDATION SCORE.
It is NOT financial truth.
It does NOT authorize execution.
"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ScoringConfig(BaseModel):
    """Configurable scoring weights and thresholds.

    All weights sum to 1.0 by default but can be adjusted.
    """

    # Component weights (must sum to 1.0)
    evidence_weight: float = Field(
        default=0.35, description="Weight for evidence score", ge=0.0, le=1.0
    )
    ml_weight: float = Field(
        default=0.20, description="Weight for ML score", ge=0.0, le=1.0
    )
    historical_weight: float = Field(
        default=0.15, description="Weight for historical similarity score", ge=0.0, le=1.0
    )
    financial_weight: float = Field(
        default=0.30, description="Weight for financial consistency score", ge=0.0, le=1.0
    )

    # Penalty multipliers
    novelty_penalty_factor: float = Field(
        default=0.15, description="Max penalty for novel cases (0.0-0.3)"
    )
    conflict_penalty_factor: float = Field(
        default=0.25, description="Max penalty for conflicting signals (0.0-0.5)"
    )

    # Thresholds
    min_similarity_for_bonus: float = Field(
        default=0.7, description="Min historical similarity for bonus"
    )
    max_historical_cases_for_bonus: int = Field(
        default=3, description="Max cases contributing to historical score"
    )

    def validate_weights(self) -> bool:
        """Check that component weights sum to approximately 1.0."""
        total = self.evidence_weight + self.ml_weight + self.historical_weight + self.financial_weight
        return abs(total - 1.0) < 0.01


class CandidateScore(BaseModel):
    """
    Detailed scoring breakdown for a single resolution candidate.

    Every score is traceable and explainable.
    This is a RECOMMENDATION SCORE — not financial truth.
    """

    # Component scores (0.0 to 1.0)
    evidence_score: float = Field(
        ..., description="Evidence coverage and consistency score", ge=0.0, le=1.0
    )
    ml_score: float = Field(
        ..., description="ML prediction confidence score", ge=0.0, le=1.0
    )
    historical_score: float = Field(
        ..., description="Historical similarity score", ge=0.0, le=1.0
    )
    financial_consistency_score: float = Field(
        ..., description="Financial adjustment consistency score", ge=0.0, le=1.0
    )

    # Penalties (0.0 to 1.0, subtracted from composite)
    novelty_penalty: float = Field(
        default=0.0, description="Penalty for novel/unfamiliar patterns", ge=0.0, le=1.0
    )
    conflict_penalty: float = Field(
        default=0.0, description="Penalty for conflicting signals", ge=0.0, le=1.0
    )

    # Composite
    final_score: float = Field(
        ..., description="Final composite score 0.0-1.0", ge=0.0, le=1.0
    )

    # Breakdown
    weighted_evidence: float = Field(default=0.0, description="evidence_score * weight")
    weighted_ml: float = Field(default=0.0, description="ml_score * weight")
    weighted_historical: float = Field(default=0.0, description="historical_score * weight")
    weighted_financial: float = Field(default=0.0, description="financial_score * weight")

    # Flags
    has_evidence_support: bool = Field(default=False)
    has_ml_support: bool = Field(default=False)
    has_historical_support: bool = Field(default=False)
    is_novel: bool = Field(default=False, description="Whether this is a novel pattern")
    has_conflicts: bool = Field(default=False, description="Whether conflicts were detected")

    def explanation(self) -> str:
        """Generate human-readable scoring explanation."""
        parts = [
            f"Evidence: {self.evidence_score:.3f} (weighted: {self.weighted_evidence:.3f})",
            f"ML: {self.ml_score:.3f} (weighted: {self.weighted_ml:.3f})",
            f"Historical: {self.historical_score:.3f} (weighted: {self.weighted_historical:.3f})",
            f"Financial: {self.financial_consistency_score:.3f} (weighted: {self.weighted_financial:.3f})",
        ]
        if self.novelty_penalty > 0:
            parts.append(f"Novelty penalty: -{self.novelty_penalty:.3f}")
        if self.conflict_penalty > 0:
            parts.append(f"Conflict penalty: -{self.conflict_penalty:.3f}")
        parts.append(f"Final: {self.final_score:.3f}")
        return " | ".join(parts)
