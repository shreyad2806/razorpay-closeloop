"""
Resolution candidate schema for Razorpay CloseLoop Phase 5A.

Defines the structured output of resolution candidate generation.
Each candidate represents a concrete, ranked financial resolution proposal.

This is a RECOMMENDATION ONLY.
It must NOT execute financial actions.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CandidateSource(str, Enum):
    """Where a resolution candidate originated."""

    DETERMINISTIC_EVIDENCE = "deterministic_evidence"
    ML_PREDICTION = "ml_prediction"
    HISTORICAL_CASE = "historical_case"
    COMBINED = "combined"


class FinancialAdjustment(BaseModel):
    """Explicit financial adjustment proposed by a candidate.

    Every adjustment must trace back to actual financial evidence.
    Never invented amounts.
    """

    adjustment_type: str = Field(
        ...,
        description="Type of adjustment: FEE_CORRECTION, REFUND_CORRECTION, TAX_CORRECTION, SETTLEMENT_ADJUSTMENT, NO_ADJUSTMENT",
    )
    amount_paise: int = Field(
        ..., description="Adjustment amount in paise (integer, never float)"
    )
    direction: str = Field(
        ...,
        description="CREDIT (add to merchant), DEBIT (deduct from merchant), NONE",
    )
    evidence_record_id: Optional[str] = Field(
        default=None,
        description="The financial record that justifies this amount",
    )
    calculation_basis: str = Field(
        ...,
        description="How this amount was derived: discrepancy, fee_record, refund_record, tax_record, settlement_diff, etc.",
    )


class EvidenceRecordRef(BaseModel):
    """Reference to a specific financial evidence record."""

    record_id: str = Field(..., description="Primary key of the financial record")
    entity_type: str = Field(
        ...,
        description="PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT",
    )
    amount: int = Field(..., description="Amount in paise")
    relationship: str = Field(
        ...,
        description="How this record relates: CALCULATION_COMPONENT, SUPPORTING_EVIDENCE, etc.",
    )
    contribution: Optional[int] = Field(
        default=None,
        description="Financial contribution to expected amount in paise",
    )


class MLSupportDetail(BaseModel):
    """Detailed ML support information for a candidate."""

    supported: bool = Field(..., description="Whether ML supports this resolution")
    predicted_resolution: Optional[str] = Field(
        default=None, description="Resolution type predicted by ML"
    )
    confidence: Optional[float] = Field(
        default=None, description="ML prediction confidence 0.0-1.0"
    )
    model_version: Optional[str] = Field(
        default=None, description="Version of the ML model"
    )
    probability: Optional[float] = Field(
        default=None, description="Probability of this specific resolution class"
    )


class HistoricalSupportDetail(BaseModel):
    """Detailed historical case support information."""

    case_id: str = Field(..., description="Historical case identifier")
    similarity_score: float = Field(
        ..., description="Cosine similarity score 0.0-1.0"
    )
    historical_resolution: str = Field(
        ..., description="Resolution applied in the historical case"
    )
    historical_outcome: Optional[str] = Field(
        default=None, description="Outcome of the historical resolution"
    )
    payment_amount: Optional[int] = Field(
        default=None, description="Payment amount in the historical case"
    )
    difference: Optional[int] = Field(
        default=None, description="Discrepancy in the historical case"
    )


class RationaleComponent(BaseModel):
    """A single component of the structured rationale."""

    component_type: str = Field(
        ...,
        description="what_happened, evidence_support, ml_support, historical_support, financial_trace, recommendation",
    )
    description: str = Field(..., description="Human-readable explanation")
    evidence_ids: List[str] = Field(
        default_factory=list, description="Related evidence record IDs"
    )
    amount_paise: Optional[int] = Field(
        default=None, description="Related amount in paise"
    )


class CandidateRanking(BaseModel):
    """Ranking metadata for a candidate."""

    rank: int = Field(..., description="Rank among all candidates (1 = best)")
    confidence_score: float = Field(
        ..., description="Combined confidence 0.0-1.0", ge=0.0, le=1.0
    )
    evidence_support: float = Field(
        ..., description="Evidence coverage supporting this candidate 0.0-1.0"
    )
    ml_support: Optional[float] = Field(
        default=None, description="ML probability for this resolution"
    )
    historical_support: Optional[float] = Field(
        default=None, description="Best historical similarity score"
    )


class ResolutionProposal(BaseModel):
    """
    A single, concrete financial resolution proposal.

    Contains everything needed to understand, evaluate, and optionally execute
    the proposed resolution. This is a recommendation ONLY.

    Financial adjustments are always derived from actual evidence — never invented.
    """

    # Identification
    candidate_id: str = Field(
        ..., description="Unique candidate identifier, e.g. CAND-001-FEE"
    )
    exception_id: str = Field(..., description="Exception this candidate addresses")
    case_id: str = Field(..., description="Case this candidate addresses")

    # Resolution
    resolution_type: str = Field(
        ..., description="Resolution type from ResolutionType taxonomy"
    )
    resolution_description: str = Field(
        ..., description="Human-readable description of the proposed resolution"
    )

    # Financial adjustment
    financial_adjustment: FinancialAdjustment = Field(
        ..., description="Explicit financial adjustment proposed"
    )

    # Evidence
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence record IDs supporting this proposal"
    )
    evidence_records: List[EvidenceRecordRef] = Field(
        default_factory=list,
        description="Detailed evidence record references with amounts and relationships",
    )
    evidence_compatible: bool = Field(
        ..., description="Whether deterministic evidence supports this proposal"
    )
    evidence_coverage: float = Field(
        default=0.0, description="Evidence coverage score 0.0-1.0"
    )
    coverage_explanation: str = Field(
        default="", description="How the candidate explains the financial discrepancy"
    )

    # ML support
    ml_support: Optional[MLSupportDetail] = Field(
        default=None, description="Detailed ML support information"
    )

    # Historical support
    historical_support: List[HistoricalSupportDetail] = Field(
        default_factory=list,
        description="Historical cases supporting this candidate",
    )

    # Sources
    sources: List[str] = Field(
        ...,
        description="All sources supporting this candidate: deterministic_evidence, ml_prediction, historical_case",
    )

    # Ranking
    ranking: CandidateRanking = Field(
        ..., description="Ranking and confidence metadata"
    )

    # Rationale
    rationale: str = Field(
        ..., description="Detailed explanation of why this resolution is proposed"
    )
    rationale_components: List[RationaleComponent] = Field(
        default_factory=list,
        description="Structured rationale components",
    )

    # Safety
    is_recommendation_only: bool = Field(
        default=True,
        description="Always True — this proposal must NOT be auto-executed",
    )

    # Metadata
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this candidate was generated",
    )
    pipeline_version: str = Field(
        default="1.0.0", description="Version of the candidate generation pipeline"
    )

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "exception_id": self.exception_id,
            "resolution_type": self.resolution_type,
            "financial_adjustment": {
                "adjustment_type": self.financial_adjustment.adjustment_type,
                "amount_paise": self.financial_adjustment.amount_paise,
                "direction": self.financial_adjustment.direction,
            },
            "sources": self.sources,
            "confidence": self.ranking.confidence_score,
            "evidence_compatible": self.evidence_compatible,
            "is_recommendation_only": True,
        }


class CandidateGenerationResult(BaseModel):
    """Result of resolution candidate generation for a single exception."""

    exception_id: str = Field(..., description="Exception analyzed")
    case_id: str = Field(..., description="Case analyzed")
    status: str = Field(
        ...,
        description="CANDIDATES_GENERATED or UNRESOLVED",
    )
    candidates: List[ResolutionProposal] = Field(
        default_factory=list, description="Ranked resolution candidates"
    )
    total_candidates: int = Field(default=0, description="Number of candidates")
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When candidates were generated",
    )
    pipeline_version: str = Field(default="1.0.0")

    def is_unresolved(self) -> bool:
        return self.status == "UNRESOLVED"

    def best_candidate(self) -> Optional[ResolutionProposal]:
        if self.candidates:
            return self.candidates[0]
        return None
