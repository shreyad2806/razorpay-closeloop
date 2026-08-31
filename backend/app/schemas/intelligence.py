"""
Exception Intelligence schema for Razorpay CloseLoop Phase 4G.

Defines the unified intelligence result that combines:
- Deterministic reconciliation
- Evidence retrieval + explanation
- ML classification + resolution prediction
- Historical similarity
- Conflict detection

This is an INTELLIGENCE OUTPUT only.
It must NOT modify financial records.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationStatus(str, Enum):
    """Status of the combined recommendation."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ResolutionCandidate(BaseModel):
    """A single resolution candidate from any source."""

    resolution_type: str = Field(
        ..., description="Proposed resolution type from ResolutionType taxonomy"
    )
    source: str = Field(
        ...,
        description="Where this candidate came from: ML_PREDICTION, HISTORICAL_SIMILARITY, DETERMINISTIC_EVIDENCE",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence record IDs supporting this resolution"
    )
    evidence_compatible: bool = Field(
        ..., description="Whether deterministic evidence supports this resolution"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score (0.0-1.0) where applicable (ML probability or similarity score)",
    )
    similarity_score: Optional[float] = Field(
        default=None, description="Historical similarity score if from similarity search"
    )
    historical_case_id: Optional[str] = Field(
        default=None, description="Historical case ID if from similarity search"
    )
    notes: Optional[str] = Field(default=None, description="Additional context")


class ClassificationResult(BaseModel):
    """Exception type classification from multiple sources."""

    deterministic_type: str = Field(
        ..., description="Exception type from deterministic reconciliation engine"
    )
    ml_predicted_type: Optional[str] = Field(
        default=None, description="Exception type from XGBoost classifier"
    )
    ml_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="ML class probabilities"
    )
    ml_model_version: Optional[str] = Field(
        default=None, description="Version of the ML model used"
    )
    agreement: bool = Field(
        ..., description="True if deterministic and ML types agree"
    )
    disagreement_note: Optional[str] = Field(
        default=None, description="Description of disagreement if any"
    )


class EvidenceIntelligence(BaseModel):
    """Evidence and explanation intelligence."""

    explanation_status: str = Field(
        ..., description="FULLY_EXPLAINED, PARTIALLY_EXPLAINED, UNEXPLAINED, CONFLICTING"
    )
    explained_amount: int = Field(
        default=0, description="Portion of discrepancy explained in paise"
    )
    remaining_difference: int = Field(
        default=0, description="Unexplained portion in paise"
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence IDs explaining the discrepancy"
    )
    evidence_coverage: float = Field(
        default=0.0, description="Coverage score (0.0-1.0)"
    )
    consistency_score: float = Field(
        default=0.0, description="Evidence consistency score (0.0-1.0)"
    )
    has_conflict: bool = Field(
        default=False, description="Whether conflicting explanations exist"
    )
    missing_evidence: List[str] = Field(
        default_factory=list, description="Entity types of missing evidence"
    )
    explanation_reason: str = Field(
        default="", description="Deterministic explanation text"
    )
    evidence_link_count: int = Field(
        default=0, description="Number of evidence links persisted"
    )


class SimilarCasesIntelligence(BaseModel):
    """Historical similarity intelligence."""

    query_embedded: bool = Field(
        default=True, description="Whether the query was successfully embedded"
    )
    total_indexed: int = Field(
        default=0, description="Total historical cases in the index"
    )
    top_k: int = Field(default=5, description="Number of results requested")
    similar_cases: List[Dict] = Field(
        default_factory=list,
        description="Ranked similar historical cases with scores",
    )
    best_similarity_score: Optional[float] = Field(
        default=None, description="Highest similarity score found"
    )
    embedding_model: str = Field(
        default="", description="Model used for embeddings"
    )


class ExceptionIntelligence(BaseModel):
    """
    Complete intelligence result for a single exception.

    Combines all Phase 4 intelligence sources into one unified output.
    This is an INTELLIGENCE OUTPUT only — it does NOT modify financial records.

    Contains:
    - Deterministic classification + ML classification
    - Evidence explanation + quality
    - Historical similarity
    - Resolution candidates from multiple sources
    - Conflict detection
    - Recommendation status
    """

    # Identification
    exception_id: str = Field(..., description="The exception analyzed")
    case_id: str = Field(..., description="Associated case")
    payment_id: str = Field(..., description="Associated payment")
    merchant_id: Optional[str] = Field(None, description="Associated merchant")

    # Financial context
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")

    # Classification
    classification: ClassificationResult = Field(
        ..., description="Exception type classification from multiple sources"
    )

    # Evidence
    evidence: EvidenceIntelligence = Field(
        ..., description="Evidence and explanation intelligence"
    )

    # Historical similarity
    similar_cases: SimilarCasesIntelligence = Field(
        default_factory=SimilarCasesIntelligence,
        description="Historical similarity intelligence",
    )

    # Resolution candidates
    resolution_candidates: List[ResolutionCandidate] = Field(
        default_factory=list, description="Ranked resolution candidates"
    )

    # Conflicts
    conflicts: List[str] = Field(
        default_factory=list,
        description="Detected conflicts between intelligence sources",
    )

    # Recommendation
    recommendation_status: RecommendationStatus = Field(
        ..., description="Overall recommendation status"
    )
    recommendation_notes: List[str] = Field(
        default_factory=list, description="Explanation of the recommendation"
    )

    # Metadata
    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When intelligence was generated",
    )
    pipeline_version: str = Field(
        default="1.0.0", description="Version of the intelligence pipeline"
    )

    # Safety
    is_intelligence_only: bool = Field(
        default=True,
        description="Always True — this output must NOT execute financial actions",
    )

    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    def is_supported(self) -> bool:
        """Check if the recommendation is fully supported."""
        return self.recommendation_status == RecommendationStatus.SUPPORTED

    def summary(self) -> str:
        """Generate a human-readable summary."""
        parts = [
            f"Exception {self.exception_id}: {self.classification.deterministic_type}",
            f"Difference: {self.difference} paise",
            f"Explanation: {self.evidence.explanation_status}",
            f"Candidates: {len(self.resolution_candidates)}",
            f"Status: {self.recommendation_status.value}",
        ]
        if self.conflicts:
            parts.append(f"Conflicts: {len(self.conflicts)}")
        return " | ".join(parts)
