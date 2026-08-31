"""
Similarity retrieval schemas for Razorpay CloseLoop Phase 4F.

Defines the structured output of semantic similarity search
for historical cases.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    """A single similar historical case returned by similarity search."""

    case_id: str = Field(..., description="Historical case identifier")
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0.0 to 1.0)", ge=-1.0, le=1.0
    )
    exception_type: str = Field(..., description="Exception type of the historical case")
    resolution_type: str = Field(..., description="Resolution that was applied")
    resolution_outcome: str = Field(..., description="Outcome of the resolution")
    payment_amount: int = Field(..., description="Payment amount in paise")
    difference: int = Field(..., description="Discrepancy in paise")
    evidence_count: int = Field(default=0, description="Number of evidence records")
    tags: List[str] = Field(default_factory=list, description="Case tags")


class SimilaritySearchResult(BaseModel):
    """Result of a semantic similarity search for a new exception."""

    query_case_id: str = Field(
        ..., description="The case_id of the query exception"
    )
    similar_cases: List[SimilarCase] = Field(
        default_factory=list, description="Ranked similar historical cases"
    )
    top_k: int = Field(..., description="Number of results requested")
    total_indexed: int = Field(
        ..., description="Total historical cases in the index"
    )
    embedding_model: str = Field(..., description="Model used for embeddings")
    embedding_dimension: int = Field(..., description="Dimension of embeddings")
    similarity_metric: str = Field(
        default="cosine", description="Similarity metric used"
    )

    def has_results(self) -> bool:
        """Check if any similar cases were found."""
        return len(self.similar_cases) > 0

    def best_match(self) -> Optional[SimilarCase]:
        """Get the most similar case, or None."""
        if self.similar_cases:
            return self.similar_cases[0]
        return None

    def above_threshold(self, threshold: float = 0.7) -> List[SimilarCase]:
        """Filter results above a similarity threshold."""
        return [c for c in self.similar_cases if c.similarity_score >= threshold]
