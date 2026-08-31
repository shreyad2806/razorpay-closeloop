"""
Historical case storage schema for Razorpay CloseLoop Phase 4E.

Defines the structured representation of resolved financial cases
for later retrieval as historical evidence for similar exceptions.

Connects: case → exception → financial context → evidence → resolution → outcome.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ResolutionOutcome(str, Enum):
    """Outcome of a resolution attempt."""

    SUCCESSFUL = "SUCCESSFUL"
    UNSUCCESSFUL = "UNSUCCESSFUL"
    REVERSED = "REVERSED"
    MANUALLY_REVIEWED = "MANUALLY_REVIEWED"


class ResolutionOrigin(str, Enum):
    """Origin of the resolution."""

    HUMAN = "HUMAN"
    DETERMINISTIC = "DETERMINISTIC"
    ML = "ML"
    AGENT = "AGENT"


class HistoricalEvidenceRef(BaseModel):
    """Reference to evidence that supported a resolution.

    Lightweight pointer to financial records — does not duplicate data.
    """

    entity_type: str = Field(
        ...,
        description="PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT",
    )
    entity_id: str = Field(..., description="Primary key of the financial record")
    relationship: str = Field(
        ...,
        description="Why this record was evidence: SUPPORTING_EVIDENCE, CALCULATION_COMPONENT, etc.",
    )
    amount: int = Field(..., description="Amount in paise")


class FinancialContext(BaseModel):
    """Financial context captured at resolution time.

    Stores the amounts that were relevant when the case was resolved.
    """

    payment_amount: int = Field(..., description="Original payment in paise")
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")
    total_refunds: int = Field(default=0, description="Total refunds in paise")
    total_fees: int = Field(default=0, description="Total fees in paise")
    total_taxes: int = Field(default=0, description="Total taxes in paise")
    total_adjustments: int = Field(default=0, description="Net adjustments in paise")


class HistoricalCase(BaseModel):
    """Structured representation of a resolved financial case.

    A historical case connects all information needed to later retrieve
    and learn from past resolutions:

    - What happened (exception type, financial context)
    - What evidence was available (evidence references)
    - How it was resolved (resolution type, outcome, origin)
    - When it happened (timestamps)
    - Metadata for retrieval (exception type, risk, tags)
    """

    # Identification
    case_id: str = Field(..., description="Unique case identifier")
    exception_id: str = Field(..., description="Associated exception identifier")
    payment_id: str = Field(..., description="Associated payment identifier")
    merchant_id: Optional[str] = Field(None, description="Associated merchant identifier")

    # Exception classification (deterministic, not ground truth)
    exception_type: str = Field(
        ..., description="Engine-classified exception type at resolution time"
    )

    # Financial context
    financial_context: FinancialContext = Field(
        ..., description="Financial amounts at resolution time"
    )

    # Resolution
    resolution_type: str = Field(
        ..., description="Resolution type from ResolutionType taxonomy"
    )
    resolution_outcome: ResolutionOutcome = Field(
        ..., description="Whether resolution was successful"
    )
    resolution_origin: ResolutionOrigin = Field(
        default=ResolutionOrigin.DETERMINISTIC,
        description="Who/what applied the resolution",
    )
    resolved_amount: Optional[int] = Field(
        default=None, description="Amount used in resolution (paise)"
    )
    resolution_notes: Optional[str] = Field(
        default=None, description="Free-text resolution notes"
    )

    # Evidence references
    evidence_refs: List[HistoricalEvidenceRef] = Field(
        default_factory=list,
        description="Evidence records that supported the resolution",
    )
    supporting_evidence_count: int = Field(
        default=0, description="Number of evidence records"
    )

    # Quality indicators
    exception_type_confidence: Optional[float] = Field(
        default=None,
        description="Confidence in exception type classification (0.0-1.0)",
    )
    evidence_coverage: Optional[float] = Field(
        default=None,
        description="How much of the discrepancy was explained (0.0-1.0)",
    )

    # Metadata
    tags: List[str] = Field(
        default_factory=list, description="Searchable tags for retrieval"
    )
    resolution_metadata: Optional[Dict] = Field(
        default=None, description="Additional structured metadata"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the historical case was created",
    )
    resolved_at: Optional[datetime] = Field(
        default=None, description="When the case was resolved"
    )

    def to_retrieval_features(self) -> Dict:
        """Extract features useful for future similarity retrieval.

        Returns a flat dict of key attributes that can be used for
        deterministic or embedding-based similarity matching.
        """
        return {
            "case_id": self.case_id,
            "exception_type": self.exception_type,
            "resolution_type": self.resolution_type,
            "resolution_outcome": self.resolution_outcome.value,
            "payment_amount": self.financial_context.payment_amount,
            "difference": self.financial_context.difference,
            "total_refunds": self.financial_context.total_refunds,
            "total_fees": self.financial_context.total_fees,
            "total_taxes": self.financial_context.total_taxes,
            "total_adjustments": self.financial_context.total_adjustments,
            "evidence_count": self.supporting_evidence_count,
            "evidence_coverage": self.evidence_coverage or 0.0,
            "tags": self.tags,
        }

    def validate_financial_integrity(self) -> List[str]:
        """Validate financial consistency.

        Returns list of errors (empty = valid).
        """
        errors = []
        fc = self.financial_context

        if fc.payment_amount < 0:
            errors.append("payment_amount must be non-negative")
        if fc.actual_amount < 0:
            errors.append("actual_amount must be non-negative")
        if fc.expected_amount < 0:
            errors.append("expected_amount must be non-negative")

        expected_diff = fc.expected_amount - fc.actual_amount
        if expected_diff != fc.difference:
            errors.append(
                f"difference mismatch: expected {expected_diff}, got {fc.difference}"
            )

        if self.resolved_amount is not None and self.resolved_amount < 0:
            errors.append("resolved_amount must be non-negative")

        return errors
