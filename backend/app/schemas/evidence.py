"""
Evidence package data contract for Razorpay CloseLoop.

Defines the structured output of evidence retrieval for a given exception.
All retrieval is deterministic — no ML, no LLM, no probabilistic reasoning.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """
    A single evidence record — one financial entity that serves as evidence
    for an exception.

    Answers:
    - What record is this?
    - How is it related to the exception?
    - What financial contribution does it make?
    - What case/payment does it belong to?
    """

    record_id: str = Field(..., description="Primary key of the financial record")
    entity_type: str = Field(
        ..., description="PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT"
    )
    relationship: str = Field(
        ..., description="Why this record is evidence: PRIMARY_RECORD, CALCULATION_COMPONENT, SUPPORTING_EVIDENCE, CONFLICTING_EVIDENCE"
    )
    amount: int = Field(..., description="Financial amount in paise")
    status: Optional[str] = Field(None, description="Record status if applicable")
    timestamp: Optional[datetime] = Field(None, description="Record timestamp if applicable")
    metadata: Optional[dict] = Field(None, description="Additional context")


class StructuralConflict(BaseModel):
    """
    A detected structural conflict in the evidence.

    Represents an anomaly that requires attention but is NOT resolved here.
    """

    conflict_type: str = Field(
        ...,
        description="MULTIPLE_SETTLEMENTS, DUPLICATE_SETTLEMENT_ID, COMPETING_EXPLANATIONS, STATUS_CONTRADICTION",
    )
    description: str = Field(..., description="Human-readable conflict description")
    affected_records: List[str] = Field(
        default_factory=list, description="IDs of records involved in the conflict"
    )


class MissingEvidence(BaseModel):
    """
    Explicit representation of missing evidence.

    Distinguishes between "record does not exist" and "was not queried".
    """

    entity_type: str = Field(..., description="What type of record is missing")
    expected: bool = Field(
        ..., description="True if this record type is expected for this payment"
    )
    reason: str = Field(
        ..., description="Why the record is considered missing"
    )


class EvidencePackage(BaseModel):
    """
    Complete evidence package for a single exception.

    Contains all financial records, missing evidence indicators,
    structural conflicts, and calculation context needed to understand
    why the exception occurred.

    This is the output of EvidenceRetrievalService.retrieve().
    """

    # Exception identification
    exception_id: str = Field(..., description="The exception being investigated")
    case_id: str = Field(..., description="Reference to Case")
    payment_id: str = Field(..., description="Reference to Payment")
    merchant_id: Optional[str] = Field(None, description="Reference to Merchant")

    # Financial context from reconciliation
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")
    exception_type: str = Field(..., description="Engine-classified exception type")

    # Evidence records (explicitly present)
    payment: Optional[EvidenceRecord] = Field(None, description="The payment record")
    settlements: List[EvidenceRecord] = Field(
        default_factory=list, description="Settlement records (0, 1, or many)"
    )
    refunds: List[EvidenceRecord] = Field(
        default_factory=list, description="Refund records"
    )
    fees: List[EvidenceRecord] = Field(default_factory=list, description="Fee records")
    taxes: List[EvidenceRecord] = Field(
        default_factory=list, description="Tax records"
    )
    adjustments: List[EvidenceRecord] = Field(
        default_factory=list, description="Adjustment records"
    )

    # Financial summary (deterministic calculation from retrieved records)
    total_settlement_amount: int = Field(
        default=0, description="Sum of all settlement amounts in paise"
    )
    total_refund_amount: int = Field(
        default=0, description="Sum of all refund amounts in paise"
    )
    total_fee_amount: int = Field(
        default=0, description="Sum of all fee amounts in paise"
    )
    total_tax_amount: int = Field(
        default=0, description="Sum of all tax amounts in paise"
    )
    total_adjustment_amount: int = Field(
        default=0, description="Net adjustments in paise (credit positive, debit negative)"
    )

    # Missing evidence
    missing_evidence: List[MissingEvidence] = Field(
        default_factory=list, description="Explicitly missing record types"
    )

    # Structural conflicts
    conflicts: List[StructuralConflict] = Field(
        default_factory=list, description="Detected structural anomalies"
    )

    # Metadata
    retrieved_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this evidence was retrieved",
    )
    evidence_link_count: int = Field(
        default=0, description="Number of evidence links persisted"
    )

    def has_missing_settlement(self) -> bool:
        """Check if settlement evidence is explicitly missing."""
        return any(m.entity_type == "SETTLEMENT" for m in self.missing_evidence)

    def has_conflicts(self) -> bool:
        """Check if any structural conflicts were detected."""
        return len(self.conflicts) > 0

    def total_evidence_records(self) -> int:
        """Count all evidence records."""
        count = 0
        if self.payment:
            count += 1
        count += len(self.settlements)
        count += len(self.refunds)
        count += len(self.fees)
        count += len(self.taxes)
        count += len(self.adjustments)
        return count
