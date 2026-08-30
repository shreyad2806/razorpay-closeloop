"""
Database model for reconciliation results.

Stores the deterministic output of the reconciliation engine.
Ground truth labels are NOT stored here — they are in separate evaluation tables.
"""

from sqlalchemy import Column, String, Integer, DateTime, Index
from datetime import datetime

from app.database.database import Base


class ReconciliationResult(Base):
    """
    Database model for reconciliation results.

    This stores the engine's independent calculation and classification.
    Ground truth is stored separately for evaluation.
    """

    __tablename__ = "reconciliation_results"

    id = Column(String, primary_key=True)  # reconciliation_id
    case_id = Column(String, nullable=False, index=True)
    payment_id = Column(String, nullable=False, index=True)
    merchant_id = Column(String, nullable=False)
    batch_id = Column(String, nullable=False, index=True)  # For idempotency

    # Financial calculation (in paise)
    payment_amount = Column(Integer, nullable=False)
    total_refunds = Column(Integer, default=0)
    total_fees = Column(Integer, default=0)
    total_taxes = Column(Integer, default=0)
    total_adjustments = Column(Integer, default=0)

    expected_amount = Column(Integer, nullable=False)
    actual_amount = Column(Integer, nullable=False)
    difference = Column(Integer, nullable=False)

    # Classification
    match_status = Column(String, nullable=False)  # MatchStatus enum value
    exception_type = Column(String, nullable=False)  # ExceptionType enum value

    # Processing metadata
    reconciliation_status = Column(String, default="PROCESSED")
    reconciliation_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_notes = Column(String, nullable=True)

    # Unique constraint for idempotency: one result per case per batch
    __table_args__ = (
        Index("ix_reconciliation_case_batch", "case_id", "batch_id", unique=True),
    )


class ReconciliationEvidence(Base):
    """
    Database model for reconciliation evidence.

    Stores supporting evidence for reconciliation decisions.
    Used for audit trail and debugging.
    """

    __tablename__ = "reconciliation_evidence"

    id = Column(String, primary_key=True)
    reconciliation_id = Column(String, nullable=False, index=True)
    evidence_type = Column(String, nullable=False)  # e.g., "CALCULATION_BREAKDOWN"
    evidence_data = Column(String, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
