"""
Database model for financial exceptions.

Stores exception records created by the reconciliation engine.
Ground truth labels are NOT stored here.
"""

from sqlalchemy import Column, String, Integer, DateTime, Index
from datetime import datetime

from app.database.database import Base


class FinancialException(Base):
    """
    Database model for financial exceptions.

    This stores exception records created when the reconciliation engine
    detects a discrepancy. Matched cases do NOT get exception records.

    Ground truth is stored separately for evaluation.
    """

    __tablename__ = "exceptions"

    id = Column(String, primary_key=True)  # exception_id
    case_id = Column(String, nullable=False, index=True)
    payment_id = Column(String, nullable=False, index=True)
    batch_id = Column(String, nullable=False, index=True)  # For idempotency

    # Financial amounts (in paise, integer)
    expected_amount = Column(Integer, nullable=False)
    actual_amount = Column(Integer, nullable=False)
    difference = Column(Integer, nullable=False)

    # Classification
    exception_type = Column(String, nullable=False)  # ExceptionType enum value
    status = Column(String, nullable=False, default="OPEN")  # ExceptionStatus

    # References
    reconciliation_id = Column(String, nullable=False, index=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint for idempotency: one exception per case per batch
    __table_args__ = (
        Index("ix_exceptions_case_batch", "case_id", "batch_id", unique=True),
    )


class ExceptionStatus:
    """Controlled status values for exception records."""
    OPEN = "OPEN"
    MATCHED = "MATCHED"
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
