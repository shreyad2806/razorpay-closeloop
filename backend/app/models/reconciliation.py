"""
Database model for reconciliation results.

Stores the deterministic output of the reconciliation engine.
Ground truth labels are NOT stored here - they are in separate evaluation tables.

CloseLoop 2.0 (architecture section 5): the result gains a foreign key to the
run that produced it (``reconciliation_run_id``), an explicit currency, and CHECK
constraints on its enum-ish string columns. The pre-2.0 unique
``(case_id, batch_id)`` idempotency index is preserved as-is;
``ix_reconciliation_case_batch`` and the newer run linkage coexist.

``total_adjustments`` and ``difference`` are SIGNED (adjustments are signed and a
difference can be negative), so no non-negative CHECK is applied to them.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.models.types import (
    CurrencyType,
    MoneyType,
    TimestampType,
    currency_check,
    enum_check,
    non_negative_check,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
MATCH_STATUS_VALUES = ("MATCHED", "EXCEPTION", "MISSING", "DUPLICATE")
RECONCILIATION_RESULT_STATUS_VALUES = (
    "PENDING",
    "PROCESSED",
    "FAILED",
    "REVIEW_REQUIRED",
)
EXCEPTION_TYPE_VALUES = (
    "EXACT_MATCH",
    "FEE_DIFFERENCE",
    "FEE_MISMATCH",
    "REFUND_ADJUSTMENT",
    "TAX_ADJUSTMENT",
    "TIMING_DIFFERENCE",
    "PARTIAL_SETTLEMENT",
    "DUPLICATE",
    "MISSING_RECORD",
    "COMPLEX_MULTI_ADJUSTMENT",
    "UNKNOWN",
)


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

    # The run that produced this result. Nullable so pre-2.0 rows (and code
    # paths that do not yet run under a ReconciliationRun) remain valid; Phase 3
    # writes it for every new result.
    reconciliation_run_id = Column(
        String,
        ForeignKey("reconciliation_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    # Financial calculation (in paise)
    payment_amount = Column(MoneyType, nullable=False)
    total_refunds = Column(MoneyType, default=0)
    total_fees = Column(MoneyType, default=0)
    total_taxes = Column(MoneyType, default=0)
    # SIGNED: the sum of signed adjustments can be negative.
    total_adjustments = Column(MoneyType, default=0)

    currency = Column(
        CurrencyType, nullable=False, default="INR", server_default="INR"
    )

    expected_amount = Column(MoneyType, nullable=False)
    actual_amount = Column(MoneyType, nullable=False)
    # SIGNED: actual - expected can be positive or negative.
    difference = Column(MoneyType, nullable=False)

    # Classification
    match_status = Column(String(16), nullable=False)  # MatchStatus enum value
    exception_type = Column(String(48), nullable=False)  # ExceptionType enum value

    # Processing metadata
    reconciliation_status = Column(String(24), default="PROCESSED")
    reconciliation_timestamp = Column(TimestampType, default=datetime.utcnow)
    processing_notes = Column(String, nullable=True)

    run = relationship("ReconciliationRun", back_populates="results")

    # Unique constraint for idempotency: one result per case per batch
    __table_args__ = (
        non_negative_check("payment_amount", "ck_reconciliation_results_payment_amount"),
        non_negative_check("total_refunds", "ck_reconciliation_results_total_refunds"),
        non_negative_check("total_fees", "ck_reconciliation_results_total_fees"),
        non_negative_check("total_taxes", "ck_reconciliation_results_total_taxes"),
        non_negative_check("expected_amount", "ck_reconciliation_results_expected"),
        non_negative_check("actual_amount", "ck_reconciliation_results_actual"),
        currency_check(),
        enum_check(
            "match_status", MATCH_STATUS_VALUES, "ck_reconciliation_results_match_status"
        ),
        enum_check(
            "exception_type",
            EXCEPTION_TYPE_VALUES,
            "ck_reconciliation_results_exception_type",
        ),
        enum_check(
            "reconciliation_status",
            RECONCILIATION_RESULT_STATUS_VALUES,
            "ck_reconciliation_results_status",
        ),
        Index("ix_reconciliation_case_batch", "case_id", "batch_id", unique=True),
    )


class ReconciliationEvidence(Base):
    """
    Database model for reconciliation evidence.

    Stores supporting evidence for reconciliation decisions.
    Used for audit trail and debugging.

    Superseded by ``evidence`` (see app.models.evidence) in CloseLoop 2.0; kept
    intact in Phase 1 because existing services and tests read it. The merge is
    Phase 4 work.
    """

    __tablename__ = "reconciliation_evidence"

    id = Column(String, primary_key=True)
    reconciliation_id = Column(String, nullable=False, index=True)
    evidence_type = Column(String, nullable=False)  # e.g., "CALCULATION_BREAKDOWN"
    evidence_data = Column(String, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
