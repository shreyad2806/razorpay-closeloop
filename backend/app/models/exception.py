"""
Database model for financial exceptions.

Stores exception records created by the reconciliation engine.
Ground truth labels are NOT stored here.

CloseLoop 2.0 refactor (architecture section 5): the record gains the lifecycle
fields the closed loop needs - status reason, assignment, SLA, risk, closure
metadata, reopen count, optimistic-locking version and currency - plus the
relationships into the investigation layer (evidence, root cause, resolutions,
model predictions, audit events, feedback, historical cases).

The investigation logic itself is NOT implemented here or anywhere in Phase 1;
these are persistence relationships only.

Status compatibility: the pre-2.0 column default is ``OPEN`` and the existing
reconciliation/evidence code writes ``OPEN``, so ``OPEN`` remains the default and
is retained as the persistence-time synonym for ``DETECTED`` in the state
machine. The legacy ``MATCHED`` and ``RESOLVED`` values are also retained. No
previous value is removed or renamed.
"""

from datetime import datetime

from sqlalchemy import Column, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.domain import state_machines
from app.models.types import (
    CurrencyType,
    MoneyType,
    TimestampType,
    currency_check,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
#
# This literal is the union of the section 7.3 lifecycle and the legacy values
# still written by pre-2.0 code, so the CHECK constraint accepts both.
EXCEPTION_STATUS_VALUES = (
    # Legacy values (still written today)
    "OPEN",
    "MATCHED",
    "RESOLVED",
    # Section 7.3 lifecycle
    "DETECTED",
    "INVESTIGATING",
    "ANALYZED",
    "RESOLUTION_PROPOSED",
    "AUTO_APPROVED",
    "HUMAN_REVIEW",
    "APPROVED",
    "REJECTED",
    "EXECUTING",
    "VERIFYING",
    "RECONCILING",
    "CLOSED",
    "ESCALATED",
    "FAILED",
    "ROLLED_BACK",
    "UNRESOLVED",
)
EXCEPTION_RISK_VALUES = ("LOW", "MEDIUM", "HIGH")


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
    payment_id = Column(
        String, nullable=False, index=True
    )  # FK to payments; see note
    batch_id = Column(String, nullable=False, index=True)  # For idempotency

    merchant_id = Column(String, nullable=True)
    currency = Column(
        CurrencyType, nullable=False, default="INR", server_default="INR"
    )

    # Financial amounts (in paise, integer)
    expected_amount = Column(MoneyType, nullable=False)
    actual_amount = Column(MoneyType, nullable=False)
    difference = Column(MoneyType, nullable=False)

    # Classification
    exception_type = Column(String(48), nullable=False)  # ExceptionType value
    status = Column(String(32), nullable=False, default="OPEN")  # ExceptionStatus
    status_reason = Column(Text, nullable=True)

    # Risk / ownership / SLA (section 7.3)
    risk_category = Column(String(16), nullable=True)
    assigned_to = Column(String(128), nullable=True)
    sla_due_at = Column(TimestampType, nullable=True)

    # Closure metadata
    closed_at = Column(TimestampType, nullable=True)
    close_reason = Column(String(64), nullable=True)
    reopen_count = Column(Integer, nullable=False, default=0, server_default="0")

    # References
    reconciliation_id = Column(String, nullable=False, index=True)

    # Optimistic locking (architecture section 6).
    version = Column(Integer, nullable=False, default=1, server_default="1")

    # Metadata (legacy defaults preserved exactly)
    created_at = Column(TimestampType, default=datetime.utcnow)
    updated_at = Column(
        TimestampType, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # -- investigation-layer relationships (foundation only) ----------------
    evidence = relationship("Evidence")
    root_causes = relationship("RootCause")
    resolutions = relationship("Resolution")
    model_predictions = relationship("ModelPrediction")
    audit_events = relationship("AuditEvent")
    feedback = relationship("Feedback")
    # NOTE: the historical-memory model is declared as ``HistoricalCaseRecord``
    # in app.services.historical_case_store (table ``historical_cases``).
    #
    # ``historical_cases.exception_id`` deliberately carries no FK constraint in
    # Phase 1 - pre-2.0 promotion paths and demo seeders write cases whose
    # exception predates the row - so the join is declared explicitly and the
    # relationship is view-only (read-only convenience, never a flush target).
    # The constraint lands with the promotion work in Phase 6.
    historical_cases = relationship(
        "HistoricalCaseRecord",
        primaryjoin=(
            "FinancialException.id == "
            "foreign(HistoricalCaseRecord.exception_id)"
        ),
        viewonly=True,
    )

    # Unique constraint for idempotency: one exception per case per batch.
    # Indexes for the queries section 5 calls out.
    __table_args__ = (
        non_negative_check("reopen_count", "ck_exceptions_reopen_count"),
        non_negative_check("version", "ck_exceptions_version"),
        currency_check(),
        enum_check("status", EXCEPTION_STATUS_VALUES, "ck_exceptions_status"),
        enum_check("risk_category", EXCEPTION_RISK_VALUES, "ck_exceptions_risk"),
        Index("ix_exceptions_case_batch", "case_id", "batch_id", unique=True),
        Index("ix_exceptions_status_created", "status", "created_at"),
        Index("ix_exceptions_type_status", "exception_type", "status"),
        Index("ix_exceptions_merchant_status", "merchant_id", "status"),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.status -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.EXCEPTION, self.status, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a lifecycle transition; raises when invalid.

        Note that ``CLOSED`` is reachable only from ``RECONCILING`` (section
        7.3); the machine rejects every attempt to close an exception
        directly.
        """
        self.status = state_machines.assert_transition(
            state_machines.EXCEPTION, self.status, new_status
        )
        return self.status


class ExceptionStatus:
    """Controlled status values for exception records.

    Retained for backwards compatibility: existing code and tests reference
    ``ExceptionStatus.OPEN`` and friends as plain string constants. The section
    7.3 lifecycle values are exposed alongside them so the full machine is
    available from one place; ``app.schemas.enums.ExceptionLifecycleStatus``
    carries the same values as an enum.
    """

    # Legacy
    OPEN = "OPEN"
    MATCHED = "MATCHED"
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"

    # Section 7.3 lifecycle
    DETECTED = "DETECTED"
    INVESTIGATING = "INVESTIGATING"
    ANALYZED = "ANALYZED"
    RESOLUTION_PROPOSED = "RESOLUTION_PROPOSED"
    AUTO_APPROVED = "AUTO_APPROVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECONCILING = "RECONCILING"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
