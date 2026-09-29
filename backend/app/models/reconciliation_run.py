"""
ReconciliationRun - one execution of the deterministic reconciliation engine.

NEW in CloseLoop 2.0 (architecture sections 5 and 7.2). Before 2.0 a "batch" was
an in-memory dictionary, so a run had no identity, no status, no counts and no
reproducibility guarantee. A run row is what makes reconciliation auditable:
results and exceptions are written inside the run's transaction, and the run's
``engine_version`` records which code produced them.

Lifecycle (section 7.2): ``PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED``.
The machine lives in :mod:`app.domain.state_machines` and is applied by
:meth:`ReconciliationRun.transition_to`.

The "no other RUNNING run for the same scope" validation required by section
7.2 is a *partial* unique index, supported by both PostgreSQL and SQLite, so it
is genuinely testable on the repository's SQLite harness.

No reconciliation logic is implemented here (task section 15).
"""

from sqlalchemy import Column, Index, Integer, String, text
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.domain import state_machines
from app.models.types import (
    JSONType,
    TimestampType,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
RECONCILIATION_RUN_STATUS_VALUES = (
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
)
RECONCILIATION_SCOPE_TYPE_VALUES = ("BATCH", "PAYMENT", "EXCEPTION")
RECONCILIATION_TRIGGER_VALUES = (
    "SCHEDULED",
    "ON_INGEST",
    "POST_RESOLUTION",
    "MANUAL",
)


class ReconciliationRun(Base):
    """One deterministic reconciliation execution over a scope."""

    __tablename__ = "reconciliation_runs"

    id = Column(String, primary_key=True)

    scope_type = Column(String(16), nullable=False)
    scope_ref = Column(String, nullable=False)

    status = Column(
        String(16), nullable=False, default="PENDING", server_default="PENDING"
    )
    trigger = Column(String(32), nullable=False)

    # Idempotency: caller-supplied run key. Unique per (scope_type, scope_ref,
    # trigger) so a retried trigger cannot start a second run.
    run_key = Column(String, nullable=True)

    # Deterministic counts produced by the run, e.g. {"checked": 120, ...}.
    counts = Column(JSONType, nullable=True)
    matched_count = Column(Integer, nullable=False, default=0, server_default="0")
    exception_count = Column(Integer, nullable=False, default=0, server_default="0")

    started_at = Column(TimestampType, nullable=True)
    finished_at = Column(TimestampType, nullable=True)

    engine_version = Column(String(64), nullable=False)
    created_by = Column(String(128), nullable=True)

    # Optimistic locking (architecture section 6).
    version = Column(Integer, nullable=False, default=1, server_default="1")

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    results = relationship(
        "ReconciliationResult",
        back_populates="run",
        cascade="save-update, merge",
    )

    __table_args__ = (
        enum_check(
            "status", RECONCILIATION_RUN_STATUS_VALUES, "ck_reconciliation_runs_status"
        ),
        enum_check(
            "scope_type",
            RECONCILIATION_SCOPE_TYPE_VALUES,
            "ck_reconciliation_runs_scope_type",
        ),
        enum_check(
            "trigger", RECONCILIATION_TRIGGER_VALUES, "ck_reconciliation_runs_trigger"
        ),
        non_negative_check("matched_count", "ck_reconciliation_runs_matched_count"),
        non_negative_check(
            "exception_count", "ck_reconciliation_runs_exception_count"
        ),
        non_negative_check("version", "ck_reconciliation_runs_version"),
        Index("ix_reconciliation_runs_status_started", "status", "started_at"),
        Index("ix_reconciliation_runs_scope", "scope_type", "scope_ref"),
        Index(
            "uq_reconciliation_runs_run_key",
            "scope_type",
            "scope_ref",
            "trigger",
            "run_key",
            unique=True,
        ),
        # Section 7.2: no other RUNNING run may exist for the same scope.
        Index(
            "uq_reconciliation_runs_one_running_per_scope",
            "scope_type",
            "scope_ref",
            unique=True,
            postgresql_where=text("status = 'RUNNING'"),
            sqlite_where=text("status = 'RUNNING'"),
        ),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.status -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.RECONCILIATION_RUN, self.status, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a status transition; raises when invalid."""
        self.status = state_machines.assert_transition(
            state_machines.RECONCILIATION_RUN, self.status, new_status
        )
        return self.status

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<ReconciliationRun id={self.id!r} scope={self.scope_type}:{self.scope_ref} "
            f"status={self.status!r}>"
        )
