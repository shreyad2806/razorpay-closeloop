"""
Approval - human/policy approval of a resolution.

NEW in CloseLoop 2.0 (architecture sections 5 and 7.6). Before 2.0, approve /
reject mutated an in-memory dictionary, so an approval left no durable trace and
could not be tied to the evidence the approver actually saw.

``evidence_digest`` is the hash of the evidence presented at approval time; if
the evidence changes afterwards the digest no longer matches and the approval is
invalid, which is exactly the failure the architecture wants to catch.

Only one PENDING approval may exist per resolution (partial unique index,
supported by PostgreSQL and SQLite alike).

Lifecycle (section 7.6): ``PENDING -> APPROVED | REJECTED | EXPIRED | REVOKED``.

No approval workflow/API is implemented in Phase 1.
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)

from app.database.database import Base
from app.domain import state_machines
from app.models.types import TimestampType, enum_check, utcnow

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
APPROVAL_DECISION_VALUES = ("PENDING", "APPROVED", "REJECTED", "EXPIRED", "REVOKED")
APPROVAL_REQUESTER_VALUES = ("SYSTEM", "HUMAN")


class Approval(Base):
    """An approval request attached to a Resolution."""

    __tablename__ = "approvals"

    id = Column(String, primary_key=True)

    resolution_id = Column(
        String,
        ForeignKey("resolutions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    requested_by = Column(String(16), nullable=False)
    required_role = Column(String(32), nullable=True)

    decision = Column(
        String(16), nullable=False, default="PENDING", server_default="PENDING"
    )
    decided_by = Column(String(128), nullable=True)
    decided_at = Column(TimestampType, nullable=True)

    expires_at = Column(TimestampType, nullable=True)

    comments = Column(Text, nullable=True)

    # Hash of the evidence shown to the approver - detects evidence drift.
    evidence_digest = Column(String(128), nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        enum_check("decision", APPROVAL_DECISION_VALUES, "ck_approvals_decision"),
        enum_check(
            "requested_by", APPROVAL_REQUESTER_VALUES, "ck_approvals_requested_by"
        ),
        Index("ix_approvals_decision_expires", "decision", "expires_at"),
        Index("ix_approvals_resolution", "resolution_id"),
        # One open request per resolution.
        Index(
            "uq_approvals_one_pending_per_resolution",
            "resolution_id",
            unique=True,
            postgresql_where=text("decision = 'PENDING'"),
            sqlite_where=text("decision = 'PENDING'"),
        ),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.decision -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.APPROVAL, self.decision, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a decision transition; raises when invalid."""
        self.decision = state_machines.assert_transition(
            state_machines.APPROVAL, self.decision, new_status
        )
        return self.decision

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Approval id={self.id!r} resolution_id={self.resolution_id!r} "
            f"decision={self.decision!r}>"
        )
