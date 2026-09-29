"""
ResolutionAction - the executable unit of a resolution.

NEW in CloseLoop 2.0 (architecture sections 5 and 7.5). This is the single row
the execution service guards, so the two properties that matter are:

* ``idempotency_key`` is UNIQUE - a retried action cannot execute twice. This
  replaces the in-memory ``ConcurrencyGuard`` with a database guarantee while
  keeping its semantics (a duplicate returns the original result).
* ``provider_reference`` is unique when present, so provider-side retries that
  produce a second reference are detected.

Lifecycle (section 7.5)::

    PROPOSED -> VALIDATED -> APPROVED -> EXECUTING -> EXECUTED -> VERIFIED
                                       -> FAILED -> ROLLBACK_PENDING
                                          -> ROLLED_BACK | ROLLBACK_FAILED

No provider execution is implemented in Phase 1 (task section 15).
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.domain import state_machines
from app.models.types import (
    MoneyType,
    TimestampType,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
RESOLUTION_ACTION_STATUS_VALUES = (
    "PROPOSED",
    "VALIDATED",
    "APPROVED",
    "EXECUTING",
    "EXECUTED",
    "VERIFIED",
    "FAILED",
    "ROLLBACK_PENDING",
    "ROLLED_BACK",
    "ROLLBACK_FAILED",
)
RESOLUTION_ACTION_TYPE_VALUES = ("ADJUSTMENT", "REVERSAL")


class ResolutionAction(Base):
    """One executable action belonging to a Resolution."""

    __tablename__ = "resolution_actions"

    id = Column(String, primary_key=True)

    resolution_id = Column(
        String,
        ForeignKey("resolutions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    action_type = Column(String(16), nullable=False)
    provider_operation = Column(String(64), nullable=True)

    amount_paise = Column(MoneyType, nullable=True)  # paise, unsigned

    # Database-backed idempotency: the only guard the execution service needs.
    idempotency_key = Column(String(128), nullable=False)

    status = Column(
        String(24), nullable=False, default="PROPOSED", server_default="PROPOSED"
    )

    provider_reference = Column(String(128), nullable=True)

    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    last_error = Column(Text, nullable=True)

    executed_at = Column(TimestampType, nullable=True)
    verified_at = Column(TimestampType, nullable=True)

    rollback_action_id = Column(
        String,
        ForeignKey("resolution_actions.id", ondelete="RESTRICT"),
        nullable=True,
    )

    version = Column(Integer, nullable=False, default=1, server_default="1")

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    resolution = relationship("Resolution", back_populates="actions")
    adjustments = relationship("Adjustment", back_populates="resolution_action")

    __table_args__ = (
        non_negative_check("amount_paise", "ck_resolution_actions_amount_paise"),
        non_negative_check("attempts", "ck_resolution_actions_attempts"),
        non_negative_check("version", "ck_resolution_actions_version"),
        enum_check(
            "status", RESOLUTION_ACTION_STATUS_VALUES, "ck_resolution_actions_status"
        ),
        enum_check(
            "action_type",
            RESOLUTION_ACTION_TYPE_VALUES,
            "ck_resolution_actions_action_type",
        ),
        # A retried action must not execute twice.
        Index(
            "uq_resolution_actions_idempotency_key",
            "idempotency_key",
            unique=True,
        ),
        # Provider-side retries that produce a second reference are detectable.
        Index(
            "uq_resolution_actions_provider_reference",
            "provider_reference",
            unique=True,
            postgresql_where=text("provider_reference IS NOT NULL"),
            sqlite_where=text("provider_reference IS NOT NULL"),
        ),
        Index("ix_resolution_actions_resolution_status", "resolution_id", "status"),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.status -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.RESOLUTION_ACTION, self.status, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a status transition; raises when invalid."""
        self.status = state_machines.assert_transition(
            state_machines.RESOLUTION_ACTION, self.status, new_status
        )
        return self.status

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<ResolutionAction id={self.id!r} resolution_id={self.resolution_id!r} "
            f"type={self.action_type!r} status={self.status!r}>"
        )
