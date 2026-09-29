"""
Resolution - a proposed and executed resolution for an exception.

NEW in CloseLoop 2.0 (architecture sections 5 and 7.4). Before 2.0 a resolution
was a transient ``ResolutionProposal`` object inside a workflow; nothing was
persisted, so a recommendation could not be reproduced, audited or linked to the
action that carried it out.

Lifecycle (section 7.4)::

    DRAFT -> PROPOSED -> POLICY_EVALUATED -> (AUTO_APPROVED | HUMAN_REVIEW | BLOCKED)
    AUTO_APPROVED / HUMAN_REVIEW+APPROVED -> EXECUTING -> EXECUTED -> VERIFIED | FAILED
    FAILED -> (ROLLBACK_PENDING -> ROLLED_BACK | ROLLBACK_FAILED)
    BLOCKED -> WITHDRAWN

Applied by :meth:`Resolution.transition_to` using
:mod:`app.domain.state_machines`.

No scoring, candidate generation or policy engine is implemented in Phase 1
(task section 15).
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.domain import state_machines
from app.models.types import (
    JSONType,
    MoneyType,
    TimestampType,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
RESOLUTION_STATUS_VALUES = (
    "DRAFT",
    "PROPOSED",
    "POLICY_EVALUATED",
    "AUTO_APPROVED",
    "HUMAN_REVIEW",
    "APPROVED",
    "REJECTED",
    "BLOCKED",
    "EXECUTING",
    "EXECUTED",
    "VERIFIED",
    "FAILED",
    "WITHDRAWN",
    "ROLLBACK_PENDING",
    "ROLLED_BACK",
    "ROLLBACK_FAILED",
)
POLICY_DECISION_VALUES = ("AUTO", "HUMAN_REVIEW", "UNRESOLVED", "BLOCKED")
RESOLUTION_RISK_VALUES = ("LOW", "MEDIUM", "HIGH")
RESOLUTION_DIRECTION_VALUES = ("CREDIT", "DEBIT")


class Resolution(Base):
    """One candidate/proposed resolution for an exception."""

    __tablename__ = "resolutions"

    id = Column(String, primary_key=True)

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # 1 = recommended candidate.
    candidate_rank = Column(Integer, nullable=False, default=1, server_default="1")

    resolution_type = Column(String(64), nullable=False)

    # Money in integer paise. Unsigned: the direction column carries the sign.
    amount_paise = Column(MoneyType, nullable=True)
    direction = Column(String(8), nullable=True)
    financial_exposure_paise = Column(MoneyType, nullable=True)

    calculation_basis = Column(Text, nullable=True)

    # Structured projection of what executing this resolution should achieve.
    expected_effect = Column(JSONType, nullable=True)

    confidence = Column(Numeric(5, 4), nullable=True)
    risk_category = Column(String(16), nullable=True)

    evidence_ids = Column(JSONType, nullable=True)
    historical_support = Column(JSONType, nullable=True)
    ml_support = Column(JSONType, nullable=True)

    status = Column(
        String(24), nullable=False, default="DRAFT", server_default="DRAFT"
    )

    # Written by the policy engine; EXECUTING requires both to be present.
    policy_decision = Column(String(16), nullable=True)
    policy_version = Column(String(32), nullable=True)

    # Optimistic locking (architecture section 6).
    version = Column(Integer, nullable=False, default=1, server_default="1")

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    actions = relationship(
        "ResolutionAction",
        back_populates="resolution",
        cascade="save-update, merge",
    )
    approvals = relationship("Approval", cascade="save-update, merge")

    __table_args__ = (
        non_negative_check("amount_paise", "ck_resolutions_amount_paise"),
        non_negative_check(
            "financial_exposure_paise", "ck_resolutions_financial_exposure"
        ),
        non_negative_check("candidate_rank", "ck_resolutions_candidate_rank"),
        non_negative_check("version", "ck_resolutions_version"),
        enum_check("status", RESOLUTION_STATUS_VALUES, "ck_resolutions_status"),
        enum_check(
            "policy_decision", POLICY_DECISION_VALUES, "ck_resolutions_policy_decision"
        ),
        enum_check("risk_category", RESOLUTION_RISK_VALUES, "ck_resolutions_risk"),
        enum_check("direction", RESOLUTION_DIRECTION_VALUES, "ck_resolutions_direction"),
        UniqueConstraint(
            "exception_id", "candidate_rank", name="uq_resolutions_exception_rank"
        ),
        Index("ix_resolutions_exception_status", "exception_id", "status"),
        Index("ix_resolutions_status", "status"),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.status -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.RESOLUTION, self.status, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a status transition; raises when invalid."""
        self.status = state_machines.assert_transition(
            state_machines.RESOLUTION, self.status, new_status
        )
        return self.status

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Resolution id={self.id!r} exception_id={self.exception_id!r} "
            f"type={self.resolution_type!r} status={self.status!r}>"
        )
