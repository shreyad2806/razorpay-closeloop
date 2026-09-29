"""
AuditEvent - the immutable audit record.

NEW table in CloseLoop 2.0 (architecture section 20). An audit-log *service*
already exists but writes nowhere durable; this is the persistence model it will
target. Phase 1 establishes the correct foundational shape only - the emit
points, the INSERT-only database role and the API endpoint are later phase work.

**Append-only.** No UPDATE, no DELETE - enforced by
:mod:`app.models.immutability`. A correction is a *new* event whose
``correction_of`` points at the event it supersedes, which is how the record
stays honest without ever rewriting history.

No unique constraint is placed on business keys: the same action may legitimately
be recorded more than once (retries, repeated reads).
"""

from sqlalchemy import Column, ForeignKey, Index, Numeric, String, Text

from app.database.database import Base
from app.models.immutability import register_append_only
from app.models.types import JSONType, TimestampType, enum_check, utcnow

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
ACTOR_TYPE_VALUES = ("SYSTEM", "HUMAN", "AGENT")
AUDIT_POLICY_DECISION_VALUES = ("AUTO", "HUMAN_REVIEW", "UNRESOLVED", "BLOCKED")


class AuditEvent(Base):
    """One immutable audit event."""

    __tablename__ = "audit_events"

    id = Column(String, primary_key=True)  # e.g. AUD-000001

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    workflow_id = Column(String, nullable=True)

    # e.g. "agent:resolution-analyst@v3", "user:u-123", "system:policy-engine"
    actor = Column(String(128), nullable=False)
    actor_type = Column(String(16), nullable=False)

    action = Column(String(64), nullable=False)

    occurred_at = Column(TimestampType, nullable=False, default=utcnow)
    correlation_id = Column(String, nullable=True)

    evidence_ids = Column(JSONType, nullable=True)

    policy_decision = Column(String(16), nullable=True)
    policy_version = Column(String(32), nullable=True)

    confidence = Column(Numeric(5, 4), nullable=True)

    before_state = Column(JSONType, nullable=True)
    after_state = Column(JSONType, nullable=True)

    model_version = Column(String(64), nullable=True)
    llm_provider = Column(String(64), nullable=True)

    # Corrections are new events, never edits.
    correction_of = Column(
        String,
        ForeignKey("audit_events.id", ondelete="RESTRICT"),
        nullable=True,
    )

    detail = Column(Text, nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        enum_check("actor_type", ACTOR_TYPE_VALUES, "ck_audit_events_actor_type"),
        enum_check(
            "policy_decision",
            AUDIT_POLICY_DECISION_VALUES,
            "ck_audit_events_policy_decision",
        ),
        Index("ix_audit_events_exception_created", "exception_id", "created_at"),
        Index("ix_audit_events_correlation_id", "correlation_id"),
        Index("ix_audit_events_actor_action", "actor_type", "action"),
        Index("ix_audit_events_occurred_at", "occurred_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<AuditEvent id={self.id!r} action={self.action!r} "
            f"actor={self.actor!r}>"
        )


register_append_only(AuditEvent)
