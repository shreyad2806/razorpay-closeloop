"""
Feedback - reviewer/system feedback on an exception or resolution outcome.

NEW table in CloseLoop 2.0 (architecture section 5). The ``FeedbackRecord``
concept already exists in a service but lives in an in-memory dictionary, so no
outcome survives a restart and the self-learning loop has nothing durable to
learn from.

**Immutable.** Rows are append-only (enforced by
:mod:`app.models.immutability`); a correction is a new row whose
``correction_of`` references the original.

No learning pipeline is implemented in Phase 1 (task section 15).
"""

from sqlalchemy import Column, ForeignKey, Index, Numeric, String, Text

from app.database.database import Base
from app.models.immutability import register_append_only
from app.models.types import JSONType, TimestampType, enum_check, utcnow

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
FEEDBACK_TYPE_VALUES = (
    "APPROVAL",
    "REJECTION",
    "CORRECTION",
    "OUTCOME",
    "REVIEW",
)
FEEDBACK_ACTOR_TYPE_VALUES = ("SYSTEM", "HUMAN", "AGENT")


class Feedback(Base):
    """One immutable feedback record."""

    __tablename__ = "feedback"

    id = Column(String, primary_key=True)

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    resolution_id = Column(
        String,
        ForeignKey("resolutions.id", ondelete="RESTRICT"),
        nullable=True,
    )

    feedback_type = Column(String(16), nullable=False)

    reviewer = Column(String(128), nullable=True)
    reviewer_role = Column(String(32), nullable=True)
    actor_type = Column(String(16), nullable=False)

    # What the system predicted, so the delta is measurable.
    system_prediction = Column(String(64), nullable=True)
    system_confidence = Column(Numeric(5, 4), nullable=True)

    correction_details = Column(JSONType, nullable=True)
    evidence_reviewed = Column(JSONType, nullable=True)

    model_version = Column(String(64), nullable=True)
    policy_version = Column(String(32), nullable=True)

    notes = Column(Text, nullable=True)

    # Corrections are new rows, never edits.
    correction_of = Column(
        String,
        ForeignKey("feedback.id", ondelete="RESTRICT"),
        nullable=True,
    )

    occurred_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        enum_check(
            "feedback_type", FEEDBACK_TYPE_VALUES, "ck_feedback_feedback_type"
        ),
        enum_check(
            "actor_type", FEEDBACK_ACTOR_TYPE_VALUES, "ck_feedback_actor_type"
        ),
        Index("ix_feedback_exception", "exception_id"),
        Index("ix_feedback_resolution", "resolution_id"),
        Index("ix_feedback_type_occurred", "feedback_type", "occurred_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Feedback id={self.id!r} type={self.feedback_type!r} "
            f"exception_id={self.exception_id!r}>"
        )


register_append_only(Feedback)
