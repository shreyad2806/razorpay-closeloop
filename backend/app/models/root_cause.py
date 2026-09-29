"""
RootCause - a ranked root-cause hypothesis for an exception.

NEW in CloseLoop 2.0 (architecture section 5). An exception may have several
candidate causes; the top-ranked candidate is what feeds resolution. Each row
records *why* it was proposed (a deterministic rule id, an ML prediction row, or
an LLM explanation) so the ranking is always explainable.

No ML inference or Layer-3 search is implemented in Phase 1 (task section 15);
this is the persistence model only.
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

from app.database.database import Base
from app.models.types import (
    JSONType,
    TimestampType,
    non_negative_check,
    utcnow,
)


class RootCause(Base):
    """One candidate explanation for an exception."""

    __tablename__ = "root_causes"

    id = Column(String, primary_key=True)

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # 1 = top-ranked candidate.
    rank = Column(Integer, nullable=False, default=1, server_default="1")

    # Reuses the ExceptionType taxonomy plus root-cause-specific labels.
    cause_type = Column(String(48), nullable=False)

    confidence = Column(Numeric(5, 4), nullable=True)

    # Deterministic provenance: the rule id that produced this candidate.
    deterministic_basis = Column(String(128), nullable=True)

    ml_prediction_id = Column(
        String,
        ForeignKey("model_predictions.id", ondelete="RESTRICT"),
        nullable=True,
    )

    llm_explanation = Column(Text, nullable=True)

    # Ids of the Evidence rows that support this candidate.
    evidence_ids = Column(JSONType, nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        non_negative_check("rank", "ck_root_causes_rank"),
        UniqueConstraint("exception_id", "rank", name="uq_root_causes_exception_rank"),
        Index("ix_root_causes_exception", "exception_id"),
        Index("ix_root_causes_exception_cause", "exception_id", "cause_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<RootCause id={self.id!r} exception_id={self.exception_id!r} "
            f"rank={self.rank!r} cause_type={self.cause_type!r}>"
        )
