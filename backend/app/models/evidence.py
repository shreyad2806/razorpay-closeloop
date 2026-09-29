"""
Evidence - the persisted evidence graph for an exception.

NEW table in CloseLoop 2.0 (architecture section 5). Two pre-2.0 tables carried
this concept: ``evidence_links`` (which records an exception points at) and
``reconciliation_evidence`` (free-form JSON blobs keyed by reconciliation id).
The architecture folds both into this single table, where ``payload`` holds the
calculation breakdown and ``relationship`` says *how* the record relates to the
exception.

Those two legacy tables are deliberately left intact in Phase 1: the merge is
Phase 4's acceptance criterion ("persist evidence table; graph renders from
DB"), and deleting tables that current tests and services read would be a
regression (see the Phase 1 checkpoint, conflict C2).

Uniqueness: ``(exception_id, entity_type, entity_id, relationship)`` replaces
today's ad-hoc string id that already encodes exactly this tuple.
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)

from app.database.database import Base
from app.models.types import (
    JSONType,
    TimestampType,
    enum_check,
    utcnow,
)

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
EVIDENCE_RELATIONSHIP_VALUES = (
    "PRIMARY",
    "CALCULATION_COMPONENT",
    "SUPPORTING",
    "CONFLICTING",
    "MISSING",
)
EVIDENCE_RECORDED_BY_VALUES = ("RECONCILIATION", "AGENT", "HUMAN")


class Evidence(Base):
    """One piece of evidence attached to an exception."""

    __tablename__ = "evidence"

    id = Column(String, primary_key=True)

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    # Denormalized for query speed, mirroring the pre-2.0 evidence_links layout.
    case_id = Column(String, nullable=True)

    entity_type = Column(String(48), nullable=False)
    entity_id = Column(String, nullable=False)

    relationship = Column(String(32), nullable=False)

    # Calculation breakdowns, provider snapshots, diffs, ...
    payload = Column(JSONType, nullable=True)

    confidence = Column(Numeric(5, 4), nullable=True)

    recorded_at = Column(TimestampType, nullable=False, default=utcnow)
    recorded_by = Column(String(16), nullable=False)

    notes = Column(Text, nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        enum_check(
            "relationship", EVIDENCE_RELATIONSHIP_VALUES, "ck_evidence_relationship"
        ),
        enum_check(
            "recorded_by", EVIDENCE_RECORDED_BY_VALUES, "ck_evidence_recorded_by"
        ),
        UniqueConstraint(
            "exception_id",
            "entity_type",
            "entity_id",
            "relationship",
            name="uq_evidence_exception_entity_relationship",
        ),
        Index("ix_evidence_exception_entity", "exception_id", "entity_type"),
        Index("ix_evidence_exception_relationship", "exception_id", "relationship"),
        Index("ix_evidence_case_entity", "case_id", "entity_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Evidence id={self.id!r} exception_id={self.exception_id!r} "
            f"{self.entity_type}:{self.entity_id} ({self.relationship})>"
        )
