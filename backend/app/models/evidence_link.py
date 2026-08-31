"""
Database model for evidence links.

Connects exceptions to the financial records that serve as evidence
for the reconciliation decision.

Supports relationships:
  Exception → Payment
  Exception → Settlement
  Exception → Refund
  Exception → Fee
  Exception → Tax
  Exception → Adjustment
"""

from sqlalchemy import Column, String, DateTime, Index
from datetime import datetime

from app.database.database import Base


class EvidenceLink(Base):
    """
    Database model for evidence relationships.

    Each row represents a single evidence connection between an exception
    and a financial record that was used in the reconciliation decision.

    This provides deterministic traceability for audit and debugging.
    Does NOT use NetworkX — that comes in a later phase.
    """

    __tablename__ = "evidence_links"

    id = Column(String, primary_key=True)  # evidence_link_id, e.g. EL-000001

    # Which exception this evidence supports
    exception_id = Column(String, nullable=False, index=True)  # FK to exceptions
    case_id = Column(String, nullable=False, index=True)  # Denormalized for query speed

    # The financial record serving as evidence
    entity_type = Column(String, nullable=False)  # PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT
    entity_id = Column(String, nullable=False, index=True)  # The record's primary key

    # Why this record is evidence
    relationship = Column(String, nullable=False)  # e.g. CALCULATION_COMPONENT, SUPPORTING_EVIDENCE

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("ix_evidence_links_exception_entity", "exception_id", "entity_type"),
        Index("ix_evidence_links_case_entity", "case_id", "entity_type"),
    )
