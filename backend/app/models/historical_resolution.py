"""
Database model for historical resolutions.

Stores previous human/system resolution outcomes that will later be
used by Phase 4 similarity retrieval.

DOES NOT implement similarity search — that comes later.
This is the storage layer only.
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, Index
from datetime import datetime

from app.database.database import Base


class HistoricalResolution(Base):
    """
    Database model for historical resolutions.

    Stores resolved exception outcomes including:
    - What resolution was applied
    - The outcome of the resolution
    - Financial details
    - Notes/metadata for future similarity matching

    This table will be the basis for Phase 4 retrieval.
    """

    __tablename__ = "historical_resolutions"

    id = Column(String, primary_key=True)  # resolution_id, e.g. HRES-000001

    # Reference to the original exception
    exception_id = Column(String, nullable=True, index=True)  # FK to exceptions
    case_id = Column(String, nullable=False, index=True)  # FK to Case

    # Resolution details
    resolution_type = Column(String, nullable=False)  # ResolutionType enum value
    outcome = Column(String, nullable=False)  # e.g. RESOLVED, ESCALATED, UNRESOLVED

    # Financial details
    resolved_amount = Column(Integer, nullable=True)  # Amount used in resolution (paise)
    difference_at_resolution = Column(Integer, nullable=True)  # Difference when resolved

    # Classification
    exception_type = Column(String, nullable=True)  # What the exception was classified as
    resolvable = Column(Boolean, nullable=True)  # Whether it was determined resolvable

    # Metadata
    notes = Column(String, nullable=True)  # Free-text notes
    resolution_metadata = Column(String, nullable=True)  # JSON string for structured metadata
    source = Column(String, nullable=True, default="deterministic")  # human, deterministic, ml, agent

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_historical_resolutions_exception", "exception_id"),
        Index("ix_historical_resolutions_resolution_type", "resolution_type"),
        Index("ix_historical_resolutions_outcome", "outcome"),
    )
