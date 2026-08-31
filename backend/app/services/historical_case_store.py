"""
Historical case storage service for Razorpay CloseLoop Phase 4E.

Stores resolved financial cases in a structured form for later retrieval
as historical evidence for similar exceptions.

Supports:
- Creation and persistence of historical cases
- Retrieval by case_id, exception_type, resolution_type
- Duplicate detection
- Financial integrity validation
- Batch operations

DOES NOT implement similarity search — that comes in a later phase.
This is the structured storage and retrieval layer only.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import Column, String, Integer, DateTime, Boolean, Float, Index, Text
from sqlalchemy.orm import Session

from app.database.database import Base
from app.schemas.historical_case import (
    HistoricalCase,
    HistoricalEvidenceRef,
    FinancialContext,
    ResolutionOutcome,
    ResolutionOrigin,
)


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy Model
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalCaseRecord(Base):
    """Database model for historical cases.

    Stores the full structured representation of a resolved financial case
    for later retrieval and learning.
    """

    __tablename__ = "historical_cases"

    id = Column(String, primary_key=True)  # case_id

    # References
    exception_id = Column(String, nullable=False)
    payment_id = Column(String, nullable=False)
    merchant_id = Column(String, nullable=True)

    # Exception classification
    exception_type = Column(String, nullable=False)

    # Financial context (stored as individual columns for queryability)
    payment_amount = Column(Integer, nullable=False)
    expected_amount = Column(Integer, nullable=False)
    actual_amount = Column(Integer, nullable=False)
    difference = Column(Integer, nullable=False)
    total_refunds = Column(Integer, default=0)
    total_fees = Column(Integer, default=0)
    total_taxes = Column(Integer, default=0)
    total_adjustments = Column(Integer, default=0)

    # Resolution
    resolution_type = Column(String, nullable=False)
    resolution_outcome = Column(String, nullable=False)
    resolution_origin = Column(String, nullable=False, default="DETERMINISTIC")
    resolved_amount = Column(Integer, nullable=True)
    resolution_notes = Column(Text, nullable=True)

    # Evidence (stored as JSON)
    evidence_refs_json = Column(Text, default="[]")
    supporting_evidence_count = Column(Integer, default=0)

    # Quality
    exception_type_confidence = Column(Float, nullable=True)
    evidence_coverage = Column(Float, nullable=True)

    # Metadata
    tags_json = Column(Text, default="[]")
    resolution_metadata_json = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_historical_cases_exception_type", "exception_type"),
        Index("ix_historical_cases_resolution_type", "resolution_type"),
        Index("ix_historical_cases_resolution_outcome", "resolution_outcome"),
        Index("ix_historical_cases_created_at", "created_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Store Service
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalCaseStore:
    """Service for storing and retrieving historical cases.

    Provides structured persistence for resolved financial cases,
    enabling later similarity retrieval and learning.
    """

    def __init__(self, session: Session):
        self.session = session

    def store(self, case: HistoricalCase) -> HistoricalCaseRecord:
        """Store a historical case.

        Args:
            case: HistoricalCase to store

        Returns:
            HistoricalCaseRecord if stored
            None if duplicate

        Raises:
            ValueError: If financial integrity validation fails
        """
        # Validate financial integrity
        errors = case.validate_financial_integrity()
        if errors:
            raise ValueError(f"Financial integrity errors: {errors}")

        # Check for duplicate
        existing = (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.id == case.case_id)
            .first()
        )
        if existing:
            return None  # Duplicate — do not overwrite

        # Convert evidence refs to JSON
        evidence_refs = [ref.model_dump() for ref in case.evidence_refs]
        tags = case.tags

        record = HistoricalCaseRecord(
            id=case.case_id,
            exception_id=case.exception_id,
            payment_id=case.payment_id,
            merchant_id=case.merchant_id,
            exception_type=case.exception_type,
            payment_amount=case.financial_context.payment_amount,
            expected_amount=case.financial_context.expected_amount,
            actual_amount=case.financial_context.actual_amount,
            difference=case.financial_context.difference,
            total_refunds=case.financial_context.total_refunds,
            total_fees=case.financial_context.total_fees,
            total_taxes=case.financial_context.total_taxes,
            total_adjustments=case.financial_context.total_adjustments,
            resolution_type=case.resolution_type,
            resolution_outcome=case.resolution_outcome.value,
            resolution_origin=case.resolution_origin.value,
            resolved_amount=case.resolved_amount,
            resolution_notes=case.resolution_notes,
            evidence_refs_json=json.dumps(evidence_refs),
            supporting_evidence_count=case.supporting_evidence_count,
            exception_type_confidence=case.exception_type_confidence,
            evidence_coverage=case.evidence_coverage,
            tags_json=json.dumps(tags),
            resolution_metadata_json=(
                json.dumps(case.resolution_metadata)
                if case.resolution_metadata
                else None
            ),
            created_at=case.created_at,
            resolved_at=case.resolved_at,
        )

        self.session.add(record)
        self.session.flush()
        return record

    def get_by_case_id(self, case_id: str) -> Optional[HistoricalCase]:
        """Retrieve a historical case by case_id."""
        record = (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.id == case_id)
            .first()
        )
        if not record:
            return None
        return self._record_to_case(record)

    def get_by_exception_type(
        self, exception_type: str, limit: int = 100
    ) -> List[HistoricalCase]:
        """Retrieve historical cases by exception type."""
        records = (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.exception_type == exception_type)
            .order_by(HistoricalCaseRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._record_to_case(r) for r in records]

    def get_by_resolution_type(
        self, resolution_type: str, limit: int = 100
    ) -> List[HistoricalCase]:
        """Retrieve historical cases by resolution type."""
        records = (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.resolution_type == resolution_type)
            .order_by(HistoricalCaseRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._record_to_case(r) for r in records]

    def get_by_outcome(
        self, outcome: ResolutionOutcome, limit: int = 100
    ) -> List[HistoricalCase]:
        """Retrieve historical cases by resolution outcome."""
        records = (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.resolution_outcome == outcome.value)
            .order_by(HistoricalCaseRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._record_to_case(r) for r in records]

    def list_all(self, limit: int = 1000) -> List[HistoricalCase]:
        """List all historical cases."""
        records = (
            self.session.query(HistoricalCaseRecord)
            .order_by(HistoricalCaseRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._record_to_case(r) for r in records]

    def count(self) -> int:
        """Count total historical cases."""
        return self.session.query(HistoricalCaseRecord).count()

    def exists(self, case_id: str) -> bool:
        """Check if a historical case exists."""
        return (
            self.session.query(HistoricalCaseRecord)
            .filter(HistoricalCaseRecord.id == case_id)
            .first()
            is not None
        )

    def _record_to_case(self, record: HistoricalCaseRecord) -> HistoricalCase:
        """Convert a database record back to a HistoricalCase schema."""
        evidence_refs = [
            HistoricalEvidenceRef(**ref)
            for ref in json.loads(record.evidence_refs_json or "[]")
        ]
        tags = json.loads(record.tags_json or "[]")
        metadata = (
            json.loads(record.resolution_metadata_json)
            if record.resolution_metadata_json
            else None
        )

        return HistoricalCase(
            case_id=record.id,
            exception_id=record.exception_id,
            payment_id=record.payment_id,
            merchant_id=record.merchant_id,
            exception_type=record.exception_type,
            financial_context=FinancialContext(
                payment_amount=record.payment_amount,
                expected_amount=record.expected_amount,
                actual_amount=record.actual_amount,
                difference=record.difference,
                total_refunds=record.total_refunds,
                total_fees=record.total_fees,
                total_taxes=record.total_taxes,
                total_adjustments=record.total_adjustments,
            ),
            resolution_type=record.resolution_type,
            resolution_outcome=ResolutionOutcome(record.resolution_outcome),
            resolution_origin=ResolutionOrigin(record.resolution_origin),
            resolved_amount=record.resolved_amount,
            resolution_notes=record.resolution_notes,
            evidence_refs=evidence_refs,
            supporting_evidence_count=record.supporting_evidence_count,
            exception_type_confidence=record.exception_type_confidence,
            evidence_coverage=record.evidence_coverage,
            tags=tags,
            resolution_metadata=metadata,
            created_at=record.created_at,
            resolved_at=record.resolved_at,
        )
