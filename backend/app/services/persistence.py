"""
Persistence service for reconciliation results and exceptions.

Provides transactional, idempotent persistence of reconciliation output.
"""

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.exception import ExceptionStatus, FinancialException
from app.models.reconciliation import ReconciliationEvidence, ReconciliationResult
from app.schemas.enums import ExceptionType, MatchStatus
from app.schemas.reconciliation import ReconciliationResult as ReconciliationResultSchema


class PersistenceService:
    """
    Service for persisting reconciliation results and exceptions.

    Features:
    - Transactional writes
    - Idempotent operations (duplicate runs don't create duplicates)
    - Error handling with rollback
    """

    def __init__(self, session: Session):
        self.session = session

    def persist_reconciliation_result(
        self,
        result: ReconciliationResultSchema,
        batch_id: str,
    ) -> ReconciliationResult:
        """
        Persist a reconciliation result to the database.

        This is idempotent - running twice with the same case_id/batch_id
        will update rather than duplicate.

        Args:
            result: ReconciliationResult schema
            batch_id: Batch identifier for idempotency

        Returns:
            Persisted ReconciliationResult database model
        """
        # Check if result already exists (idempotency)
        existing = (
            self.session.query(ReconciliationResult)
            .filter_by(case_id=result.case_id, batch_id=batch_id)
            .first()
        )

        if existing:
            # Update existing record
            existing.payment_id = result.payment_id
            existing.merchant_id = result.merchant_id
            existing.payment_amount = result.payment_amount
            existing.total_refunds = result.total_refunds
            existing.total_fees = result.total_fees
            existing.total_taxes = result.total_taxes
            existing.total_adjustments = result.total_adjustments
            existing.expected_amount = result.expected_amount
            existing.actual_amount = result.actual_amount
            existing.difference = result.difference
            existing.match_status = result.match_status.value
            existing.exception_type = result.exception_type.value
            existing.reconciliation_status = result.reconciliation_status.value
            existing.processing_notes = result.processing_notes
            self.session.flush()
            return existing

        # Create new record
        db_result = ReconciliationResult(
            id=result.reconciliation_id,
            case_id=result.case_id,
            payment_id=result.payment_id,
            merchant_id=result.merchant_id,
            batch_id=batch_id,
            payment_amount=result.payment_amount,
            total_refunds=result.total_refunds,
            total_fees=result.total_fees,
            total_taxes=result.total_taxes,
            total_adjustments=result.total_adjustments,
            expected_amount=result.expected_amount,
            actual_amount=result.actual_amount,
            difference=result.difference,
            match_status=result.match_status.value,
            exception_type=result.exception_type.value,
            reconciliation_status=result.reconciliation_status.value,
            reconciliation_timestamp=result.reconciliation_timestamp,
            processing_notes=result.processing_notes,
        )

        self.session.add(db_result)
        self.session.flush()
        return db_result

    def persist_exception(
        self,
        result: ReconciliationResultSchema,
        batch_id: str,
    ) -> Optional[FinancialException]:
        """
        Persist an exception record if the result is an exception.

        Matched cases do NOT get exception records.

        This is idempotent - running twice with the same case_id/batch_id
        will update rather than duplicate.

        Args:
            result: ReconciliationResult schema
            batch_id: Batch identifier for idempotency

        Returns:
            Persisted FinancialException or None if result is matched
        """
        # Don't create exception for matched cases
        if result.match_status == MatchStatus.MATCHED:
            return None

        # Check if exception already exists (idempotency)
        existing = (
            self.session.query(FinancialException)
            .filter_by(case_id=result.case_id, batch_id=batch_id)
            .first()
        )

        if existing:
            # Update existing record
            existing.payment_id = result.payment_id
            existing.expected_amount = result.expected_amount
            existing.actual_amount = result.actual_amount
            existing.difference = result.difference
            existing.exception_type = result.exception_type.value
            existing.status = ExceptionStatus.OPEN
            existing.reconciliation_id = result.reconciliation_id
            self.session.flush()
            return existing

        # Create new record
        exception_id = f"EXC-{result.reconciliation_id.replace('REC-', '')}"
        db_exception = FinancialException(
            id=exception_id,
            case_id=result.case_id,
            payment_id=result.payment_id,
            batch_id=batch_id,
            expected_amount=result.expected_amount,
            actual_amount=result.actual_amount,
            difference=result.difference,
            exception_type=result.exception_type.value,
            status=ExceptionStatus.OPEN,
            reconciliation_id=result.reconciliation_id,
        )

        self.session.add(db_exception)
        self.session.flush()
        return db_exception

    def persist_evidence(
        self,
        reconciliation_id: str,
        evidence_type: str,
        evidence_data: dict,
    ) -> ReconciliationEvidence:
        """
        Persist reconciliation evidence.

        Args:
            reconciliation_id: Reference to reconciliation result
            evidence_type: Type of evidence (e.g., "CALCULATION_BREAKDOWN")
            evidence_data: Evidence data as dictionary

        Returns:
            Persisted ReconciliationEvidence
        """
        evidence_id = f"EV-{reconciliation_id}-{evidence_type}"
        db_evidence = ReconciliationEvidence(
            id=evidence_id,
            reconciliation_id=reconciliation_id,
            evidence_type=evidence_type,
            evidence_data=json.dumps(evidence_data),
        )

        self.session.add(db_evidence)
        self.session.flush()
        return db_evidence

    def persist_batch(
        self,
        results: List[ReconciliationResultSchema],
        batch_id: str,
    ) -> dict:
        """
        Persist a batch of reconciliation results with transaction safety.

        If any persistence fails, the entire batch is rolled back.

        Args:
            results: List of ReconciliationResult schemas
            batch_id: Batch identifier for idempotency

        Returns:
            Dictionary with persistence statistics
        """
        stats = {
            "total": len(results),
            "matched": 0,
            "exceptions": 0,
            "errors": 0,
        }

        try:
            for result in results:
                # Persist reconciliation result
                self.persist_reconciliation_result(result, batch_id)

                # Persist exception if applicable
                exception = self.persist_exception(result, batch_id)

                if result.match_status == MatchStatus.MATCHED:
                    stats["matched"] += 1
                else:
                    stats["exceptions"] += 1

            # Commit all changes
            self.session.commit()

        except Exception as e:
            # Rollback on any error
            self.session.rollback()
            stats["errors"] = stats["total"]
            raise RuntimeError(f"Batch persistence failed: {e}")

        return stats

    def get_reconciliation_results(
        self,
        batch_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> List[ReconciliationResult]:
        """
        Retrieve reconciliation results.

        Args:
            batch_id: Filter by batch ID
            case_id: Filter by case ID

        Returns:
            List of ReconciliationResult database models
        """
        query = self.session.query(ReconciliationResult)

        if batch_id:
            query = query.filter_by(batch_id=batch_id)
        if case_id:
            query = query.filter_by(case_id=case_id)

        return query.all()

    def get_exceptions(
        self,
        batch_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[FinancialException]:
        """
        Retrieve exception records.

        Args:
            batch_id: Filter by batch ID
            status: Filter by status

        Returns:
            List of FinancialException database models
        """
        query = self.session.query(FinancialException)

        if batch_id:
            query = query.filter_by(batch_id=batch_id)
        if status:
            query = query.filter_by(status=status)

        return query.all()
