"""
Reconciliation pipeline integration tests.

Tests the complete deterministic pipeline:

  Financial records → calculate_reconciliation → ReconciliationResult → PersistenceService → DB

Each scenario creates controlled synthetic data, runs the full pipeline,
and verifies the database state matches expected ground truth.

All tests use isolated SQLite databases — no developer data affected.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Set env before any app imports
os.environ.setdefault("DATABASE_URL", "sqlite:///test_recon_pipeline.db")

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.database import Base
from app.models.exception import ExceptionStatus, FinancialException
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.reconciliation.engine import reconcile_batch
from app.schemas.enums import (
    ExceptionType,
    MatchStatus,
    ReconciliationStatus,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
    Tax,
)
from app.schemas.reconciliation import ReconciliationResult
from app.services.persistence import PersistenceService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """Create an isolated SQLite in-memory session for each test."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def persistence_service(db_session):
    """Create PersistenceService for the test session."""
    return PersistenceService(db_session)


def now():
    return datetime(2026, 6, 15, 12, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Clean Records (Exact Match)
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanRecords:
    """Payment = settlement. No refunds, fees, taxes, or adjustments."""

    def test_exact_match_pipeline(self, db_session, persistence_service):
        """Full pipeline: clean payment → exact match → no exception in DB."""
        payments = [
            Payment(
                payment_id="PAY-CLEAN-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-CLEAN-001",
                payment_id="PAY-CLEAN-001",
                merchant_id="MER-01",
                amount=100000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-CLEAN-001": "CASE-CLEAN-001"}

        # Step 1: Run reconciliation
        results = reconcile_batch(
            payments=payments,
            settlements=settlements,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            case_mapping=case_mapping,
        )

        assert len(results) == 1
        r = results[0]
        assert r.match_status == MatchStatus.MATCHED
        assert r.exception_type == ExceptionType.EXACT_MATCH
        assert r.expected_amount == 100000
        assert r.actual_amount == 100000
        assert r.difference == 0

        # Step 2: Persist to DB
        stats = persistence_service.persist_batch(results, "BATCH-CLEAN-001")
        assert stats["matched"] == 1
        assert stats["exceptions"] == 0

        # Step 3: Verify DB state
        db_results = db_session.query(DBReconciliationResult).filter_by(
            batch_id="BATCH-CLEAN-001"
        ).all()
        assert len(db_results) == 1
        assert db_results[0].match_status == MatchStatus.MATCHED.value
        assert db_results[0].exception_type == ExceptionType.EXACT_MATCH.value
        assert db_results[0].expected_amount == 100000
        assert db_results[0].actual_amount == 100000
        assert db_results[0].difference == 0

        # No exceptions for matched records
        exceptions = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-CLEAN-001"
        ).all()
        assert len(exceptions) == 0

    def test_multiple_clean_payments(self, db_session, persistence_service):
        """Multiple clean payments all match exactly."""
        payments = [
            Payment(
                payment_id=f"PAY-MC-{i:03d}",
                merchant_id="MER-01",
                amount=50000 * (i + 1),
                payment_timestamp=now(),
            )
            for i in range(4)
        ]
        settlements = [
            Settlement(
                settlement_id=f"SET-MC-{i:03d}",
                payment_id=f"PAY-MC-{i:03d}",
                merchant_id="MER-01",
                amount=50000 * (i + 1),
                settlement_timestamp=now(),
            )
            for i in range(4)
        ]
        case_mapping = {f"PAY-MC-{i:03d}": f"CASE-MC-{i:03d}" for i in range(4)}

        results = reconcile_batch(
            payments=payments,
            settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        assert len(results) == 4
        for r in results:
            assert r.match_status == MatchStatus.MATCHED
            assert r.difference == 0

        stats = persistence_service.persist_batch(results, "BATCH-MC-001")
        assert stats["matched"] == 4
        assert stats["exceptions"] == 0

    def test_clean_with_deductions_that_match(self, db_session, persistence_service):
        """Payment with refunds, fees, taxes — and settlement equals expected."""
        # expected = 100000 - 5000 (refund) - 2000 (fee) - 1000 (tax) + 0 = 92000
        payments = [
            Payment(
                payment_id="PAY-DED-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-DED-001",
                payment_id="PAY-DED-001",
                merchant_id="MER-01",
                amount=92000,
                settlement_timestamp=now(),
            ),
        ]
        refunds = [
            Refund(
                refund_id="REF-DED-001",
                payment_id="PAY-DED-001",
                amount=5000,
                refund_timestamp=now(),
            ),
        ]
        fees = [
            Fee(
                fee_id="FEE-DED-001",
                payment_id="PAY-DED-001",
                amount=2000,
                fee_type="PLATFORM",
            ),
        ]
        taxes = [
            Tax(
                tax_id="TAX-DED-001",
                payment_id="PAY-DED-001",
                amount=1000,
                tax_type="GST",
            ),
        ]
        case_mapping = {"PAY-DED-001": "CASE-DED-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=fees, taxes=taxes, adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 92000
        assert r.actual_amount == 92000
        assert r.difference == 0
        assert r.match_status == MatchStatus.MATCHED
        assert r.total_refunds == 5000
        assert r.total_fees == 2000
        assert r.total_taxes == 1000

        stats = persistence_service.persist_batch(results, "BATCH-DED-001")
        assert stats["matched"] == 1
        assert stats["exceptions"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Fee Exception
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeException:
    """Settlement differs from expected by a small fee-proportional amount."""

    def test_fee_difference_pipeline(self, db_session, persistence_service):
        """Fee error: expected 98000, actual 97500, diff=500 (5% of fees=10000)."""
        payments = [
            Payment(
                payment_id="PAY-FEE-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-FEE-001",
                payment_id="PAY-FEE-001",
                merchant_id="MER-01",
                amount=97500,
                settlement_timestamp=now(),
            ),
        ]
        fees = [
            Fee(
                fee_id="FEE-FEE-001",
                payment_id="PAY-FEE-001",
                amount=10000,
                fee_type="PLATFORM",
            ),
        ]
        case_mapping = {"PAY-FEE-001": "CASE-FEE-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=fees, taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 90000  # 100000 - 10000
        assert r.actual_amount == 97500
        assert r.difference == -7500
        assert r.match_status == MatchStatus.EXCEPTION

        # Persist and verify
        stats = persistence_service.persist_batch(results, "BATCH-FEE-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-FEE-001"
        ).first()
        assert db_exc is not None
        assert db_exc.status == ExceptionStatus.OPEN
        assert db_exc.expected_amount == 90000
        assert db_exc.actual_amount == 97500


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Refund Exception
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundException:
    """Settlement does not account for a refund."""

    def test_refund_adjustment_pipeline(self, db_session, persistence_service):
        """Refund of 5000 not reflected: expected 95000, actual 100000."""
        payments = [
            Payment(
                payment_id="PAY-REF-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-REF-001",
                payment_id="PAY-REF-001",
                merchant_id="MER-01",
                amount=100000,
                settlement_timestamp=now(),
            ),
        ]
        refunds = [
            Refund(
                refund_id="REF-REF-001",
                payment_id="PAY-REF-001",
                amount=5000,
                refund_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-REF-001": "CASE-REF-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 95000  # 100000 - 5000
        assert r.actual_amount == 100000
        assert r.difference == -5000
        assert r.match_status == MatchStatus.EXCEPTION

        stats = persistence_service.persist_batch(results, "BATCH-REF-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-REF-001"
        ).first()
        assert db_exc is not None
        assert db_exc.expected_amount == 95000
        assert db_exc.actual_amount == 100000
        assert db_exc.difference == -5000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Tax Exception
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxException:
    """Settlement has a small tax-proportional error."""

    def test_tax_adjustment_pipeline(self, db_session, persistence_service):
        """Tax error: expected=90000, actual=89000, diff=1000 (5% of taxes=20000)."""
        payments = [
            Payment(
                payment_id="PAY-TAX-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-TAX-001",
                payment_id="PAY-TAX-001",
                merchant_id="MER-01",
                amount=89000,
                settlement_timestamp=now(),
            ),
        ]
        taxes = [
            Tax(
                tax_id="TAX-TAX-001",
                payment_id="PAY-TAX-001",
                amount=20000,
                tax_type="GST",
            ),
        ]
        case_mapping = {"PAY-TAX-001": "CASE-TAX-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=taxes, adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 80000  # 100000 - 20000
        assert r.actual_amount == 89000
        assert r.difference == -9000
        assert r.match_status == MatchStatus.EXCEPTION

        stats = persistence_service.persist_batch(results, "BATCH-TAX-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-TAX-001"
        ).first()
        assert db_exc is not None
        assert db_exc.expected_amount == 80000
        assert db_exc.actual_amount == 89000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Partial Settlement
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialSettlement:
    """Settlement is only a fraction of the expected amount."""

    def test_partial_settlement_pipeline(self, db_session, persistence_service):
        """Partial: expected 100000, actual 50000, 50% settled."""
        payments = [
            Payment(
                payment_id="PAY-PS-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-PS-001",
                payment_id="PAY-PS-001",
                merchant_id="MER-01",
                amount=50000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-PS-001": "CASE-PS-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 100000
        assert r.actual_amount == 50000
        assert r.difference == 50000
        assert r.match_status == MatchStatus.EXCEPTION
        assert r.exception_type == ExceptionType.PARTIAL_SETTLEMENT

        stats = persistence_service.persist_batch(results, "BATCH-PS-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-PS-001"
        ).first()
        assert db_exc is not None
        assert db_exc.difference == 50000
        assert db_exc.exception_type == ExceptionType.PARTIAL_SETTLEMENT.value

    def test_partial_60_percent(self, db_session, persistence_service):
        """60% settled should still be PARTIAL_SETTLEMENT."""
        payments = [
            Payment(
                payment_id="PAY-PS60-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-PS60-001",
                payment_id="PAY-PS60-001",
                merchant_id="MER-01",
                amount=60000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-PS60-001": "CASE-PS60-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        assert results[0].exception_type == ExceptionType.PARTIAL_SETTLEMENT
        assert results[0].difference == 40000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Duplicate Settlement
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateSettlement:
    """Multiple identical settlement records for the same payment."""

    def test_duplicate_settlement_pipeline(self, db_session, persistence_service):
        """Two identical settlements → DUPLICATE."""
        payments = [
            Payment(
                payment_id="PAY-DUP-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-DUP-001",
                payment_id="PAY-DUP-001",
                merchant_id="MER-01",
                amount=100000,
                settlement_timestamp=now(),
            ),
            Settlement(
                settlement_id="SET-DUP-002",
                payment_id="PAY-DUP-001",
                merchant_id="MER-01",
                amount=100000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-DUP-001": "CASE-DUP-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.match_status == MatchStatus.DUPLICATE
        assert r.exception_type == ExceptionType.DUPLICATE
        assert r.actual_amount == 200000  # Sum of both settlements

        stats = persistence_service.persist_batch(results, "BATCH-DUP-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-DUP-001"
        ).first()
        assert db_exc is not None
        assert db_exc.exception_type == ExceptionType.DUPLICATE.value

    def test_different_amounts_not_duplicate(self, db_session, persistence_service):
        """Two different-amount settlements are NOT duplicate."""
        payments = [
            Payment(
                payment_id="PAY-NDUP-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-NDUP-001",
                payment_id="PAY-NDUP-001",
                merchant_id="MER-01",
                amount=50000,
                settlement_timestamp=now(),
            ),
            Settlement(
                settlement_id="SET-NDUP-002",
                payment_id="PAY-NDUP-001",
                merchant_id="MER-01",
                amount=30000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-NDUP-001": "CASE-NDUP-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.match_status != MatchStatus.DUPLICATE
        assert r.actual_amount == 80000  # 50000 + 30000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: Missing Record
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingRecord:
    """No settlement record exists for a payment."""

    def test_missing_settlement_pipeline(self, db_session, persistence_service):
        """Payment exists, no settlement → MISSING."""
        payments = [
            Payment(
                payment_id="PAY-MISS-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-MISS-001": "CASE-MISS-001"}

        results = reconcile_batch(
            payments=payments, settlements=[],
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.match_status == MatchStatus.MISSING
        assert r.exception_type == ExceptionType.MISSING_RECORD
        assert r.expected_amount == 100000
        assert r.actual_amount == 0
        assert r.difference == 100000

        stats = persistence_service.persist_batch(results, "BATCH-MISS-001")
        assert stats["exceptions"] == 1

        db_exc = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-MISS-001"
        ).first()
        assert db_exc is not None
        assert db_exc.exception_type == ExceptionType.MISSING_RECORD.value
        assert db_exc.actual_amount == 0

    def test_missing_with_refunds(self, db_session, persistence_service):
        """No settlement but refunds exist → MISSING with refunds counted."""
        payments = [
            Payment(
                payment_id="PAY-MISS2-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        refunds = [
            Refund(
                refund_id="REF-MISS2-001",
                payment_id="PAY-MISS2-001",
                amount=5000,
                refund_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-MISS2-001": "CASE-MISS2-001"}

        results = reconcile_batch(
            payments=payments, settlements=[],
            refunds=refunds, fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.match_status == MatchStatus.MISSING
        assert r.expected_amount == 95000  # 100000 - 5000
        assert r.actual_amount == 0
        assert r.total_refunds == 5000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8: Complex Multi-Adjustment
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexMultiAdjustment:
    """Multiple financial components contribute to discrepancy."""

    def test_complex_multi_adjustment_pipeline(self, db_session, persistence_service):
        """
        Payment=100000, refund=5000, fee=3000, tax=2000, adj=+2000
        Expected = 100000 - 5000 - 3000 - 2000 + 2000 = 92000
        Actual = 90000
        Diff = 2000
        """
        payments = [
            Payment(
                payment_id="PAY-CMA-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-CMA-001",
                payment_id="PAY-CMA-001",
                merchant_id="MER-01",
                amount=90000,
                settlement_timestamp=now(),
            ),
        ]
        refunds = [
            Refund(
                refund_id="REF-CMA-001",
                payment_id="PAY-CMA-001",
                amount=5000,
                refund_timestamp=now(),
            ),
        ]
        fees = [
            Fee(
                fee_id="FEE-CMA-001",
                payment_id="PAY-CMA-001",
                amount=3000,
                fee_type="PLATFORM",
            ),
        ]
        taxes = [
            Tax(
                tax_id="TAX-CMA-001",
                payment_id="PAY-CMA-001",
                amount=2000,
                tax_type="GST",
            ),
        ]
        adjustments = [
            Adjustment(
                adjustment_id="ADJ-CMA-001",
                payment_id="PAY-CMA-001",
                amount=2000,
                adjustment_type="CREDIT",
            ),
        ]
        case_mapping = {"PAY-CMA-001": "CASE-CMA-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=fees, taxes=taxes, adjustments=adjustments,
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 92000
        assert r.actual_amount == 90000
        assert r.difference == 2000
        assert r.total_refunds == 5000
        assert r.total_fees == 3000
        assert r.total_taxes == 2000
        assert r.total_adjustments == 2000
        assert r.match_status == MatchStatus.EXCEPTION

        stats = persistence_service.persist_batch(results, "BATCH-CMA-001")
        assert stats["exceptions"] == 1

        db_results = db_session.query(DBReconciliationResult).filter_by(
            batch_id="BATCH-CMA-001"
        ).all()
        assert len(db_results) == 1
        assert db_results[0].expected_amount == 92000
        assert db_results[0].total_refunds == 5000
        assert db_results[0].total_fees == 3000
        assert db_results[0].total_taxes == 2000
        assert db_results[0].total_adjustments == 2000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9: Unknown
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknown:
    """Small unexplained difference with no clear financial cause."""

    def test_unknown_exception_pipeline(self, db_session, persistence_service):
        """
        Tiny difference (99 paise) with no refunds/fees/taxes/adjustments.
        No clear explanation → UNKNOWN or TIMING_DIFFERENCE.
        """
        payments = [
            Payment(
                payment_id="PAY-UNK-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-UNK-001",
                payment_id="PAY-UNK-001",
                merchant_id="MER-01",
                amount=99901,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-UNK-001": "CASE-UNK-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 100000
        assert r.actual_amount == 99901
        assert r.difference == 99
        assert r.match_status == MatchStatus.EXCEPTION
        # 99 paise diff with no components → UNKNOWN or TIMING_DIFFERENCE
        assert r.exception_type in (
            ExceptionType.UNKNOWN,
            ExceptionType.TIMING_DIFFERENCE,
        )

    def test_large_unexplained_difference(self, db_session, persistence_service):
        """Large unexplained difference (>50K paise) with no components.
        30K/100K = 30% actual ratio → PARTIAL_SETTLEMENT (20-85% range)."""
        payments = [
            Payment(
                payment_id="PAY-UNK2-001",
                merchant_id="MER-01",
                amount=100000,
                payment_timestamp=now(),
            ),
        ]
        settlements = [
            Settlement(
                settlement_id="SET-UNK2-001",
                payment_id="PAY-UNK2-001",
                merchant_id="MER-01",
                amount=30000,
                settlement_timestamp=now(),
            ),
        ]
        case_mapping = {"PAY-UNK2-001": "CASE-UNK2-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert r.expected_amount == 100000
        assert r.actual_amount == 30000
        assert r.difference == 70000
        assert r.match_status == MatchStatus.EXCEPTION
        # 30% actual ratio falls in PARTIAL_SETTLEMENT range (20-85%)
        assert r.exception_type == ExceptionType.PARTIAL_SETTLEMENT


# ─────────────────────────────────────────────────────────────────────────────
# 10. Financial Totals Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialTotals:
    """Verify that financial totals stored in DB match calculations."""

    def test_batch_totals_match_calculations(self, db_session, persistence_service):
        """
        Batch of 5 payments with different scenarios.
        Verify each DB record has correct financial breakdown.
        """
        # Payment 1: Clean
        # Payment 2: With refund
        # Payment 3: With fee
        # Payment 4: With tax
        # Payment 5: With adjustment
        payments = [
            Payment(payment_id="PAY-TOT-001", merchant_id="MER-01", amount=50000, payment_timestamp=now()),
            Payment(payment_id="PAY-TOT-002", merchant_id="MER-01", amount=80000, payment_timestamp=now()),
            Payment(payment_id="PAY-TOT-003", merchant_id="MER-01", amount=120000, payment_timestamp=now()),
            Payment(payment_id="PAY-TOT-004", merchant_id="MER-01", amount=200000, payment_timestamp=now()),
            Payment(payment_id="PAY-TOT-005", merchant_id="MER-01", amount=150000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-TOT-001", payment_id="PAY-TOT-001", merchant_id="MER-01", amount=50000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-TOT-002", payment_id="PAY-TOT-002", merchant_id="MER-01", amount=75000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-TOT-003", payment_id="PAY-TOT-003", merchant_id="MER-01", amount=118000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-TOT-004", payment_id="PAY-TOT-004", merchant_id="MER-01", amount=199000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-TOT-005", payment_id="PAY-TOT-005", merchant_id="MER-01", amount=151000, settlement_timestamp=now()),
        ]
        refunds = [
            Refund(refund_id="REF-TOT-002", payment_id="PAY-TOT-002", amount=5000, refund_timestamp=now()),
        ]
        fees = [
            Fee(fee_id="FEE-TOT-003", payment_id="PAY-TOT-003", amount=2000, fee_type="PLATFORM"),
        ]
        taxes = [
            Tax(tax_id="TAX-TOT-004", payment_id="PAY-TOT-004", amount=1000, tax_type="GST"),
        ]
        adjustments = [
            Adjustment(adjustment_id="ADJ-TOT-005", payment_id="PAY-TOT-005", amount=1000, adjustment_type="CREDIT"),
        ]
        case_mapping = {f"PAY-TOT-{i:03d}": f"CASE-TOT-{i:03d}" for i in range(1, 6)}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=fees, taxes=taxes, adjustments=adjustments,
            case_mapping=case_mapping,
        )

        stats = persistence_service.persist_batch(results, "BATCH-TOT-001")
        assert stats["total"] == 5
        assert stats["errors"] == 0

        db_results = db_session.query(DBReconciliationResult).filter_by(
            batch_id="BATCH-TOT-001"
        ).all()
        assert len(db_results) == 5

        # Verify each record's financial breakdown
        by_payment = {r.payment_id: r for r in db_results}

        # Payment 1: clean
        r1 = by_payment["PAY-TOT-001"]
        assert r1.payment_amount == 50000
        assert r1.total_refunds == 0
        assert r1.total_fees == 0
        assert r1.total_taxes == 0
        assert r1.total_adjustments == 0
        assert r1.expected_amount == 50000
        assert r1.actual_amount == 50000
        assert r1.difference == 0

        # Payment 2: refund
        r2 = by_payment["PAY-TOT-002"]
        assert r2.total_refunds == 5000
        assert r2.expected_amount == 75000  # 80000 - 5000
        assert r2.actual_amount == 75000
        assert r2.difference == 0

        # Payment 3: fee
        r3 = by_payment["PAY-TOT-003"]
        assert r3.total_fees == 2000
        assert r3.expected_amount == 118000  # 120000 - 2000
        assert r3.actual_amount == 118000
        assert r3.difference == 0

        # Payment 4: tax
        r4 = by_payment["PAY-TOT-004"]
        assert r4.total_taxes == 1000
        assert r4.expected_amount == 199000  # 200000 - 1000
        assert r4.actual_amount == 199000
        assert r4.difference == 0

        # Payment 5: adjustment
        r5 = by_payment["PAY-TOT-005"]
        assert r5.total_adjustments == 1000
        assert r5.expected_amount == 151000  # 150000 + 1000
        assert r5.actual_amount == 151000
        assert r5.difference == 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Record Counts Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordCounts:
    """Verify correct record counts in the database after pipeline."""

    def test_mixed_batch_record_counts(self, db_session, persistence_service):
        """
        5 payments: 2 clean, 2 exception, 1 missing.
        Verify DB has exactly 5 reconciliation results and 3 exceptions.
        """
        payments = [
            Payment(payment_id="PAY-CNT-001", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
            Payment(payment_id="PAY-CNT-002", merchant_id="MER-01", amount=200000, payment_timestamp=now()),
            Payment(payment_id="PAY-CNT-003", merchant_id="MER-01", amount=150000, payment_timestamp=now()),
            Payment(payment_id="PAY-CNT-004", merchant_id="MER-01", amount=80000, payment_timestamp=now()),
            Payment(payment_id="PAY-CNT-005", merchant_id="MER-01", amount=60000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-CNT-001", payment_id="PAY-CNT-001", merchant_id="MER-01", amount=100000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-CNT-002", payment_id="PAY-CNT-002", merchant_id="MER-01", amount=195000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-CNT-003", payment_id="PAY-CNT-003", merchant_id="MER-01", amount=150000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-CNT-004", payment_id="PAY-CNT-004", merchant_id="MER-01", amount=78000, settlement_timestamp=now()),
            # No settlement for PAY-CNT-005 → MISSING
        ]
        case_mapping = {f"PAY-CNT-{i:03d}": f"CASE-CNT-{i:03d}" for i in range(1, 6)}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        # Count by match status
        matched = sum(1 for r in results if r.match_status == MatchStatus.MATCHED)
        exceptions = sum(1 for r in results if r.match_status == MatchStatus.EXCEPTION)
        missing = sum(1 for r in results if r.match_status == MatchStatus.MISSING)
        assert matched == 2  # CNT-001, CNT-003
        assert exceptions == 2  # CNT-002, CNT-004
        assert missing == 1  # CNT-005

        stats = persistence_service.persist_batch(results, "BATCH-CNT-001")
        assert stats["total"] == 5
        assert stats["matched"] == 2
        assert stats["exceptions"] == 3  # exceptions + missing both create exception records

        # DB: 5 reconciliation results
        db_results = db_session.query(DBReconciliationResult).filter_by(
            batch_id="BATCH-CNT-001"
        ).count()
        assert db_results == 5

        # DB: 3 exception records (2 EXCEPTION + 1 MISSING)
        db_exceptions = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-CNT-001"
        ).count()
        assert db_exceptions == 3


# ─────────────────────────────────────────────────────────────────────────────
# 12. Determinism / Idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Same inputs must always produce the same outputs."""

    def test_same_input_same_output(self, db_session):
        """Running reconcile_batch twice with same data gives identical results."""
        payments = [
            Payment(payment_id="PAY-DET-001", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-DET-001", payment_id="PAY-DET-001", merchant_id="MER-01", amount=98000, settlement_timestamp=now()),
        ]
        refunds = [
            Refund(refund_id="REF-DET-001", payment_id="PAY-DET-001", amount=2000, refund_timestamp=now()),
        ]
        case_mapping = {"PAY-DET-001": "CASE-DET-001"}

        results1 = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )
        results2 = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.reconciliation_id == r2.reconciliation_id
            assert r1.expected_amount == r2.expected_amount
            assert r1.actual_amount == r2.actual_amount
            assert r1.difference == r2.difference
            assert r1.match_status == r2.match_status
            assert r1.exception_type == r2.exception_type
            assert r1.total_refunds == r2.total_refunds
            assert r1.total_fees == r2.total_fees
            assert r1.total_taxes == r2.total_taxes
            assert r1.total_adjustments == r2.total_adjustments

    def test_persist_batch_idempotent(self, db_session, persistence_service):
        """Persisting same batch twice doesn't create duplicate records."""
        payments = [
            Payment(payment_id="PAY-IDEM-001", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-IDEM-001", payment_id="PAY-IDEM-001", merchant_id="MER-01", amount=100000, settlement_timestamp=now()),
        ]
        case_mapping = {"PAY-IDEM-001": "CASE-IDEM-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        # First persist
        stats1 = persistence_service.persist_batch(results, "BATCH-IDEM-001")
        assert stats1["total"] == 1

        # Second persist (same batch_id)
        stats2 = persistence_service.persist_batch(results, "BATCH-IDEM-001")
        assert stats2["total"] == 1

        # Should still be exactly 1 record in DB
        count = db_session.query(DBReconciliationResult).filter_by(
            batch_id="BATCH-IDEM-001"
        ).count()
        assert count == 1

    def test_deterministic_exception_types(self, db_session):
        """Same financial pattern always produces the same exception type."""
        scenarios = [
            # (payment_amount, settlement_amount, refunds, fees, taxes, adjustments)
            (100000, 95000, 5000, 0, 0, 0),   # refund
            (100000, 97500, 0, 10000, 0, 0),   # fee
            (100000, 90000, 0, 0, 10000, 0),   # tax
            (100000, 50000, 0, 0, 0, 0),        # partial
            (100000, 100000, 0, 0, 0, 0),       # clean
            (100000, 0, 0, 0, 0, 0),            # missing
        ]
        case_mapping = {f"PAY-DET-{i}": f"CASE-DET-{i}" for i in range(len(scenarios))}

        # Run 3 times
        all_runs = []
        for _ in range(3):
            payments = [
                Payment(payment_id=f"PAY-DET-{i}", merchant_id="MER-01", amount=s[0], payment_timestamp=now())
                for i, s in enumerate(scenarios)
            ]
            settlements = [
                Settlement(settlement_id=f"SET-DET-{i}", payment_id=f"PAY-DET-{i}", merchant_id="MER-01", amount=s[1], settlement_timestamp=now())
                for i, s in enumerate(scenarios)
                if s[1] > 0
            ]
            refunds = [
                Refund(refund_id=f"REF-DET-{i}", payment_id=f"PAY-DET-{i}", amount=s[2], refund_timestamp=now())
                for i, s in enumerate(scenarios)
                if s[2] > 0
            ]
            fees = [
                Fee(fee_id=f"FEE-DET-{i}", payment_id=f"PAY-DET-{i}", amount=s[3], fee_type="PLATFORM")
                for i, s in enumerate(scenarios)
                if s[3] > 0
            ]
            taxes = [
                Tax(tax_id=f"TAX-DET-{i}", payment_id=f"PAY-DET-{i}", amount=s[4], tax_type="GST")
                for i, s in enumerate(scenarios)
                if s[4] > 0
            ]
            results = reconcile_batch(
                payments=payments, settlements=settlements,
                refunds=refunds, fees=fees, taxes=taxes, adjustments=[],
                case_mapping=case_mapping,
            )
            all_runs.append([(r.match_status, r.exception_type) for r in results])

        # All 3 runs must produce identical classification
        assert all_runs[0] == all_runs[1] == all_runs[2]

    def test_integer_paise_throughout(self, db_session):
        """All financial values stored in DB are integers."""
        payments = [
            Payment(payment_id="PAY-INT-001", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-INT-001", payment_id="PAY-INT-001", merchant_id="MER-01", amount=99999, settlement_timestamp=now()),
        ]
        case_mapping = {"PAY-INT-001": "CASE-INT-001"}

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        r = results[0]
        assert isinstance(r.payment_amount, int)
        assert isinstance(r.expected_amount, int)
        assert isinstance(r.actual_amount, int)
        assert isinstance(r.difference, int)
        assert isinstance(r.total_refunds, int)
        assert isinstance(r.total_fees, int)
        assert isinstance(r.total_taxes, int)
        assert isinstance(r.total_adjustments, int)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Payment Isolation
# ─────────────────────────────────────────────────────────────────────────────


class TestPaymentIsolation:
    """Verify that financial records for different payments don't mix."""

    def test_refund_does_not_affect_other_payment(self, db_session):
        """Refund for PAY-A must not be counted in PAY-B's reconciliation."""
        payments = [
            Payment(payment_id="PAY-ISO-A", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
            Payment(payment_id="PAY-ISO-B", merchant_id="MER-01", amount=200000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-ISO-A", payment_id="PAY-ISO-A", merchant_id="MER-01", amount=95000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-ISO-B", payment_id="PAY-ISO-B", merchant_id="MER-01", amount=200000, settlement_timestamp=now()),
        ]
        refunds = [
            Refund(refund_id="REF-ISO-A", payment_id="PAY-ISO-A", amount=5000, refund_timestamp=now()),
        ]
        case_mapping = {
            "PAY-ISO-A": "CASE-ISO-A",
            "PAY-ISO-B": "CASE-ISO-B",
        }

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=refunds, fees=[], taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        by_payment = {r.payment_id: r for r in results}

        # PAY-A: has refund, expected=95000, actual=95000 → MATCHED
        ra = by_payment["PAY-ISO-A"]
        assert ra.total_refunds == 5000
        assert ra.expected_amount == 95000
        assert ra.actual_amount == 95000
        assert ra.match_status == MatchStatus.MATCHED

        # PAY-B: no refund, expected=200000, actual=200000 → MATCHED
        rb = by_payment["PAY-ISO-B"]
        assert rb.total_refunds == 0
        assert rb.expected_amount == 200000
        assert rb.actual_amount == 200000
        assert rb.match_status == MatchStatus.MATCHED

    def test_fee_does_not_affect_other_payment(self, db_session):
        """Fee for PAY-A must not appear in PAY-B's totals."""
        payments = [
            Payment(payment_id="PAY-FISO-A", merchant_id="MER-01", amount=100000, payment_timestamp=now()),
            Payment(payment_id="PAY-FISO-B", merchant_id="MER-01", amount=80000, payment_timestamp=now()),
        ]
        settlements = [
            Settlement(settlement_id="SET-FISO-A", payment_id="PAY-FISO-A", merchant_id="MER-01", amount=99000, settlement_timestamp=now()),
            Settlement(settlement_id="SET-FISO-B", payment_id="PAY-FISO-B", merchant_id="MER-01", amount=80000, settlement_timestamp=now()),
        ]
        fees = [
            Fee(fee_id="FEE-FISO-A", payment_id="PAY-FISO-A", amount=1000, fee_type="PLATFORM"),
        ]
        case_mapping = {
            "PAY-FISO-A": "CASE-FISO-A",
            "PAY-FISO-B": "CASE-FISO-B",
        }

        results = reconcile_batch(
            payments=payments, settlements=settlements,
            refunds=[], fees=fees, taxes=[], adjustments=[],
            case_mapping=case_mapping,
        )

        by_payment = {r.payment_id: r for r in results}

        assert by_payment["PAY-FISO-A"].total_fees == 1000
        assert by_payment["PAY-FISO-B"].total_fees == 0
        assert by_payment["PAY-FISO-B"].match_status == MatchStatus.MATCHED
