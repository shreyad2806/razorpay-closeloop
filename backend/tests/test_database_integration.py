"""
Comprehensive database integration tests for Razorpay CloseLoop.

Tests all database operations using isolated SQLite in-memory databases:
- Connection and table creation
- Model CRUD (create, read, update, delete)
- Relationships and foreign key filtering
- Idempotency of persistence operations
- Transaction rollback on failure
- Historical case retrieval
- Batch persistence

Each test gets its own isolated database session.
Tests do NOT modify developer data — all in-memory.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# Set env before importing database module — must happen before any app imports
os.environ.setdefault("DATABASE_URL", "sqlite:///test_db_integ.db")

import pytest
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.payment import Payment
from app.models.settlement import Settlement
from app.models.refund import Refund
from app.models.fee import Fee
from app.models.tax import Tax
from app.models.adjustment import Adjustment
from app.models.exception import ExceptionStatus, FinancialException
from app.models.reconciliation import ReconciliationEvidence, ReconciliationResult
from app.models.evidence_link import EvidenceLink
from app.models.historical_resolution import HistoricalResolution
from app.schemas.enums import (
    ExceptionType,
    MatchStatus,
    ReconciliationStatus,
)
from app.schemas.reconciliation import ReconciliationResult as ReconciliationResultSchema
from app.services.persistence import PersistenceService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_payment(db_session):
    """Create a sample payment record."""
    payment = Payment(
        id="PAY-TEST-001",
        merchant_id="MER-TEST-01",
        amount=100000.0,
    )
    db_session.add(payment)
    db_session.commit()
    return payment


@pytest.fixture
def sample_settlement(db_session, sample_payment):
    """Create a sample settlement linked to the payment."""
    settlement = Settlement(
        id="SET-TEST-001",
        payment_id=sample_payment.id,
        amount=98000.0,
    )
    db_session.add(settlement)
    db_session.commit()
    return settlement


@pytest.fixture
def sample_refund(db_session, sample_payment):
    """Create a sample refund."""
    refund = Refund(
        id="REF-TEST-001",
        payment_id=sample_payment.id,
        case_id="CASE-TEST-001",
        merchant_id="MER-TEST-01",
        amount=500,
        status="PROCESSED",
    )
    db_session.add(refund)
    db_session.commit()
    return refund


@pytest.fixture
def sample_fee(db_session, sample_payment):
    """Create a sample fee."""
    fee = Fee(
        id="FEE-TEST-001",
        payment_id=sample_payment.id,
        case_id="CASE-TEST-001",
        merchant_id="MER-TEST-01",
        amount=200,
        fee_type="PLATFORM_FEE",
    )
    db_session.add(fee)
    db_session.commit()
    return fee


@pytest.fixture
def sample_tax(db_session, sample_payment):
    """Create a sample tax."""
    tax = Tax(
        id="TAX-TEST-001",
        payment_id=sample_payment.id,
        case_id="CASE-TEST-001",
        merchant_id="MER-TEST-01",
        amount=1800,
        tax_type="GST",
    )
    db_session.add(tax)
    db_session.commit()
    return tax


@pytest.fixture
def sample_adjustment(db_session, sample_payment):
    """Create a sample adjustment (credit)."""
    adj = Adjustment(
        id="ADJ-TEST-001",
        payment_id=sample_payment.id,
        case_id="CASE-TEST-001",
        merchant_id="MER-TEST-01",
        amount=1000,
        adjustment_type="CREDIT",
    )
    db_session.add(adj)
    db_session.commit()
    return adj


@pytest.fixture
def sample_reconciliation_result(db_session):
    """Create a sample reconciliation result."""
    result = ReconciliationResult(
        id="REC-TEST-001",
        case_id="CASE-TEST-001",
        payment_id="PAY-TEST-001",
        merchant_id="MER-TEST-01",
        batch_id="BATCH-TEST-001",
        payment_amount=100000,
        total_refunds=500,
        total_fees=200,
        total_taxes=1800,
        total_adjustments=0,
        expected_amount=97500,
        actual_amount=97000,
        difference=500,
        match_status=MatchStatus.EXCEPTION.value,
        exception_type=ExceptionType.FEE_DIFFERENCE.value,
        reconciliation_status=ReconciliationStatus.PROCESSED.value,
    )
    db_session.add(result)
    db_session.commit()
    return result


@pytest.fixture
def sample_exception(db_session, sample_reconciliation_result):
    """Create a sample financial exception."""
    exc = FinancialException(
        id="EXC-TEST-001",
        case_id="CASE-TEST-001",
        payment_id="PAY-TEST-001",
        batch_id="BATCH-TEST-001",
        expected_amount=97500,
        actual_amount=97000,
        difference=500,
        exception_type=ExceptionType.FEE_DIFFERENCE.value,
        status=ExceptionStatus.OPEN,
        reconciliation_id="REC-TEST-001",
    )
    db_session.add(exc)
    db_session.commit()
    return exc


@pytest.fixture
def sample_evidence(db_session, sample_reconciliation_result):
    """Create sample reconciliation evidence."""
    evidence = ReconciliationEvidence(
        id="EV-TEST-001",
        reconciliation_id="REC-TEST-001",
        evidence_type="CALCULATION_BREAKDOWN",
        evidence_data=json.dumps({
            "expected": 97500,
            "actual": 97000,
            "difference": 500,
        }),
    )
    db_session.add(evidence)
    db_session.commit()
    return evidence


@pytest.fixture
def sample_evidence_link(db_session, sample_exception):
    """Create a sample evidence link."""
    link = EvidenceLink(
        id="EL-TEST-001",
        exception_id="EXC-TEST-001",
        case_id="CASE-TEST-001",
        entity_type="PAYMENT",
        entity_id="PAY-TEST-001",
        relationship="CALCULATION_COMPONENT",
    )
    db_session.add(link)
    db_session.commit()
    return link


@pytest.fixture
def sample_historical_resolution(db_session):
    """Create a sample historical resolution."""
    hr = HistoricalResolution(
        id="HRES-TEST-001",
        exception_id="EXC-TEST-001",
        case_id="CASE-TEST-001",
        resolution_type="FEE_REVERSAL",
        outcome="RESOLVED",
        resolved_amount=500,
        difference_at_resolution=500,
        exception_type=ExceptionType.FEE_DIFFERENCE.value,
        resolvable=True,
        notes="Fee difference corrected by reversing excess platform fee",
        source="deterministic",
    )
    db_session.add(hr)
    db_session.commit()
    return hr


@pytest.fixture
def persistence_service(db_session):
    """Create a PersistenceService with the test session."""
    return PersistenceService(db_session)


@pytest.fixture
def reconciliation_schema_result():
    """Create a ReconciliationResult Pydantic schema."""
    return ReconciliationResultSchema(
        reconciliation_id="REC-SCHEMA-001",
        case_id="CASE-SCHEMA-001",
        payment_id="PAY-SCHEMA-001",
        merchant_id="MER-SCHEMA-01",
        payment_amount=100000,
        total_refunds=500,
        total_fees=200,
        total_taxes=1800,
        total_adjustments=0,
        expected_amount=97500,
        actual_amount=97000,
        difference=500,
        match_status=MatchStatus.EXCEPTION,
        exception_type=ExceptionType.FEE_DIFFERENCE,
        reconciliation_status=ReconciliationStatus.PROCESSED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Database Connection and Table Creation
# ─────────────────────────────────────────────────────────────────────────────


class TestDatabaseConnection:
    """Test database connection and table creation."""

    def test_tables_created(self, db_engine):
        """All core financial tables should be created."""
        from app.database.database import Base

        import app.models.payment
        import app.models.settlement
        import app.models.refund
        import app.models.fee
        import app.models.tax
        import app.models.adjustment
        import app.models.exception
        import app.models.reconciliation
        import app.models.evidence_link
        import app.models.historical_resolution

        Base.metadata.create_all(bind=db_engine)
        tables = set(Base.metadata.tables.keys())
        required = {
            "payments", "settlements", "refunds", "fees", "taxes",
            "adjustments", "exceptions", "reconciliation_results",
            "reconciliation_evidence", "evidence_links", "historical_resolutions",
        }
        assert required.issubset(tables), f"Missing tables: {required - tables}"

    def test_foreign_keys_enabled(self, db_engine):
        """SQLite foreign keys should be enabled."""
        with db_engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text("PRAGMA foreign_keys")
            ).fetchone()
            assert result[0] == 1

    def test_session_creation(self, db_session):
        """A session should be usable immediately."""
        result = db_session.execute(
            __import__("sqlalchemy").text("SELECT 1")
        ).fetchone()
        assert result[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Creation — All 11 Models
# ─────────────────────────────────────────────────────────────────────────────


class TestModelCreation:
    """Test creating instances of every model."""

    def test_payment_creation(self, sample_payment):
        """Payment model should be persisted with correct fields."""
        assert sample_payment.id == "PAY-TEST-001"
        assert sample_payment.merchant_id == "MER-TEST-01"
        assert sample_payment.amount == 100000.0

    def test_settlement_creation(self, sample_settlement):
        """Settlement model should be persisted with correct fields."""
        assert sample_settlement.id == "SET-TEST-001"
        assert sample_settlement.payment_id == "PAY-TEST-001"
        assert sample_settlement.amount == 98000.0

    def test_refund_creation(self, sample_refund):
        """Refund model should be persisted with correct fields."""
        assert sample_refund.id == "REF-TEST-001"
        assert sample_refund.payment_id == "PAY-TEST-001"
        assert sample_refund.amount == 500
        assert sample_refund.status == "PROCESSED"

    def test_fee_creation(self, sample_fee):
        """Fee model should be persisted with correct fields."""
        assert sample_fee.id == "FEE-TEST-001"
        assert sample_fee.amount == 200
        assert sample_fee.fee_type == "PLATFORM_FEE"

    def test_tax_creation(self, sample_tax):
        """Tax model should be persisted with correct fields."""
        assert sample_tax.id == "TAX-TEST-001"
        assert sample_tax.amount == 1800
        assert sample_tax.tax_type == "GST"

    def test_adjustment_creation(self, sample_adjustment):
        """Adjustment model should be persisted with correct fields."""
        assert sample_adjustment.id == "ADJ-TEST-001"
        assert sample_adjustment.amount == 1000
        assert sample_adjustment.adjustment_type == "CREDIT"

    def test_exception_creation(self, sample_exception):
        """Exception model should be persisted with correct fields."""
        assert sample_exception.id == "EXC-TEST-001"
        assert sample_exception.case_id == "CASE-TEST-001"
        assert sample_exception.expected_amount == 97500
        assert sample_exception.actual_amount == 97000
        assert sample_exception.difference == 500
        assert sample_exception.exception_type == ExceptionType.FEE_DIFFERENCE.value
        assert sample_exception.status == ExceptionStatus.OPEN

    def test_reconciliation_result_creation(self, sample_reconciliation_result):
        """ReconciliationResult model should be persisted."""
        assert sample_reconciliation_result.id == "REC-TEST-001"
        assert sample_reconciliation_result.payment_amount == 100000
        assert sample_reconciliation_result.difference == 500

    def test_evidence_creation(self, sample_evidence):
        """ReconciliationEvidence model should be persisted."""
        assert sample_evidence.id == "EV-TEST-001"
        data = json.loads(sample_evidence.evidence_data)
        assert data["expected"] == 97500
        assert data["actual"] == 97000

    def test_evidence_link_creation(self, sample_evidence_link):
        """EvidenceLink model should be persisted."""
        assert sample_evidence_link.id == "EL-TEST-001"
        assert sample_evidence_link.entity_type == "PAYMENT"
        assert sample_evidence_link.relationship == "CALCULATION_COMPONENT"

    def test_historical_resolution_creation(self, sample_historical_resolution):
        """HistoricalResolution model should be persisted."""
        assert sample_historical_resolution.id == "HRES-TEST-001"
        assert sample_historical_resolution.resolution_type == "FEE_REVERSAL"
        assert sample_historical_resolution.outcome == "RESOLVED"
        assert sample_historical_resolution.resolved_amount == 500


# ─────────────────────────────────────────────────────────────────────────────
# 3. Read Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestReadOperations:
    """Test reading records back from the database."""

    def test_read_payment(self, db_session, sample_payment):
        """Should retrieve payment by primary key."""
        result = db_session.query(Payment).filter_by(id="PAY-TEST-001").first()
        assert result is not None
        assert result.amount == 100000.0

    def test_read_settlement_by_payment_id(self, db_session, sample_settlement):
        """Should retrieve settlement by payment_id (foreign key filter)."""
        results = db_session.query(Settlement).filter_by(
            payment_id="PAY-TEST-001"
        ).all()
        assert len(results) == 1
        assert results[0].amount == 98000.0

    def test_read_refund_by_payment_id(self, db_session, sample_refund):
        """Should retrieve refund by payment_id."""
        results = db_session.query(Refund).filter_by(payment_id="PAY-TEST-001").all()
        assert len(results) == 1
        assert results[0].amount == 500

    def test_read_fee_by_payment_id(self, db_session, sample_fee):
        """Should retrieve fee by payment_id."""
        results = db_session.query(Fee).filter_by(payment_id="PAY-TEST-001").all()
        assert len(results) == 1
        assert results[0].fee_type == "PLATFORM_FEE"

    def test_read_tax_by_payment_id(self, db_session, sample_tax):
        """Should retrieve tax by payment_id."""
        results = db_session.query(Tax).filter_by(payment_id="PAY-TEST-001").all()
        assert len(results) == 1
        assert results[0].tax_type == "GST"

    def test_read_adjustment_by_payment_id(self, db_session, sample_adjustment):
        """Should retrieve adjustment by payment_id."""
        results = db_session.query(Adjustment).filter_by(
            payment_id="PAY-TEST-001"
        ).all()
        assert len(results) == 1
        assert results[0].adjustment_type == "CREDIT"

    def test_read_exception_by_status(self, db_session, sample_exception):
        """Should retrieve exception by status filter."""
        results = db_session.query(FinancialException).filter_by(
            status=ExceptionStatus.OPEN
        ).all()
        assert len(results) == 1
        assert results[0].exception_type == ExceptionType.FEE_DIFFERENCE.value

    def test_read_exception_by_batch_id(self, db_session, sample_exception):
        """Should retrieve exception by batch_id."""
        results = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-TEST-001"
        ).all()
        assert len(results) == 1

    def test_read_exception_by_case_id(self, db_session, sample_exception):
        """Should retrieve exception by case_id."""
        result = db_session.query(FinancialException).filter_by(
            case_id="CASE-TEST-001"
        ).first()
        assert result is not None
        assert result.id == "EXC-TEST-001"

    def test_read_reconciliation_result_by_batch(self, db_session, sample_reconciliation_result):
        """Should retrieve reconciliation result by batch_id."""
        results = db_session.query(ReconciliationResult).filter_by(
            batch_id="BATCH-TEST-001"
        ).all()
        assert len(results) == 1

    def test_read_evidence_by_reconciliation_id(self, db_session, sample_evidence):
        """Should retrieve evidence by reconciliation_id."""
        results = db_session.query(ReconciliationEvidence).filter_by(
            reconciliation_id="REC-TEST-001"
        ).all()
        assert len(results) == 1

    def test_read_evidence_link_by_exception_id(self, db_session, sample_evidence_link):
        """Should retrieve evidence link by exception_id."""
        results = db_session.query(EvidenceLink).filter_by(
            exception_id="EXC-TEST-001"
        ).all()
        assert len(results) == 1
        assert results[0].entity_type == "PAYMENT"

    def test_read_evidence_link_by_entity_type(self, db_session, sample_evidence_link):
        """Should retrieve evidence links filtered by entity_type."""
        results = db_session.query(EvidenceLink).filter_by(
            entity_type="PAYMENT"
        ).all()
        assert len(results) == 1

    def test_read_historical_resolution_by_case(self, db_session, sample_historical_resolution):
        """Should retrieve historical resolution by case_id."""
        results = db_session.query(HistoricalResolution).filter_by(
            case_id="CASE-TEST-001"
        ).all()
        assert len(results) == 1
        assert results[0].outcome == "RESOLVED"

    def test_read_historical_resolution_by_exception_type(self, db_session, sample_historical_resolution):
        """Should retrieve historical resolution by exception_type."""
        results = db_session.query(HistoricalResolution).filter_by(
            exception_type=ExceptionType.FEE_DIFFERENCE.value
        ).all()
        assert len(results) == 1

    def test_read_nonexistent_returns_none(self, db_session):
        """Querying for nonexistent record should return None."""
        result = db_session.query(Payment).filter_by(id="PAY-NONEXISTENT").first()
        assert result is None

    def test_read_empty_table_returns_empty(self, db_session):
        """Querying an empty table should return empty list."""
        results = db_session.query(Payment).all()
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Update Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateOperations:
    """Test updating records in the database."""

    def test_update_exception_status(self, db_session, sample_exception):
        """Should update exception status from OPEN to RESOLVED."""
        exc = db_session.query(FinancialException).filter_by(
            id="EXC-TEST-001"
        ).first()
        exc.status = ExceptionStatus.RESOLVED
        db_session.commit()

        updated = db_session.query(FinancialException).filter_by(
            id="EXC-TEST-001"
        ).first()
        assert updated.status == ExceptionStatus.RESOLVED

    def test_update_payment_amount(self, db_session, sample_payment):
        """Should update payment amount."""
        payment = db_session.query(Payment).filter_by(id="PAY-TEST-001").first()
        payment.amount = 200000.0
        db_session.commit()

        updated = db_session.query(Payment).filter_by(id="PAY-TEST-001").first()
        assert updated.amount == 200000.0

    def test_update_historical_resolution_notes(self, db_session, sample_historical_resolution):
        """Should update historical resolution notes."""
        hr = db_session.query(HistoricalResolution).filter_by(
            id="HRES-TEST-001"
        ).first()
        hr.notes = "Updated resolution notes"
        db_session.commit()

        updated = db_session.query(HistoricalResolution).filter_by(
            id="HRES-TEST-001"
        ).first()
        assert updated.notes == "Updated resolution notes"

    def test_update_multiple_exception_fields(self, db_session, sample_exception):
        """Should update multiple fields at once."""
        exc = db_session.query(FinancialException).filter_by(
            id="EXC-TEST-001"
        ).first()
        exc.status = ExceptionStatus.RESOLVED
        exc.difference = 0
        exc.actual_amount = exc.expected_amount
        db_session.commit()

        updated = db_session.query(FinancialException).filter_by(
            id="EXC-TEST-001"
        ).first()
        assert updated.status == ExceptionStatus.RESOLVED
        assert updated.difference == 0
        assert updated.actual_amount == updated.expected_amount

    def test_update_settlement_amount(self, db_session, sample_settlement):
        """Should update settlement amount."""
        s = db_session.query(Settlement).filter_by(id="SET-TEST-001").first()
        s.amount = 99500.0
        db_session.commit()

        updated = db_session.query(Settlement).filter_by(id="SET-TEST-001").first()
        assert updated.amount == 99500.0

    def test_update_refund_status(self, db_session, sample_refund):
        """Should update refund status."""
        r = db_session.query(Refund).filter_by(id="REF-TEST-001").first()
        r.status = "REFUNDED"
        db_session.commit()

        updated = db_session.query(Refund).filter_by(id="REF-TEST-001").first()
        assert updated.status == "REFUNDED"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Delete Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteOperations:
    """Test deleting records from the database."""

    def test_delete_payment(self, db_session, sample_payment):
        """Should delete payment by primary key."""
        db_session.delete(sample_payment)
        db_session.commit()
        result = db_session.query(Payment).filter_by(id="PAY-TEST-001").first()
        assert result is None

    def test_delete_exception(self, db_session, sample_exception):
        """Should delete exception record."""
        db_session.delete(sample_exception)
        db_session.commit()
        result = db_session.query(FinancialException).filter_by(id="EXC-TEST-001").first()
        assert result is None

    def test_delete_historical_resolution(self, db_session, sample_historical_resolution):
        """Should delete historical resolution."""
        db_session.delete(sample_historical_resolution)
        db_session.commit()
        result = db_session.query(HistoricalResolution).filter_by(id="HRES-TEST-001").first()
        assert result is None

    def test_delete_does_not_affect_other_records(self, db_session, sample_exception):
        """Deleting one exception should not affect others."""
        other_exc = FinancialException(
            id="EXC-TEST-002",
            case_id="CASE-TEST-002",
            payment_id="PAY-TEST-002",
            batch_id="BATCH-TEST-001",
            expected_amount=50000,
            actual_amount=49000,
            difference=1000,
            exception_type=ExceptionType.TIMING_DIFFERENCE.value,
            status=ExceptionStatus.OPEN,
            reconciliation_id="REC-TEST-002",
        )
        db_session.add(other_exc)
        db_session.commit()

        db_session.delete(sample_exception)
        db_session.commit()

        result = db_session.query(FinancialException).filter_by(id="EXC-TEST-002").first()
        assert result is not None
        assert result.id == "EXC-TEST-002"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Relationships and Foreign Key Filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestRelationships:
    """Test that related records can be retrieved via foreign key filters."""

    def test_multiple_refunds_per_payment(self, db_session, sample_payment):
        """Payment should support multiple refunds."""
        r1 = Refund(
            id="REF-TEST-010",
            payment_id=sample_payment.id,
            amount=100,
            status="PROCESSED",
        )
        r2 = Refund(
            id="REF-TEST-011",
            payment_id=sample_payment.id,
            amount=200,
            status="PROCESSED",
        )
        db_session.add_all([r1, r2])
        db_session.commit()

        results = db_session.query(Refund).filter_by(
            payment_id=sample_payment.id
        ).all()
        assert len(results) == 2
        amounts = sorted([r.amount for r in results])
        assert amounts == [100, 200]

    def test_multiple_fees_per_payment(self, db_session, sample_payment):
        """Payment should support multiple fees."""
        f1 = Fee(
            id="FEE-TEST-010",
            payment_id=sample_payment.id,
            amount=100,
            fee_type="PLATFORM_FEE",
        )
        f2 = Fee(
            id="FEE-TEST-011",
            payment_id=sample_payment.id,
            amount=50,
            fee_type="GATEWAY_FEE",
        )
        db_session.add_all([f1, f2])
        db_session.commit()

        results = db_session.query(Fee).filter_by(
            payment_id=sample_payment.id
        ).all()
        assert len(results) == 2

    def test_multiple_taxes_per_payment(self, db_session, sample_payment):
        """Payment should support multiple taxes."""
        t1 = Tax(
            id="TAX-TEST-010",
            payment_id=sample_payment.id,
            amount=100,
            tax_type="GST",
        )
        t2 = Tax(
            id="TAX-TEST-011",
            payment_id=sample_payment.id,
            amount=50,
            tax_type="TDS",
        )
        db_session.add_all([t1, t2])
        db_session.commit()

        results = db_session.query(Tax).filter_by(
            payment_id=sample_payment.id
        ).all()
        assert len(results) == 2

    def test_multiple_adjustments_per_payment(self, db_session, sample_payment):
        """Payment should support multiple adjustments."""
        a1 = Adjustment(
            id="ADJ-TEST-010",
            payment_id=sample_payment.id,
            amount=1000,
            adjustment_type="CREDIT",
        )
        a2 = Adjustment(
            id="ADJ-TEST-011",
            payment_id=sample_payment.id,
            amount=-500,
            adjustment_type="DEBIT",
        )
        db_session.add_all([a1, a2])
        db_session.commit()

        results = db_session.query(Adjustment).filter_by(
            payment_id=sample_payment.id
        ).all()
        assert len(results) == 2

    def test_multiple_evidence_links_per_exception(self, db_session, sample_exception):
        """Exception should support multiple evidence links."""
        link1 = EvidenceLink(
            id="EL-TEST-010",
            exception_id="EXC-TEST-001",
            case_id="CASE-TEST-001",
            entity_type="PAYMENT",
            entity_id="PAY-TEST-001",
            relationship="CALCULATION_COMPONENT",
        )
        link2 = EvidenceLink(
            id="EL-TEST-011",
            exception_id="EXC-TEST-001",
            case_id="CASE-TEST-001",
            entity_type="SETTLEMENT",
            entity_id="SET-TEST-001",
            relationship="CALCULATION_COMPONENT",
        )
        db_session.add_all([link1, link2])
        db_session.commit()

        results = db_session.query(EvidenceLink).filter_by(
            exception_id="EXC-TEST-001"
        ).all()
        assert len(results) == 2
        entity_types = sorted([r.entity_type for r in results])
        assert entity_types == ["PAYMENT", "SETTLEMENT"]

    def test_exception_references_reconciliation(self, db_session, sample_reconciliation_result):
        """Exception should reference a reconciliation result via reconciliation_id."""
        exc = FinancialException(
            id="EXC-TEST-020",
            case_id="CASE-TEST-020",
            payment_id="PAY-TEST-020",
            batch_id="BATCH-TEST-001",
            expected_amount=50000,
            actual_amount=49000,
            difference=1000,
            exception_type=ExceptionType.TIMING_DIFFERENCE.value,
            status=ExceptionStatus.OPEN,
            reconciliation_id="REC-TEST-001",
        )
        db_session.add(exc)
        db_session.commit()

        results = db_session.query(FinancialException).filter_by(
            reconciliation_id="REC-TEST-001"
        ).all()
        assert len(results) == 1

    def test_payment_isolation_across_merchants(self, db_session):
        """Different payments should not affect each other even with different merchants."""
        p1 = Payment(id="PAY-M01-001", merchant_id="MER-01", amount=10000)
        p2 = Payment(id="PAY-M02-001", merchant_id="MER-02", amount=20000)
        db_session.add_all([p1, p2])
        db_session.commit()

        results_mer01 = db_session.query(Payment).filter_by(
            merchant_id="MER-01"
        ).all()
        results_mer02 = db_session.query(Payment).filter_by(
            merchant_id="MER-02"
        ).all()
        assert len(results_mer01) == 1
        assert len(results_mer02) == 1
        assert results_mer01[0].amount == 10000
        assert results_mer02[0].amount == 20000

    def test_settlement_filtered_by_payment_id(self, db_session, sample_payment):
        """Settlements for different payments should be isolated."""
        s1 = Settlement(id="SET-FILTER-01", payment_id=sample_payment.id, amount=50000)
        s2 = Settlement(id="SET-FILTER-02", payment_id="OTHER-PAY-001", amount=30000)
        db_session.add_all([s1, s2])
        db_session.commit()

        results = db_session.query(Settlement).filter_by(
            payment_id=sample_payment.id
        ).all()
        assert len(results) == 1
        assert results[0].id == "SET-FILTER-01"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Idempotency of Persistence Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    """Test that PersistenceService operations are idempotent."""

    def test_persist_reconciliation_result_idempotent(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Persisting same result twice should update, not duplicate."""
        result1 = persistence_service.persist_reconciliation_result(
            reconciliation_schema_result, "BATCH-IDEM-001"
        )
        db_session.commit()
        count1 = db_session.query(ReconciliationResult).filter_by(
            batch_id="BATCH-IDEM-001"
        ).count()

        result2 = persistence_service.persist_reconciliation_result(
            reconciliation_schema_result, "BATCH-IDEM-001"
        )
        db_session.commit()
        count2 = db_session.query(ReconciliationResult).filter_by(
            batch_id="BATCH-IDEM-001"
        ).count()

        assert count1 == 1
        assert count2 == 1
        assert result1.id == result2.id

    def test_persist_exception_idempotent(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Persisting same exception twice should update, not duplicate."""
        exc1 = persistence_service.persist_exception(
            reconciliation_schema_result, "BATCH-IDEM-002"
        )
        db_session.commit()
        assert exc1 is not None
        count1 = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-IDEM-002"
        ).count()

        exc2 = persistence_service.persist_exception(
            reconciliation_schema_result, "BATCH-IDEM-002"
        )
        db_session.commit()
        count2 = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-IDEM-002"
        ).count()

        assert count1 == 1
        assert count2 == 1
        assert exc1.id == exc2.id

    def test_persist_matched_no_exception(
        self, db_session, persistence_service
    ):
        """Matched results should NOT create exception records."""
        matched = ReconciliationResultSchema(
            reconciliation_id="REC-MATCH-001",
            case_id="CASE-MATCH-001",
            payment_id="PAY-MATCH-001",
            merchant_id="MER-MATCH-01",
            payment_amount=50000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=50000,
            actual_amount=50000,
            difference=0,
            match_status=MatchStatus.MATCHED,
            exception_type=ExceptionType.EXACT_MATCH,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        result = persistence_service.persist_reconciliation_result(
            matched, "BATCH-MATCH-001"
        )
        assert result is not None
        exc = persistence_service.persist_exception(matched, "BATCH-MATCH-001")
        assert exc is None
        count = db_session.query(FinancialException).filter_by(
            batch_id="BATCH-MATCH-001"
        ).count()
        assert count == 0

    def test_persist_different_batches_creates_separate_records(
        self, db_session, persistence_service
    ):
        """Same case in different batches should create separate records."""
        schema1 = ReconciliationResultSchema(
            reconciliation_id="REC-DIFF-001",
            case_id="CASE-DIFF-001",
            payment_id="PAY-DIFF-001",
            merchant_id="MER-DIFF-01",
            payment_amount=100000,
            total_refunds=500,
            total_fees=200,
            total_taxes=1800,
            total_adjustments=0,
            expected_amount=97500,
            actual_amount=97000,
            difference=500,
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.FEE_DIFFERENCE,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        schema2 = ReconciliationResultSchema(
            reconciliation_id="REC-DIFF-002",
            case_id="CASE-DIFF-001",
            payment_id="PAY-DIFF-001",
            merchant_id="MER-DIFF-01",
            payment_amount=100000,
            total_refunds=500,
            total_fees=200,
            total_taxes=1800,
            total_adjustments=0,
            expected_amount=97500,
            actual_amount=97000,
            difference=500,
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.FEE_DIFFERENCE,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        r1 = persistence_service.persist_reconciliation_result(
            schema1, "BATCH-DIFF-001"
        )
        r2 = persistence_service.persist_reconciliation_result(
            schema2, "BATCH-DIFF-002"
        )
        db_session.commit()

        assert r1.id == "REC-DIFF-001"
        assert r2.id == "REC-DIFF-002"
        count = db_session.query(ReconciliationResult).filter_by(
            case_id="CASE-DIFF-001"
        ).count()
        assert count == 2

    def test_persist_batch_idempotent(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Persisting the same batch twice should be idempotent."""
        stats1 = persistence_service.persist_batch(
            [reconciliation_schema_result], "BATCH-BATCH-001"
        )

        stats2 = persistence_service.persist_batch(
            [reconciliation_schema_result], "BATCH-BATCH-001"
        )

        assert stats1["total"] == 1
        assert stats2["total"] == 1
        count = db_session.query(ReconciliationResult).filter_by(
            batch_id="BATCH-BATCH-001"
        ).count()
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Transaction Rollback on Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestTransactionRollback:
    """Test that failed transactions don't leave partial data."""

    def test_persist_batch_rollback_on_error(
        self, db_session, persistence_service
    ):
        """If one item in a batch fails, the entire batch should be rolled back."""
        good_result = ReconciliationResultSchema(
            reconciliation_id="REC-ROLL-001",
            case_id="CASE-ROLL-001",
            payment_id="PAY-ROLL-001",
            merchant_id="MER-ROLL-01",
            payment_amount=50000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=50000,
            actual_amount=50000,
            difference=0,
            match_status=MatchStatus.MATCHED,
            exception_type=ExceptionType.EXACT_MATCH,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )

        # bad_result has the same reconciliation_id (primary key) to trigger IntegrityError
        bad_result = ReconciliationResultSchema(
            reconciliation_id="REC-ROLL-001",  # Same PK as good_result
            case_id="CASE-ROLL-002",
            payment_id="PAY-ROLL-002",
            merchant_id="MER-ROLL-01",
            payment_amount=30000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=30000,
            actual_amount=29000,
            difference=1000,
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.FEE_DIFFERENCE,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )

        with pytest.raises(RuntimeError, match="Batch persistence failed"):
            persistence_service.persist_batch(
                [good_result, bad_result], "BATCH-ROLL-001"
            )

        # After rollback, no records should exist for this batch
        count = db_session.query(ReconciliationResult).filter_by(
            batch_id="BATCH-ROLL-001"
        ).count()
        assert count == 0

    def test_failed_persist_does_not_corrupt_session(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """After a failed batch persist, the session should still be usable."""
        try:
            persistence_service.persist_batch(
                [reconciliation_schema_result], "BATCH-ROLL-002"
            )
        except Exception:
            pass

        # Session should still be usable after rollback
        good_result = ReconciliationResultSchema(
            reconciliation_id="REC-ROLL-010",
            case_id="CASE-ROLL-010",
            payment_id="PAY-ROLL-010",
            merchant_id="MER-ROLL-10",
            payment_amount=40000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=40000,
            actual_amount=40000,
            difference=0,
            match_status=MatchStatus.MATCHED,
            exception_type=ExceptionType.EXACT_MATCH,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        result = persistence_service.persist_reconciliation_result(
            good_result, "BATCH-ROLL-003"
        )
        db_session.commit()
        assert result is not None

    def test_exception_integrity_constraint(
        self, db_session, sample_exception
    ):
        """Violating unique constraint (case_id + batch_id) should raise IntegrityError."""
        dup = FinancialException(
            id="EXC-DUP-001",
            case_id="CASE-TEST-001",
            payment_id="PAY-TEST-001",
            batch_id="BATCH-TEST-001",  # Same case_id + batch_id
            expected_amount=97500,
            actual_amount=97000,
            difference=500,
            exception_type=ExceptionType.FEE_DIFFERENCE.value,
            status=ExceptionStatus.OPEN,
            reconciliation_id="REC-TEST-001",
        )
        db_session.add(dup)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_reconciliation_result_integrity_constraint(
        self, db_session, sample_reconciliation_result
    ):
        """Violating unique constraint on reconciliation results should raise IntegrityError."""
        dup = ReconciliationResult(
            id="REC-DUP-001",
            case_id="CASE-TEST-001",
            payment_id="PAY-TEST-001",
            merchant_id="MER-TEST-01",
            batch_id="BATCH-TEST-001",
            payment_amount=100000,
            expected_amount=97500,
            actual_amount=97000,
            difference=500,
            match_status=MatchStatus.EXCEPTION.value,
            exception_type=ExceptionType.FEE_DIFFERENCE.value,
        )
        db_session.add(dup)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Historical Case Retrieval
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalCaseRetrieval:
    """Test retrieving historical resolution records."""

    def test_retrieve_by_exception_type(
        self, db_session, sample_historical_resolution
    ):
        """Should retrieve historical cases filtered by exception type."""
        results = db_session.query(HistoricalResolution).filter_by(
            exception_type=ExceptionType.FEE_DIFFERENCE.value
        ).all()
        assert len(results) == 1
        assert results[0].resolution_type == "FEE_REVERSAL"

    def test_retrieve_by_outcome(self, db_session, sample_historical_resolution):
        """Should retrieve historical cases filtered by outcome."""
        results = db_session.query(HistoricalResolution).filter_by(
            outcome="RESOLVED"
        ).all()
        assert len(results) == 1

    def test_retrieve_by_resolution_type(
        self, db_session, sample_historical_resolution
    ):
        """Should retrieve historical cases filtered by resolution_type."""
        results = db_session.query(HistoricalResolution).filter_by(
            resolution_type="FEE_REVERSAL"
        ).all()
        assert len(results) == 1

    def test_multiple_historical_resolutions_same_exception(
        self, db_session
    ):
        """Should support multiple historical resolutions for different exceptions."""
        hr1 = HistoricalResolution(
            id="HRES-MULTI-001",
            case_id="CASE-HIST-001",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            exception_type=ExceptionType.FEE_DIFFERENCE.value,
        )
        hr2 = HistoricalResolution(
            id="HRES-MULTI-002",
            case_id="CASE-HIST-002",
            resolution_type="REFUND_ISSUE",
            outcome="RESOLVED",
            exception_type=ExceptionType.REFUND_ADJUSTMENT.value,
        )
        hr3 = HistoricalResolution(
            id="HRES-MULTI-003",
            case_id="CASE-HIST-003",
            resolution_type="ESCALATED",
            outcome="ESCALATED",
            exception_type=ExceptionType.UNKNOWN.value,
        )
        db_session.add_all([hr1, hr2, hr3])
        db_session.commit()

        resolved = db_session.query(HistoricalResolution).filter_by(
            outcome="RESOLVED"
        ).all()
        assert len(resolved) == 2

        escalated = db_session.query(HistoricalResolution).filter_by(
            outcome="ESCALATED"
        ).all()
        assert len(escalated) == 1

    def test_historical_resolution_with_metadata(
        self, db_session
    ):
        """Should store and retrieve JSON metadata."""
        hr = HistoricalResolution(
            id="HRES-META-001",
            case_id="CASE-META-001",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            resolution_metadata=json.dumps({
                "amount_reversed": 500,
                "fee_type": "PLATFORM_FEE",
                "approver": "system",
            }),
        )
        db_session.add(hr)
        db_session.commit()

        result = db_session.query(HistoricalResolution).filter_by(
            id="HRES-META-001"
        ).first()
        metadata = json.loads(result.resolution_metadata)
        assert metadata["amount_reversed"] == 500
        assert metadata["fee_type"] == "PLATFORM_FEE"

    def test_historical_resolvable_flag(self, db_session):
        """Should store and query resolvable flag."""
        hr1 = HistoricalResolution(
            id="HRES-RES-001",
            case_id="CASE-RES-001",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            resolvable=True,
        )
        hr2 = HistoricalResolution(
            id="HRES-RES-002",
            case_id="CASE-RES-002",
            resolution_type="ESCALATED",
            outcome="ESCALATED",
            resolvable=False,
        )
        db_session.add_all([hr1, hr2])
        db_session.commit()

        resolvable = db_session.query(HistoricalResolution).filter_by(
            resolvable=True
        ).all()
        assert len(resolvable) == 1

        not_resolvable = db_session.query(HistoricalResolution).filter_by(
            resolvable=False
        ).all()
        assert len(not_resolvable) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. PersistenceService Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistenceServiceIntegration:
    """Test the PersistenceService against real database operations."""

    def test_persist_and_get_reconciliation_results(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Should persist and retrieve reconciliation results."""
        persistence_service.persist_reconciliation_result(
            reconciliation_schema_result, "BATCH-INT-001"
        )
        db_session.commit()

        results = persistence_service.get_reconciliation_results(
            batch_id="BATCH-INT-001"
        )
        assert len(results) == 1
        assert results[0].case_id == "CASE-SCHEMA-001"

    def test_persist_and_get_exceptions(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Should persist and retrieve exception records."""
        persistence_service.persist_exception(
            reconciliation_schema_result, "BATCH-INT-002"
        )
        db_session.commit()

        exceptions = persistence_service.get_exceptions(
            batch_id="BATCH-INT-002"
        )
        assert len(exceptions) == 1
        assert exceptions[0].exception_type == ExceptionType.FEE_DIFFERENCE.value

    def test_get_exceptions_filtered_by_status(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Should filter exceptions by status."""
        persistence_service.persist_exception(
            reconciliation_schema_result, "BATCH-INT-003"
        )
        db_session.commit()

        open_exceptions = persistence_service.get_exceptions(
            status=ExceptionStatus.OPEN
        )
        assert len(open_exceptions) == 1

        resolved_exceptions = persistence_service.get_exceptions(
            status=ExceptionStatus.RESOLVED
        )
        assert len(resolved_exceptions) == 0

    def test_persist_evidence(
        self, db_session, persistence_service
    ):
        """Should persist evidence with JSON data."""
        evidence = persistence_service.persist_evidence(
            reconciliation_id="REC-INT-001",
            evidence_type="CALCULATION_BREAKDOWN",
            evidence_data={"expected": 50000, "actual": 49000, "difference": 1000},
        )
        db_session.commit()

        stored = db_session.query(ReconciliationEvidence).filter_by(
            id=evidence.id
        ).first()
        assert stored is not None
        data = json.loads(stored.evidence_data)
        assert data["expected"] == 50000

    def test_persist_batch_mixed_results(
        self, db_session, persistence_service
    ):
        """Should handle batch with both matched and exception results."""
        matched = ReconciliationResultSchema(
            reconciliation_id="REC-BATCH-001",
            case_id="CASE-BATCH-001",
            payment_id="PAY-BATCH-001",
            merchant_id="MER-BATCH-01",
            payment_amount=50000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=50000,
            actual_amount=50000,
            difference=0,
            match_status=MatchStatus.MATCHED,
            exception_type=ExceptionType.EXACT_MATCH,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        exception = ReconciliationResultSchema(
            reconciliation_id="REC-BATCH-002",
            case_id="CASE-BATCH-002",
            payment_id="PAY-BATCH-002",
            merchant_id="MER-BATCH-01",
            payment_amount=75000,
            total_refunds=0,
            total_fees=500,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=74500,
            actual_amount=74000,
            difference=500,
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.FEE_DIFFERENCE,
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        stats = persistence_service.persist_batch(
            [matched, exception], "BATCH-MIXED-001"
        )
        assert stats["matched"] == 1
        assert stats["exceptions"] == 1
        assert stats["errors"] == 0

    def test_persist_batch_statistics(
        self, db_session, persistence_service
    ):
        """Should return correct batch persistence statistics."""
        results = []
        for i in range(5):
            is_exception = i % 2 == 1
            results.append(
                ReconciliationResultSchema(
                    reconciliation_id=f"REC-STAT-{i:03d}",
                    case_id=f"CASE-STAT-{i:03d}",
                    payment_id=f"PAY-STAT-{i:03d}",
                    merchant_id="MER-STAT-01",
                    payment_amount=10000 * (i + 1),
                    total_refunds=0,
                    total_fees=0,
                    total_taxes=0,
                    total_adjustments=0,
                    expected_amount=10000 * (i + 1),
                    actual_amount=10000 * (i + 1) - (500 if is_exception else 0),
                    difference=500 if is_exception else 0,
                    match_status=MatchStatus.EXCEPTION if is_exception else MatchStatus.MATCHED,
                    exception_type=ExceptionType.FEE_DIFFERENCE if is_exception else ExceptionType.EXACT_MATCH,
                    reconciliation_status=ReconciliationStatus.PROCESSED,
                )
            )

        stats = persistence_service.persist_batch(results, "BATCH-STAT-001")
        assert stats["total"] == 5
        assert stats["matched"] == 3
        assert stats["exceptions"] == 2
        assert stats["errors"] == 0

    def test_get_reconciliation_results_by_case(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Should retrieve reconciliation results by case_id."""
        persistence_service.persist_reconciliation_result(
            reconciliation_schema_result, "BATCH-CASE-001"
        )
        db_session.commit()

        results = persistence_service.get_reconciliation_results(
            case_id="CASE-SCHEMA-001"
        )
        assert len(results) == 1

    def test_update_existing_reconciliation_result(
        self, db_session, persistence_service, reconciliation_schema_result
    ):
        """Persisting with updated values should update the existing record."""
        persistence_service.persist_reconciliation_result(
            reconciliation_schema_result, "BATCH-UPD-001"
        )
        db_session.commit()

        # Create updated version with different values
        updated_schema = ReconciliationResultSchema(
            reconciliation_id="REC-SCHEMA-001",
            case_id="CASE-SCHEMA-001",
            payment_id="PAY-SCHEMA-001",
            merchant_id="MER-SCHEMA-01",
            payment_amount=100000,
            total_refunds=1000,  # Changed
            total_fees=200,
            total_taxes=1800,
            total_adjustments=0,
            expected_amount=97000,  # Changed due to refund change
            actual_amount=97000,
            difference=0,  # Changed
            match_status=MatchStatus.MATCHED,  # Changed
            exception_type=ExceptionType.EXACT_MATCH,  # Changed
            reconciliation_status=ReconciliationStatus.PROCESSED,
        )
        persistence_service.persist_reconciliation_result(
            updated_schema, "BATCH-UPD-001"
        )
        db_session.commit()

        results = persistence_service.get_reconciliation_results(
            batch_id="BATCH-UPD-001"
        )
        assert len(results) == 1
        assert results[0].match_status == MatchStatus.MATCHED.value
        assert results[0].difference == 0
        assert results[0].total_refunds == 1000


# ─────────────────────────────────────────────────────────────────────────────
# 11. Data Integrity and Constraints
# ─────────────────────────────────────────────────────────────────────────────


class TestDataIntegrity:
    """Test data integrity constraints."""

    def test_exception_status_values(self, db_session):
        """Exception status should only accept defined values."""
        assert ExceptionStatus.OPEN == "OPEN"
        assert ExceptionStatus.MATCHED == "MATCHED"
        assert ExceptionStatus.UNRESOLVED == "UNRESOLVED"
        assert ExceptionStatus.RESOLVED == "RESOLVED"

    def test_exception_type_values(self):
        """ExceptionType enum should have all defined types."""
        assert ExceptionType.EXACT_MATCH.value == "EXACT_MATCH"
        assert ExceptionType.FEE_DIFFERENCE.value == "FEE_DIFFERENCE"
        assert ExceptionType.REFUND_ADJUSTMENT.value == "REFUND_ADJUSTMENT"
        assert ExceptionType.TAX_ADJUSTMENT.value == "TAX_ADJUSTMENT"
        assert ExceptionType.TIMING_DIFFERENCE.value == "TIMING_DIFFERENCE"
        assert ExceptionType.PARTIAL_SETTLEMENT.value == "PARTIAL_SETTLEMENT"
        assert ExceptionType.DUPLICATE.value == "DUPLICATE"
        assert ExceptionType.MISSING_RECORD.value == "MISSING_RECORD"
        assert ExceptionType.COMPLEX_MULTI_ADJUSTMENT.value == "COMPLEX_MULTI_ADJUSTMENT"
        assert ExceptionType.UNKNOWN.value == "UNKNOWN"

    def test_match_status_values(self):
        """MatchStatus enum should have all defined statuses."""
        assert MatchStatus.MATCHED.value == "MATCHED"
        assert MatchStatus.EXCEPTION.value == "EXCEPTION"
        assert MatchStatus.MISSING.value == "MISSING"
        assert MatchStatus.DUPLICATE.value == "DUPLICATE"

    def test_reconciliation_status_values(self):
        """ReconciliationStatus enum should have defined statuses."""
        assert ReconciliationStatus.PROCESSED.value == "PROCESSED"

    def test_negative_adjustment_amount(self, db_session, sample_payment):
        """Adjustment amount can be negative (debit)."""
        adj = Adjustment(
            id="ADJ-NEG-001",
            payment_id=sample_payment.id,
            amount=-5000,
            adjustment_type="DEBIT",
        )
        db_session.add(adj)
        db_session.commit()

        result = db_session.query(Adjustment).filter_by(id="ADJ-NEG-001").first()
        assert result.amount == -5000

    def test_zero_amount_fields(self, db_session, sample_payment):
        """Zero amounts should be valid for financial records."""
        refund = Refund(
            id="REF-ZERO-001",
            payment_id=sample_payment.id,
            amount=0,
            status="PROCESSED",
        )
        db_session.add(refund)
        db_session.commit()

        result = db_session.query(Refund).filter_by(id="REF-ZERO-001").first()
        assert result.amount == 0

    def test_large_amounts(self, db_session, sample_payment):
        """Large amounts (max Razorpay values) should be supported."""
        payment = Payment(
            id="PAY-LARGE-001",
            merchant_id="MER-LARGE-01",
            amount=100000000.0,  # 10 crore in paise
        )
        db_session.add(payment)
        db_session.commit()

        result = db_session.query(Payment).filter_by(id="PAY-LARGE-001").first()
        assert result.amount == 100000000.0

    def test_null_optional_fields(self, db_session):
        """Nullable fields should accept None."""
        hr = HistoricalResolution(
            id="HRES-NULL-001",
            case_id="CASE-NULL-001",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            resolved_amount=None,
            difference_at_resolution=None,
            exception_type=None,
            resolvable=None,
            notes=None,
            resolution_metadata=None,
        )
        db_session.add(hr)
        db_session.commit()

        result = db_session.query(HistoricalResolution).filter_by(
            id="HRES-NULL-001"
        ).first()
        assert result.resolved_amount is None
        assert result.notes is None


# ─────────────────────────────────────────────────────────────────────────────
# 12. Pagination and Filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestPaginationAndFiltering:
    """Test pagination and filtering of database queries."""

    def test_limit_and_offset(self, db_session):
        """Should support limit and offset for pagination."""
        for i in range(10):
            p = Payment(
                id=f"PAY-PAGE-{i:03d}",
                merchant_id="MER-PAGE-01",
                amount=1000.0 * (i + 1),
            )
            db_session.add(p)
        db_session.commit()

        # Page 1
        page1 = db_session.query(Payment).offset(0).limit(3).all()
        assert len(page1) == 3
        assert page1[0].id == "PAY-PAGE-000"

        # Page 2
        page2 = db_session.query(Payment).offset(3).limit(3).all()
        assert len(page2) == 3
        assert page2[0].id == "PAY-PAGE-003"

        # Last page
        page4 = db_session.query(Payment).offset(9).limit(3).all()
        assert len(page4) == 1

    def test_offset_beyond_data(self, db_session):
        """Offset beyond data should return empty list."""
        p = Payment(id="PAY-OFFSET-001", merchant_id="MER-01", amount=1000.0)
        db_session.add(p)
        db_session.commit()

        results = db_session.query(Payment).offset(100).limit(10).all()
        assert results == []

    def test_count_total(self, db_session):
        """Should support counting total records."""
        for i in range(5):
            p = Payment(
                id=f"PAY-COUNT-{i:03d}",
                merchant_id="MER-COUNT-01",
                amount=1000.0,
            )
            db_session.add(p)
        db_session.commit()

        count = db_session.query(Payment).count()
        assert count == 5

    def test_filter_by_date_range(self, db_session):
        """Should support filtering by datetime range."""
        hr1 = HistoricalResolution(
            id="HRES-DATE-001",
            case_id="CASE-DATE-001",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            created_at=datetime(2026, 1, 1),
        )
        hr2 = HistoricalResolution(
            id="HRES-DATE-002",
            case_id="CASE-DATE-002",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            created_at=datetime(2026, 6, 1),
        )
        hr3 = HistoricalResolution(
            id="HRES-DATE-003",
            case_id="CASE-DATE-003",
            resolution_type="FEE_REVERSAL",
            outcome="RESOLVED",
            created_at=datetime(2026, 12, 31),
        )
        db_session.add_all([hr1, hr2, hr3])
        db_session.commit()

        # Q2 only
        from datetime import date
        q2_start = datetime(2026, 4, 1)
        q2_end = datetime(2026, 7, 1)
        results = db_session.query(HistoricalResolution).filter(
            HistoricalResolution.created_at >= q2_start,
            HistoricalResolution.created_at < q2_end,
        ).all()
        assert len(results) == 1
        assert results[0].id == "HRES-DATE-002"

    def test_filter_by_multiple_criteria(self, db_session):
        """Should support combining multiple filter criteria."""
        for i in range(5):
            exc = FinancialException(
                id=f"EXC-FILTER-{i:03d}",
                case_id=f"CASE-FILTER-{i:03d}",
                payment_id=f"PAY-FILTER-{i:03d}",
                batch_id="BATCH-FILTER-001",
                expected_amount=50000,
                actual_amount=49000,
                difference=1000,
                exception_type=(
                    ExceptionType.FEE_DIFFERENCE.value
                    if i < 3
                    else ExceptionType.TIMING_DIFFERENCE.value
                ),
                status=(
                    ExceptionStatus.OPEN
                    if i % 2 == 0
                    else ExceptionStatus.RESOLVED
                ),
                reconciliation_id=f"REC-FILTER-{i:03d}",
            )
            db_session.add(exc)
        db_session.commit()

        # FEE_DIFFERENCE + OPEN only
        results = db_session.query(FinancialException).filter(
            FinancialException.exception_type == ExceptionType.FEE_DIFFERENCE.value,
            FinancialException.status == ExceptionStatus.OPEN,
            FinancialException.batch_id == "BATCH-FILTER-001",
        ).all()
        assert len(results) == 2  # Indices 0, 2


# ─────────────────────────────────────────────────────────────────────────────
# 13. Test Isolation Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestIsolation:
    """Verify that tests don't affect each other."""

    def test_isolation_first_insert(self, db_session):
        """First test should see an empty payments table."""
        count = db_session.query(Payment).count()
        assert count == 0

    def test_isolation_after_first_test(self, db_session):
        """Second test should also see an empty payments table (no cross-test data)."""
        count = db_session.query(Payment).count()
        assert count == 0

    def test_isolation_insert_and_verify(self, db_session):
        """Insert a record and verify isolation."""
        p = Payment(id="PAY-ISO-001", merchant_id="MER-ISO-01", amount=1000.0)
        db_session.add(p)
        db_session.commit()
        count = db_session.query(Payment).count()
        assert count == 1

    def test_isolation_after_insert(self, db_session):
        """Previous test's insert should not be visible."""
        count = db_session.query(Payment).count()
        assert count == 0
