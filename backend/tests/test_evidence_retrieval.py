"""
Tests for evidence retrieval service.

Covers:
- Normal payment retrieval
- Payment with refund
- Payment with fee
- Payment with tax
- Payment with adjustment
- Multiple settlements
- Missing settlement
- Duplicate settlement
- Complex multi-adjustment
- Unknown case
- Missing evidence representation
- Conflict representation
- EvidenceLink persistence with idempotency
- No ground truth leakage
"""

import os
import sys
from pathlib import Path

import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_evidence_retrieval.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.db_test_helper import get_test_session, reset_database
from app.models.exception import FinancialException
from app.models.refund import Refund
from app.models.fee import Fee
from app.models.tax import Tax
from app.models.adjustment import Adjustment
from app.models.evidence_link import EvidenceLink
from app.services.evidence_retrieval import (
    EvidenceRetrievalService,
    RELATIONSHIP_PRIMARY,
    RELATIONSHIP_CALCULATION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset database before each test."""
    reset_database()
    yield
    reset_database()


@pytest.fixture
def session():
    """Provide a test session."""
    s = get_test_session()
    yield s
    s.close()


@pytest.fixture
def service(session):
    """Provide an evidence retrieval service."""
    return EvidenceRetrievalService(session)


def _create_exception(
    session,
    exception_id="EXC-001",
    case_id="CASE-001",
    payment_id="PAY-001",
    batch_id="batch_001",
    expected_amount=100000,
    actual_amount=100000,
    difference=0,
    exception_type="EXACT_MATCH",
    status="OPEN",
    reconciliation_id="REC-001",
):
    """Helper to create a FinancialException."""
    exc = FinancialException(
        id=exception_id,
        case_id=case_id,
        payment_id=payment_id,
        batch_id=batch_id,
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        exception_type=exception_type,
        status=status,
        reconciliation_id=reconciliation_id,
    )
    session.add(exc)
    session.flush()
    return exc


# ─────────────────────────────────────────────────────────────────────────────
# Normal Payment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalPaymentRetrieval:
    """Tests for basic payment evidence retrieval."""

    def test_retrieve_exception_not_found(self, service):
        """Test that non-existent exception returns None."""
        result = service.retrieve_by_exception_id("EXC-NONEXISTENT")
        assert result is None

    def test_retrieve_basic_exception(self, session, service):
        """Test basic exception retrieval with no financial records."""
        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
            exception_type="EXACT_MATCH",
        )

        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg is not None
        assert pkg.exception_id == "EXC-001"
        assert pkg.case_id == "CASE-001"
        assert pkg.payment_id == "PAY-001"
        assert pkg.expected_amount == 100000
        assert pkg.actual_amount == 100000
        assert pkg.difference == 0
        assert pkg.exception_type == "EXACT_MATCH"

    def test_payment_record_is_primary(self, session, service):
        """Test that payment record is marked as PRIMARY_RECORD."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg.payment is not None
        assert pkg.payment.entity_type == "PAYMENT"
        assert pkg.payment.relationship == RELATIONSHIP_PRIMARY

    def test_empty_settlements_when_none_exist(self, session, service):
        """Test that empty settlements list is explicitly returned."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg.settlements == []
        assert pkg.total_settlement_amount == 0

    def test_retrieve_by_case_id(self, session, service):
        """Test retrieval by case_id."""
        exc = _create_exception(session, case_id="CASE-999")
        pkg = service.retrieve_by_case_id("CASE-999")

        assert pkg is not None
        assert pkg.case_id == "CASE-999"
        assert pkg.exception_id == exc.id

    def test_retrieve_by_case_id_not_found(self, service):
        """Test retrieval by case_id when no exception exists."""
        pkg = service.retrieve_by_case_id("CASE-NONEXISTENT")
        assert pkg is None


# ─────────────────────────────────────────────────────────────────────────────
# Refund Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundRetrieval:
    """Tests for refund evidence retrieval."""

    def test_refund_retrieved(self, session, service):
        """Test that refunds are retrieved and included in evidence."""
        exc = _create_exception(session)
        refund = Refund(
            id="REF-001", payment_id="PAY-001", case_id="CASE-001",
            amount=5000, status="PROCESSED"
        )
        session.add(refund)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.refunds) == 1
        assert pkg.refunds[0].record_id == "REF-001"
        assert pkg.refunds[0].amount == 5000
        assert pkg.refunds[0].entity_type == "REFUND"
        assert pkg.refunds[0].relationship == RELATIONSHIP_CALCULATION
        assert pkg.total_refund_amount == 5000

    def test_multiple_refunds(self, session, service):
        """Test that multiple refunds are all retrieved."""
        exc = _create_exception(session)
        for i in range(3):
            r = Refund(
                id=f"REF-{i:03d}", payment_id="PAY-001", case_id="CASE-001",
                amount=1000 * (i + 1), status="PROCESSED"
            )
            session.add(r)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.refunds) == 3
        assert pkg.total_refund_amount == 6000  # 1000 + 2000 + 3000

    def test_no_refunds_empty_list(self, session, service):
        """Test that no refunds produces empty list."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg.refunds == []
        assert pkg.total_refund_amount == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fee Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeRetrieval:
    """Tests for fee evidence retrieval."""

    def test_fee_retrieved(self, session, service):
        """Test that fees are retrieved and included in evidence."""
        exc = _create_exception(session)
        fee = Fee(
            id="FEE-001", payment_id="PAY-001", case_id="CASE-001",
            amount=2000, fee_type="TRANSACTION"
        )
        session.add(fee)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.fees) == 1
        assert pkg.fees[0].record_id == "FEE-001"
        assert pkg.fees[0].amount == 2000
        assert pkg.fees[0].entity_type == "FEE"
        assert pkg.total_fee_amount == 2000

    def test_multiple_fees(self, session, service):
        """Test that multiple fees are all retrieved."""
        exc = _create_exception(session)
        types = ["TRANSACTION", "PLATFORM", "TDR"]
        for i, ft in enumerate(types):
            f = Fee(
                id=f"FEE-{i:03d}", payment_id="PAY-001", case_id="CASE-001",
                amount=1000 * (i + 1), fee_type=ft
            )
            session.add(f)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.fees) == 3
        assert pkg.total_fee_amount == 6000


# ─────────────────────────────────────────────────────────────────────────────
# Tax Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxRetrieval:
    """Tests for tax evidence retrieval."""

    def test_tax_retrieved(self, session, service):
        """Test that taxes are retrieved and included in evidence."""
        exc = _create_exception(session)
        tax = Tax(
            id="TAX-001", payment_id="PAY-001", case_id="CASE-001",
            amount=1800, tax_type="GST"
        )
        session.add(tax)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.taxes) == 1
        assert pkg.taxes[0].record_id == "TAX-001"
        assert pkg.taxes[0].amount == 1800
        assert pkg.taxes[0].entity_type == "TAX"
        assert pkg.total_tax_amount == 1800

    def test_multiple_taxes(self, session, service):
        """Test that multiple taxes are all retrieved."""
        exc = _create_exception(session)
        for i, tt in enumerate(["GST", "TDS"]):
            t = Tax(
                id=f"TAX-{i:03d}", payment_id="PAY-001", case_id="CASE-001",
                amount=1000 * (i + 1), tax_type=tt
            )
            session.add(t)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.taxes) == 2
        assert pkg.total_tax_amount == 3000


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentRetrieval:
    """Tests for adjustment evidence retrieval."""

    def test_adjustment_retrieved(self, session, service):
        """Test that adjustments are retrieved and included in evidence."""
        exc = _create_exception(session)
        adj = Adjustment(
            id="ADJ-001", payment_id="PAY-001", case_id="CASE-001",
            amount=3000, adjustment_type="CREDIT"
        )
        session.add(adj)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.adjustments) == 1
        assert pkg.adjustments[0].record_id == "ADJ-001"
        assert pkg.adjustments[0].amount == 3000
        assert pkg.adjustments[0].entity_type == "ADJUSTMENT"
        assert pkg.total_adjustment_amount == 3000

    def test_negative_adjustment(self, session, service):
        """Test that negative (debit) adjustments are retrieved."""
        exc = _create_exception(session)
        adj = Adjustment(
            id="ADJ-NEG", payment_id="PAY-001", case_id="CASE-001",
            amount=-5000, adjustment_type="DEBIT"
        )
        session.add(adj)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.adjustments) == 1
        assert pkg.adjustments[0].amount == -5000
        assert pkg.total_adjustment_amount == -5000

    def test_mixed_adjustments(self, session, service):
        """Test mix of credit and debit adjustments."""
        exc = _create_exception(session)
        adj1 = Adjustment(
            id="ADJ-001", payment_id="PAY-001", case_id="CASE-001",
            amount=10000, adjustment_type="CREDIT"
        )
        adj2 = Adjustment(
            id="ADJ-002", payment_id="PAY-001", case_id="CASE-001",
            amount=-3000, adjustment_type="DEBIT"
        )
        session.add(adj1)
        session.add(adj2)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.adjustments) == 2
        assert pkg.total_adjustment_amount == 7000  # 10000 - 3000


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Settlements Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleSettlements:
    """Tests for multiple settlement evidence detection."""

    def test_multiple_settlements_detected(self, session, service):
        """Test that multiple settlements are retrieved and flagged."""
        from app.models.settlement import Settlement

        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=200000,
            difference=-100000,
            exception_type="DUPLICATE",
        )

        # Create two settlements for same payment
        s1 = Settlement(id="SET-001", payment_id="PAY-001", amount=100000)
        s2 = Settlement(id="SET-002", payment_id="PAY-001", amount=100000)
        session.add(s1)
        session.add(s2)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.settlements) == 2
        assert pkg.total_settlement_amount == 200000

    def test_multiple_different_amount_settlements(self, session, service):
        """Test conflict detection for settlements with different amounts."""
        from app.models.settlement import Settlement

        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=150000,
            difference=-50000,
        )

        s1 = Settlement(id="SET-001", payment_id="PAY-001", amount=100000)
        s2 = Settlement(id="SET-002", payment_id="PAY-001", amount=50000)
        session.add(s1)
        session.add(s2)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.settlements) == 2
        assert pkg.has_conflicts()
        conflict_types = [c.conflict_type for c in pkg.conflicts]
        assert "MULTIPLE_SETTLEMENTS" in conflict_types


# ─────────────────────────────────────────────────────────────────────────────
# Missing Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingSettlement:
    """Tests for missing settlement detection."""

    def test_missing_settlement_detected(self, session, service):
        """Test that missing settlement is explicitly represented."""
        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=0,
            difference=100000,
            exception_type="MISSING_RECORD",
        )

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.settlements) == 0
        assert pkg.has_missing_settlement()
        assert len(pkg.missing_evidence) == 1
        assert pkg.missing_evidence[0].entity_type == "SETTLEMENT"
        assert pkg.missing_evidence[0].expected is True
        assert "PAY-001" in pkg.missing_evidence[0].reason

    def test_missing_settlement_expected_is_true(self, session, service):
        """Test that missing settlement is marked as expected."""
        exc = _create_exception(session, exception_type="MISSING_RECORD")
        pkg = service.retrieve_by_exception_id(exc.id)

        for m in pkg.missing_evidence:
            if m.entity_type == "SETTLEMENT":
                assert m.expected is True


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateSettlement:
    """Tests for duplicate settlement detection."""

    def test_duplicate_settlement_detected(self, session, service):
        """Test that duplicate settlements are detected as conflicts."""
        from app.models.settlement import Settlement

        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=200000,
            difference=-100000,
            exception_type="DUPLICATE",
        )

        s1 = Settlement(id="SET-DUP-1", payment_id="PAY-001", amount=100000)
        s2 = Settlement(id="SET-DUP-2", payment_id="PAY-001", amount=100000)
        session.add(s1)
        session.add(s2)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg.has_conflicts()
        # Same amount, multiple settlements = duplicate pattern
        conflict_types = [c.conflict_type for c in pkg.conflicts]
        assert "DUPLICATE_SETTLEMENT_ID" in conflict_types


# ─────────────────────────────────────────────────────────────────────────────
# Complex Multi-Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexMultiAdjustment:
    """Tests for complex multi-adjustment evidence retrieval."""

    def test_complex_case_with_all_records(self, session, service):
        """Test evidence retrieval for a case with refunds, fees, taxes, adjustments."""
        exc = _create_exception(
            session,
            expected_amount=80000,
            actual_amount=75000,
            difference=5000,
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
        )

        # Add various financial records
        refund = Refund(id="REF-C01", payment_id="PAY-001", case_id="CASE-001", amount=5000)
        fee = Fee(id="FEE-C01", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR")
        tax = Tax(id="TAX-C01", payment_id="PAY-001", case_id="CASE-001", amount=1800, tax_type="GST")
        adj = Adjustment(id="ADJ-C01", payment_id="PAY-001", case_id="CASE-001", amount=3000, adjustment_type="CREDIT")

        session.add_all([refund, fee, tax, adj])
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.refunds) == 1
        assert len(pkg.fees) == 1
        assert len(pkg.taxes) == 1
        assert len(pkg.adjustments) == 1
        assert pkg.total_refund_amount == 5000
        assert pkg.total_fee_amount == 2000
        assert pkg.total_tax_amount == 1800
        assert pkg.total_adjustment_amount == 3000

    def test_total_evidence_records_count(self, session, service):
        """Test that total_evidence_records counts correctly."""
        exc = _create_exception(session)

        # Add records
        session.add(Refund(id="REF-C02", payment_id="PAY-001", amount=1000))
        session.add(Fee(id="FEE-C02", payment_id="PAY-001", amount=500, fee_type="TDR"))
        session.add(Tax(id="TAX-C02", payment_id="PAY-001", amount=300, tax_type="GST"))
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        # 1 payment + 1 refund + 1 fee + 1 tax = 4
        assert pkg.total_evidence_records() == 4


# ─────────────────────────────────────────────────────────────────────────────
# Unknown Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownCase:
    """Tests for unknown exception evidence retrieval."""

    def test_unknown_case_retrieved(self, session, service):
        """Test that unknown cases can still have evidence retrieved."""
        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=92000,
            difference=8000,
            exception_type="UNKNOWN",
        )

        pkg = service.retrieve_by_exception_id(exc.id)

        assert pkg is not None
        assert pkg.exception_type == "UNKNOWN"
        assert pkg.difference == 8000

    def test_unknown_case_with_records(self, session, service):
        """Test unknown case with some financial records."""
        exc = _create_exception(
            session,
            expected_amount=100000,
            actual_amount=92000,
            difference=8000,
            exception_type="UNKNOWN",
        )

        fee = Fee(id="FEE-U01", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR")
        session.add(fee)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        assert len(pkg.fees) == 1
        assert pkg.total_fee_amount == 2000


# ─────────────────────────────────────────────────────────────────────────────
# Missing Evidence Representation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingEvidenceRepresentation:
    """Tests for explicit missing evidence representation."""

    def test_empty_list_vs_not_queried(self, session, service):
        """Test that empty list means 'record does not exist', not 'was not queried'."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        # Settlements empty = record does not exist
        assert pkg.settlements == []
        # Refunds empty = record does not exist
        assert pkg.refunds == []
        # Fees empty = record does not exist
        assert pkg.fees == []
        # Taxes empty = record does not exist
        assert pkg.taxes == []
        # Adjustments empty = record does not exist
        assert pkg.adjustments == []

    def test_missing_evidence_has_reason(self, session, service):
        """Test that missing evidence includes a reason."""
        exc = _create_exception(session, exception_type="MISSING_RECORD")
        pkg = service.retrieve_by_exception_id(exc.id)

        for m in pkg.missing_evidence:
            assert m.reason is not None
            assert len(m.reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Representation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictRepresentation:
    """Tests for structural conflict detection."""

    def test_no_conflicts_when_clean(self, session, service):
        """Test that clean cases have no conflicts."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        assert not pkg.has_conflicts()
        assert pkg.conflicts == []

    def test_conflict_type_is_string(self, session, service):
        """Test that conflict types are controlled strings."""
        from app.models.settlement import Settlement

        exc = _create_exception(session)
        s1 = Settlement(id="SET-C01", payment_id="PAY-001", amount=100000)
        s2 = Settlement(id="SET-C02", payment_id="PAY-001", amount=100000)
        session.add_all([s1, s2])
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        for c in pkg.conflicts:
            assert isinstance(c.conflict_type, str)
            assert isinstance(c.description, str)
            assert isinstance(c.affected_records, list)


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceLink Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceLinkPersistence:
    """Tests for evidence link creation and idempotency."""

    def test_evidence_links_created(self, session, service):
        """Test that evidence links are persisted."""
        exc = _create_exception(session)
        fee = Fee(id="FEE-L01", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR")
        session.add(fee)
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id, persist_links=True)

        links = session.query(EvidenceLink).filter_by(exception_id=exc.id).all()
        assert len(links) > 0
        assert pkg.evidence_link_count > 0

    def test_evidence_links_idempotent(self, session, service):
        """Test that running retrieval twice does not create duplicate links."""
        exc = _create_exception(session)
        fee = Fee(id="FEE-L02", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR")
        session.add(fee)
        session.flush()

        # First retrieval
        service.retrieve_by_exception_id(exc.id, persist_links=True)
        count1 = session.query(EvidenceLink).filter_by(exception_id=exc.id).count()

        # Second retrieval
        service.retrieve_by_exception_id(exc.id, persist_links=True)
        count2 = session.query(EvidenceLink).filter_by(exception_id=exc.id).count()

        # Same number of links in DB (no duplicates created)
        assert count1 == count2

    def test_evidence_links_not_created_when_disabled(self, session, service):
        """Test that evidence links are not created when persist_links=False."""
        exc = _create_exception(session)
        session.flush()

        service.retrieve_by_exception_id(exc.id, persist_links=False)

        links = session.query(EvidenceLink).filter_by(exception_id=exc.id).count()
        assert links == 0

    def test_evidence_link_references_exception(self, session, service):
        """Test that evidence links reference the correct exception."""
        exc = _create_exception(session)
        fee = Fee(id="FEE-L03", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR")
        session.add(fee)
        session.flush()

        service.retrieve_by_exception_id(exc.id, persist_links=True)

        links = session.query(EvidenceLink).filter_by(exception_id=exc.id).all()
        for link in links:
            assert link.exception_id == exc.id
            assert link.case_id == "CASE-001"

    def test_evidence_link_covers_all_entity_types(self, session, service):
        """Test that evidence links cover all entity types present."""
        exc = _create_exception(session)

        # Add one of each type
        session.add(Refund(id="REF-L01", payment_id="PAY-001", case_id="CASE-001", amount=1000))
        session.add(Fee(id="FEE-L04", payment_id="PAY-001", case_id="CASE-001", amount=500, fee_type="TDR"))
        session.add(Tax(id="TAX-L01", payment_id="PAY-001", case_id="CASE-001", amount=300, tax_type="GST"))
        session.add(Adjustment(id="ADJ-L01", payment_id="PAY-001", case_id="CASE-001", amount=200, adjustment_type="CREDIT"))
        session.flush()

        service.retrieve_by_exception_id(exc.id, persist_links=True)

        entity_types = {
            link.entity_type
            for link in session.query(EvidenceLink).filter_by(exception_id=exc.id).all()
        }
        assert "PAYMENT" in entity_types
        assert "REFUND" in entity_types
        assert "FEE" in entity_types
        assert "TAX" in entity_types
        assert "ADJUSTMENT" in entity_types


# ─────────────────────────────────────────────────────────────────────────────
# Batch Retrieval Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchRetrieval:
    """Tests for batch evidence retrieval."""

    def test_batch_retrieval(self, session, service):
        """Test retrieving evidence for multiple exceptions."""
        exc1 = _create_exception(session, exception_id="EXC-B01", case_id="CASE-B01", payment_id="PAY-B01")
        exc2 = _create_exception(session, exception_id="EXC-B02", case_id="CASE-B02", payment_id="PAY-B02")

        packages = service.retrieve_batch(["EXC-B01", "EXC-B02"])

        assert len(packages) == 2
        assert packages[0].exception_id == "EXC-B01"
        assert packages[1].exception_id == "EXC-B02"

    def test_batch_skips_missing(self, session, service):
        """Test that batch retrieval skips non-existent exceptions."""
        exc = _create_exception(session, exception_id="EXC-B03", case_id="CASE-B03", payment_id="PAY-B03")

        packages = service.retrieve_batch(["EXC-B03", "EXC-NONEXISTENT"])

        assert len(packages) == 1
        assert packages[0].exception_id == "EXC-B03"


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Verify that evidence retrieval does not read ground truth."""

    GROUND_TRUTH_IMPORTS = [
        "ground_truth",
        "true_exception_type",
        "true_resolution",
    ]

    def test_service_code_no_ground_truth(self):
        """Test that evidence retrieval service code doesn't reference ground truth."""
        import inspect
        from app.services.evidence_retrieval import EvidenceRetrievalService

        source = inspect.getsource(EvidenceRetrievalService)
        for term in self.GROUND_TRUTH_IMPORTS:
            assert term not in source, f"Ground truth reference found: {term}"

    def test_evidence_package_no_ground_truth_fields(self, session, service):
        """Test that EvidencePackage has no ground truth fields."""
        exc = _create_exception(session)
        pkg = service.retrieve_by_exception_id(exc.id)

        assert not hasattr(pkg, "true_exception_type")
        assert not hasattr(pkg, "true_resolution")
        assert not hasattr(pkg, "resolvable")
        assert not hasattr(pkg, "risk_category")


# ─────────────────────────────────────────────────────────────────────────────
# Financial Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialSummary:
    """Tests for financial summary calculation in evidence package."""

    def test_financial_summary_matches_records(self, session, service):
        """Test that financial summary matches individual record amounts."""
        exc = _create_exception(session)

        session.add(Refund(id="REF-S01", payment_id="PAY-001", case_id="CASE-001", amount=5000))
        session.add(Refund(id="REF-S02", payment_id="PAY-001", case_id="CASE-001", amount=3000))
        session.add(Fee(id="FEE-S01", payment_id="PAY-001", case_id="CASE-001", amount=2000, fee_type="TDR"))
        session.add(Fee(id="FEE-S02", payment_id="PAY-001", case_id="CASE-001", amount=1000, fee_type="PLATFORM"))
        session.add(Tax(id="TAX-S01", payment_id="PAY-001", case_id="CASE-001", amount=1800, tax_type="GST"))
        session.add(Adjustment(id="ADJ-S01", payment_id="PAY-001", case_id="CASE-001", amount=4000, adjustment_type="CREDIT"))
        session.flush()

        pkg = service.retrieve_by_exception_id(exc.id)

        # Verify summary
        assert pkg.total_refund_amount == 8000  # 5000 + 3000
        assert pkg.total_fee_amount == 3000  # 2000 + 1000
        assert pkg.total_tax_amount == 1800
        assert pkg.total_adjustment_amount == 4000

        # Verify sums match individual records
        assert sum(r.amount for r in pkg.refunds) == pkg.total_refund_amount
        assert sum(r.amount for r in pkg.fees) == pkg.total_fee_amount
        assert sum(r.amount for r in pkg.taxes) == pkg.total_tax_amount
        assert sum(r.amount for r in pkg.adjustments) == pkg.total_adjustment_amount
