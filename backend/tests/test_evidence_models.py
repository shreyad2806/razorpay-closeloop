"""
Tests for Phase 3A evidence database models.

Covers:
- Refund creation and fields
- Fee creation and fields
- Tax creation and fields
- Adjustment creation and fields
- EvidenceLink creation and relationships
- HistoricalResolution creation and fields
- Foreign-key/reference integrity
- Unique ID enforcement
- Amount constraints
- Ground truth separation (no leakage)
"""

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_evidence.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.db_test_helper import get_test_session, reset_database, get_test_engine
from app.models.refund import Refund
from app.models.fee import Fee
from app.models.tax import Tax
from app.models.adjustment import Adjustment
from app.models.evidence_link import EvidenceLink
from app.models.historical_resolution import HistoricalResolution
from app.models.exception import FinancialException, ExceptionStatus
from app.models.reconciliation import ReconciliationResult, ReconciliationEvidence


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
def sample_payment_data():
    """Common test data for financial records."""
    return {
        "payment_id": "PAY-TEST-001",
        "case_id": "CASE-TEST-001",
        "merchant_id": "MER-TEST-001",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Refund Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundModel:
    """Tests for Refund SQLAlchemy model."""

    def test_refund_creation(self, session, sample_payment_data):
        """Test basic refund creation."""
        refund = Refund(
            id="REF-TEST-001",
            payment_id=sample_payment_data["payment_id"],
            case_id=sample_payment_data["case_id"],
            merchant_id=sample_payment_data["merchant_id"],
            amount=5000,
            status="PROCESSED",
        )
        session.add(refund)
        session.flush()

        assert refund.id == "REF-TEST-001"
        assert refund.amount == 5000
        assert refund.status == "PROCESSED"
        assert refund.payment_id == "PAY-TEST-001"
        assert refund.case_id == "CASE-TEST-001"

    def test_refund_amount_is_integer(self, session):
        """Test that refund amount is stored as integer (paise)."""
        refund = Refund(id="REF-002", payment_id="PAY-001", amount=12345)
        session.add(refund)
        session.flush()
        assert isinstance(refund.amount, int)
        assert refund.amount == 12345

    def test_refund_unique_id(self, session):
        """Test that refund IDs must be unique."""
        r1 = Refund(id="REF-DUP", payment_id="PAY-001", amount=1000)
        r2 = Refund(id="REF-DUP", payment_id="PAY-002", amount=2000)
        session.add(r1)
        session.flush()
        session.add(r2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_refund_references_payment(self, session):
        """Test that refund has valid payment reference."""
        refund = Refund(id="REF-003", payment_id="PAY-001", amount=1000)
        session.add(refund)
        session.flush()
        assert refund.payment_id.startswith("PAY-")

    def test_refund_nullable_case_id(self, session):
        """Test that case_id can be null."""
        refund = Refund(id="REF-004", payment_id="PAY-001", amount=1000, case_id=None)
        session.add(refund)
        session.flush()
        assert refund.case_id is None

    def test_refund_default_status(self, session):
        """Test that refund has default status."""
        refund = Refund(id="REF-005", payment_id="PAY-001", amount=1000)
        session.add(refund)
        session.flush()
        assert refund.status == "PROCESSED"


# ─────────────────────────────────────────────────────────────────────────────
# Fee Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeModel:
    """Tests for Fee SQLAlchemy model."""

    def test_fee_creation(self, session, sample_payment_data):
        """Test basic fee creation."""
        fee = Fee(
            id="FEE-TEST-001",
            payment_id=sample_payment_data["payment_id"],
            case_id=sample_payment_data["case_id"],
            merchant_id=sample_payment_data["merchant_id"],
            amount=2000,
            fee_type="TRANSACTION",
        )
        session.add(fee)
        session.flush()

        assert fee.id == "FEE-TEST-001"
        assert fee.amount == 2000
        assert fee.fee_type == "TRANSACTION"

    def test_fee_amount_is_integer(self, session):
        """Test that fee amount is stored as integer."""
        fee = Fee(id="FEE-002", payment_id="PAY-001", amount=9999, fee_type="PLATFORM")
        session.add(fee)
        session.flush()
        assert isinstance(fee.amount, int)

    def test_fee_unique_id(self, session):
        """Test that fee IDs must be unique."""
        f1 = Fee(id="FEE-DUP", payment_id="PAY-001", amount=1000, fee_type="TDR")
        f2 = Fee(id="FEE-DUP", payment_id="PAY-002", amount=2000, fee_type="TDR")
        session.add(f1)
        session.flush()
        session.add(f2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_fee_types(self, session):
        """Test various fee types."""
        types = ["TRANSACTION", "PLATFORM", "TDR", "GST_ON_FEES", "REFUND_FEE", "CHARGEBACK_FEE"]
        for i, ft in enumerate(types):
            fee = Fee(id=f"FEE-TYPE-{i}", payment_id="PAY-001", amount=1000, fee_type=ft)
            session.add(fee)
        session.flush()
        assert session.query(Fee).count() == len(types)


# ─────────────────────────────────────────────────────────────────────────────
# Tax Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxModel:
    """Tests for Tax SQLAlchemy model."""

    def test_tax_creation(self, session, sample_payment_data):
        """Test basic tax creation."""
        tax = Tax(
            id="TAX-TEST-001",
            payment_id=sample_payment_data["payment_id"],
            case_id=sample_payment_data["case_id"],
            merchant_id=sample_payment_data["merchant_id"],
            amount=1800,
            tax_type="GST",
        )
        session.add(tax)
        session.flush()

        assert tax.id == "TAX-TEST-001"
        assert tax.amount == 1800
        assert tax.tax_type == "GST"

    def test_tax_amount_is_integer(self, session):
        """Test that tax amount is stored as integer."""
        tax = Tax(id="TAX-002", payment_id="PAY-001", amount=5432, tax_type="TDS")
        session.add(tax)
        session.flush()
        assert isinstance(tax.amount, int)

    def test_tax_unique_id(self, session):
        """Test that tax IDs must be unique."""
        t1 = Tax(id="TAX-DUP", payment_id="PAY-001", amount=1000, tax_type="GST")
        t2 = Tax(id="TAX-DUP", payment_id="PAY-002", amount=2000, tax_type="GST")
        session.add(t1)
        session.flush()
        session.add(t2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_tax_types(self, session):
        """Test various tax types."""
        types = ["GST", "TDS", "GST_ON_FEES", "SERVICE_TAX"]
        for i, tt in enumerate(types):
            tax = Tax(id=f"TAX-TYPE-{i}", payment_id="PAY-001", amount=1000, tax_type=tt)
            session.add(tax)
        session.flush()
        assert session.query(Tax).count() == len(types)


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentModel:
    """Tests for Adjustment SQLAlchemy model."""

    def test_adjustment_creation(self, session, sample_payment_data):
        """Test basic adjustment creation."""
        adj = Adjustment(
            id="ADJ-TEST-001",
            payment_id=sample_payment_data["payment_id"],
            case_id=sample_payment_data["case_id"],
            merchant_id=sample_payment_data["merchant_id"],
            amount=3000,
            adjustment_type="CREDIT",
        )
        session.add(adj)
        session.flush()

        assert adj.id == "ADJ-TEST-001"
        assert adj.amount == 3000
        assert adj.adjustment_type == "CREDIT"

    def test_adjustment_negative_amount(self, session):
        """Test that adjustment can have negative amount (debit)."""
        adj = Adjustment(id="ADJ-NEG", payment_id="PAY-001", amount=-5000, adjustment_type="DEBIT")
        session.add(adj)
        session.flush()
        assert adj.amount == -5000

    def test_adjustment_unique_id(self, session):
        """Test that adjustment IDs must be unique."""
        a1 = Adjustment(id="ADJ-DUP", payment_id="PAY-001", amount=1000, adjustment_type="CREDIT")
        a2 = Adjustment(id="ADJ-DUP", payment_id="PAY-002", amount=2000, adjustment_type="DEBIT")
        session.add(a1)
        session.flush()
        session.add(a2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_adjustment_types(self, session):
        """Test various adjustment types."""
        types = ["CREDIT", "DEBIT", "FEE_REVERSAL", "PENALTY", "BONUS", "CORRECTION"]
        for i, at in enumerate(types):
            adj = Adjustment(id=f"ADJ-TYPE-{i}", payment_id="PAY-001", amount=1000, adjustment_type=at)
            session.add(adj)
        session.flush()
        assert session.query(Adjustment).count() == len(types)


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceLink Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceLinkModel:
    """Tests for EvidenceLink SQLAlchemy model."""

    def test_evidence_link_creation(self, session):
        """Test basic evidence link creation."""
        link = EvidenceLink(
            id="EL-TEST-001",
            exception_id="EXC-TEST-001",
            case_id="CASE-TEST-001",
            entity_type="REFUND",
            entity_id="REF-TEST-001",
            relationship="CALCULATION_COMPONENT",
        )
        session.add(link)
        session.flush()

        assert link.id == "EL-TEST-001"
        assert link.entity_type == "REFUND"
        assert link.relationship == "CALCULATION_COMPONENT"

    def test_evidence_link_entity_types(self, session):
        """Test that evidence links can reference all entity types."""
        entity_types = [
            ("PAYMENT", "PAY-001"),
            ("SETTLEMENT", "SET-001"),
            ("REFUND", "REF-001"),
            ("FEE", "FEE-001"),
            ("TAX", "TAX-001"),
            ("ADJUSTMENT", "ADJ-001"),
        ]
        for i, (et, eid) in enumerate(entity_types):
            link = EvidenceLink(
                id=f"EL-TYPE-{i}",
                exception_id="EXC-001",
                case_id="CASE-001",
                entity_type=et,
                entity_id=eid,
                relationship="SUPPORTING_EVIDENCE",
            )
            session.add(link)
        session.flush()
        assert session.query(EvidenceLink).count() == len(entity_types)

    def test_evidence_link_unique_id(self, session):
        """Test that evidence link IDs must be unique."""
        l1 = EvidenceLink(
            id="EL-DUP", exception_id="EXC-001", case_id="CASE-001",
            entity_type="FEE", entity_id="FEE-001", relationship="CALCULATION_COMPONENT"
        )
        l2 = EvidenceLink(
            id="EL-DUP", exception_id="EXC-002", case_id="CASE-002",
            entity_type="TAX", entity_id="TAX-001", relationship="CALCULATION_COMPONENT"
        )
        session.add(l1)
        session.flush()
        session.add(l2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_evidence_link_references_exception(self, session):
        """Test that evidence link references a valid exception."""
        # First create the exception
        exc = FinancialException(
            id="EXC-EL-001", case_id="CASE-001", payment_id="PAY-001",
            batch_id="batch_001", expected_amount=100000, actual_amount=98000,
            difference=2000, exception_type="FEE_DIFFERENCE", status="OPEN",
            reconciliation_id="REC-001"
        )
        session.add(exc)
        session.flush()

        link = EvidenceLink(
            id="EL-REF-001", exception_id="EXC-EL-001", case_id="CASE-001",
            entity_type="REFUND", entity_id="REF-001", relationship="CALCULATION_COMPONENT"
        )
        session.add(link)
        session.flush()
        assert link.exception_id == "EXC-EL-001"

    def test_evidence_link_multiple_for_same_exception(self, session):
        """Test that multiple evidence links can point to same exception."""
        exc = FinancialException(
            id="EXC-MULTI", case_id="CASE-001", payment_id="PAY-001",
            batch_id="batch_001", expected_amount=100000, actual_amount=98000,
            difference=2000, exception_type="FEE_DIFFERENCE", status="OPEN",
            reconciliation_id="REC-001"
        )
        session.add(exc)
        session.flush()

        for i in range(3):
            link = EvidenceLink(
                id=f"EL-MULTI-{i}", exception_id="EXC-MULTI", case_id="CASE-001",
                entity_type="FEE", entity_id=f"FEE-{i}", relationship="CALCULATION_COMPONENT"
            )
            session.add(link)
        session.flush()

        count = session.query(EvidenceLink).filter_by(exception_id="EXC-MULTI").count()
        assert count == 3


# ─────────────────────────────────────────────────────────────────────────────
# HistoricalResolution Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalResolutionModel:
    """Tests for HistoricalResolution SQLAlchemy model."""

    def test_historical_resolution_creation(self, session):
        """Test basic historical resolution creation."""
        hr = HistoricalResolution(
            id="HRES-TEST-001",
            exception_id="EXC-TEST-001",
            case_id="CASE-TEST-001",
            resolution_type="FEE_ADJUSTMENT",
            outcome="RESOLVED",
            resolved_amount=2000,
            difference_at_resolution=2000,
            exception_type="FEE_DIFFERENCE",
            resolvable=True,
            source="deterministic",
        )
        session.add(hr)
        session.flush()

        assert hr.id == "HRES-TEST-001"
        assert hr.resolution_type == "FEE_ADJUSTMENT"
        assert hr.outcome == "RESOLVED"
        assert hr.resolvable is True
        assert hr.source == "deterministic"

    def test_historical_resolution_nullable_fields(self, session):
        """Test that optional fields can be null."""
        hr = HistoricalResolution(
            id="HRES-NULL-001",
            case_id="CASE-001",
            resolution_type="UNKNOWN_UNRESOLVED",
            outcome="UNRESOLVED",
            exception_id=None,
            resolved_amount=None,
            difference_at_resolution=None,
            notes=None,
        )
        session.add(hr)
        session.flush()
        assert hr.exception_id is None
        assert hr.resolved_amount is None

    def test_historical_resolution_unique_id(self, session):
        """Test that historical resolution IDs must be unique."""
        h1 = HistoricalResolution(
            id="HRES-DUP", case_id="CASE-001",
            resolution_type="FEE_ADJUSTMENT", outcome="RESOLVED"
        )
        h2 = HistoricalResolution(
            id="HRES-DUP", case_id="CASE-002",
            resolution_type="TAX_ADJUSTMENT", outcome="RESOLVED"
        )
        session.add(h1)
        session.flush()
        session.add(h2)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_historical_resolution_source_values(self, session):
        """Test various source values."""
        sources = ["human", "deterministic", "ml", "agent"]
        for i, src in enumerate(sources):
            hr = HistoricalResolution(
                id=f"HRES-SRC-{i}", case_id=f"CASE-{i}",
                resolution_type="FEE_ADJUSTMENT", outcome="RESOLVED",
                source=src,
            )
            session.add(hr)
        session.flush()
        assert session.query(HistoricalResolution).count() == len(sources)

    def test_historical_resolution_default_source(self, session):
        """Test that default source is 'deterministic'."""
        hr = HistoricalResolution(
            id="HRES-DEFAULT-SRC", case_id="CASE-001",
            resolution_type="FEE_ADJUSTMENT", outcome="RESOLVED",
        )
        session.add(hr)
        session.flush()
        assert hr.source == "deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Model Relationship Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossModelRelationships:
    """Tests for relationships between different models."""

    def test_exception_to_evidence_link_chain(self, session):
        """Test full exception → evidence link → financial record chain."""
        # 1. Create exception
        exc = FinancialException(
            id="EXC-CHAIN-001", case_id="CASE-CHAIN-001", payment_id="PAY-CHAIN-001",
            batch_id="batch_001", expected_amount=100000, actual_amount=98000,
            difference=2000, exception_type="FEE_DIFFERENCE", status="OPEN",
            reconciliation_id="REC-CHAIN-001"
        )
        session.add(exc)
        session.flush()

        # 2. Create financial records
        fee = Fee(id="FEE-CHAIN-001", payment_id="PAY-CHAIN-001", amount=2000, fee_type="TRANSACTION")
        tax = Tax(id="TAX-CHAIN-001", payment_id="PAY-CHAIN-001", amount=1800, tax_type="GST")
        session.add(fee)
        session.add(tax)
        session.flush()

        # 3. Create evidence links
        el_fee = EvidenceLink(
            id="EL-CHAIN-001", exception_id="EXC-CHAIN-001", case_id="CASE-CHAIN-001",
            entity_type="FEE", entity_id="FEE-CHAIN-001", relationship="CALCULATION_COMPONENT"
        )
        el_tax = EvidenceLink(
            id="EL-CHAIN-002", exception_id="EXC-CHAIN-001", case_id="CASE-CHAIN-001",
            entity_type="TAX", entity_id="TAX-CHAIN-001", relationship="CALCULATION_COMPONENT"
        )
        session.add(el_fee)
        session.add(el_tax)
        session.flush()

        # 4. Verify chain
        links = session.query(EvidenceLink).filter_by(exception_id="EXC-CHAIN-001").all()
        assert len(links) == 2
        assert links[0].entity_type == "FEE"
        assert links[1].entity_type == "TAX"

    def test_reconciliation_result_to_exception_to_evidence(self, session):
        """Test full reconciliation → exception → evidence chain."""
        # ReconciliationResult
        rr = ReconciliationResult(
            id="REC-FULL-001", case_id="CASE-FULL-001", payment_id="PAY-FULL-001",
            merchant_id="MER-001", batch_id="batch_001", payment_amount=100000,
            total_refunds=0, total_fees=2000, total_taxes=18000, total_adjustments=0,
            expected_amount=80000, actual_amount=78000, difference=2000,
            match_status="EXCEPTION", exception_type="FEE_DIFFERENCE"
        )
        session.add(rr)
        session.flush()

        # Exception
        exc = FinancialException(
            id="EXC-FULL-001", case_id="CASE-FULL-001", payment_id="PAY-FULL-001",
            batch_id="batch_001", expected_amount=80000, actual_amount=78000,
            difference=2000, exception_type="FEE_DIFFERENCE", status="OPEN",
            reconciliation_id="REC-FULL-001"
        )
        session.add(exc)
        session.flush()

        # Evidence links
        el = EvidenceLink(
            id="EL-FULL-001", exception_id="EXC-FULL-001", case_id="CASE-FULL-001",
            entity_type="FEE", entity_id="FEE-CHAIN-001", relationship="CALCULATION_COMPONENT"
        )
        session.add(el)
        session.flush()

        # HistoricalResolution
        hr = HistoricalResolution(
            id="HRES-FULL-001", exception_id="EXC-FULL-001", case_id="CASE-FULL-001",
            resolution_type="FEE_ADJUSTMENT", outcome="RESOLVED",
            resolved_amount=2000, difference_at_resolution=2000,
            exception_type="FEE_DIFFERENCE", resolvable=True, source="deterministic"
        )
        session.add(hr)
        session.flush()

        # Verify the full chain
        assert session.query(ReconciliationResult).filter_by(case_id="CASE-FULL-001").count() == 1
        assert session.query(FinancialException).filter_by(case_id="CASE-FULL-001").count() == 1
        assert session.query(EvidenceLink).filter_by(case_id="CASE-FULL-001").count() == 1
        assert session.query(HistoricalResolution).filter_by(case_id="CASE-FULL-001").count() == 1

    def test_multiple_payments_evidence_links(self, session):
        """Test evidence links across multiple payments."""
        for i in range(5):
            exc = FinancialException(
                id=f"EXC-MULTI-{i}", case_id=f"CASE-{i}", payment_id=f"PAY-{i}",
                batch_id="batch_001", expected_amount=100000, actual_amount=95000,
                difference=5000, exception_type="FEE_DIFFERENCE", status="OPEN",
                reconciliation_id=f"REC-{i}"
            )
            session.add(exc)

            link = EvidenceLink(
                id=f"EL-MULTI-{i}", exception_id=f"EXC-MULTI-{i}", case_id=f"CASE-{i}",
                entity_type="FEE", entity_id=f"FEE-{i}", relationship="CALCULATION_COMPONENT"
            )
            session.add(link)

        session.flush()
        assert session.query(EvidenceLink).count() == 5
        assert session.query(FinancialException).count() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Verify that no ground truth labels leak into evidence models."""

    GROUND_TRUTH_FIELDS = [
        "true_exception_type",
        "true_resolution",
        "resolvable_as_ground_truth",
        "risk_category",
    ]

    def test_refund_no_ground_truth_fields(self):
        """Test that Refund model has no ground truth fields."""
        columns = [c.name for c in Refund.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"Refund has ground truth field: {field}"

    def test_fee_no_ground_truth_fields(self):
        """Test that Fee model has no ground truth fields."""
        columns = [c.name for c in Fee.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"Fee has ground truth field: {field}"

    def test_tax_no_ground_truth_fields(self):
        """Test that Tax model has no ground truth fields."""
        columns = [c.name for c in Tax.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"Tax has ground truth field: {field}"

    def test_adjustment_no_ground_truth_fields(self):
        """Test that Adjustment model has no ground truth fields."""
        columns = [c.name for c in Adjustment.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"Adjustment has ground truth field: {field}"

    def test_evidence_link_no_ground_truth_fields(self):
        """Test that EvidenceLink model has no ground truth fields."""
        columns = [c.name for c in EvidenceLink.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"EvidenceLink has ground truth field: {field}"

    def test_financial_exception_no_ground_truth_fields(self):
        """Test that FinancialException model has no ground truth fields."""
        columns = [c.name for c in FinancialException.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"FinancialException has ground truth field: {field}"

    def test_reconciliation_result_no_ground_truth_fields(self):
        """Test that ReconciliationResult model has no ground truth fields."""
        columns = [c.name for c in ReconciliationResult.__table__.columns]
        for field in self.GROUND_TRUTH_FIELDS:
            assert field not in columns, f"ReconciliationResult has ground truth field: {field}"


# ─────────────────────────────────────────────────────────────────────────────
# Amount Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAmountIntegrity:
    """Verify all financial amounts are integers (paise)."""

    def test_all_amounts_are_integer(self, session):
        """Test that all financial entity amounts are integers."""
        # Refund
        r = Refund(id="REF-INT-001", payment_id="PAY-001", amount=5000)
        session.add(r)

        # Fee
        f = Fee(id="FEE-INT-001", payment_id="PAY-001", amount=2000, fee_type="TDR")
        session.add(f)

        # Tax
        t = Tax(id="TAX-INT-001", payment_id="PAY-001", amount=1800, tax_type="GST")
        session.add(t)

        # Adjustment (positive)
        a1 = Adjustment(id="ADJ-INT-001", payment_id="PAY-001", amount=1000, adjustment_type="CREDIT")
        session.add(a1)

        # Adjustment (negative)
        a2 = Adjustment(id="ADJ-INT-002", payment_id="PAY-001", amount=-500, adjustment_type="DEBIT")
        session.add(a2)

        session.flush()

        assert isinstance(r.amount, int)
        assert isinstance(f.amount, int)
        assert isinstance(t.amount, int)
        assert isinstance(a1.amount, int)
        assert isinstance(a2.amount, int)

    def test_no_float_amounts(self, session):
        """Test that no amount is a float."""
        records = [
            Refund(id="REF-FLT-001", payment_id="PAY-001", amount=5000),
            Fee(id="FEE-FLT-001", payment_id="PAY-001", amount=2000, fee_type="TDR"),
            Tax(id="TAX-FLT-001", payment_id="PAY-001", amount=1800, tax_type="GST"),
            Adjustment(id="ADJ-FLT-001", payment_id="PAY-001", amount=1000, adjustment_type="CREDIT"),
        ]
        for r in records:
            session.add(r)
        session.flush()

        for r in records:
            assert not isinstance(r.amount, float), f"{r.__class__.__name__} has float amount"
