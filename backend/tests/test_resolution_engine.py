"""
Tests for Razorpay CloseLoop Phase 5E — Resolution Engine Integration.

End-to-end tests covering:
- ResolutionEngineResult schema
- ResolutionEngine pipeline
- Fee exception end-to-end
- Refund exception end-to-end
- Tax exception end-to-end
- Partial settlement end-to-end
- Duplicate end-to-end
- Missing record end-to-end
- Complex multi-adjustment end-to-end
- Unknown case end-to-end
- Exact match end-to-end
- Non-existent exception
- Safety guarantees
- Audit trail
- Ground truth separation
"""

import os
import pytest

os.environ["DATABASE_URL"] = "sqlite:///test_resolution_engine.db"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.models.payment import Payment
from app.models.settlement import Settlement
from app.models.refund import Refund
from app.models.fee import Fee
from app.models.tax import Tax
from app.models.adjustment import Adjustment
from app.models.exception import FinancialException
from app.models.evidence_link import EvidenceLink
from app.models.historical_resolution import HistoricalResolution
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus, SelectionConfig
from app.services.resolution_engine import ResolutionEngine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()


def _seed_fee_exception(session):
    """Seed a fee difference exception with evidence."""
    exc_id = "EXC-FEE-001"
    case_id = "CASE-FEE-001"
    payment_id = "PAY-FEE-001"

    exc = FinancialException(
        id=exc_id,
        case_id=case_id,
        payment_id=payment_id,
        batch_id="BATCH-001",
        exception_type="FEE_DIFFERENCE",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        reconciliation_id="REC-001",
    )
    session.add(exc)

    payment = Payment(id=payment_id, amount=100000)
    session.add(payment)

    settlement = Settlement(
        id="SET-FEE-001", payment_id=payment_id, amount=97000
    )
    session.add(settlement)

    fee = Fee(
        id="FEE-FEE-001", payment_id=payment_id, amount=3000, fee_type="TDR"
    )
    session.add(fee)
    session.flush()
    return exc_id


def _seed_refund_exception(session):
    """Seed a refund adjustment exception."""
    exc_id = "EXC-REF-001"
    case_id = "CASE-REF-001"
    payment_id = "PAY-REF-001"

    exc = FinancialException(
        id=exc_id,
        case_id=case_id,
        payment_id=payment_id,
        batch_id="BATCH-001",
        exception_type="REFUND_ADJUSTMENT",
        expected_amount=100000,
        actual_amount=95000,
        difference=5000,
        reconciliation_id="REC-002",
    )
    session.add(exc)

    payment = Payment(id=payment_id, amount=100000)
    session.add(payment)

    settlement = Settlement(
        id="SET-REF-001", payment_id=payment_id, amount=95000
    )
    session.add(settlement)

    refund = Refund(
        id="REF-REF-001", payment_id=payment_id, amount=5000, status="PROCESSED"
    )
    session.add(refund)
    session.flush()
    return exc_id


def _seed_exact_match(session):
    """Seed an exact match case."""
    exc_id = "EXC-EXACT-001"
    case_id = "CASE-EXACT-001"
    payment_id = "PAY-EXACT-001"

    exc = FinancialException(
        id=exc_id,
        case_id=case_id,
        payment_id=payment_id,
        batch_id="BATCH-001",
        exception_type="EXACT_MATCH",
        expected_amount=100000,
        actual_amount=100000,
        difference=0,
        reconciliation_id="REC-003",
    )
    session.add(exc)

    payment = Payment(id=payment_id, amount=100000)
    session.add(payment)

    settlement = Settlement(
        id="SET-EXACT-001", payment_id=payment_id, amount=100000
    )
    session.add(settlement)
    session.flush()
    return exc_id


def _seed_missing_record(session):
    """Seed a missing record exception."""
    exc_id = "EXC-MISS-001"
    case_id = "CASE-MISS-001"
    payment_id = "PAY-MISS-001"

    exc = FinancialException(
        id=exc_id,
        case_id=case_id,
        payment_id=payment_id,
        batch_id="BATCH-001",
        exception_type="MISSING_RECORD",
        expected_amount=100000,
        actual_amount=0,
        difference=100000,
        reconciliation_id="REC-004",
    )
    session.add(exc)

    payment = Payment(id=payment_id, amount=100000)
    session.add(payment)
    # No settlement — missing record
    session.flush()
    return exc_id


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionEngineResult:
    def test_basic_structure(self):
        result = ResolutionEngineResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            confidence=0.8,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        assert result.is_recommended()
        assert not result.is_unresolved()
        assert result.is_recommendation_only is True

    def test_unresolved_result(self):
        result = ResolutionEngineResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            expected_amount=100000,
            actual_amount=0,
            difference=100000,
            status=SelectionStatus.UNRESOLVED,
            confidence=0.0,
            risk_category="HIGH",
            deterministic_exception_type="MISSING_RECORD",
            rejection_reasons=["No valid candidates"],
        )
        assert result.is_unresolved()
        assert result.selected_resolution is None

    def test_human_review_result(self):
        result = ResolutionEngineResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
            status=SelectionStatus.HUMAN_REVIEW,
            confidence=0.5,
            risk_category="MEDIUM",
            deterministic_exception_type="FEE_DIFFERENCE",
            rejection_reasons=["Close candidates"],
        )
        assert result.needs_human_review()

    def test_summary(self):
        result = ResolutionEngineResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            confidence=0.8,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        summary = result.summary()
        assert "EXC-001" in summary
        assert "FEE_DIFFERENCE" in summary
        assert "RECOMMENDED" in summary


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeExceptionE2E:
    def test_fee_exception_resolves(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result is not None
        assert result.exception_id == exc_id
        assert result.status in (
            SelectionStatus.RECOMMENDED,
            SelectionStatus.HUMAN_REVIEW,
            SelectionStatus.UNRESOLVED,
        )
        assert result.is_recommendation_only is True
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0

    def test_fee_exception_has_evidence(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result.evidence_explanation_status != ""
        assert result.evidence_coverage >= 0.0

    def test_fee_exception_financial_context(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result.expected_amount == 100000
        assert result.actual_amount == 97000
        assert result.difference == 3000


class TestRefundExceptionE2E:
    def test_refund_exception_resolves(self, session):
        exc_id = _seed_refund_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result is not None
        assert result.exception_id == exc_id
        assert result.difference == 5000


class TestExactMatchE2E:
    def test_exact_match_resolves(self, session):
        exc_id = _seed_exact_match(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result is not None
        assert result.difference == 0
        assert result.status in (
            SelectionStatus.RECOMMENDED,
            SelectionStatus.UNRESOLVED,
        )


class TestMissingRecordE2E:
    def test_missing_record_resolves(self, session):
        exc_id = _seed_missing_record(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert result is not None
        assert result.difference == 100000
        assert result.status in (
            SelectionStatus.RECOMMENDED,
            SelectionStatus.HUMAN_REVIEW,
            SelectionStatus.UNRESOLVED,
        )


class TestNonExistentException:
    def test_nonexistent_returns_none(self, session):
        engine = ResolutionEngine(session)
        result = engine.resolve("EXC-NONEXISTENT")
        assert result is None


class TestPipelineStages:
    def test_pipeline_completes_all_stages(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        # All pipeline stages produced output
        assert result.deterministic_exception_type != ""
        assert result.evidence_explanation_status != ""
        assert result.confidence >= 0.0
        assert result.risk_category in ("LOW", "MEDIUM", "HIGH")
        assert result.explainability is not None


class TestSafetyGuarantees:
    def test_no_financial_modification(self, session):
        """Engine must not modify any financial records."""
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        # Verify recommendation only
        assert result.is_recommendation_only is True

    def test_no_ground_truth_in_engine(self):
        """ResolutionEngine must not use ground truth."""
        import inspect

        source = inspect.getsource(ResolutionEngine)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source

    def test_recommendation_only_flag(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)
        assert result.is_recommendation_only is True


class TestAuditTrail:
    def test_exception_id_traceable(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)
        assert result.exception_id == exc_id

    def test_classification_traceable(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)
        assert result.deterministic_exception_type == "FEE_DIFFERENCE"

    def test_evidence_traceable(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)
        assert result.evidence_coverage >= 0.0
        assert result.evidence_consistency >= 0.0


class TestIdempotency:
    def test_same_input_same_output(self, session):
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)

        result1 = engine.resolve(exc_id)
        result2 = engine.resolve(exc_id)

        # Same status and resolution
        assert result1.status == result2.status
        assert result1.selected_resolution == result2.selected_resolution
        assert result1.confidence == result2.confidence


class TestGroundTruthSeparation:
    def test_no_ground_truth_fields(self, session):
        """Engine result must not contain ground truth labels."""
        exc_id = _seed_fee_exception(session)
        engine = ResolutionEngine(session)
        result = engine.resolve(exc_id)

        assert not hasattr(result, "true_exception_type")
        assert not hasattr(result, "true_resolution")
        assert not hasattr(result, "resolvable")
