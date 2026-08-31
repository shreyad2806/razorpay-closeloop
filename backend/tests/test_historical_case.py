"""
Tests for Razorpay CloseLoop Phase 4E — Historical Case Store.

Tests cover:
- HistoricalCase schema validation
- FinancialContext integrity
- HistoricalEvidenceRef
- ResolutionOutcome/Origin enums
- HistoricalCaseStore CRUD
- Duplicate handling
- Retrieval by type/outcome
- Data integrity validation
- Ground truth separation
"""

import os
import pytest
from datetime import datetime

os.environ["DATABASE_URL"] = "sqlite:///test_historical_case.db"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.schemas.historical_case import (
    FinancialContext,
    HistoricalCase,
    HistoricalEvidenceRef,
    ResolutionOrigin,
    ResolutionOutcome,
)
from app.services.historical_case_store import (
    HistoricalCaseRecord,
    HistoricalCaseStore,
)


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


@pytest.fixture
def store(session):
    return HistoricalCaseStore(session)


def _make_case(
    case_id="CASE-001",
    exception_type="FEE_DIFFERENCE",
    resolution_type="FEE_ADJUSTMENT",
    resolution_outcome=ResolutionOutcome.SUCCESSFUL,
    difference=3000,
    evidence_refs=None,
    tags=None,
):
    """Create a minimal HistoricalCase for testing."""
    return HistoricalCase(
        case_id=case_id,
        exception_id="EXC-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        exception_type=exception_type,
        financial_context=FinancialContext(
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=difference,
            total_fees=3000,
        ),
        resolution_type=resolution_type,
        resolution_outcome=resolution_outcome,
        resolution_origin=ResolutionOrigin.DETERMINISTIC,
        resolved_amount=97000,
        evidence_refs=evidence_refs or [],
        supporting_evidence_count=len(evidence_refs) if evidence_refs else 0,
        tags=tags or ["fee", "known-pattern"],
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        resolved_at=datetime(2026, 1, 15, 10, 5, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalCaseSchema:
    def test_basic_creation(self):
        case = _make_case()
        assert case.case_id == "CASE-001"
        assert case.exception_type == "FEE_DIFFERENCE"
        assert case.resolution_type == "FEE_ADJUSTMENT"
        assert case.resolution_outcome == ResolutionOutcome.SUCCESSFUL

    def test_financial_context(self):
        case = _make_case()
        assert case.financial_context.payment_amount == 100000
        assert case.financial_context.expected_amount == 100000
        assert case.financial_context.actual_amount == 97000
        assert case.financial_context.difference == 3000

    def test_financial_integrity_valid(self):
        case = _make_case()
        errors = case.validate_financial_integrity()
        assert errors == []

    def test_financial_integrity_difference_mismatch(self):
        case = _make_case()
        case.financial_context.difference = 999  # Wrong difference
        errors = case.validate_financial_integrity()
        assert len(errors) == 1
        assert "difference mismatch" in errors[0]

    def test_financial_integrity_negative_payment(self):
        case = _make_case()
        case.financial_context.payment_amount = -100
        errors = case.validate_financial_integrity()
        assert any("payment_amount" in e for e in errors)

    def test_financial_integrity_negative_resolved(self):
        case = _make_case()
        case.resolved_amount = -500
        errors = case.validate_financial_integrity()
        assert any("resolved_amount" in e for e in errors)

    def test_to_retrieval_features(self):
        case = _make_case()
        features = case.to_retrieval_features()
        assert features["case_id"] == "CASE-001"
        assert features["exception_type"] == "FEE_DIFFERENCE"
        assert features["payment_amount"] == 100000
        assert features["difference"] == 3000
        assert "tags" in features

    def test_optional_fields(self):
        case = HistoricalCase(
            case_id="CASE-OPT",
            exception_id="EXC-OPT",
            payment_id="PAY-OPT",
            exception_type="EXACT_MATCH",
            financial_context=FinancialContext(
                payment_amount=50000,
                expected_amount=50000,
                actual_amount=50000,
                difference=0,
            ),
            resolution_type="NO_ACTION",
            resolution_outcome=ResolutionOutcome.SUCCESSFUL,
        )
        assert case.merchant_id is None
        assert case.resolved_amount is None
        assert case.evidence_refs == []
        assert case.tags == []
        assert case.created_at is not None

    def test_resolution_outcome_values(self):
        assert ResolutionOutcome.SUCCESSFUL.value == "SUCCESSFUL"
        assert ResolutionOutcome.UNSUCCESSFUL.value == "UNSUCCESSFUL"
        assert ResolutionOutcome.REVERSED.value == "REVERSED"
        assert ResolutionOutcome.MANUALLY_REVIEWED.value == "MANUALLY_REVIEWED"

    def test_resolution_origin_values(self):
        assert ResolutionOrigin.HUMAN.value == "HUMAN"
        assert ResolutionOrigin.DETERMINISTIC.value == "DETERMINISTIC"
        assert ResolutionOrigin.ML.value == "ML"
        assert ResolutionOrigin.AGENT.value == "AGENT"


class TestHistoricalEvidenceRef:
    def test_basic_creation(self):
        ref = HistoricalEvidenceRef(
            entity_type="FEE",
            entity_id="FEE-001",
            relationship="SUPPORTING_EVIDENCE",
            amount=3000,
        )
        assert ref.entity_type == "FEE"
        assert ref.entity_id == "FEE-001"
        assert ref.amount == 3000

    def test_in_case(self):
        refs = [
            HistoricalEvidenceRef(
                entity_type="FEE",
                entity_id="FEE-001",
                relationship="CALCULATION_COMPONENT",
                amount=3000,
            ),
            HistoricalEvidenceRef(
                entity_type="REFUND",
                entity_id="REF-001",
                relationship="SUPPORTING_EVIDENCE",
                amount=1500,
            ),
        ]
        case = _make_case(evidence_refs=refs)
        case.supporting_evidence_count = 2
        assert len(case.evidence_refs) == 2
        assert case.supporting_evidence_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Store Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalCaseStore:
    def test_store_and_retrieve(self, store):
        case = _make_case(case_id="CASE-S01")
        record = store.store(case)
        assert record is not None
        assert record.id == "CASE-S01"

        retrieved = store.get_by_case_id("CASE-S01")
        assert retrieved is not None
        assert retrieved.case_id == "CASE-S01"
        assert retrieved.exception_type == "FEE_DIFFERENCE"
        assert retrieved.financial_context.payment_amount == 100000

    def test_store_duplicate_returns_none(self, store):
        case = _make_case(case_id="CASE-DUP")
        record1 = store.store(case)
        assert record1 is not None

        record2 = store.store(case)
        assert record2 is None  # Duplicate

    def test_exists(self, store):
        assert not store.exists("CASE-EXIST-NO")
        store.store(_make_case(case_id="CASE-EXIST-YES"))
        assert store.exists("CASE-EXIST-YES")

    def test_count(self, store):
        initial = store.count()
        store.store(_make_case(case_id="CASE-CNT1"))
        store.store(_make_case(case_id="CASE-CNT2"))
        assert store.count() == initial + 2

    def test_get_by_exception_type(self, store):
        store.store(_make_case(case_id="CASE-EXC1", exception_type="FEE_DIFFERENCE"))
        store.store(_make_case(case_id="CASE-EXC2", exception_type="EXACT_MATCH"))
        store.store(_make_case(case_id="CASE-EXC3", exception_type="FEE_DIFFERENCE"))

        fee_cases = store.get_by_exception_type("FEE_DIFFERENCE")
        assert len(fee_cases) >= 2
        assert all(c.exception_type == "FEE_DIFFERENCE" for c in fee_cases)

    def test_get_by_resolution_type(self, store):
        store.store(_make_case(case_id="CASE-RES1", resolution_type="FEE_ADJUSTMENT"))
        store.store(_make_case(case_id="CASE-RES2", resolution_type="NO_ACTION"))

        fee_resolved = store.get_by_resolution_type("FEE_ADJUSTMENT")
        assert len(fee_resolved) >= 1
        assert all(c.resolution_type == "FEE_ADJUSTMENT" for c in fee_resolved)

    def test_get_by_outcome(self, store):
        store.store(
            _make_case(
                case_id="CASE-OUT1",
                resolution_outcome=ResolutionOutcome.SUCCESSFUL,
            )
        )
        store.store(
            _make_case(
                case_id="CASE-OUT2",
                resolution_outcome=ResolutionOutcome.UNSUCCESSFUL,
            )
        )

        successful = store.get_by_outcome(ResolutionOutcome.SUCCESSFUL)
        assert len(successful) >= 1
        assert all(
            c.resolution_outcome == ResolutionOutcome.SUCCESSFUL for c in successful
        )

    def test_list_all(self, store):
        store.store(_make_case(case_id="CASE-LA1"))
        store.store(_make_case(case_id="CASE-LA2"))
        cases = store.list_all()
        assert isinstance(cases, list)
        assert len(cases) >= 2

    def test_financial_integrity_preserved(self, store):
        case = _make_case(case_id="CASE-FIN")
        store.store(case)
        retrieved = store.get_by_case_id("CASE-FIN")
        assert retrieved.financial_context.payment_amount == 100000
        assert retrieved.financial_context.expected_amount == 100000
        assert retrieved.financial_context.actual_amount == 97000
        assert retrieved.financial_context.difference == 3000

    def test_evidence_refs_preserved(self, store):
        refs = [
            HistoricalEvidenceRef(
                entity_type="FEE",
                entity_id="FEE-PRES",
                relationship="CALCULATION_COMPONENT",
                amount=3000,
            ),
        ]
        case = _make_case(case_id="CASE-EVR", evidence_refs=refs)
        case.supporting_evidence_count = 1
        store.store(case)
        retrieved = store.get_by_case_id("CASE-EVR")
        assert len(retrieved.evidence_refs) == 1
        assert retrieved.evidence_refs[0].entity_id == "FEE-PRES"
        assert retrieved.evidence_refs[0].amount == 3000

    def test_tags_preserved(self, store):
        case = _make_case(case_id="CASE-TAG", tags=["fee", "known", "v1"])
        store.store(case)
        retrieved = store.get_by_case_id("CASE-TAG")
        assert retrieved.tags == ["fee", "known", "v1"]

    def test_resolution_metadata_preserved(self, store):
        case = _make_case(case_id="CASE-META")
        case.resolution_metadata = {"key": "value", "count": 42}
        store.store(case)
        retrieved = store.get_by_case_id("CASE-META")
        assert retrieved.resolution_metadata == {"key": "value", "count": 42}

    def test_store_rejects_invalid_financial(self, store):
        case = _make_case(case_id="CASE-BAD")
        case.financial_context.difference = 999  # Mismatch
        with pytest.raises(ValueError, match="Financial integrity"):
            store.store(case)

    def test_timestamps_preserved(self, store):
        ts = datetime(2026, 6, 15, 12, 0, 0)
        case = _make_case(case_id="CASE-TS")
        case.created_at = ts
        case.resolved_at = ts
        store.store(case)
        retrieved = store.get_by_case_id("CASE-TS")
        assert retrieved.created_at == ts
        assert retrieved.resolved_at == ts

    def test_resolution_origin_preserved(self, store):
        case = _make_case(case_id="CASE-ORI")
        case.resolution_origin = ResolutionOrigin.ML
        store.store(case)
        retrieved = store.get_by_case_id("CASE-ORI")
        assert retrieved.resolution_origin == ResolutionOrigin.ML

    def test_quality_indicators_preserved(self, store):
        case = _make_case(case_id="CASE-QLT")
        case.exception_type_confidence = 0.95
        case.evidence_coverage = 0.87
        store.store(case)
        retrieved = store.get_by_case_id("CASE-QLT")
        assert retrieved.exception_type_confidence == 0.95
        assert retrieved.evidence_coverage == 0.87

    def test_merchant_id_optional(self, store):
        case = _make_case(case_id="CASE-MNO")
        case.merchant_id = None
        store.store(case)
        retrieved = store.get_by_case_id("CASE-MNO")
        assert retrieved.merchant_id is None

    def test_get_nonexistent_returns_none(self, store):
        result = store.get_by_case_id("CASE-NONEXISTENT")
        assert result is None

    def test_limit_on_queries(self, store):
        """Ensure limit parameter works."""
        for i in range(5):
            store.store(_make_case(case_id=f"CASE-LIM{i:03d}"))
        results = store.get_by_exception_type("FEE_DIFFERENCE", limit=2)
        assert len(results) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    def test_no_ground_truth_fields_in_historical_case(self):
        """HistoricalCase must not contain ground-truth evaluation labels."""
        import inspect

        source = inspect.getsource(HistoricalCase)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source

    def test_no_ground_truth_fields_in_store(self):
        """HistoricalCaseStore must not use ground truth."""
        import inspect

        source = inspect.getsource(HistoricalCaseStore)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "ground_truth" not in source

    def test_no_ground_truth_in_record_model(self):
        """HistoricalCaseRecord must not store ground truth."""
        import inspect

        source = inspect.getsource(HistoricalCaseRecord)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "ground_truth" not in source

    def test_exception_type_is_deterministic_not_label(self):
        """exception_type in HistoricalCase is the engine's classification, not ground truth."""
        case = _make_case()
        # The exception_type is "FEE_DIFFERENCE" — this represents the engine's classification
        # It is NOT labeled as true_exception_type
        assert case.exception_type == "FEE_DIFFERENCE"
        assert not hasattr(case, "true_exception_type")


# ─────────────────────────────────────────────────────────────────────────────
# Financial Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialIntegrity:
    def test_all_amounts_are_integers(self):
        case = _make_case()
        fc = case.financial_context
        assert isinstance(fc.payment_amount, int)
        assert isinstance(fc.expected_amount, int)
        assert isinstance(fc.actual_amount, int)
        assert isinstance(fc.difference, int)
        assert isinstance(fc.total_refunds, int)
        assert isinstance(fc.total_fees, int)
        assert isinstance(fc.total_taxes, int)
        assert isinstance(fc.total_adjustments, int)

    def test_no_floating_point_amounts(self):
        """Ensure no float amounts sneak in."""
        fc = FinancialContext(
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
        )
        # All amounts should be int, not float
        assert type(fc.payment_amount) is int
        assert type(fc.difference) is int

    def test_difference_consistency(self):
        fc = FinancialContext(
            payment_amount=100000,
            expected_amount=95000,
            actual_amount=92000,
            difference=3000,
        )
        assert fc.expected_amount - fc.actual_amount == fc.difference

    def test_zero_amounts_allowed(self):
        case = _make_case(case_id="CASE-ZERO")
        case.financial_context = FinancialContext(
            payment_amount=0,
            expected_amount=0,
            actual_amount=0,
            difference=0,
        )
        errors = case.validate_financial_integrity()
        assert errors == []
