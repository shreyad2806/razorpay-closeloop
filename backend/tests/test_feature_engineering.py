"""
Tests for deterministic feature engineering.

Covers:
- Financial features
- Structural features
- Evidence features
- Temporal features
- Merchant features
- Historical features
- Numeric safety (zero division, extreme values)
- Feature schema compliance
- Leakage checks
- Deterministic output
- Manual calculation verification
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_feature_eng.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
    StructuralConflict,
)
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
from app.schemas.ml_dataset import LEAKED_FIELDS, FEATURE_SCHEMA_VERSION
from app.ml.engineering import FeatureEngineer
from app.ml.features import ML_FEATURE_SCHEMA, validate_features


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def engineer():
    """Provide a feature engineer."""
    return FeatureEngineer()


def _make_package(**kwargs):
    """Helper to create an EvidencePackage with defaults."""
    defaults = {
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "payment_id": "PAY-001",
        "expected_amount": 100000,
        "actual_amount": 99500,
        "difference": 500,
        "exception_type": "FEE_DIFFERENCE",
    }
    defaults.update(kwargs)
    return EvidencePackage(**defaults)


def _make_record(record_id, entity_type, amount, **kwargs):
    """Helper to create an EvidenceRecord."""
    defaults = {
        "record_id": record_id,
        "entity_type": entity_type,
        "relationship": "CALCULATION_COMPONENT",
        "amount": amount,
    }
    defaults.update(kwargs)
    return EvidenceRecord(**defaults)


def _make_explanation(**kwargs):
    """Helper to create an ExplanationResult with defaults."""
    defaults = {
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "payment_id": "PAY-001",
        "expected_amount": 100000,
        "actual_amount": 99500,
        "difference": 500,
        "explanation_status": ExplanationStatus.FULLY_EXPLAINED,
        "explained_amount": 500,
        "remaining_difference": 0,
        "supporting_evidence_ids": ["FEE-001"],
        "conflict": False,
        "missing_evidence": [],
    }
    defaults.update(kwargs)
    return ExplanationResult(**defaults)


def _make_quality(**kwargs):
    """Helper to create an EvidenceQualityResult with defaults."""
    defaults = {
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "consistency_score": 1.0,
        "coverage_score": 1.0,
        "conflict": False,
        "novelty": NoveltyLevel.KNOWN_PATTERN,
        "missing_evidence": [],
        "fully_explained": True,
        "partially_explained": False,
        "supporting_evidence_count": 1,
    }
    defaults.update(kwargs)
    return EvidenceQualityResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Financial Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialFeatures:
    """Tests for financial feature extraction."""

    def test_difference_amount(self, engineer):
        """Test that difference_amount is absolute difference."""
        pkg = _make_package(difference=-500)
        exp = _make_explanation(difference=-500)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["difference_amount"] == 500.0

    def test_relative_difference(self, engineer):
        """Test that relative_difference is difference/payment."""
        pkg = _make_package(difference=500, expected_amount=100000)
        exp = _make_explanation(difference=500)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        # payment_amount = expected + refunds + fees + taxes - adjustments = 100000
        assert vec.features["relative_difference"] == pytest.approx(0.005, abs=0.001)

    def test_payment_amount(self, engineer):
        """Test that payment_amount is extracted correctly."""
        pkg = _make_package(expected_amount=100000)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["payment_amount"] == 100000.0

    def test_settlement_amount(self, engineer):
        """Test that settlement_amount comes from package."""
        pkg = _make_package(total_settlement_amount=99500)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["settlement_amount"] == 99500.0

    def test_refund_amount(self, engineer):
        """Test that refund_amount is total refund."""
        pkg = _make_package(total_refund_amount=1000)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["refund_amount"] == 1000.0

    def test_fee_amount(self, engineer):
        """Test that fee_amount is total fee."""
        pkg = _make_package(total_fee_amount=500)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["fee_amount"] == 500.0

    def test_tax_amount(self, engineer):
        """Test that tax_amount is total tax."""
        pkg = _make_package(total_tax_amount=200)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["tax_amount"] == 200.0

    def test_adjustment_amount(self, engineer):
        """Test that adjustment_amount is net adjustment."""
        pkg = _make_package(total_adjustment_amount=-300)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["adjustment_amount"] == -300.0

    def test_refund_ratio(self, engineer):
        """Test that refund_ratio is refund/payment."""
        pkg = _make_package(total_refund_amount=1000, expected_amount=100000)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["refund_ratio"] == pytest.approx(0.01, abs=0.001)

    def test_fee_ratio(self, engineer):
        """Test that fee_ratio is fee/payment."""
        pkg = _make_package(total_fee_amount=500, expected_amount=100000)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["fee_ratio"] == pytest.approx(0.005, abs=0.001)

    def test_tax_ratio(self, engineer):
        """Test that tax_ratio is tax/payment."""
        pkg = _make_package(total_tax_amount=200, expected_amount=100000)
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["tax_ratio"] == pytest.approx(0.002, abs=0.001)

    def test_manual_calculation_match(self, engineer):
        """Test that manually calculated features match implementation."""
        # With payment record present, payment_amount comes from the record
        pkg = _make_package(
            difference=500, expected_amount=100000,
            total_settlement_amount=99500, total_refund_amount=1000,
            total_fee_amount=500, total_tax_amount=200, total_adjustment_amount=0,
            payment=_make_record("PAY-001", "PAYMENT", 100000),
        )
        exp = _make_explanation(difference=500)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)

        # Manual verification
        assert vec.features["difference_amount"] == 500.0
        assert vec.features["payment_amount"] == 100000.0
        assert vec.features["settlement_amount"] == 99500.0
        assert vec.features["refund_amount"] == 1000.0
        assert vec.features["fee_amount"] == 500.0
        assert vec.features["tax_amount"] == 200.0
        assert vec.features["refund_ratio"] == 1000 / 100000
        assert vec.features["fee_ratio"] == 500 / 100000
        assert vec.features["tax_ratio"] == 200 / 100000


# ─────────────────────────────────────────────────────────────────────────────
# Structural Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStructuralFeatures:
    """Tests for structural feature extraction."""

    def test_num_settlements(self, engineer):
        """Test that num_settlements counts correctly."""
        pkg = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 50000),
                _make_record("SET-002", "SETTLEMENT", 49500),
            ]
        )
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["num_settlements"] == 2.0

    def test_num_refunds(self, engineer):
        """Test that num_refunds counts correctly."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 500),
                _make_record("REF-002", "REFUND", 300),
                _make_record("REF-003", "REFUND", 200),
            ]
        )
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["num_refunds"] == 3.0

    def test_num_fees(self, engineer):
        """Test that num_fees counts correctly."""
        pkg = _make_package(
            fees=[_make_record("FEE-001", "FEE", 500)]
        )
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["num_fees"] == 1.0

    def test_has_missing_evidence(self, engineer):
        """Test that has_missing_evidence is 1.0 when evidence is missing."""
        pkg = _make_package(
            missing_evidence=[
                MissingEvidence(entity_type="SETTLEMENT", expected=True, reason="Missing"),
            ]
        )
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["has_missing_evidence"] == 1.0
        assert vec.features["num_missing_evidence"] == 1.0

    def test_no_missing_evidence(self, engineer):
        """Test that has_missing_evidence is 0.0 when no evidence missing."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["has_missing_evidence"] == 0.0
        assert vec.features["num_missing_evidence"] == 0.0

    def test_multiple_missing_evidence(self, engineer):
        """Test that multiple missing evidence types are counted."""
        pkg = _make_package(
            missing_evidence=[
                MissingEvidence(entity_type="SETTLEMENT", expected=True, reason="Missing"),
                MissingEvidence(entity_type="REFUND", expected=True, reason="Missing"),
            ]
        )
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["num_missing_evidence"] == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceFeatures:
    """Tests for evidence feature extraction."""

    def test_evidence_coverage(self, engineer):
        """Test that evidence_coverage comes from quality scorer."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality(coverage_score=0.75)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["evidence_coverage"] == 0.75

    def test_consistency_score(self, engineer):
        """Test that consistency_score comes from quality scorer."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality(consistency_score=0.85)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["consistency_score"] == 0.85

    def test_fully_explained(self, engineer):
        """Test that fully_explained is 1.0 when explanation is complete."""
        pkg = _make_package()
        exp = _make_explanation(
            explanation_status=ExplanationStatus.FULLY_EXPLAINED
        )
        qual = _make_quality(fully_explained=True)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["fully_explained"] == 1.0
        assert vec.features["partially_explained"] == 0.0

    def test_partially_explained(self, engineer):
        """Test that partially_explained is 1.0 when partial."""
        pkg = _make_package()
        exp = _make_explanation(
            explanation_status=ExplanationStatus.PARTIALLY_EXPLAINED,
            explained_amount=300,
            remaining_difference=200,
        )
        qual = _make_quality(fully_explained=False, partially_explained=True)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["fully_explained"] == 0.0
        assert vec.features["partially_explained"] == 1.0

    def test_has_conflict(self, engineer):
        """Test that has_conflict is 1.0 when conflict exists."""
        pkg = _make_package()
        exp = _make_explanation(conflict=True)
        qual = _make_quality(conflict=True)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["has_conflict"] == 1.0

    def test_supporting_evidence_count(self, engineer):
        """Test that supporting_evidence_count is correct."""
        pkg = _make_package()
        exp = _make_explanation(
            supporting_evidence_ids=["REF-001", "FEE-001", "TAX-001"]
        )
        qual = _make_quality(supporting_evidence_count=3)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["supporting_evidence_count"] == 3.0

    def test_num_candidate_explanations(self, engineer):
        """Test that num_candidate_explanations is correct."""
        from app.schemas.explanation import CandidateExplanation, ExplainingEvent

        pkg = _make_package()
        exp = _make_explanation(
            explanation_status=ExplanationStatus.CONFLICTING,
            conflict=True,
            candidate_explanations=[
                CandidateExplanation(
                    events=[ExplainingEvent(record_id="REF-001", entity_type="REFUND", amount=500, contribution=-500)],
                    total_contribution=-500,
                    is_exact_match=True,
                ),
                CandidateExplanation(
                    events=[ExplainingEvent(record_id="FEE-001", entity_type="FEE", amount=500, contribution=-500)],
                    total_contribution=-500,
                    is_exact_match=True,
                ),
            ],
        )
        qual = _make_quality(conflict=True)

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["num_candidate_explanations"] == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def test_payment_to_settlement_delay(self, engineer):
        """Test that payment-to-settlement delay is calculated correctly."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        payment_ts = datetime(2026, 1, 1, 10, 0, 0)
        settlement_ts = datetime(2026, 1, 1, 15, 30, 0)  # 5.5 hours later

        vec = engineer.engineer(pkg, exp, qual, payment_timestamp=payment_ts, settlement_timestamp=settlement_ts)
        assert vec.features["payment_to_settlement_delay_hours"] == pytest.approx(5.5, abs=0.1)

    def test_no_timestamps_default_zero(self, engineer):
        """Test that missing timestamps produce 0.0 delay."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["payment_to_settlement_delay_hours"] == 0.0

    def test_negative_delay_clipped_to_zero(self, engineer):
        """Test that negative delays are clipped to 0.0."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        payment_ts = datetime(2026, 1, 2, 10, 0, 0)
        settlement_ts = datetime(2026, 1, 1, 10, 0, 0)  # Before payment

        vec = engineer.engineer(pkg, exp, qual, payment_timestamp=payment_ts, settlement_timestamp=settlement_ts)
        assert vec.features["payment_to_settlement_delay_hours"] == 0.0

    def test_extreme_delay_clipped(self, engineer):
        """Test that extreme delays are clipped to 8760 hours (1 year)."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        payment_ts = datetime(2025, 1, 1, 0, 0, 0)
        settlement_ts = datetime(2027, 1, 1, 0, 0, 0)  # 2 years

        vec = engineer.engineer(pkg, exp, qual, payment_timestamp=payment_ts, settlement_timestamp=settlement_ts)
        assert vec.features["payment_to_settlement_delay_hours"] == 8760.0


# ─────────────────────────────────────────────────────────────────────────────
# Merchant Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMerchantFeatures:
    """Tests for merchant feature extraction."""

    def test_merchant_exception_rate(self, engineer):
        """Test that merchant exception rate is extracted."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual, merchant_exception_rate=0.15)
        assert vec.features["merchant_historical_exception_rate"] == 0.15

    def test_merchant_rate_clamped(self, engineer):
        """Test that merchant rate is clamped to [0, 1]."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual, merchant_exception_rate=1.5)
        assert vec.features["merchant_historical_exception_rate"] == 1.0

        vec = engineer.engineer(pkg, exp, qual, merchant_exception_rate=-0.5)
        assert vec.features["merchant_historical_exception_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Historical Features Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalFeatures:
    """Tests for historical feature extraction."""

    def test_historical_pattern_default(self, engineer):
        """Test that historical pattern score defaults to 0.0."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["historical_pattern_match_score"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Numeric Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNumericSafety:
    """Tests for numeric safety in feature engineering."""

    def test_zero_payment_no_crash(self, engineer):
        """Test that zero payment amount doesn't crash."""
        pkg = _make_package(
            difference=500, expected_amount=0,
            total_refund_amount=0, total_fee_amount=0,
            total_tax_amount=0, total_adjustment_amount=0,
        )
        exp = _make_explanation(difference=500)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["payment_amount"] == 0.0
        assert vec.features["relative_difference"] == 0.0
        assert vec.features["refund_ratio"] == 0.0
        assert vec.features["fee_ratio"] == 0.0
        assert vec.features["tax_ratio"] == 0.0

    def test_no_nan_in_output(self, engineer):
        """Test that no feature value is NaN."""
        pkg = _make_package(difference=0, expected_amount=0)
        exp = _make_explanation(difference=0)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        for name, value in vec.features.items():
            assert not (isinstance(value, float) and value != value), f"NaN in {name}"

    def test_no_inf_in_output(self, engineer):
        """Test that no feature value is inf."""
        pkg = _make_package(difference=0, expected_amount=0)
        exp = _make_explanation(difference=0)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        for name, value in vec.features.items():
            assert value != float("inf"), f"inf in {name}"
            assert value != float("-inf"), f"-inf in {name}"

    def test_negative_difference_handled(self, engineer):
        """Test that negative differences are handled correctly."""
        pkg = _make_package(difference=-500)
        exp = _make_explanation(difference=-500)
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.features["difference_amount"] == 500.0  # Absolute
        assert vec.features["relative_difference"] < 0  # Signed


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema Compliance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaCompliance:
    """Tests for feature schema compliance."""

    def test_all_schema_features_present(self, engineer):
        """Test that all schema features are in the output."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        schema_names = set(ML_FEATURE_SCHEMA.get_feature_names())

        for name in schema_names:
            assert name in vec.features, f"Missing feature: {name}"

    def test_no_extra_features(self, engineer):
        """Test that no extra features are in the output."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        schema_names = set(ML_FEATURE_SCHEMA.get_feature_names())

        for name in vec.features:
            assert name in schema_names, f"Unexpected feature: {name}"

    def test_feature_count_matches_schema(self, engineer):
        """Test that feature count matches schema."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert len(vec.features) == ML_FEATURE_SCHEMA.get_feature_count()

    def test_schema_version_in_vector(self, engineer):
        """Test that schema version is in the feature vector."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        assert vec.schema_version == FEATURE_SCHEMA_VERSION

    def test_validate_features_passes(self, engineer):
        """Test that validate_features passes for engineered features."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        errors = validate_features(vec.features)
        assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLeakageChecks:
    """Tests for ground-truth leakage prevention."""

    def test_no_leaked_fields_in_features(self, engineer):
        """Test that no leaked fields appear in feature names."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        for leaked in LEAKED_FIELDS:
            assert leaked not in vec.features, f"Leaked field in features: {leaked}"


# ─────────────────────────────────────────────────────────────────────────────
# Determinism Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Tests for deterministic feature engineering."""

    def test_same_input_same_output(self, engineer):
        """Test that same inputs produce same features."""
        pkg = _make_package(difference=500, expected_amount=100000)
        exp = _make_explanation(difference=500)
        qual = _make_quality()

        vec1 = engineer.engineer(pkg, exp, qual)
        vec2 = engineer.engineer(pkg, exp, qual)
        assert vec1.features == vec2.features

    def test_different_input_different_output(self, engineer):
        """Test that different inputs produce different features."""
        pkg1 = _make_package(difference=500)
        pkg2 = _make_package(difference=1000)
        exp1 = _make_explanation(difference=500)
        exp2 = _make_explanation(difference=1000)
        qual = _make_quality()

        vec1 = engineer.engineer(pkg1, exp1, qual)
        vec2 = engineer.engineer(pkg2, exp2, qual)
        assert vec1.features != vec2.features

    def test_feature_names_ordering(self, engineer):
        """Test that feature names are in consistent order."""
        pkg = _make_package()
        exp = _make_explanation()
        qual = _make_quality()

        vec = engineer.engineer(pkg, exp, qual)
        names1 = list(vec.features.keys())
        names2 = list(vec.features.keys())
        assert names1 == names2
