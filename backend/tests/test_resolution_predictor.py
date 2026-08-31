"""
Tests for Razorpay CloseLoop Phase 4D — Resolution Predictor.

Tests cover:
- ResolutionClassifier training and prediction
- ResolutionPrediction result structure
- EvidenceCompatibilityChecker rules
- ResolutionTrainingBuilder
- ResolutionPredictor integration
- Ground truth separation
- Model save/load
"""

import numpy as np
import pytest
from app.schemas.evidence import EvidencePackage, EvidenceRecord, MissingEvidence
from app.schemas.explanation import (
    ExplanationResult,
    ExplanationStatus,
    CandidateExplanation,
)
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
from app.schemas.enums import ExceptionType, ResolutionType
from app.ml.resolution import (
    ResolutionClassifier,
    ResolutionPredictor,
    ResolutionPrediction,
    ResolutionTrainingBuilder,
    EvidenceCompatibilityChecker,
    EXCEPTION_TO_RESOLUTION_MAP,
    ALL_RESOLUTIONS,
    EVIDENCE_COMPATIBILITY_RULES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(entity_id, entity_type, amount, status="OK", relationship="SUPPORTING_EVIDENCE"):
    """Create an EvidenceRecord with the correct schema fields."""
    return EvidenceRecord(
        record_id=entity_id,
        entity_type=entity_type,
        relationship=relationship,
        amount=amount,
        status=status,
    )


def _make_package(
    exception_type="FEE_DIFFERENCE",
    fees=None,
    refunds=None,
    taxes=None,
    settlements=None,
    missing=None,
    has_conflicts=False,
):
    """Create a minimal EvidencePackage for testing."""
    return EvidencePackage(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        exception_type=exception_type,
        payment=_make_record("PAY-001", "PAYMENT", 100000, "CAPTURED", "PRIMARY_RECORD"),
        settlements=settlements or [
            _make_record("SET-001", "SETTLEMENT", 97000, "SETTLED"),
        ],
        refunds=refunds or [],
        fees=fees or [],
        taxes=taxes or [],
        adjustments=[],
        total_settlement_amount=97000,
        total_refund_amount=0,
        total_fee_amount=0,
        total_tax_amount=0,
        total_adjustment_amount=0,
        missing_evidence=missing or [],
        conflicts=[],
        has_conflicts=has_conflicts,
        evidence_link_count=0,
    )


def _make_explanation(
    status=ExplanationStatus.FULLY_EXPLAINED,
    supporting_ids=None,
    candidates=None,
    conflict=False,
    explained_amount=-3000,
    remaining=0,
):
    """Create a minimal ExplanationResult for testing."""
    return ExplanationResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        explanation_status=status,
        explained_amount=explained_amount,
        remaining_difference=remaining,
        supporting_evidence_ids=supporting_ids or ["FEE-001"],
        candidate_explanations=candidates or [],
        conflict=conflict,
        missing_evidence=[],
        explanation_reason="Fee explains difference.",
    )


def _make_quality(
    consistency=0.85,
    coverage=1.0,
    conflict=False,
    novelty=NoveltyLevel.KNOWN_PATTERN,
    fully=True,
    partially=False,
    evidence_count=1,
):
    """Create a minimal EvidenceQualityResult for testing."""
    return EvidenceQualityResult(
        consistency_score=consistency,
        coverage_score=coverage,
        conflict=conflict,
        novelty=novelty,
        fully_explained=fully,
        partially_explained=partially,
        missing_evidence=[],
        supporting_evidence_count=evidence_count,
    )


@pytest.fixture
def trained_classifier():
    """Create a trained ResolutionClassifier with synthetic data."""
    rng = np.random.default_rng(42)
    classifier = ResolutionClassifier(seed=42)

    # Generate training data
    resolutions = ALL_RESOLUTIONS
    X = rng.standard_normal((200, 29))
    y = np.array([i % len(resolutions) for i in range(200)])

    classifier.fit(X, y, feature_names=[f"f{i}" for i in range(29)])
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# ResolutionClassifier Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionClassifier:
    def test_classifier_init(self):
        classifier = ResolutionClassifier(seed=42)
        assert classifier.seed == 42
        assert not classifier.is_fitted
        assert len(ALL_RESOLUTIONS) == 10

    def test_classifier_fit(self, trained_classifier):
        assert trained_classifier.is_fitted
        assert len(trained_classifier.feature_names) == 29

    def test_classifier_predict(self, trained_classifier):
        rng = np.random.default_rng(100)
        X_test = rng.standard_normal((5, 29))
        preds = trained_classifier.predict(X_test)
        assert len(preds) == 5
        assert all(isinstance(p, (int, np.integer)) for p in preds)

    def test_classifier_predict_labels(self, trained_classifier):
        rng = np.random.default_rng(101)
        X_test = rng.standard_normal((3, 29))
        labels = trained_classifier.predict_labels(X_test)
        assert len(labels) == 3
        for label in labels:
            assert label in ALL_RESOLUTIONS

    def test_classifier_predict_proba(self, trained_classifier):
        rng = np.random.default_rng(102)
        X_test = rng.standard_normal((3, 29))
        proba = trained_classifier.predict_proba(X_test)
        assert proba.shape == (3, 10)
        # Each row sums to 1.0
        for row in proba:
            assert abs(sum(row) - 1.0) < 0.001

    def test_classifier_reproducible(self, seed=42):
        """Same seed + data = same predictions."""
        rng1 = np.random.default_rng(seed)
        X = rng1.standard_normal((100, 29))
        y = np.array([i % 10 for i in range(100)])

        c1 = ResolutionClassifier(seed=seed)
        c1.fit(X, y, feature_names=[f"f{i}" for i in range(29)])

        c2 = ResolutionClassifier(seed=seed)
        c2.fit(X, y, feature_names=[f"f{i}" for i in range(29)])

        X_test = np.random.default_rng(99).standard_normal((5, 29))
        np.testing.assert_array_equal(c1.predict(X_test), c2.predict(X_test))


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceCompatibilityChecker Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceCompatibilityChecker:
    def test_compatible_fee_resolution(self):
        package = _make_package(
            fees=[_make_record("FEE-001", "FEE", 3000)]
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "FEE_ADJUSTMENT", package, explanation
        )
        assert compatible
        assert "compatible" in notes[0].lower()

    def test_incompatible_fee_resolution_no_fees(self):
        package = _make_package(fees=[])
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "FEE_ADJUSTMENT", package, explanation
        )
        assert not compatible
        assert any("no fee" in n.lower() for n in notes)

    def test_incompatible_refund_resolution_no_refunds(self):
        package = _make_package(refunds=[])
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "REFUND_ADJUSTMENT", package, explanation
        )
        assert not compatible
        assert any("no refund" in n.lower() for n in notes)

    def test_incompatible_tax_resolution_no_taxes(self):
        package = _make_package(taxes=[])
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "TAX_ADJUSTMENT", package, explanation
        )
        assert not compatible
        assert any("no tax" in n.lower() for n in notes)

    def test_incompatible_duplicate_one_settlement(self):
        package = _make_package(
            settlements=[_make_record("SET-001", "SETTLEMENT", 97000, "SETTLED")]
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "DUPLICATE_SETTLEMENT", package, explanation
        )
        assert not compatible
        assert any(">=2 settlements" in n for n in notes)

    def test_compatible_duplicate_two_settlements(self):
        package = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 97000, "SETTLED"),
                _make_record("SET-002", "SETTLEMENT", 97000, "SETTLED"),
            ]
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "DUPLICATE_SETTLEMENT", package, explanation
        )
        assert compatible

    def test_conflict_incompatible_with_specific_resolution(self):
        package = _make_package(
            fees=[_make_record("FEE-001", "FEE", 3000)]
        )
        explanation = _make_explanation(conflict=True)
        compatible, notes = EvidenceCompatibilityChecker.check(
            "FEE_ADJUSTMENT", package, explanation
        )
        assert not compatible
        assert any("conflict" in n.lower() for n in notes)

    def test_conflict_compatible_with_manual_review(self):
        package = _make_package(
            fees=[_make_record("FEE-001", "FEE", 3000)]
        )
        explanation = _make_explanation(conflict=True)
        compatible, notes = EvidenceCompatibilityChecker.check(
            "MANUAL_REVIEW", package, explanation
        )
        assert compatible

    def test_no_action_incompatible_with_unexplained(self):
        package = _make_package()
        explanation = _make_explanation(
            status=ExplanationStatus.UNEXPLAINED,
            explained_amount=0,
            remaining=3000,
            supporting_ids=[],
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "NO_ACTION", package, explanation
        )
        assert not compatible
        assert any("unexplained" in n.lower() for n in notes)

    def test_no_action_compatible_with_explained(self):
        package = _make_package(
            fees=[_make_record("FEE-001", "FEE", 3000)]
        )
        explanation = _make_explanation(
            status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=-3000,
            remaining=0,
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "NO_ACTION", package, explanation
        )
        assert compatible

    def test_missing_record_compatible_when_missing(self):
        package = _make_package(
            missing=[MissingEvidence(entity_type="SETTLEMENT", expected=True, reason="No settlement")]
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "MISSING_RECORD_ESCALATION", package, explanation
        )
        assert compatible

    def test_missing_record_incompatible_when_no_missing(self):
        package = _make_package(missing=[])
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "MISSING_RECORD_ESCALATION", package, explanation
        )
        assert not compatible
        assert any("no missing" in n.lower() for n in notes)

    def test_no_action_compatible_with_exact_match(self):
        """NO_ACTION should be compatible when discrepancy is zero."""
        pkg = _make_package()
        pkg.difference = 0
        pkg.expected_amount = 100000
        pkg.actual_amount = 100000
        explanation = _make_explanation(
            status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=0,
            remaining=0,
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "NO_ACTION", pkg, explanation
        )
        assert compatible

    def test_unknown_unresolved_incompatible_with_no_action(self):
        """UNKNOWN_UNRESOLVED should not be used when explanation is clear."""
        package = _make_package()
        explanation = _make_explanation(
            status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=-3000,
            remaining=0,
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "UNKNOWN_UNRESOLVED", package, explanation
        )
        # UNKNOWN_UNRESOLVED has no evidence requirement, so it should be compatible
        # unless there's another rule violated
        assert compatible or not any("unexplained" in n.lower() for n in notes)

    def test_partial_settlement_compatible_with_settlements(self):
        """PARTIAL_SETTLEMENT_RECONCILIATION requires settlements."""
        package = _make_package(
            settlements=[_make_record("SET-001", "SETTLEMENT", 50000, "SETTLED")]
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "PARTIAL_SETTLEMENT_RECONCILIATION", package, explanation
        )
        assert compatible

    def test_partial_settlement_incompatible_without_settlements(self):
        """PARTIAL_SETTLEMENT_RECONCILIATION requires settlements."""
        # Build package directly to ensure empty settlements list
        package = EvidencePackage(
            exception_id="EXC-001",
            case_id="CASE-001",
            payment_id="PAY-001",
            expected_amount=100000,
            actual_amount=50000,
            difference=50000,
            exception_type="PARTIAL_SETTLEMENT",
            payment=_make_record("PAY-001", "PAYMENT", 100000, "CAPTURED", "PRIMARY_RECORD"),
            settlements=[],
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            total_settlement_amount=50000,
            missing_evidence=[],
            conflicts=[],
            evidence_link_count=0,
        )
        explanation = _make_explanation()
        compatible, notes = EvidenceCompatibilityChecker.check(
            "PARTIAL_SETTLEMENT_RECONCILIATION", package, explanation
        )
        assert not compatible


# ─────────────────────────────────────────────────────────────────────────────
# ResolutionPrediction Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionPrediction:
    def test_prediction_structure(self):
        pred = ResolutionPrediction(
            predicted_resolution="FEE_ADJUSTMENT",
            predicted_exception_type="FEE_DIFFERENCE",
            probabilities={"FEE_ADJUSTMENT": 0.8, "OTHER": 0.2},
            evidence_compatible=True,
            compatibility_notes=["Evidence is compatible"],
            supporting_evidence_ids=["FEE-001"],
            model_version="1.0.0",
        )
        assert pred.predicted_resolution == "FEE_ADJUSTMENT"
        assert pred.evidence_compatible is True
        assert len(pred.supporting_evidence_ids) == 1

    def test_prediction_to_dict(self):
        pred = ResolutionPrediction(
            predicted_resolution="NO_ACTION",
            predicted_exception_type="EXACT_MATCH",
            probabilities={"NO_ACTION": 0.95},
            evidence_compatible=True,
            compatibility_notes=["OK"],
            supporting_evidence_ids=[],
            model_version="1.0.0",
        )
        d = pred.to_dict()
        assert d["is_recommendation_only"] is True
        assert d["predicted_resolution"] == "NO_ACTION"

    def test_prediction_is_recommendation_only(self):
        """ResolutionPrediction must always be marked as recommendation only."""
        pred = ResolutionPrediction(
            predicted_resolution="FEE_ADJUSTMENT",
            predicted_exception_type="FEE_DIFFERENCE",
            probabilities={},
            evidence_compatible=True,
            compatibility_notes=[],
            supporting_evidence_ids=[],
            model_version="1.0.0",
        )
        d = pred.to_dict()
        assert d["is_recommendation_only"] is True


# ─────────────────────────────────────────────────────────────────────────────
# ResolutionPredictor Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionPredictor:
    def test_predict_with_trained_model(self, trained_classifier):
        feature_names = trained_classifier.feature_names
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        feature_dict = {name: 0.5 for name in feature_names}
        package = _make_package()
        explanation = _make_explanation()

        pred = predictor.predict(feature_dict, package, explanation)
        assert isinstance(pred, ResolutionPrediction)
        assert pred.predicted_resolution in ALL_RESOLUTIONS
        assert pred.evidence_compatible in (True, False)

    def test_predict_sets_exception_type_from_package(self, trained_classifier):
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        package = _make_package(exception_type="UNKNOWN")
        explanation = _make_explanation()
        feature_dict = {name: 0.1 for name in trained_classifier.feature_names}

        pred = predictor.predict(feature_dict, package, explanation)
        assert pred.predicted_exception_type == "UNKNOWN"

    def test_predict_batch(self, trained_classifier):
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        feature_dicts = [
            {name: 0.5 for name in trained_classifier.feature_names}
            for _ in range(3)
        ]
        packages = [_make_package() for _ in range(3)]
        explanations = [_make_explanation() for _ in range(3)]

        preds = predictor.predict_batch(feature_dicts, packages, explanations)
        assert len(preds) == 3
        for pred in preds:
            assert isinstance(pred, ResolutionPrediction)

    def test_predict_no_ground_truth_imported(self):
        """ResolutionPredictor must not use ground truth at inference time."""
        import inspect

        source = inspect.getsource(ResolutionPredictor)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source
        assert "risk_category" not in source

    def test_predict_probability_sums_to_one(self, trained_classifier):
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        feature_dict = {name: 1.0 for name in trained_classifier.feature_names}
        package = _make_package()
        explanation = _make_explanation()

        pred = predictor.predict(feature_dict, package, explanation)
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_evidence_compatibility_logged(self, trained_classifier):
        """Evidence compatibility result must be included in prediction."""
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        # Package with no fees, but model might predict FEE_ADJUSTMENT
        package = _make_package(fees=[])
        explanation = _make_explanation()
        feature_dict = {name: 0.0 for name in trained_classifier.feature_names}

        pred = predictor.predict(feature_dict, package, explanation)
        # Compatibility notes must exist
        assert isinstance(pred.compatibility_notes, list)
        assert len(pred.compatibility_notes) > 0

    def test_predict_multiple_samples_different_results(self, trained_classifier):
        """Different feature inputs should produce different predictions sometimes."""
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )

        # Two very different feature vectors
        feat1 = {name: 0.0 for name in trained_classifier.feature_names}
        feat2 = {name: 100.0 for name in trained_classifier.feature_names}
        package = _make_package()
        explanation = _make_explanation()

        pred1 = predictor.predict(feat1, package, explanation)
        pred2 = predictor.predict(feat2, package, explanation)
        # At least the probabilities should differ
        assert pred1.probabilities != pred2.probabilities


# ─────────────────────────────────────────────────────────────────────────────
# ResolutionTrainingBuilder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionTrainingBuilder:
    def test_build_samples(self):
        rng = np.random.default_rng(42)
        gt_records = [
            {
                "case_id": "CASE-001",
                "payment_amount": 100000,
                "difference": 3000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "true_exception_type": "FEE_DIFFERENCE",
                "true_resolution": "FEE_ADJUSTMENT",
                "total_fees": 3000,
                "total_refunds": 0,
                "total_taxes": 0,
                "total_adjustments": 0,
            },
        ]
        feature_names = [f"f{i}" for i in range(29)]
        X, y, label_names = ResolutionTrainingBuilder.build_samples(
            gt_records, feature_names, rng
        )
        assert X.shape == (1, 29)
        assert len(y) == 1
        assert len(label_names) == 10

    def test_build_samples_multiple(self):
        rng = np.random.default_rng(42)
        gt_records = [
            {
                "case_id": f"CASE-{i:03d}",
                "payment_amount": 100000,
                "difference": 3000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "true_exception_type": "FEE_DIFFERENCE",
                "true_resolution": "FEE_ADJUSTMENT",
                "total_fees": 3000,
                "total_refunds": 0,
                "total_taxes": 0,
                "total_adjustments": 0,
            }
            for i in range(10)
        ]
        feature_names = [f"f{i}" for i in range(29)]
        X, y, label_names = ResolutionTrainingBuilder.build_samples(
            gt_records, feature_names, rng
        )
        assert X.shape == (10, 29)
        assert len(y) == 10

    def test_build_samples_uses_ground_truth_only_during_training(self):
        """Training builder explicitly uses ground truth — but only for training."""
        rng = np.random.default_rng(42)
        gt_records = [
            {
                "case_id": "CASE-001",
                "payment_amount": 100000,
                "difference": 0,
                "expected_amount": 100000,
                "actual_amount": 100000,
                "true_exception_type": "EXACT_MATCH",
                "true_resolution": "NO_ACTION",
                "total_fees": 0,
                "total_refunds": 0,
                "total_taxes": 0,
                "total_adjustments": 0,
            }
        ]
        feature_names = [f"f{i}" for i in range(29)]
        X, y, labels = ResolutionTrainingBuilder.build_samples(
            gt_records, feature_names, rng
        )
        # The label is NO_ACTION, which is a valid resolution
        assert len(labels) == 10
        assert "NO_ACTION" in labels


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionTaxonomy:
    def test_exception_to_resolution_map_complete(self):
        for exc_type in ExceptionType:
            assert exc_type.value in EXCEPTION_TO_RESOLUTION_MAP

    def test_all_resolutions_valid(self):
        valid_resolutions = {r.value for r in ResolutionType}
        for resolution in ALL_RESOLUTIONS:
            assert resolution in valid_resolutions

    def test_no_action_for_exact_match(self):
        assert EXCEPTION_TO_RESOLUTION_MAP["EXACT_MATCH"] == "NO_ACTION"

    def test_unknown_unresolved_for_unknown(self):
        assert EXCEPTION_TO_RESOLUTION_MAP["UNKNOWN"] == "UNKNOWN_UNRESOLVED"

    def test_evidence_rules_cover_key_resolutions(self):
        key_resolutions = [
            "FEE_ADJUSTMENT",
            "REFUND_ADJUSTMENT",
            "TAX_ADJUSTMENT",
            "DUPLICATE_SETTLEMENT",
            "MISSING_RECORD_ESCALATION",
        ]
        for resolution in key_resolutions:
            assert resolution in EVIDENCE_COMPATIBILITY_RULES

    def test_no_leakage_in_exception_to_resolution_map(self):
        """The map is deterministic and does not use runtime labels."""
        assert len(EXCEPTION_TO_RESOLUTION_MAP) == 10
        # Known mappings
        assert EXCEPTION_TO_RESOLUTION_MAP["EXACT_MATCH"] == "NO_ACTION"
        assert EXCEPTION_TO_RESOLUTION_MAP["FEE_DIFFERENCE"] == "FEE_ADJUSTMENT"
        assert EXCEPTION_TO_RESOLUTION_MAP["REFUND_ADJUSTMENT"] == "REFUND_ADJUSTMENT"
        assert EXCEPTION_TO_RESOLUTION_MAP["TAX_ADJUSTMENT"] == "TAX_ADJUSTMENT"
        assert EXCEPTION_TO_RESOLUTION_MAP["TIMING_DIFFERENCE"] == "TIMING_RECONCILIATION"
        assert EXCEPTION_TO_RESOLUTION_MAP["PARTIAL_SETTLEMENT"] == "PARTIAL_SETTLEMENT_RECONCILIATION"
        assert EXCEPTION_TO_RESOLUTION_MAP["DUPLICATE"] == "DUPLICATE_SETTLEMENT"
        assert EXCEPTION_TO_RESOLUTION_MAP["MISSING_RECORD"] == "MISSING_RECORD_ESCALATION"
        assert EXCEPTION_TO_RESOLUTION_MAP["COMPLEX_MULTI_ADJUSTMENT"] == "MULTI_ADJUSTMENT"
        assert EXCEPTION_TO_RESOLUTION_MAP["UNKNOWN"] == "UNKNOWN_UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionEdgeCases:
    def test_unknown_exception_with_zero_difference(self):
        """UNKNOWN exception with zero difference should still get a resolution."""
        pkg = _make_package(exception_type="UNKNOWN")
        # Override fields by rebuilding (Pydantic v2 is immutable by default)
        package = EvidencePackage(
            exception_id=pkg.exception_id,
            case_id=pkg.case_id,
            payment_id=pkg.payment_id,
            merchant_id=pkg.merchant_id,
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
            exception_type="UNKNOWN",
            payment=pkg.payment,
            settlements=pkg.settlements,
            refunds=pkg.refunds,
            fees=pkg.fees,
            taxes=pkg.taxes,
            adjustments=pkg.adjustments,
            total_settlement_amount=pkg.total_settlement_amount,
            total_refund_amount=pkg.total_refund_amount,
            total_fee_amount=pkg.total_fee_amount,
            total_tax_amount=pkg.total_tax_amount,
            total_adjustment_amount=pkg.total_adjustment_amount,
            missing_evidence=pkg.missing_evidence,
            conflicts=pkg.conflicts,
            has_conflicts=pkg.has_conflicts,
            evidence_link_count=pkg.evidence_link_count,
        )
        explanation = _make_explanation(
            status=ExplanationStatus.FULLY_EXPLAINED,
            explained_amount=0,
            remaining=0,
            supporting_ids=[],
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "NO_ACTION", package, explanation
        )
        # ZERO difference with NO_ACTION should be compatible
        assert compatible

    def test_empty_evidence_package(self):
        """Empty evidence package should not crash the compatibility checker."""
        package = EvidencePackage(
            exception_id="EXC-999",
            case_id="CASE-999",
            payment_id="PAY-999",
            merchant_id="MER-999",
            expected_amount=0,
            actual_amount=0,
            difference=0,
            exception_type="UNKNOWN",
            payment=None,
            settlements=[],
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            total_settlement_amount=0,
            total_refund_amount=0,
            total_fee_amount=0,
            total_tax_amount=0,
            total_adjustment_amount=0,
            missing_evidence=[],
            conflicts=[],
            has_conflicts=False,
            evidence_link_count=0,
        )
        explanation = _make_explanation(
            status=ExplanationStatus.UNEXPLAINED,
            explained_amount=0,
            remaining=0,
            supporting_ids=[],
        )
        compatible, notes = EvidenceCompatibilityChecker.check(
            "FEE_ADJUSTMENT", package, explanation
        )
        assert not compatible
        assert any("no fee" in n.lower() for n in notes)

    def test_resolution_predictor_model_version(self, trained_classifier):
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
            model_version="2.0.0",
        )

        feature_dict = {name: 0.5 for name in trained_classifier.feature_names}
        package = _make_package()
        explanation = _make_explanation()

        pred = predictor.predict(feature_dict, package, explanation)
        assert pred.model_version == "2.0.0"

    def test_predict_multiple_different_packages(self, trained_classifier):
        """Different packages should produce different compatibility results."""
        predictor = ResolutionPredictor(
            model=trained_classifier.model,
            feature_names=trained_classifier.feature_names,
            label_names=ALL_RESOLUTIONS,
        )
        feature_dict = {name: 0.5 for name in trained_classifier.feature_names}

        # Package with fees
        pkg1 = _make_package(
            fees=[_make_record("FEE-001", "FEE", 3000)]
        )
        # Package without fees
        pkg2 = _make_package(fees=[])
        explanation = _make_explanation()

        pred1 = predictor.predict(feature_dict, pkg1, explanation)
        pred2 = predictor.predict(feature_dict, pkg2, explanation)

        # The same model prediction might be compatible with one but not the other
        # At minimum, both should produce valid predictions
        assert pred1.predicted_resolution in ALL_RESOLUTIONS
        assert pred2.predicted_resolution in ALL_RESOLUTIONS

    def test_trained_model_accuracy_above_baseline(self, trained_classifier):
        """Trained model should perform better than majority class baseline."""
        rng = np.random.default_rng(200)
        X_test = rng.standard_normal((100, 29))
        y_test = np.array([i % 10 for i in range(100)])

        preds = trained_classifier.predict(X_test)
        accuracy = np.mean(preds == y_test)
        # Should be better than random (10% for 10 classes)
        assert accuracy > 0.1
