"""
Phase 5 — Exception Intelligence + XGBoost + Deterministic Features Tests.

Covers:
1. Feature determinism
2. Feature schema version
3. Feature ordering
4. Feature types
5. Missing-value handling
6. Leakage prevention
7. Dataset construction
8. Class distribution
9. Deterministic train/validation split
10. XGBoost training
11. Model artifact creation
12. Artifact reload
13. Prediction
14. Probability validity
15. ModelPrediction persistence
16. Feature schema mismatch failure
17. Missing artifact failure
18. Corrupt artifact failure
19. Financial truth preservation
20. Deterministic-vs-ML disagreement
21. Evaluation metrics
22. Confusion matrix
23. Golden FEE_MISMATCH vertical slice
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_phase5.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.db_test_helper import (
    ensure_financial_parents,
    ensure_merchant,
    get_test_session,
    reset_database,
)
from app.models.exception import FinancialException
from app.models.model_prediction import ModelPrediction
from app.models.root_cause import RootCause
from app.schemas.enums import ExceptionType
from app.schemas.ml_dataset import (
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    LEAKED_FIELDS,
    MLLabels,
    MLSample,
)
from app.ml.features import ML_FEATURE_SCHEMA, extract_features, validate_features
from app.ml.engineering import FeatureEngineer
from app.ml.classifier import (
    CLASSIFIER_VERSION,
    DatasetBuilder,
    ExceptionClassifier,
    ExceptionClassifierService,
    ModelArtifact,
    ModelEvaluator,
)
from app.providers.mock_provider import MockProvider
from app.ingestion.service import IngestionService
from app.services.evidence_retrieval import EvidenceRetrievalService


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
def feature_names():
    """Provide feature names from schema."""
    return ML_FEATURE_SCHEMA.get_feature_names()


@pytest.fixture
def synthetic_samples(feature_names):
    """Create synthetic MLSamples for testing."""
    rng = np.random.default_rng(42)
    samples = []

    for i in range(100):
        exc_type = list(ExceptionType)[i % len(ExceptionType)]
        features = {name: float(rng.uniform(0, 1)) for name in feature_names}

        sample = MLSample(
            case_id=f"CASE-{i:04d}",
            payment_id=f"PAY-{i:04d}",
            expected_amount=100000,
            actual_amount=100000 + rng.integers(-5000, 5000),
            difference=0,
            payment_amount=100000,
            features=FeatureVector(features=features, schema_version=FEATURE_SCHEMA_VERSION),
            labels=MLLabels(
                true_exception_type=exc_type,
                resolvable=True,
            ),
        )
        sample.difference = sample.expected_amount - sample.actual_amount
        samples.append(sample)

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 1-6: Feature Contract, Determinism, & Leakage Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_determinism():
    """Test that extract_features produces identical results for identical inputs."""
    f1 = extract_features(
        difference=25000,
        payment_amount=1000000,
        settlement_amount=925000,
        refund_amount=0,
        fee_amount=50000,
        tax_amount=0,
        adjustment_amount=0,
        num_settlements=1,
        num_refunds=0,
        num_fees=1,
        num_taxes=0,
        num_adjustments=0,
        has_missing_evidence=False,
        num_missing_evidence=0,
        evidence_coverage=1.0,
        consistency_score=1.0,
        fully_explained=True,
        partially_explained=False,
        has_conflict=False,
        supporting_evidence_count=2,
        num_candidate_explanations=1,
    )
    f2 = extract_features(
        difference=25000,
        payment_amount=1000000,
        settlement_amount=925000,
        refund_amount=0,
        fee_amount=50000,
        tax_amount=0,
        adjustment_amount=0,
        num_settlements=1,
        num_refunds=0,
        num_fees=1,
        num_taxes=0,
        num_adjustments=0,
        has_missing_evidence=False,
        num_missing_evidence=0,
        evidence_coverage=1.0,
        consistency_score=1.0,
        fully_explained=True,
        partially_explained=False,
        has_conflict=False,
        supporting_evidence_count=2,
        num_candidate_explanations=1,
    )

    assert f1 == f2


def test_feature_schema_version():
    """Test feature schema versioning contract."""
    assert ML_FEATURE_SCHEMA.version == FEATURE_SCHEMA_VERSION
    assert len(ML_FEATURE_SCHEMA.features) == 28


def test_feature_ordering(feature_names):
    """Test that feature names are ordered and non-empty."""
    assert len(feature_names) == 28
    assert feature_names[0] == "difference_amount"
    assert "relative_difference" in feature_names


def test_feature_types():
    """Test that extracted feature values are float types."""
    f = extract_features(
        difference=100, payment_amount=1000, settlement_amount=900,
        refund_amount=0, fee_amount=100, tax_amount=0, adjustment_amount=0,
        num_settlements=1, num_refunds=0, num_fees=1, num_taxes=0, num_adjustments=0,
        has_missing_evidence=False, num_missing_evidence=0, evidence_coverage=1.0,
        consistency_score=1.0, fully_explained=True, partially_explained=False,
        has_conflict=False, supporting_evidence_count=1, num_candidate_explanations=1
    )
    for k, v in f.items():
        assert isinstance(v, float), f"Feature {k} is not float"


def test_missing_value_handling():
    """Test zero division and missing value handling produce safe 0.0."""
    f = extract_features(
        difference=100, payment_amount=0, settlement_amount=0,
        refund_amount=0, fee_amount=0, tax_amount=0, adjustment_amount=0,
        num_settlements=0, num_refunds=0, num_fees=0, num_taxes=0, num_adjustments=0,
        has_missing_evidence=True, num_missing_evidence=1, evidence_coverage=0.0,
        consistency_score=0.0, fully_explained=False, partially_explained=False,
        has_conflict=False, supporting_evidence_count=0, num_candidate_explanations=0
    )
    assert f["relative_difference"] == 0.0
    assert f["refund_ratio"] == 0.0
    assert f["fee_ratio"] == 0.0
    assert f["tax_ratio"] == 0.0


def test_leakage_prevention(feature_names):
    """Test that no forbidden post-outcome fields exist in the feature schema."""
    for leaked in LEAKED_FIELDS:
        assert leaked not in feature_names, f"Leaked field {leaked} found in ML feature schema"
    for forbidden in ["resolution_status", "post_resolution", "human_approval", "final_closed_state"]:
        assert forbidden not in feature_names, f"Prohibited field {forbidden} found in feature schema"


# ─────────────────────────────────────────────────────────────────────────────
# 7-10: Dataset, Training, & Classifier Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_dataset_construction(synthetic_samples, feature_names):
    """Test numpy matrix construction from samples."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    assert X.shape == (100, 28)
    assert y.shape == (100,)
    assert len(labels) == len(ExceptionType)


def test_class_distribution(synthetic_samples, feature_names):
    """Test dataset class distribution calculation."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42)
    meta = clf.fit(X, y, feature_names=feature_names)

    assert "class_distribution" in meta
    assert meta["n_samples"] == 100
    assert meta["n_features"] == 28


def test_deterministic_train_validation_split(synthetic_samples, feature_names):
    """Test reproducible train/validation split using fixed seed."""
    builder = DatasetBuilder(feature_names)
    X, y, _ = builder.build(synthetic_samples)

    clf1 = ExceptionClassifier(seed=42)
    clf1.fit(X[:80], y[:80], feature_names=feature_names)

    clf2 = ExceptionClassifier(seed=42)
    clf2.fit(X[:80], y[:80], feature_names=feature_names)

    p1 = clf1.predict(X[80:])
    p2 = clf2.predict(X[80:])

    assert np.array_equal(p1, p2)


def test_xgboost_training(synthetic_samples, feature_names):
    """Test that real XGBoost classifier trains successfully."""
    builder = DatasetBuilder(feature_names)
    X, y, _ = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    meta = clf.fit(X, y, feature_names=feature_names)

    assert clf.is_fitted is True
    assert len(clf.get_feature_importance()) == 28


# ─────────────────────────────────────────────────────────────────────────────
# 11-15: Artifact, Reload, Prediction, & Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_model_artifact_creation(synthetic_samples, feature_names):
    """Test saving model artifact to disk."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    meta = clf.fit(X, y, feature_names=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = os.path.join(tmpdir, "model_v1")
        ModelArtifact.save(
            model=clf.model,
            path=art_dir,
            feature_names=feature_names,
            label_names=labels,
            training_metadata=meta,
            evaluation={"accuracy": 0.9},
        )

        assert os.path.exists(os.path.join(art_dir, "model.joblib"))
        assert os.path.exists(os.path.join(art_dir, "metadata.json"))


def test_artifact_reload(synthetic_samples, feature_names):
    """Test reloading model artifact and using it for inference."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    meta = clf.fit(X, y, feature_names=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = os.path.join(tmpdir, "model_v1")
        ModelArtifact.save(
            model=clf.model,
            path=art_dir,
            feature_names=feature_names,
            label_names=labels,
            training_metadata=meta,
            evaluation={"accuracy": 0.9},
        )

        service = ExceptionClassifierService.from_artifact(
            art_dir, expected_schema_version=FEATURE_SCHEMA_VERSION
        )
        pred = service.predict(synthetic_samples[0].features.features)

        assert pred.predicted_type in labels
        assert len(pred.probabilities) == len(labels)


def test_prediction(synthetic_samples, feature_names):
    """Test inference output structure."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    service = ExceptionClassifierService(
        model=clf.model,
        feature_names=feature_names,
        label_names=labels,
        model_version="1.0.0",
    )

    pred = service.predict(synthetic_samples[0].features.features)
    assert pred.predicted_type in labels
    assert pred.model_version == "1.0.0"


def test_probability_validity(synthetic_samples, feature_names):
    """Test that predicted probabilities sum to 1.0 and are in [0, 1]."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    service = ExceptionClassifierService(
        model=clf.model, feature_names=feature_names, label_names=labels
    )
    pred = service.predict(synthetic_samples[0].features.features)

    for p in pred.probabilities.values():
        assert 0.0 <= p <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_model_prediction_persistence(session, synthetic_samples, feature_names):
    """Test persisting ModelPrediction to database."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    # Seed parent exception
    exc = FinancialException(
        id="EXC-MPRED-001", case_id="CASE-001", payment_id="PAY-001",
        batch_id="batch_001", expected_amount=100000, actual_amount=95000,
        difference=5000, exception_type="FEE_DIFFERENCE", status="OPEN",
        reconciliation_id="REC-001"
    )
    session.add(exc)
    session.flush()

    service = ExceptionClassifierService(
        model=clf.model, feature_names=feature_names, label_names=labels
    )

    pred_row = service.persist_prediction(
        session=session,
        exception_id=exc.id,
        feature_dict=synthetic_samples[0].features.features,
    )

    db_pred = session.query(ModelPrediction).filter_by(id=pred_row.id).first()
    assert db_pred is not None
    assert db_pred.exception_id == exc.id
    assert db_pred.model_version == CLASSIFIER_VERSION
    assert db_pred.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert db_pred.predicted_type in labels


# ─────────────────────────────────────────────────────────────────────────────
# 16-18: Error Handling & Schema Mismatch Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_schema_mismatch_failure(synthetic_samples, feature_names):
    """Test that schema version mismatch or missing features fail clearly."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = os.path.join(tmpdir, "model_v1")
        ModelArtifact.save(
            model=clf.model,
            path=art_dir,
            feature_names=feature_names,
            label_names=labels,
            training_metadata={},
            evaluation={},
        )

        with pytest.raises(ValueError, match="Incompatible feature schema version"):
            ExceptionClassifierService.from_artifact(
                art_dir, expected_schema_version="phase5.v999"
            )

    service = ExceptionClassifierService(
        model=clf.model, feature_names=feature_names, label_names=labels
    )
    incomplete_features = {"difference_amount": 100.0}
    with pytest.raises(ValueError, match="Incompatible feature schema"):
        service.predict(incomplete_features, validate_schema=True)


def test_missing_artifact_failure():
    """Test that missing artifact path fails cleanly without crashing Python."""
    with pytest.raises(FileNotFoundError):
        ExceptionClassifierService.from_artifact("/nonexistent/path/to/artifact")


def test_corrupt_artifact_failure():
    """Test that corrupt artifact file fails safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = os.path.join(tmpdir, "corrupt_model")
        os.makedirs(art_dir, exist_ok=True)

        with open(os.path.join(art_dir, "model.joblib"), "w") as f:
            f.write("corrupt binary data")
        with open(os.path.join(art_dir, "metadata.json"), "w") as f:
            f.write("{}")

        with pytest.raises(Exception):
            ExceptionClassifierService.from_artifact(art_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 19-22: Financial Truth Preservation, Disagreement, & Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_financial_truth_preservation(session, synthetic_samples, feature_names):
    """Test that ML prediction does NOT alter FinancialException fields."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    exc = FinancialException(
        id="EXC-TRUTH-001", case_id="CASE-TRUTH", payment_id="PAY-TRUTH",
        batch_id="batch_001", expected_amount=950000, actual_amount=925000,
        difference=25000, exception_type="FEE_MISMATCH", status="OPEN",
        reconciliation_id="REC-TRUTH"
    )
    session.add(exc)
    session.flush()

    service = ExceptionClassifierService(
        model=clf.model, feature_names=feature_names, label_names=labels
    )
    service.persist_prediction(session, exc.id, synthetic_samples[0].features.features)

    db_exc = session.query(FinancialException).filter_by(id="EXC-TRUTH-001").first()
    assert db_exc.exception_type == "FEE_MISMATCH"
    assert db_exc.expected_amount == 950000
    assert db_exc.actual_amount == 925000
    assert db_exc.difference == 25000
    assert db_exc.status == "OPEN"


def test_deterministic_vs_ml_disagreement(session, synthetic_samples, feature_names):
    """Test that deterministic truth and ML prediction coexist independently when they disagree."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    exc = FinancialException(
        id="EXC-DISAGREE-001", case_id="CASE-DISAGREE", payment_id="PAY-DISAGREE",
        batch_id="batch_001", expected_amount=950000, actual_amount=925000,
        difference=25000, exception_type="FEE_MISMATCH", status="OPEN",
        reconciliation_id="REC-DISAGREE"
    )
    session.add(exc)
    session.flush()

    service = ExceptionClassifierService(
        model=clf.model, feature_names=feature_names, label_names=labels
    )
    # Manually override predicted type in mock result to guarantee disagreement
    pred = service.predict(synthetic_samples[0].features.features)
    pred.predicted_type = "TIMING_DIFFERENCE"

    pred_row = ModelPrediction(
        id=f"MPRED-{exc.id}-v1",
        exception_id=exc.id,
        model_name="XGBoostExceptionClassifier",
        model_version="1.0.0",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        predicted_type="TIMING_DIFFERENCE",
        probabilities=pred.probabilities,
        confidence=0.85,
    )
    session.add(pred_row)
    session.flush()

    db_exc = session.query(FinancialException).filter_by(id=exc.id).first()
    db_pred = session.query(ModelPrediction).filter_by(exception_id=exc.id).first()

    assert db_exc.exception_type == "FEE_MISMATCH"
    assert db_pred.predicted_type == "TIMING_DIFFERENCE"
    assert db_exc.exception_type != db_pred.predicted_type


def test_evaluation_metrics(synthetic_samples, feature_names):
    """Test evaluation harness outputs macro F1, weighted F1, accuracy."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X[:70], y[:70], feature_names=feature_names)

    preds = clf.predict(X[70:])
    eval_res = ModelEvaluator.evaluate(y[70:], preds, labels)

    assert "accuracy" in eval_res
    assert "macro_f1" in eval_res
    assert "weighted_f1" in eval_res
    assert "per_class" in eval_res


def test_confusion_matrix(synthetic_samples, feature_names):
    """Test confusion matrix structure."""
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)

    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X[:70], y[:70], feature_names=feature_names)

    preds = clf.predict(X[70:])
    eval_res = ModelEvaluator.evaluate(y[70:], preds, labels)

    cm = eval_res["confusion_matrix"]
    assert len(cm) == len(labels)
    assert len(cm[0]) == len(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 23: Golden FEE_MISMATCH End-to-End Vertical Slice
# ─────────────────────────────────────────────────────────────────────────────


def _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-GOLDEN-P5"):
    from datetime import datetime, timezone
    from app.models.payment import Payment as DBPayment
    from app.models.settlement import Settlement as DBSettlement
    from app.models.fee import Fee as DBFee
    from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
    from app.schemas.enums import FeeType, ExceptionType
    from app.schemas.financial import Payment, Settlement, Fee
    from app.reconciliation.engine import calculate_reconciliation
    from app.services.persistence import PersistenceService

    NOW = datetime.now(timezone.utc)
    batch_id = f"BATCH-{case_id}"

    provider = MockProvider(seed=42)
    provider.add_merchant("MER-P5-001", name="Golden Merchant P5")
    provider.add_payment("PAY-P5-001", merchant_id="MER-P5-001", amount=1_000_000, captured_at=NOW)
    provider.add_fee("FEE-P5-001", payment_id="PAY-P5-001", amount=50_000, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    provider.add_settlement("SET-P5-001", payment_id="PAY-P5-001", merchant_id="MER-P5-001", amount=925_000, settled_at=NOW)

    ingestion_service = IngestionService(db_session)
    ingestion_service.ingest(provider)

    run = DBReconciliationRun(
        id=f"RUN-{case_id}",
        scope_type="BATCH",
        scope_ref=batch_id,
        status="PENDING",
        trigger="MANUAL",
        started_at=NOW,
        engine_version="5.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    payment_db = db_session.get(DBPayment, "PAY-P5-001")
    settlement_db = db_session.get(DBSettlement, "SET-P5-001")
    fee_db = db_session.get(DBFee, "FEE-P5-001")

    p_domain = Payment(payment_id=payment_db.id, merchant_id=payment_db.merchant_id, amount=payment_db.amount, payment_timestamp=NOW)
    f_domain = Fee(fee_id=fee_db.id, payment_id=fee_db.payment_id, amount=fee_db.amount, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    s_domain = Settlement(settlement_id=settlement_db.id, payment_id=settlement_db.payment_id, merchant_id=settlement_db.merchant_id, amount=settlement_db.amount, settlement_timestamp=NOW)

    result_schema = calculate_reconciliation(
        payment=p_domain,
        settlements=[s_domain],
        refunds=[],
        fees=[f_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id=f"REC-{case_id}",
    )

    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(result_schema, batch_id=batch_id)
    db_result.reconciliation_run_id = run.id
    db_exception = persistence.persist_exception(result_schema, batch_id=batch_id)

    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = datetime.now(timezone.utc)
    run.transition_to("COMPLETED")
    db_session.commit()

    return db_exception


def test_golden_fee_mismatch_phase5_end_to_end(session, synthetic_samples, feature_names):
    """
    End-to-end Phase 5 vertical slice:
    MockProvider -> Ingestion -> Reconciliation -> FinancialException (FEE_MISMATCH)
    -> EvidenceRetrievalService -> FeatureEngineer -> XGBoost Prediction -> ModelPrediction persistence
    """
    # 1. Train classifier
    builder = DatasetBuilder(feature_names)
    X, y, labels = builder.build(synthetic_samples)
    clf = ExceptionClassifier(seed=42, n_estimators=10)
    clf.fit(X, y, feature_names=feature_names)

    service = ExceptionClassifierService(
        model=clf.model,
        feature_names=feature_names,
        label_names=labels,
    )

    # 2. Pipeline setup for golden exception
    exc = _setup_golden_fee_mismatch_exception(session, case_id="CASE-GOLDEN-P5")
    assert exc is not None
    assert exc.exception_type == "FEE_MISMATCH"
    assert exc.expected_amount == 950000
    assert exc.actual_amount == 925000
    assert exc.difference == 25000

    # 3. Evidence Retrieval
    retrieval_service = EvidenceRetrievalService(session)
    pkg = retrieval_service.retrieve_by_exception_id(exc.id, persist_links=True)
    assert pkg is not None

    # 4. Feature Engineering
    engineer = FeatureEngineer()
    fv = engineer.engineer_from_package(pkg)
    assert len(fv.features) == 28
    assert fv.schema_version == FEATURE_SCHEMA_VERSION

    # 5. Prediction & ModelPrediction Persistence
    db_pred = service.persist_prediction(session, exc.id, fv.features)
    assert db_pred is not None
    assert db_pred.exception_id == exc.id
    assert db_pred.model_version == CLASSIFIER_VERSION
    assert db_pred.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert db_pred.predicted_type in labels
    assert 0.0 <= float(db_pred.confidence) <= 1.0

    # 6. Assert Financial Truth Invariants
    db_exc = session.query(FinancialException).filter_by(id=exc.id).first()
    assert db_exc.exception_type == "FEE_MISMATCH"
    assert db_exc.expected_amount == 950000
    assert db_exc.actual_amount == 925000
    assert db_exc.difference == 25000
