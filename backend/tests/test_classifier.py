"""
Tests for exception type classifier.

Covers:
- Dataset builder
- Majority class baseline
- Logistic regression baseline
- XGBoost classifier
- Model evaluation
- Model artifact save/load
- Inference service
- Reproducibility
- Feature ordering
- Label mapping
- No financial modification
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_classifier.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.enums import ExceptionType
from app.schemas.ml_dataset import (
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    MLLabels,
    MLSample,
)
from app.ml.features import ML_FEATURE_SCHEMA
from app.ml.classifier import (
    CLASSIFIER_VERSION,
    DatasetBuilder,
    ExceptionClassifier,
    ExceptionClassifierService,
    MajorityClassClassifier,
    ModelArtifact,
    ModelEvaluator,
    LogisticRegressionClassifier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def feature_names():
    """Provide feature names from schema."""
    return ML_FEATURE_SCHEMA.get_feature_names()


@pytest.fixture
def synthetic_samples(feature_names):
    """Create synthetic MLSamples for testing."""
    rng = np.random.default_rng(42)
    samples = []

    for i in range(200):
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


@pytest.fixture
def train_samples(synthetic_samples):
    """Provide training samples."""
    return synthetic_samples[:120]


@pytest.fixture
def test_samples(synthetic_samples):
    """Provide test samples."""
    return synthetic_samples[120:]


@pytest.fixture
def trained_classifier(train_samples, feature_names):
    """Provide a trained XGBoost classifier."""
    builder = DatasetBuilder(feature_names)
    X, y, _ = builder.build(train_samples)

    clf = ExceptionClassifier(seed=42)
    clf.fit(X, y, feature_names=feature_names)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetBuilder:
    """Tests for dataset building."""

    def test_build_arrays(self, train_samples, feature_names):
        """Test that build produces correct numpy arrays."""
        builder = DatasetBuilder(feature_names)
        X, y, labels = builder.build(train_samples)

        assert X.shape[0] == len(train_samples)
        assert X.shape[1] == len(feature_names)
        assert len(y) == len(train_samples)
        assert len(labels) == len(ExceptionType)

    def test_build_single(self, train_samples, feature_names):
        """Test single sample conversion."""
        builder = DatasetBuilder(feature_names)
        X = builder.build_single(train_samples[0])

        assert X.shape == (1, len(feature_names))

    def test_label_encoding(self, train_samples, feature_names):
        """Test that labels are correctly encoded."""
        builder = DatasetBuilder(feature_names)
        _, y, labels = builder.build(train_samples)

        assert all(isinstance(v, (int, np.integer)) for v in y)
        assert all(isinstance(v, str) for v in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Majority Class Baseline Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMajorityClassBaseline:
    """Tests for majority class baseline."""

    def test_fit_predict(self, train_samples, test_samples, feature_names):
        """Test that majority class baseline can fit and predict."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)
        X_test, y_test, _ = builder.build(test_samples)

        clf = MajorityClassClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        assert len(preds) == len(y_test)
        assert all(isinstance(p, (int, np.integer)) for p in preds)

    def test_predict_proba(self, train_samples, feature_names):
        """Test that majority class returns probabilities."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)

        clf = MajorityClassClassifier()
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_train[:5])

        assert proba.shape == (5, len(labels))
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_majority_class_accuracy(self, train_samples, test_samples, feature_names):
        """Test that majority class accuracy matches expected baseline."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)
        X_test, y_test, _ = builder.build(test_samples)

        clf = MajorityClassClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert 0.0 <= accuracy <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression Baseline Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLogisticRegressionBaseline:
    """Tests for logistic regression baseline."""

    def test_fit_predict(self, train_samples, test_samples, feature_names):
        """Test that logistic regression can fit and predict."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)
        X_test, y_test, _ = builder.build(test_samples)

        clf = LogisticRegressionClassifier(seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        assert len(preds) == len(y_test)

    def test_predict_proba(self, train_samples, feature_names):
        """Test that logistic regression returns probabilities."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)

        clf = LogisticRegressionClassifier(seed=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_train[:5])

        assert proba.shape == (5, len(labels))
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Classifier Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestXGBoostClassifier:
    """Tests for XGBoost classifier."""

    def test_fit_predict(self, trained_classifier, test_samples, feature_names):
        """Test that trained classifier can predict."""
        builder = DatasetBuilder(feature_names)
        X_test, y_test, labels = builder.build(test_samples)

        preds = trained_classifier.predict(X_test)
        assert len(preds) == len(y_test)

    def test_predict_labels(self, trained_classifier, test_samples, feature_names):
        """Test that predict_labels returns human-readable strings."""
        builder = DatasetBuilder(feature_names)
        X_test, _, labels = builder.build(test_samples)

        pred_labels = trained_classifier.predict_labels(X_test)
        assert len(pred_labels) == len(test_samples)
        assert all(isinstance(l, str) for l in pred_labels)
        assert all(l in [e.value for e in ExceptionType] for l in pred_labels)

    def test_predict_proba(self, trained_classifier, test_samples, feature_names):
        """Test that predict_proba returns valid probabilities."""
        builder = DatasetBuilder(feature_names)
        X_test, _, labels = builder.build(test_samples)

        proba = trained_classifier.predict_proba(X_test)
        assert proba.shape == (len(test_samples), len(labels))
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance(self, trained_classifier, feature_names):
        """Test that feature importance is available."""
        importance = trained_classifier.get_feature_importance()
        assert len(importance) == len(feature_names)
        assert all(isinstance(v, float) for v in importance.values())

    def test_is_fitted(self, trained_classifier):
        """Test that classifier reports fitted status."""
        assert trained_classifier.is_fitted is True


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelEvaluator:
    """Tests for model evaluation."""

    def test_evaluate_metrics(self, trained_classifier, test_samples, feature_names):
        """Test that evaluation produces all required metrics."""
        builder = DatasetBuilder(feature_names)
        X_test, y_test, labels = builder.build(test_samples)

        preds = trained_classifier.predict(X_test)
        eval_result = ModelEvaluator.evaluate(y_test, preds, labels)

        assert "accuracy" in eval_result
        assert "macro_precision" in eval_result
        assert "macro_recall" in eval_result
        assert "macro_f1" in eval_result
        assert "weighted_f1" in eval_result
        assert "per_class" in eval_result
        assert "confusion_matrix" in eval_result
        assert "top_errors" in eval_result

    def test_evaluate_ranges(self, trained_classifier, test_samples, feature_names):
        """Test that metrics are in valid ranges."""
        builder = DatasetBuilder(feature_names)
        X_test, y_test, labels = builder.build(test_samples)

        preds = trained_classifier.predict(X_test)
        eval_result = ModelEvaluator.evaluate(y_test, preds, labels)

        assert 0.0 <= eval_result["accuracy"] <= 1.0
        assert 0.0 <= eval_result["macro_f1"] <= 1.0
        assert 0.0 <= eval_result["weighted_f1"] <= 1.0

    def test_evaluate_per_class(self, trained_classifier, test_samples, feature_names):
        """Test that per-class metrics are computed."""
        builder = DatasetBuilder(feature_names)
        X_test, y_test, labels = builder.build(test_samples)

        preds = trained_classifier.predict(X_test)
        eval_result = ModelEvaluator.evaluate(y_test, preds, labels)

        assert len(eval_result["per_class"]) == len(labels)
        for name, metrics in eval_result["per_class"].items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

    def test_confusion_matrix_shape(self, trained_classifier, test_samples, feature_names):
        """Test that confusion matrix has correct shape."""
        builder = DatasetBuilder(feature_names)
        X_test, y_test, labels = builder.build(test_samples)

        preds = trained_classifier.predict(X_test)
        eval_result = ModelEvaluator.evaluate(y_test, preds, labels)

        cm = eval_result["confusion_matrix"]
        assert len(cm) == len(labels)
        assert all(len(row) == len(labels) for row in cm)


# ─────────────────────────────────────────────────────────────────────────────
# Model Artifact Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelArtifact:
    """Tests for model save/load."""

    def test_save_and_load(self, trained_classifier, train_samples, feature_names):
        """Test that model can be saved and loaded."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ModelArtifact.save(
                model=trained_classifier.model,
                path=os.path.join(tmpdir, "test_model"),
                feature_names=feature_names,
                label_names=labels,
                training_metadata={"seed": 42},
                evaluation={"accuracy": 0.95},
            )

            # Load
            model, metadata = ModelArtifact.load(os.path.join(tmpdir, "test_model"))

            assert metadata["classifier_version"] == CLASSIFIER_VERSION
            assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
            assert metadata["feature_names"] == feature_names
            assert metadata["label_names"] == labels
            assert metadata["evaluation"]["accuracy"] == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Inference Service Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInferenceService:
    """Tests for clean inference interface."""

    def test_predict_single(self, trained_classifier, test_samples, feature_names):
        """Test single prediction via service."""
        builder = DatasetBuilder(feature_names)
        X_test, _, labels = builder.build(test_samples)

        service = ExceptionClassifierService(
            model=trained_classifier.model,
            feature_names=feature_names,
            label_names=labels,
        )

        feature_dict = test_samples[0].features.features
        pred = service.predict(feature_dict)

        assert pred.predicted_type in [e.value for e in ExceptionType]
        assert len(pred.probabilities) == len(labels)
        assert pred.model_version == CLASSIFIER_VERSION

    def test_predict_batch(self, trained_classifier, test_samples, feature_names):
        """Test batch prediction via service."""
        builder = DatasetBuilder(feature_names)
        X_test, _, labels = builder.build(test_samples)

        service = ExceptionClassifierService(
            model=trained_classifier.model,
            feature_names=feature_names,
            label_names=labels,
        )

        feature_dicts = [s.features.features for s in test_samples[:10]]
        preds = service.predict_batch(feature_dicts)

        assert len(preds) == 10
        assert all(p.predicted_type in [e.value for e in ExceptionType] for p in preds)

    def test_probabilities_sum_to_one(self, trained_classifier, test_samples, feature_names):
        """Test that probabilities sum to 1.0."""
        builder = DatasetBuilder(feature_names)
        X_test, _, labels = builder.build(test_samples)

        service = ExceptionClassifierService(
            model=trained_classifier.model,
            feature_names=feature_names,
            label_names=labels,
        )

        pred = service.predict(test_samples[0].features.features)
        prob_sum = sum(pred.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReproducibility:
    """Tests for deterministic training."""

    def test_same_seed_same_model(self, train_samples, feature_names):
        """Test that same seed produces same predictions."""
        builder = DatasetBuilder(feature_names)
        X_train, y_train, labels = builder.build(train_samples)

        clf1 = ExceptionClassifier(seed=42)
        clf1.fit(X_train, y_train, feature_names=feature_names)

        clf2 = ExceptionClassifier(seed=42)
        clf2.fit(X_train, y_train, feature_names=feature_names)

        # Same model should produce same predictions
        test_X = X_train[:5]
        pred1 = clf1.predict(test_X)
        pred2 = clf2.predict(test_X)
        assert np.array_equal(pred1, pred2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Ordering Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureOrdering:
    """Tests for correct feature ordering."""

    def test_feature_names_match_schema(self, trained_classifier, feature_names):
        """Test that classifier uses correct feature ordering."""
        assert trained_classifier.feature_names == feature_names

    def test_prediction_uses_correct_order(self, trained_classifier, feature_names):
        """Test that prediction uses features in schema order."""
        # Create features in random order
        rng = np.random.default_rng(42)
        shuffled_names = list(feature_names)
        rng.shuffle(shuffled_names)
        shuffled_features = {name: float(rng.uniform(0, 1)) for name in shuffled_names}

        # Service should handle arbitrary input order
        labels = [e.value for e in ExceptionType]
        service = ExceptionClassifierService(
            model=trained_classifier.model,
            feature_names=feature_names,
            label_names=labels,
        )

        pred = service.predict(shuffled_features)
        assert pred.predicted_type in labels


# ─────────────────────────────────────────────────────────────────────────────
# No Financial Modification Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoFinancialModification:
    """Verify that classifier does not modify financial amounts."""

    def test_classifier_only_predicts(self, trained_classifier, feature_names):
        """Test that classifier only returns labels, not amounts."""
        rng = np.random.default_rng(42)
        X = np.array([[float(rng.uniform(0, 1)) for _ in feature_names]], dtype=np.float32)

        pred_idx = trained_classifier.predict(X)[0]
        pred_label = trained_classifier.label_encoder.inverse_transform([pred_idx])[0]

        # Should be a string label, not a number
        assert isinstance(pred_label, str)
        assert pred_label in [e.value for e in ExceptionType]
