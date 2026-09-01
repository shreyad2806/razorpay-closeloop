"""
Tests for Phase 9E — Candidate Model Training.

Tests cover:
- Training reproducibility
- Dataset versioning
- Model versioning
- Metric calculation
- Baseline comparison
- Candidate rejection
- Safety-critical metrics
- Edge cases
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from app.schemas.learning_dataset import (
    DataSplit,
    FeatureSnapshot,
    LearningDataset,
    LearningExample,
    LearningLabels,
    SplitType,
)
from app.schemas.model_training import (
    CandidateModelComparison,
    EvaluationMetrics,
    ModelMetadata,
    ModelStatus,
    ModelType,
    SafetyMetricCheck,
    TrainingConfig,
)
from app.services.model_training import (
    ModelComparator,
    ModelEvaluator,
    ModelTrainer,
    extract_features_and_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_example(
    example_id: str = "LEX-001",
    case_id: str = "CASE-001",
    exception_id: str = "EXC-001",
    true_exception_type: str = "FEE_DIFFERENCE",
    resolution_correct: bool = True,
    verification_passed: bool = True,
    adjustment: float = 3000.0,
    decision: str = "AUTO",
) -> LearningExample:
    return LearningExample(
        example_id=example_id,
        case_id=case_id,
        exception_id=exception_id,
        workflow_id=f"WF-{case_id}",
        features=FeatureSnapshot(
            financial_features={
                "requested_adjustment_paise": adjustment,
                "actual_adjustment_paise": adjustment,
                "difference_before_paise": adjustment,
                "difference_after_paise": 0.0,
                "discrepancy_eliminated": 1.0,
                "unintended_changes": 0.0,
            },
            structural_features={
                "has_resolution": 1.0,
                "has_verification": float(verification_passed),
                "has_human_feedback": 0.0,
                "human_override": 0.0,
            },
            evidence_features={
                "evidence_count": 1.0,
                "has_execution": 1.0,
                "has_verification_ref": 1.0,
            },
            temporal_features={"confidence": 0.85},
        ),
        labels=LearningLabels(
            true_exception_type=true_exception_type,
            predicted_exception_type=true_exception_type,
            exception_prediction_correct=True,
            true_resolution="FEE_ADJUSTMENT",
            predicted_resolution="FEE_ADJUSTMENT",
            resolution_correct=resolution_correct,
            verification_passed=verification_passed,
        ),
        guardrail_decision=decision,
        confidence=0.85,
    )


def _make_dataset(
    n_examples: int = 30,
    seed: int = 42,
) -> LearningDataset:
    """Create a synthetic learning dataset."""
    rng = np.random.RandomState(seed)
    types = ["FEE_DIFFERENCE", "REFUND_ADJUSTMENT", "TAX_ADJUSTMENT", "UNKNOWN"]
    examples = []
    for i in range(n_examples):
        etype = types[i % len(types)]
        correct = rng.random() > 0.2
        verification = rng.random() > 0.1
        adjustment = float(rng.randint(1000, 50000))
        examples.append(_make_example(
            example_id=f"LEX-{i:03d}",
            case_id=f"CASE-{i:03d}",
            exception_id=f"EXC-{i:03d}",
            true_exception_type=etype,
            resolution_correct=correct,
            verification_passed=verification,
            adjustment=adjustment,
        ))

    # Create splits
    ids = [e.example_id for e in examples]
    split = DataSplit(
        split_type=SplitType.TRAIN,
        example_ids=ids[:20],
        example_count=20,
    )
    val_split = DataSplit(
        split_type=SplitType.VALIDATION,
        example_ids=ids[20:25],
        example_count=5,
    )
    test_split = DataSplit(
        split_type=SplitType.TEST,
        example_ids=ids[25:],
        example_count=5,
    )

    return LearningDataset(
        dataset_id="LDS-TEST",
        version="1.0.0",
        feature_schema_version="1.0.0",
        examples=examples,
        splits={
            "train": split,
            "validation": val_split,
            "test": test_split,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureExtraction:
    """Test feature and label extraction from learning examples."""

    def test_extract_features_and_labels(self):
        """Extract feature matrix and labels from examples."""
        examples = [_make_example() for _ in range(5)]
        X, y, feature_names, label_classes = extract_features_and_labels(examples)
        assert X.shape == (5, len(feature_names))
        assert len(y) == 5
        assert "FEE_DIFFERENCE" in label_classes

    def test_extract_handles_empty(self):
        """Empty examples produce empty arrays."""
        X, y, feature_names, label_classes = extract_features_and_labels([])
        assert X.shape == (0,)
        assert len(y) == 0

    def test_extract_deterministic(self):
        """Same examples produce same features."""
        examples = [_make_example() for _ in range(3)]
        X1, y1, fn1, lc1 = extract_features_and_labels(examples)
        X2, y2, fn2, lc2 = extract_features_and_labels(examples)
        np.testing.assert_array_equal(X1, X2)
        assert fn1 == fn2


# ─────────────────────────────────────────────────────────────────────────────
# Model Trainer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelTrainer:
    """Test model training and reproducibility."""

    def test_train_xgboost(self):
        """Train XGBoost classifier."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        config = TrainingConfig(algorithm="xgboost", random_seed=42)
        metadata = trainer.train(dataset, config, model_version="1.0.0")
        assert metadata.model_id.startswith("MOD-")
        assert metadata.status == ModelStatus.CANDIDATE
        assert metadata.training_examples == 20  # train split
        assert metadata.feature_count > 0

    def test_train_decision_tree(self):
        """Train Decision Tree classifier."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        config = TrainingConfig(algorithm="decision_tree", random_seed=42)
        metadata = trainer.train(dataset, config, model_version="1.0.0")
        assert metadata.training_examples > 0

    def test_train_logistic_regression(self):
        """Train Logistic Regression classifier."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        config = TrainingConfig(algorithm="logistic_regression", random_seed=42)
        metadata = trainer.train(dataset, config, model_version="1.0.0")
        assert metadata.training_examples > 0

    def test_training_reproducible(self):
        """Same data + same seed → same model metadata."""
        dataset = _make_dataset(n_examples=30)
        config = TrainingConfig(algorithm="xgboost", random_seed=42)

        trainer1 = ModelTrainer()
        m1 = trainer1.train(dataset, config, model_version="1.0.0")

        trainer2 = ModelTrainer()
        m2 = trainer2.train(dataset, config, model_version="1.0.0")

        assert m1.training_examples == m2.training_examples
        assert m1.feature_count == m2.feature_count
        assert m1.feature_names == m2.feature_names

    def test_prediction_works(self):
        """Model can make predictions."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        config = TrainingConfig(algorithm="xgboost", random_seed=42)
        metadata = trainer.train(dataset, config)

        # Predict on some examples
        test_examples = dataset.get_examples_by_split(SplitType.TEST)
        X, _, _, _ = extract_features_and_labels(test_examples)
        predictions, probabilities = trainer.predict(metadata.model_id, X)
        assert len(predictions) == len(test_examples)
        assert probabilities is not None

    def test_predict_nonexistent_model_raises(self):
        """Predicting with non-existent model raises error."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError):
            trainer.predict("MOD-NONE", np.array([[1.0, 2.0]]))

    def test_train_insufficient_examples_raises(self):
        """Training with too few examples raises error."""
        dataset = LearningDataset(
            dataset_id="LDS-SMALL",
            examples=[_make_example(example_id="LEX-1", case_id="CASE-1", exception_id="EXC-1")],
        )
        trainer = ModelTrainer()
        with pytest.raises(ValueError):
            trainer.train(dataset)

    def test_dataset_version_recorded(self):
        """Model records dataset version."""
        dataset = _make_dataset(n_examples=30)
        dataset.version = "2.1.0"
        trainer = ModelTrainer()
        metadata = trainer.train(dataset, model_version="3.0.0")
        assert metadata.dataset_version == "2.1.0"
        assert metadata.version == "3.0.0"

    def test_feature_names_recorded(self):
        """Model records feature names."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)
        assert len(metadata.feature_names) > 0
        assert all(isinstance(f, str) for f in metadata.feature_names)

    def test_label_classes_recorded(self):
        """Model records label classes."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)
        assert len(metadata.label_classes) > 0

    def test_training_duration_recorded(self):
        """Model records training duration."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)
        assert metadata.training_duration_seconds is not None
        assert metadata.training_duration_seconds >= 0

    def test_model_stored_and_retrievable(self):
        """Trained model is stored and retrievable."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)
        assert trainer.get_model(metadata.model_id) is not None
        assert trainer.get_metadata(metadata.model_id) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelEvaluator:
    """Test model evaluation with safety-critical metrics."""

    def test_evaluate_on_test_split(self):
        """Evaluate model on test split."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        config = TrainingConfig(algorithm="xgboost", random_seed=42)
        metadata = trainer.train(dataset, config)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            trainer, metadata.model_id, dataset, SplitType.TEST
        )
        assert metrics.model_id == metadata.model_id
        assert metrics.model_version == metadata.version
        assert metrics.total_samples > 0
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_evaluate_records_safety_metrics(self):
        """Evaluation records false automation and high-value errors."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        assert metrics.false_automation >= 0
        assert metrics.high_value_errors >= 0
        assert metrics.unknown_case_errors >= 0

    def test_evaluate_confusion_matrix(self):
        """Evaluation produces confusion matrix."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        assert len(metrics.confusion_matrix) > 0
        assert len(metrics.confusion_labels) > 0

    def test_evaluate_per_class_metrics(self):
        """Evaluation produces per-class precision/recall/F1."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        assert len(metrics.per_class_precision) > 0
        assert len(metrics.per_class_recall) > 0
        assert len(metrics.per_class_f1) > 0

    def test_evaluate_nonexistent_model_raises(self):
        """Evaluating non-existent model raises error."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(trainer, "MOD-NONE", dataset)

    def test_evaluate_verification_failure_rate(self):
        """Evaluation records verification failure rate."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        assert metrics.verification_failure_rate is not None
        assert 0.0 <= metrics.verification_failure_rate <= 1.0

    def test_evaluate_resolution_accuracy(self):
        """Evaluation records resolution accuracy."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        assert metrics.resolution_accuracy is not None
        assert 0.0 <= metrics.resolution_accuracy <= 1.0

    def test_metrics_summary(self):
        """Metrics summary is readable."""
        dataset = _make_dataset(n_examples=30)
        trainer = ModelTrainer()
        metadata = trainer.train(dataset)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trainer, metadata.model_id, dataset)
        summary = metrics.summary()
        assert "Accuracy:" in summary
        assert "F1" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelComparator:
    """Test model comparison with safety-critical checks."""

    def _make_current(self) -> EvaluationMetrics:
        return EvaluationMetrics(
            model_id="MOD-CURRENT",
            model_version="1.0.0",
            total_samples=100,
            accuracy=0.80,
            precision_macro=0.78,
            recall_macro=0.75,
            f1_macro=0.76,
            precision_weighted=0.80,
            recall_weighted=0.80,
            f1_weighted=0.80,
            incorrect_auto_resolution=5,
            high_value_errors=1,
            false_automation=5,
            unknown_case_errors=2,
            verification_failure_rate=0.05,
            resolution_accuracy=0.85,
        )

    def _make_candidate(self, **overrides) -> EvaluationMetrics:
        current = self._make_current()
        data = current.model_dump()
        data["model_id"] = "MOD-CANDIDATE"
        data["model_version"] = "2.0.0"
        data.update(overrides)
        return EvaluationMetrics(**data)

    def test_candidate_improves_accuracy(self):
        """Better accuracy + same safety → PROMOTE."""
        current = self._make_current()
        candidate = self._make_candidate(accuracy=0.90, f1_macro=0.88)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "PROMOTE"
        assert any("accuracy" in i for i in comparison.improvements)

    def test_candidate_accuracy_below_threshold(self):
        """Accuracy below 50% → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(accuracy=0.30, f1_macro=0.25)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "REJECT"
        assert not comparison.all_safety_passed

    def test_candidate_high_value_error_increase(self):
        """More high-value errors → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(high_value_errors=3)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "REJECT"
        hv_check = [c for c in comparison.safety_checks if c.metric_name == "high_value_errors"]
        assert len(hv_check) == 1
        assert not hv_check[0].passed

    def test_candidate_reduces_false_automation(self):
        """Fewer false automations → improvement."""
        current = self._make_current()
        candidate = self._make_candidate(false_automation=2, accuracy=0.85)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert any("false_automation" in i for i in comparison.improvements)

    def test_candidate_increases_false_automation_critical(self):
        """More false automation from zero → REJECT."""
        current = EvaluationMetrics(
            model_id="MOD-CURRENT", model_version="1.0.0",
            false_automation=0, high_value_errors=0,
        )
        candidate = EvaluationMetrics(
            model_id="MOD-CAND", model_version="2.0.0",
            false_automation=5, high_value_errors=0,
        )
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "REJECT"

    def test_equal_metrics_defer(self):
        """Equal improvements and regressions → DEFER."""
        current = self._make_current()
        candidate = self._make_candidate(
            accuracy=0.85,  # improvement
            precision_macro=0.73,  # regression
        )
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "DEFER"

    def test_only_regressions_reject(self):
        """Only regressions → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(
            accuracy=0.70,  # regression
            f1_macro=0.65,  # regression
            precision_macro=0.68,  # regression
        )
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == "REJECT"

    def test_safety_checks_all_passed(self):
        """All safety checks passed when no regressions."""
        current = self._make_current()
        candidate = self._make_candidate(accuracy=0.90, f1_macro=0.88)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.all_safety_passed is True

    def test_comparison_summary(self):
        """Comparison summary is readable."""
        current = self._make_current()
        candidate = self._make_candidate(accuracy=0.85)
        comparator = ModelComparator()
        comparison = comparator.compare(current, candidate)
        summary = comparison.summary()
        assert "Current:" in summary
        assert "Candidate:" in summary


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Training Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingE2E:
    """End-to-end model training pipeline."""

    def test_full_pipeline(self):
        """Dataset → Train → Evaluate → Compare → Decide."""
        # 1. Create dataset
        dataset = _make_dataset(n_examples=40)

        # 2. Train candidate
        trainer = ModelTrainer()
        config = TrainingConfig(
            algorithm="xgboost",
            random_seed=42,
        )
        candidate_meta = trainer.train(
            dataset, config, model_version="2.0.0", model_name="candidate_v2"
        )

        # 3. Evaluate
        evaluator = ModelEvaluator()
        candidate_metrics = evaluator.evaluate(
            trainer, candidate_meta.model_id, dataset, SplitType.TEST
        )

        # 4. Create "current" baseline (simulate existing model metrics)
        current_metrics = EvaluationMetrics(
            model_id="MOD-CURRENT",
            model_version="1.0.0",
            total_samples=candidate_metrics.total_samples,
            accuracy=0.70,
            precision_macro=0.68,
            recall_macro=0.65,
            f1_macro=0.66,
            false_automation=8,
            high_value_errors=2,
        )

        # 5. Compare
        comparator = ModelComparator()
        comparison = comparator.compare(current_metrics, candidate_metrics)

        # 6. Result
        assert comparison.recommendation in ("PROMOTE", "REJECT", "DEFER")
        assert len(comparison.safety_checks) > 0
        assert comparison.current_version == "1.0.0"
        assert comparison.candidate_version == "2.0.0"

    def test_multiple_algorithms(self):
        """Train and evaluate multiple algorithms."""
        dataset = _make_dataset(n_examples=30)
        evaluator = ModelEvaluator()
        results = {}

        for algo in ["xgboost", "decision_tree", "logistic_regression"]:
            trainer = ModelTrainer()
            config = TrainingConfig(algorithm=algo, random_seed=42)
            meta = trainer.train(dataset, config, model_version=f"1.0-{algo}")
            metrics = evaluator.evaluate(trainer, meta.model_id, dataset)
            results[algo] = metrics

        # All should produce valid metrics
        for algo, metrics in results.items():
            assert metrics.accuracy >= 0.0
            assert metrics.f1_macro >= 0.0
