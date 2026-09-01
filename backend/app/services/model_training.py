"""
Model Training services for Razorpay CloseLoop Phase 9E.

Implements candidate model training, evaluation, and baseline comparison.

Safety principle:
  A trained candidate is a CANDIDATE.
  It is NOT automatically promoted to production.
  Phase 6 hard safety constraints remain mandatory.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import LabelEncoder

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


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_features_and_labels(
    examples: List[LearningExample],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract feature matrix and label array from learning examples.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label array (n_samples,)
        feature_names: ordered feature names
        label_classes: ordered unique labels
    """
    if not examples:
        return np.array([]), np.array([]), [], []

    # Collect all feature names
    all_features: set = set()
    for ex in examples:
        all_features.update(ex.features.to_flat_dict().keys())
    feature_names = sorted(all_features)

    # Build feature matrix
    X = np.array([
        [ex.features.to_flat_dict().get(f, 0.0) for f in feature_names]
        for ex in examples
    ], dtype=np.float64)

    # Build label array
    labels = []
    for ex in examples:
        lbl = ex.labels.true_exception_type
        labels.append(lbl if lbl else "UNKNOWN")

    label_classes = sorted(set(labels))
    y = np.array(labels)

    return X, y, feature_names, label_classes


# ─────────────────────────────────────────────────────────────────────────────
# Model Trainer
# ─────────────────────────────────────────────────────────────────────────────


class ModelTrainer:
    """Trains candidate models from learning datasets.

    Supports XGBoost, Decision Tree, and Logistic Regression.
    Training is deterministic given the same seed and data.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}

    def train(
        self,
        dataset: LearningDataset,
        config: Optional[TrainingConfig] = None,
        model_version: str = "1.0.0",
        model_name: str = "candidate_model",
    ) -> ModelMetadata:
        """Train a candidate model from a learning dataset.

        Args:
            dataset: The learning dataset with examples and splits.
            config: Training configuration.
            model_version: Version string for the new model.
            model_name: Human-readable name.

        Returns:
            ModelMetadata with training results.
        """
        config = config or TrainingConfig()

        # Extract training data from the train split
        train_examples = dataset.get_examples_by_split(SplitType.TRAIN)
        if not train_examples:
            # Fall back to all examples if no split exists
            train_examples = [e for e in dataset.examples if e.is_valid()]

        if len(train_examples) < 2:
            raise ValueError(
                f"Insufficient training examples: {len(train_examples)}"
            )

        X, y, feature_names, label_classes = extract_features_and_labels(train_examples)

        # Train model
        start_time = time.time()
        model = self._train_model(X, y, config, label_classes)
        duration = time.time() - start_time

        # Build metadata
        model_id = _gen_id("MOD")
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=model_version,
            model_type=config.model_type,
            status=ModelStatus.CANDIDATE,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            feature_schema_version=dataset.feature_schema_version,
            training_examples=len(train_examples),
            feature_count=len(feature_names),
            feature_names=feature_names,
            label_classes=label_classes,
            config=config,
            trained_at=datetime.utcnow(),
            training_duration_seconds=duration,
        )

        self._models[model_id] = model
        self._metadata[model_id] = metadata
        return metadata

    def get_model(self, model_id: str) -> Optional[Any]:
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        return self._metadata.get(model_id)

    def predict(
        self, model_id: str, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run prediction using a trained model.

        Returns:
            predictions: predicted labels (strings)
            probabilities: class probabilities (if supported)
        """
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        raw_predictions = model.predict(X)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)

        # Decode integer predictions back to strings (for XGBoost)
        if hasattr(model, "_label_encoder"):
            predictions = model._label_encoder.inverse_transform(
                raw_predictions.astype(int)
            )
        else:
            predictions = raw_predictions

        return predictions, probabilities

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        label_classes: List[str],
    ) -> Any:
        """Train the actual ML model."""
        if config.algorithm == "xgboost":
            return self._train_xgboost(X, y, config, label_classes)
        elif config.algorithm == "decision_tree":
            return self._train_decision_tree(X, y, config)
        elif config.algorithm == "logistic_regression":
            return self._train_logistic_regression(X, y, config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    def _train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        label_classes: List[str],
    ) -> Any:
        """Train XGBoost classifier."""
        import xgboost as xgb

        # XGBoost requires integer-encoded labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        n_classes = len(label_classes)
        params = {
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
            "num_class": n_classes if n_classes > 2 else None,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "max_depth": config.hyperparameters.get("max_depth", 6),
            "learning_rate": config.hyperparameters.get("learning_rate", 0.1),
            "n_estimators": config.hyperparameters.get("n_estimators", 100),
            "random_state": config.random_seed,
            "verbosity": 0,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        model = xgb.XGBClassifier(**params)
        model.fit(X, y_encoded)
        # Store encoder for prediction
        model._label_encoder = le
        return model

    def _train_decision_tree(
        self, X: np.ndarray, y: np.ndarray, config: TrainingConfig
    ) -> Any:
        """Train Decision Tree classifier."""
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            max_depth=config.hyperparameters.get("max_depth", 10),
            random_state=config.random_seed,
            class_weight=config.class_weight,
        )
        model.fit(X, y)
        return model

    def _train_logistic_regression(
        self, X: np.ndarray, y: np.ndarray, config: TrainingConfig
    ) -> Any:
        """Train Logistic Regression classifier."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=config.hyperparameters.get("max_iter", 1000),
            random_state=config.random_seed,
            class_weight=config.class_weight,
        )
        model.fit(X, y)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluator
# ─────────────────────────────────────────────────────────────────────────────


class ModelEvaluator:
    """Evaluates models with standard + safety-critical metrics."""

    def evaluate(
        self,
        trainer: ModelTrainer,
        model_id: str,
        dataset: LearningDataset,
        split: SplitType = SplitType.TEST,
        high_value_threshold: int = 100000,
    ) -> EvaluationMetrics:
        """Evaluate a model on a dataset split.

        Args:
            trainer: The model trainer containing the trained model.
            model_id: ID of the model to evaluate.
            dataset: Dataset with examples and splits.
            split: Which split to evaluate on.
            high_value_threshold: Threshold for high-value errors.

        Returns:
            Comprehensive evaluation metrics.
        """
        metadata = trainer.get_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        # Get evaluation examples
        eval_examples = dataset.get_examples_by_split(split)
        if not eval_examples:
            eval_examples = [e for e in dataset.examples if e.is_valid()]

        if not eval_examples:
            return EvaluationMetrics(
                model_id=model_id,
                model_version=metadata.version,
                evaluated_on=split.value,
            )

        X, y_true, feature_names, label_classes = extract_features_and_labels(
            eval_examples
        )

        # Get predictions
        y_pred, probabilities = trainer.predict(model_id, X)

        # Standard metrics
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        accuracy = float(accuracy_score(y_true, y_pred))
        precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro_val = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        per_class_support = {}
        for cls in label_classes:
            if cls in report:
                per_class_precision[cls] = float(report[cls]["precision"])
                per_class_recall[cls] = float(report[cls]["recall"])
                per_class_f1[cls] = float(report[cls]["f1-score"])
                per_class_support[cls] = int(report[cls]["support"])

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=label_classes)
        cm_list = cm.tolist()

        # Safety-critical metrics
        incorrect_auto = 0
        high_value_errors = 0
        unknown_errors = 0
        false_auto = 0

        for i, ex in enumerate(eval_examples):
            predicted = y_pred[i]
            actual = y_true[i]
            is_correct = predicted == actual

            # Count incorrect predictions as "false automation"
            if not is_correct:
                false_auto += 1
                incorrect_auto += 1

                # High-value errors
                adj = ex.features.financial_features.get(
                    "actual_adjustment_paise", 0
                )
                if adj >= high_value_threshold:
                    high_value_errors += 1

                # Unknown case errors
                if actual == "UNKNOWN" or predicted == "UNKNOWN":
                    unknown_errors += 1

        # Verification failure rate
        ver_fail_count = sum(
            1 for ex in eval_examples
            if not ex.labels.verification_passed
        )
        ver_fail_rate = (
            ver_fail_count / len(eval_examples) if eval_examples else None
        )

        # Resolution accuracy
        resolution_correct = sum(
            1 for ex in eval_examples
            if ex.labels.resolution_correct is True
        )
        resolution_with_labels = sum(
            1 for ex in eval_examples
            if ex.labels.resolution_correct is not None
        )
        resolution_accuracy = (
            resolution_correct / resolution_with_labels
            if resolution_with_labels > 0
            else None
        )

        return EvaluationMetrics(
            model_id=model_id,
            model_version=metadata.version,
            evaluated_on=split.value,
            total_samples=len(eval_examples),
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro_val,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            confusion_matrix=cm_list,
            confusion_labels=label_classes,
            incorrect_auto_resolution=incorrect_auto,
            high_value_errors=high_value_errors,
            unknown_case_errors=unknown_errors,
            false_automation=false_auto,
            verification_failure_rate=ver_fail_rate,
            resolution_accuracy=resolution_accuracy,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparator
# ─────────────────────────────────────────────────────────────────────────────


class ModelComparator:
    """Compares current vs candidate model with safety-critical checks."""

    # Safety thresholds
    MIN_ACCURACY = 0.50
    MAX_FALSE_AUTO_INCREASE_RATIO = 1.20  # Max 20% increase
    NO_HIGH_VALUE_ERROR_INCREASE = True
    MIN_F1_MACRO = 0.40

    def compare(
        self,
        current: EvaluationMetrics,
        candidate: EvaluationMetrics,
        high_value_threshold: int = 100000,
    ) -> CandidateModelComparison:
        """Compare current and candidate model metrics."""
        improvements: List[str] = []
        regressions: List[str] = []
        safety_checks: List[SafetyMetricCheck] = []

        # 1. Accuracy check
        acc_check = SafetyMetricCheck(
            metric_name="accuracy",
            current_value=current.accuracy,
            candidate_value=candidate.accuracy,
            passed=candidate.accuracy >= self.MIN_ACCURACY,
            threshold=self.MIN_ACCURACY,
            description=f"Candidate accuracy {candidate.accuracy:.1%} vs minimum {self.MIN_ACCURACY:.1%}",
        )
        safety_checks.append(acc_check)

        if candidate.accuracy > current.accuracy:
            improvements.append(
                f"accuracy: {current.accuracy:.1%} → {candidate.accuracy:.1%}"
            )
        elif candidate.accuracy < current.accuracy:
            regressions.append(
                f"accuracy: {current.accuracy:.1%} → {candidate.accuracy:.1%}"
            )

        # 2. F1 macro check
        f1_check = SafetyMetricCheck(
            metric_name="f1_macro",
            current_value=current.f1_macro,
            candidate_value=candidate.f1_macro,
            passed=candidate.f1_macro >= self.MIN_F1_MACRO,
            threshold=self.MIN_F1_MACRO,
            description=f"Candidate F1(macro) {candidate.f1_macro:.1%} vs minimum {self.MIN_F1_MACRO:.1%}",
        )
        safety_checks.append(f1_check)

        if candidate.f1_macro > current.f1_macro:
            improvements.append(
                f"f1_macro: {current.f1_macro:.1%} → {candidate.f1_macro:.1%}"
            )
        elif candidate.f1_macro < current.f1_macro:
            regressions.append(
                f"f1_macro: {current.f1_macro:.1%} → {candidate.f1_macro:.1%}"
            )

        # 3. False automation check
        if current.false_automation > 0:
            increase_ratio = (
                candidate.false_automation / current.false_automation
            )
            fa_check = SafetyMetricCheck(
                metric_name="false_automation",
                current_value=float(current.false_automation),
                candidate_value=float(candidate.false_automation),
                passed=increase_ratio <= self.MAX_FALSE_AUTO_INCREASE_RATIO,
                threshold=float(
                    current.false_automation * self.MAX_FALSE_AUTO_INCREASE_RATIO
                ),
                description=(
                    f"False automation: {current.false_automation} → {candidate.false_automation} "
                    f"(ratio: {increase_ratio:.2f})"
                ),
            )
        else:
            fa_check = SafetyMetricCheck(
                metric_name="false_automation",
                current_value=0.0,
                candidate_value=float(candidate.false_automation),
                passed=candidate.false_automation == 0,
                threshold=0.0,
                description=(
                    f"False automation from zero: {candidate.false_automation}"
                ),
            )
        safety_checks.append(fa_check)

        if candidate.false_automation < current.false_automation:
            improvements.append(
                f"false_automation: {current.false_automation} → {candidate.false_automation}"
            )
        elif candidate.false_automation > current.false_automation:
            regressions.append(
                f"false_automation: {current.false_automation} → {candidate.false_automation}"
            )

        # 4. High-value errors — no increase allowed
        hv_check = SafetyMetricCheck(
            metric_name="high_value_errors",
            current_value=float(current.high_value_errors),
            candidate_value=float(candidate.high_value_errors),
            passed=candidate.high_value_errors <= current.high_value_errors,
            threshold=float(current.high_value_errors),
            description=(
                f"High-value errors: {current.high_value_errors} → {candidate.high_value_errors}"
            ),
        )
        safety_checks.append(hv_check)

        if candidate.high_value_errors < current.high_value_errors:
            improvements.append(
                f"high_value_errors: {current.high_value_errors} → {candidate.high_value_errors}"
            )
        elif candidate.high_value_errors > current.high_value_errors:
            regressions.append(
                f"high_value_errors: {current.high_value_errors} → {candidate.high_value_errors}"
            )

        # 5. Unknown case errors
        if candidate.unknown_case_errors > current.unknown_case_errors:
            regressions.append(
                f"unknown_case_errors: {current.unknown_case_errors} → {candidate.unknown_case_errors}"
            )
        elif candidate.unknown_case_errors < current.unknown_case_errors:
            improvements.append(
                f"unknown_case_errors: {current.unknown_case_errors} → {candidate.unknown_case_errors}"
            )

        # 6. Precision / recall
        if candidate.precision_macro > current.precision_macro:
            improvements.append(
                f"precision_macro: {current.precision_macro:.1%} → {candidate.precision_macro:.1%}"
            )
        elif candidate.precision_macro < current.precision_macro:
            regressions.append(
                f"precision_macro: {current.precision_macro:.1%} → {candidate.precision_macro:.1%}"
            )

        if candidate.recall_macro > current.recall_macro:
            improvements.append(
                f"recall_macro: {current.recall_macro:.1%} → {candidate.recall_macro:.1%}"
            )
        elif candidate.recall_macro < current.recall_macro:
            regressions.append(
                f"recall_macro: {current.recall_macro:.1%} → {candidate.recall_macro:.1%}"
            )

        # 7. Verification failure rate
        if (
            candidate.verification_failure_rate is not None
            and current.verification_failure_rate is not None
        ):
            if candidate.verification_failure_rate > current.verification_failure_rate:
                regressions.append(
                    f"verification_failure_rate: {current.verification_failure_rate:.1%} → "
                    f"{candidate.verification_failure_rate:.1%}"
                )
            elif candidate.verification_failure_rate < current.verification_failure_rate:
                improvements.append(
                    f"verification_failure_rate: {current.verification_failure_rate:.1%} → "
                    f"{candidate.verification_failure_rate:.1%}"
                )

        # Determine all safety passed
        all_safety_passed = all(check.passed for check in safety_checks)

        # Determine recommendation
        critical_failures = [
            c for c in safety_checks
            if not c.passed and c.metric_name in (
                "accuracy", "f1_macro", "false_automation", "high_value_errors"
            )
        ]

        if critical_failures:
            recommendation = "REJECT"
            reason = (
                f"Rejected: {len(critical_failures)} critical safety check(s) failed"
            )
        elif not all_safety_passed:
            recommendation = "DEFER"
            reason = "Deferred: safety checks need investigation"
        elif len(improvements) > len(regressions):
            recommendation = "PROMOTE"
            reason = (
                f"Promoted: {len(improvements)} improvements, "
                f"{len(regressions)} regressions, all safety checks passed"
            )
        elif len(improvements) == len(regressions):
            recommendation = "DEFER"
            reason = "Deferred: equal improvements and regressions"
        else:
            recommendation = "REJECT"
            reason = (
                f"Rejected: {len(regressions)} regressions "
                f"outweigh {len(improvements)} improvements"
            )

        return CandidateModelComparison(
            current_model_id=current.model_id,
            current_version=current.model_version,
            candidate_model_id=candidate.model_id,
            candidate_version=candidate.model_version,
            current_metrics=current,
            candidate_metrics=candidate,
            safety_checks=safety_checks,
            all_safety_passed=all_safety_passed,
            improvements=improvements,
            regressions=regressions,
            recommendation=recommendation,
            recommendation_reason=reason,
        )
