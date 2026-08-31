"""
Exception type classifier for Razorpay CloseLoop Phase 4.

Provides:
- Baseline classifiers (majority class, logistic regression)
- XGBoost multiclass classifier
- Training pipeline with evaluation
- Clean inference interface
- Model artifact save/load

Deterministic reconciliation remains the source of financial truth.
This classifier predicts a category — it does NOT modify financial amounts.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from app.schemas.enums import ExceptionType
from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION, MLSample


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_VERSION = "1.0.0"
ALL_LABELS = [e.value for e in ExceptionType]
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────


class DatasetBuilder:
    """
    Builds numpy arrays from MLSample lists for model training/evaluation.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ALL_LABELS)

    def build(
        self, samples: List[MLSample]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert MLSample list to numpy arrays.

        Returns:
            (X, y, label_names) where X is features, y is encoded labels,
            label_names is the ordered class names.
        """
        X = np.array(
            [s.features.to_list(self.feature_names) for s in samples],
            dtype=np.float32,
        )
        y_str = [s.labels.true_exception_type.value for s in samples]
        y = self.label_encoder.transform(y_str)

        return X, y, list(self.label_encoder.classes_)

    def build_single(self, sample: MLSample) -> np.ndarray:
        """Convert a single MLSample to a feature array."""
        return np.array(
            [sample.features.to_list(self.feature_names)], dtype=np.float32
        )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Classifiers
# ─────────────────────────────────────────────────────────────────────────────


class MajorityClassClassifier:
    """
    Simple baseline: always predicts the most common class.
    """

    def __init__(self):
        self.majority_class: Optional[str] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ALL_LABELS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train by finding the majority class."""
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = self.label_encoder.inverse_transform(
            [unique[np.argmax(counts)]]
        )[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict majority class for all samples."""
        idx = self.label_encoder.transform([self.majority_class])[0]
        return np.full(len(X), idx, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability 1.0 for majority class."""
        proba = np.zeros((len(X), len(self.label_encoder.classes_)), dtype=np.float32)
        idx = self.label_encoder.transform([self.majority_class])[0]
        proba[:, idx] = 1.0
        return proba


class LogisticRegressionClassifier:
    """
    Baseline: logistic regression with balanced class weights.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ALL_LABELS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train logistic regression."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.model.predict_proba(X)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────


class ExceptionClassifier:
    """
    XGBoost multiclass exception type classifier.

    This is a production-oriented baseline model, not a Kaggle optimization.
    Uses fixed random seed, reasonable hyperparameters, and class weighting
    for imbalanced data.
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        import xgboost as xgb

        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            tree_method="hist",
            verbosity=0,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ALL_LABELS)
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train the XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels (encoded)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional)
            feature_names: Feature names for feature importance

        Returns:
            Training metadata dict
        """
        self.feature_names = feature_names or []

        # Compute class weights for imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {int(c): total / (len(unique) * int(cnt)) for c, cnt in zip(unique, counts)}
        sample_weights = np.array([class_weights[int(y)] for y in y_train])

        # Train with sample weights
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True

        # Compute class distribution
        class_dist = {self.label_encoder.inverse_transform([int(c)])[0]: int(cnt) for c, cnt in zip(unique, counts)}

        return {
            "n_samples": len(y_train),
            "n_features": X_train.shape[1],
            "n_classes": len(unique),
            "class_distribution": class_dist,
            "class_weights": {self.label_encoder.inverse_transform([int(c)])[0]: round(w, 4) for c, w in class_weights.items()},
            "training_time": datetime.utcnow().isoformat(),
            "seed": self.seed,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict exception type classes."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.model.predict_proba(X)

    def predict_labels(self, X: np.ndarray) -> List[str]:
        """Predict and return human-readable labels."""
        preds = self.predict(X)
        return self.label_encoder.inverse_transform(preds).tolist()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted or not self.feature_names:
            return {}
        importance = self.model.feature_importances_
        return {
            name: round(float(imp), 6)
            for name, imp in zip(self.feature_names, importance)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


class ModelEvaluator:
    """
    Evaluates classifier performance with comprehensive metrics.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str],
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.

        Returns dict with accuracy, precision, recall, F1, per-class metrics,
        confusion matrix, and error analysis.
        """
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(label_names):
            mask_true = y_true == i
            mask_pred = y_pred == i
            tp = int(np.sum(mask_true & mask_pred))
            fp = int(np.sum(~mask_true & mask_pred))
            fn = int(np.sum(mask_true & ~mask_pred))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(mask_true)),
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_list = cm.tolist()

        # Error analysis
        errors = []
        for i, true_name in enumerate(label_names):
            for j, pred_name in enumerate(label_names):
                if i != j and cm[i][j] > 0:
                    errors.append({
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(cm[i][j]),
                    })
        errors.sort(key=lambda x: x["count"], reverse=True)

        # Find weak classes (F1 < 0.5)
        weak_classes = [name for name, m in per_class.items() if m["f1"] < 0.5 and m["support"] > 0]

        return {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class": per_class,
            "confusion_matrix": cm_list,
            "label_names": label_names,
            "top_errors": errors[:10],
            "weak_classes": weak_classes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Artifact
# ─────────────────────────────────────────────────────────────────────────────


class ModelArtifact:
    """
    Manages saving and loading of trained model artifacts.

    Saves:
    - trained model
    - feature schema/version
    - label mapping
    - training metadata
    - evaluation metrics
    """

    @staticmethod
    def save(
        model,
        path: str,
        feature_names: List[str],
        label_names: List[str],
        training_metadata: Dict,
        evaluation: Dict,
        classifier_type: str = "xgboost",
    ) -> None:
        """Save model artifact to disk."""
        os.makedirs(path, exist_ok=True)

        # Save model
        model_path = os.path.join(path, "model.joblib")
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "classifier_type": classifier_type,
            "classifier_version": CLASSIFIER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": feature_names,
            "label_names": label_names,
            "training_metadata": training_metadata,
            "evaluation": evaluation,
            "saved_at": datetime.utcnow().isoformat(),
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load(path: str) -> Tuple[object, Dict]:
        """Load model artifact from disk. Returns (model, metadata)."""
        model_path = os.path.join(path, "model.joblib")
        metadata_path = os.path.join(path, "metadata.json")

        model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return model, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Inference Interface
# ─────────────────────────────────────────────────────────────────────────────


class ExceptionPrediction:
    """Clean prediction output for a single sample."""

    def __init__(
        self,
        predicted_type: str,
        probabilities: Dict[str, float],
        model_version: str,
    ):
        self.predicted_type = predicted_type
        self.probabilities = probabilities
        self.model_version = model_version

    def to_dict(self) -> Dict:
        return {
            "predicted_type": self.predicted_type,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
        }


class ExceptionClassifierService:
    """
    Clean inference service for exception type prediction.

    Wraps a trained model with a clean interface.
    Does NOT modify any financial amounts.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        label_names: List[str],
        model_version: str = CLASSIFIER_VERSION,
    ):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        self.model_version = model_version

    def predict(self, feature_dict: Dict[str, float]) -> ExceptionPrediction:
        """
        Predict exception type from a feature dictionary.

        Args:
            feature_dict: Feature name → value mapping

        Returns:
            ExceptionPrediction with predicted type and probabilities
        """
        # Convert to array in correct order
        X = np.array(
            [[feature_dict.get(name, 0.0) for name in self.feature_names]],
            dtype=np.float32,
        )

        # Predict
        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        predicted_type = self.label_names[int(pred_idx)]
        probabilities = {
            name: round(float(p), 6)
            for name, p in zip(self.label_names, proba)
        }

        return ExceptionPrediction(
            predicted_type=predicted_type,
            probabilities=probabilities,
            model_version=self.model_version,
        )

    def predict_batch(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[ExceptionPrediction]:
        """Predict for multiple samples."""
        return [self.predict(fd) for fd in feature_dicts]

    @classmethod
    def from_artifact(cls, path: str) -> "ExceptionClassifierService":
        """Load service from saved artifact."""
        model, metadata = ModelArtifact.load(path)
        return cls(
            model=model,
            feature_names=metadata["feature_names"],
            label_names=metadata["label_names"],
            model_version=metadata.get("classifier_version", CLASSIFIER_VERSION),
        )
