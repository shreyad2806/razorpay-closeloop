"""
ML classification evaluation (Phase 15).

Uses the **production** XGBoost classifier (:class:`app.ml.classifier.
ExceptionClassifier`) with its production random seed. The model architecture,
features and hyperparameters are untouched — this module only measures.

Because the system is safety-sensitive, accuracy alone is never reported as the
verdict. Per-class metrics, false-positive/negative rates, calibration and a
selective-prediction table are computed alongside it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvaluationCase, EvaluationDataset, FEATURE_NAMES
from app.evaluation.metrics import (
    accuracy as cm_accuracy,
    confusion,
    confidence_threshold_table,
    expected_calibration_error,
    macro_average,
    per_class_metrics,
    round4,
    weighted_average,
)
from app.evaluation.versioning import DEFAULT_EVAL_SEED, EVALUATOR_VERSION


class ConfidenceThresholdRow(BaseModel):
    threshold: float
    coverage: float
    samples: int
    accuracy: float
    macro_f1: float
    error_rate: float


class MLClassificationReport(BaseModel):
    """Result of evaluating the production classifier on a dataset split."""

    evaluator_version: str = EVALUATOR_VERSION
    seed: int = DEFAULT_EVAL_SEED
    dataset_version: str = ""
    dataset_source: str = ""
    model_version: str = "exception_classifier-xgboost"
    feature_schema_version: str = ""

    train_samples: int = 0
    test_samples: int = 0
    label_count: int = 0

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0

    per_class: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    confusion_labels: List[str] = Field(default_factory=list)

    false_positive_rate_macro: float = 0.0
    false_negative_rate_macro: float = 0.0
    unknown_handling: Dict[str, float] = Field(default_factory=dict)

    mean_confidence: float = 0.0
    confidence_bins: List[Dict[str, Any]] = Field(default_factory=list)
    calibration_ece: float = 0.0
    confidence_thresholds: List[ConfidenceThresholdRow] = Field(default_factory=list)

    production_evaluator: Dict[str, object] = Field(default_factory=dict)
    split_strategy: str = "temporal"
    notes: List[str] = Field(default_factory=list)


def train_production_classifier(
    cases: Sequence[EvaluationCase],
    feature_names: Sequence[str] = tuple(FEATURE_NAMES),
    seed: int = DEFAULT_EVAL_SEED,
):
    """Fit the production classifier on a set of cases.

    Returns ``(model, feature_names, label_universe)``.

    Evaluation-specific detail: the production classifier presets its label
    encoder to the full exception taxonomy, but XGBoost requires the encoded
    labels to be contiguous from 0. A temporal split may omit classes, so the
    evaluation fits the encoder to the classes actually present in *training*
    data. No model architecture, hyperparameter or production code is changed —
    only which label space this evaluation run uses.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    from app.ml.classifier import ExceptionClassifier

    names = list(feature_names)
    labels = [c.exception_type for c in cases]
    X = np.array([[c.features.get(n, 0.0) for n in names] for c in cases], dtype=np.float64)

    model = ExceptionClassifier(seed=seed)
    encoder = LabelEncoder().fit(sorted(set(labels)))
    model.label_encoder = encoder
    y = encoder.transform(labels)

    model.fit(X, y, feature_names=names)
    return model, names, list(encoder.classes_)


def _production_evaluator_report(y_true_int: Sequence[int], y_pred_int: Sequence[int]) -> Dict[str, object]:
    """Reuse the production ModelEvaluator when the split permits it.

    The production evaluator indexes its confusion matrix by the full label
    universe, so it is only invoked when every label is represented. Otherwise
    the evaluation record documents why it was skipped rather than crashing.
    """
    try:
        from app.ml.classifier import ALL_LABELS, ModelEvaluator

        full = set(range(len(ALL_LABELS)))
        if (set(y_true_int) | set(y_pred_int)) != full:
            return {
                "available": False,
                "reason": "not every class is represented in this split",
            }
        return {"available": True, "metrics": ModelEvaluator.evaluate(list(y_true_int), list(y_pred_int), list(ALL_LABELS))}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": type(exc).__name__}


def evaluate_classifier(
    dataset: EvaluationDataset,
    seed: int = DEFAULT_EVAL_SEED,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> MLClassificationReport:
    """Train on the train split and evaluate on the test split."""
    import numpy as np

    from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION

    train_cases = dataset.cases_in_split("train")
    test_cases = dataset.cases_in_split("test")
    if not train_cases or not test_cases:  # pragma: no cover - defensive
        return MLClassificationReport(
            dataset_version=dataset.dataset_version,
            dataset_source=dataset.source,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            notes=["dataset produced an empty train or test split"],
        )

    model, names, all_labels = train_production_classifier(train_cases, seed=seed)

    X_test = np.array([[c.features.get(n, 0.0) for n in names] for c in test_cases], dtype=np.float64)
    y_true = [c.exception_type for c in test_cases]
    y_pred_int = list(model.predict(X_test))
    encoder = model.label_encoder
    y_pred = [encoder.inverse_transform([int(v)])[0] for v in y_pred_int]

    # ── Labels actually present in the test split drive the metric universe ──
    labels = sorted(set(y_true) | set(y_pred))

    matrix = confusion(y_true, y_pred, labels)
    per_class = per_class_metrics(matrix, labels)

    probs = model.predict_proba(X_test)
    classes = list(encoder.classes_)
    confidences: List[float] = []
    correct: List[bool] = []
    for i, row in enumerate(probs):
        best = int(np.argmax(row))
        confidences.append(round4(float(row[best])))
        predicted_label = classes[best] if best < len(classes) else "UNKNOWN"
        correct.append(predicted_label == y_true[i])

    threshold_rows = [
        ConfidenceThresholdRow(**row) for row in confidence_threshold_table(confidences, y_true, y_pred, thresholds, labels)
    ]
    ece = expected_calibration_error(confidences, correct)

    unknown_handling = {
        "actual_unknown": float(sum(1 for t in y_true if t == "UNKNOWN")),
        "predicted_unknown": float(sum(1 for p in y_pred if p == "UNKNOWN")),
        "unknown_recall": per_class.get("UNKNOWN", {}).get("recall", 0.0),
        "unknown_precision": per_class.get("UNKNOWN", {}).get("precision", 0.0),
    }

    y_true_int = [classes.index(t) if t in classes else -1 for t in y_true]

    return MLClassificationReport(
        dataset_version=dataset.dataset_version,
        dataset_source=dataset.source,
        seed=seed,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        train_samples=len(train_cases),
        test_samples=len(test_cases),
        label_count=len(labels),
        accuracy=cm_accuracy(matrix),
        precision_macro=macro_average(per_class, "precision"),
        recall_macro=macro_average(per_class, "recall"),
        f1_macro=macro_average(per_class, "f1"),
        precision_weighted=weighted_average(per_class, "precision"),
        recall_weighted=weighted_average(per_class, "recall"),
        f1_weighted=weighted_average(per_class, "f1"),
        per_class=per_class,
        confusion_matrix=matrix,
        confusion_labels=labels,
        false_positive_rate_macro=macro_average(per_class, "false_positive_rate"),
        false_negative_rate_macro=macro_average(per_class, "false_negative_rate"),
        unknown_handling=unknown_handling,
        mean_confidence=round4(sum(confidences) / len(confidences)) if confidences else 0.0,
        confidence_bins=list(ece["bins"]),  # type: ignore[arg-type]
        calibration_ece=float(ece["ece"]),
        confidence_thresholds=threshold_rows,
        production_evaluator=_production_evaluator_report(y_true_int, y_pred_int),
        split_strategy=dataset.split_strategy,
        notes=[
            "Synthetic evaluation dataset — these numbers are fixture performance, "
            "not production performance.",
            f"Trained on {len(train_cases)} cases, evaluated on {len(test_cases)} cases.",
        ],
    )


def predictions_for_cases(
    dataset: EvaluationDataset,
    cases: Sequence[EvaluationCase],
    seed: int = DEFAULT_EVAL_SEED,
) -> Dict[str, Tuple[str, float]]:
    """Return {case_id: (predicted_label, confidence)} for arbitrary cases."""
    import numpy as np

    train_cases = dataset.cases_in_split("train")
    model, names, _ = train_production_classifier(train_cases, seed=seed)
    X = np.array([[c.features.get(n, 0.0) for n in names] for c in cases], dtype=np.float64)
    probs = model.predict_proba(X)
    classes = list(model.label_encoder.classes_)
    out: Dict[str, Tuple[str, float]] = {}
    for i, case in enumerate(cases):
        row = probs[i]
        best = int(np.argmax(row))
        out[case.case_id] = (classes[best], round4(float(row[best])))
    return out
