"""
Pure metric primitives for evaluation (Phase 15).

No production service is imported here. These are small, deterministic,
side-effect-free helpers used by the evaluation modules and tests.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def safe_rate(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def round4(value: float) -> float:
    """Round to 4 decimals so results are byte-stable across runs."""
    return round(float(value) + 0.0, 4)


def confusion(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> List[List[int]]:
    """Confusion matrix (rows = actual, cols = predicted) over ``labels``."""
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for true, pred in zip(y_true, y_pred):
        i = index.get(true)
        j = index.get(pred)
        if i is None or j is None:
            continue
        matrix[i][j] += 1
    return matrix


def binary_counts(matrix: List[List[int]], label_index: int) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for one class in a multiclass confusion matrix."""
    tp = matrix[label_index][label_index]
    fp = sum(row[label_index] for i, row in enumerate(matrix) if i != label_index)
    fn = sum(v for j, v in enumerate(matrix[label_index]) if j != label_index)
    total = sum(sum(row) for row in matrix)
    tn = total - tp - fp - fn
    return tp, fp, fn, tn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall and F1 from raw counts (0.0 when undefined)."""
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    return round4(precision), round4(recall), round4(f1)


def per_class_metrics(
    matrix: List[List[int]],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 / support / FPR / FNR."""
    out: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp, fp, fn, tn = binary_counts(matrix, i)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "false_positive_rate": round4(safe_rate(fp, fp + tn)),
            "false_negative_rate": round4(safe_rate(fn, fn + tp)),
        }
    return out


def macro_average(per_class: Dict[str, Dict[str, float]], key: str) -> float:
    """Unweighted mean of a per-class metric (macro average)."""
    if not per_class:
        return 0.0
    return round4(sum(v.get(key, 0.0) for v in per_class.values()) / len(per_class))


def weighted_average(per_class: Dict[str, Dict[str, float]], key: str) -> float:
    """Support-weighted mean of a per-class metric (weighted average)."""
    total = sum(v.get("support", 0.0) for v in per_class.values())
    if total == 0:
        return 0.0
    acc = sum(v.get(key, 0.0) * v.get("support", 0.0) for v in per_class.values())
    return round4(acc / total)


def accuracy(matrix: List[List[int]]) -> float:
    """Overall accuracy from a confusion matrix."""
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return round4(safe_rate(correct, total))


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 5,
) -> Dict[str, object]:
    """Expected calibration error with equal-width confidence bins."""
    if not confidences:
        return {"ece": 0.0, "bins": []}
    edges = [i / bins for i in range(bins + 1)]
    detail = []
    ece = 0.0
    n = len(confidences)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        members = [
            (c, ok)
            for c, ok in zip(confidences, correct)
            if (lo < c <= hi) or (b == 0 and c <= lo)
        ]
        if not members:
            detail.append({"range": [lo, hi], "count": 0, "avg_confidence": 0.0, "accuracy": 0.0})
            continue
        avg_conf = sum(c for c, _ in members) / len(members)
        acc = sum(1 for _, ok in members if ok) / len(members)
        gap = abs(avg_conf - acc)
        ece += gap * (len(members) / n)
        detail.append(
            {
                "range": [round(lo, 4), round(hi, 4)],
                "count": len(members),
                "avg_confidence": round4(avg_conf),
                "accuracy": round4(acc),
            }
        )
    return {"ece": round4(ece), "bins": detail}


def confidence_threshold_table(
    confidences: Sequence[float],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Selective-prediction table: coverage / accuracy / macro F1 / error rate."""
    labels = list(labels or sorted(set(y_true) | set(y_pred)))
    rows: List[Dict[str, object]] = []
    n = len(confidences)
    for t in thresholds:
        idx = [i for i, c in enumerate(confidences) if c >= t]
        if not idx:
            rows.append(
                {
                    "threshold": round4(t),
                    "coverage": 0.0,
                    "samples": 0,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "error_rate": 0.0,
                }
            )
            continue
        t_true = [y_true[i] for i in idx]
        t_pred = [y_pred[i] for i in idx]
        cm = confusion(t_true, t_pred, labels)
        pcm = per_class_metrics(cm, labels)
        rows.append(
            {
                "threshold": round4(t),
                "coverage": round4(safe_rate(len(idx), n)),
                "samples": len(idx),
                "accuracy": accuracy(cm),
                "macro_f1": macro_average(pcm, "f1"),
                "error_rate": round4(1.0 - accuracy(cm)),
            }
        )
    return rows


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Recall@K for a single query."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return safe_rate(len(set(retrieved[:k]) & rel), len(rel))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Precision@K for a single query."""
    if k <= 0:
        return 0.0
    return safe_rate(len(set(retrieved[:k]) & set(relevant)), k)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result (0.0 when absent)."""
    rel = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in rel:
            return round4(1.0 / (i + 1))
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty input."""
    if not values:
        return 0.0
    return round4(sum(values) / len(values))


def category_rate(flags: Sequence[bool]) -> float:
    """Fraction of True values."""
    return round4(safe_rate(sum(1 for f in flags if f), len(flags)))
