#!/usr/bin/env python3
"""
Train the exception type classifier on synthetic financial data.

Loads Phase 1 ground truth + generates synthetic features,
trains baseline + XGBoost classifiers, evaluates, and saves artifacts.

Usage:
    python scripts/train_classifier.py [--batch-dir data] [--output models]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.schemas.enums import ExceptionType
from app.schemas.ml_dataset import MLSample, MLLabels, FeatureVector, FEATURE_SCHEMA_VERSION
from app.ml.features import ML_FEATURE_SCHEMA, extract_features
from app.ml.classifier import (
    DatasetBuilder,
    MajorityClassClassifier,
    LogisticRegressionClassifier,
    ExceptionClassifier,
    ModelEvaluator,
    ModelArtifact,
)

ALL_LABELS = [e.value for e in ExceptionType]


def load_ground_truth(batch_dir: str) -> list:
    """Load ground truth from batch directories."""
    all_gt = []
    batch_path = Path(batch_dir)
    for batch_dir in sorted(batch_path.iterdir()):
        gt_file = batch_dir / "ground_truth" / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file) as f:
                gt = json.load(f)
                for record in gt:
                    record["batch_id"] = batch_dir.name
                all_gt.extend(gt)
    return all_gt


def generate_synthetic_features(gt_record: dict, rng: np.random.Generator) -> dict:
    """
    Generate synthetic features for a ground truth record.

    In production, these would come from the evidence pipeline.
    For training, we simulate realistic features correlated with the label.
    """
    exc_type = gt_record["true_exception_type"]
    payment_amount = gt_record["payment_amount"]
    difference = gt_record["difference"]
    expected = gt_record["expected_amount"]
    actual = gt_record["actual_amount"]

    # Base financial features from ground truth
    abs_diff = abs(difference)
    relative_diff = difference / payment_amount if payment_amount > 0 else 0.0

    # Simulate evidence features correlated with exception type
    if exc_type == "EXACT_MATCH":
        coverage = 1.0
        consistency = 1.0
        fully_explained = 1.0
        partially = 0.0
        conflict = 0.0
        evidence_count = 0
        num_candidates = 0
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 1
        num_refunds = 0
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "FEE_DIFFERENCE":
        coverage = min(1.0, gt_record["total_fees"] / max(abs_diff, 1))
        consistency = 0.85 + rng.uniform(-0.1, 0.1)
        fully_explained = 1.0 if coverage > 0.9 else 0.0
        partially = 0.0 if fully_explained else 1.0
        conflict = 0.0
        evidence_count = max(1, gt_record["total_fees"] // 500)
        num_candidates = 1
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 1
        num_refunds = 0
        num_fees = max(1, gt_record["total_fees"] // 1000)
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "REFUND_ADJUSTMENT":
        coverage = min(1.0, gt_record["total_refunds"] / max(abs_diff, 1))
        consistency = 0.85 + rng.uniform(-0.1, 0.1)
        fully_explained = 1.0 if coverage > 0.9 else 0.0
        partially = 0.0 if fully_explained else 1.0
        conflict = 0.0
        evidence_count = max(1, gt_record["total_refunds"] // 500)
        num_candidates = 1
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 1
        num_refunds = max(1, gt_record["total_refunds"] // 1000)
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "TAX_ADJUSTMENT":
        coverage = min(1.0, gt_record["total_taxes"] / max(abs_diff, 1))
        consistency = 0.85 + rng.uniform(-0.1, 0.1)
        fully_explained = 1.0 if coverage > 0.9 else 0.0
        partially = 0.0 if fully_explained else 1.0
        conflict = 0.0
        evidence_count = max(1, gt_record["total_taxes"] // 500)
        num_candidates = 1
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 1
        num_refunds = 0
        num_fees = 0
        num_taxes = max(1, gt_record["total_taxes"] // 1000)
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "PARTIAL_SETTLEMENT":
        coverage = actual / expected if expected > 0 else 0.0
        consistency = 0.7 + rng.uniform(-0.1, 0.1)
        fully_explained = 0.0
        partially = 1.0
        conflict = 0.0
        evidence_count = 1
        num_candidates = 0
        has_missing = 1.0
        num_missing = 1.0
        num_settlements = 1
        num_refunds = 0
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "DUPLICATE":
        coverage = 1.0
        consistency = 0.65 + rng.uniform(-0.1, 0.1)
        fully_explained = 0.0
        partially = 0.0
        conflict = 1.0
        evidence_count = 2
        num_candidates = 2
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 2
        num_refunds = 0
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual
    elif exc_type == "MISSING_RECORD":
        coverage = 0.0
        consistency = 0.25 + rng.uniform(-0.1, 0.1)
        fully_explained = 0.0
        partially = 0.0
        conflict = 0.0
        evidence_count = 0
        num_candidates = 0
        has_missing = 1.0
        num_missing = 1.0
        num_settlements = 0
        num_refunds = 0
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = 0
    elif exc_type == "COMPLEX_MULTI_ADJUSTMENT":
        coverage = 0.8 + rng.uniform(-0.1, 0.1)
        consistency = 0.75 + rng.uniform(-0.1, 0.1)
        fully_explained = 1.0 if coverage > 0.9 else 0.0
        partially = 0.0 if fully_explained else 1.0
        conflict = 0.0
        evidence_count = 3
        num_candidates = 1
        has_missing = 0.0
        num_missing = 0.0
        num_settlements = 1
        num_refunds = 1
        num_fees = 1
        num_taxes = 1
        num_adj = 1
        settlement_amt = actual
    else:  # UNKNOWN
        coverage = rng.uniform(0.0, 0.3)
        consistency = 0.3 + rng.uniform(-0.1, 0.1)
        fully_explained = 0.0
        partially = 1.0 if coverage > 0.1 else 0.0
        conflict = 0.0
        evidence_count = rng.integers(0, 2)
        num_candidates = 0
        has_missing = 1.0 if rng.random() > 0.5 else 0.0
        num_missing = float(rng.integers(0, 2))
        num_settlements = 1
        num_refunds = 0
        num_fees = 0
        num_taxes = 0
        num_adj = 0
        settlement_amt = actual

    # Clamp values
    consistency = max(0.0, min(1.0, consistency))
    coverage = max(0.0, min(1.0, coverage))

    refund_total = gt_record["total_refunds"]
    fee_total = gt_record["total_fees"]
    tax_total = gt_record["total_taxes"]
    adj_total = gt_record["total_adjustments"]

    return extract_features(
        difference=difference,
        payment_amount=payment_amount,
        settlement_amount=settlement_amt,
        refund_amount=refund_total,
        fee_amount=fee_total,
        tax_amount=tax_total,
        adjustment_amount=adj_total,
        num_settlements=num_settlements,
        num_refunds=num_refunds,
        num_fees=num_fees,
        num_taxes=num_taxes,
        num_adjustments=num_adj,
        has_missing_evidence=has_missing > 0.5,
        num_missing_evidence=int(num_missing),
        evidence_coverage=coverage,
        consistency_score=consistency,
        fully_explained=fully_explained > 0.5,
        partially_explained=partially > 0.5,
        has_conflict=conflict > 0.5,
        supporting_evidence_count=evidence_count,
        num_candidate_explanations=num_candidates,
    )


def main():
    parser = argparse.ArgumentParser(description="Train exception classifier")
    parser.add_argument("--batch-dir", default="data", help="Directory containing batches")
    parser.add_argument("--output", default="models", help="Output directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("EXCEPTION TYPE CLASSIFIER — TRAINING")
    print("=" * 70)
    print()

    # 1. Load ground truth
    print("Loading ground truth...")
    gt_records = load_ground_truth(args.batch_dir)
    print(f"  Loaded {len(gt_records)} ground truth records")

    # 2. Build MLSamples
    print("Building ML samples with synthetic features...")
    rng = np.random.default_rng(args.seed)
    feature_names = ML_FEATURE_SCHEMA.get_feature_names()
    samples = []

    for gt in gt_records:
        features = generate_synthetic_features(gt, rng)
        sample = MLSample(
            case_id=gt["case_id"],
            payment_id=gt["payment_id"],
            batch_id=gt.get("batch_id"),
            expected_amount=gt["expected_amount"],
            actual_amount=gt["actual_amount"],
            difference=gt["difference"],
            payment_amount=gt["payment_amount"],
            features=FeatureVector(features=features, schema_version=FEATURE_SCHEMA_VERSION),
            labels=MLLabels(
                true_exception_type=ExceptionType(gt["true_exception_type"]),
                true_resolution=gt.get("true_resolution"),
                resolvable=gt.get("resolvable", True),
                risk_category=gt.get("risk_category"),
            ),
        )
        samples.append(sample)
    print(f"  Built {len(samples)} samples")

    # 3. Split by batch
    print("Splitting by batch...")
    batch_001 = [s for s in samples if s.batch_id == "batch_001"]
    batch_002 = [s for s in samples if s.batch_id == "batch_002"]
    batch_003 = [s for s in samples if s.batch_id == "batch_003"]
    print(f"  Train (batch_001): {len(batch_001)}")
    print(f"  Val (batch_002): {len(batch_002)}")
    print(f"  Test (batch_003): {len(batch_003)}")

    # 4. Build numpy arrays
    builder = DatasetBuilder(feature_names)
    X_train, y_train, label_names = builder.build(batch_001)
    X_val, y_val, _ = builder.build(batch_002)
    X_test, y_test, _ = builder.build(batch_003)

    # 5. Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution (train):")
    for c, cnt in zip(unique, counts):
        name = label_names[int(c)]
        print(f"  {name}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

    # 6. Train baselines
    print("\n--- Baseline: Majority Class ---")
    majority = MajorityClassClassifier()
    majority.fit(X_train, y_train)
    maj_pred = majority.predict(X_test)
    maj_eval = ModelEvaluator.evaluate(y_test, maj_pred, label_names)
    print(f"  Accuracy: {maj_eval['accuracy']:.4f}")
    print(f"  Macro F1: {maj_eval['macro_f1']:.4f}")

    print("\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegressionClassifier(seed=args.seed)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_eval = ModelEvaluator.evaluate(y_test, lr_pred, label_names)
    print(f"  Accuracy: {lr_eval['accuracy']:.4f}")
    print(f"  Macro F1: {lr_eval['macro_f1']:.4f}")

    # 7. Train XGBoost
    print("\n--- XGBoost Classifier ---")
    start = time.time()
    xgb = ExceptionClassifier(seed=args.seed)
    train_meta = xgb.fit(X_train, y_train, X_val, y_val, feature_names)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")

    # 8. Evaluate XGBoost
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    xgb_eval = ModelEvaluator.evaluate(y_test, xgb_pred, label_names)

    print(f"\n  Accuracy: {xgb_eval['accuracy']:.4f}")
    print(f"  Macro Precision: {xgb_eval['macro_precision']:.4f}")
    print(f"  Macro Recall: {xgb_eval['macro_recall']:.4f}")
    print(f"  Macro F1: {xgb_eval['macro_f1']:.4f}")
    print(f"  Weighted F1: {xgb_eval['weighted_f1']:.4f}")

    print("\n  Per-class performance:")
    for name, metrics in xgb_eval["per_class"].items():
        if metrics["support"] > 0:
            print(f"    {name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} (n={metrics['support']})")

    print("\n  Top errors:")
    for err in xgb_eval["top_errors"][:5]:
        print(f"    {err['true']} -> {err['predicted']}: {err['count']}")

    if xgb_eval["weak_classes"]:
        print(f"\n  Weak classes (F1<0.5): {', '.join(xgb_eval['weak_classes'])}")

    # 9. Feature importance
    importance = xgb.get_feature_importance()
    if importance:
        print("\n  Top 10 features:")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, imp in sorted_imp:
            print(f"    {name}: {imp:.4f}")

    # 10. Save artifacts
    print(f"\nSaving model artifacts to {args.output}/...")
    os.makedirs(args.output, exist_ok=True)
    ModelArtifact.save(
        model=xgb.model,
        path=os.path.join(args.output, "xgboost"),
        feature_names=feature_names,
        label_names=label_names,
        training_metadata=train_meta,
        evaluation=xgb_eval,
        classifier_type="xgboost",
    )
    print(f"  Saved to {args.output}/xgboost/")

    # Save comparison
    comparison = {
        "majority_class": {"accuracy": maj_eval["accuracy"], "macro_f1": maj_eval["macro_f1"]},
        "logistic_regression": {"accuracy": lr_eval["accuracy"], "macro_f1": lr_eval["macro_f1"]},
        "xgboost": {"accuracy": xgb_eval["accuracy"], "macro_f1": xgb_eval["macro_f1"], "weighted_f1": xgb_eval["weighted_f1"]},
    }
    with open(os.path.join(args.output, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
