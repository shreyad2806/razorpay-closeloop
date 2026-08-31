"""
Resolution predictor for Razorpay CloseLoop Phase 4.

Given an exception and its evidence, predicts the most likely resolution.

IMPORTANT: A predicted resolution is ONLY a recommendation.
It must NOT automatically modify financial records.

Uses the centralized ResolutionType taxonomy from enums.py.
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from app.schemas.enums import ExceptionType, ResolutionType
from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult
from app.schemas.evidence_quality import EvidenceQualityResult
from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION
from app.ml.features import ML_FEATURE_SCHEMA


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RESOLUTION_VERSION = "1.0.0"
ALL_RESOLUTIONS = [r.value for r in ResolutionType]
RANDOM_SEED = 42

# Deterministic mapping from exception type to expected resolution.
# Used for evidence compatibility checking, NOT as a shortcut for prediction.
EXCEPTION_TO_RESOLUTION_MAP = {
    "EXACT_MATCH": "NO_ACTION",
    "FEE_DIFFERENCE": "FEE_ADJUSTMENT",
    "REFUND_ADJUSTMENT": "REFUND_ADJUSTMENT",
    "TAX_ADJUSTMENT": "TAX_ADJUSTMENT",
    "TIMING_DIFFERENCE": "TIMING_RECONCILIATION",
    "PARTIAL_SETTLEMENT": "PARTIAL_SETTLEMENT_RECONCILIATION",
    "DUPLICATE": "DUPLICATE_SETTLEMENT",
    "MISSING_RECORD": "MISSING_RECORD_ESCALATION",
    "COMPLEX_MULTI_ADJUSTMENT": "MULTI_ADJUSTMENT",
    "UNKNOWN": "UNKNOWN_UNRESOLVED",
}

# Evidence compatibility rules: which resolution types require specific evidence
EVIDENCE_COMPATIBILITY_RULES = {
    "FEE_ADJUSTMENT": {"requires": ["fees"], "description": "Requires fee records"},
    "REFUND_ADJUSTMENT": {"requires": ["refunds"], "description": "Requires refund records"},
    "TAX_ADJUSTMENT": {"requires": ["taxes"], "description": "Requires tax records"},
    "PARTIAL_SETTLEMENT_RECONCILIATION": {"requires": ["settlements"], "description": "Requires settlement records"},
    "DUPLICATE_SETTLEMENT": {"requires": ["settlements"], "description": "Requires multiple settlements"},
    "MISSING_RECORD_ESCALATION": {"requires": ["missing_evidence"], "description": "Requires missing evidence"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Prediction Result
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionPrediction:
    """
    Resolution prediction result.

    Contains the ML recommendation plus evidence compatibility analysis.
    This is a recommendation ONLY — it does not modify financial records.
    """

    def __init__(
        self,
        predicted_resolution: str,
        predicted_exception_type: str,
        probabilities: Dict[str, float],
        evidence_compatible: bool,
        compatibility_notes: List[str],
        supporting_evidence_ids: List[str],
        model_version: str,
    ):
        self.predicted_resolution = predicted_resolution
        self.predicted_exception_type = predicted_exception_type
        self.probabilities = probabilities
        self.evidence_compatible = evidence_compatible
        self.compatibility_notes = compatibility_notes
        self.supporting_evidence_ids = supporting_evidence_ids
        self.model_version = model_version

    def to_dict(self) -> Dict:
        return {
            "predicted_resolution": self.predicted_resolution,
            "predicted_exception_type": self.predicted_exception_type,
            "probabilities": self.probabilities,
            "evidence_compatible": self.evidence_compatible,
            "compatibility_notes": self.compatibility_notes,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "model_version": self.model_version,
            "is_recommendation_only": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Compatibility Checker
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceCompatibilityChecker:
    """
    Checks whether a predicted resolution is compatible with deterministic evidence.

    This is a safety layer: if the ML recommends a resolution that contradicts
    the available evidence, the conflict is flagged.
    """

    @staticmethod
    def check(
        predicted_resolution: str,
        package: EvidencePackage,
        explanation: ExplanationResult,
    ) -> Tuple[bool, List[str]]:
        """
        Check evidence compatibility for a predicted resolution.

        Returns:
            (is_compatible, list_of_notes)
        """
        notes = []
        compatible = True

        # Check resolution-specific evidence requirements
        rule = EVIDENCE_COMPATIBILITY_RULES.get(predicted_resolution)
        if rule:
            for requirement in rule["requires"]:
                if requirement == "fees" and len(package.fees) == 0:
                    notes.append(
                        f"ML recommends {predicted_resolution} but no fee evidence found"
                    )
                    compatible = False
                elif requirement == "refunds" and len(package.refunds) == 0:
                    notes.append(
                        f"ML recommends {predicted_resolution} but no refund evidence found"
                    )
                    compatible = False
                elif requirement == "taxes" and len(package.taxes) == 0:
                    notes.append(
                        f"ML recommends {predicted_resolution} but no tax evidence found"
                    )
                    compatible = False
                elif requirement == "settlements" and len(package.settlements) == 0:
                    notes.append(
                        f"ML recommends {predicted_resolution} but no settlement evidence found"
                    )
                    compatible = False
                elif requirement == "missing_evidence" and len(package.missing_evidence) == 0:
                    notes.append(
                        f"ML recommends {predicted_resolution} but no missing evidence detected"
                    )
                    compatible = False

        # Check for explanation conflicts
        if explanation.conflict and predicted_resolution not in (
            "MANUAL_REVIEW", "UNKNOWN_UNRESOLVED"
        ):
            notes.append(
                f"ML recommends {predicted_resolution} but explanation has conflicts"
            )
            compatible = False

        # Check for unexplained cases
        if (
            explanation.explanation_status.value == "UNEXPLAINED"
            and predicted_resolution == "NO_ACTION"
        ):
            notes.append(
                "ML recommends NO_ACTION but discrepancy is unexplained"
            )
            compatible = False

        # Check DUPLICATE_SETTLEMENT requires multiple settlements
        if predicted_resolution == "DUPLICATE_SETTLEMENT":
            if len(package.settlements) < 2:
                notes.append(
                    "DUPLICATE_SETTLEMENT requires >=2 settlements, "
                    f"found {len(package.settlements)}"
                )
                compatible = False

        if not notes:
            notes.append("Evidence is compatible with predicted resolution")

        return compatible, notes


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Classifier
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionClassifier:
    """
    XGBoost multiclass resolution type classifier.

    Predicts the most likely resolution given exception features.
    This is a recommendation ONLY — it does not modify financial records.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        import xgboost as xgb

        self.seed = seed
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            tree_method="hist",
            verbosity=0,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ALL_RESOLUTIONS)
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Train the resolution classifier."""
        self.feature_names = feature_names or []

        # Class weights for imbalance
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weights = {
            int(c): total / (len(unique) * int(cnt))
            for c, cnt in zip(unique, counts)
        }
        sample_weights = np.array([class_weights[int(yi)] for yi in y])

        self.model.fit(X, y, sample_weight=sample_weights, verbose=False)
        self.is_fitted = True

        class_dist = {
            self.label_encoder.inverse_transform([int(c)])[0]: int(cnt)
            for c, cnt in zip(unique, counts)
        }
        return {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": len(unique),
            "class_distribution": class_dist,
            "seed": self.seed,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_labels(self, X: np.ndarray) -> List[str]:
        preds = self.predict(X)
        return self.label_encoder.inverse_transform(preds).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Prediction Service
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionPredictor:
    """
    Resolution prediction service.

    Combines ML prediction with evidence compatibility checking.
    This is a recommendation ONLY — it does not modify financial records.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        label_names: List[str],
        model_version: str = RESOLUTION_VERSION,
    ):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        self.model_version = model_version
        self.compat_checker = EvidenceCompatibilityChecker()

    def predict(
        self,
        feature_dict: Dict[str, float],
        package: EvidencePackage,
        explanation: ExplanationResult,
    ) -> ResolutionPrediction:
        """
        Predict resolution with evidence compatibility check.

        Args:
            feature_dict: Feature name → value mapping
            package: EvidencePackage from evidence retrieval
            explanation: ExplanationResult from explanation engine

        Returns:
            ResolutionPrediction with ML recommendation + compatibility
        """
        # ML prediction
        X = np.array(
            [[feature_dict.get(name, 0.0) for name in self.feature_names]],
            dtype=np.float32,
        )
        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        predicted_resolution = self.label_names[int(pred_idx)]
        probabilities = {
            name: round(float(p), 6)
            for name, p in zip(self.label_names, proba)
        }

        # Evidence compatibility check
        compatible, notes = self.compat_checker.check(
            predicted_resolution, package, explanation
        )

        # Get exception type from package
        exception_type = package.exception_type

        return ResolutionPrediction(
            predicted_resolution=predicted_resolution,
            predicted_exception_type=exception_type,
            probabilities=probabilities,
            evidence_compatible=compatible,
            compatibility_notes=notes,
            supporting_evidence_ids=explanation.supporting_evidence_ids,
            model_version=self.model_version,
        )

    def predict_batch(
        self,
        feature_dicts: List[Dict[str, float]],
        packages: List[EvidencePackage],
        explanations: List[ExplanationResult],
    ) -> List[ResolutionPrediction]:
        """Predict for multiple samples."""
        return [
            self.predict(fd, pkg, exp)
            for fd, pkg, exp in zip(feature_dicts, packages, explanations)
        ]

    @classmethod
    def from_artifact(cls, path: str) -> "ResolutionPredictor":
        """Load from saved artifact."""
        model_path = os.path.join(path, "model.joblib")
        metadata_path = os.path.join(path, "metadata.json")

        model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return cls(
            model=model,
            feature_names=metadata["feature_names"],
            label_names=metadata["label_names"],
            model_version=metadata.get("resolution_version", RESOLUTION_VERSION),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training Data Builder
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionTrainingBuilder:
    """
    Builds training data for resolution prediction from ground truth.
    """

    @staticmethod
    def build_samples(
        ground_truth_records: List[Dict],
        feature_names: List[str],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build training arrays from ground truth records.

        Returns (X, y, label_names).
        """
        label_encoder = LabelEncoder()
        label_encoder.fit(ALL_RESOLUTIONS)

        X_rows = []
        y_rows = []

        for gt in ground_truth_records:
            # Generate features correlated with the resolution
            features = ResolutionTrainingBuilder._generate_features(gt, rng)
            X_rows.append([features.get(name, 0.0) for name in feature_names])
            y_rows.append(gt["true_resolution"])

        X = np.array(X_rows, dtype=np.float32)
        y = label_encoder.transform(y_rows)

        return X, y, list(label_encoder.classes_)

    @staticmethod
    def _generate_features(gt: dict, rng: np.random.Generator) -> dict:
        """Generate features for a resolution training sample."""
        from app.ml.features import extract_features

        exc_type = gt["true_exception_type"]
        resolution = gt["true_resolution"]

        # Base features from ground truth
        payment_amount = gt["payment_amount"]
        difference = gt["difference"]
        expected = gt["expected_amount"]
        actual = gt["actual_amount"]

        # Determine structural features based on exception type
        if exc_type == "EXACT_MATCH":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 0, 0, 0, 0
            coverage, consistency = 1.0, 1.0
            fully, partially, conflict = True, False, False
            evidence_count, num_candidates = 0, 0
            has_missing, num_missing = False, 0
            settlement_amt = actual
        elif exc_type == "FEE_DIFFERENCE":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 0, 1, 0, 0
            coverage = min(1.0, gt["total_fees"] / max(abs(difference), 1))
            consistency = 0.85
            fully, partially, conflict = coverage > 0.9, coverage <= 0.9, False
            evidence_count, num_candidates = 1, 1
            has_missing, num_missing = False, 0
            settlement_amt = actual
        elif exc_type == "REFUND_ADJUSTMENT":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 1, 0, 0, 0
            coverage = min(1.0, gt["total_refunds"] / max(abs(difference), 1))
            consistency = 0.85
            fully, partially, conflict = coverage > 0.9, coverage <= 0.9, False
            evidence_count, num_candidates = 1, 1
            has_missing, num_missing = False, 0
            settlement_amt = actual
        elif exc_type == "TAX_ADJUSTMENT":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 0, 0, 1, 0
            coverage = min(1.0, gt["total_taxes"] / max(abs(difference), 1))
            consistency = 0.85
            fully, partially, conflict = coverage > 0.9, coverage <= 0.9, False
            evidence_count, num_candidates = 1, 1
            has_missing, num_missing = False, 0
            settlement_amt = actual
        elif exc_type == "PARTIAL_SETTLEMENT":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 0, 0, 0, 0
            coverage = actual / expected if expected > 0 else 0.0
            consistency = 0.7
            fully, partially, conflict = False, True, False
            evidence_count, num_candidates = 1, 0
            has_missing, num_missing = True, 1
            settlement_amt = actual
        elif exc_type == "DUPLICATE":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 2, 0, 0, 0, 0
            coverage, consistency = 1.0, 0.65
            fully, partially, conflict = False, False, True
            evidence_count, num_candidates = 2, 2
            has_missing, num_missing = False, 0
            settlement_amt = actual
        elif exc_type == "MISSING_RECORD":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 0, 0, 0, 0, 0
            coverage, consistency = 0.0, 0.25
            fully, partially, conflict = False, False, False
            evidence_count, num_candidates = 0, 0
            has_missing, num_missing = True, 1
            settlement_amt = 0
        elif exc_type == "COMPLEX_MULTI_ADJUSTMENT":
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 1, 1, 1, 1
            coverage, consistency = 0.8, 0.75
            fully = coverage > 0.9
            partially, conflict = not fully, False
            evidence_count, num_candidates = 3, 1
            has_missing, num_missing = False, 0
            settlement_amt = actual
        else:  # UNKNOWN
            num_settlements, num_refunds, num_fees, num_taxes, num_adj = 1, 0, 0, 0, 0
            coverage = rng.uniform(0.0, 0.3)
            consistency = 0.3
            fully, partially, conflict = False, coverage > 0.1, False
            evidence_count, num_candidates = 0, 0
            has_missing = rng.random() > 0.5
            num_missing = int(has_missing)
            settlement_amt = actual

        return extract_features(
            difference=difference,
            payment_amount=payment_amount,
            settlement_amount=settlement_amt,
            refund_amount=gt["total_refunds"],
            fee_amount=gt["total_fees"],
            tax_amount=gt["total_taxes"],
            adjustment_amount=gt["total_adjustments"],
            num_settlements=num_settlements,
            num_refunds=num_refunds,
            num_fees=num_fees,
            num_taxes=num_taxes,
            num_adjustments=num_adj,
            has_missing_evidence=has_missing,
            num_missing_evidence=num_missing,
            evidence_coverage=max(0.0, min(1.0, coverage)),
            consistency_score=max(0.0, min(1.0, consistency)),
            fully_explained=fully,
            partially_explained=partially,
            has_conflict=conflict,
            supporting_evidence_count=evidence_count,
            num_candidate_explanations=num_candidates,
        )
