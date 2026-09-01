"""
Model Training schemas for Razorpay CloseLoop Phase 9E.

Defines candidate model metadata, training configuration,
evaluation metrics, and baseline comparison.

Safety principle:
  A trained candidate model is a CANDIDATE.
  It is NOT automatically promoted to production.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Model Enums
# ─────────────────────────────────────────────────────────────────────────────


class ModelStatus(str, Enum):
    """Status of a model version."""
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class ModelType(str, Enum):
    """Types of models supported."""
    EXCEPTION_CLASSIFIER = "exception_classifier"
    RESOLUTION_PREDICTOR = "resolution_predictor"


# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────────────


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    model_type: ModelType = Field(
        default=ModelType.EXCEPTION_CLASSIFIER,
        description="Type of model to train",
    )
    algorithm: str = Field(
        default="xgboost",
        description="Training algorithm: xgboost, decision_tree, logistic_regression",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    random_seed: int = Field(
        default=42, description="Random seed for reproducibility"
    )
    max_features: Optional[int] = Field(
        default=None, description="Max features to use (None = all)"
    )
    class_weight: Optional[str] = Field(
        default=None, description="Class weighting: None, balanced"
    )
    early_stopping_rounds: Optional[int] = Field(
        default=10, description="Early stopping rounds (XGBoost)"
    )
    train_ratio: float = Field(
        default=0.7, description="Training set ratio"
    )
    val_ratio: float = Field(
        default=0.15, description="Validation set ratio"
    )
    test_ratio: float = Field(
        default=0.15, description="Test set ratio"
    )

    def summary(self) -> str:
        return (
            f"Algorithm: {self.algorithm} | "
            f"Type: {self.model_type.value} | "
            f"Seed: {self.random_seed}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Metadata
# ─────────────────────────────────────────────────────────────────────────────


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    version: str = Field(..., description="Model version string")
    model_type: ModelType = Field(
        default=ModelType.EXCEPTION_CLASSIFIER, description="Model type"
    )
    status: ModelStatus = Field(
        default=ModelStatus.CANDIDATE, description="Model status"
    )

    # Dataset provenance
    dataset_id: Optional[str] = Field(
        default=None, description="Training dataset ID"
    )
    dataset_version: str = Field(
        default="1.0.0", description="Training dataset version"
    )
    feature_schema_version: str = Field(
        default="1.0.0", description="Feature schema version"
    )
    training_examples: int = Field(
        default=0, description="Number of training examples"
    )
    feature_count: int = Field(
        default=0, description="Number of features used"
    )
    feature_names: List[str] = Field(
        default_factory=list, description="Ordered feature names"
    )
    label_classes: List[str] = Field(
        default_factory=list, description="Label classes"
    )

    # Training config
    config: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    trained_at: Optional[datetime] = Field(
        default=None, description="When model was trained"
    )
    training_duration_seconds: Optional[float] = Field(
        default=None, description="Training time in seconds"
    )

    # Promotion
    promoted_at: Optional[datetime] = Field(
        default=None, description="When promoted to active"
    )
    retired_at: Optional[datetime] = Field(
        default=None, description="When retired"
    )

    def summary(self) -> str:
        return (
            f"Model: {self.model_name} v{self.version} | "
            f"Status: {self.status.value} | "
            f"Features: {self.feature_count} | "
            f"Examples: {self.training_examples}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics for a model."""
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    evaluated_on: str = Field(
        default="test", description="Dataset split used for evaluation"
    )
    total_samples: int = Field(default=0, description="Total samples evaluated")

    # Standard classification metrics
    accuracy: float = Field(default=0.0, description="Overall accuracy")
    precision_macro: float = Field(default=0.0, description="Macro-averaged precision")
    recall_macro: float = Field(default=0.0, description="Macro-averaged recall")
    f1_macro: float = Field(default=0.0, description="Macro-averaged F1")
    precision_weighted: float = Field(default=0.0, description="Weighted precision")
    recall_weighted: float = Field(default=0.0, description="Weighted recall")
    f1_weighted: float = Field(default=0.0, description="Weighted F1")

    # Per-class metrics
    per_class_precision: Dict[str, float] = Field(
        default_factory=dict, description="Per-class precision"
    )
    per_class_recall: Dict[str, float] = Field(
        default_factory=dict, description="Per-class recall"
    )
    per_class_f1: Dict[str, float] = Field(
        default_factory=dict, description="Per-class F1"
    )
    per_class_support: Dict[str, int] = Field(
        default_factory=dict, description="Per-class sample count"
    )

    # Confusion matrix
    confusion_matrix: List[List[int]] = Field(
        default_factory=list, description="Confusion matrix (rows=actual, cols=predicted)"
    )
    confusion_labels: List[str] = Field(
        default_factory=list, description="Labels for confusion matrix"
    )

    # Safety-critical metrics
    incorrect_auto_resolution: int = Field(
        default=0, description="Incorrect AUTO decisions"
    )
    high_value_errors: int = Field(
        default=0, description="Errors on high-value cases"
    )
    unknown_case_errors: int = Field(
        default=0, description="Errors on UNKNOWN cases"
    )
    novel_pattern_errors: int = Field(
        default=0, description="Errors on novel patterns"
    )
    verification_failure_rate: Optional[float] = Field(
        default=None, description="Rate of verification failures"
    )
    false_automation: int = Field(
        default=0, description="Incorrect auto-resolutions"
    )

    # Resolution-specific
    resolution_accuracy: Optional[float] = Field(
        default=None, description="Resolution prediction accuracy"
    )

    # Metadata
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When evaluation was performed"
    )
    evaluation_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Evaluation configuration"
    )

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.1%} | "
            f"F1(macro): {self.f1_macro:.1%} | "
            f"False auto: {self.false_automation} | "
            f"HV errors: {self.high_value_errors}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Model Comparison
# ─────────────────────────────────────────────────────────────────────────────


class SafetyMetricCheck(BaseModel):
    """A single safety metric check result."""
    metric_name: str = Field(..., description="Metric name")
    current_value: float = Field(..., description="Current model value")
    candidate_value: float = Field(..., description="Candidate model value")
    passed: bool = Field(..., description="Whether check passed")
    threshold: Optional[float] = Field(
        default=None, description="Threshold if applicable"
    )
    description: str = Field(default="", description="Check description")


class CandidateModelComparison(BaseModel):
    """Comparison between current and candidate model."""
    current_model_id: str = Field(...)
    current_version: str = Field(...)
    candidate_model_id: str = Field(...)
    candidate_version: str = Field(...)

    current_metrics: EvaluationMetrics = Field(...)
    candidate_metrics: EvaluationMetrics = Field(...)

    # Safety checks
    safety_checks: List[SafetyMetricCheck] = Field(
        default_factory=list, description="Safety metric checks"
    )
    all_safety_passed: bool = Field(
        default=False, description="Whether all safety checks passed"
    )

    # Improvements and regressions
    improvements: List[str] = Field(
        default_factory=list, description="Metrics that improved"
    )
    regressions: List[str] = Field(
        default_factory=list, description="Metrics that regressed"
    )

    # Recommendation
    recommendation: str = Field(
        default="DEFER",
        description="PROMOTE, REJECT, or DEFER"
    )
    recommendation_reason: str = Field(
        default="", description="Why this recommendation"
    )

    compared_at: datetime = Field(
        default_factory=datetime.utcnow, description="When comparison was made"
    )

    def summary(self) -> str:
        return (
            f"Current: v{self.current_version} | "
            f"Candidate: v{self.candidate_version} | "
            f"Recommendation: {self.recommendation} | "
            f"Safety: {'PASS' if self.all_safety_passed else 'FAIL'} | "
            f"Improvements: {len(self.improvements)} | "
            f"Regressions: {len(self.regressions)}"
        )
