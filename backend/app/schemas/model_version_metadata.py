"""
Model Version Metadata schemas for Razorpay CloseLoop Phase 10E.

Connects MLflow runs with the existing Phase 9 model-versioning system.

Goal: Know exactly which model produced every prediction.

Prediction lineage:
  Prediction → Model version → MLflow run → Training config → Dataset → Features

Safety principle:
  Model version metadata is OBSERVATIONAL ONLY.
  It records provenance but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Model Version Enums
# ─────────────────────────────────────────────────────────────────────────────


class ModelVersionStatus(str, Enum):
    """Status of a model version in the MLflow-linked registry."""
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


# ─────────────────────────────────────────────────────────────────────────────
# Model Version Metadata
# ─────────────────────────────────────────────────────────────────────────────


class ModelVersionMetadata(BaseModel):
    """Complete metadata for a model version, linking Phase 9 models to MLflow.

    Every trained model must have enough information to:
    - Locate the exact MLflow run
    - Retrieve the exact training configuration
    - Identify the dataset and feature versions
    - Trace every prediction back to this model
    """
    # Identity
    model_id: str = Field(..., description="Unique model identifier (Phase 9)")
    model_name: str = Field(..., description="Human-readable model name")
    model_version: str = Field(..., description="Model version string")
    status: ModelVersionStatus = Field(
        default=ModelVersionStatus.CANDIDATE,
        description="Current status of this model version",
    )

    # MLflow linkage
    mlflow_run_id: Optional[str] = Field(
        None, description="MLflow run ID this model was trained in"
    )
    mlflow_experiment_name: Optional[str] = Field(
        None, description="MLflow experiment name"
    )

    # Dataset lineage
    dataset_id: Optional[str] = Field(None, description="Training dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(
        None, description="Feature schema version"
    )
    label_schema_version: Optional[str] = Field(
        None, description="Label schema version"
    )

    # Training configuration
    algorithm: Optional[str] = Field(None, description="Training algorithm")
    training_config: Dict[str, Any] = Field(
        default_factory=dict, description="Full training configuration"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    random_seed: Optional[int] = Field(None, description="Random seed")

    # Feature details
    feature_count: int = Field(default=0, description="Number of features")
    feature_names: List[str] = Field(
        default_factory=list, description="Ordered feature names"
    )
    label_classes: List[str] = Field(
        default_factory=list, description="Label classes"
    )

    # Training info
    training_examples: int = Field(default=0, description="Training examples")
    training_duration_seconds: Optional[float] = Field(
        None, description="Training duration"
    )

    # Evaluation metrics snapshot (key metrics for quick reference)
    accuracy: Optional[float] = Field(None, description="Test accuracy")
    f1_macro: Optional[float] = Field(None, description="Test F1 macro")
    precision_macro: Optional[float] = Field(None, description="Test precision macro")
    false_automation: Optional[int] = Field(
        None, description="False automation count"
    )
    high_value_errors: Optional[int] = Field(
        None, description="High-value error count"
    )

    # Timestamps
    trained_at: Optional[datetime] = Field(None, description="Training timestamp")
    promoted_at: Optional[datetime] = Field(
        None, description="Promotion timestamp"
    )
    retired_at: Optional[datetime] = Field(None, description="Retirement timestamp")

    # Policy linkage
    policy_version: Optional[str] = Field(
        None, description="Policy version used during this model's lifetime"
    )

    def summary(self) -> str:
        return (
            f"Model: {self.model_name} v{self.model_version} | "
            f"Status: {self.status.value} | "
            f"Run: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'} | "
            f"Dataset: v{self.dataset_version or '?'}"
        )

    def get_lineage_dict(self) -> Dict[str, Any]:
        """Return the full lineage as a serializable dict."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "feature_schema_version": self.feature_schema_version,
            "label_schema_version": self.label_schema_version,
            "algorithm": self.algorithm,
            "feature_count": self.feature_count,
            "label_classes": self.label_classes,
            "training_examples": self.training_examples,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "false_automation": self.false_automation,
            "high_value_errors": self.high_value_errors,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "policy_version": self.policy_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Lineage
# ─────────────────────────────────────────────────────────────────────────────


class PredictionLineage(BaseModel):
    """Traceability record for a single prediction.

    Every prediction must be traceable:
    Prediction → Model version → MLflow run → Training config → Dataset → Features
    """
    prediction_id: str = Field(..., description="Unique prediction identifier")
    model_id: str = Field(..., description="Model that produced the prediction")
    model_version: str = Field(..., description="Model version")
    mlflow_run_id: Optional[str] = Field(
        None, description="MLflow run ID"
    )
    dataset_version: Optional[str] = Field(
        None, description="Dataset version used for training"
    )
    feature_schema_version: Optional[str] = Field(
        None, description="Feature schema version"
    )
    algorithm: Optional[str] = Field(None, description="Algorithm used")
    predicted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the prediction was made",
    )

    # Prediction details (what was predicted)
    prediction: Optional[str] = Field(None, description="Predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence")

    def summary(self) -> str:
        return (
            f"Prediction: {self.prediction_id[:8]}... | "
            f"Model: v{self.model_version} | "
            f"Run: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Version Summary
# ─────────────────────────────────────────────────────────────────────────────


class ModelVersionSummary(BaseModel):
    """Summary of all model versions for observability."""
    total_versions: int = Field(default=0, description="Total model versions")
    active_count: int = Field(default=0, description="Currently active models")
    candidate_count: int = Field(default=0, description="Candidate models")
    retired_count: int = Field(default=0, description="Retired models")
    rejected_count: int = Field(default=0, description="Rejected models")
    active_model: Optional[ModelVersionMetadata] = Field(
        None, description="Currently active model metadata"
    )
    candidate_model: Optional[ModelVersionMetadata] = Field(
        None, description="Current candidate model metadata"
    )

    def summary(self) -> str:
        return (
            f"Versions: {self.total_versions} | "
            f"Active: {self.active_count} | "
            f"Candidate: {self.candidate_count} | "
            f"Retired: {self.retired_count}"
        )
