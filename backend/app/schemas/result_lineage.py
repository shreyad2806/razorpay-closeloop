"""
Result Lineage schemas for Razorpay CloseLoop Phase 10H.

Defines the complete traceability record for model-generated results.

Given any model prediction or resolution result, the system must answer:
  "Which exact model produced this result?"

Result → Model Version → MLflow Run → Parameters → Metrics → Artifacts → Dataset → Features

Safety principle:
  Result lineage is OBSERVATIONAL ONLY.
  It records provenance but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Result Enums
# ─────────────────────────────────────────────────────────────────────────────


class ResultType(str, Enum):
    """Types of model-generated results."""
    CLASSIFICATION = "classification"
    RESOLUTION = "resolution"
    EXCEPTION_TYPE = "exception_type"
    RISK_ASSESSMENT = "risk_assessment"
    CUSTOM = "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Result Record
# ─────────────────────────────────────────────────────────────────────────────


class ResultRecord(BaseModel):
    """A model-generated result with complete lineage.

    Every result must be traceable:
    Result → Model Version → MLflow Run → Dataset → Features
    """
    # Result identity
    result_id: str = Field(..., description="Unique result identifier")
    result_type: ResultType = Field(
        default=ResultType.CLASSIFICATION, description="Type of result"
    )

    # Workflow context
    workflow_id: Optional[str] = Field(None, description="Workflow ID")
    exception_id: Optional[str] = Field(None, description="Exception ID")

    # Model provenance (snapshot at time of prediction)
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_id: Optional[str] = Field(None, description="Model ID")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow experiment")

    # Dataset lineage (snapshot at time of prediction)
    dataset_id: Optional[str] = Field(None, description="Training dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(
        None, description="Feature schema version"
    )
    label_schema_version: Optional[str] = Field(
        None, description="Label schema version"
    )

    # Training provenance
    algorithm: Optional[str] = Field(None, description="Training algorithm")
    training_config: Dict[str, Any] = Field(
        default_factory=dict, description="Training configuration snapshot"
    )

    # Model quality snapshot (at time of prediction)
    model_accuracy: Optional[float] = Field(None, description="Model accuracy at prediction time")
    model_f1_macro: Optional[float] = Field(None, description="Model F1 macro at prediction time")
    model_precision_macro: Optional[float] = Field(None, description="Model precision at prediction time")

    # Prediction details
    prediction: Optional[str] = Field(None, description="Predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    prediction_raw: Optional[Dict[str, Any]] = Field(
        None, description="Raw prediction output"
    )

    # Policy context
    policy_version: Optional[str] = Field(None, description="Active policy version")

    # Timestamps
    predicted_at: datetime = Field(
        default_factory=datetime.utcnow, description="When prediction was made"
    )

    def summary(self) -> str:
        return (
            f"Result: {self.result_id[:8]}... | "
            f"Model: {self.model_name} v{self.model_version} | "
            f"Run: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'} | "
            f"Exception: {self.exception_id or 'none'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Complete Lineage Chain
# ─────────────────────────────────────────────────────────────────────────────


class ModelLineageChain(BaseModel):
    """Complete lineage chain from result to training provenance."""
    # Result
    result_id: str = Field(..., description="Result ID")
    result_type: str = Field(..., description="Result type")
    exception_id: Optional[str] = Field(None, description="Exception ID")
    workflow_id: Optional[str] = Field(None, description="Workflow ID")

    # Model
    model_id: Optional[str] = Field(None, description="Model ID")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_status: Optional[str] = Field(None, description="Model status at query time")

    # MLflow
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow experiment")

    # Dataset
    dataset_id: Optional[str] = Field(None, description="Dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature version")
    label_schema_version: Optional[str] = Field(None, description="Label version")

    # Training
    algorithm: Optional[str] = Field(None, description="Algorithm")
    training_config: Dict[str, Any] = Field(
        default_factory=dict, description="Training config"
    )
    training_examples: Optional[int] = Field(None, description="Training examples")
    feature_count: Optional[int] = Field(None, description="Feature count")
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    label_classes: List[str] = Field(default_factory=list, description="Label classes")

    # Quality
    model_accuracy: Optional[float] = Field(None, description="Model accuracy")
    model_f1_macro: Optional[float] = Field(None, description="Model F1 macro")
    model_precision_macro: Optional[float] = Field(None, description="Model precision")

    # Prediction
    prediction: Optional[str] = Field(None, description="Prediction")
    confidence: Optional[float] = Field(None, description="Confidence")
    predicted_at: Optional[datetime] = Field(None, description="Prediction timestamp")

    # Policy
    policy_version: Optional[str] = Field(None, description="Policy version")

    # MLflow run metadata (if available)
    mlflow_run_status: Optional[str] = Field(None, description="MLflow run status")
    mlflow_run_params: Dict[str, str] = Field(
        default_factory=dict, description="MLflow run parameters"
    )

    def summary(self) -> str:
        return (
            f"Chain: {self.result_id[:8]}... → "
            f"{self.model_name} v{self.model_version} → "
            f"Run {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'} → "
            f"Dataset v{self.dataset_version or '?'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Audit Response
# ─────────────────────────────────────────────────────────────────────────────


class AuditResponse(BaseModel):
    """Complete audit response for a given exception or result."""
    exception_id: Optional[str] = Field(None, description="Exception ID")
    result_id: Optional[str] = Field(None, description="Result ID")

    # Model identification
    model_name: Optional[str] = Field(None, description="Which model classified it")
    model_version: Optional[str] = Field(None, description="Which version")
    model_id: Optional[str] = Field(None, description="Model ID")

    # MLflow traceability
    mlflow_run_id: Optional[str] = Field(None, description="Which MLflow run")
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow experiment")

    # Training provenance
    dataset_version: Optional[str] = Field(None, description="Which dataset trained it")
    feature_schema_version: Optional[str] = Field(None, description="Which features")
    algorithm: Optional[str] = Field(None, description="Which algorithm")
    training_examples: Optional[int] = Field(None, description="Training set size")

    # Prediction details
    prediction: Optional[str] = Field(None, description="Prediction made")
    confidence: Optional[float] = Field(None, description="Model confidence")

    # Model quality at prediction time
    model_accuracy: Optional[float] = Field(None, description="Model accuracy")
    model_f1_macro: Optional[float] = Field(None, description="Model F1")

    # Policy context
    policy_version: Optional[str] = Field(None, description="Active policy version")

    # Timestamps
    predicted_at: Optional[datetime] = Field(None, description="When prediction was made")
    trained_at: Optional[datetime] = Field(None, description="When model was trained")

    # Full lineage chain
    lineage_chain: Optional[ModelLineageChain] = Field(
        None, description="Complete lineage chain"
    )

    def summary(self) -> str:
        return (
            f"Audit: {self.exception_id or self.result_id or 'unknown'} | "
            f"Model: {self.model_name} v{self.model_version} | "
            f"Run: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'}"
        )
