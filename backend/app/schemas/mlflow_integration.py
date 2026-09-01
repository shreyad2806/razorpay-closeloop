"""
MLflow Integration schemas for Razorpay CloseLoop Phase 10J.

Defines the unified integration that ties together all Phase 10 components:
  MLflow Tracking → Parameter Logging → Metrics → Artifacts →
  Model Registry → Result Lineage → Experiment Comparison

Complete flow:
  Learning Dataset → Training → MLflow Experiment → MLflow Run →
  Candidate Model → Evaluation → Safety Evaluation → Model Registry →
  Prediction → Result Lineage → Outcome → Feedback → Learning

Safety principle:
  MLflow integration is OBSERVATIONAL + LIFECYCLE only.
  It never authorizes execution or bypasses Phase 6 guardrails.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Integration Status
# ─────────────────────────────────────────────────────────────────────────────


class IntegrationPhase(str, Enum):
    """Phases of the integrated MLflow lifecycle."""
    TRAINING = "training"
    TRACKING = "tracking"
    EVALUATION = "evaluation"
    REGISTRY = "registry"
    PREDICTION = "prediction"
    OUTCOME = "outcome"
    LEARNING = "learning"


class IntegrationStatus(str, Enum):
    """Status of an integration step."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle Record
# ─────────────────────────────────────────────────────────────────────────────


class MLflowLifecycleStep(BaseModel):
    """A single step in the MLflow lifecycle."""
    phase: IntegrationPhase = Field(..., description="Integration phase")
    status: IntegrationStatus = Field(default=IntegrationStatus.PENDING, description="Status")
    timestamp: Optional[datetime] = Field(None, description="When this step completed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Phase-specific details")
    error: Optional[str] = Field(None, description="Error if failed")


class MLflowLifecycleRecord(BaseModel):
    """Complete lifecycle record for a model through the integrated pipeline."""
    record_id: str = Field(..., description="Unique lifecycle record ID")
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")

    # MLflow linkage
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow experiment")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")

    # Dataset linkage
    dataset_id: Optional[str] = Field(None, description="Training dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")

    # Registry linkage
    registry_state: Optional[str] = Field(None, description="Registry lifecycle state")
    previous_model_id: Optional[str] = Field(None, description="Previous model (for rollback)")

    # Lifecycle steps
    steps: List[MLflowLifecycleStep] = Field(default_factory=list, description="Lifecycle steps")

    # Summary
    current_phase: IntegrationPhase = Field(
        default=IntegrationPhase.TRAINING, description="Current phase"
    )
    all_completed: bool = Field(default=False, description="All phases completed")
    any_failed: bool = Field(default=False, description="Any phase failed")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Record creation time")
    completed_at: Optional[datetime] = Field(None, description="Record completion time")

    def get_step(self, phase: IntegrationPhase) -> Optional[MLflowLifecycleStep]:
        """Get a specific phase step."""
        for s in self.steps:
            if s.phase == phase:
                return s
        return None

    def summary(self) -> str:
        return (
            f"Lifecycle: {self.model_version} | "
            f"MLflow: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'} | "
            f"Registry: {self.registry_state or 'none'} | "
            f"Status: {'COMPLETE' if self.all_completed else 'IN PROGRESS'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training → MLflow Request
# ─────────────────────────────────────────────────────────────────────────────


class TrainAndTrackRequest(BaseModel):
    """Request to train a model and track via MLflow."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    experiment_name: str = Field(..., description="MLflow experiment name")
    algorithm: str = Field(default="xgboost", description="Algorithm")
    dataset_id: Optional[str] = Field(None, description="Dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training config")


class TrainAndTrackResult(BaseModel):
    """Result of a tracked training run."""
    lifecycle_record_id: str = Field(..., description="Lifecycle record ID")
    model_id: str = Field(..., description="Trained model ID")
    model_version: str = Field(..., description="Model version")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    mlflow_experiment: str = Field(..., description="MLflow experiment")
    registry_model_id: Optional[str] = Field(None, description="Registry model ID")
    parameters_logged: bool = Field(default=False, description="Parameters logged")
    metrics_logged: bool = Field(default=False, description="Metrics logged")
    artifacts_logged: bool = Field(default=False, description="Artifacts logged")


# ─────────────────────────────────────────────────────────────────────────────
# Predict and Track Request
# ─────────────────────────────────────────────────────────────────────────────


class PredictAndTrackRequest(BaseModel):
    """Request to make a prediction and record lineage."""
    exception_id: str = Field(..., description="Exception ID")
    model_id: str = Field(..., description="Model ID making the prediction")
    workflow_id: Optional[str] = Field(None, description="Workflow ID")
    prediction: Optional[str] = Field(None, description="Predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    policy_version: Optional[str] = Field(None, description="Active policy version")


class PredictAndTrackResult(BaseModel):
    """Result of a tracked prediction."""
    result_id: str = Field(..., description="Result lineage ID")
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    prediction: Optional[str] = Field(None, description="Prediction")
    confidence: Optional[float] = Field(None, description="Confidence")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Summary
# ─────────────────────────────────────────────────────────────────────────────


class MLflowIntegrationSummary(BaseModel):
    """Summary of the complete MLflow integration state."""
    total_lifecycle_records: int = Field(default=0, description="Total lifecycle records")
    completed_records: int = Field(default=0, description="Completed records")
    failed_records: int = Field(default=0, description="Failed records")
    in_progress_records: int = Field(default=0, description="In-progress records")

    total_models_tracked: int = Field(default=0, description="Models tracked")
    total_predictions_tracked: int = Field(default=0, description="Predictions tracked")
    total_experiments: int = Field(default=0, description="Experiments")

    registry_models: int = Field(default=0, description="Registry models")
    production_model: Optional[str] = Field(None, description="Current production model")

    safety_boundary: str = Field(default="ENFORCED", description="Safety boundary status")
