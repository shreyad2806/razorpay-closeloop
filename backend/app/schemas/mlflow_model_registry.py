"""
MLflow Model Registry schemas for Razorpay CloseLoop Phase 10G.

Defines a controlled lifecycle for model versions:
  CANDIDATE → VALIDATION → PRODUCTION → ARCHIVED

Safety principle:
  Model Registry controls model lifecycle.
  Phase 6 controls financial safety.
  Registry MUST NOT bypass Phase 6.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle States
# ─────────────────────────────────────────────────────────────────────────────


class ModelLifecycleState(str, Enum):
    """Lifecycle states for a model version in the MLflow registry."""
    CANDIDATE = "CANDIDATE"
    VALIDATION = "VALIDATION"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


# Valid state transitions
VALID_TRANSITIONS: Dict[ModelLifecycleState, List[ModelLifecycleState]] = {
    ModelLifecycleState.CANDIDATE: [ModelLifecycleState.VALIDATION, ModelLifecycleState.ARCHIVED],
    ModelLifecycleState.VALIDATION: [ModelLifecycleState.PRODUCTION, ModelLifecycleState.CANDIDATE, ModelLifecycleState.ARCHIVED],
    ModelLifecycleState.PRODUCTION: [ModelLifecycleState.ARCHIVED, ModelLifecycleState.VALIDATION],
    ModelLifecycleState.ARCHIVED: [ModelLifecycleState.PRODUCTION],  # Rollback only
}


def is_valid_transition(from_state: ModelLifecycleState, to_state: ModelLifecycleState) -> bool:
    """Check if a state transition is valid."""
    return to_state in VALID_TRANSITIONS.get(from_state, [])


# ─────────────────────────────────────────────────────────────────────────────
# Registry Model Entry
# ─────────────────────────────────────────────────────────────────────────────


class RegistryModelEntry(BaseModel):
    """A single model version entry in the MLflow Model Registry."""
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    model_version: str = Field(..., description="Model version string")
    state: ModelLifecycleState = Field(
        default=ModelLifecycleState.CANDIDATE,
        description="Current lifecycle state",
    )

    # MLflow linkage
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    mlflow_experiment_name: Optional[str] = Field(None, description="MLflow experiment")

    # Dataset lineage
    dataset_version: Optional[str] = Field(None, description="Training dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    algorithm: Optional[str] = Field(None, description="Training algorithm")

    # Metrics snapshot
    accuracy: Optional[float] = Field(None, description="Test accuracy")
    f1_macro: Optional[float] = Field(None, description="Test F1 macro")
    precision_macro: Optional[float] = Field(None, description="Test precision macro")
    false_automation: Optional[int] = Field(None, description="False automation count")
    high_value_errors: Optional[int] = Field(None, description="High-value error count")

    # Lifecycle timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When registered as CANDIDATE"
    )
    validated_at: Optional[datetime] = Field(None, description="When moved to VALIDATION")
    promoted_at: Optional[datetime] = Field(None, description="When promoted to PRODUCTION")
    archived_at: Optional[datetime] = Field(None, description="When moved to ARCHIVED")

    # Previous model (for rollback reference)
    previous_model_id: Optional[str] = Field(
        None, description="Model ID that this version replaces"
    )
    previous_model_version: Optional[str] = Field(
        None, description="Version that this version replaces"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def summary(self) -> str:
        return (
            f"Model: {self.model_name} v{self.model_version} | "
            f"State: {self.state.value} | "
            f"Run: {self.mlflow_run_id[:8] + '...' if self.mlflow_run_id else 'none'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle Transition Record
# ─────────────────────────────────────────────────────────────────────────────


class LifecycleTransition(BaseModel):
    """Immutable record of a lifecycle state transition."""
    transition_id: str = Field(..., description="Unique transition ID")
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    from_state: ModelLifecycleState = Field(..., description="Previous state")
    to_state: ModelLifecycleState = Field(..., description="New state")
    reason: str = Field(default="", description="Reason for transition")
    performed_by: str = Field(default="system", description="Who performed transition")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When transition occurred"
    )

    def summary(self) -> str:
        return (
            f"{self.from_state.value} → {self.to_state.value} | "
            f"v{self.model_version} | {self.reason}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validation Gate Configuration
# ─────────────────────────────────────────────────────────────────────────────


class ValidationGateConfig(BaseModel):
    """Configuration for the validation gate (CANDIDATE → VALIDATION)."""
    require_training_success: bool = Field(
        default=True, description="Training must have succeeded"
    )
    require_evaluation_metrics: bool = Field(
        default=True, description="Evaluation metrics must exist"
    )
    require_safety_checks: bool = Field(
        default=True, description="Safety checks must pass"
    )
    min_accuracy: float = Field(
        default=0.50, description="Minimum accuracy"
    )
    min_f1_macro: float = Field(
        default=0.40, description="Minimum F1 macro"
    )
    max_false_automation: int = Field(
        default=10, description="Maximum false automation count"
    )
    max_high_value_errors: int = Field(
        default=0, description="Maximum high-value errors"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate Configuration
# ─────────────────────────────────────────────────────────────────────────────


class PromotionGateConfig(BaseModel):
    """Configuration for the promotion gate (VALIDATION → PRODUCTION)."""
    require_validation_pass: bool = Field(
        default=True, description="Validation gate must have passed"
    )
    require_unified_evaluation: bool = Field(
        default=True, description="Unified evaluation must have been performed"
    )
    evaluation_verdict_must_be: str = Field(
        default="PROMOTE", description="Required evaluation verdict"
    )
    min_accuracy: float = Field(default=0.60, description="Minimum accuracy")
    min_f1_macro: float = Field(default=0.50, description="Minimum F1 macro")
    max_false_automation: int = Field(default=5, description="Maximum false automation")
    max_high_value_errors: int = Field(default=0, description="Maximum HV errors")


# ─────────────────────────────────────────────────────────────────────────────
# Registry Summary
# ─────────────────────────────────────────────────────────────────────────────


class RegistrySummary(BaseModel):
    """Summary of the MLflow Model Registry state."""
    total_models: int = Field(default=0, description="Total model versions")
    candidate_count: int = Field(default=0, description="CANDIDATE models")
    validation_count: int = Field(default=0, description="VALIDATION models")
    production_count: int = Field(default=0, description="PRODUCTION models")
    archived_count: int = Field(default=0, description="ARCHIVED models")
    production_model: Optional[RegistryModelEntry] = Field(
        None, description="Current production model"
    )
    total_transitions: int = Field(default=0, description="Total lifecycle transitions")
    last_promotion_at: Optional[datetime] = Field(
        None, description="Last promotion timestamp"
    )

    def summary(self) -> str:
        return (
            f"Registry: {self.total_models} models | "
            f"Production: {self.production_count} | "
            f"Candidate: {self.candidate_count} | "
            f"Validation: {self.validation_count} | "
            f"Archived: {self.archived_count}"
        )
