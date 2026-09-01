"""
Model Promotion schemas for Razorpay CloseLoop Phase 9F.

Defines the promotion gate, promotion records, and rollback mechanisms.

Safety principle:
  Promotion requires passing explicit safety criteria.
  Higher accuracy alone is NOT sufficient for promotion.
  Safety-critical metrics must NOT regress.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Enums
# ─────────────────────────────────────────────────────────────────────────────


class PromotionDecision(str, Enum):
    """Decision on model promotion."""
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class PromotionGateStatus(str, Enum):
    """Status of individual gate checks."""
    PASSED = "PASSED"
    FAILED = "FAILED"


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate Configuration
# ─────────────────────────────────────────────────────────────────────────────


class PromotionThresholds(BaseModel):
    """Configurable thresholds for the promotion gate."""
    min_precision: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Minimum precision for promotion",
    )
    min_recall: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum recall for promotion",
    )
    min_f1: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Minimum F1 for promotion",
    )
    max_false_automation: int = Field(
        default=5, ge=0,
        description="Maximum false automation count allowed",
    )
    max_high_value_errors: int = Field(
        default=0, ge=0,
        description="Maximum high-value errors (0 = none allowed)",
    )
    max_verification_failure_rate: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Maximum verification failure rate",
    )
    max_unknown_case_errors: int = Field(
        default=3, ge=0,
        description="Maximum unknown case errors",
    )
    min_accuracy: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum accuracy for promotion",
    )
    max_false_auto_increase_pct: float = Field(
        default=20.0, ge=0.0,
        description="Maximum % increase in false automation allowed",
    )
    no_hv_error_increase: bool = Field(
        default=True,
        description="Whether high-value errors must not increase at all",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate Check
# ─────────────────────────────────────────────────────────────────────────────


class GateCheck(BaseModel):
    """Result of a single promotion gate check."""
    check_name: str = Field(..., description="Name of the check")
    status: PromotionGateStatus = Field(..., description="PASSED or FAILED")
    current_value: float = Field(..., description="Current model value")
    candidate_value: float = Field(..., description="Candidate model value")
    threshold: float = Field(..., description="Threshold compared against")
    description: str = Field(default="", description="Check description")


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate Result
# ─────────────────────────────────────────────────────────────────────────────


class PromotionGateResult(BaseModel):
    """Complete promotion gate evaluation."""
    gate_id: str = Field(..., description="Unique gate evaluation ID")
    candidate_model_id: str = Field(..., description="Candidate model ID")
    candidate_version: str = Field(..., description="Candidate version")
    current_model_id: Optional[str] = Field(
        default=None, description="Current active model ID"
    )
    current_version: Optional[str] = Field(
        default=None, description="Current active model version"
    )

    # Gate checks
    checks: List[GateCheck] = Field(
        default_factory=list, description="All gate checks performed"
    )
    all_passed: bool = Field(
        default=False, description="Whether all checks passed"
    )
    failed_checks: List[str] = Field(
        default_factory=list, description="Names of failed checks"
    )

    # Decision
    decision: PromotionDecision = Field(
        default=PromotionDecision.PENDING,
        description="Promotion decision",
    )
    decision_reason: str = Field(
        default="", description="Why this decision was made"
    )

    # Thresholds used
    thresholds: PromotionThresholds = Field(
        default_factory=PromotionThresholds,
        description="Thresholds applied",
    )

    # Metadata
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When gate was evaluated",
    )
    evaluated_by: str = Field(
        default="system", description="Who/what evaluated the gate"
    )

    def summary(self) -> str:
        return (
            f"Gate: {self.decision.value} | "
            f"Candidate: v{self.candidate_version} | "
            f"Checks: {len(self.checks)} | "
            f"Failed: {len(self.failed_checks)} | "
            f"Reason: {self.decision_reason}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Record
# ─────────────────────────────────────────────────────────────────────────────


class PromotionRecord(BaseModel):
    """Immutable record of a model promotion attempt."""
    record_id: str = Field(..., description="Unique record ID")
    gate_id: str = Field(..., description="Gate evaluation ID")

    # Models
    old_model_id: Optional[str] = Field(
        default=None, description="Previous active model ID"
    )
    old_version: Optional[str] = Field(
        default=None, description="Previous active model version"
    )
    new_model_id: str = Field(..., description="New model ID")
    new_version: str = Field(..., description="New model version")

    # Decision
    decision: PromotionDecision = Field(..., description="Promotion decision")
    reason: str = Field(default="", description="Decision reason")

    # Metrics snapshot
    old_metrics_summary: Optional[Dict[str, float]] = Field(
        default=None, description="Summary of old model metrics"
    )
    new_metrics_summary: Optional[Dict[str, float]] = Field(
        default=None, description="Summary of new model metrics"
    )

    # Safety thresholds
    thresholds: PromotionThresholds = Field(
        default_factory=PromotionThresholds,
        description="Safety thresholds applied",
    )

    # Audit
    performed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When promotion was attempted",
    )
    performed_by: str = Field(
        default="system", description="Who performed the promotion"
    )
    dataset_version: Optional[str] = Field(
        default=None, description="Dataset version used for training"
    )
    feature_schema_version: Optional[str] = Field(
        default=None, description="Feature schema version"
    )

    def summary(self) -> str:
        return (
            f"Promotion: {self.decision.value} | "
            f"v{self.old_version or 'none'} → v{self.new_version} | "
            f"Reason: {self.reason}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Record
# ─────────────────────────────────────────────────────────────────────────────


class RollbackRecord(BaseModel):
    """Immutable record of a model rollback."""
    record_id: str = Field(..., description="Unique rollback ID")
    promotion_record_id: Optional[str] = Field(
        default=None, description="Promotion record being rolled back"
    )

    # Models
    rolled_back_from_id: str = Field(..., description="Model being rolled back")
    rolled_back_from_version: str = Field(..., description="Version being rolled back")
    restored_to_id: str = Field(..., description="Model restored to")
    restored_to_version: str = Field(..., description="Version restored to")

    # Reason
    reason: str = Field(..., description="Why rollback was performed")

    # Audit
    performed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When rollback was performed",
    )
    performed_by: str = Field(
        default="system", description="Who performed the rollback"
    )

    def summary(self) -> str:
        return (
            f"Rollback: v{self.rolled_back_from_version} → "
            f"v{self.restored_to_version} | "
            f"Reason: {self.reason}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry State
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistryState(BaseModel):
    """Complete state of the model registry."""
    active_model_id: Optional[str] = Field(
        default=None, description="Currently active model ID"
    )
    active_version: Optional[str] = Field(
        default=None, description="Currently active model version"
    )
    archived_model_ids: List[str] = Field(
        default_factory=list, description="Archived model IDs"
    )
    total_promotions: int = Field(default=0, description="Total promotion attempts")
    total_rollbacks: int = Field(default=0, description="Total rollbacks")
    last_promotion_at: Optional[datetime] = Field(
        default=None, description="When last promotion occurred"
    )
    last_rollback_at: Optional[datetime] = Field(
        default=None, description="When last rollback occurred"
    )
