"""
Self-Learning Loop schemas for Razorpay CloseLoop Phase 9I.

Defines the complete learning cycle record and observability structures
for the integrated self-learning loop.

Safety principle:
  Learning can improve classification, resolution prediction, similar-case
  retrieval, policy recommendations, and candidate ranking.

  Learning CANNOT bypass:
    Phase 6 guardrails
    financial exposure limits
    conflict checks
    novelty checks
    verification
    execution authorization
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Learning Cycle Enums
# ─────────────────────────────────────────────────────────────────────────────

class LearningCycleStatus(str, Enum):
    """Status of a learning cycle."""
    RECORDING = "RECORDING"             # Recording outcome/feedback
    REWARD_CALCULATED = "REWARD_CALCULATED"
    EXAMPLE_BUILT = "EXAMPLE_BUILT"
    BATCH_READY = "BATCH_READY"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    COMPARING = "COMPARING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class PromotionAction(str, Enum):
    """Action taken on candidate model."""
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    NO_CANDIDATE = "NO_CANDIDATE"


# ─────────────────────────────────────────────────────────────────────────────
# Learning Cycle Record
# ─────────────────────────────────────────────────────────────────────────────

class LearningCycleRecord(BaseModel):
    """Complete record of a single learning cycle.

    Tracks a case from outcome recording through feedback, reward,
    dataset construction, training, evaluation, and promotion.
    """
    cycle_id: str = Field(..., description="Unique cycle identifier")
    cycle_number: int = Field(..., description="Sequential cycle number")

    # Source case
    exception_id: str = Field(..., description="Source exception ID")
    workflow_id: str = Field(..., description="Source workflow ID")
    case_id: Optional[str] = Field(default=None, description="Source case ID")

    # Status
    status: LearningCycleStatus = Field(
        default=LearningCycleStatus.RECORDING,
        description="Current cycle status",
    )

    # Outcome
    outcome_id: Optional[str] = Field(
        default=None, description="OutcomeRecord ID",
    )
    prediction_correct: Optional[bool] = Field(
        default=None, description="Whether system prediction was correct",
    )
    resolution_type: Optional[str] = Field(
        default=None, description="Resolution type applied",
    )
    was_auto_resolved: bool = Field(
        default=False, description="Whether system auto-resolved",
    )

    # Feedback
    feedback_id: Optional[str] = Field(
        default=None, description="FeedbackRecord ID",
    )
    feedback_type: Optional[str] = Field(
        default=None, description="APPROVE/REJECT/CORRECT/ESCALATE",
    )

    # Reward
    reward_id: Optional[str] = Field(
        default=None, description="RewardRecord ID",
    )
    reward_value: Optional[float] = Field(
        default=None, description="Reward value (-1.0 to 1.0)",
    )
    reward_category: Optional[str] = Field(
        default=None, description="Reward category",
    )

    # Learning dataset
    learning_example_id: Optional[str] = Field(
        default=None, description="LearningExample ID",
    )
    dataset_id: Optional[str] = Field(
        default=None, description="LearningDataset ID (batch)",
    )

    # Training
    candidate_model_id: Optional[str] = Field(
        default=None, description="Trained candidate model ID",
    )
    candidate_model_version: Optional[str] = Field(
        default=None, description="Candidate model version",
    )
    training_duration_seconds: Optional[float] = Field(
        default=None, description="Training duration",
    )

    # Evaluation
    evaluation_accuracy: Optional[float] = Field(
        default=None, description="Candidate evaluation accuracy",
    )
    evaluation_f1: Optional[float] = Field(
        default=None, description="Candidate evaluation F1",
    )
    evaluation_precision: Optional[float] = Field(
        default=None, description="Candidate evaluation precision",
    )

    # Promotion
    promotion_action: PromotionAction = Field(
        default=PromotionAction.NO_CANDIDATE,
        description="Promotion action taken",
    )
    promotion_reason: Optional[str] = Field(
        default=None, description="Why promoted/rejected/deferred",
    )
    promoted_model_version: Optional[str] = Field(
        default=None, description="Model version after promotion",
    )

    # Safety
    safety_maintained: bool = Field(
        default=True, description="Whether safety was maintained",
    )
    guardrail_decision: Optional[str] = Field(
        default=None, description="Phase 6 guardrail decision",
    )

    # Timestamps
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When cycle started",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When cycle completed",
    )
    outcome_recorded_at: Optional[datetime] = Field(
        default=None, description="When outcome was recorded",
    )
    feedback_received_at: Optional[datetime] = Field(
        default=None, description="When feedback was received",
    )
    reward_calculated_at: Optional[datetime] = Field(
        default=None, description="When reward was calculated",
    )
    trained_at: Optional[datetime] = Field(
        default=None, description="When candidate was trained",
    )
    evaluated_at: Optional[datetime] = Field(
        default=None, description="When candidate was evaluated",
    )
    promoted_at: Optional[datetime] = Field(
        default=None, description="When promotion was decided",
    )

    def summary(self) -> str:
        return (
            f"Cycle #{self.cycle_number} ({self.cycle_id}) | "
            f"Exception: {self.exception_id} | "
            f"Status: {self.status.value} | "
            f"Reward: {self.reward_value or 'N/A'} | "
            f"Promotion: {self.promotion_action.value}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Learning System State
# ─────────────────────────────────────────────────────────────────────────────

class LearningSystemState(BaseModel):
    """Complete state of the learning system."""
    total_cycles: int = Field(default=0, description="Total learning cycles")
    completed_cycles: int = Field(default=0, description="Completed cycles")

    # Current model
    active_model_id: Optional[str] = Field(
        default=None, description="Current active model ID",
    )
    active_model_version: Optional[str] = Field(
        default=None, description="Current active model version",
    )

    # Dataset state
    total_learning_examples: int = Field(
        default=0, description="Total learning examples generated",
    )
    current_dataset_version: Optional[str] = Field(
        default=None, description="Current dataset version",
    )

    # Batch state
    total_batches: int = Field(default=0, description="Total batches processed")
    active_batch_id: Optional[str] = Field(
        default=None, description="Currently collecting batch ID",
    )

    # Policy state
    active_policy_id: Optional[str] = Field(
        default=None, description="Current active policy ID",
    )
    active_policy_version: Optional[str] = Field(
        default=None, description="Current active policy version",
    )

    # Performance
    total_rewards: int = Field(default=0, description="Total rewards computed")
    avg_reward: Optional[float] = Field(
        default=None, description="Average reward across all cycles",
    )
    total_promotions: int = Field(default=0, description="Total promotions")
    total_rejections: int = Field(default=0, description="Total rejections")

    # Safety
    safety_maintained_all_cycles: bool = Field(
        default=True, description="Safety maintained across all cycles",
    )

    # Timestamps
    system_started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When learning system started",
    )
    last_cycle_completed_at: Optional[datetime] = Field(
        default=None, description="When last cycle completed",
    )
    last_promotion_at: Optional[datetime] = Field(
        default=None, description="When last promotion occurred",
    )

    def summary(self) -> str:
        return (
            f"Learning System | "
            f"Cycles: {self.completed_cycles}/{self.total_cycles} | "
            f"Model: {self.active_model_version or 'none'} | "
            f"Batches: {self.total_batches} | "
            f"Avg reward: {self.avg_reward or 'N/A'} | "
            f"Safety: {'OK' if self.safety_maintained_all_cycles else 'BREACH'}"
        )
