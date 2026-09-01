"""
Reward Engine schemas for Razorpay CloseLoop Phase 9B.

Defines the expanded reward taxonomy and transparent reward calculation.

Safety principle:
  Reward is an evaluation signal.
  Reward MUST NOT directly authorize financial execution.
  Phase 6 remains the final safety authority.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Reward Categories
# ─────────────────────────────────────────────────────────────────────────────


class RewardCategory(str, Enum):
    """Controlled taxonomy of reward categories.

    Categories reflect the full spectrum from correct automation
    to dangerous errors. Ordering matters: the most positive is
    first, the most negative is last.
    """
    CORRECT_AUTO_RESOLUTION = "CORRECT_AUTO_RESOLUTION"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    CORRECT_ESCALATION = "CORRECT_ESCALATION"
    UNNECESSARY_ESCALATION = "UNNECESSARY_ESCALATION"
    INCORRECT_AUTO_RESOLUTION = "INCORRECT_AUTO_RESOLUTION"
    HIGH_VALUE_ERROR = "HIGH_VALUE_ERROR"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"


class FinancialRiskLevel(str, Enum):
    """Financial risk classification for reward weighting."""
    NEGLIGIBLE = "NEGLIGIBLE"   # < ₹100
    LOW = "LOW"                 # ₹100 – ₹1,000
    MEDIUM = "MEDIUM"           # ₹1,000 – ₹10,000
    HIGH = "HIGH"               # ₹10,000 – ₹1,00,000
    CRITICAL = "CRITICAL"       # > ₹1,00,000


# ─────────────────────────────────────────────────────────────────────────────
# Reward Configuration
# ─────────────────────────────────────────────────────────────────────────────


class RewardWeights(BaseModel):
    """Configurable weights for reward calculation.

    These determine how much each factor contributes to the
    final reward value. All weights are transparent and auditable.
    """
    # Base reward by category
    base_rewards: Dict[str, float] = Field(
        default_factory=lambda: {
            "CORRECT_AUTO_RESOLUTION": 0.8,
            "HUMAN_CONFIRMED": 0.6,
            "CORRECT_ESCALATION": 0.3,
            "UNNECESSARY_ESCALATION": -0.3,
            "INCORRECT_AUTO_RESOLUTION": -0.7,
            "HIGH_VALUE_ERROR": -0.95,
            "VERIFICATION_FAILURE": -0.8,
        },
        description="Base reward value for each category",
    )

    # Verification modifiers
    verification_passed_bonus: float = Field(
        default=0.1, description="Bonus when verification passes"
    )
    verification_failed_penalty: float = Field(
        default=-0.3, description="Penalty when verification fails"
    )

    # Financial risk modifiers
    financial_risk_penalties: Dict[str, float] = Field(
        default_factory=lambda: {
            "NEGLIGIBLE": 0.0,
            "LOW": -0.05,
            "MEDIUM": -0.15,
            "HIGH": -0.3,
            "CRITICAL": -0.5,
        },
        description="Additional penalty by financial risk level",
    )

    # Human feedback modifiers
    human_approve_bonus: float = Field(
        default=0.15, description="Bonus when human approves"
    )
    human_reject_penalty: float = Field(
        default=-0.2, description="Penalty when human rejects"
    )
    human_correct_penalty: float = Field(
        default=-0.15, description="Penalty when human corrects prediction"
    )

    # Confidence modifier
    confidence_bonus_scale: float = Field(
        default=0.1,
        description="Max bonus/penalty scaled by confidence (0-0.1)",
        ge=0.0,
        le=0.5,
    )

    # Discrepancy resolution
    discrepancy_eliminated_bonus: float = Field(
        default=0.1, description="Bonus when discrepancy fully eliminated"
    )
    discrepancy_remainder_penalty: float = Field(
        default=-0.15, description="Penalty when discrepancy remains"
    )

    # Unintended changes penalty
    unintended_change_penalty: float = Field(
        default=-0.2, description="Penalty per unintended financial change"
    )


class RewardConfig(BaseModel):
    """Complete reward configuration."""
    weights: RewardWeights = Field(
        default_factory=RewardWeights, description="Reward weights"
    )
    high_value_threshold_paise: int = Field(
        default=10_000_000,  # ₹1,00,000
        description="Threshold above which HIGH_VALUE_ERROR applies",
    )
    policy_version: str = Field(
        default="1.0.0", description="Reward policy version"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reward Calculation Components
# ─────────────────────────────────────────────────────────────────────────────


class RewardComponent(BaseModel):
    """A single component of the reward calculation.

    Every component is independently explainable.
    """
    component_name: str = Field(
        ..., description="Name of the reward component"
    )
    value: float = Field(
        ..., description="Contribution to final reward"
    )
    reason: str = Field(
        ..., description="Why this component has this value"
    )
    weight_used: Optional[float] = Field(
        default=None, description="Weight/config value that produced this component"
    )


class RewardBreakdown(BaseModel):
    """Complete transparent breakdown of reward calculation.

    Every reward must be explainable through this breakdown.
    No opaque calculations.
    """
    base_reward: RewardComponent = Field(
        ..., description="Base reward from category"
    )
    verification_component: RewardComponent = Field(
        ..., description="Verification impact"
    )
    financial_risk_component: RewardComponent = Field(
        ..., description="Financial risk impact"
    )
    human_feedback_component: RewardComponent = Field(
        ..., description="Human feedback impact"
    )
    confidence_component: RewardComponent = Field(
        ..., description="Confidence impact"
    )
    discrepancy_component: RewardComponent = Field(
        ..., description="Discrepancy resolution impact"
    )
    unintended_changes_component: RewardComponent = Field(
        ..., description="Unintended changes impact"
    )

    def all_components(self) -> List[RewardComponent]:
        """Return all components in order."""
        return [
            self.base_reward,
            self.verification_component,
            self.financial_risk_component,
            self.human_feedback_component,
            self.confidence_component,
            self.discrepancy_component,
            self.unintended_changes_component,
        ]

    def total(self) -> float:
        """Sum of all components, clamped to [-1.0, 1.0]."""
        raw = sum(c.value for c in self.all_components())
        return max(-1.0, min(1.0, raw))


# ─────────────────────────────────────────────────────────────────────────────
# Reward Record
# ─────────────────────────────────────────────────────────────────────────────


class RewardRecord(BaseModel):
    """Complete reward record with full explanation.

    Stores everything needed for auditing and learning.
    """
    # Identity
    reward_id: str = Field(..., description="Unique reward identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")

    # Category
    category: RewardCategory = Field(
        ..., description="Reward category"
    )

    # Final reward
    reward_value: float = Field(
        ..., description="Final reward value (-1.0 to 1.0)", ge=-1.0, le=1.0
    )
    reward_reason: str = Field(
        ..., description="Human-readable explanation of the reward"
    )

    # Breakdown
    breakdown: RewardBreakdown = Field(
        ..., description="Transparent reward calculation breakdown"
    )

    # Input signals (what fed into the calculation)
    resolution_correct: Optional[bool] = Field(
        default=None, description="Whether resolution was correct"
    )
    was_auto_resolved: bool = Field(
        default=False, description="Whether system auto-resolved"
    )
    was_human_approved: bool = Field(
        default=False, description="Whether human approved"
    )
    was_human_rejected: bool = Field(
        default=False, description="Whether human rejected"
    )
    was_human_corrected: bool = Field(
        default=False, description="Whether human corrected"
    )
    was_escalated: bool = Field(
        default=False, description="Whether case was escalated"
    )
    escalation_was_correct: Optional[bool] = Field(
        default=None, description="Whether escalation was the right call"
    )
    verification_passed: bool = Field(
        default=False, description="Whether verification passed"
    )

    # Financial
    financial_impact_paise: int = Field(
        default=0, description="Financial impact in paise"
    )
    financial_risk_level: FinancialRiskLevel = Field(
        default=FinancialRiskLevel.NEGLIGIBLE,
        description="Financial risk classification",
    )
    discrepancy_eliminated: bool = Field(
        default=False, description="Whether discrepancy was eliminated"
    )
    unintended_changes: int = Field(
        default=0, description="Number of unintended financial changes"
    )

    # Confidence
    confidence: Optional[float] = Field(
        default=None, description="System confidence at time of decision"
    )

    # Ground truth (evaluation only)
    ground_truth_exception_type: Optional[str] = Field(
        default=None, description="Ground truth (evaluation only)"
    )
    ground_truth_resolution: Optional[str] = Field(
        default=None, description="Ground truth resolution (evaluation only)"
    )

    # Metadata
    policy_version: str = Field(
        default="1.0.0", description="Reward policy version"
    )
    model_version: Optional[str] = Field(
        default=None, description="Model version at time of reward"
    )
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When reward was calculated"
    )

    def summary(self) -> str:
        return (
            f"Reward: {self.reward_value:+.3f} | "
            f"Category: {self.category.value} | "
            f"Workflow: {self.workflow_id} | "
            f"Correct: {self.resolution_correct} | "
            f"Risk: {self.financial_risk_level.value}"
        )

    def is_positive(self) -> bool:
        return self.reward_value > 0

    def is_negative(self) -> bool:
        return self.reward_value < 0

    def is_neutral(self) -> bool:
        return self.reward_value == 0.0

    def magnitude(self) -> float:
        return abs(self.reward_value)
