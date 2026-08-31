"""
Outcome and Reward schemas for Razorpay CloseLoop Phase 7J.

Defines outcome recording and reward generation for resolved cases.

Ground truth may be used for:
- offline evaluation
- reward calculation
- model training
- analysis

It must NOT be used to decide the current case.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Enums
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowOutcome(str, Enum):
    """Final outcome of the workflow."""
    RESOLVED_AUTO = "RESOLVED_AUTO"
    RESOLVED_HUMAN = "RESOLVED_HUMAN"
    REJECTED_BY_HUMAN = "REJECTED_BY_HUMAN"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class RewardType(str, Enum):
    """Type of reward signal."""
    CORRECT_RESOLUTION = "CORRECT_RESOLUTION"
    INCORRECT_RESOLUTION = "INCORRECT_RESOLUTION"
    PARTIAL_CREDIT = "PARTIAL_CREDIT"
    NO_REWARD = "NO_REWARD"
    PENALTY = "PENALTY"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Outcome Record
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowOutcomeRecord(BaseModel):
    """Complete record of workflow outcome.

    Captures everything that happened during the workflow
    for auditing, learning, and historical storage.
    """
    # Identity
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(default=None, description="Selected candidate")

    # Decision path
    decision: str = Field(..., description="Final guardrail decision")
    resolution_type: Optional[str] = Field(default=None, description="Resolution applied")
    authorization_source: Optional[str] = Field(default=None, description="AUTO_GUARDRAIL or HUMAN_APPROVAL")
    human_approved: bool = Field(default=False, description="Whether human approved")

    # Verification
    verification_passed: bool = Field(default=False, description="Whether verification passed")
    verification_action: Optional[str] = Field(default=None, description="Verification result")

    # Financial
    financial_adjustment_paise: int = Field(default=0, description="Proposed adjustment in paise")
    action_created: bool = Field(default=False, description="Whether action request was created")

    # Final outcome
    outcome: WorkflowOutcome = Field(..., description="Final workflow outcome")
    outcome_reason: Optional[str] = Field(default=None, description="Why this outcome occurred")

    # Metadata
    confidence: Optional[float] = Field(default=None, description="Final confidence")
    risk: Optional[str] = Field(default=None, description="Risk category")
    exception_type: Optional[str] = Field(default=None, description="Classified exception type")
    nodes_executed: List[str] = Field(default_factory=list, description="Nodes that ran")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When record was created")
    completed_at: Optional[datetime] = Field(default=None, description="When workflow completed")

    def summary(self) -> str:
        return (
            f"Outcome: {self.outcome.value} | "
            f"Decision: {self.decision} | "
            f"Resolution: {self.resolution_type or 'none'} | "
            f"Amount: {self.financial_adjustment_paise} paise"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reward Signal
# ─────────────────────────────────────────────────────────────────────────────


class RewardSignal(BaseModel):
    """Reward signal for future learning.

    This signal is generated AFTER workflow completion.
    It may use ground truth for evaluation purposes ONLY.

    The reward does NOT affect the current case's financial decision.
    It is used for:
    - model training data
    - offline evaluation
    - performance tracking
    - agent learning
    """
    # Identity
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")

    # Reward
    reward_type: RewardType = Field(..., description="Type of reward")
    reward_value: float = Field(..., description="Numeric reward (-1.0 to 1.0)")
    reward_reason: str = Field(..., description="Why this reward was assigned")

    # Components
    resolution_correct: Optional[bool] = Field(
        default=None, description="Whether resolution was correct (uses ground truth)"
    )
    verification_bonus: float = Field(
        default=0.0, description="Bonus for passing verification"
    )
    financial_accuracy: Optional[float] = Field(
        default=None, description="Financial accuracy score (0.0-1.0)"
    )
    human_approval_bonus: float = Field(
        default=0.0, description="Bonus for human approval"
    )

    # Ground truth context (ONLY for offline evaluation)
    ground_truth_exception_type: Optional[str] = Field(
        default=None, description="Ground truth exception type (evaluation only)"
    )
    ground_truth_resolvable: Optional[bool] = Field(
        default=None, description="Whether case was resolvable (evaluation only)"
    )
    ground_truth_risk: Optional[str] = Field(
        default=None, description="Ground truth risk category (evaluation only)"
    )

    # Metadata
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: Optional[str] = Field(default=None, description="Model version at time of reward")


# ─────────────────────────────────────────────────────────────────────────────
# Historical Learning Record
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalLearningRecord(BaseModel):
    """Record for future Phase 4 retrieval.

    Combines outcome + reward into a single retrievable unit.
    """
    # Identity
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")

    # What happened
    exception_type: str = Field(..., description="Classified exception type")
    resolution_type: Optional[str] = Field(default=None, description="Resolution applied")
    outcome: WorkflowOutcome = Field(..., description="Final outcome")

    # Financial context
    financial_adjustment_paise: int = Field(default=0, description="Adjustment in paise")
    confidence: Optional[float] = Field(default=None, description="Final confidence")
    risk: Optional[str] = Field(default=None, description="Risk category")

    # Evidence context
    evidence_coverage: Optional[float] = Field(default=None)
    evidence_consistency: Optional[float] = Field(default=None)
    supporting_evidence_count: int = Field(default=0)

    # Resolution context
    verification_passed: bool = Field(default=False)
    human_approved: bool = Field(default=False)
    authorization_source: Optional[str] = Field(default=None)

    # Reward
    reward: Optional[RewardSignal] = Field(default=None, description="Reward signal")

    # Ground truth (evaluation only — not used in production decisions)
    ground_truth_exception_type: Optional[str] = Field(default=None)
    ground_truth_resolution: Optional[str] = Field(default=None)
    ground_truth_resolvable: Optional[bool] = Field(default=None)

    # Metadata
    nodes_executed: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(default=None)

    def to_retrieval_features(self) -> Dict[str, Any]:
        """Extract features for future similarity retrieval."""
        return {
            "exception_type": self.exception_type,
            "resolution_type": self.resolution_type,
            "outcome": self.outcome.value,
            "financial_adjustment_paise": self.financial_adjustment_paise,
            "confidence": self.confidence,
            "risk": self.risk,
            "evidence_coverage": self.evidence_coverage,
            "supporting_evidence_count": self.supporting_evidence_count,
            "verification_passed": self.verification_passed,
            "human_approved": self.human_approved,
        }
