"""
Agent State schema for Razorpay CloseLoop Phase 7A.

Defines the typed state representation for the LangGraph workflow.

This state carries data produced by workflow nodes.
Nodes perform work; state contains data.

The state is the single source of truth for workflow execution.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Status
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowStatus(str, Enum):
    """Status of the overall workflow."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class NodeStatus(str, Enum):
    """Status of individual workflow nodes."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "SKIPPED"
    SKIPPED = "SKIPPED"


class HumanApprovalStatus(str, Enum):
    """Human-in-the-loop approval status."""

    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RESUMED = "RESUMED"


class VerificationStatus(str, Enum):
    """Post-resolution verification status."""

    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class RewardStatus(str, Enum):
    """Reward calculation status."""

    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    CALCULATED = "CALCULATED"
    FAILED = "FAILED"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Metadata
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowMetadata(BaseModel):
    """Metadata for workflow execution tracking."""

    workflow_id: str = Field(
        ..., description="Unique workflow identifier"
    )
    exception_id: str = Field(
        ..., description="Exception being investigated"
    )
    case_id: Optional[str] = Field(
        default=None, description="Case being investigated"
    )

    # Current state
    current_node: Optional[str] = Field(
        default=None, description="Currently executing node name"
    )
    workflow_status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Overall workflow status",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the workflow was created",
    )
    started_at: Optional[datetime] = Field(
        default=None, description="When execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When execution completed"
    )
    last_updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last state update timestamp",
    )

    # Retry
    retry_count: int = Field(
        default=0, description="Number of retry attempts", ge=0
    )
    max_retries: int = Field(
        default=3, description="Maximum retry attempts", ge=0
    )

    # Errors
    errors: List[str] = Field(
        default_factory=list, description="Accumulated error messages"
    )

    # Audit
    nodes_executed: List[str] = Field(
        default_factory=list,
        description="Ordered list of nodes that have executed",
    )
    execution_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed execution log entries",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Human-in-the-Loop State
# ─────────────────────────────────────────────────────────────────────────────


class HumanReviewState(BaseModel):
    """State for human-in-the-loop approval workflow.

    Do not implement human approval logic yet.
    Only prepare the state structure.
    """

    approval_status: HumanApprovalStatus = Field(
        default=HumanApprovalStatus.NOT_REQUIRED,
        description="Current approval status",
    )
    assigned_reviewer: Optional[str] = Field(
        default=None, description="Assigned human reviewer ID"
    )
    review_requested_at: Optional[datetime] = Field(
        default=None, description="When review was requested"
    )
    review_completed_at: Optional[datetime] = Field(
        default=None, description="When review was completed"
    )
    reviewer_notes: Optional[str] = Field(
        default=None, description="Notes from the reviewer"
    )
    review_reason: Optional[str] = Field(
        default=None, description="Why human review was requested"
    )
    review_priority: Optional[str] = Field(
        default=None, description="Review priority level"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verification State
# ─────────────────────────────────────────────────────────────────────────────


class VerificationState(BaseModel):
    """State for post-resolution verification.

    Do not implement verification logic yet.
    Only prepare the state structure.
    """

    verification_status: VerificationStatus = Field(
        default=VerificationStatus.NOT_REQUIRED,
        description="Current verification status",
    )
    verification_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Verification result details"
    )
    verification_errors: List[str] = Field(
        default_factory=list, description="Verification error messages"
    )
    verified_at: Optional[datetime] = Field(
        default=None, description="When verification was performed"
    )
    verified_by: Optional[str] = Field(
        default=None, description="What performed verification"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reward State
# ─────────────────────────────────────────────────────────────────────────────


class RewardState(BaseModel):
    """State for resolution reward/feedback tracking.

    Do not implement reward calculation yet.
    Only prepare the state structure.
    """

    reward_status: RewardStatus = Field(
        default=RewardStatus.NOT_REQUIRED,
        description="Current reward status",
    )
    reward: Optional[float] = Field(
        default=None, description="Calculated reward value"
    )
    reward_reason: Optional[str] = Field(
        default=None, description="Explanation of reward calculation"
    )
    reward_calculated_at: Optional[datetime] = Field(
        default=None, description="When reward was calculated"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent State (Main)
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """
    Typed state for the LangGraph workflow.

    Contains all data produced by workflow nodes.
    Nodes perform work; state contains data.

    This state is the single source of truth for workflow execution.
    Each node returns only the fields it changes.
    """

    # ── Workflow Metadata ──
    metadata: WorkflowMetadata = Field(
        ..., description="Workflow execution metadata"
    )

    # ── Phase 2: Reconciliation ──
    reconciliation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 2 deterministic reconciliation result",
    )

    # ── Phase 3: Evidence ──
    evidence_package: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 3 evidence retrieval package",
    )
    evidence_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 3 NetworkX evidence graph (serialized)",
    )
    explanation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 3 deterministic explanation",
    )
    evidence_quality: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 3 evidence quality scores",
    )

    # ── Phase 4: Intelligence ──
    classification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 4 ML exception classification",
    )
    similar_cases: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 4 historical similar cases",
    )
    intelligence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 4 aggregated intelligence",
    )

    # ── Phase 5: Resolution ──
    candidates: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 5 resolution candidates",
    )
    selected_candidate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 5 selected resolution candidate",
    )
    candidate_scores: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 5 candidate scores",
    )

    # ── Phase 6: Guardrails ──
    confidence: Optional[float] = Field(
        default=None,
        description="Phase 6 final confidence",
        ge=0.0,
        le=1.0,
    )
    risk: Optional[str] = Field(
        default=None,
        description="Phase 6 risk category",
    )
    guardrail_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 6 guardrail engine result",
    )
    decision: Optional[str] = Field(
        default=None,
        description="Phase 6 final decision: AUTO, HUMAN_REVIEW, UNRESOLVED",
    )

    # ── Phase 7: Human / Verification / Reward ──
    human_review: HumanReviewState = Field(
        default_factory=HumanReviewState,
        description="Human-in-the-loop state",
    )
    verification: VerificationState = Field(
        default_factory=VerificationState,
        description="Post-resolution verification state",
    )
    reward: RewardState = Field(
        default_factory=RewardState,
        description="Resolution reward tracking state",
    )

    # ── Error tracking ──
    errors: List[str] = Field(
        default_factory=list,
        description="Accumulated errors across all nodes",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Accumulated warnings",
    )

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [
            f"Workflow: {self.metadata.workflow_id}",
            f"Exception: {self.metadata.exception_id}",
            f"Status: {self.metadata.workflow_status.value}",
            f"Current: {self.metadata.current_node or 'none'}",
        ]
        if self.decision:
            parts.append(f"Decision: {self.decision}")
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        return " | ".join(parts)
