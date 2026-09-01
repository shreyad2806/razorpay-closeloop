"""
Feedback and Outcome schemas for Razorpay CloseLoop Phase 9A.

Defines structured feedback recording and outcome tracking.

Key design principle:
  PREDICTION ≠ ACTUAL OUTCOME ≠ HUMAN FEEDBACK ≠ VERIFICATION RESULT

These must remain separate and never collapse into one field.

Ground truth may be used for offline evaluation, dataset labeling, model
evaluation, and reward calculation. It must NOT be used to fake a
production/current outcome.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Types
# ─────────────────────────────────────────────────────────────────────────────


class FeedbackType(str, Enum):
    """Explicit human feedback outcomes."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    CORRECT = "CORRECT"
    ESCALATE = "ESCALATE"


class OutcomeStatus(str, Enum):
    """Status of the recorded outcome."""
    RECORDED = "RECORDED"
    PENDING_FEEDBACK = "PENDING_FEEDBACK"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
    REWARD_CALCULATED = "REWARD_CALCULATED"
    STORED_FOR_LEARNING = "STORED_FOR_LEARNING"


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Record
# ─────────────────────────────────────────────────────────────────────────────


class CorrectionDetail(BaseModel):
    """Details when a human corrects the system prediction."""
    original_resolution: str = Field(
        ..., description="Resolution type originally predicted by the system"
    )
    corrected_resolution: str = Field(
        ..., description="Resolution type the human says is correct"
    )
    correction_reason: str = Field(
        ..., description="Why the correction was necessary"
    )
    original_confidence: Optional[float] = Field(
        default=None, description="System confidence on the original prediction"
    )
    corrected_amount_paise: Optional[int] = Field(
        default=None,
        description="Corrected financial adjustment if different from predicted",
    )


class RejectionDetail(BaseModel):
    """Details when a human rejects the proposed resolution."""
    rejection_reason: str = Field(
        ..., description="Why the resolution was rejected"
    )
    suggested_alternative: Optional[str] = Field(
        default=None,
        description="Alternative resolution the reviewer suggests, if any",
    )
    risk_concern: Optional[str] = Field(
        default=None,
        description="Specific risk concern that caused rejection",
    )


class EscalationDetail(BaseModel):
    """Details when a human escalates for further review."""
    escalation_reason: str = Field(
        ..., description="Why the case was escalated"
    )
    escalation_target: Optional[str] = Field(
        default=None,
        description="Who/what the case is escalated to",
    )
    additional_context: Optional[str] = Field(
        default=None, description="Additional context for the escalation"
    )


class FeedbackRecord(BaseModel):
    """Structured human feedback record.

    Captures explicit human interaction with the resolution system.

    Each feedback record is immutable once created.
    If a reviewer changes their mind, create a new feedback record
    referencing the previous one via correction_of.
    """
    # Identity
    feedback_id: str = Field(..., description="Unique feedback identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(
        default=None, description="Candidate identifier that was reviewed"
    )

    # Feedback type
    feedback_type: FeedbackType = Field(
        ..., description="Type of human feedback"
    )

    # Actor
    reviewer: str = Field(
        ..., description="Who provided the feedback (reviewer ID/name)"
    )
    reviewer_role: Optional[str] = Field(
        default=None, description="Role of the reviewer (e.g., finance_ops, auditor)"
    )

    # System prediction being reviewed
    system_prediction: str = Field(
        ..., description="Resolution type the system predicted"
    )
    system_confidence: Optional[float] = Field(
        default=None, description="System confidence at time of prediction"
    )
    financial_adjustment_paise: int = Field(
        default=0, description="Proposed financial adjustment in paise"
    )

    # Feedback-specific details (one of these will be set based on feedback_type)
    correction: Optional[CorrectionDetail] = Field(
        default=None, description="Correction details (for CORRECT feedback)"
    )
    rejection: Optional[RejectionDetail] = Field(
        default=None, description="Rejection details (for REJECT feedback)"
    )
    escalation: Optional[EscalationDetail] = Field(
        default=None, description="Escalation details (for ESCALATE feedback)"
    )

    # APPROVE has no additional detail needed — just the feedback_type

    # Reason (generic fallback for simple approve/escalate)
    reason: Optional[str] = Field(
        default=None, description="General reason or comment"
    )

    # Evidence references reviewed
    evidence_references_reviewed: List[str] = Field(
        default_factory=list,
        description="Evidence record IDs the reviewer examined",
    )

    # Immutability
    correction_of: Optional[str] = Field(
        default=None,
        description="Feedback ID this record supersedes/corrects",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When feedback was recorded",
    )
    review_duration_seconds: Optional[float] = Field(
        default=None,
        description="How long the reviewer spent reviewing (seconds)",
    )

    # Metadata
    model_version: Optional[str] = Field(
        default=None, description="ML model version at time of review"
    )
    policy_version: Optional[str] = Field(
        default=None, description="Guardrail policy version at time of review"
    )

    def summary(self) -> str:
        parts = [
            f"Feedback: {self.feedback_type.value}",
            f"Workflow: {self.workflow_id}",
            f"System predicted: {self.system_prediction}",
        ]
        if self.correction:
            parts.append(f"Corrected to: {self.correction.corrected_resolution}")
        if self.rejection:
            parts.append(f"Reason: {self.rejection.rejection_reason}")
        if self.escalation:
            parts.append(f"Escalated: {self.escalation.escalation_reason}")
        return " | ".join(parts)

    def is_correction(self) -> bool:
        return self.feedback_type == FeedbackType.CORRECT and self.correction is not None

    def is_rejection(self) -> bool:
        return self.feedback_type == FeedbackType.REJECT and self.rejection is not None

    def is_escalation(self) -> bool:
        return self.feedback_type == FeedbackType.ESCALATE and self.escalation is not None

    def is_approval(self) -> bool:
        return self.feedback_type == FeedbackType.APPROVE


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Record
# ─────────────────────────────────────────────────────────────────────────────


class PredictionRecord(BaseModel):
    """What the system predicted — separated from actual outcome."""
    exception_type: Optional[str] = Field(
        default=None, description="ML-predicted exception type"
    )
    resolution_type: Optional[str] = Field(
        default=None, description="Predicted resolution type"
    )
    resolution_confidence: Optional[float] = Field(
        default=None, description="Confidence in resolution prediction"
    )
    exception_confidence: Optional[float] = Field(
        default=None, description="Confidence in exception classification"
    )
    model_version: Optional[str] = Field(
        default=None, description="Model version used for prediction"
    )


class ActualOutcomeRecord(BaseModel):
    """What actually happened — separate from what was predicted."""
    actual_resolution: Optional[str] = Field(
        default=None,
        description="Resolution that was actually applied (may differ from prediction)",
    )
    actual_exception_type: Optional[str] = Field(
        default=None,
        description="True exception type (ground truth, for offline evaluation only)",
    )
    resolution_correct: Optional[bool] = Field(
        default=None,
        description="Whether prediction matched actual (uses ground truth for eval)",
    )
    financial_impact_paise: int = Field(
        default=0, description="Actual financial impact in paise"
    )
    was_executed: bool = Field(
        default=False, description="Whether the resolution was actually executed"
    )
    was_verified: bool = Field(
        default=False, description="Whether the execution was verified"
    )
    was_rolled_back: bool = Field(
        default=False, description="Whether the execution was rolled back"
    )


class FinancialImpact(BaseModel):
    """Detailed financial impact of the resolution."""
    requested_adjustment_paise: int = Field(
        default=0, description="Original requested adjustment"
    )
    actual_adjustment_paise: int = Field(
        default=0, description="Actual adjustment applied"
    )
    difference_before_paise: int = Field(
        default=0, description="Financial discrepancy before resolution"
    )
    difference_after_paise: int = Field(
        default=0, description="Financial discrepancy after resolution"
    )
    discrepancy_eliminated: bool = Field(
        default=False, description="Whether the discrepancy was eliminated"
    )
    unintended_changes: int = Field(
        default=0, description="Number of unintended financial changes"
    )


class DataLineage(BaseModel):
    """Traceability for every learning example.

    Every learning example must be traceable to the full chain:
    exception → evidence → prediction → decision → guardrail →
    execution → verification → feedback → actual outcome
    """
    exception_id: str = Field(..., description="Source exception")
    evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence records used"
    )
    prediction_id: Optional[str] = Field(
        default=None, description="ML prediction reference"
    )
    decision_id: Optional[str] = Field(
        default=None, description="Guardrail decision reference"
    )
    execution_id: Optional[str] = Field(
        default=None, description="Execution record reference"
    )
    verification_id: Optional[str] = Field(
        default=None, description="Verification record reference"
    )
    feedback_id: Optional[str] = Field(
        default=None, description="Human feedback reference"
    )
    audit_event_ids: List[str] = Field(
        default_factory=list, description="Related audit event IDs"
    )
    reward_id: Optional[str] = Field(
        default=None, description="Reward signal reference"
    )
    historical_case_id: Optional[str] = Field(
        default=None, description="Stored historical case ID for future retrieval"
    )


class OutcomeRecord(BaseModel):
    """Complete outcome record separating prediction, actual outcome,
    human feedback, and verification result.

    This record is the primary input for learning.

    Key design:
    - prediction: what the system thought
    - actual_outcome: what really happened
    - human_feedback: what the human said
    - verification: what independent verification found
    - financial_impact: financial details
    - lineage: full traceability
    """
    # Identity
    outcome_id: str = Field(..., description="Unique outcome identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(
        default=None, description="Selected candidate identifier"
    )

    # Prediction (what the system predicted)
    prediction: PredictionRecord = Field(
        ..., description="System prediction record"
    )

    # Actual outcome (what really happened)
    actual_outcome: ActualOutcomeRecord = Field(
        ..., description="Actual outcome record"
    )

    # Human feedback
    human_feedback_id: Optional[str] = Field(
        default=None, description="Feedback record ID, if feedback was received"
    )
    human_feedback_type: Optional[FeedbackType] = Field(
        default=None, description="Type of feedback received"
    )
    human_override: bool = Field(
        default=False,
        description="Whether human overrode the system prediction",
    )

    # Verification
    verification_passed: bool = Field(
        default=False, description="Whether independent verification passed"
    )
    verification_notes: Optional[str] = Field(
        default=None, description="Verification details"
    )

    # Financial impact
    financial_impact: FinancialImpact = Field(
        default_factory=FinancialImpact,
        description="Financial impact details",
    )

    # Data lineage
    lineage: DataLineage = Field(
        ..., description="Full traceability for learning"
    )

    # Status
    status: OutcomeStatus = Field(
        default=OutcomeStatus.RECORDED,
        description="Outcome recording status",
    )

    def model_post_init(self, __context: object) -> None:
        """Auto-set status when feedback is present at construction time."""
        if self.human_feedback_id and self.status == OutcomeStatus.RECORDED:
            self.status = OutcomeStatus.FEEDBACK_RECEIVED

    # Ground truth (evaluation only — MUST NOT influence current decisions)
    ground_truth_exception_type: Optional[str] = Field(
        default=None,
        description="Ground truth exception type (evaluation only)",
    )
    ground_truth_resolution: Optional[str] = Field(
        default=None,
        description="Ground truth resolution (evaluation only)",
    )
    ground_truth_resolvable: Optional[bool] = Field(
        default=None,
        description="Whether case was truly resolvable (evaluation only)",
    )

    # Metadata
    decision: Optional[str] = Field(
        default=None, description="Guardrail decision: AUTO, HUMAN_REVIEW, UNRESOLVED"
    )
    confidence: Optional[float] = Field(
        default=None, description="Final confidence"
    )
    risk: Optional[str] = Field(
        default=None, description="Risk category"
    )
    nodes_executed: List[str] = Field(
        default_factory=list, description="Workflow nodes that executed"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When outcome was recorded",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When the full workflow completed"
    )
    feedback_received_at: Optional[datetime] = Field(
        default=None, description="When human feedback was received"
    )

    def summary(self) -> str:
        pred = self.prediction.resolution_type or "none"
        actual = self.actual_outcome.actual_resolution or "none"
        correct = "✓" if self.actual_outcome.resolution_correct else (
            "?" if self.actual_outcome.resolution_correct is None else "✗"
        )
        return (
            f"Outcome: {self.outcome_id} | "
            f"Predicted: {pred} | Actual: {actual} | "
            f"Correct: {correct} | "
            f"Executed: {self.actual_outcome.was_executed} | "
            f"Verified: {self.verification_passed} | "
            f"Status: {self.status.value}"
        )

    def is_learning_ready(self) -> bool:
        """Check if this outcome has enough data for learning."""
        has_prediction = (
            self.prediction.resolution_type is not None
        )
        has_outcome = (
            self.actual_outcome.actual_resolution is not None
            or self.actual_outcome.was_executed
        )
        has_feedback = self.human_feedback_id is not None
        return has_prediction and has_outcome

    def prediction_matches_actual(self) -> Optional[bool]:
        """Check if prediction matched the actual resolution.

        Returns None if comparison is not possible.
        """
        if (
            self.prediction.resolution_type is not None
            and self.actual_outcome.actual_resolution is not None
        ):
            return self.prediction.resolution_type == self.actual_outcome.actual_resolution
        return None
