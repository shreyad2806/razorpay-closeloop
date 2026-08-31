"""
Audit Event schema for Razorpay CloseLoop Phase 8F.

Immutable-style audit records for major workflow events.
Every automated resolution must be explainable after the fact.

Audit history must not be silently overwritten.
If a correction is necessary, create another audit event.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Audit Enums
# ─────────────────────────────────────────────────────────────────────────────


class AuditEventType(str, Enum):
    """Types of audit events."""
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    EVIDENCE_GATHERED = "EVIDENCE_GATHERED"
    CLASSIFICATION_COMPLETE = "CLASSIFICATION_COMPLETE"
    CANDIDATES_GENERATED = "CANDIDATES_GENERATED"
    GUARDRAIL_EVALUATED = "GUARDRAIL_EVALUATED"
    HUMAN_REVIEW_REQUESTED = "HUMAN_REVIEW_REQUESTED"
    HUMAN_DECISION_RECEIVED = "HUMAN_DECISION_RECEIVED"
    VERIFICATION_PERFORMED = "VERIFICATION_PERFORMED"
    EXECUTION_REQUESTED = "EXECUTION_REQUESTED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    ROLLBACK_INITIATED = "ROLLBACK_INITIATED"
    ROLLBACK_COMPLETED = "ROLLBACK_COMPLETED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    RESOLUTION_VERIFIED = "RESOLUTION_VERIFIED"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"
    CASE_ESCALATED = "CASE_ESCALATED"
    CASE_UNRESOLVED = "CASE_UNRESOLVED"
    OUTCOME_RECORDED = "OUTCOME_RECORDED"
    REWARD_CALCULATED = "REWARD_CALCULATED"
    CORRECTION_APPLIED = "CORRECTION_APPLIED"


class ActorType(str, Enum):
    """Who or what performed the action."""
    SYSTEM = "SYSTEM"
    AGENT = "AGENT"
    HUMAN = "HUMAN"
    SERVICE = "SERVICE"


class FinalOutcome(str, Enum):
    """Final outcome of the resolution attempt."""
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    ESCALATED = "ESCALATED"
    UNRESOLVED = "UNRESOLVED"
    PENDING = "PENDING"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Sub-schemas
# ─────────────────────────────────────────────────────────────────────────────


class ModelMetadata(BaseModel):
    """Model information where applicable."""
    model_name: Optional[str] = Field(default=None, description="Model name")
    model_version: Optional[str] = Field(default=None, description="Model version")
    classifier_version: Optional[str] = Field(default=None, description="Classifier version")
    embedding_model_version: Optional[str] = Field(default=None, description="Embedding model version")
    retrieval_config: Optional[Dict[str, Any]] = Field(default=None, description="Retrieval configuration")


class GuardrailMetadata(BaseModel):
    """Guardrail evaluation metadata."""
    decision: Optional[str] = Field(default=None, description="Guardrail decision")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    exposure_paise: Optional[int] = Field(default=None, description="Financial exposure in paise")
    passed_checks: List[str] = Field(default_factory=list, description="Checks that passed")
    failed_checks: List[str] = Field(default_factory=list, description="Checks that failed")
    reason_codes: List[str] = Field(default_factory=list, description="Reason codes")
    policy_version: Optional[str] = Field(default=None, description="Policy version")


class ActionMetadata(BaseModel):
    """Execution action metadata."""
    resolution_type: Optional[str] = Field(default=None, description="Resolution type")
    requested_adjustment_paise: Optional[int] = Field(default=None, description="Requested adjustment")
    actual_adjustment_paise: Optional[int] = Field(default=None, description="Actual adjustment")
    execution_status: Optional[str] = Field(default=None, description="Execution status")
    idempotency_key: Optional[str] = Field(default=None, description="Idempotency key")
    execution_id: Optional[str] = Field(default=None, description="Execution ID")


class VerificationMetadata(BaseModel):
    """Verification result metadata."""
    expected_result: Optional[Dict[str, Any]] = Field(default=None, description="Expected result")
    actual_result: Optional[Dict[str, Any]] = Field(default=None, description="Actual result")
    difference_before: Optional[int] = Field(default=None, description="Difference before")
    difference_after: Optional[int] = Field(default=None, description="Difference after")
    discrepancy_eliminated: Optional[bool] = Field(default=None, description="Discrepancy eliminated")
    unintended_changes: Optional[int] = Field(default=None, description="Unintended change count")
    verification_status: Optional[str] = Field(default=None, description="Verification status")
    verification_failure_reason: Optional[str] = Field(default=None, description="Failure reason")


class RollbackMetadata(BaseModel):
    """Rollback metadata."""
    rollback_id: Optional[str] = Field(default=None, description="Rollback ID")
    rollback_status: Optional[str] = Field(default=None, description="Rollback status")
    reversal_amount_paise: Optional[int] = Field(default=None, description="Amount reversed")
    rollback_verified: Optional[bool] = Field(default=None, description="Rollback verified")
    rollback_reason: Optional[str] = Field(default=None, description="Why rollback occurred")


# ─────────────────────────────────────────────────────────────────────────────
# Audit Event
# ─────────────────────────────────────────────────────────────────────────────


class AuditEvent(BaseModel):
    """Immutable-style audit record for a major workflow event.

    Once created, an audit event must not be modified.
    If correction is needed, create a new CORRECTION_APPLIED event.
    """
    # Identity
    event_id: str = Field(..., description="Unique event ID")
    event_type: AuditEventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When event occurred")

    # Context
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(default=None, description="Candidate identifier")

    # Actor
    actor: str = Field(default="system", description="Who performed the action")
    actor_type: ActorType = Field(default=ActorType.SYSTEM, description="Type of actor")

    # Decision
    decision: Optional[str] = Field(default=None, description="Decision made")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    risk: Optional[str] = Field(default=None, description="Risk category")

    # Metadata
    model_metadata: Optional[ModelMetadata] = Field(default=None, description="Model information")
    guardrail_metadata: Optional[GuardrailMetadata] = Field(default=None, description="Guardrail info")
    action_metadata: Optional[ActionMetadata] = Field(default=None, description="Action info")
    verification_metadata: Optional[VerificationMetadata] = Field(default=None, description="Verification info")
    rollback_metadata: Optional[RollbackMetadata] = Field(default=None, description="Rollback info")

    # Evidence
    evidence_references: List[str] = Field(default_factory=list, description="Evidence record IDs")

    # State references
    before_state_reference: Optional[str] = Field(default=None, description="Before state snapshot ID")
    after_state_reference: Optional[str] = Field(default=None, description="After state snapshot ID")

    # Outcome
    final_outcome: Optional[FinalOutcome] = Field(default=None, description="Final outcome")

    # Error
    error: Optional[str] = Field(default=None, description="Error information")

    # Correction (for immutability)
    correction_of: Optional[str] = Field(default=None, description="Event ID this corrects")
    correction_reason: Optional[str] = Field(default=None, description="Why correction was needed")

    def summary(self) -> str:
        return (
            f"Audit: {self.event_type.value} | "
            f"Workflow: {self.workflow_id} | "
            f"Exception: {self.exception_id} | "
            f"Actor: {self.actor_type.value} | "
            f"Outcome: {self.final_outcome.value if self.final_outcome else 'N/A'}"
        )
