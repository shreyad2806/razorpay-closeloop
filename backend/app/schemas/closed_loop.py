"""
Closed-loop verification schemas for Razorpay CloseLoop Phase 10.

The single rule these types exist to express:

    PROVIDER SUCCESS != FINANCIAL VERIFICATION.

An exception may be closed only after post-execution deterministic
reconciliation proves the discrepancy is resolved. Nothing produced by an
LLM, an ML model, a similarity lookup or an agent can reach ``VERIFIED`` —
only the existing reconciliation engine's match status can.

Decision vocabulary deliberately reuses the states already established by the
pre-2.0 workflow (``app.schemas.outcome.WorkflowOutcome``) rather than
inventing a parallel one.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class VerificationDecision(str, Enum):
    """Outcome of post-execution closed-loop verification.

    * ``VERIFIED``      - reconciliation MATCHED; the discrepancy is gone.
    * ``FAILED``        - provider executed but the financial state did not
      reconcile (or reconciliation itself failed).
    * ``HUMAN_REVIEW``  - verification is inconclusive; a human must decide.
    * ``UNRESOLVED``    - the post-execution financial state could not be
      established at all (fail-closed).
    * ``ESCALATED``     - the exception was legally escalated.
    """

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"


class ClosedLoopVerificationResult(BaseModel):
    """Structured, auditable outcome of a Phase 10 verification.

    Every financial number in here comes from the existing deterministic
    reconciliation engine applied to *fresh* post-execution provider state.
    ``execution_result.after_state`` is never used as financial truth.
    """

    # Identity
    verification_id: str = Field(..., description="Unique verification identifier")
    execution_id: str = Field(..., description="Phase 9 execution identifier")
    action_id: str = Field(default="", description="Resolution action identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")

    # Reconciliation context
    run_id: Optional[str] = Field(default=None, description="ReconciliationRun id")
    reconciliation_id: str = Field(..., description="ReconciliationResult id")

    # Verification
    decision: VerificationDecision = Field(..., description="Verification decision")
    financially_verified: bool = Field(
        default=False, description="True only when reconciliation MATCHED"
    )
    execution_status: str = Field(
        default="", description="Phase 9 execution status after verification"
    )
    fresh_state_used: bool = Field(
        default=False, description="True when fresh post-execution state was read"
    )

    # Deterministic financial truth (integer paise)
    expected_amount: int = Field(default=0, description="Engine expected settlement")
    actual_amount: int = Field(default=0, description="Engine observed settlement")
    difference: int = Field(default=0, description="expected - actual")
    match_status: Optional[str] = Field(default=None, description="MatchStatus value")
    exception_type: Optional[str] = Field(
        default=None, description="Engine-determined exception type"
    )

    # Closure
    exception_closed: bool = Field(
        default=False, description="True when this call closed the exception"
    )
    exception_already_closed: bool = Field(
        default=False, description="Exception was already CLOSED before this call"
    )
    resolution_verified: Optional[bool] = Field(
        default=None, description="Resolution moved to VERIFIED (when one was given)"
    )
    action_verified: Optional[bool] = Field(
        default=None, description="ResolutionAction moved to VERIFIED"
    )
    exception_escalated: bool = Field(
        default=False, description="Exception legally escalated instead of closed"
    )

    # Persistence / audit
    reconciliation_persisted: bool = Field(
        default=False, description="Run + result were persisted"
    )
    audit_event_ids: List[str] = Field(
        default_factory=list, description="Audit event ids recorded"
    )

    # Failure detail
    failure_reason: Optional[str] = Field(default=None, description="Why not verified")

    verified_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        return (
            f"ClosedLoop {self.verification_id} | {self.decision.value} | "
            f"expected={self.expected_amount} actual={self.actual_amount} "
            f"diff={self.difference} | closed={self.exception_closed}"
        )


__all__ = ["VerificationDecision", "ClosedLoopVerificationResult"]
