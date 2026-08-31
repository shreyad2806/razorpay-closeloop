"""
Execution schemas for Razorpay CloseLoop Phase 8A.

Defines execution state machine, before/after state capture,
and execution results for resolution execution.

Execution and verification are separate stages:
- Execution success = "The requested action was performed"
- Verification success = "The financial state now matches expected outcome"

Only verification success produces FINAL_SUCCESS.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Execution Status
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionStatus(str, Enum):
    """Explicit execution states — do not collapse execution and verification."""
    NOT_EXECUTED = "NOT_EXECUTED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"
    VERIFIED = "VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    ESCALATED = "ESCALATED"


class ExecutionTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Financial State Snapshot
# ─────────────────────────────────────────────────────────────────────────────


class FinancialStateSnapshot(BaseModel):
    """Snapshot of financial state at a point in time.

    Captures enough information to determine:
    - what existed before
    - what the system intended to change
    - what could potentially be affected
    """
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")

    # Core financial amounts (integer paise)
    payment_amount: int = Field(default=0, description="Payment amount in paise")
    expected_amount: int = Field(default=0, description="Expected settlement in paise")
    actual_amount: int = Field(default=0, description="Actual settlement in paise")
    difference: int = Field(default=0, description="expected - actual in paise")

    # Component totals
    total_refunds: int = Field(default=0, description="Total refunds in paise")
    total_fees: int = Field(default=0, description="Total fees in paise")
    total_taxes: int = Field(default=0, description="Total taxes in paise")
    total_adjustments: int = Field(default=0, description="Net adjustments in paise")

    # Related records
    settlement_count: int = Field(default=0, description="Number of settlement records")
    refund_count: int = Field(default=0, description="Number of refund records")
    fee_count: int = Field(default=0, description="Number of fee records")
    tax_count: int = Field(default=0, description="Number of tax records")
    adjustment_count: int = Field(default=0, description="Number of adjustment records")

    # Metadata
    captured_at: datetime = Field(default_factory=datetime.utcnow)
    snapshot_reason: str = Field(default="pre_execution", description="Why snapshot was taken")


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Record
# ─────────────────────────────────────────────────────────────────────────────


class AdjustmentRecord(BaseModel):
    """Record of a financial adjustment applied during execution."""
    adjustment_id: str = Field(..., description="Unique adjustment identifier")
    adjustment_type: str = Field(..., description="Type of adjustment")
    amount_paise: int = Field(..., description="Adjustment amount in paise")
    requested_amount_paise: int = Field(..., description="Originally requested amount")
    affected_records: List[str] = Field(default_factory=list, description="IDs of affected records")
    status: str = Field(default="applied", description="Adjustment status")


# ─────────────────────────────────────────────────────────────────────────────
# Execution Result
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionResult(BaseModel):
    """Complete result of resolution execution.

    Stores everything needed for:
    - verification
    - audit
    - rollback decisions
    - historical learning
    """
    # Identity
    execution_id: str = Field(..., description="Unique execution identifier")
    action_id: str = Field(..., description="Source action request ID")
    idempotency_key: str = Field(..., description="Idempotency key")

    # Workflow context
    workflow_id: str = Field(..., description="Workflow identifier")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(default=None, description="Candidate identifier")

    # Resolution
    resolution_type: str = Field(..., description="Resolution type applied")
    authorization_source: str = Field(..., description="Authorization source")

    # State snapshots
    before_state: FinancialStateSnapshot = Field(
        ..., description="Financial state before execution"
    )
    after_state: Optional[FinancialStateSnapshot] = Field(
        default=None, description="Financial state after execution"
    )

    # Adjustment
    adjustment: Optional[AdjustmentRecord] = Field(
        default=None, description="Applied adjustment"
    )
    requested_adjustment_paise: int = Field(
        default=0, description="Requested adjustment in paise"
    )
    actual_adjustment_paise: int = Field(
        default=0, description="Actual adjustment applied in paise"
    )

    # Status
    status: ExecutionStatus = Field(
        default=ExecutionStatus.NOT_EXECUTED,
        description="Execution status",
    )

    # Decision metadata
    decision: Optional[str] = Field(default=None, description="Guardrail decision")
    confidence: Optional[float] = Field(default=None, description="Confidence at execution")
    risk: Optional[str] = Field(default=None, description="Risk at execution")
    guardrail_reason_codes: List[str] = Field(
        default_factory=list, description="Guardrail reason codes"
    )

    # Evidence
    evidence_references: List[str] = Field(
        default_factory=list, description="Evidence record IDs"
    )

    # Errors
    error: Optional[str] = Field(default=None, description="Execution error if failed")
    rollback_reason: Optional[str] = Field(default=None, description="Why rollback occurred")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = Field(default=None, description="When execution occurred")
    verified_at: Optional[datetime] = Field(default=None, description="When verification occurred")
    rolled_back_at: Optional[datetime] = Field(default=None, description="When rollback occurred")

    def summary(self) -> str:
        return (
            f"Execution: {self.execution_id} | "
            f"Status: {self.status.value} | "
            f"Requested: {self.requested_adjustment_paise} paise | "
            f"Actual: {self.actual_adjustment_paise} paise"
        )

    def transition_to(self, new_status: ExecutionStatus) -> None:
        """Validate and apply a state transition.

        Raises:
            ExecutionTransitionError: If transition is not allowed.
        """
        if not is_valid_transition(self.status, new_status):
            raise ExecutionTransitionError(
                f"Invalid transition: {self.status.value} → {new_status.value}"
            )
        self.status = new_status


# ─────────────────────────────────────────────────────────────────────────────
# State Machine: Valid Transitions
# ─────────────────────────────────────────────────────────────────────────────


# Centralized transition policy.
# Key = current status, Value = set of allowed next statuses.
VALID_TRANSITIONS: Dict[ExecutionStatus, Set[ExecutionStatus]] = {
    ExecutionStatus.NOT_EXECUTED: {
        ExecutionStatus.EXECUTING,
        ExecutionStatus.ESCALATED,
    },
    ExecutionStatus.EXECUTING: {
        ExecutionStatus.EXECUTED,
        ExecutionStatus.EXECUTION_FAILED,
    },
    ExecutionStatus.EXECUTED: {
        ExecutionStatus.VERIFICATION_PENDING,
        ExecutionStatus.ESCALATED,
    },
    ExecutionStatus.EXECUTION_FAILED: {
        ExecutionStatus.ESCALATED,
        ExecutionStatus.NOT_EXECUTED,  # retry
    },
    ExecutionStatus.VERIFICATION_PENDING: {
        ExecutionStatus.VERIFIED,
        ExecutionStatus.VERIFICATION_FAILED,
    },
    ExecutionStatus.VERIFIED: {
        # Terminal state — no further transitions
    },
    ExecutionStatus.VERIFICATION_FAILED: {
        ExecutionStatus.ROLLED_BACK,
        ExecutionStatus.ESCALATED,
        ExecutionStatus.EXECUTING,  # retry
    },
    ExecutionStatus.ROLLED_BACK: {
        ExecutionStatus.ESCALATED,
    },
    ExecutionStatus.ESCALATED: {
        # Terminal state — no further transitions
    },
}


def is_valid_transition(
    current: ExecutionStatus,
    target: ExecutionStatus,
) -> bool:
    """Check if a state transition is allowed."""
    allowed = VALID_TRANSITIONS.get(current, set())
    return target in allowed


def get_allowed_transitions(
    current: ExecutionStatus,
) -> Set[ExecutionStatus]:
    """Get all allowed transitions from a given status."""
    return VALID_TRANSITIONS.get(current, set()).copy()


def is_terminal(status: ExecutionStatus) -> bool:
    """Check if a status is terminal (no outgoing transitions)."""
    return len(VALID_TRANSITIONS.get(status, set())) == 0
