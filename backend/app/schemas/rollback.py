"""
Rollback schema for Razorpay CloseLoop Phase 8E.

Defines rollback state machine, rollback verification,
and escalation behavior for verification failures.

If verification fails, the system must never silently report success.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Enums
# ─────────────────────────────────────────────────────────────────────────────


class RollbackStatus(str, Enum):
    """Rollback lifecycle states."""
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    ESCALATED = "ESCALATED"


class RollbackReason(str, Enum):
    """Why rollback was initiated."""
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNINTENDED_CHANGES = "UNINTENDED_CHANGES"
    DISCREPANCY_REMAINS = "DISCREPANCY_REMAINS"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Audit Entry
# ─────────────────────────────────────────────────────────────────────────────


class RollbackAuditEntry(BaseModel):
    """Audit entry for a rollback attempt."""
    entry_id: str = Field(..., description="Unique audit entry ID")
    execution_id: str = Field(..., description="Execution being rolled back")
    exception_id: str = Field(..., description="Exception identifier")
    action: str = Field(..., description="Rollback action performed")
    status: str = Field(..., description="Action status")
    expected_state: Optional[Dict[str, Any]] = Field(default=None)
    actual_state: Optional[Dict[str, Any]] = Field(default=None)
    match: Optional[bool] = Field(default=None, description="Whether states match")
    error: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Result
# ─────────────────────────────────────────────────────────────────────────────


class RollbackResult(BaseModel):
    """Complete result of a rollback operation."""
    rollback_id: str = Field(..., description="Unique rollback ID")
    execution_id: str = Field(..., description="Execution being rolled back")
    exception_id: str = Field(..., description="Exception identifier")

    # Status
    status: RollbackStatus = Field(..., description="Rollback status")
    reason: RollbackReason = Field(..., description="Why rollback was initiated")

    # State
    before_rollback_state: Optional[Dict[str, Any]] = Field(default=None)
    expected_rollback_state: Optional[Dict[str, Any]] = Field(default=None)
    after_rollback_state: Optional[Dict[str, Any]] = Field(default=None)

    # Verification
    rollback_verified: bool = Field(default=False, description="Whether rollback was verified")
    rollback_state_match: Optional[bool] = Field(default=None, description="States match after rollback")

    # Adjustment
    adjustment_reversed: bool = Field(default=False, description="Whether adjustment was reversed")
    reversal_amount_paise: int = Field(default=0, description="Amount reversed in paise")

    # Audit
    audit_trail: List[RollbackAuditEntry] = Field(
        default_factory=list, description="Rollback audit trail"
    )

    # Errors
    error: Optional[str] = Field(default=None, description="Rollback error if failed")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)

    def summary(self) -> str:
        return (
            f"Rollback: {self.rollback_id} | "
            f"Status: {self.status.value} | "
            f"Verified: {self.rollback_verified} | "
            f"Reversed: {self.reversal_amount_paise} paise"
        )
