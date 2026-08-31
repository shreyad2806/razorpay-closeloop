"""
Financial State Diff schema for Razorpay CloseLoop Phase 8C.

Compares before/after financial state snapshots to determine:
- intended changes (from the authorized resolution)
- actual changes (what really happened)
- missing changes (intended but not applied)
- unintended changes (unexpected modifications)

All financial values remain integer paise.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Diff Enums
# ─────────────────────────────────────────────────────────────────────────────


class ChangeType(str, Enum):
    """Type of change detected."""
    INTENDED = "INTENDED"
    UNINTENDED = "UNINTENDED"
    MISSING = "MISSING"
    NO_CHANGE = "NO_CHANGE"


class FieldChange(BaseModel):
    """A single field change between before and after states."""
    field_name: str = Field(..., description="Name of the field")
    before_value: int = Field(..., description="Value before execution (paise)")
    after_value: int = Field(..., description="Value after execution (paise)")
    delta: int = Field(..., description="after - before (paise)")
    change_type: ChangeType = Field(..., description="Whether change was intended")


class RecordChange(BaseModel):
    """A change to record counts."""
    record_type: str = Field(..., description="Type of record (fee, refund, etc.)")
    before_count: int = Field(..., description="Count before")
    after_count: int = Field(..., description="Count after")
    delta: int = Field(..., description="after - before")
    change_type: ChangeType = Field(..., description="Whether change was intended")


# ─────────────────────────────────────────────────────────────────────────────
# Financial State Diff
# ─────────────────────────────────────────────────────────────────────────────


class FinancialStateDiff(BaseModel):
    """Complete comparison of before/after financial state.

    Answers:
    - What was the financial state before execution?
    - What is the state after execution?
    - What exactly changed?
    - Were changes intended?
    - Were there unintended changes?
    """
    exception_id: str = Field(..., description="Exception identifier")
    execution_id: Optional[str] = Field(default=None, description="Execution identifier")

    # Field-level changes
    field_changes: List[FieldChange] = Field(
        default_factory=list, description="Changes to financial amount fields"
    )

    # Record count changes
    record_changes: List[RecordChange] = Field(
        default_factory=list, description="Changes to record counts"
    )

    # Classification
    intended_changes: List[FieldChange] = Field(
        default_factory=list, description="Changes that match the authorized resolution"
    )
    unintended_changes: List[FieldChange] = Field(
        default_factory=list, description="Unexpected changes not in the resolution"
    )
    missing_changes: List[FieldChange] = Field(
        default_factory=list, description="Intended changes that were not applied"
    )

    # Record-level unintended changes
    unintended_record_changes: List[RecordChange] = Field(
        default_factory=list, description="Unexpected record count changes"
    )

    # Summary
    total_intended_paise: int = Field(default=0, description="Sum of intended deltas")
    total_unintended_paise: int = Field(default=0, description="Sum of unintended deltas")
    has_unintended_changes: bool = Field(default=False, description="Any unintended changes detected")
    has_missing_changes: bool = Field(default=False, description="Any intended changes not applied")

    # Integrity
    all_integer_paise: bool = Field(default=True, description="All values are integer paise")

    # Metadata
    compared_at: datetime = Field(default_factory=datetime.utcnow)
    resolution_type: Optional[str] = Field(default=None, description="Resolution type applied")
    requested_adjustment_paise: int = Field(default=0, description="Requested adjustment")

    def summary(self) -> str:
        return (
            f"Diff: {len(self.field_changes)} fields changed | "
            f"Intended: {len(self.intended_changes)} | "
            f"Unintended: {len(self.unintended_changes)} | "
            f"Missing: {len(self.missing_changes)} | "
            f"Unintended paise: {self.total_unintended_paise}"
        )
