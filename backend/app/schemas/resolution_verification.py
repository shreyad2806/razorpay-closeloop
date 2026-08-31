"""
Resolution Verification Engine schema for Razorpay CloseLoop Phase 8D.

Independent verification that a resolution actually achieved its goal.

Verification uses:
- actual database state
- deterministic financial calculations
- authorized resolution request

Ground truth is NOT used for verification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Verification Enums
# ─────────────────────────────────────────────────────────────────────────────


class VerificationStatus(str, Enum):
    """Verification outcome."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    CALCULATION_ERROR = "CALCULATION_ERROR"


class VerificationCheckType(str, Enum):
    """Individual verification check."""
    DISCREPANCY_ELIMINATED = "DISCREPANCY_ELIMINATED"
    CORRECT_ADJUSTMENT = "CORRECT_ADJUSTMENT"
    NO_UNINTENDED_CHANGES = "NO_UNINTENDED_CHANGES"
    AFFECTED_RECORDS_CORRECT = "AFFECTED_RECORDS_CORRECT"
    FINANCIAL_INTEGRITY = "FINANCIAL_INTEGRITY"
    AMOUNT_CONSISTENCY = "AMOUNT_CONSISTENCY"


class CheckResult(str, Enum):
    """Result of an individual check."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
# Verification Check
# ─────────────────────────────────────────────────────────────────────────────


class VerificationCheck(BaseModel):
    """A single verification check."""
    check_type: VerificationCheckType = Field(...)
    result: CheckResult = Field(...)
    expected: Optional[Any] = Field(default=None)
    actual: Optional[Any] = Field(default=None)
    message: Optional[str] = Field(default=None)
    details: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Expected Result (recalculated)
# ─────────────────────────────────────────────────────────────────────────────


class ExpectedFinancialResult(BaseModel):
    """Expected financial outcome recalculated from deterministic rules.

    NOT from ground truth.
    NOT from execution response.
    Recalculated independently.
    """
    expected_adjustment_paise: int = Field(default=0, description="Expected adjustment")
    expected_new_actual: int = Field(default=0, description="Expected new actual amount")
    expected_new_difference: int = Field(default=0, description="Expected new difference")
    expected_new_total_adjustments: int = Field(default=0, description="Expected new total adjustments")
    resolution_type: str = Field(default="", description="Resolution type used")


# ─────────────────────────────────────────────────────────────────────────────
# Actual Result (read from state)
# ─────────────────────────────────────────────────────────────────────────────


class ActualFinancialResult(BaseModel):
    """Actual financial state after execution.

    Read from the post-execution financial state.
    """
    actual_adjustment_paise: int = Field(default=0, description="Actual adjustment applied")
    actual_new_actual: int = Field(default=0, description="Actual new actual amount")
    actual_new_difference: int = Field(default=0, description="Actual new difference")
    actual_new_total_adjustments: int = Field(default=0, description="Actual new total adjustments")


# ─────────────────────────────────────────────────────────────────────────────
# Verification Result
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionVerificationResult(BaseModel):
    """Complete verification result.

    Answers:
    - Was the resolution successful?
    - What was expected vs what happened?
    - Were there unintended changes?
    """
    # Identity
    verification_id: str = Field(..., description="Unique verification ID")
    execution_id: str = Field(..., description="Execution being verified")
    exception_id: str = Field(..., description="Exception identifier")

    # Status
    status: VerificationStatus = Field(..., description="Overall verification status")

    # Expected vs Actual
    expected_result: ExpectedFinancialResult = Field(
        ..., description="Recalculated expected outcome"
    )
    actual_result: ActualFinancialResult = Field(
        ..., description="Read from post-execution state"
    )

    # Discrepancy tracking
    difference_before: int = Field(default=0, description="Original discrepancy (paise)")
    difference_after: int = Field(default=0, description="Remaining discrepancy after execution (paise)")
    discrepancy_eliminated: bool = Field(default=False, description="Whether original discrepancy is gone")

    # Change tracking
    has_unintended_changes: bool = Field(default=False, description="Any unintended changes detected")
    unintended_change_count: int = Field(default=0, description="Number of unintended changes")

    # Checks
    checks: List[VerificationCheck] = Field(
        default_factory=list, description="Individual verification checks"
    )
    passed_checks: int = Field(default=0, description="Number of checks passed")
    failed_checks: int = Field(default=0, description="Number of checks failed")

    # Errors
    verification_errors: List[str] = Field(
        default_factory=list, description="Errors during verification"
    )

    # Metadata
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    verified_by: str = Field(default="verification_engine", description="What performed verification")

    def summary(self) -> str:
        return (
            f"Verification: {self.status.value} | "
            f"Discrepancy: {self.difference_before} → {self.difference_after} | "
            f"Checks: {self.passed_checks} passed, {self.failed_checks} failed | "
            f"Unintended: {self.unintended_change_count}"
        )
