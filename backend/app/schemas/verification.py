"""
Verification schema for Razorpay CloseLoop Phase 7H.

Defines verification checks, results, and staleness detection
for post-resolution validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Verification Enums
# ─────────────────────────────────────────────────────────────────────────────


class VerificationAction(str, Enum):
    """Action after verification."""
    VERIFIED = "VERIFIED"
    STALE_STATE = "STALE_STATE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    RECONCILE_AGAIN = "RECONCILE_AGAIN"


class CheckStatus(str, Enum):
    """Status of individual verification checks."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ─────────────────────────────────────────────────────────────────────────────
# Individual Check
# ─────────────────────────────────────────────────────────────────────────────


class VerificationCheck(BaseModel):
    """A single verification check."""
    check_name: str = Field(..., description="Name of the check")
    status: CheckStatus = Field(..., description="Check result")
    expected: Optional[Any] = Field(default=None, description="Expected value")
    actual: Optional[Any] = Field(default=None, description="Actual value")
    message: Optional[str] = Field(default=None, description="Human-readable message")


# ─────────────────────────────────────────────────────────────────────────────
# Verification Result
# ─────────────────────────────────────────────────────────────────────────────


class VerificationResult(BaseModel):
    """Complete verification result."""
    exception_id: str = Field(..., description="Exception being verified")
    candidate_id: Optional[str] = Field(default=None, description="Candidate being verified")

    # Overall outcome
    action: VerificationAction = Field(..., description="Verification action")
    passed: bool = Field(..., description="Whether all checks passed")

    # Individual checks
    checks: List[VerificationCheck] = Field(
        default_factory=list, description="All verification checks"
    )

    # Staleness details
    stale_checks: List[str] = Field(
        default_factory=list, description="Names of checks that detected staleness"
    )
    changed_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Records that changed since recommendation"
    )

    # Financial consistency
    expected_amount_at_recommendation: Optional[int] = Field(
        default=None, description="Expected amount when recommendation was made"
    )
    expected_amount_now: Optional[int] = Field(
        default=None, description="Current expected amount"
    )
    amount_consistent: bool = Field(
        default=True, description="Whether financial amounts are still consistent"
    )

    # Evidence consistency
    evidence_exists: bool = Field(
        default=True, description="Whether evidence records still exist"
    )
    candidate_exists: bool = Field(
        default=True, description="Whether the candidate still exists"
    )

    # Metadata
    verified_at: datetime = Field(
        default_factory=datetime.utcnow, description="When verification was performed"
    )
    verified_by: str = Field(
        default="system", description="What performed verification"
    )
    elapsed_ms: Optional[float] = Field(
        default=None, description="Verification time in milliseconds"
    )

    def summary(self) -> str:
        """Human-readable summary."""
        check_passed = sum(1 for c in self.checks if c.status == CheckStatus.PASSED)
        check_failed = sum(1 for c in self.checks if c.status == CheckStatus.FAILED)
        return (
            f"Verification: {self.action.value} | "
            f"Checks: {check_passed} passed, {check_failed} failed | "
            f"Amount consistent: {self.amount_consistent} | "
            f"Evidence exists: {self.evidence_exists}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification Config
# ─────────────────────────────────────────────────────────────────────────────


class VerificationConfig(BaseModel):
    """Configuration for verification checks."""
    check_exception_exists: bool = Field(
        default=True, description="Verify exception still exists"
    )
    check_candidate_exists: bool = Field(
        default=True, description="Verify candidate still exists"
    )
    check_evidence_exists: bool = Field(
        default=True, description="Verify evidence records still exist"
    )
    check_financial_consistency: bool = Field(
        default=True, description="Verify financial amounts unchanged"
    )
    check_guardrail_valid: bool = Field(
        default=True, description="Verify guardrail decision still valid"
    )
    check_no_conflicting_update: bool = Field(
        default=True, description="No conflicting update occurred"
    )
    require_all_passed: bool = Field(
        default=True, description="All checks must pass for VERIFIED"
    )
