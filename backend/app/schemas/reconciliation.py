"""
Reconciliation result data contract for Razorpay CloseLoop.

Defines the deterministic output of the reconciliation engine.
No AI, no ML, no LLM — pure deterministic calculation.

The reconciliation engine:
1. Reads financial input records (payments, settlements, refunds, fees, taxes, adjustments)
2. Independently calculates expected settlement amount
3. Compares with actual settlement amount
4. Produces a ReconciliationResult with match status and exception type

Ground truth is NOT read by the reconciliation engine.
Ground truth is used only for evaluation after reconciliation completes.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.enums import (
    ExceptionType,
    MatchStatus,
    ReconciliationStatus,
    ResolutionType,
    RiskCategory,
)


class ReconciliationResult(BaseModel):
    """
    Deterministic reconciliation result for a single payment/case.

    This is the output of the reconciliation engine.
    It contains the engine's independent calculation and classification.

    Ground truth labels (true_exception_type, true_resolution) are NOT included here.
    They are stored separately for evaluation.
    """

    reconciliation_id: str = Field(
        ..., description="Unique reconciliation result identifier, e.g. REC-000001"
    )
    case_id: str = Field(..., description="Reference to Case.case_id")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    merchant_id: str = Field(..., description="Reference to Merchant.merchant_id")

    # Financial calculation (independent of ground truth)
    payment_amount: int = Field(..., description="Original payment amount in paise")
    total_refunds: int = Field(
        default=0, description="Sum of refund amounts in paise"
    )
    total_fees: int = Field(default=0, description="Sum of fee amounts in paise")
    total_taxes: int = Field(default=0, description="Sum of tax amounts in paise")
    total_adjustments: int = Field(
        default=0, description="Net adjustments in paise (positive=credit, negative=debit)"
    )

    expected_amount: int = Field(
        ..., description="Engine-calculated expected settlement in paise"
    )
    actual_amount: int = Field(
        ..., description="Observed actual settlement in paise"
    )
    difference: int = Field(
        ..., description="expected_amount - actual_amount in paise"
    )

    # Matching and classification
    match_status: MatchStatus = Field(
        ..., description="Deterministic match classification"
    )
    exception_type: ExceptionType = Field(
        ..., description="Engine-determined exception type"
    )

    # Processing metadata
    reconciliation_status: ReconciliationStatus = Field(
        default=ReconciliationStatus.PROCESSED,
        description="Processing status",
    )
    reconciliation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When reconciliation was performed",
    )
    processing_notes: Optional[str] = Field(
        default=None, description="Deterministic processing notes"
    )

    def compute_expected_amount(self) -> int:
        """
        Compute expected settlement amount using integer arithmetic.

        Formula: payment - refunds - fees - taxes + adjustments

        This is the same formula used in Phase 1 ground truth.
        The reconciliation engine must independently compute this.
        """
        return (
            self.payment_amount
            - self.total_refunds
            - self.total_fees
            - self.total_taxes
            + self.total_adjustments
        )

    def verify_calculation(self) -> bool:
        """Verify that expected_amount matches the financial breakdown."""
        return self.compute_expected_amount() == self.expected_amount

    def compute_difference(self) -> int:
        """Compute difference as expected - actual."""
        return self.expected_amount - self.actual_amount


# ─────────────────────────────────────────────────────────────────────────────
# Calculation Contract
# ─────────────────────────────────────────────────────────────────────────────


class CalculationBreakdown(BaseModel):
    """
    Detailed breakdown of the expected settlement calculation.

    Provides transparency into how the expected amount was computed.
    Used for debugging and evidence collection.
    """

    payment_amount: int = Field(..., description="Base payment amount in paise")
    refund_deduction: int = Field(
        default=0, description="Total refunds deducted in paise"
    )
    fee_deduction: int = Field(
        default=0, description="Total fees deducted in paise"
    )
    tax_deduction: int = Field(
        default=0, description="Total taxes deducted in paise"
    )
    adjustment_addition: int = Field(
        default=0, description="Net adjustments added in paise"
    )
    expected_amount: int = Field(
        ..., description="Final expected settlement in paise"
    )

    @classmethod
    def from_financial_records(
        cls,
        payment_amount: int,
        total_refunds: int,
        total_fees: int,
        total_taxes: int,
        total_adjustments: int,
    ) -> "CalculationBreakdown":
        """
        Create a breakdown from financial records.

        All values in paise (integer minor units).
        """
        expected = (
            payment_amount
            - total_refunds
            - total_fees
            - total_taxes
            + total_adjustments
        )
        return cls(
            payment_amount=payment_amount,
            refund_deduction=total_refunds,
            fee_deduction=total_fees,
            tax_deduction=total_taxes,
            adjustment_addition=total_adjustments,
            expected_amount=expected,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Matching Contract
# ─────────────────────────────────────────────────────────────────────────────


class MatchingRule(BaseModel):
    """
    Definition of a deterministic matching rule.

    Rules are applied in priority order:
    1. Exact ID match
    2. Valid payment relationship
    3. Case relationship
    4. Missing record detection
    5. Duplicate detection
    """

    rule_id: str = Field(..., description="Unique rule identifier")
    priority: int = Field(..., description="Rule priority (lower = higher priority)")
    description: str = Field(..., description="Human-readable rule description")
    is_deterministic: bool = Field(
        default=True, description="Whether rule is fully deterministic"
    )


# Default matching rules (applied in priority order)
DEFAULT_MATCHING_RULES = [
    MatchingRule(
        rule_id="EXACT_ID_MATCH",
        priority=1,
        description="Exact payment/settlement ID relationship exists",
        is_deterministic=True,
    ),
    MatchingRule(
        rule_id="VALID_PAYMENT_RELATIONSHIP",
        priority=2,
        description="Valid payment relationship with matching amounts",
        is_deterministic=True,
    ),
    MatchingRule(
        rule_id="CASE_RELATIONSHIP",
        priority=3,
        description="Case relationship exists with expected amounts",
        is_deterministic=True,
    ),
    MatchingRule(
        rule_id="MISSING_RECORD_DETECTION",
        priority=4,
        description="Expected record is missing from settlement data",
        is_deterministic=True,
    ),
    MatchingRule(
        rule_id="DUPLICATE_DETECTION",
        priority=5,
        description="Duplicate settlement detected for same payment",
        is_deterministic=True,
    ),
]
