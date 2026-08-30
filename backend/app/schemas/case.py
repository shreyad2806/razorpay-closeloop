"""
Case and Ground Truth data contracts for the Razorpay CloseLoop synthetic dataset.

A Case represents a financial situation that will be reconciled.
Ground Truth stores the definitive answer key, separate from generated financial records.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.enums import ExceptionType, ResolutionType, RiskCategory


class Case(BaseModel):
    """
    A reconciliation case representing a financial situation to be analyzed.

    Contains the observed discrepancy and enough context for the reconciliation
    engine to determine expected vs actual amounts.
    """

    case_id: str = Field(..., description="Unique case identifier, e.g. CASE-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    merchant_id: str = Field(..., description="Reference to Merchant.merchant_id")
    expected_amount: int = Field(
        ..., description="Expected settlement amount in minor units (paise)"
    )
    actual_amount: int = Field(
        ..., description="Actual settlement amount in minor units (paise)"
    )
    difference: int = Field(
        ..., description="actual_amount - expected_amount in minor units (paise)"
    )
    scenario: ExceptionType = Field(
        ..., description="Exception scenario type from centralized taxonomy"
    )
    observation_timestamp: datetime = Field(
        ..., description="When the discrepancy was observed"
    )
    resolvable: bool = Field(
        ..., description="Whether the case can be automatically resolved"
    )
    risk_category: RiskCategory = Field(
        ..., description="Risk level independent of exception type"
    )


class GroundTruth(BaseModel):
    """
    Definitive answer key for a reconciliation case.

    Stored separately from generated financial records and model outputs.
    Allows verification of the expected financial calculation:

        payment
        - refunds
        - fees
        - taxes
        +/- adjustments
        = expected settlement
    """

    case_id: str = Field(..., description="Reference to Case.case_id")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")

    # Financial breakdown (all in minor units)
    payment_amount: int = Field(..., description="Original payment amount in paise")
    total_refunds: int = Field(
        default=0, description="Sum of all refund amounts in paise"
    )
    total_fees: int = Field(default=0, description="Sum of all fee amounts in paise")
    total_taxes: int = Field(default=0, description="Sum of all tax amounts in paise")
    total_adjustments: int = Field(
        default=0,
        description="Net adjustments in paise (positive=credit, negative=debit)",
    )

    # Expected vs actual
    expected_amount: int = Field(
        ..., description="Computed expected settlement in paise"
    )
    actual_amount: int = Field(..., description="Observed actual settlement in paise")
    difference: int = Field(..., description="actual_amount - expected_amount in paise")

    # Truth labels
    true_exception_type: ExceptionType = Field(
        ..., description="The actual exception category"
    )
    true_resolution: ResolutionType = Field(
        ..., description="The correct resolution for this case"
    )
    resolvable: bool = Field(
        ..., description="Whether this case has a deterministic resolution"
    )
    risk_category: RiskCategory = Field(
        ..., description="Risk level assigned to this case"
    )

    def verify_expected_amount(self) -> bool:
        """
        Verify that expected_amount matches the financial breakdown.

        Formula: payment - refunds - fees - taxes + adjustments = expected
        """
        computed = (
            self.payment_amount
            - self.total_refunds
            - self.total_fees
            - self.total_taxes
            + self.total_adjustments
        )
        return computed == self.expected_amount
