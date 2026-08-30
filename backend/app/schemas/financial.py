"""
Financial entity data contracts for the Razorpay CloseLoop synthetic dataset.

All monetary amounts are represented as integers in minor units (paise for INR).
This avoids floating-point rounding issues throughout the pipeline.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.enums import (
    AdjustmentType,
    Currency,
    FeeType,
    PaymentStatus,
    RefundStatus,
    SettlementStatus,
    TaxType,
)


class Merchant(BaseModel):
    """Merchant entity representing a payment gateway merchant."""

    merchant_id: str = Field(..., description="Unique merchant identifier, e.g. MER-001")
    name: str = Field(..., description="Merchant display name")
    reference: str = Field(..., description="Internal reference code")
    currency: Currency = Field(default=Currency.INR, description="Primary currency")
    created_at: datetime = Field(..., description="Merchant creation timestamp")


class Payment(BaseModel):
    """Payment entity representing a captured payment transaction."""

    payment_id: str = Field(..., description="Unique payment identifier, e.g. PAY-001")
    merchant_id: str = Field(..., description="Reference to Merchant.merchant_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Payment amount in minor units (paise)")
    currency: Currency = Field(default=Currency.INR)
    status: PaymentStatus = Field(default=PaymentStatus.CAPTURED)
    payment_timestamp: datetime = Field(..., description="When the payment was captured")


class Settlement(BaseModel):
    """Settlement entity representing a payout to a merchant."""

    settlement_id: str = Field(..., description="Unique settlement identifier, e.g. SET-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    merchant_id: str = Field(..., description="Reference to Merchant.merchant_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Settlement amount in minor units (paise)")
    currency: Currency = Field(default=Currency.INR)
    status: SettlementStatus = Field(default=SettlementStatus.SETTLED)
    settlement_timestamp: datetime = Field(..., description="When the settlement was processed")


class Refund(BaseModel):
    """Refund entity representing a refund against a payment."""

    refund_id: str = Field(..., description="Unique refund identifier, e.g. REF-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Refund amount in minor units (paise)")
    status: RefundStatus = Field(default=RefundStatus.PROCESSED)
    refund_timestamp: datetime = Field(..., description="When the refund was processed")


class Fee(BaseModel):
    """Fee entity representing a charge applied to a payment."""

    fee_id: str = Field(..., description="Unique fee identifier, e.g. FEE-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Fee amount in minor units (paise)")
    fee_type: FeeType = Field(..., description="Category of fee")


class Tax(BaseModel):
    """Tax entity representing tax applied to a payment or fee."""

    tax_id: str = Field(..., description="Unique tax identifier, e.g. TAX-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Tax amount in minor units (paise)")
    tax_type: TaxType = Field(..., description="Category of tax")


class Adjustment(BaseModel):
    """Adjustment entity representing a financial correction or modification."""

    adjustment_id: str = Field(..., description="Unique adjustment identifier, e.g. ADJ-001")
    payment_id: str = Field(..., description="Reference to Payment.payment_id")
    case_id: Optional[str] = Field(None, description="Reference to Case.case_id if applicable")
    amount: int = Field(..., description="Adjustment amount in minor units (paise); positive=credit, negative=debit")
    adjustment_type: AdjustmentType = Field(..., description="Category of adjustment")
