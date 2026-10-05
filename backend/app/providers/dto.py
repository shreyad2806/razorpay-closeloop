"""
Provider DTOs — external-system record shapes.

Architecture section 14: the provider boundary must not leak ORM models in
either direction. These pydantic models describe what an *external financial
system* reports; they deliberately differ from ``app.models``:

* ids are the provider's native string references (``pay_*``, ``setl_*``);
* every record carries the reporting ``provider`` name and an integer
  ``version`` so ingestion can be idempotent and version-aware;
* timestamps are datetimes (providers report instants, not columns);
* money is integer minor units — the same invariant as the database, enforced
  here with ``StrictInt`` so a float can never masquerade as paise.

The ingestion layer (:mod:`app.ingestion`) is the only code allowed to convert
these DTOs into domain rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional, Dict

from pydantic import BeforeValidator, BaseModel, ConfigDict, Field, StrictInt, StrictStr

# Reuse the existing enum vocabulary — CloseLoop already has one source of
# truth for statuses and the drift tests guard it.
from app.schemas.enums import (
    ChargebackStatus,
    Currency,
    FeeType,
    PaymentAttemptKind,
    PaymentAttemptStatus,
    PaymentStatus,
    RefundStatus,
    SettlementComponentType,
    SettlementStatus,
    TaxType,
    ProviderExecutionStatus,
)


def _strict_datetime(value):
    """Reject string timestamps at the boundary.

    A provider that reports "2026-01-01T00:00:00Z" as a string has broken the
    record contract; parsing it here would silently guess its intent. Callers
    must hand over real datetime objects.
    """
    if not isinstance(value, datetime):
        raise ValueError(
            "timestamp must be a datetime object, not "
            f"{type(value).__name__} (malformed timestamps are rejected)"
        )
    return value


# Runs BEFORE pydantic's lenient string->datetime parsing.
DateTimeField = Annotated[datetime, BeforeValidator(_strict_datetime)]


class _ProviderRecord(BaseModel):
    """Common shape of every provider-reported record."""

    model_config = ConfigDict(extra="forbid")

    provider: StrictStr = Field(..., description="Name of the reporting provider")
    version: StrictInt = Field(
        ..., ge=1, description="Provider-side record version (1 = first report)"
    )


class ProviderMerchant(_ProviderRecord):
    """A merchant as the external system reports it."""

    merchant_id: StrictStr
    name: StrictStr
    status: StrictStr = "ACTIVE"
    category_code: Optional[StrictStr] = None
    onboarded_at: Optional[DateTimeField] = None


class ProviderPayment(_ProviderRecord):
    """A payment as the external system reports it."""

    payment_id: StrictStr
    merchant_id: StrictStr
    order_id: Optional[StrictStr] = None
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    status: PaymentStatus = PaymentStatus.CREATED
    method: Optional[StrictStr] = None
    captured_at: Optional[DateTimeField] = None
    provider_created_at: Optional[DateTimeField] = None


class ProviderPaymentAttempt(_ProviderRecord):
    """One authorisation/capture attempt against a payment."""

    attempt_id: StrictStr
    payment_id: StrictStr
    attempt_no: StrictInt = Field(..., ge=1)
    kind: PaymentAttemptKind
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    status: PaymentAttemptStatus = PaymentAttemptStatus.CREATED
    provider_attempt_id: Optional[StrictStr] = None
    occurred_at: Optional[DateTimeField] = None


class ProviderSettlement(_ProviderRecord):
    """A payout the external system settled to a merchant."""

    settlement_id: StrictStr
    payment_id: Optional[StrictStr] = None
    merchant_id: Optional[StrictStr] = None
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    status: SettlementStatus = SettlementStatus.PENDING
    settled_at: Optional[DateTimeField] = None
    provider_batch_id: Optional[StrictStr] = None


class ProviderSettlementLine(_ProviderRecord):
    """One signed component of a settlement."""

    line_id: StrictStr
    settlement_id: StrictStr
    component_type: SettlementComponentType
    amount: StrictInt = Field(
        ..., description="Signed minor units: credits positive, debits negative"
    )
    currency: Currency = Currency.INR
    source_entity_type: Optional[StrictStr] = None
    source_entity_id: Optional[StrictStr] = None
    description: Optional[StrictStr] = None


class ProviderRefund(_ProviderRecord):
    """A refund against a payment."""

    refund_id: StrictStr
    payment_id: StrictStr
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    status: RefundStatus = RefundStatus.PROCESSED
    reason: Optional[StrictStr] = None
    refund_timestamp: Optional[DateTimeField] = None


class ProviderChargeback(_ProviderRecord):
    """A dispute raised against a payment."""

    chargeback_id: StrictStr
    payment_id: StrictStr
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    status: ChargebackStatus = ChargebackStatus.OPEN
    reason_code: Optional[StrictStr] = None
    evidence_due_at: Optional[DateTimeField] = None
    resolved_at: Optional[DateTimeField] = None
    notes: Optional[StrictStr] = None


class ProviderFee(_ProviderRecord):
    """A fee charged against a payment."""

    fee_id: StrictStr
    payment_id: StrictStr
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    fee_type: FeeType
    processed_at: Optional[DateTimeField] = None


class ProviderTax(_ProviderRecord):
    """A tax applied to a payment."""

    tax_id: StrictStr
    payment_id: StrictStr
    amount: StrictInt = Field(..., ge=0, description="Minor units")
    currency: Currency = Currency.INR
    tax_type: TaxType
    jurisdiction: Optional[StrictStr] = None


__all__ = [
    "ProviderMerchant",
    "ProviderPayment",
    "ProviderPaymentAttempt",
    "ProviderSettlement",
    "ProviderSettlementLine",
    "ProviderRefund",
    "ProviderChargeback",
    "ProviderFee",
    "ProviderTax",
    "ProviderExecutionRequest",
    "ProviderExecutionResult",
]


class ProviderExecutionRequest(BaseModel):
    """Payload to request mutation across the provider boundary."""

    model_config = ConfigDict(extra="forbid")

    action_type: StrictStr = Field(..., description="Type of operation, e.g. ADJUSTMENT, FEE_CORRECTION")
    action_id: StrictStr = Field(..., description="Identifier for this action request")
    target_entity_id: StrictStr = Field(..., description="Provider reference ID to mutate (e.g. payment_id)")
    amount_paise: StrictInt = Field(..., ge=0, description="Amount to adjust in paise")
    currency: Currency = Currency.INR
    idempotency_key: StrictStr = Field(..., description="Strict idempotency token for provider deduplication")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Safe primitives for provider execution")


class ProviderExecutionResult(BaseModel):
    """Result payload returned by the provider boundary after a mutation attempt."""

    model_config = ConfigDict(extra="forbid")

    status: ProviderExecutionStatus = Field(..., description="Outcome of the mutation")
    provider_reference_id: Optional[StrictStr] = Field(default=None, description="Provider ID generated for this mutation")
    amount_paise: StrictInt = Field(default=0, description="Amount actually mutated in paise")
    currency: Currency = Currency.INR
    error_code: Optional[StrictStr] = Field(default=None, description="Explicit error code if failed")
    error_message: Optional[StrictStr] = Field(default=None, description="Explicit error message if failed")
    idempotency_key: StrictStr = Field(..., description="The idempotency key that was requested")
    executed_at: Optional[DateTimeField] = None
