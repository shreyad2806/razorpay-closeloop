"""
DTO -> ORM normalization.

The ingestion layer is the only code allowed to convert provider DTOs into
domain rows. Normalization is explicit and boring:

* statuses arrive as pydantic enum members and are stored as their string
  values (the database columns are CHECK-constrained strings);
* currency arrives as an enum and is stored as its 3-letter value;
* timestamps are passed through unchanged — they were already validated as
  real datetimes, and the columns are timezone-aware;
* money is passed through as int — it was already validated as integer minor
  units.

No reconciliation logic, no exception detection, no state *decisions*: payment
status changes follow the provider payload through the domain state machine.
"""

from __future__ import annotations

from app.models.adjustment import Adjustment  # noqa: F401  (import symmetry)
from app.models.chargeback import Chargeback
from app.models.fee import Fee
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.payment_attempt import PaymentAttempt
from app.models.refund import Refund
from app.models.settlement import Settlement
from app.models.settlement_line import SettlementLine
from app.models.tax import Tax


def normalize_merchant(dto) -> dict:
    return {
        "id": dto.merchant_id,
        "name": dto.name,
        "status": dto.status,
        "category_code": dto.category_code,
        "onboarded_at": dto.onboarded_at,
    }


def normalize_payment(dto) -> dict:
    return {
        "id": dto.payment_id,
        "merchant_id": dto.merchant_id,
        "order_id": dto.order_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "status": dto.status.value,
        "method": dto.method,
        "captured_at": dto.captured_at,
        "provider_created_at": dto.provider_created_at,
        "version": dto.version,
    }


def normalize_payment_attempt(dto) -> dict:
    return {
        "id": dto.attempt_id,
        "payment_id": dto.payment_id,
        "attempt_no": dto.attempt_no,
        "kind": dto.kind.value,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "status": dto.status.value,
        "provider_attempt_id": dto.provider_attempt_id,
        "occurred_at": dto.occurred_at,
    }


def normalize_settlement(dto) -> dict:
    return {
        "id": dto.settlement_id,
        "payment_id": dto.payment_id,
        "merchant_id": dto.merchant_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "status": dto.status.value,
        "settled_at": dto.settled_at,
        "provider_batch_id": dto.provider_batch_id,
        "version": dto.version,
    }


def normalize_settlement_line(dto) -> dict:
    return {
        "id": dto.line_id,
        "settlement_id": dto.settlement_id,
        "component_type": dto.component_type.value,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "source_entity_type": dto.source_entity_type,
        "source_entity_id": dto.source_entity_id,
        "description": dto.description,
    }


def normalize_refund(dto) -> dict:
    return {
        "id": dto.refund_id,
        "payment_id": dto.payment_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "status": dto.status.value,
        "reason": dto.reason,
        "refund_timestamp": dto.refund_timestamp,
    }


def normalize_chargeback(dto) -> dict:
    return {
        "id": dto.chargeback_id,
        "payment_id": dto.payment_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "status": dto.status.value,
        "reason_code": dto.reason_code,
        "evidence_due_at": dto.evidence_due_at,
        "resolved_at": dto.resolved_at,
        "notes": dto.notes,
    }


def normalize_fee(dto) -> dict:
    return {
        "id": dto.fee_id,
        "payment_id": dto.payment_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "fee_type": dto.fee_type.value,
        "processed_at": dto.processed_at,
    }


def normalize_tax(dto) -> dict:
    return {
        "id": dto.tax_id,
        "payment_id": dto.payment_id,
        "amount": dto.amount,
        "currency": dto.currency.value,
        "tax_type": dto.tax_type.value,
        "jurisdiction": dto.jurisdiction,
    }
