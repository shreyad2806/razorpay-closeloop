"""
Provider record validation.

Validation happens BEFORE normalization and BEFORE any database work: a
malformed provider record must be rejected with a precise reason, never
silently coerced into nonsense. The rules implement the Phase 2 requirements:

* identity  — provider ids are non-empty strings;
* currency  — must be a currency CloseLoop knows (INR/USD, matching the
  database CHECK constraint, so a record accepted here can never violate it);
* amount    — integer minor units, never a float, never a string of digits;
* parents   — child records must reference a parent id that exists in the
  provider snapshot (the ingestion service supplies the snapshot's id sets);
* version   — a positive integer;
* timestamps — datetimes only; a string timestamp is malformed and rejected,
  not parsed "best effort".
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.models.types import CURRENCY_VALUES


class IngestionValidationError(ValueError):
    """A provider record failed validation; it must not be persisted."""

    def __init__(self, entity: str, record_id: Optional[str], reason: str) -> None:
        self.entity = entity
        self.record_id = record_id
        self.reason = reason
        super().__init__(f"{entity}[{record_id}]: {reason}")


def _require_id(entity: str, record_id: Optional[str]) -> None:
    if not isinstance(record_id, str) or not record_id.strip():
        raise IngestionValidationError(
            entity, record_id, f"{entity} id must be a non-empty string"
        )


def _require_currency(entity: str, record_id: str, currency) -> None:
    if not isinstance(currency, str) or currency.upper() not in CURRENCY_VALUES:
        raise IngestionValidationError(
            entity,
            record_id,
            f"currency {currency!r} is not a supported currency "
            f"(expected one of {CURRENCY_VALUES})",
        )


def _require_amount(
    entity: str, record_id: str, amount, signed: bool = False
) -> None:
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise IngestionValidationError(
            entity,
            record_id,
            f"amount must be an integer in minor units, got {amount!r}",
        )
    if not signed and amount < 0:
        raise IngestionValidationError(
            entity, record_id, f"amount must be non-negative, got {amount!r}"
        )


def _require_version(entity: str, record_id: str, version) -> None:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise IngestionValidationError(
            entity,
            record_id,
            f"version must be a positive integer, got {version!r}",
        )


def _require_timestamp(
    entity: str, record_id: str, field_name: str, value
) -> None:
    if value is None:
        return
    if not isinstance(value, datetime):
        raise IngestionValidationError(
            entity,
            record_id,
            f"{field_name} must be a datetime, got {type(value).__name__} "
            "(malformed timestamps are rejected, not parsed)",
        )


def _require_parent(
    entity: str,
    record_id: str,
    parent_field: str,
    parent_id,
    known_parents: Optional[set],
) -> None:
    if parent_id is None:
        # Settlements may legitimately have no payment linkage yet; any other
        # required parent is enforced by the DTO being required (pydantic).
        return
    _require_id(f"{entity}.{parent_field}", parent_id)
    if known_parents is not None and parent_id not in known_parents:
        raise IngestionValidationError(
            entity,
            record_id,
            f"parent {parent_field}={parent_id!r} is not present in the "
            "provider snapshot",
        )


def _check_common(
    entity: str,
    record_id: str,
    version,
) -> None:
    _require_id(entity, record_id)
    _require_version(entity, record_id, version)


def validate_merchant(dto, known_parents=None) -> None:
    _check_common("merchant", dto.merchant_id, dto.version)
    _require_currency("merchant", dto.merchant_id, getattr(dto, "currency", "INR"))


def validate_payment(dto, known_parents) -> None:
    _check_common("payment", dto.payment_id, dto.version)
    _require_parent("payment", dto.payment_id, "merchant_id", dto.merchant_id, known_parents)
    _require_currency("payment", dto.payment_id, dto.currency)
    _require_amount("payment", dto.payment_id, dto.amount)
    _require_timestamp("payment", dto.payment_id, "captured_at", dto.captured_at)
    _require_timestamp("payment", dto.payment_id, "provider_created_at", dto.provider_created_at)


def validate_payment_attempt(dto, known_parents) -> None:
    _check_common("payment_attempt", dto.attempt_id, dto.version)
    _require_parent(
        "payment_attempt", dto.attempt_id, "payment_id", dto.payment_id, known_parents
    )
    _require_currency("payment_attempt", dto.attempt_id, dto.currency)
    _require_amount("payment_attempt", dto.attempt_id, dto.amount)
    _require_timestamp("payment_attempt", dto.attempt_id, "occurred_at", dto.occurred_at)


def validate_settlement(dto, known_parents) -> None:
    _check_common("settlement", dto.settlement_id, dto.version)
    # A settlement may be unlinked (payment_id=None); when present it must
    # exist in the snapshot.
    if dto.payment_id is not None:
        _require_parent(
            "settlement", dto.settlement_id, "payment_id", dto.payment_id, known_parents
        )
    if dto.merchant_id is not None:
        _require_parent(
            "settlement", dto.settlement_id, "merchant_id", dto.merchant_id, known_parents
        )
    _require_currency("settlement", dto.settlement_id, dto.currency)
    _require_amount("settlement", dto.settlement_id, dto.amount)
    _require_timestamp("settlement", dto.settlement_id, "settled_at", dto.settled_at)


def validate_settlement_line(dto, known_parents) -> None:
    _check_common("settlement_line", dto.line_id, dto.version)
    _require_parent(
        "settlement_line", dto.line_id, "settlement_id", dto.settlement_id, known_parents
    )
    _require_currency("settlement_line", dto.line_id, dto.currency)
    # Settlement lines are signed (fees/refunds/chargebacks reduce payouts).
    _require_amount("settlement_line", dto.line_id, dto.amount, signed=True)


def validate_refund(dto, known_parents) -> None:
    _check_common("refund", dto.refund_id, dto.version)
    _require_parent("refund", dto.refund_id, "payment_id", dto.payment_id, known_parents)
    _require_currency("refund", dto.refund_id, dto.currency)
    _require_amount("refund", dto.refund_id, dto.amount)
    _require_timestamp("refund", dto.refund_id, "refund_timestamp", dto.refund_timestamp)


def validate_chargeback(dto, known_parents) -> None:
    _check_common("chargeback", dto.chargeback_id, dto.version)
    _require_parent(
        "chargeback", dto.chargeback_id, "payment_id", dto.payment_id, known_parents
    )
    _require_currency("chargeback", dto.chargeback_id, dto.currency)
    _require_amount("chargeback", dto.chargeback_id, dto.amount)
    _require_timestamp("chargeback", dto.chargeback_id, "evidence_due_at", dto.evidence_due_at)
    _require_timestamp("chargeback", dto.chargeback_id, "resolved_at", dto.resolved_at)


def validate_fee(dto, known_parents) -> None:
    _check_common("fee", dto.fee_id, dto.version)
    _require_parent("fee", dto.fee_id, "payment_id", dto.payment_id, known_parents)
    _require_currency("fee", dto.fee_id, dto.currency)
    _require_amount("fee", dto.fee_id, dto.amount)
    _require_timestamp("fee", dto.fee_id, "processed_at", dto.processed_at)


def validate_tax(dto, known_parents) -> None:
    _check_common("tax", dto.tax_id, dto.version)
    _require_parent("tax", dto.tax_id, "payment_id", dto.payment_id, known_parents)
    _require_currency("tax", dto.tax_id, dto.currency)
    _require_amount("tax", dto.tax_id, dto.amount)
