"""
IngestionService — provider snapshot to normalized database state.

Pipeline (architecture section 14):

    provider snapshot -> validate -> normalize -> ordered upsert -> DB

Invariants:

* parent-before-child: merchants, then payments, then payment attempts and the
  payment-level financial events, then settlements and their lines. A child is
  never persisted before its parent exists.
* idempotent: re-ingesting the same snapshot changes nothing. Upserts are
  keyed on the provider-native primary key; a record already at the reported
  version is a NO-OP, not a duplicate.
* version-aware: same id + newer version updates; same id + same version
  no-ops; same id + older version is refused (newer state preserved).
* transactional: the whole snapshot is one transaction. Any unexpected failure
  rolls back everything — no half-created parents or children.
* no reconciliation, no exceptions, no ML: ingestion only records what the
  provider reported.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.domain.state_machines import InvalidStateTransitionError
from app.ingestion.normalizer import (
    normalize_chargeback,
    normalize_fee,
    normalize_merchant,
    normalize_payment,
    normalize_payment_attempt,
    normalize_refund,
    normalize_settlement,
    normalize_settlement_line,
    normalize_tax,
)
from app.ingestion.result import IngestionResult
from app.ingestion.validation import (
    IngestionValidationError,
    validate_chargeback,
    validate_fee,
    validate_merchant,
    validate_payment,
    validate_payment_attempt,
    validate_refund,
    validate_settlement,
    validate_settlement_line,
    validate_tax,
)
from app.models.chargeback import Chargeback
from app.models.fee import Fee
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.payment_attempt import PaymentAttempt
from app.models.refund import Refund
from app.models.settlement import Settlement
from app.models.settlement_line import SettlementLine
from app.models.tax import Tax


class IngestionService:
    """Ingest a provider snapshot into the database, idempotently."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # -- public API -----------------------------------------------------------

    def ingest(self, provider) -> IngestionResult:
        """Ingest a full provider snapshot.

        Args:
            provider: a ``PaymentProvider`` implementation.

        Returns:
            The structured :class:`IngestionResult`. Status is ``COMPLETED``
            when every record was persisted or no-op'd, ``PARTIAL`` when some
            records were rejected as invalid, ``FAILED`` only when the
            transaction itself aborted (the error is re-raised).
        """
        result = IngestionResult(provider=provider.name)
        try:
            self._ingest_merchants(provider, result)
            self._ingest_payments(provider, result)
            self._ingest_children(provider, result)
            self._ingest_settlements(provider, result)
        except Exception:
            self.session.rollback()
            result.finish("FAILED")
            raise
        self.session.commit()
        return result.finish()

    # -- stages ---------------------------------------------------------------

    def _ingest_merchants(self, provider, result: IngestionResult) -> None:
        for dto in provider.get_merchants():
            try:
                validate_merchant(dto)
            except IngestionValidationError as error:
                result.record_error(error.entity, error.record_id, error.reason)
                continue
            fields = normalize_merchant(dto)
            merchant = self.session.get(Merchant, fields["id"])
            if merchant is None:
                self.session.add(Merchant(**fields))
                result.record_created("merchants")
            else:
                # Merchants carry no provider version column; idempotency is
                # field equality: identical replay is a no-op, changed data
                # updates.
                changed = {
                    key: value
                    for key, value in fields.items()
                    if getattr(merchant, key) != value
                }
                if changed:
                    for key, value in changed.items():
                        setattr(merchant, key, value)
                    result.record_updated("merchants")
                else:
                    result.record_noop("merchants")

    def _ingest_payments(self, provider, result: IngestionResult) -> None:
        merchant_ids = {m.merchant_id for m in provider.get_merchants()}
        for dto in provider.get_payments():
            try:
                validate_payment(dto, merchant_ids)
            except IngestionValidationError as error:
                result.record_error(error.entity, error.record_id, error.reason)
                continue
            fields = normalize_payment(dto)
            payment = self.session.get(Payment, fields["id"])
            if payment is None:
                payment = Payment(**fields)
                self.session.add(payment)
                result.record_created("payments")
            else:
                self._apply_payment_update(payment, fields, dto.version, result)

    def _ingest_children(self, provider, result: IngestionResult) -> None:
        payment_ids = {p.payment_id for p in provider.get_payments()}

        for dto in provider.get_payment_attempts():
            self._ingest_child(
                dto,
                validator=validate_payment_attempt,
                normalizer=normalize_payment_attempt,
                parents=payment_ids,
                model=PaymentAttempt,
                entity="payment_attempts",
                result=result,
            )
        for dto in provider.get_refunds():
            self._ingest_child(
                dto,
                validator=validate_refund,
                normalizer=normalize_refund,
                parents=payment_ids,
                model=Refund,
                entity="refunds",
                result=result,
            )
        for dto in provider.get_chargebacks():
            self._ingest_child(
                dto,
                validator=validate_chargeback,
                normalizer=normalize_chargeback,
                parents=payment_ids,
                model=Chargeback,
                entity="chargebacks",
                result=result,
            )
        for dto in provider.get_fees():
            self._ingest_child(
                dto,
                validator=validate_fee,
                normalizer=normalize_fee,
                parents=payment_ids,
                model=Fee,
                entity="fees",
                result=result,
            )
        for dto in provider.get_taxes():
            self._ingest_child(
                dto,
                validator=validate_tax,
                normalizer=normalize_tax,
                parents=payment_ids,
                model=Tax,
                entity="taxes",
                result=result,
            )

    def _ingest_settlements(self, provider, result: IngestionResult) -> None:
        payment_ids = {p.payment_id for p in provider.get_payments()}
        merchant_ids = {m.merchant_id for m in provider.get_merchants()}
        # A settlement may reference both a payment and a merchant; either
        # linkage must exist in the snapshot.
        known_parents = payment_ids | merchant_ids
        for dto in provider.get_settlements():
            try:
                validate_settlement(dto, known_parents)
            except IngestionValidationError as error:
                result.record_error(error.entity, error.record_id, error.reason)
                continue
            fields = normalize_settlement(dto)
            settlement = self.session.get(Settlement, fields["id"])
            if settlement is None:
                self.session.add(Settlement(**fields))
                result.record_created("settlements")
            else:
                self._apply_update(
                    settlement, fields, dto.version, "settlements", result
                )

        settlement_ids = {s.settlement_id for s in provider.get_settlements()}
        for dto in provider.get_settlement_lines():
            self._ingest_child(
                dto,
                validator=validate_settlement_line,
                normalizer=normalize_settlement_line,
                parents=settlement_ids,
                model=SettlementLine,
                entity="settlement_lines",
                result=result,
            )

    # -- record upserts ---------------------------------------------------------

    def _ingest_child(self, dto, validator, normalizer, parents, model, entity, result):
        """Validate and upsert one child record, tolerating invalid records."""
        try:
            validator(dto, parents)
        except IngestionValidationError as error:
            result.record_error(error.entity, error.record_id, error.reason)
            return
        fields = normalizer(dto)
        existing = self.session.get(model, fields["id"])
        if existing is None:
            self.session.add(model(**fields))
            result.record_created(entity)
        else:
            self._apply_update(existing, fields, dto.version, entity, result)

    # -- version handling ---------------------------------------------------------

    def _apply_update(self, existing, fields, version, entity, result) -> None:
        """Version-gated update for versioned entities.

        Payment and Settlement carry a provider version column: same version
        is an idempotent no-op, an older version never overwrites newer
        state. Entities without a version column (refunds, fees, taxes,
        chargebacks, settlement lines) fall back to field comparison.
        """
        if hasattr(existing, "version"):
            if version <= existing.version:
                return result.record_noop(entity)
            for key, value in fields.items():
                setattr(existing, key, value)
            return result.record_updated(entity)

        changed = {
            key: value
            for key, value in fields.items()
            if getattr(existing, key) != value
        }
        if changed:
            for key, value in changed.items():
                setattr(existing, key, value)
            result.record_updated(entity)
        else:
            result.record_noop(entity)

    def _apply_payment_update(self, payment, fields, version, result) -> None:
        """Payment updates additionally respect the domain state machine.

        The provider payload drives the status, but an invalid transition
        (e.g. CAPTURED -> AUTHORIZED replayed from a stale snapshot) is a
        validation failure, not silent corruption.
        """
        if version <= payment.version:
            result.record_noop("payments")
            return
        if payment.status != fields["status"]:
            try:
                payment.transition_to(fields["status"])
            except InvalidStateTransitionError as error:
                result.record_error(
                    "payments",
                    fields["id"],
                    f"provider status transition rejected: {error}",
                )
                return
        for key, value in fields.items():
            if key == "status":
                continue  # already applied through the state machine
            setattr(payment, key, value)
        result.record_updated("payments")
