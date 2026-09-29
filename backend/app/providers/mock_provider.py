"""
MockProvider — a simulated external financial provider.

This is NOT a fake implementation hidden inside the domain layer: it models an
external system with its own record versions, its own ordering, and the ability
to emit arbitrary scenarios (clean payments, refunds, disputes, multi-capture
histories, settlements with component lines).

Determinism: ``seed`` controls the generated ids and the RNG; the same seed and
the same scenario always produce byte-identical DTOs. Scenarios are composable:
call :meth:`add_scenario` for each behaviour the caller wants present, or use
:meth:`add_payment` and friends to feed provider-shaped records directly (this
is how tests simulate a provider that has changed its state between polls).

The provider knows nothing about the database, reconciliation, or exceptions.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.providers.base import PaymentProvider
from app.providers.dto import (
    ProviderChargeback,
    ProviderFee,
    ProviderMerchant,
    ProviderPayment,
    ProviderPaymentAttempt,
    ProviderRefund,
    ProviderSettlement,
    ProviderSettlementLine,
    ProviderTax,
)
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
)

PROVIDER_NAME = "mock"

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class MockProvider(PaymentProvider):
    """Deterministic, in-memory simulated financial provider."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._merchants: dict[str, ProviderMerchant] = {}
        self._payments: dict[str, ProviderPayment] = {}
        self._attempts: dict[str, ProviderPaymentAttempt] = {}
        self._settlements: dict[str, ProviderSettlement] = {}
        self._settlement_lines: dict[str, ProviderSettlementLine] = {}
        self._refunds: dict[str, ProviderRefund] = {}
        self._chargebacks: dict[str, ProviderChargeback] = {}
        self._fees: dict[str, ProviderFee] = {}
        self._taxes: dict[str, ProviderTax] = {}

        # A fresh provider always has at least one merchant, like a real
        # gateway account.
        self.add_merchant("merchant_demo_001", name="Demo Merchant 001")

    # -- PaymentProvider contract -------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def get_merchants(self) -> list[ProviderMerchant]:
        return list(self._merchants.values())

    def get_payments(self) -> list[ProviderPayment]:
        return list(self._payments.values())

    def get_payment_attempts(self) -> list[ProviderPaymentAttempt]:
        return list(self._attempts.values())

    def get_settlements(self) -> list[ProviderSettlement]:
        return list(self._settlements.values())

    def get_settlement_lines(self) -> list[ProviderSettlementLine]:
        return list(self._settlement_lines.values())

    def get_refunds(self) -> list[ProviderRefund]:
        return list(self._refunds.values())

    def get_chargebacks(self) -> list[ProviderChargeback]:
        return list(self._chargebacks.values())

    def get_fees(self) -> list[ProviderFee]:
        return list(self._fees.values())

    def get_taxes(self) -> list[ProviderTax]:
        return list(self._taxes.values())

    # -- record staging (the "external world changes" surface) ---------------

    def add_merchant(
        self,
        merchant_id: str,
        name: str,
        status: str = "ACTIVE",
        category_code: Optional[str] = None,
        onboarded_at: Optional[datetime] = None,
        version: int = 1,
    ) -> ProviderMerchant:
        record = ProviderMerchant(
            provider=PROVIDER_NAME,
            version=version,
            merchant_id=merchant_id,
            name=name,
            status=status,
            category_code=category_code,
            onboarded_at=onboarded_at or _EPOCH,
        )
        self._merchants[merchant_id] = record
        return record

    def add_payment(
        self,
        payment_id: str,
        merchant_id: str,
        amount: int,
        currency: Currency = Currency.INR,
        status: PaymentStatus = PaymentStatus.CAPTURED,
        method: Optional[str] = "upi",
        captured_at: Optional[datetime] = None,
        order_id: Optional[str] = None,
        version: int = 1,
    ) -> ProviderPayment:
        record = ProviderPayment(
            provider=PROVIDER_NAME,
            version=version,
            payment_id=payment_id,
            merchant_id=merchant_id,
            order_id=order_id or f"order_{payment_id}",
            amount=amount,
            currency=currency,
            status=status,
            method=method,
            captured_at=captured_at,
            provider_created_at=captured_at,
        )
        self._payments[payment_id] = record
        return record

    def add_payment_attempt(
        self,
        attempt_id: str,
        payment_id: str,
        amount: int,
        attempt_no: int = 1,
        kind: PaymentAttemptKind = PaymentAttemptKind.CAPTURE,
        status: PaymentAttemptStatus = PaymentAttemptStatus.CAPTURED,
        occurred_at: Optional[datetime] = None,
        version: int = 1,
    ) -> ProviderPaymentAttempt:
        record = ProviderPaymentAttempt(
            provider=PROVIDER_NAME,
            version=version,
            attempt_id=attempt_id,
            payment_id=payment_id,
            attempt_no=attempt_no,
            kind=kind,
            amount=amount,
            status=status,
            provider_attempt_id=attempt_id,
            occurred_at=occurred_at,
        )
        self._attempts[attempt_id] = record
        return record

    def add_settlement(
        self,
        settlement_id: str,
        payment_id: str,
        merchant_id: str,
        amount: int,
        currency: Currency = Currency.INR,
        status: SettlementStatus = SettlementStatus.SETTLED,
        settled_at: Optional[datetime] = None,
        provider_batch_id: Optional[str] = None,
        version: int = 1,
    ) -> ProviderSettlement:
        record = ProviderSettlement(
            provider=PROVIDER_NAME,
            version=version,
            settlement_id=settlement_id,
            payment_id=payment_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            status=status,
            settled_at=settled_at,
            provider_batch_id=provider_batch_id or f"payout_{settlement_id}",
        )
        self._settlements[settlement_id] = record
        return record

    def add_settlement_line(
        self,
        line_id: str,
        settlement_id: str,
        component_type: SettlementComponentType,
        amount: int,
        currency: Currency = Currency.INR,
        source_entity_type: Optional[str] = None,
        source_entity_id: Optional[str] = None,
        description: Optional[str] = None,
        version: int = 1,
    ) -> ProviderSettlementLine:
        record = ProviderSettlementLine(
            provider=PROVIDER_NAME,
            version=version,
            line_id=line_id,
            settlement_id=settlement_id,
            component_type=component_type,
            amount=amount,
            currency=currency,
            source_entity_type=source_entity_type,
            source_entity_id=source_entity_id,
            description=description,
        )
        self._settlement_lines[line_id] = record
        return record

    def add_refund(
        self,
        refund_id: str,
        payment_id: str,
        amount: int,
        currency: Currency = Currency.INR,
        status: RefundStatus = RefundStatus.PROCESSED,
        reason: Optional[str] = None,
        refund_timestamp: Optional[datetime] = None,
        version: int = 1,
    ) -> ProviderRefund:
        record = ProviderRefund(
            provider=PROVIDER_NAME,
            version=version,
            refund_id=refund_id,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
            status=status,
            reason=reason,
            refund_timestamp=refund_timestamp,
        )
        self._refunds[refund_id] = record
        return record

    def add_chargeback(
        self,
        chargeback_id: str,
        payment_id: str,
        amount: int,
        currency: Currency = Currency.INR,
        status: ChargebackStatus = ChargebackStatus.OPEN,
        reason_code: Optional[str] = None,
        evidence_due_at: Optional[datetime] = None,
        resolved_at: Optional[datetime] = None,
        notes: Optional[str] = None,
        version: int = 1,
    ) -> ProviderChargeback:
        record = ProviderChargeback(
            provider=PROVIDER_NAME,
            version=version,
            chargeback_id=chargeback_id,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
            status=status,
            reason_code=reason_code,
            evidence_due_at=evidence_due_at,
            resolved_at=resolved_at,
            notes=notes,
        )
        self._chargebacks[chargeback_id] = record
        return record

    def add_fee(
        self,
        fee_id: str,
        payment_id: str,
        amount: int,
        fee_type: FeeType = FeeType.TRANSACTION,
        currency: Currency = Currency.INR,
        processed_at: Optional[datetime] = None,
        version: int = 1,
    ) -> ProviderFee:
        record = ProviderFee(
            provider=PROVIDER_NAME,
            version=version,
            fee_id=fee_id,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
            fee_type=fee_type,
            processed_at=processed_at,
        )
        self._fees[fee_id] = record
        return record

    def add_tax(
        self,
        tax_id: str,
        payment_id: str,
        amount: int,
        tax_type: TaxType = TaxType.GST,
        currency: Currency = Currency.INR,
        jurisdiction: Optional[str] = None,
        version: int = 1,
    ) -> ProviderTax:
        record = ProviderTax(
            provider=PROVIDER_NAME,
            version=version,
            tax_id=tax_id,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
            tax_type=tax_type,
            jurisdiction=jurisdiction,
        )
        self._taxes[tax_id] = record
        return record

    # -- scenarios ------------------------------------------------------------

    def add_scenario(self, scenario: str, scale: int = 3) -> None:
        """Stage a named financial scenario onto the provider.

        Available scenarios (composable, none is privileged):

        * ``clean``       — captured payments settled in full, no discrepancies
        * ``refunds``     — partially refunded payments
        * ``chargebacks`` — disputed payments
        * ``fees_taxes``  — fees and taxes recorded against payments
        * ``multi_attempt`` — auth-then-capture payment histories
        * ``settlement_lines`` — settlements carrying component breakdowns

        Scale controls how many payments the scenario touches.
        """
        builders = {
            "clean": self._scenario_clean,
            "refunds": self._scenario_refunds,
            "chargebacks": self._scenario_chargebacks,
            "fees_taxes": self._scenario_fees_taxes,
            "multi_attempt": self._scenario_multi_attempt,
            "settlement_lines": self._scenario_settlement_lines,
        }
        try:
            builder = builders[scenario]
        except KeyError:
            raise ValueError(
                f"Unknown scenario {scenario!r}; known: {sorted(builders)}"
            ) from None
        builder(scale)

    def _timestamp(self, day_offset: int) -> datetime:
        return _EPOCH + timedelta(days=day_offset)

    def _scenario_clean(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            amount = self._rng.randrange(10_000, 1_000_000)
            self.add_payment(
                payment_id,
                merchant_id="merchant_demo_001",
                amount=amount,
                captured_at=self._timestamp(i),
            )
            self.add_settlement(
                f"settlement_demo_{i:03d}",
                payment_id=payment_id,
                merchant_id="merchant_demo_001",
                amount=amount,
                settled_at=self._timestamp(i + 2),
            )

    def _scenario_refunds(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            if payment_id not in self._payments:
                amount = self._rng.randrange(10_000, 1_000_000)
                self.add_payment(
                    payment_id,
                    merchant_id="merchant_demo_001",
                    amount=amount,
                    captured_at=self._timestamp(i),
                )
            else:
                amount = self._payments[payment_id].amount
            refund_amount = amount // 2
            self.add_refund(
                f"refund_demo_{i:03d}",
                payment_id=payment_id,
                amount=refund_amount,
                reason="customer_request",
                refund_timestamp=self._timestamp(i + 1),
            )
            payment = self._payments[payment_id]
            self._payments[payment_id] = payment.model_copy(
                update={
                    "version": payment.version + 1,
                    "status": PaymentStatus.PARTIALLY_REFUNDED,
                }
            )

    def _scenario_chargebacks(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            if payment_id not in self._payments:
                amount = self._rng.randrange(10_000, 1_000_000)
                self.add_payment(
                    payment_id,
                    merchant_id="merchant_demo_001",
                    amount=amount,
                    captured_at=self._timestamp(i),
                )
            else:
                amount = self._payments[payment_id].amount
            self.add_chargeback(
                f"chargeback_demo_{i:03d}",
                payment_id=payment_id,
                amount=amount,
                reason_code="fraudulent",
                evidence_due_at=self._timestamp(i + 7),
            )
            payment = self._payments[payment_id]
            self._payments[payment_id] = payment.model_copy(
                update={
                    "version": payment.version + 1,
                    "status": PaymentStatus.CHARGEBACK,
                }
            )

    def _scenario_fees_taxes(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            if payment_id not in self._payments:
                self.add_payment(
                    payment_id,
                    merchant_id="merchant_demo_001",
                    amount=self._rng.randrange(10_000, 1_000_000),
                    captured_at=self._timestamp(i),
                )
            self.add_fee(
                f"fee_demo_{i:03d}",
                payment_id=payment_id,
                amount=2_000 + i,
                fee_type=FeeType.TRANSACTION,
                processed_at=self._timestamp(i),
            )
            self.add_tax(
                f"tax_demo_{i:03d}",
                payment_id=payment_id,
                amount=360 + i,
                tax_type=TaxType.GST,
                jurisdiction="KA",
            )

    def _scenario_multi_attempt(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            if payment_id not in self._payments:
                self.add_payment(
                    payment_id,
                    merchant_id="merchant_demo_001",
                    amount=500_000,
                    status=PaymentStatus.AUTHORIZED,
                    method="card",
                    captured_at=None,
                )
            self.add_payment_attempt(
                f"attempt_demo_{i * 10:03d}",
                payment_id=payment_id,
                amount=500_000,
                attempt_no=1,
                kind=PaymentAttemptKind.AUTH,
                status=PaymentAttemptStatus.AUTHORIZED,
                occurred_at=self._timestamp(i),
            )
            self.add_payment_attempt(
                f"attempt_demo_{i * 10 + 1:03d}",
                payment_id=payment_id,
                amount=500_000,
                attempt_no=2,
                kind=PaymentAttemptKind.CAPTURE,
                occurred_at=self._timestamp(i),
            )

    def _scenario_settlement_lines(self, scale: int) -> None:
        for i in range(1, scale + 1):
            payment_id = f"payment_demo_{i:03d}"
            settlement_id = f"settlement_demo_{i:03d}"
            if settlement_id not in self._settlements:
                if payment_id not in self._payments:
                    self.add_payment(
                        payment_id,
                        merchant_id="merchant_demo_001",
                        amount=100_000,
                        captured_at=self._timestamp(i),
                    )
                self.add_settlement(
                    settlement_id,
                    payment_id=payment_id,
                    merchant_id="merchant_demo_001",
                    amount=98_000,
                    settled_at=self._timestamp(i + 2),
                )
            self.add_settlement_line(
                f"line_demo_{i * 10:03d}",
                settlement_id=settlement_id,
                component_type=SettlementComponentType.GROSS,
                amount=100_000,
            )
            self.add_settlement_line(
                f"line_demo_{i * 10 + 1:03d}",
                settlement_id=settlement_id,
                component_type=SettlementComponentType.FEE,
                amount=-2_000,
                source_entity_type="FEE",
                source_entity_id=f"fee_demo_{i:03d}",
            )
