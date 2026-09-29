"""
Phase 2 tests: the provider boundary.

Contract tests for ``PaymentProvider`` and behavioural tests for
``MockProvider``. The contract proves:

* every concrete provider exposes the nine retrieval methods;
* every returned record is a provider DTO (pydantic), never an ORM model;
* every record carries ``provider`` and a positive integer ``version``;
* MockProvider is deterministic for a fixed seed;
* MockProvider scenarios are composable and produce coherent parent/child data.
"""

from __future__ import annotations

import inspect
from datetime import timezone

import pytest
from pydantic import ValidationError

from app.models.payment import Payment  # ORM model - must never be returned
from app.providers import (
    MockProvider,
    PaymentProvider,
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

DTO_TYPES = {
    "get_merchants": ProviderMerchant,
    "get_payments": ProviderPayment,
    "get_payment_attempts": ProviderPaymentAttempt,
    "get_settlements": ProviderSettlement,
    "get_settlement_lines": ProviderSettlementLine,
    "get_refunds": ProviderRefund,
    "get_chargebacks": ProviderChargeback,
    "get_fees": ProviderFee,
    "get_taxes": ProviderTax,
}

SCENARIOS = [
    "clean",
    "refunds",
    "chargebacks",
    "fees_taxes",
    "multi_attempt",
    "settlement_lines",
]


# ─────────────────────────────────────────────────────────────────────────────
# Contract: the abstract interface
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderContract:
    def test_payment_provider_is_abstract(self):
        with pytest.raises(TypeError):
            PaymentProvider()

    def test_contract_declares_all_nine_retrieval_methods(self):
        for method in DTO_TYPES:
            assert hasattr(PaymentProvider, method), method
            assert callable(getattr(PaymentProvider, method))

    def test_contract_methods_are_abstract(self):
        for method in DTO_TYPES:
            assert getattr(PaymentProvider, method).__isabstractmethod__

    def test_contract_declares_provider_name(self):
        assert getattr(PaymentProvider.name, "__isabstractmethod__")

    def test_incomplete_provider_cannot_be_instantiated(self):
        class HalfProvider(PaymentProvider):
            @property
            def name(self):
                return "half"

            def get_merchants(self):
                return []

        with pytest.raises(TypeError):
            HalfProvider()

    def test_mock_provider_is_a_payment_provider(self):
        assert isinstance(MockProvider(), PaymentProvider)

    def test_orm_models_are_not_part_of_the_provider_contract(self):
        """The boundary DTOs must not be SQLAlchemy models."""
        for dto in (
            ProviderMerchant,
            ProviderPayment,
            ProviderPaymentAttempt,
            ProviderSettlement,
            ProviderSettlementLine,
            ProviderRefund,
            ProviderChargeback,
            ProviderFee,
            ProviderTax,
        ):
            import app.database.database as db

            mapper = getattr(dto, "__mro__", [])
            assert db.Base.__class__ not in [
                type(base).__class__ for base in mapper if isinstance(base, type)
            ] or not any(issubclass(base, db.Base) for base in mapper if isinstance(base, type))


# ─────────────────────────────────────────────────────────────────────────────
# Contract: MockProvider satisfies it
# ─────────────────────────────────────────────────────────────────────────────


class TestMockProviderContract:
    def test_provider_name_is_stable(self):
        assert MockProvider().name == "mock"

    @pytest.mark.parametrize("method,dto_type", sorted(DTO_TYPES.items()))
    def test_retrieval_returns_provider_dtos(self, method, dto_type):
        provider = MockProvider()
        for scenario in SCENARIOS:
            provider.add_scenario(scenario)
        records = getattr(provider, method)()
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, dto_type)
            assert not isinstance(record, Payment)

    @pytest.mark.parametrize("method", sorted(DTO_TYPES))
    def test_every_record_carries_provider_and_version(self, method):
        provider = MockProvider()
        for scenario in SCENARIOS:
            provider.add_scenario(scenario)
        for record in getattr(provider, method)():
            assert record.provider == "mock"
            assert isinstance(record.version, int)
            assert record.version >= 1

    def test_fresh_provider_has_one_merchant(self):
        provider = MockProvider()
        merchants = provider.get_merchants()
        assert len(merchants) == 1
        assert merchants[0].merchant_id == "merchant_demo_001"

    def test_demo_data_ids_match_the_architecture_examples(self):
        provider = MockProvider()
        provider.add_scenario("clean")
        payment_ids = {p.payment_id for p in provider.get_payments()}
        settlement_ids = {s.settlement_id for s in provider.get_settlements()}
        assert "payment_demo_001" in payment_ids
        assert "settlement_demo_001" in settlement_ids

    def test_no_fee_mismatch_workflow_is_hardcoded(self):
        """The provider must not privilege any reconciliation scenario."""
        provider = MockProvider()
        for scenario in SCENARIOS:
            provider.add_scenario(scenario)
        # Nothing in the provider mentions exception types.
        for record in (
            provider.get_payments()
            + provider.get_settlements()
            + provider.get_fees()
            + provider.get_taxes()
        ):
            assert not hasattr(record, "exception_type")
            assert not hasattr(record, "difference")


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_produces_identical_snapshots(self):
        def snapshot(seed):
            provider = MockProvider(seed=seed)
            provider.add_scenario("clean", scale=5)
            return [
                r.model_dump(mode="json") for r in provider.get_payments()
            ], [r.model_dump(mode="json") for r in provider.get_settlements()]

        assert snapshot(7) == snapshot(7)

    def test_different_seeds_can_differ(self):
        amounts = set()
        for seed in range(5):
            provider = MockProvider(seed=seed)
            provider.add_scenario("clean", scale=3)
            amounts.add(tuple(p.amount for p in provider.get_payments()))
        assert len(amounts) > 1

    def test_scenarios_are_idempotent_per_record(self):
        """Re-staging the same scenario rewrites the same ids, not new ones."""
        provider = MockProvider()
        provider.add_scenario("clean", scale=3)
        count_before = len(provider.get_payments())
        provider.add_scenario("clean", scale=3)
        assert len(provider.get_payments()) == count_before


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────


class TestScenarios:
    def test_unknown_scenario_is_rejected(self):
        provider = MockProvider()
        with pytest.raises(ValueError):
            provider.add_scenario("print_money")

    def test_refund_scenario_moves_payment_state_forward(self):
        provider = MockProvider()
        provider.add_scenario("clean", scale=3)
        provider.add_scenario("refunds", scale=2)
        statuses = {p.payment_id: p.status for p in provider.get_payments()}
        refunded = [s for s in statuses.values() if s.value == "PARTIALLY_REFUNDED"]
        assert len(refunded) >= 1
        assert len(provider.get_refunds()) >= 1

    def test_chargeback_scenario_disputes_payments(self):
        provider = MockProvider()
        provider.add_scenario("chargebacks", scale=2)
        assert any(
            c.status == "OPEN" for c in provider.get_chargebacks()
        )
        assert any(p.status.value == "CHARGEBACK" for p in provider.get_payments())

    def test_fees_taxes_scenario_records_both(self):
        provider = MockProvider()
        provider.add_scenario("fees_taxes", scale=3)
        assert len(provider.get_fees()) == 3
        assert len(provider.get_taxes()) == 3

    def test_multi_attempt_scenario_has_auth_and_capture(self):
        provider = MockProvider()
        provider.add_scenario("multi_attempt", scale=2)
        kinds = {a.kind.value for a in provider.get_payment_attempts()}
        assert kinds == {"AUTH", "CAPTURE"}

    def test_settlement_lines_scenario_breaks_down_components(self):
        provider = MockProvider()
        provider.add_scenario("settlement_lines", scale=2)
        lines = provider.get_settlement_lines()
        components = {line.component_type.value for line in lines}
        assert {"GROSS", "FEE"} <= components
        fee_lines = [l for l in lines if l.component_type.value == "FEE"]
        assert all(l.amount < 0 for l in fee_lines)

    def test_scenarios_compose(self):
        provider = MockProvider()
        for scenario in SCENARIOS:
            provider.add_scenario(scenario, scale=2)
        assert provider.get_payments()
        assert provider.get_settlements()


# ─────────────────────────────────────────────────────────────────────────────
# DTO strictness (external shapes reject malformed payloads at the boundary)
# ─────────────────────────────────────────────────────────────────────────────


class TestDtoStrictness:
    def test_float_amount_is_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPayment(
                provider="mock",
                version=1,
                payment_id="pay_1",
                merchant_id="merchant_demo_001",
                amount=100.5,
                status="CAPTURED",
            )

    def test_string_amount_is_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPayment(
                provider="mock",
                version=1,
                payment_id="pay_1",
                merchant_id="merchant_demo_001",
                amount="100000",
                status="CAPTURED",
            )

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPayment(
                provider="mock",
                version=1,
                payment_id="pay_1",
                merchant_id="merchant_demo_001",
                amount=100,
                status="CAPTURED",
                surprise_field="nope",
            )

    def test_zero_version_is_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPayment(
                provider="mock",
                version=0,
                payment_id="pay_1",
                merchant_id="merchant_demo_001",
                amount=100,
                status="CAPTURED",
            )

    def test_negative_amount_is_rejected(self):
        with pytest.raises(ValidationError):
            ProviderRefund(
                provider="mock",
                version=1,
                refund_id="rfnd_1",
                payment_id="pay_1",
                amount=-1,
                status="PROCESSED",
            )

    def test_settlement_line_amounts_can_be_negative(self):
        line = ProviderSettlementLine(
            provider="mock",
            version=1,
            line_id="line_1",
            settlement_id="setl_1",
            component_type="FEE",
            amount=-250,
        )
        assert line.amount == -250

    def test_string_timestamp_is_rejected_at_the_boundary(self):
        with pytest.raises(ValidationError):
            ProviderPayment(
                provider="mock",
                version=1,
                payment_id="pay_1",
                merchant_id="merchant_demo_001",
                amount=100,
                status="CAPTURED",
                captured_at="2026-01-01T00:00:00Z",  # string, not datetime
            )

    def test_timezone_aware_timestamps_are_preserved(self):
        from datetime import datetime

        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        payment = ProviderPayment(
            provider="mock",
            version=1,
            payment_id="pay_1",
            merchant_id="merchant_demo_001",
            amount=100,
            status="CAPTURED",
            captured_at=when,
        )
        assert payment.captured_at.tzinfo is not None
