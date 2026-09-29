"""
Phase 2 tests: ingestion service.

Covers the mandatory Phase 2 matrix:

* idempotency (ingest once -> N; ingest again -> still N);
* replay (three ingests of the same snapshot -> no duplicates);
* version handling (same version no-op, newer updates, older preserved);
* parent-before-child ordering (Merchant -> Payment -> Attempt -> Refund,
  Settlement -> SettlementLine);
* validation failures (missing id, invalid currency, non-integer amount,
  missing parent, invalid version, malformed timestamps) reject deterministically;
* database-level uniqueness backs the application-level upsert;
* ingestion never produces reconciliation output (no exceptions created).
"""

from __future__ import annotations

import inspect

import pytest
from sqlalchemy import func, select

from app.ingestion import IngestionService, IngestionResult
from app.ingestion.validation import IngestionValidationError
from app.models.chargeback import Chargeback
from app.models.fee import Fee
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.payment_attempt import PaymentAttempt
from app.models.refund import Refund
from app.models.settlement import Settlement
from app.models.settlement_line import SettlementLine
from app.models.tax import Tax
from app.providers import MockProvider


@pytest.fixture
def provider():
    p = MockProvider(seed=42)
    p.add_scenario("clean", scale=3)
    return p


@pytest.fixture
def service(db_session):
    return IngestionService(db_session)


def _counts(session):
    return {
        "merchants": session.scalar(select(func.count()).select_from(Merchant)),
        "payments": session.scalar(select(func.count()).select_from(Payment)),
        "attempts": session.scalar(select(func.count()).select_from(PaymentAttempt)),
        "settlements": session.scalar(select(func.count()).select_from(Settlement)),
        "lines": session.scalar(select(func.count()).select_from(SettlementLine)),
        "refunds": session.scalar(select(func.count()).select_from(Refund)),
        "chargebacks": session.scalar(select(func.count()).select_from(Chargeback)),
        "fees": session.scalar(select(func.count()).select_from(Fee)),
        "taxes": session.scalar(select(func.count()).select_from(Tax)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency (tests 1-2) and replay (mandatory test)
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_ingest_once_creates_n_records(self, db_session, service, provider):
        result = service.ingest(provider)
        assert result.status == "COMPLETED"
        assert result.count("payments", "created") == 3
        assert result.count("settlements", "created") == 3
        assert result.count("merchants", "created") == 1
        counts = _counts(db_session)
        assert counts["payments"] == 3
        assert counts["settlements"] == 3

    def test_ingest_twice_still_n_records(self, db_session, service, provider):
        service.ingest(provider)
        counts_after_first = _counts(db_session)

        result = service.ingest(provider)

        assert result.count("payments", "created") == 0
        assert result.count("payments", "noop") == 3
        counts_after_second = _counts(db_session)
        assert counts_after_second == counts_after_first

    def test_replay_three_times_no_duplicates(self, db_session, service, provider):
        """The mandatory replay test: 3x ingest of the same snapshot."""
        for _ in range(3):
            service.ingest(provider)
        counts = _counts(db_session)
        assert counts == {
            "merchants": 1,
            "payments": 3,
            "attempts": 0,
            "settlements": 3,
            "lines": 0,
            "refunds": 0,
            "chargebacks": 0,
            "fees": 0,
            "taxes": 0,
        }

    def test_replay_preserves_field_values(self, db_session, service, provider):
        service.ingest(provider)
        original_amount = db_session.get(Payment, "payment_demo_001").amount
        service.ingest(provider)
        assert db_session.get(Payment, "payment_demo_001").amount == original_amount

    def test_duplicate_provider_id_violates_db_uniqueness(
        self, db_session, service, provider
    ):
        """Test 5: the PK itself is the uniqueness guarantee."""
        service.ingest(provider)
        merchant = db_session.get(Merchant, "merchant_demo_001")
        duplicate = Payment(
            id="payment_demo_001",  # same provider id
            merchant_id=merchant.id,
            amount=1,
            currency="INR",
            status="CREATED",
            version=99,
        )
        db_session.add(duplicate)
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Version handling (tests 3-4)
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionHandling:
    def test_newer_version_updates_the_record(self, db_session, service, provider):
        service.ingest(provider)
        provider.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=999,
            version=2,
        )
        result = service.ingest(provider)
        assert result.count("payments", "updated") == 1
        payment = db_session.get(Payment, "payment_demo_001")
        assert payment.amount == 999
        assert payment.version == 2

    def test_same_version_is_a_noop(self, db_session, service, provider):
        service.ingest(provider)
        provider.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=12345,
            version=1,
        )
        result = service.ingest(provider)
        assert result.count("payments", "noop") == 3
        # Same version must NOT overwrite state.
        assert db_session.get(Payment, "payment_demo_001").amount != 12345

    def test_older_version_does_not_overwrite_newer_state(
        self, db_session, service, provider
    ):
        service.ingest(provider)
        provider.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=111,
            version=3,
        )
        service.ingest(provider)
        assert db_session.get(Payment, "payment_demo_001").amount == 111

        stale = MockProvider(seed=1)
        stale.add_merchant("merchant_demo_001", name="Demo Merchant 001")
        stale.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=222,
            version=2,
        )
        result = service.ingest(stale)
        assert result.count("payments", "noop") >= 1
        assert db_session.get(Payment, "payment_demo_001").amount == 111
        assert db_session.get(Payment, "payment_demo_001").version == 3

    def test_version_update_for_children(self, db_session, service, provider):
        service.ingest(provider)
        provider.add_fee(
            "fee_demo_001", payment_id="payment_demo_001", amount=5000, version=1
        )
        service.ingest(provider)
        provider.add_fee(
            "fee_demo_001", payment_id="payment_demo_001", amount=7777, version=2
        )
        result = service.ingest(provider)
        assert result.count("fees", "updated") == 1
        assert db_session.get(Fee, "fee_demo_001").amount == 7777

    def test_payment_status_moves_through_state_machine(
        self, db_session, service, provider
    ):
        from app.schemas.enums import PaymentStatus

        service.ingest(provider)
        payment = db_session.get(Payment, "payment_demo_001")
        assert payment.status == "CAPTURED"

        provider.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=payment.amount,
            status=PaymentStatus.SETTLED,
            version=2,
        )
        service.ingest(provider)
        assert db_session.get(Payment, "payment_demo_001").status == "SETTLED"

    def test_invalid_status_transition_is_rejected_not_corrupted(
        self, db_session, service, provider
    ):
        from app.schemas.enums import PaymentStatus

        service.ingest(provider)
        # CAPTURED -> AUTHORIZED is not a legal PAYMENT transition.
        provider.add_payment(
            "payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=100,
            status=PaymentStatus.AUTHORIZED,
            version=2,
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("transition rejected" in e.reason for e in result.errors)
        payment = db_session.get(Payment, "payment_demo_001")
        assert payment.status == "CAPTURED"  # unchanged
        assert payment.amount != 100  # not partially updated either


# ─────────────────────────────────────────────────────────────────────────────
# Parent-before-child ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestParentBeforeChild:
    def test_full_chain_merchant_payment_attempt_refund(
        self, db_session, service
    ):
        provider = MockProvider()
        provider.add_merchant("merchant_chain", name="Chain Merchant")
        provider.add_payment(
            "pay_chain", merchant_id="merchant_chain", amount=500_000,
            captured_at=provider._timestamp(1),
        )
        provider.add_payment_attempt(
            "att_chain", payment_id="pay_chain", amount=500_000
        )
        provider.add_refund("rfnd_chain", payment_id="pay_chain", amount=100_000)
        provider.add_fee("fee_chain", payment_id="pay_chain", amount=2_000)
        provider.add_tax("tax_chain", payment_id="pay_chain", amount=360)
        provider.add_chargeback("cb_chain", payment_id="pay_chain", amount=1_000)

        result = service.ingest(provider)
        assert result.status == "COMPLETED"
        assert result.errors == []

        assert db_session.get(Merchant, "merchant_chain") is not None
        assert db_session.get(Payment, "pay_chain") is not None
        assert db_session.get(PaymentAttempt, "att_chain") is not None
        assert db_session.get(Refund, "rfnd_chain") is not None
        assert db_session.get(Fee, "fee_chain") is not None
        assert db_session.get(Tax, "tax_chain") is not None
        assert db_session.get(Chargeback, "cb_chain") is not None

    def test_settlement_then_line(self, db_session, service):
        provider = MockProvider()
        provider.add_merchant("merchant_chain2", name="Chain Merchant 2")
        provider.add_payment(
            "pay_chain2", merchant_id="merchant_chain2", amount=80_000
        )
        provider.add_settlement(
            "setl_chain2", payment_id="pay_chain2", merchant_id="merchant_chain2",
            amount=78_000,
        )
        provider.add_settlement_line(
            "line_gross", settlement_id="setl_chain2",
            component_type="GROSS", amount=80_000,
        )
        provider.add_settlement_line(
            "line_fee", settlement_id="setl_chain2",
            component_type="FEE", amount=-2_000,
        )
        result = service.ingest(provider)
        assert result.status == "COMPLETED"
        settlement = db_session.get(Settlement, "setl_chain2")
        assert settlement is not None
        assert len(settlement.lines) == 2

    def test_child_with_missing_parent_in_snapshot_is_rejected(
        self, db_session, service
    ):
        provider = MockProvider()
        # No merchant staged for this payment.
        provider.add_payment(
            "pay_orphan", merchant_id="merchant_missing", amount=1_000
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any(
            "not present in the provider snapshot" in e.reason
            for e in result.errors
        )
        assert db_session.scalar(select(func.count()).select_from(Payment)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Validation failures (deterministic rejection, no silent corruption)
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationFailures:
    def test_missing_payment_id(self, db_session, service):
        provider = MockProvider()
        record = provider.add_payment(
            "pay_x", merchant_id="merchant_demo_001", amount=100
        )
        # Simulate a provider bug: an empty id slips through.
        broken = record.model_copy(update={"payment_id": ""})
        provider._payments["pay_x"] = broken
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("non-empty" in e.reason for e in result.errors)

    def test_invalid_currency(self, db_session, service):
        from app.providers.dto import ProviderPayment

        provider = MockProvider()
        # model_construct bypasses pydantic so the raw invalid value reaches
        # ingestion validation (a real provider adapter would reject it at
        # the boundary; ingestion must not depend on that).
        provider._payments["pay_bad"] = ProviderPayment.model_construct(
            provider="mock",
            version=1,
            payment_id="pay_bad",
            merchant_id="merchant_demo_001",
            amount=100,
            currency="GBP",  # not in CURRENCY_VALUES
            status="CAPTURED",
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("currency" in e.reason for e in result.errors)
        assert db_session.scalar(select(func.count()).select_from(Payment)) == 0

    def test_non_integer_amount_rejected(self, db_session, service):
        from app.providers.dto import ProviderPayment

        provider = MockProvider()
        provider._payments["pay_float"] = ProviderPayment.model_construct(
            provider="mock",
            version=1,
            payment_id="pay_float",
            merchant_id="merchant_demo_001",
            amount=100.5,
            currency="INR",
            status="CAPTURED",
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("integer" in e.reason for e in result.errors)

    def test_invalid_version_rejected(self, db_session, service):
        from app.providers.dto import ProviderPayment

        provider = MockProvider()
        provider._payments["pay_v"] = ProviderPayment.model_construct(
            provider="mock",
            version=0,
            payment_id="pay_v",
            merchant_id="merchant_demo_001",
            amount=100,
            status="CAPTURED",
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("version" in e.reason for e in result.errors)

    def test_malformed_timestamp_rejected(self, db_session, service):
        from datetime import datetime

        from app.providers.dto import ProviderPayment

        provider = MockProvider()
        # Bypass pydantic to emulate a provider that emits a date-like string
        # where the contract demands a datetime.
        broken = ProviderPayment.model_construct(
            provider="mock",
            version=1,
            payment_id="pay_ts",
            merchant_id="merchant_demo_001",
            amount=100,
            status="CAPTURED",
            captured_at="not-a-timestamp",
            provider_created_at=None,
            currency="INR",
        )
        assert not isinstance(broken.captured_at, datetime)
        provider._payments["pay_ts"] = broken
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert any("malformed timestamps are rejected" in e.reason for e in result.errors)

    def test_valid_records_survive_invalid_ones(self, db_session, service):
        from app.providers.dto import ProviderPayment

        provider = MockProvider()
        provider.add_payment(
            "pay_good", merchant_id="merchant_demo_001", amount=5_000
        )
        provider._payments["pay_bad"] = ProviderPayment.model_construct(
            provider="mock",
            version=1,
            payment_id="pay_bad",
            merchant_id="merchant_demo_001",
            amount=100,
            currency="XYZ",
            status="CAPTURED",
        )
        result = service.ingest(provider)
        assert result.status == "PARTIAL"
        assert db_session.get(Payment, "pay_good") is not None
        assert db_session.get(Payment, "pay_bad") is None


# ─────────────────────────────────────────────────────────────────────────────
# Transaction boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestTransactionBoundary:
    def test_unexpected_failure_rolls_back_the_whole_batch(
        self, db_session, service
    ):
        class ExplodingProvider(MockProvider):
            def get_settlements(self):
                raise RuntimeError("provider connection lost")

            def get_settlement_lines(self):
                return []

        provider = ExplodingProvider()
        provider.add_payment("pay_boom", merchant_id="merchant_demo_001", amount=1)

        with pytest.raises(RuntimeError):
            service.ingest(provider)

        # Rollback: nothing persisted, not even the parent records.
        db_session.rollback()
        assert db_session.scalar(select(func.count()).select_from(Payment)) == 0
        assert db_session.scalar(select(func.count()).select_from(Merchant)) == 0

    def test_ingestion_result_shape(self, db_session, service, provider):
        result = service.ingest(provider)
        assert isinstance(result, IngestionResult)
        summary = result.summary()
        assert summary["provider"] == "mock"
        assert summary["status"] == "COMPLETED"
        assert summary["total_created"] == 7
        assert summary["completed_at"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Scope guards: ingestion does NOT do reconciliation's job
# ─────────────────────────────────────────────────────────────────────────────


class TestNoReconciliationLeakage:
    def test_ingestion_creates_no_exceptions(self, db_session, service, provider):
        from app.models.exception import FinancialException

        # A deliberately "weird" snapshot: settlement far below payment amount.
        provider.add_settlement(
            "setl_weird",
            payment_id="payment_demo_001",
            merchant_id="merchant_demo_001",
            amount=1,
        )
        service.ingest(provider)
        assert db_session.scalar(select(func.count()).select_from(FinancialException)) == 0

    def test_ingestion_module_does_not_import_reconciliation(self):
        import app.ingestion.service as service_module

        banned = (
            "app.reconciliation",
            "app.ml",
            "app.llm",
            "app.agent",
            "app.services.resolution_engine",
        )
        source = inspect.getsource(service_module)
        for module in banned:
            assert module not in source
