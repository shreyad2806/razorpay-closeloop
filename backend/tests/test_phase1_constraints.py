"""
Phase 1 tests: database-level constraints.

Everything asserted here is enforced by the *database*, not by Python, and the
tests run on an engine with ``PRAGMA foreign_keys=ON`` so the checks are real:

* NOT NULL on required financial fields and on explicit currency;
* ``CHECK (amount >= 0)`` on unsigned amounts, and its deliberate absence on the
  entities that carry signed movements;
* CHECK constraints on enum-ish status/type strings;
* real foreign keys, including ``ON DELETE RESTRICT`` for financial records;
* uniqueness and idempotency: idempotency keys, one-result-per-case-per-run,
  one-pending-approval-per-resolution, one-running-run-per-scope;
* append-only enforcement for the ledger, audit and feedback tables.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.adjustment import Adjustment
from app.models.approval import Approval
from app.models.audit_event import AuditEvent
from app.models.chargeback import Chargeback
from app.models.evidence import Evidence
from app.models.exception import FinancialException
from app.models.feedback import Feedback
from app.models.fee import Fee
from app.models.immutability import ImmutableRecordError
from app.models.ledger_entry import LedgerEntry
from app.models.merchant import Merchant
from app.models.model_prediction import ModelPrediction
from app.models.payment import Payment
from app.models.payment_attempt import PaymentAttempt
from app.models.reconciliation import ReconciliationResult
from app.models.reconciliation_run import ReconciliationRun
from app.models.refund import Refund
from app.models.resolution import Resolution
from app.models.resolution_action import ResolutionAction
from app.models.root_cause import RootCause
from app.models.settlement import Settlement
from app.models.settlement_line import SettlementLine
from app.models.tax import Tax
from app.services.historical_case_store import HistoricalCaseRecord
from app.services.similarity_service import CaseEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────────────────────────────────────


def seed_merchant(session, merchant_id="MER-1"):
    session.add(Merchant(id=merchant_id, name="Test Merchant"))
    session.flush()
    return merchant_id


def seed_payment(session, payment_id="PAY-1", amount=25000):
    seed_merchant(session)
    session.add(
        Payment(id=payment_id, merchant_id="MER-1", amount=amount, currency="INR")
    )
    session.flush()
    return payment_id


def seed_exception(session, exception_id="EXC-1"):
    exc = FinancialException(
        id=exception_id,
        case_id="CASE-1",
        payment_id="PAY-1",
        batch_id="B-1",
        expected_amount=1000,
        actual_amount=900,
        difference=100,
        exception_type="FEE_DIFFERENCE",
        reconciliation_id="REC-1",
    )
    session.add(exc)
    session.flush()
    return exc


def seed_resolution(session, resolution_id="RES-1", rank=1):
    resolution = Resolution(
        id=resolution_id,
        exception_id="EXC-1",
        candidate_rank=rank,
        resolution_type="FEE_ADJUSTMENT",
    )
    session.add(resolution)
    session.flush()
    return resolution


def seed_run(session, run_id="RUN-1", scope_ref="B-1", status="PENDING"):
    run = ReconciliationRun(
        id=run_id,
        scope_type="BATCH",
        scope_ref=scope_ref,
        trigger="MANUAL",
        engine_version="1.0.0",
        status=status,
    )
    session.add(run)
    session.flush()
    return run


def seed_historical_case(session, case_id="HC-1"):
    case = HistoricalCaseRecord(
        id=case_id,
        exception_id="EXC-1",
        payment_id="PAY-1",
        exception_type="FEE_DIFFERENCE",
        payment_amount=1000,
        expected_amount=1000,
        actual_amount=900,
        difference=100,
        resolution_type="FEE_ADJUSTMENT",
        resolution_outcome="RESOLVED",
    )
    session.add(case)
    session.flush()
    return case


# ─────────────────────────────────────────────────────────────────────────────
# NOT NULL
# ─────────────────────────────────────────────────────────────────────────────


class TestRequiredFields:
    def test_payment_amount_is_required(self, db_session):
        seed_merchant(db_session)
        db_session.add(Payment(id="PAY-X", merchant_id="MER-1", amount=None))
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_currency_column_is_not_nullable(self):
        """Declared NOT NULL, so the database refuses a NULL currency.

        The ORM fills the column default when the attribute is unset, so this is
        asserted against the schema (and then against the database below with a
        raw INSERT) rather than by constructing a model instance.
        """
        from app.database.database import Base

        assert Base.metadata.tables["payments"].columns["currency"].nullable is False

    def test_database_rejects_a_null_currency(self, db_session):
        from sqlalchemy import text

        seed_merchant(db_session)
        with pytest.raises(IntegrityError):
            db_session.execute(
                text(
                    "INSERT INTO payments (id, merchant_id, amount, currency, "
                    "status, version) VALUES ('PAY-X', 'MER-1', 100, NULL, "
                    "'CREATED', 1)"
                )
            )
        db_session.rollback()

    def test_refund_requires_payment_and_amount(self, db_session):
        db_session.add(
            Refund(id="REF-X", payment_id=None, amount=100, currency="INR")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_resolution_action_requires_idempotency_key(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add(
            ResolutionAction(
                id="ACT-X",
                resolution_id="RES-1",
                action_type="ADJUSTMENT",
                idempotency_key=None,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Amount constraints
# ─────────────────────────────────────────────────────────────────────────────


class TestAmountConstraints:
    @pytest.mark.parametrize(
        "table_name",
        ["payments", "settlements", "refunds", "fees", "taxes", "chargebacks"],
    )
    def test_unsigned_amount_tables_declare_a_non_negative_check(
        self, table_name
    ):
        from app.database.database import Base

        table = Base.metadata.tables[table_name]
        checks = [
            str(constraint.sqltext)
            for constraint in table.constraints
            if constraint.__class__.__name__ == "CheckConstraint"
        ]
        assert any("amount >= 0" in sql for sql in checks), checks

    @pytest.mark.parametrize(
        "table_name", ["adjustments", "settlement_lines", "ledger_entries"]
    )
    def test_signed_tables_do_not_forbid_negative_amounts(self, table_name):
        from app.database.database import Base

        table = Base.metadata.tables[table_name]
        checks = [
            str(constraint.sqltext)
            for constraint in table.constraints
            if constraint.__class__.__name__ == "CheckConstraint"
        ]
        assert not any("amount >= 0" in sql for sql in checks), checks

    def test_negative_payment_amount_is_rejected(self, db_session):
        seed_merchant(db_session)
        db_session.add(
            Payment(id="PAY-NEG", merchant_id="MER-1", amount=-1, currency="INR")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_negative_refund_amount_is_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(
            Refund(id="REF-NEG", payment_id="PAY-1", amount=-1, currency="INR")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_negative_fee_and_tax_amounts_are_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(
            Fee(id="FEE-NEG", payment_id="PAY-1", amount=-1, currency="INR",
                fee_type="PLATFORM")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

        db_session.add(
            Tax(id="TAX-NEG", payment_id="PAY-1", amount=-1, currency="INR",
                tax_type="GST")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_negative_settlement_and_chargeback_amounts_are_rejected(
        self, db_session
    ):
        seed_payment(db_session)
        db_session.add(
            Settlement(id="SET-NEG", payment_id="PAY-1", amount=-1, currency="INR")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

        db_session.add(
            Chargeback(id="CB-NEG", payment_id="PAY-1", amount=-1, currency="INR")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_negative_resolution_amount_is_rejected(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add(
            Resolution(
                id="RES-NEG",
                exception_id="EXC-1",
                candidate_rank=1,
                resolution_type="FEE_ADJUSTMENT",
                amount_paise=-1,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_signed_entities_accept_negative_amounts(self, db_session):
        seed_payment(db_session)
        db_session.add_all(
            [
                Adjustment(
                    id="ADJ-NEG", payment_id="PAY-1", amount=-500,
                    currency="INR", adjustment_type="DEBIT",
                ),
                SettlementLine(
                    id="SL-NEG", settlement_id="SET-1", component_type="FEE",
                    amount=-500, currency="INR",
                ),
            ]
        )
        db_session.add(Settlement(id="SET-1", payment_id="PAY-1", amount=0,
                                  currency="INR"))
        db_session.flush()
        db_session.add(
            LedgerEntry(
                id="LED-NEG",
                payment_id="PAY-1",
                entry_type="ADJUSTMENT",
                amount=-500,
                currency="INR",
                direction="DEBIT",
                source="EXECUTION",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        db_session.flush()
        assert db_session.get(Adjustment, "ADJ-NEG").amount == -500
        assert db_session.get(LedgerEntry, "LED-NEG").amount == -500


# ─────────────────────────────────────────────────────────────────────────────
# Enum-ish CHECK constraints
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusAndTypeChecks:
    def test_invalid_currency_is_rejected(self, db_session):
        seed_merchant(db_session)
        db_session.add(
            Payment(id="PAY-CUR", merchant_id="MER-1", amount=100, currency="XYZ")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_payment_status_is_rejected(self, db_session):
        seed_merchant(db_session)
        db_session.add(
            Payment(
                id="PAY-ST", merchant_id="MER-1", amount=100,
                currency="INR", status="NOT_A_STATUS",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_exception_status_is_rejected(self, db_session):
        db_session.add(
            FinancialException(
                id="EXC-ST", case_id="CASE-1", payment_id="PAY-1", batch_id="B-1",
                expected_amount=1, actual_amount=1, difference=0,
                exception_type="FEE_DIFFERENCE", reconciliation_id="REC-1",
                status="SOMETHING_ELSE",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_reconciliation_run_status_and_trigger_are_rejected(
        self, db_session
    ):
        db_session.add(
            ReconciliationRun(
                id="RUN-ST", scope_type="BATCH", scope_ref="B-1",
                trigger="MANUAL", engine_version="1.0.0", status="WAT",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

        db_session.add(
            ReconciliationRun(
                id="RUN-TR", scope_type="BATCH", scope_ref="B-1",
                trigger="SOMETIMES", engine_version="1.0.0",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_approval_decision_is_rejected(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add(
            Approval(
                id="APR-ST", resolution_id="RES-1", requested_by="SYSTEM",
                decision="MAYBE",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_fee_type_is_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(
            Fee(id="FEE-T", payment_id="PAY-1", amount=100, currency="INR",
                fee_type="PLATFORM_FEE")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_evidence_relationship_is_rejected(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add(
            Evidence(
                id="EV-R", exception_id="EXC-1", entity_type="PAYMENT",
                entity_id="PAY-1", relationship="VIBES", recorded_by="HUMAN",
                payload={},
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_ledger_direction_and_source_are_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(
            LedgerEntry(
                id="LED-D", payment_id="PAY-1", entry_type="PAYMENT",
                amount=100, currency="INR", direction="SIDEWAYS", source="INGEST",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

        db_session.add(
            LedgerEntry(
                id="LED-S", payment_id="PAY-1", entry_type="PAYMENT",
                amount=100, currency="INR", direction="CREDIT", source="MAGIC",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_invalid_adjustment_origin_is_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(
            Adjustment(
                id="ADJ-O", payment_id="PAY-1", amount=100, currency="INR",
                adjustment_type="CREDIT", origin="SOMEWHERE_ELSE",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Foreign keys
# ─────────────────────────────────────────────────────────────────────────────


class TestForeignKeys:
    def test_payment_requires_an_existing_merchant(self, db_session):
        db_session.add(
            Payment(id="PAY-ORPHAN", merchant_id="MER-MISSING", amount=100)
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_payment_attempt_requires_an_existing_payment(self, db_session):
        db_session.add(
            PaymentAttempt(
                id="ATT-ORPHAN", payment_id="PAY-MISSING", attempt_no=1,
                kind="CAPTURE", amount=100, currency="INR",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: Refund(id="R", payment_id="PAY-MISSING", amount=1, currency="INR"),
            lambda: Fee(
                id="F", payment_id="PAY-MISSING", amount=1, currency="INR",
                fee_type="PLATFORM",
            ),
            lambda: Tax(
                id="T", payment_id="PAY-MISSING", amount=1, currency="INR",
                tax_type="GST",
            ),
            lambda: Adjustment(
                id="A", payment_id="PAY-MISSING", amount=1, currency="INR",
                adjustment_type="CREDIT",
            ),
            lambda: Chargeback(
                id="C", payment_id="PAY-MISSING", amount=1, currency="INR"
            ),
            lambda: Settlement(
                id="S", payment_id="PAY-MISSING", amount=1, currency="INR"
            ),
        ],
    )
    def test_payment_children_require_an_existing_payment(self, db_session, factory):
        db_session.add(factory())
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_settlement_line_requires_an_existing_settlement(self, db_session):
        db_session.add(
            SettlementLine(
                id="SL-ORPHAN", settlement_id="SET-MISSING",
                component_type="GROSS", amount=1, currency="INR",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_reconciliation_result_requires_an_existing_run(self, db_session):
        db_session.add(
            ReconciliationResult(
                id="REC-ORPHAN", case_id="CASE-1", payment_id="PAY-1",
                merchant_id="MER-1", batch_id="B-1",
                reconciliation_run_id="RUN-MISSING",
                payment_amount=1, expected_amount=1, actual_amount=1, difference=0,
                match_status="MATCHED", exception_type="EXACT_MATCH",
                currency="INR",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_evidence_requires_an_existing_exception(self, db_session):
        db_session.add(
            Evidence(
                id="EV-ORPHAN", exception_id="EXC-MISSING", entity_type="PAYMENT",
                entity_id="PAY-1", relationship="PRIMARY",
                recorded_by="RECONCILIATION",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_root_cause_requires_an_existing_exception(self, db_session):
        db_session.add(
            RootCause(id="RC-ORPHAN", exception_id="EXC-MISSING", cause_type="X")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_model_prediction_requires_an_existing_exception(self, db_session):
        db_session.add(
            ModelPrediction(
                id="MP-ORPHAN", exception_id="EXC-MISSING", model_name="m",
                model_version="1", feature_schema_version="v1", predicted_type="X",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_resolution_requires_an_existing_exception(self, db_session):
        db_session.add(
            Resolution(
                id="RES-ORPHAN", exception_id="EXC-MISSING", candidate_rank=1,
                resolution_type="FEE_ADJUSTMENT",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_resolution_action_requires_an_existing_resolution(self, db_session):
        db_session.add(
            ResolutionAction(
                id="ACT-ORPHAN", resolution_id="RES-MISSING",
                action_type="ADJUSTMENT", idempotency_key="k",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_approval_requires_an_existing_resolution(self, db_session):
        db_session.add(
            Approval(id="APR-ORPHAN", resolution_id="RES-MISSING",
                     requested_by="SYSTEM")
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_case_embedding_requires_an_existing_historical_case(self, db_session):
        db_session.add(
            CaseEmbedding(
                id="HC-MISSING", embedding_json="[]",
                exception_type="X", resolution_type="Y", resolution_outcome="Z",
                payment_amount=1, difference=0, case_text="t",
                embedding_model="m", embedding_dimension=384,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_deleting_a_merchant_with_payments_is_restricted(self, db_session):
        """ON DELETE RESTRICT: financial records are never orphaned."""
        seed_payment(db_session)
        merchant = db_session.get(Merchant, "MER-1")
        db_session.delete(merchant)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_deleting_a_payment_with_children_is_restricted(self, db_session):
        seed_payment(db_session)
        db_session.add(
            Refund(id="REF-1", payment_id="PAY-1", amount=100, currency="INR")
        )
        db_session.flush()
        payment = db_session.get(Payment, "PAY-1")
        db_session.delete(payment)
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_nullable_foreign_keys_are_allowed(self, db_session):
        """Unlinkable optionals stay optional after Phase 2 tightening.

        Phase 1 allowed a payment without a merchant; Phase 2 ingestion
        guarantees every payment has one, so ``payments.merchant_id`` is now
        NOT NULL (migration 0002) and that case is asserted as REJECTED in
        TestForeignKeys. A settlement without a payment linkage, and a
        reconciliation result without a run, remain valid.
        """
        seed_merchant(db_session)
        db_session.add(Payment(id="PAY-NOMERCH", merchant_id="MER-1", amount=100))
        db_session.add(
            Settlement(id="SET-NOPAY", payment_id=None, amount=100, currency="INR")
        )
        db_session.add(
            ReconciliationResult(
                id="REC-NORUN", case_id="CASE-1", payment_id="PAY-NOMERCH",
                merchant_id="MER-1", batch_id="B-2", reconciliation_run_id=None,
                payment_amount=100, expected_amount=100, actual_amount=100,
                difference=0, match_status="MATCHED", exception_type="EXACT_MATCH",
                currency="INR",
            )
        )
        db_session.flush()
        assert db_session.get(Settlement, "SET-NOPAY").payment_id is None

    def test_payment_without_a_merchant_is_rejected(self, db_session):
        """Phase 2: payments.merchant_id is NOT NULL (migration 0002)."""
        db_session.add(Payment(id="PAY-NOMERCH", merchant_id=None, amount=100))
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Uniqueness and idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestUniqueness:
    def test_duplicate_primary_key_is_rejected(self, db_session):
        seed_payment(db_session)
        db_session.add(Payment(id="PAY-1", merchant_id="MER-1", amount=1))
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_evidence_unique_per_exception_entity_relationship(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        for _ in range(2):
            db_session.add(
                Evidence(
                    id=f"EV-{_}",
                    exception_id="EXC-1",
                    entity_type="FEE",
                    entity_id="FEE-1",
                    relationship="CALCULATION_COMPONENT",
                    recorded_by="RECONCILIATION",
                )
            )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_evidence_allows_the_same_entity_with_a_different_relationship(
        self, db_session
    ):
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add_all(
            [
                Evidence(
                    id="EV-A", exception_id="EXC-1", entity_type="FEE",
                    entity_id="FEE-1", relationship="CALCULATION_COMPONENT",
                    recorded_by="RECONCILIATION",
                ),
                Evidence(
                    id="EV-B", exception_id="EXC-1", entity_type="FEE",
                    entity_id="FEE-1", relationship="CONFLICTING",
                    recorded_by="AGENT",
                ),
            ]
        )
        db_session.flush()
        assert db_session.query(Evidence).count() == 2

    def test_resolution_candidate_rank_is_unique_per_exception(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session, "RES-1", rank=1)
        db_session.add(
            Resolution(
                id="RES-2", exception_id="EXC-1", candidate_rank=1,
                resolution_type="NO_ACTION",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_root_cause_rank_is_unique_per_exception(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add_all(
            [
                RootCause(id="RC-1", exception_id="EXC-1", rank=1, cause_type="A"),
                RootCause(id="RC-2", exception_id="EXC-1", rank=1, cause_type="B"),
            ]
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_idempotency_key_must_be_unique(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add_all(
            [
                ResolutionAction(
                    id="ACT-1", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="same-key",
                ),
                ResolutionAction(
                    id="ACT-2", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="same-key",
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_provider_reference_is_unique_when_present(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add_all(
            [
                ResolutionAction(
                    id="ACT-1", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="k1", provider_reference="adj_1",
                ),
                ResolutionAction(
                    id="ACT-2", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="k2", provider_reference="adj_1",
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_multiple_null_provider_references_are_allowed(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add_all(
            [
                ResolutionAction(
                    id="ACT-1", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="k1", provider_reference=None,
                ),
                ResolutionAction(
                    id="ACT-2", resolution_id="RES-1", action_type="ADJUSTMENT",
                    idempotency_key="k2", provider_reference=None,
                ),
            ]
        )
        db_session.flush()
        assert db_session.query(ResolutionAction).count() == 2

    def test_one_pending_approval_per_resolution(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        db_session.add_all(
            [
                Approval(id="APR-1", resolution_id="RES-1", requested_by="SYSTEM"),
                Approval(id="APR-2", resolution_id="RES-1", requested_by="HUMAN"),
            ]
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_a_decided_approval_frees_the_slot(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        seed_resolution(db_session)
        first = Approval(
            id="APR-1", resolution_id="RES-1", requested_by="SYSTEM"
        )
        db_session.add(first)
        db_session.flush()
        first.transition_to("APPROVED")
        db_session.flush()

        db_session.add(
            Approval(id="APR-2", resolution_id="RES-1", requested_by="HUMAN")
        )
        db_session.flush()
        assert db_session.query(Approval).count() == 2

    def test_one_running_run_per_scope(self, db_session):
        seed_run(db_session, "RUN-1", scope_ref="B-1", status="RUNNING")
        db_session.add(
            ReconciliationRun(
                id="RUN-2", scope_type="BATCH", scope_ref="B-1", trigger="MANUAL",
                engine_version="1.0.0", status="RUNNING",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_finished_runs_do_not_block_a_new_run(self, db_session):
        seed_run(db_session, "RUN-1", scope_ref="B-1", status="COMPLETED")
        seed_run(db_session, "RUN-2", scope_ref="B-1", status="PENDING")
        db_session.flush()
        assert db_session.query(ReconciliationRun).count() == 2

    def test_run_key_is_unique_per_scope_and_trigger(self, db_session):
        first = seed_run(db_session, "RUN-1", scope_ref="B-1")
        first.run_key = "key-1"
        db_session.flush()
        db_session.add(
            ReconciliationRun(
                id="RUN-2", scope_type="BATCH", scope_ref="B-1", trigger="MANUAL",
                engine_version="1.0.0", run_key="key-1",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_exception_unique_per_case_and_batch(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add(
            FinancialException(
                id="EXC-2", case_id="CASE-1", payment_id="PAY-1", batch_id="B-1",
                expected_amount=1, actual_amount=1, difference=0,
                exception_type="FEE_DIFFERENCE", reconciliation_id="REC-2",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_reconciliation_result_unique_per_case_and_batch(self, db_session):
        seed_payment(db_session)
        for identifier in ("REC-1", "REC-2"):
            db_session.add(
                ReconciliationResult(
                    id=identifier, case_id="CASE-1", payment_id="PAY-1",
                    merchant_id="MER-1", batch_id="B-1",
                    payment_amount=100, expected_amount=100, actual_amount=100,
                    difference=0, match_status="MATCHED",
                    exception_type="EXACT_MATCH", currency="INR",
                )
            )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_case_embedding_is_one_to_one(self, db_session):
        seed_payment(db_session)
        seed_historical_case(db_session)
        db_session.add(
            CaseEmbedding(
                id="HC-1", embedding_json="[]", exception_type="X",
                resolution_type="Y", resolution_outcome="Z", payment_amount=1,
                difference=0, case_text="t", embedding_model="m",
                embedding_dimension=384,
            )
        )
        db_session.flush()
        db_session.add(
            CaseEmbedding(
                id="HC-1", embedding_json="[]", exception_type="X",
                resolution_type="Y", resolution_outcome="Z", payment_amount=1,
                difference=0, case_text="t", embedding_model="m",
                embedding_dimension=384,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Append-only behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendOnly:
    def test_ledger_entry_cannot_be_updated(self, db_session):
        seed_payment(db_session)
        entry = LedgerEntry(
            id="LED-1", payment_id="PAY-1", entry_type="PAYMENT", amount=100,
            currency="INR", direction="CREDIT", source="INGEST",
            occurred_at=datetime.now(timezone.utc),
        )
        db_session.add(entry)
        db_session.flush()

        entry.amount = 999
        with pytest.raises(ImmutableRecordError):
            db_session.flush()
        db_session.rollback()

    def test_ledger_entry_cannot_be_deleted(self, db_session):
        seed_payment(db_session)
        entry = LedgerEntry(
            id="LED-1", payment_id="PAY-1", entry_type="PAYMENT", amount=100,
            currency="INR", direction="CREDIT", source="INGEST",
            occurred_at=datetime.now(timezone.utc),
        )
        db_session.add(entry)
        db_session.flush()

        db_session.delete(entry)
        with pytest.raises(ImmutableRecordError):
            db_session.flush()
        db_session.rollback()

    def test_audit_event_cannot_be_updated(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        event = AuditEvent(
            id="AUD-1", exception_id="EXC-1", actor="system",
            actor_type="SYSTEM", action="EXCEPTION_DETECTED",
        )
        db_session.add(event)
        db_session.flush()

        event.action = "REWRITTEN"
        with pytest.raises(ImmutableRecordError):
            db_session.flush()
        db_session.rollback()

    def test_audit_event_cannot_be_deleted(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        event = AuditEvent(
            id="AUD-1", exception_id="EXC-1", actor="system",
            actor_type="SYSTEM", action="EXCEPTION_DETECTED",
        )
        db_session.add(event)
        db_session.flush()

        db_session.delete(event)
        with pytest.raises(ImmutableRecordError):
            db_session.flush()
        db_session.rollback()

    def test_feedback_cannot_be_updated_or_deleted(self, db_session):
        seed_payment(db_session)
        seed_exception(db_session)
        feedback = Feedback(
            id="FB-1", exception_id="EXC-1", feedback_type="OUTCOME",
            actor_type="HUMAN", reviewer="u-1",
        )
        db_session.add(feedback)
        db_session.flush()

        feedback.reviewer = "someone-else"
        with pytest.raises(ImmutableRecordError):
            db_session.flush()
        db_session.rollback()

    def test_corrections_are_new_rows_not_edits(self, db_session):
        """The sanctioned way to fix an audit record is a new row."""
        seed_payment(db_session)
        seed_exception(db_session)
        db_session.add(
            AuditEvent(
                id="AUD-1", exception_id="EXC-1", actor="system",
                actor_type="SYSTEM", action="EXCEPTION_DETECTED",
            )
        )
        db_session.flush()
        db_session.add(
            AuditEvent(
                id="AUD-2", exception_id="EXC-1", actor="system",
                actor_type="SYSTEM", action="EXCEPTION_DETECTED",
                correction_of="AUD-1",
            )
        )
        db_session.flush()
        assert db_session.query(AuditEvent).count() == 2
        assert db_session.get(AuditEvent, "AUD-2").correction_of == "AUD-1"

    def test_immutable_tables_are_importable_from_the_registry(self):
        import app.models as models

        assert models.ImmutableRecordError is ImmutableRecordError
