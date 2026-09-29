"""
Phase 1 tests: domain model foundation.

Covers CloseLoop 2.0 architecture section 5 (domain model) and the task's
section 14 requirements:

* every Phase 1 entity exists as a table;
* a valid instance of each can be created and persisted;
* the relationships of the domain map actually work in both directions;
* money is integer minor units and currency is always explicit;
* the CHECK-constraint value literals in the models match the enums in
  ``app.schemas.enums`` (drift guard).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import BigInteger, Float, Numeric

from app.database.database import Base

from app.models.adjustment import Adjustment
from app.models.approval import Approval
from app.models.audit_event import AuditEvent
from app.models.chargeback import Chargeback
from app.models.evidence import Evidence
from app.models.exception import FinancialException
from app.models.feedback import Feedback
from app.models.fee import Fee
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


# The complete Phase 1 schema. A table missing from here means an entity from
# architecture section 5 was never implemented; an extra one means something
# was added that Phase 1 did not approve.
PHASE1_TABLES = {
    # financial layer
    "merchants",
    "payments",
    "payment_attempts",
    "settlements",
    "settlement_lines",
    "refunds",
    "chargebacks",
    "fees",
    "taxes",
    "adjustments",
    "ledger_entries",
    # reconciliation
    "reconciliation_runs",
    "reconciliation_results",
    "reconciliation_evidence",
    # exception & investigation
    "exceptions",
    "evidence",
    "evidence_links",
    "root_causes",
    "model_predictions",
    "audit_events",
    "feedback",
    # resolution
    "resolutions",
    "resolution_actions",
    "approvals",
    # historical memory
    "historical_cases",
    "historical_resolutions",
    "case_embeddings",
}

# Columns that hold money and therefore must be integer minor units.
MONEY_COLUMN_NAMES = {
    "amount",
    "amount_paise",
    "expected_amount",
    "actual_amount",
    "difference",
    "payment_amount",
    "total_refunds",
    "total_fees",
    "total_taxes",
    "total_adjustments",
    "resolved_amount",
    "financial_exposure_paise",
    "difference_at_resolution",
}

# ─────────────────────────────────────────────────────────────────────────────
# Schema registration
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaRegistration:
    def test_every_phase1_table_is_registered(self):
        registered = set(Base.metadata.tables)
        assert PHASE1_TABLES <= registered, (
            f"missing tables: {sorted(PHASE1_TABLES - registered)}"
        )

    def test_importing_app_models_registers_the_whole_schema(self):
        """The registry promise: `import app.models` must register everything."""
        import app.models as models

        assert len(models.__all__) == 29
        for name in models.__all__:
            assert hasattr(models, name), name

    def test_phase1_entities_expose_expected_column_counts(self):
        # A cheap canary: if a model silently loses columns, this fails.
        assert len(Payment.__table__.columns) == 12
        assert len(Merchant.__table__.columns) == 9
        assert len(ReconciliationRun.__table__.columns) == 16
        assert len(FinancialException.__table__.columns) == 22


# ─────────────────────────────────────────────────────────────────────────────
# Creation and persistence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def merchant(db_session):
    m = Merchant(
        id="MER-1",
        name="Acme Retail",
        status="ACTIVE",
        category_code="5411",
        metadata_json={"tier": "gold"},
    )
    db_session.add(m)
    db_session.flush()
    return m


@pytest.fixture
def payment(db_session, merchant):
    p = Payment(
        id="PAY-1",
        merchant_id=merchant.id,
        order_id="ORD-1",
        amount=25000,  # 250.00 INR
        currency="INR",
        status="CAPTURED",
        method="card",
    )
    db_session.add(p)
    db_session.flush()
    return p


@pytest.fixture
def exception(db_session, payment):
    e = FinancialException(
        id="EXC-1",
        case_id="CASE-1",
        payment_id=payment.id,
        batch_id="B-1",
        expected_amount=25000,
        actual_amount=24500,
        difference=500,
        exception_type="FEE_DIFFERENCE",
        status="OPEN",
        reconciliation_id="REC-1",
        risk_category="LOW",
    )
    db_session.add(e)
    db_session.flush()
    return e


@pytest.fixture
def resolution(db_session, exception):
    r = Resolution(
        id="RES-1",
        exception_id=exception.id,
        candidate_rank=1,
        resolution_type="FEE_ADJUSTMENT",
        amount_paise=500,
        direction="CREDIT",
        confidence=0.92,
        financial_exposure_paise=500,
        evidence_ids=["EV-1"],
    )
    db_session.add(r)
    db_session.flush()
    return r


class TestCreationAndPersistence:
    def test_merchant_persists(self, db_session, merchant):
        stored = db_session.get(Merchant, "MER-1")
        assert stored.name == "Acme Retail"
        assert stored.metadata_json == {"tier": "gold"}

    def test_payment_persists_with_merchant(self, db_session, payment):
        stored = db_session.get(Payment, "PAY-1")
        assert stored.merchant_id == "MER-1"
        assert stored.amount == 25000
        assert stored.currency == "INR"
        assert stored.order_id == "ORD-1"
        assert stored.version == 1

    def test_payment_attempt_persists(self, db_session, payment):
        attempt = PaymentAttempt(
            id="ATT-1",
            payment_id=payment.id,
            attempt_no=1,
            kind="CAPTURE",
            amount=25000,
            currency="INR",
            status="CAPTURED",
        )
        db_session.add(attempt)
        db_session.flush()
        assert db_session.get(PaymentAttempt, "ATT-1").kind == "CAPTURE"

    def test_settlement_and_lines_persist(self, db_session, payment):
        settlement = Settlement(
            id="SET-1",
            payment_id=payment.id,
            merchant_id="MER-1",
            amount=24500,
            currency="INR",
            status="SETTLED",
            provider_batch_id="SETL-BATCH-1",
        )
        db_session.add(settlement)
        db_session.flush()

        db_session.add_all(
            [
                SettlementLine(
                    id="SL-1",
                    settlement_id="SET-1",
                    component_type="GROSS",
                    amount=25000,
                    currency="INR",
                    source_entity_type="PAYMENT",
                    source_entity_id=payment.id,
                ),
                SettlementLine(
                    id="SL-2",
                    settlement_id="SET-1",
                    component_type="FEE",
                    amount=-500,  # signed
                    currency="INR",
                    source_entity_type="FEE",
                    source_entity_id="FEE-1",
                ),
            ]
        )
        db_session.flush()

        lines = db_session.query(SettlementLine).filter_by(settlement_id="SET-1").all()
        assert sum(line.amount for line in lines) == 24500
        assert {line.component_type for line in lines} == {"GROSS", "FEE"}

    def test_refund_fee_tax_adjustment_persist(self, db_session, payment):
        db_session.add_all(
            [
                Refund(
                    id="REF-1",
                    payment_id=payment.id,
                    case_id="CASE-1",
                    merchant_id="MER-1",
                    amount=5000,
                    currency="INR",
                    status="PROCESSED",
                    reason="customer_request",
                ),
                Fee(
                    id="FEE-1",
                    payment_id=payment.id,
                    case_id="CASE-1",
                    merchant_id="MER-1",
                    amount=200,
                    currency="INR",
                    fee_type="PLATFORM",
                ),
                Tax(
                    id="TAX-1",
                    payment_id=payment.id,
                    case_id="CASE-1",
                    merchant_id="MER-1",
                    amount=1800,
                    currency="INR",
                    tax_type="GST",
                    jurisdiction="IN-MH",
                ),
                Adjustment(
                    id="ADJ-1",
                    payment_id=payment.id,
                    case_id="CASE-1",
                    merchant_id="MER-1",
                    amount=-1000,  # signed: a debit
                    currency="INR",
                    adjustment_type="DEBIT",
                    origin="PROVIDER",
                ),
            ]
        )
        db_session.flush()
        assert db_session.get(Refund, "REF-1").reason == "customer_request"
        assert db_session.get(Fee, "FEE-1").fee_type == "PLATFORM"
        assert db_session.get(Tax, "TAX-1").jurisdiction == "IN-MH"
        assert db_session.get(Adjustment, "ADJ-1").origin == "PROVIDER"

    def test_chargeback_persists(self, db_session, payment):
        chargeback = Chargeback(
            id="CB-1",
            payment_id=payment.id,
            merchant_id="MER-1",
            amount=25000,
            currency="INR",
            status="OPEN",
            reason_code="FRAUD",
        )
        db_session.add(chargeback)
        db_session.flush()
        stored = db_session.get(Chargeback, "CB-1")
        assert stored.status == "OPEN"
        assert stored.reason_code == "FRAUD"

    def test_ledger_entry_persists(self, db_session, payment):
        entry = LedgerEntry(
            id="LED-1",
            payment_id=payment.id,
            merchant_id="MER-1",
            entry_type="PAYMENT",
            amount=25000,
            currency="INR",
            direction="CREDIT",
            source="INGEST",
            source_entity_type="PAYMENT",
            source_entity_id=payment.id,
            # occurred_at is NOT NULL with no default on purpose: the ledger
            # records when the financial event happened, so the writer must say.
            occurred_at=datetime.now(timezone.utc),
            correlation_id="req-1",
        )
        db_session.add(entry)
        db_session.flush()
        assert db_session.get(LedgerEntry, "LED-1").direction == "CREDIT"

    def test_reconciliation_run_and_result_persist(self, db_session, payment):
        run = ReconciliationRun(
            id="RUN-1",
            scope_type="BATCH",
            scope_ref="BATCH-1",
            trigger="ON_INGEST",
            engine_version="1.0.0",
            counts={"checked": 1},
            status="RUNNING",
        )
        db_session.add(run)
        db_session.flush()

        result = ReconciliationResult(
            id="REC-1",
            case_id="CASE-1",
            payment_id=payment.id,
            merchant_id="MER-1",
            batch_id="B-1",
            reconciliation_run_id=run.id,
            payment_amount=25000,
            total_refunds=0,
            total_fees=500,
            total_taxes=0,
            total_adjustments=0,
            expected_amount=24500,
            actual_amount=24500,
            difference=0,
            match_status="EXCEPTION",
            exception_type="FEE_DIFFERENCE",
            reconciliation_status="PROCESSED",
            currency="INR",
        )
        db_session.add(result)
        db_session.flush()
        assert db_session.get(ReconciliationResult, "REC-1").reconciliation_run_id == "RUN-1"

    def test_exception_lifecycle_columns_persist(self, db_session, exception):
        stored = db_session.get(FinancialException, "EXC-1")
        assert stored.risk_category == "LOW"
        assert stored.status == "OPEN"
        assert stored.reopen_count == 0
        assert stored.version == 1
        assert stored.currency == "INR"

    def test_evidence_root_cause_and_prediction_persist(self, db_session, exception):
        db_session.add_all(
            [
                Evidence(
                    id="EV-1",
                    exception_id=exception.id,
                    case_id="CASE-1",
                    entity_type="FEE",
                    entity_id="FEE-1",
                    relationship="CALCULATION_COMPONENT",
                    payload={"expected": 500, "actual": 700},
                    confidence=0.9,
                    recorded_by="RECONCILIATION",
                ),
                RootCause(
                    id="RC-1",
                    exception_id=exception.id,
                    rank=1,
                    cause_type="FEE_DIFFERENCE",
                    confidence=0.8,
                    deterministic_basis="rule.fee.floor",
                    evidence_ids=["EV-1"],
                ),
                ModelPrediction(
                    id="MP-1",
                    exception_id=exception.id,
                    model_name="classifier",
                    model_version="1.0.0",
                    feature_schema_version="v1",
                    predicted_type="FEE_DIFFERENCE",
                    probabilities={"FEE_DIFFERENCE": 0.8},
                    confidence=0.8,
                    features_hash="abc123",
                ),
            ]
        )
        db_session.flush()
        assert db_session.get(Evidence, "EV-1").payload == {
            "expected": 500,
            "actual": 700,
        }
        assert db_session.get(RootCause, "RC-1").deterministic_basis == "rule.fee.floor"
        assert db_session.get(ModelPrediction, "MP-1").probabilities == {
            "FEE_DIFFERENCE": 0.8
        }

    def test_audit_event_persists(self, db_session, exception):
        event = AuditEvent(
            id="AUD-1",
            exception_id=exception.id,
            workflow_id="WF-1",
            actor="system:policy-engine",
            actor_type="SYSTEM",
            action="POLICY_DECIDED",
            correlation_id="req-1",
            evidence_ids=["EL-1"],
            policy_decision="AUTO",
            policy_version="2026.09.1",
            confidence=0.96,
            before_state={"status": "RESOLUTION_PROPOSED"},
            after_state={"status": "AUTO_APPROVED"},
            model_version="clf-1.2.0",
            llm_provider="none",
        )
        db_session.add(event)
        db_session.flush()
        stored = db_session.get(AuditEvent, "AUD-1")
        assert stored.after_state == {"status": "AUTO_APPROVED"}
        assert stored.actor_type == "SYSTEM"

    def test_resolution_action_and_approval_persist(self, db_session, resolution):
        action = ResolutionAction(
            id="ACT-1",
            resolution_id=resolution.id,
            action_type="ADJUSTMENT",
            provider_operation="create_adjustment",
            amount_paise=500,
            idempotency_key="idem-1",
        )
        approval = Approval(
            id="APR-1",
            resolution_id=resolution.id,
            requested_by="SYSTEM",
            required_role="REVIEWER",
            evidence_digest="sha256:abc",
        )
        db_session.add_all([action, approval])
        db_session.flush()
        assert db_session.get(ResolutionAction, "ACT-1").provider_operation == (
            "create_adjustment"
        )
        assert db_session.get(Approval, "APR-1").required_role == "REVIEWER"

    def test_feedback_persists(self, db_session, exception, resolution):
        feedback = Feedback(
            id="FB-1",
            exception_id=exception.id,
            resolution_id=resolution.id,
            feedback_type="APPROVAL",
            reviewer="u-123",
            reviewer_role="REVIEWER",
            actor_type="HUMAN",
            system_prediction="FEE_ADJUSTMENT",
            system_confidence=0.92,
            correction_details={"note": "confirmed"},
            evidence_reviewed=["EV-1"],
            model_version="clf-1.2.0",
            policy_version="2026.09.1",
        )
        db_session.add(feedback)
        db_session.flush()
        assert db_session.get(Feedback, "FB-1").actor_type == "HUMAN"

    def test_historical_case_and_embedding_persist(self, db_session, payment):
        case = HistoricalCaseRecord(
            id="HC-1",
            exception_id="EXC-1",
            payment_id=payment.id,
            merchant_id="MER-1",
            exception_type="FEE_DIFFERENCE",
            payment_amount=25000,
            expected_amount=24500,
            actual_amount=24000,
            difference=500,
            resolution_type="FEE_ADJUSTMENT",
            resolution_outcome="RESOLVED",
            resolution_origin="DETERMINISTIC",
            financial_exposure_paise=500,
            reconciliation_verified=True,
        )
        db_session.add(case)
        db_session.flush()

        embedding = CaseEmbedding(
            id="HC-1",
            embedding_json="[0.1, 0.2]",
            exception_type="FEE_DIFFERENCE",
            resolution_type="FEE_ADJUSTMENT",
            resolution_outcome="RESOLVED",
            payment_amount=25000,
            difference=500,
            case_text="FEE_DIFFERENCE 250.00 vs 245.00",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            embedding_template_version="v1",
        )
        db_session.add(embedding)
        db_session.flush()
        assert db_session.get(CaseEmbedding, "HC-1").embedding_template_version == "v1"


# ─────────────────────────────────────────────────────────────────────────────
# Relationships (task section 10)
# ─────────────────────────────────────────────────────────────────────────────


class TestRelationships:
    def test_merchant_to_payments(self, db_session, merchant, payment):
        db_session.add(
            Payment(id="PAY-2", merchant_id=merchant.id, amount=1000, currency="INR")
        )
        db_session.flush()
        payments = db_session.query(Payment).filter_by(merchant_id="MER-1").all()
        assert sorted(p.id for p in payments) == ["PAY-1", "PAY-2"]

    def test_payment_to_attempts(self, db_session, payment):
        payment.attempts.append(
            PaymentAttempt(
                id="ATT-1",
                attempt_no=1,
                kind="AUTH",
                amount=25000,
                currency="INR",
            )
        )
        payment.attempts.append(
            PaymentAttempt(
                id="ATT-2",
                attempt_no=2,
                kind="CAPTURE",
                amount=25000,
                currency="INR",
            )
        )
        db_session.flush()
        db_session.refresh(payment)
        assert [a.id for a in payment.attempts] == ["ATT-1", "ATT-2"]
        assert payment.attempts[1].payment.id == "PAY-1"

    def test_payment_to_refunds_and_chargebacks(self, db_session, payment):
        payment.refunds.append(
            Refund(id="REF-1", amount=1000, currency="INR", status="PROCESSED")
        )
        payment.chargebacks.append(
            Chargeback(id="CB-1", amount=25000, currency="INR", status="OPEN")
        )
        db_session.flush()
        db_session.refresh(payment)
        assert [r.id for r in payment.refunds] == ["REF-1"]
        assert [c.id for c in payment.chargebacks] == ["CB-1"]
        assert payment.refunds[0].payment.id == "PAY-1"
        assert payment.chargebacks[0].payment.id == "PAY-1"

    def test_settlement_to_lines(self, db_session, payment):
        settlement = Settlement(
            id="SET-1", payment_id=payment.id, amount=24500, currency="INR"
        )
        settlement.lines.append(
            SettlementLine(
                id="SL-1",
                component_type="GROSS",
                amount=25000,
                currency="INR",
            )
        )
        db_session.add(settlement)
        db_session.flush()
        db_session.refresh(settlement)
        assert settlement.lines[0].settlement.id == "SET-1"
        assert settlement.payment.id == "PAY-1"

    def test_reconciliation_run_to_results(self, db_session, payment):
        run = ReconciliationRun(
            id="RUN-1",
            scope_type="BATCH",
            scope_ref="B-1",
            trigger="MANUAL",
            engine_version="1.0.0",
        )
        run.results.append(
            ReconciliationResult(
                id="REC-1",
                case_id="CASE-1",
                payment_id=payment.id,
                merchant_id="MER-1",
                batch_id="B-1",
                payment_amount=25000,
                expected_amount=24500,
                actual_amount=24500,
                difference=0,
                match_status="MATCHED",
                exception_type="EXACT_MATCH",
                currency="INR",
            )
        )
        db_session.add(run)
        db_session.flush()
        db_session.refresh(run)
        assert [r.id for r in run.results] == ["REC-1"]
        assert run.results[0].run.id == "RUN-1"

    def test_exception_to_investigation_entities(self, db_session, exception, resolution):
        db_session.add_all(
            [
                Evidence(
                    id="EV-1",
                    exception_id=exception.id,
                    entity_type="PAYMENT",
                    entity_id="PAY-1",
                    relationship="PRIMARY",
                    recorded_by="RECONCILIATION",
                ),
                RootCause(id="RC-1", exception_id=exception.id, cause_type="FEE_DIFFERENCE"),
                ModelPrediction(
                    id="MP-1",
                    exception_id=exception.id,
                    model_name="classifier",
                    model_version="1.0.0",
                    feature_schema_version="v1",
                    predicted_type="FEE_DIFFERENCE",
                ),
                AuditEvent(
                    id="AUD-1",
                    exception_id=exception.id,
                    actor="system:reconciliation",
                    actor_type="SYSTEM",
                    action="EXCEPTION_DETECTED",
                ),
                Feedback(
                    id="FB-1",
                    exception_id=exception.id,
                    feedback_type="OUTCOME",
                    actor_type="SYSTEM",
                ),
                HistoricalCaseRecord(
                    id="HC-1",
                    exception_id=exception.id,
                    payment_id="PAY-1",
                    exception_type="FEE_DIFFERENCE",
                    payment_amount=25000,
                    expected_amount=24500,
                    actual_amount=24000,
                    difference=500,
                    resolution_type="FEE_ADJUSTMENT",
                    resolution_outcome="RESOLVED",
                ),
            ]
        )
        db_session.flush()
        db_session.refresh(exception)

        assert [e.id for e in exception.evidence] == ["EV-1"]
        assert [r.id for r in exception.root_causes] == ["RC-1"]
        assert [r.id for r in exception.resolutions] == ["RES-1"]
        assert [p.id for p in exception.model_predictions] == ["MP-1"]
        assert [a.id for a in exception.audit_events] == ["AUD-1"]
        assert [f.id for f in exception.feedback] == ["FB-1"]
        assert [h.id for h in exception.historical_cases] == ["HC-1"]

    def test_resolution_to_actions_and_approvals(self, db_session, resolution):
        resolution.actions.append(
            ResolutionAction(
                id="ACT-1",
                action_type="ADJUSTMENT",
                idempotency_key="idem-1",
            )
        )
        resolution.approvals.append(
            Approval(id="APR-1", requested_by="SYSTEM")
        )
        db_session.flush()
        db_session.refresh(resolution)
        assert [a.id for a in resolution.actions] == ["ACT-1"]
        assert resolution.actions[0].resolution.id == "RES-1"
        assert [a.id for a in resolution.approvals] == ["APR-1"]

    def test_resolution_action_to_adjustment(self, db_session, resolution):
        action = ResolutionAction(
            id="ACT-1",
            resolution_id=resolution.id,
            action_type="ADJUSTMENT",
            idempotency_key="idem-1",
        )
        action.adjustments.append(
            Adjustment(
                id="ADJ-1",
                payment_id="PAY-1",
                amount=500,
                currency="INR",
                adjustment_type="CREDIT",
                origin="CLOSELOOP_EXECUTION",
            )
        )
        db_session.add(action)
        db_session.flush()
        db_session.refresh(action)
        assert [a.id for a in action.adjustments] == ["ADJ-1"]
        assert action.adjustments[0].resolution_action.id == "ACT-1"
        assert action.adjustments[0].origin == "CLOSELOOP_EXECUTION"

    def test_historical_case_to_embedding(self, db_session, payment):
        case = HistoricalCaseRecord(
            id="HC-1",
            exception_id="EXC-1",
            payment_id=payment.id,
            exception_type="FEE_DIFFERENCE",
            payment_amount=25000,
            expected_amount=24500,
            actual_amount=24000,
            difference=500,
            resolution_type="FEE_ADJUSTMENT",
            resolution_outcome="RESOLVED",
        )
        db_session.add(case)
        db_session.flush()
        db_session.add(
            CaseEmbedding(
                id="HC-1",
                embedding_json="[0.1]",
                exception_type="FEE_DIFFERENCE",
                resolution_type="FEE_ADJUSTMENT",
                resolution_outcome="RESOLVED",
                payment_amount=25000,
                difference=500,
                case_text="x",
                embedding_model="m",
                embedding_dimension=384,
            )
        )
        db_session.flush()
        # Shared primary key expresses the 1:1 relationship, and the FK now
        # enforces it.
        assert db_session.get(CaseEmbedding, case.id) is not None

    def test_adjustment_reversal_self_reference(self, db_session, payment):
        original = Adjustment(
            id="ADJ-1",
            payment_id=payment.id,
            amount=500,
            currency="INR",
            adjustment_type="CREDIT",
        )
        db_session.add(original)
        db_session.flush()

        reversal = Adjustment(
            id="ADJ-2",
            payment_id=payment.id,
            amount=-500,
            currency="INR",
            adjustment_type="CORRECTION",
        )
        db_session.add(reversal)
        db_session.flush()

        # The reversing row must exist before the link is written - the FK
        # enforces the ordering, which is the point of having it.
        original.reversed_by_adjustment_id = reversal.id
        db_session.flush()
        assert db_session.get(Adjustment, "ADJ-1").reversed_by_adjustment_id == "ADJ-2"

    def test_audit_event_correction_self_reference(self, db_session, exception):
        original = AuditEvent(
            id="AUD-1",
            exception_id=exception.id,
            actor="system",
            actor_type="SYSTEM",
            action="EXCEPTION_DETECTED",
        )
        db_session.add(original)
        db_session.flush()
        correction = AuditEvent(
            id="AUD-2",
            exception_id=exception.id,
            actor="system",
            actor_type="SYSTEM",
            action="EXCEPTION_DETECTED",
            correction_of="AUD-1",
        )
        db_session.add(correction)
        db_session.flush()
        assert db_session.get(AuditEvent, "AUD-2").correction_of == "AUD-1"


# ─────────────────────────────────────────────────────────────────────────────
# Money representation (task section 5)
# ─────────────────────────────────────────────────────────────────────────────


class TestMoneyRepresentation:
    def test_rupees_are_stored_as_integer_minor_units(self, db_session, payment):
        """The invariant: INR 250.00 is 25000 paise, an int, never a float."""
        stored = db_session.get(Payment, "PAY-1")
        assert stored.amount == 25000
        assert isinstance(stored.amount, int)
        assert not isinstance(stored.amount, float)

    def test_every_money_column_is_an_integer_type(self):
        offenders = []
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if column.name not in MONEY_COLUMN_NAMES:
                    continue
                if not isinstance(column.type, BigInteger):
                    offenders.append(
                        f"{table.name}.{column.name} is {type(column.type).__name__}"
                    )
        assert not offenders, f"non-integer money columns: {offenders}"

    def test_no_money_column_is_a_float_or_decimal_type(self):
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if column.name not in MONEY_COLUMN_NAMES:
                    continue
                assert not isinstance(column.type, (Float, Numeric)), (
                    f"{table.name}.{column.name} must not be float/decimal"
                )

    def test_fractional_columns_are_scores_never_money(self):
        """Non-integer numerics must be named scores, and never a money name."""
        SCORE_COLUMNS = {
            "confidence",
            "system_confidence",
            "exception_type_confidence",
            "evidence_coverage",
        }
        fractional = {
            (table.name, column.name)
            for table in Base.metadata.tables.values()
            for column in table.columns
            if isinstance(column.type, (Numeric, Float))
        }
        assert fractional
        unexpected = {(t, c) for t, c in fractional if c not in SCORE_COLUMNS}
        assert not unexpected, f"unexpected fractional columns: {sorted(unexpected)}"
        assert not (fractional & {
            (t, c) for t in Base.metadata.tables for c in MONEY_COLUMN_NAMES
        })

    def test_signed_columns_accept_negative_minor_units(self, db_session, payment):
        db_session.add(
            Adjustment(
                id="ADJ-NEG",
                payment_id=payment.id,
                amount=-25000,
                currency="INR",
                adjustment_type="DEBIT",
            )
        )
        db_session.flush()
        assert db_session.get(Adjustment, "ADJ-NEG").amount == -25000

    def test_large_amounts_fit_in_bigint(self, db_session, payment):
        huge = 10**15  # INR 10,000,000,000,000.00 in paise
        db_session.add(
            Payment(id="PAY-BIG", merchant_id="MER-1", amount=huge, currency="INR")
        )
        db_session.flush()
        assert db_session.get(Payment, "PAY-BIG").amount == huge

    def test_paise_round_trip_documented_convention(self):
        """Guard the convention itself: 2 decimal places, integer paise."""
        def to_paise(rupees: str) -> int:
            whole, _, fraction = rupees.partition(".")
            fraction = (fraction + "00")[:2]
            return int(whole) * 100 + int(fraction)

        assert to_paise("250.00") == 25000
        assert to_paise("10000.00") == 1000000
        assert to_paise("0.01") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Currency (task section 5 / architecture section 6)
# ─────────────────────────────────────────────────────────────────────────────


class TestCurrency:
    EXPECTED = {"payments", "settlements", "refunds", "fees", "taxes", "adjustments",
                "chargebacks", "payment_attempts", "settlement_lines", "ledger_entries",
                "exceptions", "reconciliation_results", "historical_cases"}

    def test_every_amount_bearing_table_declares_currency(self):
        missing = [
            name for name in sorted(self.EXPECTED)
            if "currency" not in Base.metadata.tables[name].columns
        ]
        assert not missing, f"tables without an explicit currency: {missing}"

    def test_currency_is_not_nullable_and_has_a_default(self):
        for name in sorted(self.EXPECTED):
            column = Base.metadata.tables[name].columns["currency"]
            assert column.nullable is False, f"{name}.currency must be NOT NULL"
            assert column.server_default is not None, (
                f"{name}.currency needs a server default for ALTER TABLE safety"
            )
            assert column.type.length == 3, f"{name}.currency must be a 3-char code"

    def test_currency_is_explicit_on_persisted_rows(self, db_session, payment):
        assert db_session.get(Payment, "PAY-1").currency == "INR"


# ─────────────────────────────────────────────────────────────────────────────
# Enum alignment (drift guard)
# ─────────────────────────────────────────────────────────────────────────────


class TestEnumAlignment:
    """The CHECK constraints use literal value tuples so the model layer does not
    import app.schemas. These tests fail loudly if the two drift apart."""

    def test_currency_values_match_the_enum(self):
        from app.models.types import CURRENCY_VALUES
        from app.schemas.enums import Currency

        assert set(CURRENCY_VALUES) == {member.value for member in Currency}

    @pytest.mark.parametrize(
        "module_constant,enum_class",
        [
            ("app.models.merchant:MERCHANT_STATUS_VALUES", "MerchantStatus"),
            ("app.models.payment:PAYMENT_STATUS_VALUES", "PaymentStatus"),
            ("app.models.payment_attempt:PAYMENT_ATTEMPT_KIND_VALUES", "PaymentAttemptKind"),
            ("app.models.payment_attempt:PAYMENT_ATTEMPT_STATUS_VALUES", "PaymentAttemptStatus"),
            ("app.models.settlement:SETTLEMENT_STATUS_VALUES", "SettlementStatus"),
            ("app.models.settlement_line:SETTLEMENT_COMPONENT_VALUES", "SettlementComponentType"),
            ("app.models.refund:REFUND_STATUS_VALUES", "RefundStatus"),
            ("app.models.chargeback:CHARGEBACK_STATUS_VALUES", "ChargebackStatus"),
            ("app.models.fee:FEE_TYPE_VALUES", "FeeType"),
            ("app.models.tax:TAX_TYPE_VALUES", "TaxType"),
            ("app.models.adjustment:ADJUSTMENT_TYPE_VALUES", "AdjustmentType"),
            ("app.models.adjustment:ADJUSTMENT_ORIGIN_VALUES", "AdjustmentOrigin"),
            ("app.models.ledger_entry:LEDGER_ENTRY_TYPE_VALUES", "LedgerEntryType"),
            ("app.models.ledger_entry:LEDGER_DIRECTION_VALUES", "LedgerDirection"),
            ("app.models.ledger_entry:LEDGER_SOURCE_VALUES", "LedgerSource"),
            ("app.models.reconciliation_run:RECONCILIATION_RUN_STATUS_VALUES", "ReconciliationRunStatus"),
            ("app.models.reconciliation_run:RECONCILIATION_SCOPE_TYPE_VALUES", "ReconciliationScopeType"),
            ("app.models.reconciliation_run:RECONCILIATION_TRIGGER_VALUES", "ReconciliationTrigger"),
            ("app.models.reconciliation:MATCH_STATUS_VALUES", "MatchStatus"),
            ("app.models.reconciliation:RECONCILIATION_RESULT_STATUS_VALUES", "ReconciliationStatus"),
            ("app.models.reconciliation:EXCEPTION_TYPE_VALUES", "ExceptionType"),
            ("app.models.evidence:EVIDENCE_RELATIONSHIP_VALUES", "EvidenceRelationship"),
            ("app.models.evidence:EVIDENCE_RECORDED_BY_VALUES", "EvidenceRecordedBy"),
            ("app.models.resolution:RESOLUTION_STATUS_VALUES", "ResolutionStatus"),
            ("app.models.resolution:POLICY_DECISION_VALUES", "PolicyDecision"),
            ("app.models.resolution:RESOLUTION_RISK_VALUES", "RiskCategory"),
            ("app.models.resolution_action:RESOLUTION_ACTION_STATUS_VALUES", "ResolutionActionStatus"),
            ("app.models.resolution_action:RESOLUTION_ACTION_TYPE_VALUES", "ResolutionActionType"),
            ("app.models.approval:APPROVAL_DECISION_VALUES", "ApprovalDecision"),
            ("app.models.approval:APPROVAL_REQUESTER_VALUES", "ApprovalRequester"),
            ("app.models.audit_event:ACTOR_TYPE_VALUES", "ActorType"),
            ("app.models.audit_event:AUDIT_POLICY_DECISION_VALUES", "PolicyDecision"),
            ("app.models.feedback:FEEDBACK_TYPE_VALUES", "FeedbackType"),
            ("app.models.feedback:FEEDBACK_ACTOR_TYPE_VALUES", "ActorType"),
        ],
    )
    def test_model_literals_match_enum(self, module_constant, enum_class):
        import importlib

        module_name, constant_name = module_constant.split(":")
        module = importlib.import_module(module_name)
        enums = importlib.import_module("app.schemas.enums")
        enum = getattr(enums, enum_class)

        assert set(getattr(module, constant_name)) == {
            member.value for member in enum
        }, f"{module_constant} drifted from {enum_class}"

    def test_exception_status_literals_cover_the_lifecycle_enum(self):
        from app.models.exception import EXCEPTION_STATUS_VALUES
        from app.schemas.enums import ExceptionLifecycleStatus

        assert {member.value for member in ExceptionLifecycleStatus} <= set(
            EXCEPTION_STATUS_VALUES
        )

    def test_exception_status_legacy_values_survive(self):
        from app.models.exception import EXCEPTION_STATUS_VALUES

        for legacy in ("OPEN", "MATCHED", "RESOLVED"):
            assert legacy in EXCEPTION_STATUS_VALUES

    def test_state_machine_states_are_accepted_by_the_check_constraints(self):
        """A state machine that produced a value the CHECK rejects would be a bug."""
        from app.domain import state_machines as sm
        from app.models.exception import EXCEPTION_STATUS_VALUES
        from app.models.payment import PAYMENT_STATUS_VALUES
        from app.models.reconciliation_run import RECONCILIATION_RUN_STATUS_VALUES
        from app.models.resolution import RESOLUTION_STATUS_VALUES
        from app.models.resolution_action import RESOLUTION_ACTION_STATUS_VALUES
        from app.models.approval import APPROVAL_DECISION_VALUES

        pairs = {
            sm.PAYMENT: PAYMENT_STATUS_VALUES,
            sm.RECONCILIATION_RUN: RECONCILIATION_RUN_STATUS_VALUES,
            sm.EXCEPTION: EXCEPTION_STATUS_VALUES,
            sm.RESOLUTION: RESOLUTION_STATUS_VALUES,
            sm.RESOLUTION_ACTION: RESOLUTION_ACTION_STATUS_VALUES,
            sm.APPROVAL: APPROVAL_DECISION_VALUES,
        }
        for machine, allowed in pairs.items():
            unrepresentable = sm.known_states(machine) - set(allowed)
            assert not unrepresentable, (
                f"{machine} can produce states the CHECK constraint rejects: "
                f"{sorted(unrepresentable)}"
            )
