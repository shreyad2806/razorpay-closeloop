"""
Phase 1 tests: state machines.

Covers CloseLoop 2.0 architecture section 7. Every machine gets:

* its documented valid transitions asserted as passing;
* a set of invalid transitions asserted as *rejected*, including the ones the
  task calls out explicitly (``COMPLETED -> RUNNING``, ``COMPLETED -> PENDING``);
* the guarantee that a rejected transition leaves the entity's status unchanged.

The model-level tests drive the same tables through
``Model.transition_to`` so the machines cannot quietly drift away from the
persisted entities.
"""

from __future__ import annotations

import pytest

from app.domain import state_machines as sm
from app.domain.state_machines import InvalidStateTransitionError
from app.models.approval import Approval
from app.models.exception import FinancialException
from app.models.payment import Payment
from app.models.reconciliation_run import ReconciliationRun
from app.models.resolution import Resolution
from app.models.resolution_action import ResolutionAction
from app.schemas.enums import (
    ApprovalDecision,
    PaymentStatus,
    ReconciliationRunStatus,
    ResolutionActionStatus,
    ResolutionStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry / helper tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    """The registry must expose exactly the six section 7 machines."""

    EXPECTED = {
        "PAYMENT",
        "RECONCILIATION_RUN",
        "EXCEPTION",
        "RESOLUTION",
        "RESOLUTION_ACTION",
        "APPROVAL",
    }

    def test_all_section_7_machines_registered(self):
        assert sm.machine_names() == frozenset(self.EXPECTED)

    def test_unknown_machine_is_rejected(self):
        with pytest.raises(KeyError):
            sm.known_states("NOT_A_MACHINE")

    def test_unknown_state_in_machine_is_rejected(self):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(sm.APPROVAL, "PENDING", "TELEPORTED")

    def test_unknown_current_state_has_no_transitions(self):
        assert sm.allowed_transitions(sm.APPROVAL, "TELEPORTED") == frozenset()

    def test_enum_members_are_accepted_as_statuses(self):
        assert sm.is_valid_transition(
            sm.RECONCILIATION_RUN,
            ReconciliationRunStatus.PENDING,
            ReconciliationRunStatus.RUNNING,
        )

    def test_next_status_returns_the_new_status(self):
        assert (
            sm.next_status(sm.APPROVAL, "PENDING", ApprovalDecision.APPROVED)
            is ApprovalDecision.APPROVED
        )

    def test_error_message_names_the_machine_and_states(self):
        error = InvalidStateTransitionError("APPROVAL", "APPROVED", "PENDING")
        assert "APPROVAL" in str(error)
        assert "APPROVED" in str(error) and "PENDING" in str(error)
        assert "terminal" in str(error)


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationRun (section 7.2)
# ─────────────────────────────────────────────────────────────────────────────


class TestReconciliationRunMachine:
    MACHINE = sm.RECONCILIATION_RUN

    @pytest.mark.parametrize(
        "current,new",
        [
            ("PENDING", "RUNNING"),
            ("RUNNING", "COMPLETED"),
            ("RUNNING", "FAILED"),
            ("RUNNING", "CANCELLED"),
            ("PENDING", "CANCELLED"),
        ],
    )
    def test_valid_transitions_pass(self, current, new):
        assert sm.is_valid_transition(self.MACHINE, current, new)
        assert sm.assert_transition(self.MACHINE, current, new) == new

    @pytest.mark.parametrize(
        "current,new",
        [
            ("COMPLETED", "RUNNING"),
            ("COMPLETED", "PENDING"),
            ("COMPLETED", "FAILED"),
            ("FAILED", "RUNNING"),
            ("CANCELLED", "RUNNING"),
            ("PENDING", "COMPLETED"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        assert not sm.is_valid_transition(self.MACHINE, current, new)
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_terminal_states(self):
        assert sm.is_terminal(self.MACHINE, "COMPLETED")
        assert sm.is_terminal(self.MACHINE, "FAILED")
        assert sm.is_terminal(self.MACHINE, "CANCELLED")
        assert not sm.is_terminal(self.MACHINE, "PENDING")
        assert not sm.is_terminal(self.MACHINE, "RUNNING")

    def test_model_transition_to_applies_and_rejects(self):
        # ``status`` is passed explicitly: SQLAlchemy column defaults are applied
        # by the database layer at INSERT time, so they are not visible on an
        # unflushed instance. Their values are asserted after a flush in
        # TestPersistedDefaults below.
        run = ReconciliationRun(
            id="RUN-1",
            scope_type="BATCH",
            scope_ref="B-1",
            trigger="MANUAL",
            engine_version="1.0",
            status="PENDING",
        )
        assert run.status == "PENDING"
        run.transition_to(ReconciliationRunStatus.RUNNING)
        assert run.status == "RUNNING"
        run.transition_to("COMPLETED")
        assert run.status == "COMPLETED"

        with pytest.raises(InvalidStateTransitionError):
            run.transition_to("RUNNING")
        # A rejected transition must not mutate the record.
        assert run.status == "COMPLETED"
        assert run.can_transition_to("RUNNING") is False


# ─────────────────────────────────────────────────────────────────────────────
# Payment (section 7.1)
# ─────────────────────────────────────────────────────────────────────────────


class TestPaymentMachine:
    MACHINE = sm.PAYMENT

    @pytest.mark.parametrize(
        "current,new",
        [
            ("CREATED", "AUTHORIZED"),
            ("AUTHORIZED", "CAPTURED"),
            ("CAPTURED", "SETTLED"),
            ("CAPTURED", "PARTIALLY_REFUNDED"),
            ("PARTIALLY_REFUNDED", "REFUNDED"),
            ("CAPTURED", "CHARGEBACK"),
            ("CAPTURED", "FAILED"),
            ("PENDING", "AUTHORIZED"),
        ],
    )
    def test_valid_transitions_pass(self, current, new):
        assert sm.is_valid_transition(self.MACHINE, current, new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("REFUNDED", "CAPTURED"),
            ("FAILED", "CAPTURED"),
            ("CHARGEBACK", "CAPTURED"),
            ("CREATED", "SETTLED"),
            ("AUTHORIZED", "REFUNDED"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_every_non_terminal_state_can_fail(self):
        """Section 7.1 documents ``any -> FAILED``."""
        for state in sm.known_states(self.MACHINE):
            if sm.is_terminal(self.MACHINE, state):
                continue
            assert sm.is_valid_transition(self.MACHINE, state, "FAILED"), state

    def test_model_transition_to(self):
        payment = Payment(
            id="PAY-1", merchant_id="MER-1", amount=25000, status="CREATED"
        )
        assert payment.status == "CREATED"
        payment.transition_to(PaymentStatus.AUTHORIZED)
        payment.transition_to("CAPTURED")
        assert payment.status == "CAPTURED"
        with pytest.raises(InvalidStateTransitionError):
            payment.transition_to("AUTHORIZED")
        assert payment.status == "CAPTURED"


# ─────────────────────────────────────────────────────────────────────────────
# Exception (section 7.3)
# ─────────────────────────────────────────────────────────────────────────────


class TestExceptionMachine:
    MACHINE = sm.EXCEPTION

    @pytest.mark.parametrize(
        "current,new",
        [
            ("OPEN", "INVESTIGATING"),
            ("DETECTED", "INVESTIGATING"),
            ("INVESTIGATING", "ANALYZED"),
            ("ANALYZED", "RESOLUTION_PROPOSED"),
            ("RESOLUTION_PROPOSED", "AUTO_APPROVED"),
            ("RESOLUTION_PROPOSED", "HUMAN_REVIEW"),
            ("RESOLUTION_PROPOSED", "UNRESOLVED"),
            ("AUTO_APPROVED", "EXECUTING"),
            ("HUMAN_REVIEW", "APPROVED"),
            ("HUMAN_REVIEW", "REJECTED"),
            ("APPROVED", "EXECUTING"),
            ("EXECUTING", "VERIFYING"),
            ("VERIFYING", "RECONCILING"),
            ("RECONCILING", "CLOSED"),
            ("EXECUTING", "FAILED"),
            ("VERIFYING", "FAILED"),
            ("RECONCILING", "FAILED"),
            ("FAILED", "ROLLED_BACK"),
            ("ROLLED_BACK", "HUMAN_REVIEW"),
            ("UNRESOLVED", "HUMAN_REVIEW"),
            ("ESCALATED", "INVESTIGATING"),
        ],
    )
    def test_valid_transitions_pass(self, current, new):
        assert sm.is_valid_transition(self.MACHINE, current, new)

    def test_closed_is_reachable_only_from_reconciling(self):
        """Section 7.3: nothing other than re-reconciliation may close an exception."""
        for state in sm.known_states(self.MACHINE):
            if state == "RECONCILING":
                continue
            assert not sm.is_valid_transition(
                self.MACHINE, state, "CLOSED"
            ), f"{state} must not reach CLOSED"

    @pytest.mark.parametrize(
        "current,new",
        [
            ("CLOSED", "INVESTIGATING"),
            ("CLOSED", "HUMAN_REVIEW"),
            ("CLOSED", "RECONCILING"),
            ("DETECTED", "CLOSED"),
            ("MATCHED", "INVESTIGATING"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_every_pre_closed_state_can_escalate(self):
        """Section 7.3: any pre-CLOSED state -> ESCALATED.

        ``ESCALATED`` itself is excluded - it is already escalated, and a
        self-transition would be meaningless (see TestMachineInvariants).
        """
        for state in sm.known_states(self.MACHINE):
            if state in {"CLOSED", "MATCHED", "RESOLVED", "ESCALATED"}:
                continue
            assert sm.is_valid_transition(self.MACHINE, state, "ESCALATED"), state

    def test_legacy_states_are_known(self):
        """Pre-2.0 values must remain representable."""
        for legacy in ("OPEN", "MATCHED", "RESOLVED"):
            assert sm.is_known_state(self.MACHINE, legacy)

    def test_model_transition_to(self):
        exc = FinancialException(
            id="EXC-1",
            case_id="CASE-1",
            payment_id="PAY-1",
            batch_id="B-1",
            expected_amount=1000,
            actual_amount=900,
            difference=100,
            exception_type="FEE_DIFFERENCE",
            reconciliation_id="REC-1",
            status="OPEN",
        )
        assert exc.status == "OPEN"
        exc.transition_to("INVESTIGATING")
        exc.transition_to("ANALYZED")
        assert exc.status == "ANALYZED"
        with pytest.raises(InvalidStateTransitionError):
            exc.transition_to("CLOSED")
        assert exc.status == "ANALYZED"


# ─────────────────────────────────────────────────────────────────────────────
# Resolution (section 7.4)
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionMachine:
    MACHINE = sm.RESOLUTION

    @pytest.mark.parametrize(
        "current,new",
        [
            ("DRAFT", "PROPOSED"),
            ("PROPOSED", "POLICY_EVALUATED"),
            ("POLICY_EVALUATED", "AUTO_APPROVED"),
            ("POLICY_EVALUATED", "HUMAN_REVIEW"),
            ("POLICY_EVALUATED", "BLOCKED"),
            ("AUTO_APPROVED", "EXECUTING"),
            ("HUMAN_REVIEW", "APPROVED"),
            ("APPROVED", "EXECUTING"),
            ("EXECUTING", "EXECUTED"),
            ("EXECUTED", "VERIFIED"),
            ("EXECUTING", "FAILED"),
            ("EXECUTED", "FAILED"),
            ("FAILED", "ROLLBACK_PENDING"),
            ("ROLLBACK_PENDING", "ROLLED_BACK"),
            ("ROLLBACK_PENDING", "ROLLBACK_FAILED"),
            ("BLOCKED", "WITHDRAWN"),
        ],
    )
    def test_valid_transitions_pass(self, current, new):
        assert sm.is_valid_transition(self.MACHINE, current, new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("DRAFT", "EXECUTING"),
            ("BLOCKED", "EXECUTING"),
            ("BLOCKED", "HUMAN_REVIEW"),
            ("PROPOSED", "AUTO_APPROVED"),
            ("VERIFIED", "EXECUTING"),
            ("WITHDRAWN", "PROPOSED"),
            ("ROLLED_BACK", "EXECUTING"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_model_transition_to(self):
        resolution = Resolution(
            id="RES-1",
            exception_id="EXC-1",
            candidate_rank=1,
            resolution_type="FEE_ADJUSTMENT",
            status="DRAFT",
        )
        assert resolution.status == "DRAFT"
        resolution.transition_to(ResolutionStatus.PROPOSED)
        resolution.transition_to("POLICY_EVALUATED")
        resolution.transition_to("HUMAN_REVIEW")
        resolution.transition_to("APPROVED")
        resolution.transition_to("EXECUTING")
        assert resolution.status == "EXECUTING"
        with pytest.raises(InvalidStateTransitionError):
            resolution.transition_to("DRAFT")
        assert resolution.status == "EXECUTING"


# ─────────────────────────────────────────────────────────────────────────────
# ResolutionAction (section 7.5)
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionActionMachine:
    MACHINE = sm.RESOLUTION_ACTION

    @pytest.mark.parametrize(
        "current,new",
        [
            ("PROPOSED", "VALIDATED"),
            ("VALIDATED", "APPROVED"),
            ("APPROVED", "EXECUTING"),
            ("EXECUTING", "EXECUTED"),
            ("EXECUTING", "FAILED"),
            ("EXECUTED", "VERIFIED"),
            ("FAILED", "ROLLBACK_PENDING"),
            ("ROLLBACK_PENDING", "ROLLED_BACK"),
            ("ROLLBACK_PENDING", "ROLLBACK_FAILED"),
        ],
    )
    def test_valid_transitions_pass(self, current, new):
        assert sm.is_valid_transition(self.MACHINE, current, new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("PROPOSED", "EXECUTING"),
            ("VALIDATED", "EXECUTING"),
            ("VERIFIED", "FAILED"),
            ("ROLLED_BACK", "EXECUTING"),
            ("EXECUTED", "EXECUTING"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_model_transition_to(self):
        action = ResolutionAction(
            id="ACT-1",
            resolution_id="RES-1",
            action_type="ADJUSTMENT",
            idempotency_key="idem-1",
            status="PROPOSED",
        )
        assert action.status == "PROPOSED"
        action.transition_to(ResolutionActionStatus.VALIDATED)
        action.transition_to("APPROVED")
        action.transition_to("EXECUTING")
        action.transition_to("EXECUTED")
        action.transition_to("VERIFIED")
        assert action.status == "VERIFIED"
        with pytest.raises(InvalidStateTransitionError):
            action.transition_to("FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Approval (section 7.6)
# ─────────────────────────────────────────────────────────────────────────────


class TestApprovalMachine:
    MACHINE = sm.APPROVAL

    @pytest.mark.parametrize(
        "new", ["APPROVED", "REJECTED", "EXPIRED", "REVOKED"]
    )
    def test_pending_can_reach_every_decision(self, new):
        assert sm.is_valid_transition(self.MACHINE, "PENDING", new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("APPROVED", "PENDING"),
            ("APPROVED", "REJECTED"),
            ("REJECTED", "APPROVED"),
            ("EXPIRED", "APPROVED"),
            ("REVOKED", "APPROVED"),
        ],
    )
    def test_invalid_transitions_fail(self, current, new):
        with pytest.raises(InvalidStateTransitionError):
            sm.assert_transition(self.MACHINE, current, new)

    def test_every_decision_is_terminal(self):
        for decision in ("APPROVED", "REJECTED", "EXPIRED", "REVOKED"):
            assert sm.is_terminal(self.MACHINE, decision)

    def test_model_transition_to(self):
        approval = Approval(
            id="APR-1",
            resolution_id="RES-1",
            requested_by="SYSTEM",
            decision="PENDING",
        )
        assert approval.decision == "PENDING"
        approval.transition_to("APPROVED")
        assert approval.decision == "APPROVED"
        with pytest.raises(InvalidStateTransitionError):
            approval.transition_to("REVOKED")
        assert approval.decision == "APPROVED"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-machine invariants
# ─────────────────────────────────────────────────────────────────────────────


class TestMachineInvariants:
    def test_no_machine_transitions_to_itself(self):
        for machine in sm.machine_names():
            for state in sm.known_states(machine):
                assert not sm.is_valid_transition(machine, state, state), (
                    f"{machine}.{state} must not self-transition"
                )

    def test_every_transition_target_is_a_known_state(self):
        for machine, table in sm.MACHINES.items():
            states = sm.known_states(machine)
            for state, targets in table.items():
                assert targets <= states, (
                    f"{machine}.{state} points at unknown states: "
                    f"{sorted(targets - states)}"
                )

    def test_terminal_states_match_derivation(self):
        for machine in sm.machine_names():
            for state in sm.known_states(machine):
                assert sm.is_terminal(machine, state) == (
                    state in sm.TERMINAL_STATES[machine]
                )


# ─────────────────────────────────────────────────────────────────────────────
# Defaults on the persisted rows
#
# Column defaults are applied by the database layer, so a freshly constructed
# instance does not show them. These tests flush and assert that the entity the
# state machines will actually guard starts in the state each machine declares.
# ─────────────────────────────────────────────────────────────────────────────


def _seed_merchant(session, merchant_id="MER-1"):
    from app.models.merchant import Merchant

    merchant = Merchant(id=merchant_id, name="Test Merchant")
    session.add(merchant)
    session.flush()
    return merchant


def _seed_exception(session, exception_id="EXC-1"):
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


def _seed_resolution(session, resolution_id="RES-1"):
    resolution = Resolution(
        id=resolution_id,
        exception_id="EXC-1",
        candidate_rank=1,
        resolution_type="FEE_ADJUSTMENT",
    )
    session.add(resolution)
    session.flush()
    return resolution


class TestPersistedDefaults:
    """Each stateful entity must start in the state its machine declares."""

    def test_payment_starts_created(self, db_session):
        _seed_merchant(db_session)
        payment = Payment(id="PAY-1", merchant_id="MER-1", amount=25000)
        db_session.add(payment)
        db_session.flush()
        assert payment.status == "CREATED"
        assert payment.currency == "INR"
        assert payment.version == 1

    def test_reconciliation_run_starts_pending(self, db_session):
        run = ReconciliationRun(
            id="RUN-1",
            scope_type="BATCH",
            scope_ref="B-1",
            trigger="MANUAL",
            engine_version="1.0",
        )
        db_session.add(run)
        db_session.flush()
        assert run.status == "PENDING"
        assert run.version == 1
        assert run.matched_count == 0 and run.exception_count == 0

    def test_exception_starts_open(self, db_session):
        exc = _seed_exception(db_session)
        assert exc.status == "OPEN"
        assert exc.version == 1
        assert exc.reopen_count == 0
        assert exc.currency == "INR"

    def test_resolution_starts_draft(self, db_session):
        _seed_exception(db_session)
        resolution = _seed_resolution(db_session)
        assert resolution.status == "DRAFT"
        assert resolution.version == 1

    def test_resolution_action_starts_proposed(self, db_session):
        _seed_exception(db_session)
        _seed_resolution(db_session)
        action = ResolutionAction(
            id="ACT-1",
            resolution_id="RES-1",
            action_type="ADJUSTMENT",
            idempotency_key="idem-1",
        )
        db_session.add(action)
        db_session.flush()
        assert action.status == "PROPOSED"
        assert action.attempts == 0
        assert action.version == 1

    def test_approval_starts_pending(self, db_session):
        _seed_exception(db_session)
        _seed_resolution(db_session)
        approval = Approval(
            id="APR-1", resolution_id="RES-1", requested_by="SYSTEM"
        )
        db_session.add(approval)
        db_session.flush()
        assert approval.decision == "PENDING"

    def test_chargeback_starts_open(self, db_session):
        from app.models.chargeback import Chargeback

        _seed_merchant(db_session)
        payment = Payment(id="PAY-1", merchant_id="MER-1", amount=25000)
        db_session.add(payment)
        db_session.flush()
        chargeback = Chargeback(id="CB-1", payment_id="PAY-1", amount=25000)
        db_session.add(chargeback)
        db_session.flush()
        assert chargeback.status == "OPEN"
        assert chargeback.currency == "INR"

    def test_historical_case_starts_unverified(self, db_session):
        from app.services.historical_case_store import HistoricalCaseRecord

        case = HistoricalCaseRecord(
            id="HC-1",
            exception_id="EXC-1",
            payment_id="PAY-1",
            exception_type="FEE_DIFFERENCE",
            payment_amount=100000,
            expected_amount=95000,
            actual_amount=92500,
            difference=2500,
            resolution_type="FEE_ADJUSTMENT",
            resolution_outcome="RESOLVED",
        )
        db_session.add(case)
        db_session.flush()
        assert case.reconciliation_verified is False
        assert case.currency == "INR"
