"""
Tests for Razorpay CloseLoop Phase 8B — Execution State Machine.

Tests valid/invalid transitions, terminal states, success semantics,
and the centralized transition policy.
"""

import pytest
from app.schemas.execution import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTransitionError,
    FinancialStateSnapshot,
    VALID_TRANSITIONS,
    get_allowed_transitions,
    is_terminal,
    is_valid_transition,
)
from app.services.execution import ResolutionExecutionService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_result(status: ExecutionStatus = ExecutionStatus.EXECUTED) -> ExecutionResult:
    """Build an ExecutionResult with the given status."""
    return ExecutionResult(
        execution_id="EXE-001",
        action_id="ACT-001",
        idempotency_key="key-001",
        workflow_id="WF-001",
        exception_id="EXC-001",
        resolution_type="APPLY_FEE_CORRECTION",
        authorization_source="AUTO_GUARDRAIL",
        before_state=FinancialStateSnapshot(exception_id="EXC-001"),
        status=status,
    )


def _make_request(**overrides) -> dict:
    """Build a valid action request."""
    base = {
        "action_id": "ACT-001",
        "idempotency_key": "key-001",
        "workflow_id": "WF-001",
        "exception_id": "EXC-001",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "financial_adjustment_paise": 3000,
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Transition Policy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTransitionPolicy:
    def test_all_statuses_have_transitions_defined(self):
        """Every status has an entry in the transition table."""
        for status in ExecutionStatus:
            assert status in VALID_TRANSITIONS

    def test_valid_transition_function(self):
        """is_valid_transition returns correct results."""
        assert is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.EXECUTING)
        assert is_valid_transition(ExecutionStatus.EXECUTING, ExecutionStatus.EXECUTED)
        assert is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.VERIFICATION_PENDING)
        assert is_valid_transition(ExecutionStatus.VERIFICATION_PENDING, ExecutionStatus.VERIFIED)

    def test_invalid_transition_function(self):
        """is_valid_transition rejects invalid transitions."""
        assert not is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.VERIFIED)
        assert not is_valid_transition(ExecutionStatus.EXECUTING, ExecutionStatus.VERIFIED)
        assert not is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.VERIFIED)

    def test_terminal_states(self):
        """VERIFIED and ESCALATED are terminal."""
        assert is_terminal(ExecutionStatus.VERIFIED)
        assert is_terminal(ExecutionStatus.ESCALATED)

    def test_non_terminal_states(self):
        """Other states are not terminal."""
        assert not is_terminal(ExecutionStatus.NOT_EXECUTED)
        assert not is_terminal(ExecutionStatus.EXECUTING)
        assert not is_terminal(ExecutionStatus.EXECUTED)
        assert not is_terminal(ExecutionStatus.VERIFICATION_PENDING)

    def test_get_allowed_transitions(self):
        """get_allowed_transitions returns the correct set."""
        allowed = get_allowed_transitions(ExecutionStatus.NOT_EXECUTED)
        assert ExecutionStatus.EXECUTING in allowed
        assert ExecutionStatus.ESCALATED in allowed
        assert ExecutionStatus.VERIFIED not in allowed


# ─────────────────────────────────────────────────────────────────────────────
# Happy Path Transitions
# ─────────────────────────────────────────────────────────────────────────────


class TestHappyPathTransitions:
    def test_not_executed_to_executing(self):
        """NOT_EXECUTED → EXECUTING ✓"""
        assert is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.EXECUTING)

    def test_executing_to_executed(self):
        """EXECUTING → EXECUTED ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTING, ExecutionStatus.EXECUTED)

    def test_executed_to_verification_pending(self):
        """EXECUTED → VERIFICATION_PENDING ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.VERIFICATION_PENDING)

    def test_verification_pending_to_verified(self):
        """VERIFICATION_PENDING → VERIFIED ✓"""
        assert is_valid_transition(ExecutionStatus.VERIFICATION_PENDING, ExecutionStatus.VERIFIED)

    def test_full_happy_path(self):
        """Complete happy path: NOT_EXECUTED → EXECUTING → EXECUTED → VERIFICATION_PENDING → VERIFIED."""
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        result.transition_to(ExecutionStatus.EXECUTING)
        assert result.status == ExecutionStatus.EXECUTING

        result.transition_to(ExecutionStatus.EXECUTED)
        assert result.status == ExecutionStatus.EXECUTED

        result.transition_to(ExecutionStatus.VERIFICATION_PENDING)
        assert result.status == ExecutionStatus.VERIFICATION_PENDING

        result.transition_to(ExecutionStatus.VERIFIED)
        assert result.status == ExecutionStatus.VERIFIED


# ─────────────────────────────────────────────────────────────────────────────
# Failure Path Transitions
# ─────────────────────────────────────────────────────────────────────────────


class TestFailurePathTransitions:
    def test_executing_to_execution_failed(self):
        """EXECUTING → EXECUTION_FAILED ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTING, ExecutionStatus.EXECUTION_FAILED)

    def test_execution_failed_to_escalated(self):
        """EXECUTION_FAILED → ESCALATED ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTION_FAILED, ExecutionStatus.ESCALATED)

    def test_execution_failed_to_retry(self):
        """EXECUTION_FAILED → NOT_EXECUTED (retry) ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTION_FAILED, ExecutionStatus.NOT_EXECUTED)

    def test_verification_pending_to_verification_failed(self):
        """VERIFICATION_PENDING → VERIFICATION_FAILED ✓"""
        assert is_valid_transition(ExecutionStatus.VERIFICATION_PENDING, ExecutionStatus.VERIFICATION_FAILED)

    def test_verification_failed_to_rolled_back(self):
        """VERIFICATION_FAILED → ROLLED_BACK ✓"""
        assert is_valid_transition(ExecutionStatus.VERIFICATION_FAILED, ExecutionStatus.ROLLED_BACK)

    def test_verification_failed_to_escalated(self):
        """VERIFICATION_FAILED → ESCALATED ✓"""
        assert is_valid_transition(ExecutionStatus.VERIFICATION_FAILED, ExecutionStatus.ESCALATED)

    def test_verification_failed_to_retry(self):
        """VERIFICATION_FAILED → EXECUTING (retry) ✓"""
        assert is_valid_transition(ExecutionStatus.VERIFICATION_FAILED, ExecutionStatus.EXECUTING)

    def test_rolled_back_to_escalated(self):
        """ROLLED_BACK → ESCALATED ✓"""
        assert is_valid_transition(ExecutionStatus.ROLLED_BACK, ExecutionStatus.ESCALATED)

    def test_not_executed_to_escalated(self):
        """NOT_EXECUTED → ESCALATED ✓"""
        assert is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.ESCALATED)

    def test_executed_to_escalated(self):
        """EXECUTED → ESCALATED ✓"""
        assert is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.ESCALATED)


# ─────────────────────────────────────────────────────────────────────────────
# Invalid Transition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInvalidTransitions:
    def test_not_executed_to_executed(self):
        """NOT_EXECUTED → EXECUTED ✗ (must go through EXECUTING)."""
        assert not is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.EXECUTED)

    def test_not_executed_to_verified(self):
        """NOT_EXECUTED → VERIFIED ✗ (skips execution + verification)."""
        assert not is_valid_transition(ExecutionStatus.NOT_EXECUTED, ExecutionStatus.VERIFIED)

    def test_executing_to_verified(self):
        """EXECUTING → VERIFIED ✗ (skips execution + verification)."""
        assert not is_valid_transition(ExecutionStatus.EXECUTING, ExecutionStatus.VERIFIED)

    def test_executed_to_verified(self):
        """EXECUTED → VERIFIED ✗ (must go through verification)."""
        assert not is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.VERIFIED)

    def test_executed_to_execution_failed(self):
        """EXECUTED → EXECUTION_FAILED ✗ (already executed)."""
        assert not is_valid_transition(ExecutionStatus.EXECUTED, ExecutionStatus.EXECUTION_FAILED)

    def test_execution_failed_to_verified(self):
        """EXECUTION_FAILED → VERIFIED ✗ (never executed successfully)."""
        assert not is_valid_transition(ExecutionStatus.EXECUTION_FAILED, ExecutionStatus.VERIFIED)

    def test_execution_failed_to_executed(self):
        """EXECUTION_FAILED → EXECUTED ✗ (must retry)."""
        assert not is_valid_transition(ExecutionStatus.EXECUTION_FAILED, ExecutionStatus.EXECUTED)

    def test_verification_failed_to_verified(self):
        """VERIFICATION_FAILED → VERIFIED ✗ (verification failed)."""
        assert not is_valid_transition(ExecutionStatus.VERIFICATION_FAILED, ExecutionStatus.VERIFIED)

    def test_verification_failed_to_executed(self):
        """VERIFICATION_FAILED → EXECUTED ✗ (must retry or rollback)."""
        assert not is_valid_transition(ExecutionStatus.VERIFICATION_FAILED, ExecutionStatus.EXECUTED)

    def test_rolled_back_to_executed(self):
        """ROLLED_BACK → EXECUTED ✗ (rolled back)."""
        assert not is_valid_transition(ExecutionStatus.ROLLED_BACK, ExecutionStatus.EXECUTED)

    def test_rolled_back_to_verified(self):
        """ROLLED_BACK → VERIFIED ✗ (rolled back)."""
        assert not is_valid_transition(ExecutionStatus.ROLLED_BACK, ExecutionStatus.VERIFIED)

    def test_verified_to_anything(self):
        """VERIFIED is terminal — no transitions allowed."""
        for status in ExecutionStatus:
            if status != ExecutionStatus.VERIFIED:
                assert not is_valid_transition(ExecutionStatus.VERIFIED, status), \
                    f"VERIFIED → {status.value} should be invalid"

    def test_escalated_to_anything(self):
        """ESCALATED is terminal — no transitions allowed."""
        for status in ExecutionStatus:
            if status != ExecutionStatus.ESCALATED:
                assert not is_valid_transition(ExecutionStatus.ESCALATED, status), \
                    f"ESCALATED → {status.value} should be invalid"


# ─────────────────────────────────────────────────────────────────────────────
# TransitionTo Method Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTransitionToMethod:
    def test_valid_transition_applied(self):
        """Valid transition is applied to result."""
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        result.transition_to(ExecutionStatus.EXECUTING)
        assert result.status == ExecutionStatus.EXECUTING

    def test_invalid_transition_raises(self):
        """Invalid transition raises ExecutionTransitionError."""
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        with pytest.raises(ExecutionTransitionError, match="NOT_EXECUTED → VERIFIED"):
            result.transition_to(ExecutionStatus.VERIFIED)

    def test_cannot_skip_to_verified(self):
        """Cannot skip execution and jump to VERIFIED."""
        result = _make_result(ExecutionStatus.EXECUTING)
        with pytest.raises(ExecutionTransitionError):
            result.transition_to(ExecutionStatus.VERIFIED)

    def test_cannot_go_from_failed_to_verified(self):
        """EXECUTION_FAILED → VERIFIED must fail."""
        result = _make_result(ExecutionStatus.EXECUTION_FAILED)
        with pytest.raises(ExecutionTransitionError):
            result.transition_to(ExecutionStatus.VERIFIED)


# ─────────────────────────────────────────────────────────────────────────────
# Service Transition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestServiceTransitions:
    def test_transition_status_valid(self):
        """Service transitions valid status."""
        service = ResolutionExecutionService()
        result = _make_result(ExecutionStatus.EXECUTED)
        updated = service.transition_status(result, ExecutionStatus.VERIFICATION_PENDING)
        assert updated.status == ExecutionStatus.VERIFICATION_PENDING

    def test_transition_status_invalid(self):
        """Service rejects invalid transition."""
        service = ResolutionExecutionService()
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        with pytest.raises(ExecutionTransitionError):
            service.transition_status(result, ExecutionStatus.VERIFIED)

    def test_transition_to_verified_updates_timestamp(self):
        """Transition to VERIFIED sets verified_at."""
        service = ResolutionExecutionService()
        result = _make_result(ExecutionStatus.VERIFICATION_PENDING)
        updated = service.transition_status(result, ExecutionStatus.VERIFIED)
        assert updated.verified_at is not None

    def test_transition_to_rolled_back_sets_reason(self):
        """Transition to ROLLED_BACK sets rollback_reason."""
        service = ResolutionExecutionService()
        result = _make_result(ExecutionStatus.VERIFICATION_FAILED)
        updated = service.transition_status(result, ExecutionStatus.ROLLED_BACK, reason="Stale state")
        assert updated.rollback_reason == "Stale state"
        assert updated.rolled_back_at is not None

    def test_service_stores_updated_result(self):
        """Service updates idempotency store after transition."""
        service = ResolutionExecutionService()
        request = _make_request()
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTED

        service.transition_status(result, ExecutionStatus.VERIFICATION_PENDING)
        stored = service.get_execution("key-001")
        assert stored.status == ExecutionStatus.VERIFICATION_PENDING


# ─────────────────────────────────────────────────────────────────────────────
# Success Semantics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuccessSemantics:
    def test_executed_is_not_success(self):
        """EXECUTED does not represent final success."""
        result = _make_result(ExecutionStatus.EXECUTED)
        assert result.status != ExecutionStatus.VERIFIED
        assert not is_terminal(result.status)

    def test_only_verified_is_success(self):
        """Only VERIFIED represents successful resolution."""
        result = _make_result(ExecutionStatus.VERIFIED)
        assert is_terminal(result.status)

    def test_verification_pending_not_success(self):
        """VERIFICATION_PENDING is not success."""
        result = _make_result(ExecutionStatus.VERIFICATION_PENDING)
        assert not is_terminal(result.status)

    def test_full_success_requires_verification(self):
        """Full success requires going through verification."""
        result = _make_result(ExecutionStatus.EXECUTED)
        # Cannot go directly to VERIFIED
        with pytest.raises(ExecutionTransitionError):
            result.transition_to(ExecutionStatus.VERIFIED)
        # Must go through verification
        result.transition_to(ExecutionStatus.VERIFICATION_PENDING)
        result.transition_to(ExecutionStatus.VERIFIED)
        assert result.status == ExecutionStatus.VERIFIED


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_retry_from_execution_failed(self):
        """Can retry from EXECUTION_FAILED → NOT_EXECUTED."""
        result = _make_result(ExecutionStatus.EXECUTION_FAILED)
        result.transition_to(ExecutionStatus.NOT_EXECUTED)
        result.transition_to(ExecutionStatus.EXECUTING)
        result.transition_to(ExecutionStatus.EXECUTED)
        assert result.status == ExecutionStatus.EXECUTED

    def test_retry_from_verification_failed(self):
        """Can retry from VERIFICATION_FAILED → EXECUTING."""
        result = _make_result(ExecutionStatus.VERIFICATION_FAILED)
        result.transition_to(ExecutionStatus.EXECUTING)
        result.transition_to(ExecutionStatus.EXECUTED)
        assert result.status == ExecutionStatus.EXECUTED

    def test_escalation_from_any_non_terminal(self):
        """Can escalate from most non-terminal states."""
        for status in ExecutionStatus:
            if not is_terminal(status) and status != ExecutionStatus.VERIFIED:
                allowed = get_allowed_transitions(status)
                # Most states can escalate (checked individually below)
                if ExecutionStatus.ESCALATED in allowed:
                    result = _make_result(status)
                    result.transition_to(ExecutionStatus.ESCALATED)
                    assert result.status == ExecutionStatus.ESCALATED

    def test_multiple_transitions_chained(self):
        """Chain multiple valid transitions."""
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        transitions = [
            ExecutionStatus.EXECUTING,
            ExecutionStatus.EXECUTED,
            ExecutionStatus.VERIFICATION_PENDING,
            ExecutionStatus.VERIFIED,
        ]
        for target in transitions:
            result.transition_to(target)
        assert result.status == ExecutionStatus.VERIFIED

    def test_failure_then_retry_then_success(self):
        """Execute → fail → retry → succeed → verify."""
        result = _make_result(ExecutionStatus.NOT_EXECUTED)
        result.transition_to(ExecutionStatus.EXECUTING)
        result.transition_to(ExecutionStatus.EXECUTION_FAILED)
        result.transition_to(ExecutionStatus.NOT_EXECUTED)  # retry
        result.transition_to(ExecutionStatus.EXECUTING)
        result.transition_to(ExecutionStatus.EXECUTED)
        result.transition_to(ExecutionStatus.VERIFICATION_PENDING)
        result.transition_to(ExecutionStatus.VERIFIED)
        assert result.status == ExecutionStatus.VERIFIED
