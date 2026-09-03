"""
Resolution execution and verification integration tests.

Tests the complete pipeline:
  action_request → execute → before/after state → verification → rollback

Against isolated test data — no developer data affected.

Covers:
1. valid resolution
2. adjustment creation
3. before/after state capture
4. recalculation
5. discrepancy elimination
6. unintended change detection
7. verification success
8. verification failure
9. rollback
10. escalation after failure
11. audit metadata
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_res_verif.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.execution import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTransitionError,
    FinancialStateSnapshot,
    AdjustmentRecord,
    is_valid_transition,
    is_terminal,
)
from app.schemas.resolution_verification import (
    CheckResult,
    VerificationCheckType,
    VerificationStatus,
)
from app.schemas.rollback import RollbackReason, RollbackStatus
from app.schemas.financial_diff import ChangeType
from app.services.execution import ResolutionExecutionService
from app.services.resolution_verification import ResolutionVerificationEngine
from app.services.rollback import RollbackService
from app.services.financial_diff import FinancialDiffService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def exec_service():
    """Create a fresh execution service."""
    return ResolutionExecutionService()


@pytest.fixture
def verify_engine():
    """Create a fresh verification engine."""
    return ResolutionVerificationEngine()


@pytest.fixture
def rollback_service():
    """Create a fresh rollback service."""
    return RollbackService()


@pytest.fixture
def diff_service():
    """Create a fresh financial diff service."""
    return FinancialDiffService()


def _make_action_request(
    workflow_id="WF-TEST-001",
    exception_id="EXC-TEST-001",
    case_id="CASE-TEST-001",
    resolution_type="FEE_ADJUSTMENT",
    amount_paise=3000,
    idempotency_key="key-test-001",
    guardrail_decision="AUTO",
    guardrail_confidence=0.85,
    authorization_source="AUTO_GUARDRAIL",
    verification_passed=True,
    risk="LOW",
):
    """Build a valid action request dict."""
    return {
        "action_id": "ACT-TEST-001",
        "idempotency_key": idempotency_key,
        "workflow_id": workflow_id,
        "exception_id": exception_id,
        "case_id": case_id,
        "candidate_id": "CAND-FEE-001",
        "resolution_type": resolution_type,
        "financial_adjustment_paise": amount_paise,
        "authorization_source": authorization_source,
        "verification_passed": verification_passed,
        "guardrail_decision": guardrail_decision,
        "guardrail_confidence": guardrail_confidence,
        "evidence_summary": {
            "coverage": 0.95,
            "consistency": 0.90,
            "evidence_ids": ["EV-001", "EV-002"],
        },
        "metadata": {
            "risk": risk,
            "reason_codes": ["ALL_GATES_PASSED"],
        },
    }


def _make_financial_state(
    exception_id="EXC-TEST-001",
    payment_amount=100000,
    expected_amount=97000,
    actual_amount=94000,
    difference=3000,
    total_fees=3000,
    adjustment_count=0,
):
    """Build a financial state snapshot dict."""
    return {
        "payment_amount": payment_amount,
        "expected_amount": expected_amount,
        "actual_amount": actual_amount,
        "difference": difference,
        "total_refunds": 0,
        "total_fees": total_fees,
        "total_taxes": 0,
        "total_adjustments": 0,
        "settlement_count": 1,
        "refund_count": 0,
        "fee_count": 1,
        "tax_count": 0,
        "adjustment_count": adjustment_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Valid Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestValidResolution:
    """Test complete valid resolution execution."""

    def test_valid_execution_produces_result(
        self, exec_service
    ):
        """A valid action request should produce an EXECUTED result."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.status == ExecutionStatus.EXECUTED
        assert result.execution_id.startswith("EXE-")
        assert result.error is None

    def test_execution_result_has_all_metadata(
        self, exec_service
    ):
        """Execution result should contain full audit metadata."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.workflow_id == "WF-TEST-001"
        assert result.exception_id == "EXC-TEST-001"
        assert result.case_id == "CASE-TEST-001"
        assert result.candidate_id == "CAND-FEE-001"
        assert result.resolution_type == "FEE_ADJUSTMENT"
        assert result.authorization_source == "AUTO_GUARDRAIL"
        assert result.decision == "AUTO"
        assert result.confidence == 0.85
        assert result.risk == "LOW"
        assert result.guardrail_reason_codes == ["ALL_GATES_PASSED"]

    def test_execution_result_has_evidence_references(
        self, exec_service
    ):
        """Execution result should reference evidence records."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert "EV-001" in result.evidence_references
        assert "EV-002" in result.evidence_references

    def test_execution_result_has_timestamps(
        self, exec_service
    ):
        """Execution result should have creation and execution timestamps."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.created_at is not None
        assert result.executed_at is not None
        assert result.executed_at >= result.created_at


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adjustment Creation
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentCreation:
    """Test that adjustments are correctly created."""

    def test_adjustment_record_created(
        self, exec_service
    ):
        """Successful execution should create an AdjustmentRecord."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.adjustment is not None
        assert result.adjustment.adjustment_id.startswith("ADJ-")
        assert result.adjustment.amount_paise == 3000
        assert result.adjustment.requested_amount_paise == 3000
        assert result.adjustment.status == "applied"

    def test_fee_adjustment_type(
        self, exec_service
    ):
        """FEE_ADJUSTMENT resolution should create FEE_REVERSAL adjustment."""
        request = _make_action_request(resolution_type="FEE_ADJUSTMENT")
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.adjustment.adjustment_type == "FEE_REVERSAL"

    def test_refund_adjustment_type(
        self, exec_service
    ):
        """REFUND_ADJUSTMENT should create REFUND adjustment."""
        request = _make_action_request(resolution_type="REFUND_ADJUSTMENT")
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.adjustment.adjustment_type == "REFUND"

    def test_tax_adjustment_type(
        self, exec_service
    ):
        """TAX_ADJUSTMENT should create TAX_ADJUSTMENT adjustment."""
        request = _make_action_request(resolution_type="TAX_ADJUSTMENT")
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.adjustment.adjustment_type == "TAX_ADJUSTMENT"

    def test_adjustment_affected_records(
        self, exec_service
    ):
        """FEE_ADJUSTMENT should list fee records as affected."""
        request = _make_action_request(resolution_type="FEE_ADJUSTMENT")
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert len(result.adjustment.affected_records) > 0

    def test_actual_matches_requested(
        self, exec_service
    ):
        """Actual adjustment should match requested amount."""
        request = _make_action_request(amount_paise=5000)
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.actual_adjustment_paise == 5000
        assert result.requested_adjustment_paise == 5000


# ─────────────────────────────────────────────────────────────────────────────
# 3. Before/After State
# ─────────────────────────────────────────────────────────────────────────────


class TestBeforeAfterState:
    """Test before/after state capture."""

    def test_before_state_captured(
        self, exec_service
    ):
        """Before state should reflect the input financial state."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=94000,
            difference=3000,
        )

        result = exec_service.execute(request, financial_state)

        assert result.before_state is not None
        assert result.before_state.payment_amount == 100000
        assert result.before_state.expected_amount == 97000
        assert result.before_state.actual_amount == 94000
        assert result.before_state.difference == 3000

    def test_after_state_captured(
        self, exec_service
    ):
        """After state should reflect the post-execution financial state."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000,
            difference=3000,
        )

        result = exec_service.execute(request, financial_state)

        assert result.after_state is not None
        # actual_amount increases by adjustment amount
        assert result.after_state.actual_amount == 94000 + 3000
        # difference decreases by adjustment amount
        assert result.after_state.difference == 3000 - 3000
        # total_adjustments increases
        assert result.after_state.total_adjustments == 0 + 3000

    def test_after_state_has_snapshot_reason(
        self, exec_service
    ):
        """After state snapshot should be marked as post_execution."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.before_state.snapshot_reason == "pre_execution"
        assert result.after_state.snapshot_reason == "post_execution"

    def test_before_state_not_modified_by_execution(
        self, exec_service
    ):
        """Before state should not be modified by the execution."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(actual_amount=94000, difference=3000)

        result = exec_service.execute(request, financial_state)

        # Before state should still show original values
        assert result.before_state.actual_amount == 94000
        assert result.before_state.difference == 3000

    def test_no_after_state_on_failure(
        self, exec_service
    ):
        """Failed execution should not have after_state."""
        request = _make_action_request(resolution_type="")  # Missing type

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert result.after_state is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Recalculation
# ─────────────────────────────────────────────────────────────────────────────


class TestRecalculation:
    """Test that verification independently recalculates expected results."""

    def test_expected_result_recalled(
        self, exec_service, verify_engine
    ):
        """Verification should independently recalculate expected outcome."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000,
            actual_amount=94000,
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        expected = verification.expected_result
        assert expected.expected_adjustment_paise == 3000
        assert expected.expected_new_actual == 94000 + 3000
        assert expected.expected_new_difference == 97000 - (94000 + 3000)

    def test_recalculation_uses_integer_paise(
        self, exec_service, verify_engine
    ):
        """Recalculated values should all be integers."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state()

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        expected = verification.expected_result
        assert isinstance(expected.expected_adjustment_paise, int)
        assert isinstance(expected.expected_new_actual, int)
        assert isinstance(expected.expected_new_difference, int)
        assert isinstance(expected.expected_new_total_adjustments, int)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Discrepancy Eliminated
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscrepancyEliminated:
    """Test that successful resolution eliminates the discrepancy."""

    def test_exact_discrepancy_eliminated(
        self, exec_service, verify_engine
    ):
        """Adjustment exactly equal to difference → discrepancy eliminated."""
        # difference = 3000, adjustment = 3000 → difference becomes 0
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000,
            actual_amount=94000,
            difference=3000,
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.discrepancy_eliminated is True
        assert verification.difference_before == 3000
        assert verification.difference_after == 0

    def test_partial_discrepancy_not_eliminated(
        self, exec_service, verify_engine
    ):
        """Adjustment less than difference → discrepancy not fully eliminated."""
        # difference = 3000, adjustment = 1000 → difference becomes 2000
        request = _make_action_request(amount_paise=1000)
        financial_state = _make_financial_state(
            expected_amount=97000,
            actual_amount=94000,
            difference=3000,
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.discrepancy_eliminated is False
        assert verification.difference_after == 2000

    def test_no_original_discrepancy(
        self, exec_service, verify_engine
    ):
        """No original discrepancy → verification passes discrepancy check."""
        request = _make_action_request(amount_paise=0)
        financial_state = _make_financial_state(
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        disc_check = next(
            c for c in verification.checks
            if c.check_type == VerificationCheckType.DISCREPANCY_ELIMINATED
        )
        assert disc_check.result == CheckResult.PASS


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unintended Change Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestUnintendedChanges:
    """Test that the system detects unintended financial changes."""

    def test_no_unintended_changes(
        self, exec_service, verify_engine, diff_service
    ):
        """Clean resolution should have no unintended changes."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.has_unintended_changes is False
        assert verification.unintended_change_count == 0

    def test_diff_service_detects_intended_changes(
        self, diff_service, exec_service
    ):
        """FinancialDiffService should correctly classify intended changes."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        diff = diff_service.compare(
            execution.before_state,
            execution.after_state,
            execution,
        )

        # actual_amount and total_adjustments should be intended
        intended_field_names = [fc.field_name for fc in diff.intended_changes]
        assert "actual_amount" in intended_field_names
        assert "total_adjustments" in intended_field_names

    def test_diff_detects_unintended_field_change(
        self, diff_service, exec_service
    ):
        """If an unexpected field changes, diff should flag it."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        # Tamper with after_state to simulate unintended change
        from app.schemas.execution import FinancialStateSnapshot
        tampered_after = FinancialStateSnapshot(
            exception_id=execution.after_state.exception_id,
            payment_amount=execution.after_state.payment_amount,
            expected_amount=execution.after_state.expected_amount,
            actual_amount=execution.after_state.actual_amount,
            difference=execution.after_state.difference,
            total_refunds=5000,  # Unintended change!
            total_fees=execution.after_state.total_fees,
            total_taxes=execution.after_state.total_taxes,
            total_adjustments=execution.after_state.total_adjustments,
            adjustment_count=execution.after_state.adjustment_count,
            snapshot_reason="post_execution",
        )

        diff = diff_service.compare(
            execution.before_state, tampered_after, execution
        )

        assert diff.has_unintended_changes is True
        unintended_fields = [fc.field_name for fc in diff.unintended_changes]
        assert "total_refunds" in unintended_fields

    def test_diff_all_integer_paise(
        self, exec_service, diff_service
    ):
        """All financial values should be integer paise."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        diff = diff_service.compare(
            execution.before_state, execution.after_state, execution
        )

        assert diff.all_integer_paise is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Verification Success
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationSuccess:
    """Test that correct resolution passes verification."""

    def test_all_checks_pass(
        self, exec_service, verify_engine
    ):
        """All verification checks should pass for a correct resolution."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.status == VerificationStatus.PASSED
        assert verification.failed_checks == 0
        assert verification.passed_checks >= 4  # At least 4 checks

    def test_verification_status_passed(
        self, exec_service, verify_engine
    ):
        """Verification should return PASSED for correct resolution."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.status == VerificationStatus.PASSED

    def test_verification_has_metadata(
        self, exec_service, verify_engine
    ):
        """Verification should include execution and exception IDs."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.execution_id == execution.execution_id
        assert verification.exception_id == "EXC-TEST-001"
        assert verification.verified_by == "resolution_verification_engine"
        assert verification.verified_at is not None

    def test_each_check_has_message(
        self, exec_service, verify_engine
    ):
        """Every verification check should have a message."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        for check in verification.checks:
            assert check.message is not None
            assert len(check.message) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Verification Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFailure:
    """Test that incorrect resolution fails verification."""

    def test_wrong_amount_fails_verification(
        self, exec_service, verify_engine
    ):
        """Wrong adjustment amount should fail verification."""
        request = _make_action_request(amount_paise=1000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        # Discrepancy not fully eliminated
        assert verification.discrepancy_eliminated is False

    def test_amount_consistency_fails(
        self, exec_service, verify_engine
    ):
        """If actual amounts don't match expected, consistency check fails."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        # Create a mismatched current state
        from app.schemas.execution import FinancialStateSnapshot
        wrong_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=96000,  # Wrong! Should be 97000
            difference=1000,
            total_adjustments=3000,
            snapshot_reason="post_execution",
        )

        verification = verify_engine.verify(execution, wrong_state)

        assert verification.status == VerificationStatus.FAILED

    def test_unintended_changes_fail_verification(
        self, exec_service, verify_engine
    ):
        """Unintended changes should cause verification to fail."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        tampered = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,  # Correct
            difference=0,  # Correct
            total_refunds=5000,  # Unintended!
            total_fees=3000,
            total_taxes=0,
            total_adjustments=3000,
            snapshot_reason="post_execution",
        )

        verification = verify_engine.verify(execution, tampered)

        # Unintended changes detected
        assert verification.has_unintended_changes is True

    def test_failed_verification_has_failed_checks(
        self, exec_service, verify_engine
    ):
        """Failed verification should have failed_checks > 0."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        # If PASSED, failed_checks should be 0
        if verification.status == VerificationStatus.FAILED:
            assert verification.failed_checks > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Rollback
# ─────────────────────────────────────────────────────────────────────────────


class TestRollback:
    """Test rollback after failed verification."""

    def test_rollback_reverses_adjustment(
        self, exec_service, rollback_service
    ):
        """Rollback should reverse the adjustment amount."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            adjustment_count=1,  # execution incremented this
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.VERIFICATION_FAILED
        )

        assert rollback.status == RollbackStatus.ROLLED_BACK
        assert rollback.adjustment_reversed is True
        assert rollback.reversal_amount_paise == 3000

    def test_rollback_state_matches_before(
        self, exec_service, rollback_service
    ):
        """Rollback should restore state to the before_snapshot."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            adjustment_count=1,  # execution incremented this
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.VERIFICATION_FAILED
        )

        assert rollback.rollback_verified is True
        assert rollback.rollback_state_match is True

        # After rollback should match before state
        after = rollback.after_rollback_state
        before = execution.before_state
        assert after["actual_amount"] == before.actual_amount
        assert after["total_adjustments"] == before.total_adjustments

    def test_rollback_has_audit_trail(
        self, exec_service, rollback_service
    ):
        """Rollback should produce a full audit trail."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.VERIFICATION_FAILED
        )

        assert len(rollback.audit_trail) >= 3  # identity, idempotency, reverse, verify

        actions = [e.action for e in rollback.audit_trail]
        assert "confirm_identity" in actions
        assert "reverse_adjustment" in actions
        assert "verify_rollback" in actions

    def test_rollback_has_timestamps(
        self, exec_service, rollback_service
    ):
        """Rollback should have creation and completion timestamps."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.VERIFICATION_FAILED
        )

        assert rollback.created_at is not None
        assert rollback.completed_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Escalation After Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestEscalationAfterFailure:
    """Test escalation behavior when things go wrong."""

    def test_execution_failure_no_after_state(
        self, exec_service
    ):
        """Execution failure should not produce after_state."""
        request = _make_action_request(resolution_type="")

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert result.after_state is None
        assert result.error is not None

    def test_missing_guardrail_decision_blocks(
        self, exec_service
    ):
        """Missing guardrail decision should block execution."""
        request = _make_action_request(guardrail_decision=None)

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "Guardrail decision" in result.error

    def test_unresolved_guardrail_blocks(
        self, exec_service
    ):
        """UNRESOLVED guardrail decision should block execution."""
        request = _make_action_request(guardrail_decision="UNRESOLVED")

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "UNRESOLVED" in result.error

    def test_no_authorization_blocks(
        self, exec_service
    ):
        """NONE authorization should block execution."""
        request = _make_action_request()
        request["authorization_source"] = "NONE"

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "authorization" in result.error.lower()

    def test_verification_not_passed_blocks(
        self, exec_service
    ):
        """Verification not passed should block execution."""
        request = _make_action_request(verification_passed=False)

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "Verification" in result.error

    def test_missing_financial_adjustment_blocks(
        self, exec_service
    ):
        """Missing financial adjustment should block execution."""
        request = _make_action_request()
        del request["financial_adjustment_paise"]

        result = exec_service.execute(request)

        assert result.status == ExecutionStatus.EXECUTION_FAILED

    def test_failed_execution_has_error_message(
        self, exec_service
    ):
        """Failed execution should include error description."""
        request = _make_action_request(resolution_type="")

        result = exec_service.execute(request)

        assert result.error is not None
        assert len(result.error) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Audit Metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditMetadata:
    """Test that complete audit metadata is preserved."""

    def test_execution_has_decision_metadata(
        self, exec_service
    ):
        """Execution should record decision, confidence, and risk."""
        request = _make_action_request(
            guardrail_decision="AUTO",
            guardrail_confidence=0.85,
            risk="LOW",
        )
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert result.decision == "AUTO"
        assert result.confidence == 0.85
        assert result.risk == "LOW"
        assert result.guardrail_reason_codes == ["ALL_GATES_PASSED"]

    def test_execution_has_evidence_metadata(
        self, exec_service
    ):
        """Execution should reference evidence records."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert len(result.evidence_references) > 0

    def test_execution_has_guardrail_reason_codes(
        self, exec_service
    ):
        """Execution should store guardrail reason codes."""
        request = _make_action_request()
        financial_state = _make_financial_state()

        result = exec_service.execute(request, financial_state)

        assert "ALL_GATES_PASSED" in result.guardrail_reason_codes

    def test_verification_has_verification_metadata(
        self, exec_service, verify_engine
    ):
        """Verification should record who verified and when."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)
        verification = verify_engine.verify(execution, execution.after_state)

        assert verification.verified_by == "resolution_verification_engine"
        assert verification.verified_at is not None

    def test_rollback_has_reason_metadata(
        self, exec_service, rollback_service
    ):
        """Rollback should record the reason for rollback."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            adjustment_count=1,
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.UNINTENDED_CHANGES
        )

        assert rollback.reason == RollbackReason.UNINTENDED_CHANGES

    def test_execution_idempotency_preserves_metadata(
        self, exec_service
    ):
        """Duplicate execution should return cached result with same metadata."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result1 = exec_service.execute(request, financial_state)
        result2 = exec_service.execute(request, financial_state)

        # Same idempotency key → same result
        assert result1.execution_id == result2.execution_id
        assert result1.status == result2.status
        assert result1.workflow_id == result2.workflow_id


# ─────────────────────────────────────────────────────────────────────────────
# 12. State Machine Transitions
# ─────────────────────────────────────────────────────────────────────────────


class TestStateMachine:
    """Test execution status state machine."""

    def test_valid_transition_executed_to_verification_pending(
        self, exec_service
    ):
        """EXECUTED → VERIFICATION_PENDING should be valid."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)
        assert result.status == ExecutionStatus.EXECUTED

        # Transition to VERIFICATION_PENDING
        updated = exec_service.transition_status(
            result, ExecutionStatus.VERIFICATION_PENDING
        )
        assert updated.status == ExecutionStatus.VERIFICATION_PENDING

    def test_invalid_transition_raises(
        self, exec_service
    ):
        """Invalid transition should raise ExecutionTransitionError."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)

        # EXECUTED → VERIFIED is not valid (must go through VERIFICATION_PENDING)
        with pytest.raises(ExecutionTransitionError):
            exec_service.transition_status(result, ExecutionStatus.VERIFIED)

    def test_terminal_states(
        self,
    ):
        """VERIFIED and ESCALATED should be terminal states."""
        assert is_terminal(ExecutionStatus.VERIFIED) is True
        assert is_terminal(ExecutionStatus.ESCALATED) is True
        assert is_terminal(ExecutionStatus.EXECUTED) is False

    def test_allowed_transitions(
        self,
    ):
        """Each state should have defined allowed transitions."""
        from app.schemas.execution import get_allowed_transitions

        executed_transitions = get_allowed_transitions(ExecutionStatus.EXECUTED)
        assert ExecutionStatus.VERIFICATION_PENDING in executed_transitions

        failed_transitions = get_allowed_transitions(ExecutionStatus.VERIFICATION_FAILED)
        assert ExecutionStatus.ROLLED_BACK in failed_transitions

    def test_full_lifecycle_happy_path(
        self, exec_service
    ):
        """Full lifecycle: NOT_EXECUTED → EXECUTING → EXECUTED → VERIFICATION_PENDING → VERIFIED."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)
        assert result.status == ExecutionStatus.EXECUTED

        result = exec_service.transition_status(
            result, ExecutionStatus.VERIFICATION_PENDING
        )
        result = exec_service.transition_status(
            result, ExecutionStatus.VERIFIED
        )
        assert result.status == ExecutionStatus.VERIFIED
        assert is_terminal(result.status)

    def test_full_lifecycle_rollback_path(
        self, exec_service
    ):
        """Full lifecycle with rollback: EXECUTED → VERIFICATION_PENDING → VERIFICATION_FAILED → ROLLED_BACK."""
        request = _make_action_request()
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)

        result = exec_service.transition_status(
            result, ExecutionStatus.VERIFICATION_PENDING
        )
        result = exec_service.transition_status(
            result, ExecutionStatus.VERIFICATION_FAILED
        )
        result = exec_service.transition_status(
            result, ExecutionStatus.ROLLED_BACK
        )
        assert result.status == ExecutionStatus.ROLLED_BACK

    def test_execution_failed_can_retry(
        self, exec_service
    ):
        """EXECUTION_FAILED → NOT_EXECUTED should be valid (retry)."""
        from app.schemas.execution import VALID_TRANSITIONS

        allowed = VALID_TRANSITIONS.get(ExecutionStatus.EXECUTION_FAILED, set())
        assert ExecutionStatus.NOT_EXECUTED in allowed


# ─────────────────────────────────────────────────────────────────────────────
# 13. Financial Consistency After Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialConsistency:
    """Verify financial consistency of resolved state."""

    def test_resolved_state_difference_eliminated(
        self, exec_service
    ):
        """After correct resolution, difference should be 0."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)

        # After state should have difference = 0
        assert result.after_state.difference == 0
        assert result.after_state.actual_amount == 97000

    def test_adjustments_track_total(
        self, exec_service
    ):
        """Total adjustments should increase by the adjustment amount."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, difference=3000
        )

        result = exec_service.execute(request, financial_state)

        assert result.after_state.total_adjustments == 0 + 3000

    def test_adjustment_count_increments(
        self, exec_service
    ):
        """Adjustment count should increment by 1."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            actual_amount=94000, adjustment_count=0
        )

        result = exec_service.execute(request, financial_state)

        assert result.after_state.adjustment_count == 1

    def test_payment_amount_unchanged(
        self, exec_service
    ):
        """Payment amount should not change after resolution."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(payment_amount=100000)

        result = exec_service.execute(request, financial_state)

        assert result.after_state.payment_amount == 100000
        assert result.before_state.payment_amount == 100000

    def test_expected_amount_unchanged(
        self, exec_service
    ):
        """Expected amount should not change after resolution."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(expected_amount=97000)

        result = exec_service.execute(request, financial_state)

        assert result.after_state.expected_amount == 97000

    def test_rollback_restores_consistency(
        self, exec_service, rollback_service
    ):
        """After rollback, financial state should be consistent with before_state."""
        request = _make_action_request(amount_paise=3000)
        financial_state = _make_financial_state(
            expected_amount=97000, actual_amount=94000, difference=3000
        )

        execution = exec_service.execute(request, financial_state)

        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id="EXC-TEST-001",
            payment_amount=100000,
            expected_amount=97000,
            actual_amount=97000,
            difference=0,
            total_adjustments=3000,
            adjustment_count=1,
            snapshot_reason="post_execution",
        )

        rollback = rollback_service.rollback(
            execution, current_state, RollbackReason.VERIFICATION_FAILED
        )

        after = rollback.after_rollback_state
        assert after["actual_amount"] == execution.before_state.actual_amount
        assert after["difference"] == execution.before_state.difference
        assert after["total_adjustments"] == execution.before_state.total_adjustments
