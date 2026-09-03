"""
Adversarial safety tests for verification failure scenarios.

Verifies that when resolution execution appears successful but independent
verification detects problems, the system fails safely.

Verification failure chain:
  1. Resolution executes (appears successful)
  2. Independent verification recalculates expected state
  3. Verification reads actual post-execution state
  4. Checks: discrepancy eliminated, correct adjustment, no unintended changes
  5. If ANY check fails → VerificationStatus.FAILED
  6. Failed verification → rollback (where supported)
  7. Rollback failure → escalation

Safety invariant:
  Failed verification → NEVER reported as FINAL_SUCCESS
  Failed verification → rollback or escalation

No production logic is modified.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_verify.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.execution import (
    ExecutionStatus,
    ExecutionResult,
    FinancialStateSnapshot,
)
from app.schemas.resolution_verification import (
    CheckResult,
    VerificationCheckType,
    VerificationStatus,
)
from app.schemas.rollback import RollbackReason, RollbackStatus
from app.services.resolution_verification import ResolutionVerificationEngine
from app.services.rollback import RollbackService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot(
    exception_id="EXC-VERIFY-001",
    payment_amount=100_000,
    expected_amount=95_000,
    actual_amount=90_000,
    difference=5_000,
    total_adjustments=0,
    adjustment_count=0,
    total_refunds=0,
    total_fees=0,
    total_taxes=0,
):
    """Build a FinancialStateSnapshot."""
    return FinancialStateSnapshot(
        exception_id=exception_id,
        payment_amount=payment_amount,
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        total_adjustments=total_adjustments,
        adjustment_count=adjustment_count,
        total_refunds=total_refunds,
        total_fees=total_fees,
        total_taxes=total_taxes,
    )


def _make_execution_result(
    exception_id="EXC-VERIFY-001",
    status=ExecutionStatus.VERIFIED,
    before_difference=5_000,
    adjustment_amount=5_000,
    after_actual=None,
    after_adjustments=None,
    after_difference=0,
):
    """Build an ExecutionResult for testing."""
    before = _make_snapshot(
        exception_id=exception_id,
        difference=before_difference,
        actual_amount=90_000,
        expected_amount=95_000,
    )
    if after_actual is None:
        after_actual = 90_000 + adjustment_amount
    if after_adjustments is None:
        after_adjustments = adjustment_amount
    after = _make_snapshot(
        exception_id=exception_id,
        actual_amount=after_actual,
        difference=after_difference,
        total_adjustments=after_adjustments,
        adjustment_count=1,
    )
    return ExecutionResult(
        execution_id="EXEC-VERIFY-001",
        action_id="ACT-VERIFY-001",
        exception_id=exception_id,
        workflow_id="WF-VERIFY-001",
        status=status,
        resolution_type="FEE_ADJUSTMENT",
        adjustment_amount_paise=adjustment_amount,
        requested_adjustment_paise=adjustment_amount,
        actual_adjustment_paise=adjustment_amount,
        authorization_source="guardrail_engine",
        idempotency_key="IDEM-VERIFY-001",
        before_state=before,
        after_state=after,
    )


def _make_mismatched_execution(
    exception_id="EXC-VERIFY-002",
    requested=5_000,
    actual_applied=3_000,
):
    """Build execution where actual adjustment differs from requested."""
    before = _make_snapshot(
        exception_id=exception_id,
        difference=5_000,
        actual_amount=90_000,
        expected_amount=95_000,
    )
    after = _make_snapshot(
        exception_id=exception_id,
        actual_amount=90_000 + actual_applied,
        difference=5_000 - actual_applied,
        total_adjustments=actual_applied,
        adjustment_count=1,
    )
    return ExecutionResult(
        execution_id="EXEC-VERIFY-003",
        action_id="ACT-VERIFY-003",
        exception_id=exception_id,
        workflow_id="WF-VERIFY-001",
        status=ExecutionStatus.VERIFIED,
        resolution_type="FEE_ADJUSTMENT",
        adjustment_amount_paise=actual_applied,
        requested_adjustment_paise=requested,
        actual_adjustment_paise=actual_applied,
        authorization_source="guardrail_engine",
        idempotency_key="IDEM-VERIFY-002",
        before_state=before,
        after_state=after,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Successful Verification (Baseline)
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationSuccessBaseline:
    """Test that correct resolution passes verification."""

    def test_correct_resolution_passes(self):
        """Correct resolution → all checks pass → VERIFIED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=95_000,
            after_difference=0,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.PASSED
        assert result.discrepancy_eliminated is True
        assert result.failed_checks == 0

    def test_no_discrepancy_passes(self):
        """No original discrepancy → verification passes."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=0,
            adjustment_amount=0,
            after_actual=90_000,
            after_difference=0,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.PASSED

    def test_verification_records_id(self):
        """Verification records unique verification ID."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result()
        result = engine.verify(exec_result)
        assert result.verification_id.startswith("VER-")
        assert len(result.verification_id) > 4

    def test_verification_records_execution_id(self):
        """Verification records the execution ID being verified."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result()
        result = engine.verify(exec_result)
        assert result.execution_id == exec_result.execution_id


# ─────────────────────────────────────────────────────────────────────────────
# Test: Discrepancy Remains
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationDiscrepancyRemains:
    """Test verification when discrepancy is not eliminated."""

    def test_discrepancy_reduced_not_eliminated_fails(self):
        """Discrepancy reduced but not eliminated → FAILED."""
        engine = ResolutionVerificationEngine()
        # Before: difference=5000, adjustment=3000 (partial)
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=3_000,
            after_actual=93_000,
            after_difference=2_000,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.discrepancy_eliminated is False
        assert result.difference_after == 2_000

    def test_discrepancy_increased_fails(self):
        """Discrepancy increased after execution → FAILED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=-2_000,  # Wrong direction
            after_actual=88_000,
            after_difference=7_000,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.discrepancy_eliminated is False
        assert result.difference_after == 7_000

    def test_discrepancy_check_fails(self):
        """DISCREPANCY_ELIMINATED check specifically fails."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=2_000,
            after_actual=92_000,
            after_difference=3_000,
        )
        result = engine.verify(exec_result)
        disc_checks = [c for c in result.checks if c.check_type == VerificationCheckType.DISCREPANCY_ELIMINATED]
        assert len(disc_checks) == 1
        assert disc_checks[0].result == CheckResult.FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Test: Incorrect Adjustment Amount
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationIncorrectAmount:
    """Test verification when wrong adjustment amount was applied."""

    def test_adjustment_too_small_fails(self):
        """Adjustment smaller than requested → FAILED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_mismatched_execution(requested=5_000, actual_applied=3_000)
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED

    def test_adjustment_too_large_fails(self):
        """Adjustment larger than requested → FAILED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_mismatched_execution(requested=5_000, actual_applied=8_000)
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED

    def test_adjustment_check_fails(self):
        """CORRECT_ADJUSTMENT check specifically fails."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_mismatched_execution(requested=5_000, actual_applied=3_000)
        result = engine.verify(exec_result)
        adj_checks = [c for c in result.checks if c.check_type == VerificationCheckType.CORRECT_ADJUSTMENT]
        assert len(adj_checks) == 1
        assert adj_checks[0].result == CheckResult.FAIL

    def test_adjustment_check_passes_when_correct(self):
        """CORRECT_ADJUSTMENT check passes when amounts match."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(adjustment_amount=5_000)
        result = engine.verify(exec_result)
        adj_checks = [c for c in result.checks if c.check_type == VerificationCheckType.CORRECT_ADJUSTMENT]
        assert len(adj_checks) == 1
        assert adj_checks[0].result == CheckResult.PASS


# ─────────────────────────────────────────────────────────────────────────────
# Test: Unexpected Financial Change
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationUnexpectedChange:
    """Test verification when unexpected financial changes occur."""

    def test_amount_consistency_fails_on_wrong_actual(self):
        """Amount consistency check fails when actual_amount is wrong."""
        engine = ResolutionVerificationEngine()
        # Before: actual=90000, adjustment=5000 → expected new actual=95000
        # But after_state shows actual=94000 (unexpected change)
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=94_000,  # Should be 95000
            after_difference=1_000,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        amt_checks = [c for c in result.checks if c.check_type == VerificationCheckType.AMOUNT_CONSISTENCY]
        assert len(amt_checks) == 1
        assert amt_checks[0].result == CheckResult.FAIL

    def test_total_adjustments_mismatch_fails(self):
        """Amount consistency fails when total_adjustments is wrong."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=95_000,
            after_difference=0,
            after_adjustments=10_000,  # Should be 5000
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test: Missing Adjustment
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationMissingAdjustment:
    """Test verification when adjustment was not applied."""

    def test_zero_adjustment_when_expected_fails(self):
        """Zero adjustment when one was expected → FAILED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_mismatched_execution(requested=5_000, actual_applied=0)
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.discrepancy_eliminated is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Multiple Check Failures
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationMultipleFailures:
    """Test verification with multiple simultaneous failures."""

    def test_multiple_checks_fail(self):
        """Multiple checks failing → FAILED with correct count."""
        engine = ResolutionVerificationEngine()
        # Wrong amount AND discrepancy remains
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=92_000,  # Should be 95000
            after_difference=3_000,
            after_adjustments=2_000,  # Should be 5000
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.failed_checks >= 2

    def test_all_checks_fail(self):
        """All checks failing → FAILED."""
        engine = ResolutionVerificationEngine()
        # Completely wrong: no adjustment, discrepancy remains
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=90_000,  # No change
            after_difference=5_000,
            after_adjustments=0,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.failed_checks >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Test: Financial Integrity
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFinancialIntegrity:
    """Test verification financial integrity checks."""

    def test_integer_paise_values(self):
        """All financial values are integer paise."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            adjustment_amount=5_000,
            after_actual=95_000,
            after_difference=0,
        )
        result = engine.verify(exec_result)
        int_checks = [c for c in result.checks if c.check_type == VerificationCheckType.FINANCIAL_INTEGRITY]
        assert len(int_checks) == 1
        assert int_checks[0].result == CheckResult.PASS

    def test_adjustment_is_integer(self):
        """Adjustment amount must be integer."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(adjustment_amount=5_000)
        result = engine.verify(exec_result)
        assert isinstance(exec_result.requested_adjustment_paise, int)
        assert isinstance(exec_result.actual_adjustment_paise, int)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Failed Verification → NEVER FINAL_SUCCESS
# ─────────────────────────────────────────────────────────────────────────────


class TestFailedVerificationNeverSuccess:
    """Verify failed verification is never reported as success."""

    @pytest.mark.parametrize("before_diff,adjustment,after_actual,after_diff", [
        (5_000, 3_000, 93_000, 2_000),   # Partial fix
        (5_000, 0, 90_000, 5_000),       # No adjustment
        (5_000, 8_000, 98_000, -3_000),   # Over-adjustment
        (10_000, 5_000, 95_000, 5_000),   # Under-adjustment
    ])
    def test_failed_never_passes(self, before_diff, adjustment, after_actual, after_diff):
        """Failed verification never reports PASSED."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=before_diff,
            adjustment_amount=adjustment,
            after_actual=after_actual,
            after_difference=after_diff,
        )
        result = engine.verify(exec_result)
        assert result.status != VerificationStatus.PASSED

    def test_failed_result_has_failed_checks(self):
        """Failed verification has at least one failed check."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=3_000,
            after_actual=93_000,
            after_difference=2_000,
        )
        result = engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.failed_checks > 0

    def test_failed_result_has_discrepancy_info(self):
        """Failed verification records discrepancy info."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=3_000,
            after_actual=93_000,
            after_difference=2_000,
        )
        result = engine.verify(exec_result)
        assert result.difference_before == 5_000
        assert result.difference_after == 2_000
        assert result.discrepancy_eliminated is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Rollback After Verification Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackAfterVerificationFailure:
    """Test rollback behavior after verification failure."""

    def test_rollback_triggered_on_verification_failure(self):
        """Verification failure → rollback can be initiated."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=95_000,
            after_difference=0,
        )
        result = rollback_service.rollback(
            exec_result,
            reason=RollbackReason.VERIFICATION_FAILED,
        )
        assert result.status in (RollbackStatus.ROLLED_BACK, RollbackStatus.ROLLBACK_FAILED, RollbackStatus.ESCALATED)

    def test_rollback_records_audit_trail(self):
        """Rollback records complete audit trail."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        assert len(result.audit_trail) >= 2

    def test_rollback_records_reason(self):
        """Rollback records the reason."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(
            exec_result,
            reason=RollbackReason.VERIFICATION_FAILED,
        )
        assert result.reason == RollbackReason.VERIFICATION_FAILED

    def test_rollback_records_execution_id(self):
        """Rollback records the execution ID."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        assert result.execution_id == exec_result.execution_id

    def test_rollback_reverses_adjustment(self):
        """Rollback reverses the adjustment when adjustment record exists."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result(adjustment_amount=5_000)
        # Add adjustment record for rollback to reverse
        from app.schemas.execution import AdjustmentRecord
        exec_result.adjustment = AdjustmentRecord(
            adjustment_id="ADJ-VERIFY-001",
            adjustment_type="FEE_CORRECTION",
            amount_paise=5_000,
            requested_amount_paise=5_000,
        )
        result = rollback_service.rollback(exec_result)
        assert result.adjustment_reversed is True
        assert result.reversal_amount_paise == 5_000

    def test_rollback_id_format(self):
        """Rollback ID has correct format."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        assert result.rollback_id.startswith("RBK-")

    def test_rollback_different_reasons(self):
        """Rollback supports different failure reasons."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        for reason in RollbackReason:
            result = rollback_service.rollback(exec_result, reason=reason)
            assert result.reason == reason


# ─────────────────────────────────────────────────────────────────────────────
# Test: Verification → Rollback → Escalation
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationRollbackEscalation:
    """Test the complete verification → rollback → escalation chain."""

    def test_verification_failure_leads_to_rollback(self):
        """Verification failure → rollback initiated."""
        # Verify
        v_engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=3_000,
            after_actual=93_000,
            after_difference=2_000,
        )
        v_result = v_engine.verify(exec_result)
        assert v_result.status == VerificationStatus.FAILED

        # Rollback
        r_service = RollbackService()
        r_result = r_service.rollback(
            exec_result,
            reason=RollbackReason.VERIFICATION_FAILED,
        )
        assert r_result.status != RollbackStatus.NOT_REQUIRED

    def test_rollback_audit_entries_correlate(self):
        """Rollback audit entries reference the same execution."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        for entry in result.audit_trail:
            assert entry.execution_id == exec_result.execution_id
            assert entry.exception_id == exec_result.exception_id

    def test_rollback_captures_before_state(self):
        """Rollback captures the state before rollback."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        assert result.before_rollback_state is not None

    def test_rollback_computes_expected_state(self):
        """Rollback computes expected state after rollback."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        from app.schemas.execution import AdjustmentRecord
        exec_result.adjustment = AdjustmentRecord(
            adjustment_id="ADJ-VERIFY-002",
            adjustment_type="FEE_CORRECTION",
            amount_paise=5_000,
            requested_amount_paise=5_000,
        )
        result = rollback_service.rollback(exec_result)
        assert result.expected_rollback_state is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Verification Check Types
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationCheckTypes:
    """Test all verification check types."""

    def test_all_check_types_present(self):
        """Verification runs all check types."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result(
            before_difference=5_000,
            adjustment_amount=5_000,
            after_actual=95_000,
            after_difference=0,
        )
        result = engine.verify(exec_result)
        check_types = {c.check_type for c in result.checks}
        expected_types = {
            VerificationCheckType.DISCREPANCY_ELIMINATED,
            VerificationCheckType.CORRECT_ADJUSTMENT,
            VerificationCheckType.NO_UNINTENDED_CHANGES,
            VerificationCheckType.AMOUNT_CONSISTENCY,
            VerificationCheckType.FINANCIAL_INTEGRITY,
        }
        assert expected_types.issubset(check_types)

    def test_check_result_enum(self):
        """CheckResult has PASS, FAIL, SKIP."""
        assert CheckResult.PASS.value == "PASS"
        assert CheckResult.FAIL.value == "FAIL"
        assert CheckResult.SKIP.value == "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Verification Cannot Execute Financial Actions
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationNoExecution:
    """Verify verification engine cannot execute financial actions."""

    def test_engine_has_no_execute_method(self):
        """VerificationEngine has no execute/apply/authorize method."""
        engine = ResolutionVerificationEngine()
        assert not hasattr(engine, "execute")
        assert not hasattr(engine, "apply")
        assert not hasattr(engine, "authorize")

    def test_rollback_has_no_execute_method(self):
        """RollbackService has no execute/apply/authorize method."""
        service = RollbackService()
        assert not hasattr(service, "execute")
        assert not hasattr(service, "apply")
        assert not hasattr(service, "authorize")

    def test_verification_result_no_financial_write_fields(self):
        """ResolutionVerificationResult has no financial write fields."""
        from app.schemas.resolution_verification import ResolutionVerificationResult
        fields = set(ResolutionVerificationResult.model_fields.keys())
        dangerous = {"execute", "apply", "authorize", "create_adjustment", "modify_records"}
        assert dangerous.isdisjoint(fields)

    def test_rollback_result_no_financial_write_fields(self):
        """RollbackResult has no financial write fields."""
        from app.schemas.rollback import RollbackResult
        fields = set(RollbackResult.model_fields.keys())
        dangerous = {"execute", "apply", "authorize", "create_adjustment", "modify_records"}
        assert dangerous.isdisjoint(fields)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Audit Trail Integrity
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationAuditTrail:
    """Test audit trail integrity for verification and rollback."""

    def test_verification_records_timestamp(self):
        """Verification records timestamp."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result()
        result = engine.verify(exec_result)
        assert result.verified_at is not None

    def test_verification_records_verifier(self):
        """Verification records who performed verification."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result()
        result = engine.verify(exec_result)
        assert result.verified_by == "resolution_verification_engine"

    def test_rollback_records_timestamp(self):
        """Rollback records completion timestamp."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        assert result.completed_at is not None

    def test_rollback_audit_entry_has_timestamp(self):
        """Each rollback audit entry has a timestamp."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        for entry in result.audit_trail:
            assert entry.timestamp is not None

    def test_verification_summary_includes_status(self):
        """Verification summary includes status."""
        engine = ResolutionVerificationEngine()
        exec_result = _make_execution_result()
        result = engine.verify(exec_result)
        summary = result.summary()
        assert result.status.value in summary

    def test_rollback_summary_includes_status(self):
        """Rollback summary includes status."""
        rollback_service = RollbackService()
        exec_result = _make_execution_result()
        result = rollback_service.rollback(exec_result)
        summary = result.summary()
        assert result.status.value in summary
