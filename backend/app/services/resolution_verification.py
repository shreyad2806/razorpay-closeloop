"""
Resolution Verification Engine for Razorpay CloseLoop Phase 8D.

Independent verification that a resolution actually achieved its goal.

CRITICAL:
- Verification does NOT use ground truth
- Verification recalculates expected results independently
- Verification reads actual post-execution state
- Only verification success produces FINAL_SUCCESS

Ground truth may only be used later for offline evaluation.
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional

from app.schemas.execution import ExecutionResult, FinancialStateSnapshot
from app.schemas.financial_diff import FinancialStateDiff
from app.schemas.resolution_verification import (
    ActualFinancialResult,
    CheckResult,
    ExpectedFinancialResult,
    ResolutionVerificationResult,
    VerificationCheck,
    VerificationCheckType,
    VerificationStatus,
)
from app.services.financial_diff import FinancialDiffService


# ─────────────────────────────────────────────────────────────────────────────
# Verification Engine
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionVerificationEngine:
    """Independently verifies that a resolution achieved its goal.

    Uses deterministic financial calculations to verify:
    1. Discrepancy was eliminated
    2. Correct adjustment was applied
    3. No unintended changes occurred
    4. Affected records are correct
    """

    def __init__(self):
        self.diff_service = FinancialDiffService()

    def verify(
        self,
        execution_result: ExecutionResult,
        current_financial_state: Optional[FinancialStateSnapshot] = None,
    ) -> ResolutionVerificationResult:
        """Verify a resolution.

        Args:
            execution_result: The completed execution result
            current_financial_state: Current financial state (post-execution)

        Returns:
            ResolutionVerificationResult with all checks
        """
        start_time = time.perf_counter()
        verification_id = f"VER-{uuid.uuid4().hex[:8].upper()}"

        try:
            # ── Step 1: Recalculate expected result ──
            expected = self._recalculate_expected(execution_result)

            # ── Step 2: Read actual result ──
            actual = self._read_actual(execution_result, current_financial_state)

            # ── Step 3: Compute before/after diff ──
            diff = None
            if current_financial_state:
                diff = self.diff_service.compare(
                    execution_result.before_state,
                    current_financial_state,
                    execution_result,
                )

            # ── Step 4: Run all verification checks ──
            checks = self._run_checks(
                execution_result, expected, actual, diff
            )

            # ── Step 5: Determine overall status ──
            passed = sum(1 for c in checks if c.result == CheckResult.PASS)
            failed = sum(1 for c in checks if c.result == CheckResult.FAIL)
            status = VerificationStatus.PASSED if failed == 0 else VerificationStatus.FAILED

            # ── Step 6: Compute discrepancy elimination ──
            difference_before = execution_result.before_state.difference
            difference_after = actual.actual_new_difference
            discrepancy_eliminated = difference_after == 0

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ResolutionVerificationResult(
                verification_id=verification_id,
                execution_id=execution_result.execution_id,
                exception_id=execution_result.exception_id,
                status=status,
                expected_result=expected,
                actual_result=actual,
                difference_before=difference_before,
                difference_after=difference_after,
                discrepancy_eliminated=discrepancy_eliminated,
                has_unintended_changes=diff.has_unintended_changes if diff else False,
                unintended_change_count=len(diff.unintended_changes) if diff else 0,
                checks=checks,
                passed_checks=passed,
                failed_checks=failed,
                verified_at=datetime.utcnow(),
                verified_by="resolution_verification_engine",
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ResolutionVerificationResult(
                verification_id=verification_id,
                execution_id=execution_result.execution_id,
                exception_id=execution_result.exception_id,
                status=VerificationStatus.CALCULATION_ERROR,
                expected_result=ExpectedFinancialResult(),
                actual_result=ActualFinancialResult(),
                checks=[],
                verification_errors=[f"Verification calculation failed: {str(e)}"],
                verified_at=datetime.utcnow(),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Recalculate Expected Result
    # ─────────────────────────────────────────────────────────────────────────

    def _recalculate_expected(
        self, execution_result: ExecutionResult,
    ) -> ExpectedFinancialResult:
        """Recalculate expected financial outcome independently.

        Uses deterministic financial rules, NOT ground truth.
        NOT from the execution response.
        """
        before = execution_result.before_state
        requested = execution_result.requested_adjustment_paise
        resolution_type = execution_result.resolution_type.upper()

        # Recalculate what the result should be
        expected_new_actual = before.actual_amount + requested
        expected_new_difference = before.expected_amount - expected_new_actual
        expected_new_total_adjustments = before.total_adjustments + requested

        return ExpectedFinancialResult(
            expected_adjustment_paise=requested,
            expected_new_actual=expected_new_actual,
            expected_new_difference=expected_new_difference,
            expected_new_total_adjustments=expected_new_total_adjustments,
            resolution_type=execution_result.resolution_type,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Read Actual Result
    # ─────────────────────────────────────────────────────────────────────────

    def _read_actual(
        self,
        execution_result: ExecutionResult,
        current_state: Optional[FinancialStateSnapshot],
    ) -> ActualFinancialResult:
        """Read actual financial state after execution.

        If no current state provided, use execution's after_state.
        """
        if current_state:
            return ActualFinancialResult(
                actual_adjustment_paise=execution_result.actual_adjustment_paise,
                actual_new_actual=current_state.actual_amount,
                actual_new_difference=current_state.difference,
                actual_new_total_adjustments=current_state.total_adjustments,
            )

        # Fall back to execution's after_state
        after = execution_result.after_state
        if after:
            return ActualFinancialResult(
                actual_adjustment_paise=execution_result.actual_adjustment_paise,
                actual_new_actual=after.actual_amount,
                actual_new_difference=after.difference,
                actual_new_total_adjustments=after.total_adjustments,
            )

        # No state available
        return ActualFinancialResult(
            actual_adjustment_paise=execution_result.actual_adjustment_paise,
            actual_new_actual=execution_result.before_state.actual_amount,
            actual_new_difference=execution_result.before_state.difference,
            actual_new_total_adjustments=execution_result.before_state.total_adjustments,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Verification Checks
    # ─────────────────────────────────────────────────────────────────────────

    def _run_checks(
        self,
        execution_result: ExecutionResult,
        expected: ExpectedFinancialResult,
        actual: ActualFinancialResult,
        diff: Optional[FinancialStateDiff],
    ) -> List[VerificationCheck]:
        """Run all verification checks."""
        checks = []

        # Check 1: Discrepancy eliminated
        checks.append(self._check_discrepancy_eliminated(execution_result, expected, actual))

        # Check 2: Correct adjustment amount
        checks.append(self._check_correct_adjustment(expected, actual))

        # Check 3: No unintended changes
        checks.append(self._check_no_unintended_changes(diff))

        # Check 4: Amount consistency
        checks.append(self._check_amount_consistency(expected, actual))

        # Check 5: Financial integrity (integer paise)
        checks.append(self._check_financial_integrity(execution_result, diff))

        return checks

    def _check_discrepancy_eliminated(
        self,
        execution_result: ExecutionResult,
        expected: ExpectedFinancialResult,
        actual: ActualFinancialResult,
    ) -> VerificationCheck:
        """Check: original discrepancy is eliminated."""
        before_diff = execution_result.before_state.difference
        after_diff = actual.actual_new_difference

        if before_diff == 0:
            return VerificationCheck(
                check_type=VerificationCheckType.DISCREPANCY_ELIMINATED,
                result=CheckResult.PASS,
                expected=0,
                actual=after_diff,
                message="No original discrepancy to eliminate",
            )

        if after_diff == 0:
            return VerificationCheck(
                check_type=VerificationCheckType.DISCREPANCY_ELIMINATED,
                result=CheckResult.PASS,
                expected=0,
                actual=after_diff,
                message=f"Discrepancy {before_diff} eliminated",
            )

        return VerificationCheck(
            check_type=VerificationCheckType.DISCREPANCY_ELIMINATED,
            result=CheckResult.FAIL,
            expected=0,
            actual=after_diff,
            message=f"Discrepancy reduced from {before_diff} to {after_diff} but not eliminated",
        )

    def _check_correct_adjustment(
        self,
        expected: ExpectedFinancialResult,
        actual: ActualFinancialResult,
    ) -> VerificationCheck:
        """Check: correct adjustment amount was applied."""
        if expected.expected_adjustment_paise == actual.actual_adjustment_paise:
            return VerificationCheck(
                check_type=VerificationCheckType.CORRECT_ADJUSTMENT,
                result=CheckResult.PASS,
                expected=expected.expected_adjustment_paise,
                actual=actual.actual_adjustment_paise,
                message=f"Adjustment correct: {actual.actual_adjustment_paise} paise",
            )

        return VerificationCheck(
            check_type=VerificationCheckType.CORRECT_ADJUSTMENT,
            result=CheckResult.FAIL,
            expected=expected.expected_adjustment_paise,
            actual=actual.actual_adjustment_paise,
            message=f"Adjustment mismatch: expected {expected.expected_adjustment_paise}, got {actual.actual_adjustment_paise}",
        )

    def _check_no_unintended_changes(
        self,
        diff: Optional[FinancialStateDiff],
    ) -> VerificationCheck:
        """Check: no unintended financial changes."""
        if diff is None:
            return VerificationCheck(
                check_type=VerificationCheckType.NO_UNINTENDED_CHANGES,
                result=CheckResult.SKIP,
                message="No diff available to check",
            )

        if not diff.has_unintended_changes:
            return VerificationCheck(
                check_type=VerificationCheckType.NO_UNINTENDED_CHANGES,
                result=CheckResult.PASS,
                message="No unintended changes detected",
            )

        return VerificationCheck(
            check_type=VerificationCheckType.NO_UNINTENDED_CHANGES,
            result=CheckResult.FAIL,
            actual=diff.total_unintended_paise,
            message=f"{len(diff.unintended_changes)} unintended changes totaling {diff.total_unintended_paise} paise",
        )

    def _check_amount_consistency(
        self,
        expected: ExpectedFinancialResult,
        actual: ActualFinancialResult,
    ) -> VerificationCheck:
        """Check: actual amounts match expected recalculated amounts."""
        errors = []

        if actual.actual_new_actual != expected.expected_new_actual:
            errors.append(
                f"actual_amount: expected {expected.expected_new_actual}, got {actual.actual_new_actual}"
            )

        if actual.actual_new_total_adjustments != expected.expected_new_total_adjustments:
            errors.append(
                f"total_adjustments: expected {expected.expected_new_total_adjustments}, got {actual.actual_new_total_adjustments}"
            )

        if errors:
            return VerificationCheck(
                check_type=VerificationCheckType.AMOUNT_CONSISTENCY,
                result=CheckResult.FAIL,
                expected=expected.model_dump(),
                actual=actual.model_dump(),
                message="; ".join(errors),
            )

        return VerificationCheck(
            check_type=VerificationCheckType.AMOUNT_CONSISTENCY,
            result=CheckResult.PASS,
            message="All amounts consistent with expected",
        )

    def _check_financial_integrity(
        self,
        execution_result: ExecutionResult,
        diff: Optional[FinancialStateDiff],
    ) -> VerificationCheck:
        """Check: all financial values are integer paise."""
        if diff and not diff.all_integer_paise:
            return VerificationCheck(
                check_type=VerificationCheckType.FINANCIAL_INTEGRITY,
                result=CheckResult.FAIL,
                message="Non-integer financial values detected",
            )

        # Also check the adjustment amounts
        if execution_result.adjustment:
            adj = execution_result.adjustment
            if not isinstance(adj.amount_paise, int) or adj.amount_paise != int(adj.amount_paise):
                return VerificationCheck(
                    check_type=VerificationCheckType.FINANCIAL_INTEGRITY,
                    result=CheckResult.FAIL,
                    message=f"Adjustment amount is not integer: {adj.amount_paise}",
                )

        return VerificationCheck(
            check_type=VerificationCheckType.FINANCIAL_INTEGRITY,
            result=CheckResult.PASS,
            message="All financial values are integer paise",
        )
