"""
Tests for Razorpay CloseLoop Phase 8C — Before/After Financial State.

Tests state comparison, intended/unintended change detection,
missing change detection, and integer paise integrity.
"""

import pytest
from app.schemas.execution import ExecutionResult, ExecutionStatus, FinancialStateSnapshot
from app.schemas.financial_diff import (
    ChangeType,
    FieldChange,
    FinancialStateDiff,
    RecordChange,
)
from app.services.financial_diff import FinancialDiffService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot(**overrides) -> FinancialStateSnapshot:
    """Build a financial state snapshot with defaults."""
    defaults = dict(
        exception_id="EXC-001",
        payment_amount=50000,
        expected_amount=47000,
        actual_amount=44000,
        difference=3000,
        total_refunds=0,
        total_fees=3000,
        total_taxes=0,
        total_adjustments=0,
        settlement_count=1,
        refund_count=0,
        fee_count=1,
        tax_count=0,
        adjustment_count=0,
    )
    defaults.update(overrides)
    return FinancialStateSnapshot(**defaults)


def _make_execution_result(**overrides) -> ExecutionResult:
    """Build an execution result for testing."""
    defaults = dict(
        execution_id="EXE-001",
        action_id="ACT-001",
        idempotency_key="key-001",
        workflow_id="WF-001",
        exception_id="EXC-001",
        resolution_type="APPLY_FEE_CORRECTION",
        authorization_source="AUTO_GUARDRAIL",
        before_state=_make_snapshot(),
        status=ExecutionStatus.EXECUTED,
        requested_adjustment_paise=3000,
        actual_adjustment_paise=3000,
    )
    defaults.update(overrides)
    return ExecutionResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_change_type_values(self):
        assert ChangeType.INTENDED.value == "INTENDED"
        assert ChangeType.UNINTENDED.value == "UNINTENDED"
        assert ChangeType.MISSING.value == "MISSING"
        assert ChangeType.NO_CHANGE.value == "NO_CHANGE"

    def test_field_change(self):
        fc = FieldChange(
            field_name="actual_amount",
            before_value=44000,
            after_value=47000,
            delta=3000,
            change_type=ChangeType.INTENDED,
        )
        assert fc.delta == 3000
        assert fc.change_type == ChangeType.INTENDED

    def test_record_change(self):
        rc = RecordChange(
            record_type="fee",
            before_count=1,
            after_count=1,
            delta=0,
            change_type=ChangeType.NO_CHANGE,
        )
        assert rc.delta == 0

    def test_diff_summary(self):
        diff = FinancialStateDiff(
            exception_id="EXC-001",
            field_changes=[FieldChange(field_name="x", before_value=0, after_value=100, delta=100, change_type=ChangeType.INTENDED)],
            intended_changes=[FieldChange(field_name="x", before_value=0, after_value=100, delta=100, change_type=ChangeType.INTENDED)],
        )
        s = diff.summary()
        assert "1 fields changed" in s
        assert "Intended: 1" in s


# ─────────────────────────────────────────────────────────────────────────────
# Exact Expected Change Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExactExpectedChange:
    def test_fee_correction_exact_change(self):
        """Fee correction with exact expected change → all intended."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000, total_adjustments=0)
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)

        assert diff.has_unintended_changes is False
        assert diff.has_missing_changes is False
        assert len(diff.unintended_changes) == 0
        assert len(diff.missing_changes) == 0

    def test_refund_exact_change(self):
        """Refund adjustment with exact change → all intended."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000, total_adjustments=0)
        after = _make_snapshot(actual_amount=46500, difference=500, total_adjustments=2500)
        exec_result = _make_execution_result(
            resolution_type="APPLY_REFUND_ADJUSTMENT",
            requested_adjustment_paise=2500,
        )

        diff = service.compare(before, after, exec_result)
        assert diff.has_unintended_changes is False


# ─────────────────────────────────────────────────────────────────────────────
# Partial Change Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialChange:
    def test_partial_fee_change(self):
        """Partial fee change → intended (partial is still intended)."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000)
        after = _make_snapshot(actual_amount=45500, difference=1500, total_adjustments=1500)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        # Partial change should be classified as intended
        assert len(diff.unintended_changes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Excessive Change Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExcessiveChange:
    def test_excessive_fee_change(self):
        """Excessive fee change → unintended excess."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000)
        after = _make_snapshot(actual_amount=50000, difference=-3000, total_adjustments=6000)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        assert diff.has_unintended_changes is True


# ─────────────────────────────────────────────────────────────────────────────
# Wrong Record Changed Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWrongRecordChanged:
    def test_unrelated_field_changed(self):
        """Payment amount changed → unintended."""
        service = FinancialDiffService()
        before = _make_snapshot(payment_amount=50000, actual_amount=44000, difference=3000)
        after = _make_snapshot(payment_amount=55000, actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        # payment_amount changed but is not in intended targets
        payment_changes = [fc for fc in diff.unintended_changes if fc.field_name == "payment_amount"]
        assert len(payment_changes) == 1

    def test_expected_amount_changed(self):
        """Expected amount changed → unintended."""
        service = FinancialDiffService()
        before = _make_snapshot(expected_amount=47000, actual_amount=44000, difference=3000)
        after = _make_snapshot(expected_amount=50000, actual_amount=47000, difference=3000, total_adjustments=3000)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        expected_changes = [fc for fc in diff.unintended_changes if fc.field_name == "expected_amount"]
        assert len(expected_changes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateAdjustment:
    def test_duplicate_adjustment_detected(self):
        """Double the intended adjustment → unintended excess."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000, total_adjustments=0)
        # After: applied 6000 instead of 3000
        after = _make_snapshot(actual_amount=50000, difference=-3000, total_adjustments=6000)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        assert diff.has_unintended_changes is True
        # The actual_amount change is excessive
        actual_changes = [fc for fc in diff.unintended_changes if fc.field_name == "actual_amount"]
        assert len(actual_changes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Unrelated Record Changed Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnrelatedRecordChanged:
    def test_refund_count_changed_for_fee_resolution(self):
        """Refund count changed during fee resolution → unintended."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000, refund_count=0)
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000, refund_count=1)
        exec_result = _make_execution_result(
            resolution_type="APPLY_FEE_CORRECTION",
            requested_adjustment_paise=3000,
        )

        diff = service.compare(before, after, exec_result)
        refund_changes = [rc for rc in diff.unintended_record_changes if rc.record_type == "refund"]
        assert len(refund_changes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# No Change Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoChange:
    def test_identical_states_zero_adjustment(self):
        """Identical before/after with zero adjustment → no changes."""
        service = FinancialDiffService()
        snapshot = _make_snapshot()
        exec_result = _make_execution_result(requested_adjustment_paise=0)

        diff = service.compare(snapshot, snapshot, exec_result)
        assert len(diff.field_changes) == 0
        assert diff.has_unintended_changes is False
        assert diff.has_missing_changes is False

    def test_identical_states_with_nonzero_request(self):
        """Identical states with nonzero request → missing changes detected."""
        service = FinancialDiffService()
        snapshot = _make_snapshot(actual_amount=44000, difference=3000)
        exec_result = _make_execution_result(requested_adjustment_paise=3000)

        diff = service.compare(snapshot, snapshot, exec_result)
        # No actual changes happened, but intended changes are missing
        assert diff.has_missing_changes is True
        assert len(diff.missing_changes) > 0

    def test_zero_adjustment(self):
        """Zero adjustment → no amount changes."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000)
        exec_result = _make_execution_result(requested_adjustment_paise=0, actual_adjustment_paise=0)

        diff = service.compare(before, before, exec_result)
        assert len(diff.field_changes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Mismatch Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSnapshotMismatch:
    def test_no_execution_result(self):
        """No execution result → all changes are unintended."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000)
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)

        diff = service.compare(before, after)
        assert diff.has_unintended_changes is True
        assert len(diff.unintended_changes) > 0

    def test_exception_id_preserved(self):
        """Exception ID is preserved in diff."""
        service = FinancialDiffService()
        before = _make_snapshot(exception_id="EXC-42")
        exec_result = _make_execution_result(exception_id="EXC-42")

        diff = service.compare(before, before, exec_result)
        assert diff.exception_id == "EXC-42"

    def test_execution_id_preserved(self):
        """Execution ID is preserved in diff."""
        service = FinancialDiffService()
        before = _make_snapshot()
        exec_result = _make_execution_result(execution_id="EXE-42")

        diff = service.compare(before, before, exec_result)
        assert diff.execution_id == "EXE-42"


# ─────────────────────────────────────────────────────────────────────────────
# Integer Paise Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegerPaise:
    def test_all_integer_values(self):
        """All integer values → integrity check passes."""
        service = FinancialDiffService()
        before = _make_snapshot()
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result()

        diff = service.compare(before, after, exec_result)
        assert diff.all_integer_paise is True

    def test_zero_values_are_integer(self):
        """Zero values are valid integers."""
        service = FinancialDiffService()
        before = _make_snapshot(payment_amount=0, actual_amount=0, difference=0)
        after = _make_snapshot(payment_amount=0, actual_amount=0, difference=0)

        diff = service.compare(before, after)
        assert diff.all_integer_paise is True


# ─────────────────────────────────────────────────────────────────────────────
# Field Delta Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFieldDelta:
    def test_delta_calculation(self):
        """Delta is correctly calculated as after - before."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000)
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result()

        diff = service.compare(before, after, exec_result)
        actual_change = [fc for fc in diff.field_changes if fc.field_name == "actual_amount"][0]
        assert actual_change.delta == 3000
        assert actual_change.before_value == 44000
        assert actual_change.after_value == 47000

    def test_negative_delta(self):
        """Negative delta when value decreases."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)
        after = _make_snapshot(actual_amount=44000, difference=3000, total_adjustments=0)

        diff = service.compare(before, after)
        actual_change = [fc for fc in diff.field_changes if fc.field_name == "actual_amount"][0]
        assert actual_change.delta == -3000

    def test_multiple_fields_changed(self):
        """Multiple fields change simultaneously."""
        service = FinancialDiffService()
        before = _make_snapshot(actual_amount=44000, difference=3000, total_adjustments=0)
        after = _make_snapshot(actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result()

        diff = service.compare(before, after, exec_result)
        changed_fields = {fc.field_name for fc in diff.field_changes}
        assert "actual_amount" in changed_fields
        assert "difference" in changed_fields
        assert "total_adjustments" in changed_fields


# ─────────────────────────────────────────────────────────────────────────────
# Record Count Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordCounts:
    def test_adjustment_count_increments(self):
        """Adjustment count increments on execution → intended."""
        service = FinancialDiffService()
        before = _make_snapshot(adjustment_count=0)
        after = _make_snapshot(adjustment_count=1, actual_amount=47000, difference=0, total_adjustments=3000)
        exec_result = _make_execution_result()

        diff = service.compare(before, after, exec_result)
        adj_change = [rc for rc in diff.record_changes if rc.record_type == "adjustment"][0]
        assert adj_change.delta == 1
        assert adj_change.change_type == ChangeType.INTENDED

    def test_no_record_count_changes(self):
        """No record count changes → all NO_CHANGE."""
        service = FinancialDiffService()
        snapshot = _make_snapshot()
        exec_result = _make_execution_result()

        diff = service.compare(snapshot, snapshot, exec_result)
        for rc in diff.record_changes:
            assert rc.change_type == ChangeType.NO_CHANGE
