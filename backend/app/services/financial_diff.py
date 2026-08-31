"""
Financial State Comparison Service for Razorpay CloseLoop Phase 8C.

Compares before/after financial state snapshots to determine:
- intended changes (from the authorized resolution)
- actual changes (what really happened)
- missing changes (intended but not applied)
- unintended changes (unexpected modifications)

The expected change comes from the authorized resolution candidate,
NEVER from ground truth.
"""

from typing import List, Optional

from app.schemas.execution import ExecutionResult, FinancialStateSnapshot
from app.schemas.financial_diff import (
    ChangeType,
    FieldChange,
    FinancialStateDiff,
    RecordChange,
)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Service
# ─────────────────────────────────────────────────────────────────────────────


class FinancialDiffService:
    """Compares before/after financial state snapshots.

    Determines what changed, whether changes were intended,
    and detects unintended modifications.
    """

    # Fields that represent financial amounts (integer paise)
    AMOUNT_FIELDS = [
        "payment_amount",
        "expected_amount",
        "actual_amount",
        "difference",
        "total_refunds",
        "total_fees",
        "total_taxes",
        "total_adjustments",
    ]

    # Fields that represent record counts
    COUNT_FIELDS = [
        "settlement_count",
        "refund_count",
        "fee_count",
        "tax_count",
        "adjustment_count",
    ]

    def compare(
        self,
        before: FinancialStateSnapshot,
        after: FinancialStateSnapshot,
        execution_result: Optional[ExecutionResult] = None,
    ) -> FinancialStateDiff:
        """Compare before and after financial state.

        Args:
            before: Financial state before execution
            after: Financial state after execution
            execution_result: Optional execution result for context

        Returns:
            FinancialStateDiff with full comparison
        """
        # ── Compute field-level changes ──
        field_changes = self._compute_field_changes(before, after)

        # ── Compute record count changes ──
        record_changes = self._compute_record_changes(before, after)

        # ── Determine intended changes from resolution ──
        intended = self._determine_intended_changes(
            field_changes, execution_result
        )

        # ── Classify changes ──
        intended_fields = [fc for fc in field_changes if fc.change_type == ChangeType.INTENDED]
        unintended_fields = [fc for fc in field_changes if fc.change_type == ChangeType.UNINTENDED]
        missing_fields = [fc for fc in field_changes if fc.change_type == ChangeType.MISSING]

        # ── Classify record changes ──
        unintended_records = self._classify_record_changes(
            record_changes, execution_result
        )

        # ── Calculate totals ──
        total_intended = sum(abs(fc.delta) for fc in intended_fields)
        total_unintended = sum(abs(fc.delta) for fc in unintended_fields)

        # ── Check integer paise integrity ──
        all_integer = self._verify_integer_paise(before, after)

        return FinancialStateDiff(
            exception_id=before.exception_id,
            execution_id=execution_result.execution_id if execution_result else None,
            field_changes=field_changes,
            record_changes=record_changes,
            intended_changes=intended_fields,
            unintended_changes=unintended_fields,
            missing_changes=missing_fields,
            unintended_record_changes=unintended_records,
            total_intended_paise=total_intended,
            total_unintended_paise=total_unintended,
            has_unintended_changes=len(unintended_fields) > 0 or len(unintended_records) > 0,
            has_missing_changes=len(missing_fields) > 0,
            all_integer_paise=all_integer,
            resolution_type=execution_result.resolution_type if execution_result else None,
            requested_adjustment_paise=execution_result.requested_adjustment_paise if execution_result else 0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Field Change Computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_field_changes(
        self,
        before: FinancialStateSnapshot,
        after: FinancialStateSnapshot,
    ) -> List[FieldChange]:
        """Compute changes for all financial amount fields."""
        changes = []
        for field_name in self.AMOUNT_FIELDS:
            before_val = getattr(before, field_name)
            after_val = getattr(after, field_name)
            delta = after_val - before_val

            if delta != 0:
                changes.append(FieldChange(
                    field_name=field_name,
                    before_value=before_val,
                    after_value=after_val,
                    delta=delta,
                    change_type=ChangeType.NO_CHANGE,  # classified later
                ))

        return changes

    # ─────────────────────────────────────────────────────────────────────────
    # Record Change Computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_record_changes(
        self,
        before: FinancialStateSnapshot,
        after: FinancialStateSnapshot,
    ) -> List[RecordChange]:
        """Compute changes for record counts."""
        changes = []
        for field_name in self.COUNT_FIELDS:
            before_val = getattr(before, field_name)
            after_val = getattr(after, field_name)
            delta = after_val - before_val

            record_type = field_name.replace("_count", "")
            changes.append(RecordChange(
                record_type=record_type,
                before_count=before_val,
                after_count=after_val,
                delta=delta,
                change_type=ChangeType.NO_CHANGE,  # classified later
            ))

        return changes

    # ─────────────────────────────────────────────────────────────────────────
    # Intended Change Determination
    # ─────────────────────────────────────────────────────────────────────────

    def _determine_intended_changes(
        self,
        field_changes: List[FieldChange],
        execution_result: Optional[ExecutionResult],
    ) -> List[FieldChange]:
        """Determine which changes were intended by the resolution.

        The expected change comes from the authorized resolution candidate,
        NEVER from ground truth.
        """
        if not execution_result:
            # No execution context — all changes are unintended
            for fc in field_changes:
                fc.change_type = ChangeType.UNINTENDED
            return field_changes

        resolution_type = execution_result.resolution_type.upper()
        requested = execution_result.requested_adjustment_paise

        # Determine which fields should change based on resolution type
        intended_fields = self._get_intended_field_targets(resolution_type, requested)

        for fc in field_changes:
            if fc.field_name in intended_fields:
                expected_delta = intended_fields[fc.field_name]
                if fc.delta == expected_delta:
                    fc.change_type = ChangeType.INTENDED
                elif abs(fc.delta) > abs(expected_delta):
                    # Changed more than expected — partially intended + excess
                    fc.change_type = ChangeType.UNINTENDED
                else:
                    # Changed less than expected — partial
                    fc.change_type = ChangeType.INTENDED  # partial is still intended
            else:
                # Field not in intended targets — unintended change
                fc.change_type = ChangeType.UNINTENDED

        # Check for missing intended changes
        changed_fields = {fc.field_name for fc in field_changes}
        for field_name, expected_delta in intended_fields.items():
            if field_name not in changed_fields and expected_delta != 0:
                # Intended change did not happen
                field_changes.append(FieldChange(
                    field_name=field_name,
                    before_value=0,  # would need actual before value
                    after_value=0,
                    delta=0,
                    change_type=ChangeType.MISSING,
                ))

        return field_changes

    def _get_intended_field_targets(
        self,
        resolution_type: str,
        requested_paise: int,
    ) -> dict:
        """Determine which fields should change for a given resolution type.

        Returns dict of {field_name: expected_delta}.
        """
        targets = {}

        if "FEE" in resolution_type:
            # Fee adjustment: actual_amount increases, total_adjustments increases
            targets["actual_amount"] = requested_paise
            targets["total_adjustments"] = requested_paise
            targets["difference"] = -requested_paise  # difference decreases

        elif "REFUND" in resolution_type:
            # Refund adjustment: actual_amount increases, total_adjustments increases
            targets["actual_amount"] = requested_paise
            targets["total_adjustments"] = requested_paise
            targets["difference"] = -requested_paise

        elif "TAX" in resolution_type:
            # Tax adjustment
            targets["actual_amount"] = requested_paise
            targets["total_adjustments"] = requested_paise
            targets["difference"] = -requested_paise

        else:
            # Generic adjustment
            targets["actual_amount"] = requested_paise
            targets["total_adjustments"] = requested_paise
            targets["difference"] = -requested_paise

        return targets

    # ─────────────────────────────────────────────────────────────────────────
    # Record Change Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_record_changes(
        self,
        record_changes: List[RecordChange],
        execution_result: Optional[ExecutionResult],
    ) -> List[RecordChange]:
        """Classify record count changes as intended or unintended."""
        unintended = []

        if not execution_result:
            for rc in record_changes:
                if rc.delta != 0:
                    rc.change_type = ChangeType.UNINTENDED
                    unintended.append(rc)
            return unintended

        resolution_type = execution_result.resolution_type.upper()

        for rc in record_changes:
            if rc.delta == 0:
                rc.change_type = ChangeType.NO_CHANGE
                continue

            # Determine if this record type change is expected
            expected_record_type = None
            if "FEE" in resolution_type:
                expected_record_type = "fee"  # adjustment_count
            elif "REFUND" in resolution_type:
                expected_record_type = "refund"

            if rc.record_type == "adjustment":
                # Adjustment count always increases on execution
                rc.change_type = ChangeType.INTENDED
            elif rc.record_type == expected_record_type:
                rc.change_type = ChangeType.INTENDED
            else:
                rc.change_type = ChangeType.UNINTENDED
                unintended.append(rc)

        return unintended

    # ─────────────────────────────────────────────────────────────────────────
    # Integer Paise Verification
    # ─────────────────────────────────────────────────────────────────────────

    def _verify_integer_paise(
        self,
        before: FinancialStateSnapshot,
        after: FinancialStateSnapshot,
    ) -> bool:
        """Verify all financial values are integers (paise)."""
        for snapshot in [before, after]:
            for field_name in self.AMOUNT_FIELDS:
                val = getattr(snapshot, field_name)
                if not isinstance(val, int):
                    return False
                # Check it's a whole number (no fractional part)
                if val != int(val):
                    return False
        return True
