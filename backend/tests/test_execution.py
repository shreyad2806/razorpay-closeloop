"""
Tests for Razorpay CloseLoop Phase 8A — Resolution Execution Foundation.

Tests execution service, idempotency, precondition validation,
before/after state capture, and adjustment calculation.
"""

import pytest
from app.schemas.execution import (
    AdjustmentRecord,
    ExecutionResult,
    ExecutionStatus,
    FinancialStateSnapshot,
)
from app.services.execution import ResolutionExecutionService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_action_request(**overrides) -> dict:
    """Build a valid action request."""
    base = {
        "action_id": "ACT-001",
        "idempotency_key": "key-001",
        "workflow_id": "WF-001",
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "candidate_id": "CAND-001",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "financial_adjustment_paise": 3000,
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
        "guardrail_confidence": 0.85,
        "evidence_summary": {"coverage": 0.9},
    }
    base.update(overrides)
    return base


def _make_financial_state(**overrides) -> dict:
    """Build a financial state snapshot."""
    base = {
        "payment_amount": 50000,
        "expected_amount": 47000,
        "actual_amount": 44000,
        "difference": 3000,
        "total_refunds": 0,
        "total_fees": 3000,
        "total_taxes": 0,
        "total_adjustments": 0,
        "settlement_count": 1,
        "refund_count": 0,
        "fee_count": 1,
        "tax_count": 0,
        "adjustment_count": 0,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_execution_status_values(self):
        assert ExecutionStatus.NOT_EXECUTED.value == "NOT_EXECUTED"
        assert ExecutionStatus.EXECUTING.value == "EXECUTING"
        assert ExecutionStatus.EXECUTED.value == "EXECUTED"
        assert ExecutionStatus.EXECUTION_FAILED.value == "EXECUTION_FAILED"
        assert ExecutionStatus.VERIFICATION_PENDING.value == "VERIFICATION_PENDING"
        assert ExecutionStatus.VERIFIED.value == "VERIFIED"
        assert ExecutionStatus.VERIFICATION_FAILED.value == "VERIFICATION_FAILED"
        assert ExecutionStatus.ROLLED_BACK.value == "ROLLED_BACK"
        assert ExecutionStatus.ESCALATED.value == "ESCALATED"

    def test_financial_state_snapshot(self):
        snapshot = FinancialStateSnapshot(
            exception_id="EXC-001",
            payment_amount=50000,
            expected_amount=47000,
            actual_amount=44000,
            difference=3000,
        )
        assert snapshot.exception_id == "EXC-001"
        assert snapshot.payment_amount == 50000
        assert snapshot.captured_at is not None

    def test_adjustment_record(self):
        adj = AdjustmentRecord(
            adjustment_id="ADJ-001",
            adjustment_type="FEE_REVERSAL",
            amount_paise=3000,
            requested_amount_paise=3000,
        )
        assert adj.adjustment_id == "ADJ-001"
        assert adj.amount_paise == 3000

    def test_execution_result_summary(self):
        result = ExecutionResult(
            execution_id="EXE-001",
            action_id="ACT-001",
            idempotency_key="key-001",
            workflow_id="WF-001",
            exception_id="EXC-001",
            resolution_type="APPLY_FEE_CORRECTION",
            authorization_source="AUTO_GUARDRAIL",
            before_state=FinancialStateSnapshot(exception_id="EXC-001"),
            status=ExecutionStatus.EXECUTED,
            requested_adjustment_paise=3000,
            actual_adjustment_paise=3000,
        )
        s = result.summary()
        assert "EXE-001" in s
        assert "EXECUTED" in s
        assert "3000" in s


# ─────────────────────────────────────────────────────────────────────────────
# Valid Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidExecution:
    def test_valid_execution_succeeds(self):
        """Valid request with all preconditions → EXECUTED."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        state = _make_financial_state()

        result = service.execute(request, state)

        assert result.status == ExecutionStatus.EXECUTED
        assert result.adjustment is not None
        assert result.adjustment.amount_paise == 3000
        assert result.before_state is not None
        assert result.after_state is not None
        assert result.executed_at is not None

    def test_execution_captures_before_state(self):
        """Before state captures financial snapshot."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        state = _make_financial_state(payment_amount=50000, difference=3000)

        result = service.execute(request, state)

        assert result.before_state.payment_amount == 50000
        assert result.before_state.difference == 3000
        assert result.before_state.snapshot_reason == "pre_execution"

    def test_execution_captures_after_state(self):
        """After state reflects the adjustment."""
        service = ResolutionExecutionService()
        request = _make_action_request(
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
        )
        state = _make_financial_state(actual_amount=44000)

        result = service.execute(request, state)

        assert result.after_state is not None
        assert result.after_state.snapshot_reason == "post_execution"
        # Fee correction adds to actual amount
        assert result.after_state.actual_amount == 44000 + 3000

    def test_adjustment_calculation(self):
        """Actual adjustment matches requested."""
        service = ResolutionExecutionService()
        request = _make_action_request(financial_adjustment_paise=5000)
        result = service.execute(request)

        assert result.requested_adjustment_paise == 5000
        assert result.actual_adjustment_paise == 5000

    def test_metadata_persisted(self):
        """Decision metadata is stored in result."""
        service = ResolutionExecutionService()
        request = _make_action_request(
            guardrail_decision="AUTO",
            guardrail_confidence=0.85,
        )
        result = service.execute(request)

        assert result.decision == "AUTO"
        assert result.confidence == 0.85
        assert result.workflow_id == "WF-001"
        assert result.exception_id == "EXC-001"

    def test_execution_has_unique_id(self):
        """Each execution gets a unique ID."""
        service = ResolutionExecutionService()
        r1 = service.execute(_make_action_request(idempotency_key="key-1"))
        r2 = service.execute(_make_action_request(idempotency_key="key-2"))
        assert r1.execution_id != r2.execution_id

    def test_human_approved_execution(self):
        """HUMAN_REVIEW + HUMAN_APPROVAL → EXECUTED."""
        service = ResolutionExecutionService()
        request = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
        )
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTED


# ─────────────────────────────────────────────────────────────────────────────
# Precondition Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPreconditions:
    def test_missing_workflow_id(self):
        """Missing workflow_id → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(workflow_id="")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "workflow_id" in result.error

    def test_missing_exception_id(self):
        """Missing exception_id → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(exception_id="")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "exception_id" in result.error

    def test_missing_idempotency_key(self):
        """Missing idempotency_key → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(idempotency_key="")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED

    def test_guardrail_blocked(self):
        """UNRESOLVED guardrail → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(guardrail_decision="UNRESOLVED")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "does not allow execution" in result.error

    def test_no_authorization(self):
        """No authorization → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(authorization_source="NONE")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "authorization" in result.error.lower()

    def test_verification_not_passed(self):
        """Verification not passed → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(verification_passed=False)
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        assert "verification" in result.error.lower()

    def test_missing_adjustment(self):
        """Missing adjustment amount → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(financial_adjustment_paise=None)
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED

    def test_missing_resolution_type(self):
        """Missing resolution type → EXECUTION_FAILED."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="")
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_request_returns_same_result(self):
        """Same idempotency key → same result."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        r1 = service.execute(request)
        r2 = service.execute(request)
        assert r1.execution_id == r2.execution_id
        assert r1.status == r2.status

    def test_same_request_no_duplicate_adjustment(self):
        """Repeated request → no duplicate financial adjustment."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        r1 = service.execute(request)
        r2 = service.execute(request)
        assert r1.actual_adjustment_paise == r2.actual_adjustment_paise

    def test_different_keys_different_results(self):
        """Different idempotency keys → different results."""
        service = ResolutionExecutionService()
        r1 = service.execute(_make_action_request(idempotency_key="key-1"))
        r2 = service.execute(_make_action_request(idempotency_key="key-2"))
        assert r1.execution_id != r2.execution_id

    def test_has_executed_check(self):
        """has_executed returns correct state."""
        service = ResolutionExecutionService()
        assert not service.has_executed("key-1")
        service.execute(_make_action_request(idempotency_key="key-1"))
        assert service.has_executed("key-1")

    def test_get_execution(self):
        """get_execution returns previous result."""
        service = ResolutionExecutionService()
        request = _make_action_request(idempotency_key="key-1")
        r1 = service.execute(request)
        r2 = service.get_execution("key-1")
        assert r2 is not None
        assert r2.execution_id == r1.execution_id

    def test_failed_execution_not_idempotent(self):
        """Failed execution is NOT stored for idempotency."""
        service = ResolutionExecutionService()
        request = _make_action_request(guardrail_decision="UNRESOLVED")
        r1 = service.execute(request)
        assert r1.status == ExecutionStatus.EXECUTION_FAILED
        # Second call should also fail (not return cached failure)
        r2 = service.execute(request)
        assert r2.status == ExecutionStatus.EXECUTION_FAILED
        assert r1.execution_id != r2.execution_id  # different execution IDs


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentTypes:
    def test_fee_correction_type(self):
        """Fee correction → FEE_REVERSAL adjustment."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="APPLY_FEE_CORRECTION")
        result = service.execute(request)
        assert result.adjustment.adjustment_type == "FEE_REVERSAL"

    def test_refund_type(self):
        """Refund → REFUND adjustment."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="APPLY_REFUND_ADJUSTMENT")
        result = service.execute(request)
        assert result.adjustment.adjustment_type == "REFUND"

    def test_tax_type(self):
        """Tax → TAX_ADJUSTMENT."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="APPLY_TAX_CORRECTION")
        result = service.execute(request)
        assert result.adjustment.adjustment_type == "TAX_ADJUSTMENT"

    def test_generic_type(self):
        """Generic → CORRECTION."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="GENERIC_ADJUSTMENT")
        result = service.execute(request)
        assert result.adjustment.adjustment_type == "CORRECTION"

    def test_affected_records_fee(self):
        """Fee correction affects fee records."""
        service = ResolutionExecutionService()
        request = _make_action_request(resolution_type="APPLY_FEE_CORRECTION")
        result = service.execute(request)
        assert len(result.adjustment.affected_records) > 0

    def test_zero_adjustment(self):
        """Zero adjustment → EXECUTED with zero amount."""
        service = ResolutionExecutionService()
        request = _make_action_request(financial_adjustment_paise=0)
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTED
        assert result.actual_adjustment_paise == 0


# ─────────────────────────────────────────────────────────────────────────────
# State Snapshot Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStateSnapshots:
    def test_before_state_default(self):
        """Before state with no financial data → zeros."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        result = service.execute(request)
        assert result.before_state.payment_amount == 0
        assert result.before_state.settlement_count == 0

    def test_before_state_captured(self):
        """Before state captures provided financial data."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        state = _make_financial_state(payment_amount=100000)
        result = service.execute(request, state)
        assert result.before_state.payment_amount == 100000

    def test_after_state_none_on_failure(self):
        """After state is None when execution fails."""
        service = ResolutionExecutionService()
        request = _make_action_request(guardrail_decision="UNRESOLVED")
        result = service.execute(request)
        assert result.after_state is None

    def test_after_state_reflects_adjustment(self):
        """After state difference is recalculated."""
        service = ResolutionExecutionService()
        request = _make_action_request(financial_adjustment_paise=3000)
        state = _make_financial_state(expected_amount=47000, actual_amount=44000)
        result = service.execute(request, state)
        # After: actual = 44000 + 3000 = 47000, diff = 47000 - 47000 = 0
        assert result.after_state.actual_amount == 47000
        assert result.after_state.difference == 0

    def test_after_state_adjustment_count_increments(self):
        """After state has one more adjustment than before."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        state = _make_financial_state(adjustment_count=2)
        result = service.execute(request, state)
        assert result.after_state.adjustment_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_financial_state(self):
        """Execution without financial state → uses defaults."""
        service = ResolutionExecutionService()
        request = _make_action_request()
        result = service.execute(request, None)
        assert result.status == ExecutionStatus.EXECUTED
        assert result.before_state.payment_amount == 0

    def test_execution_has_timestamps(self):
        """Execution result has created_at and executed_at."""
        service = ResolutionExecutionService()
        result = service.execute(_make_action_request())
        assert result.created_at is not None
        assert result.executed_at is not None

    def test_failed_execution_has_no_adjustment(self):
        """Failed execution → no adjustment record."""
        service = ResolutionExecutionService()
        request = _make_action_request(guardrail_decision="UNRESOLVED")
        result = service.execute(request)
        assert result.adjustment is None
        assert result.actual_adjustment_paise == 0

    def test_multiple_precondition_failures(self):
        """Multiple precondition failures → single error message."""
        service = ResolutionExecutionService()
        request = _make_action_request(
            workflow_id="",
            exception_id="",
            guardrail_decision="UNRESOLVED",
            verification_passed=False,
        )
        result = service.execute(request)
        assert result.status == ExecutionStatus.EXECUTION_FAILED
        # Error should mention multiple issues
        assert result.error is not None

    def test_request_preserved_in_result(self):
        """Action request fields preserved in result."""
        service = ResolutionExecutionService()
        request = _make_action_request(
            case_id="CASE-001",
            candidate_id="CAND-001",
            authorization_source="HUMAN_APPROVAL",
        )
        result = service.execute(request)
        assert result.case_id == "CASE-001"
        assert result.candidate_id == "CAND-001"
        assert result.authorization_source == "HUMAN_APPROVAL"
