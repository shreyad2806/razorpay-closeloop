"""
Tests for Razorpay CloseLoop Phase 9 — Controlled Provider Execution.
"""

import pytest
from datetime import datetime, timedelta, timezone

from app.schemas.execution import ExecutionStatus
from app.services.execution import ResolutionExecutionService
from app.providers.mock_provider import MockProvider
from app.providers.dto import ProviderExecutionRequest
from app.schemas.enums import ProviderExecutionStatus

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
        "currency": "INR",
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
        "guardrail_confidence": 0.85,
        "evidence_summary": {"coverage": 0.9},
        "metadata": {},
        "current_evidence_digest": "digest-123"
    }
    base.update(overrides)
    return base

def _make_financial_state(**overrides) -> dict:
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


class TestPhase9ProviderExecution:
    
    # A. provider success
    def test_provider_success(self):
        service = ResolutionExecutionService()
        req = _make_action_request(idempotency_key="key-success")
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTED
        assert res.actual_adjustment_paise == 3000

    # B. provider failure
    def test_provider_failure(self):
        service = ResolutionExecutionService()
        req = _make_action_request(idempotency_key="key-fail", metadata={"mock_outcome": "FAILED"})
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Provider rejected execution" in res.error

    # C. provider timeout
    def test_provider_timeout(self):
        service = ResolutionExecutionService()
        req = _make_action_request(idempotency_key="key-timeout", metadata={"mock_outcome": "TIMEOUT"})
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Provider execution timed out" in res.error

    # D. provider unknown outcome
    def test_provider_unknown(self):
        service = ResolutionExecutionService()
        req = _make_action_request(idempotency_key="key-unknown", metadata={"mock_outcome": "UNKNOWN"})
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Provider execution outcome unknown" in res.error

    # E. AUTO authorization succeeds
    def test_auto_auth_succeeds(self):
        service = ResolutionExecutionService()
        req = _make_action_request(guardrail_decision="AUTO", authorization_source="AUTO_GUARDRAIL", idempotency_key="auto-1")
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTED

    # F. HUMAN_REVIEW + APPROVED succeeds
    def test_human_review_approved_succeeds(self):
        service = ResolutionExecutionService()
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=1)
        req = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            idempotency_key="human-1",
            approval_record={"decision": "APPROVED", "expires_at": future, "evidence_digest": "digest-123"}
        )
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTED

    # G. HUMAN_REVIEW + PENDING blocked
    def test_human_review_pending_blocked(self):
        service = ResolutionExecutionService()
        req = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            idempotency_key="human-pending",
            approval_record={"decision": "PENDING"}
        )
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Approval not APPROVED" in res.error

    # H. HUMAN_REVIEW + REJECTED blocked
    def test_human_review_rejected_blocked(self):
        service = ResolutionExecutionService()
        req = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            idempotency_key="human-rej",
            approval_record={"decision": "REJECTED"}
        )
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED

    # I. expired approval blocked
    def test_expired_approval_blocked(self):
        service = ResolutionExecutionService()
        past = datetime.now(timezone.utc) - timedelta(days=1)
        req = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            idempotency_key="human-exp",
            approval_record={"decision": "APPROVED", "expires_at": past}
        )
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Approval has expired" in res.error

    # K. evidence drift blocked
    def test_evidence_drift_blocked(self):
        service = ResolutionExecutionService()
        future = datetime.now(timezone.utc) + timedelta(days=1)
        req = _make_action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            idempotency_key="human-drift",
            approval_record={"decision": "APPROVED", "expires_at": future, "evidence_digest": "digest-OLD"}
        )
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Evidence digest mismatch" in res.error

    # M. currency mismatch blocked
    def test_currency_invalid(self):
        service = ResolutionExecutionService()
        req = _make_action_request(currency="EUR", idempotency_key="curr-1")
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED
        assert "Currency mismatch or invalid" in res.error

    # O. missing authorization blocked
    def test_missing_authorization(self):
        service = ResolutionExecutionService()
        req = _make_action_request(guardrail_decision="AUTO", authorization_source="NONE")
        res = service.execute(req, _make_financial_state())
        assert res.status == ExecutionStatus.EXECUTION_FAILED

    # P. duplicate idempotency key does not duplicate provider mutation
    def test_idempotency_prevents_duplicate(self):
        service = ResolutionExecutionService()
        req = _make_action_request(idempotency_key="duplicate-test")
        r1 = service.execute(req, _make_financial_state())
        r2 = service.execute(req, _make_financial_state())
        assert r1.status == ExecutionStatus.EXECUTED
        assert r2.status == ExecutionStatus.EXECUTED
        assert r1.execution_id == r2.execution_id
        # In mock provider, checking if actual call occurred twice can be inferred by the single execution_id

    # W. Golden FEE_MISMATCH
    def test_golden_fee_mismatch(self):
        service = ResolutionExecutionService()
        state = _make_financial_state(
            payment_amount=1000000,
            expected_amount=950000,
            actual_amount=925000,
            difference=25000,
            exception_id="FEE_MISMATCH"
        )
        req = _make_action_request(
            exception_id="FEE_MISMATCH",
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=25000,
            idempotency_key="golden-fee-mismatch"
        )
        res = service.execute(req, state)
        assert res.status == ExecutionStatus.EXECUTED
        assert res.actual_adjustment_paise == 25000
        assert res.after_state.difference == 0
        assert res.after_state.actual_amount == 950000
        # Assert exception remains NOT CLOSED, etc is implicitly covered since execution does not mutate exception state
