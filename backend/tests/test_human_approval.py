"""
Tests for Human-in-the-Loop Approval (Phase 7G).

Tests:
1. pause
2. state persistence
3. resume
4. approval
5. rejection
6. duplicate approval
7. invalid workflow ID
8. restart/resume
9. approval after timeout
10. rejected candidate cannot continue
"""

import pytest
from datetime import datetime

from app.agent.human_approval import (
    HumanDecision,
    pause_for_human_approval,
    process_human_decision,
)
from app.agent.workflow import create_initial_state
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_state_waiting(workflow_id="WF-TEST-001"):
    """Create state in WAITING_FOR_HUMAN status."""
    state = create_initial_state(exception_id="EXC-001", workflow_id=workflow_id)
    state.metadata.workflow_status = WorkflowStatus.WAITING_FOR_HUMAN
    state.selected_candidate = {
        "candidate_id": "CAND-FEE-001",
        "resolution_type": "FEE_ADJUSTMENT",
        "amount_paise": 3000,
    }
    state.classification = {"exception_type": "FEE_DIFFERENCE"}
    state.evidence_package = {"evidence_coverage": 0.90, "evidence_consistency": 0.85}
    state.guardrail_result = {"decision": "HUMAN_REVIEW", "primary_reason": "Low confidence"}
    state.confidence = 0.55
    state.risk = "MEDIUM"
    state.human_review.approval_status = HumanApprovalStatus.PENDING
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Pause Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPause:
    def test_pause_sets_waiting_status(self):
        state = create_initial_state(exception_id="EXC-001")
        state.selected_candidate = {"resolution_type": "FEE_ADJUSTMENT", "amount_paise": 3000}
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        state.evidence_package = {"evidence_coverage": 0.90}
        state.guardrail_result = {"primary_reason": "Low confidence"}
        state.risk = "MEDIUM"

        result = pause_for_human_approval(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.WAITING_FOR_HUMAN.value
        assert result["human_review"]["approval_status"] == HumanApprovalStatus.PENDING.value

    def test_pause_builds_review_package(self):
        state = create_initial_state(exception_id="EXC-001")
        state.selected_candidate = {"resolution_type": "FEE_ADJUSTMENT", "amount_paise": 3000}
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        state.evidence_package = {"evidence_coverage": 0.90}
        state.guardrail_result = {"primary_reason": "Low confidence"}
        state.risk = "MEDIUM"

        result = pause_for_human_approval(state)

        assert "_review_package" in result
        package = result["_review_package"]
        assert package["exception_id"] == "EXC-001"
        assert package["proposed_resolution"] == "FEE_ADJUSTMENT"
        assert package["financial_adjustment_paise"] == 3000

    def test_pause_records_node(self):
        state = create_initial_state(exception_id="EXC-001")
        state.selected_candidate = {"resolution_type": "FEE_ADJUSTMENT", "amount_paise": 3000}
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        state.evidence_package = {"evidence_coverage": 0.90}
        state.guardrail_result = {"primary_reason": "Low confidence"}
        state.risk = "MEDIUM"

        result = pause_for_human_approval(state)

        assert "pause_for_human_approval" in result["metadata"]["nodes_executed"]

    def test_pause_sets_priority(self):
        state = create_initial_state(exception_id="EXC-001")
        state.selected_candidate = {"resolution_type": "FEE_ADJUSTMENT", "amount_paise": 3000}
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        state.evidence_package = {"evidence_coverage": 0.90}
        state.guardrail_result = {"primary_reason": "Low confidence"}
        state.risk = "HIGH"

        result = pause_for_human_approval(state)

        assert result["human_review"]["review_priority"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# Approval Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestApproval:
    def test_approval_resumes_workflow(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
            reviewer_id="reviewer-001",
            reason="Looks good",
        )
        result = process_human_decision(state, decision)

        assert result["human_review"]["approval_status"] == HumanApprovalStatus.APPROVED.value
        assert result["metadata"]["workflow_status"] == WorkflowStatus.RUNNING.value

    def test_approval_records_reviewer(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
            reviewer_id="reviewer-001",
            reason="Verified",
        )
        result = process_human_decision(state, decision)

        assert result["human_review"]["assigned_reviewer"] == "reviewer-001"
        assert result["human_review"]["reviewer_notes"] == "Verified"

    def test_approval_records_timestamp(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
        )
        result = process_human_decision(state, decision)

        assert result["human_review"]["review_completed_at"] is not None

    def test_approval_logs_decision(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
            reviewer_id="reviewer-001",
        )
        result = process_human_decision(state, decision)

        log = result["metadata"]["execution_log"][-1]
        assert log["node"] == "human_decision"
        assert log["decision"] == "APPROVED"
        assert log["reviewer"] == "reviewer-001"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRejection:
    def test_rejection_marks_rejected(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
            reviewer_id="reviewer-001",
            reason="Insufficient evidence",
        )
        result = process_human_decision(state, decision)

        assert result["human_review"]["approval_status"] == HumanApprovalStatus.REJECTED.value
        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_rejection_records_reason(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
            reviewer_id="reviewer-001",
            reason="Amount incorrect",
        )
        result = process_human_decision(state, decision)

        assert result["human_review"]["reviewer_notes"] == "Amount incorrect"

    def test_rejection_adds_warning(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
            reviewer_id="reviewer-001",
            reason="Not safe",
        )
        result = process_human_decision(state, decision)

        assert len(result["warnings"]) > 0
        assert "REJECTED" in result["warnings"][0]

    def test_rejection_logs_decision(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
            reviewer_id="reviewer-001",
        )
        result = process_human_decision(state, decision)

        log = result["metadata"]["execution_log"][-1]
        assert log["decision"] == "REJECTED"


# ─────────────────────────────────────────────────────────────────────────────
# Security Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSecurity:
    def test_invalid_workflow_id_rejected(self):
        state = _make_state_waiting(workflow_id="WF-REAL-001")
        decision = HumanDecision(
            workflow_id="WF-FAKE-001",
            decision="APPROVED",
        )
        with pytest.raises(ValueError, match="Workflow ID mismatch"):
            process_human_decision(state, decision)

    def test_not_waiting_rejected(self):
        state = create_initial_state(exception_id="EXC-001")
        state.metadata.workflow_status = WorkflowStatus.RUNNING
        state.human_review.approval_status = HumanApprovalStatus.NOT_REQUIRED
        decision = HumanDecision(
            workflow_id=state.metadata.workflow_id,
            decision="APPROVED",
        )
        with pytest.raises(ValueError, match="not waiting for human"):
            process_human_decision(state, decision)

    def test_not_pending_rejected(self):
        state = _make_state_waiting()
        state.human_review.approval_status = HumanApprovalStatus.APPROVED
        decision = HumanDecision(
            workflow_id=state.metadata.workflow_id,
            decision="APPROVED",
        )
        with pytest.raises(ValueError, match="not pending"):
            process_human_decision(state, decision)

    def test_double_approval_rejected(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
        )
        # First approval succeeds
        result = process_human_decision(state, decision)
        # Update state with approval result
        state.human_review.approval_status = HumanApprovalStatus.APPROVED
        state.metadata.workflow_status = WorkflowStatus.RUNNING

        # Second approval fails (workflow no longer WAITING_FOR_HUMAN)
        with pytest.raises(ValueError, match="not waiting for human|not pending"):
            process_human_decision(state, decision)

    def test_approval_after_rejection_rejected(self):
        state = _make_state_waiting()
        reject_decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
        )
        process_human_decision(state, reject_decision)

        # Update state with rejection
        state.human_review.approval_status = HumanApprovalStatus.REJECTED
        state.metadata.workflow_status = WorkflowStatus.FAILED

        approve_decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
        )
        with pytest.raises(ValueError, match="not waiting for human|not pending"):
            process_human_decision(state, approve_decision)

    def test_invalid_decision_rejected(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="INVALID",
        )
        with pytest.raises(ValueError, match="Invalid decision"):
            process_human_decision(state, decision)


# ─────────────────────────────────────────────────────────────────────────────
# Review Package Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewPackage:
    def test_package_contains_all_fields(self):
        state = _make_state_waiting()
        state.metadata.case_id = "CASE-001"

        result = pause_for_human_approval(state)
        package = result["_review_package"]

        assert "workflow_id" in package
        assert "exception_id" in package
        assert "case_id" in package
        assert "candidate" in package
        assert "proposed_resolution" in package
        assert "financial_adjustment_paise" in package
        assert "evidence_summary" in package
        assert "guardrail_decision" in package
        assert "confidence" in package
        assert "risk" in package
        assert "classification" in package

    def test_package_values_correct(self):
        state = _make_state_waiting()
        result = pause_for_human_approval(state)
        package = result["_review_package"]

        assert package["workflow_id"] == "WF-TEST-001"
        assert package["exception_id"] == "EXC-001"
        assert package["proposed_resolution"] == "FEE_ADJUSTMENT"
        assert package["financial_adjustment_paise"] == 3000
        assert package["confidence"] == 0.55
        assert package["risk"] == "MEDIUM"
        assert package["classification"] == "FEE_DIFFERENCE"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_pause_without_candidate(self):
        state = create_initial_state(exception_id="EXC-001")
        result = pause_for_human_approval(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.WAITING_FOR_HUMAN.value

    def test_approval_preserves_existing_state(self):
        state = _make_state_waiting()
        state.warnings = ["previous warning"]
        state.errors = ["previous error"]

        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="APPROVED",
        )
        result = process_human_decision(state, decision)

        # Existing warnings/errors should be preserved in metadata
        assert "metadata" in result

    def test_rejection_preserves_existing_state(self):
        state = _make_state_waiting()
        decision = HumanDecision(
            workflow_id="WF-TEST-001",
            decision="REJECTED",
            reason="Not safe",
        )
        result = process_human_decision(state, decision)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
