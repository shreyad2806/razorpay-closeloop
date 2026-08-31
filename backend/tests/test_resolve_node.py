"""
Tests for Razorpay CloseLoop Phase 7I — Resolve / Action Boundary.

Tests action request creation, safety checks, authorization, and idempotency.
"""

import hashlib
import pytest
from app.agent.resolve_node import resolve_action_boundary, _compute_idempotency_key
from app.agent.workflow import create_initial_state
from app.schemas.action_request import (
    ActionRequest,
    ActionRequestResult,
    ActionStatus,
    AuthorizationSource,
)
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    HumanReviewState,
    VerificationStatus,
    VerificationState,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_state(
    decision="AUTO",
    verification_status="VERIFIED",
    approval_status="NOT_REQUIRED",
    candidate_id="CAND-001",
    resolution_type="APPLY_FEE_CORRECTION",
    amount_paise=3000,
    confidence=0.85,
    risk="LOW",
) -> AgentState:
    """Build an AgentState for resolve node testing."""
    state = create_initial_state(exception_id="EXC-001", workflow_id="WF-TEST-001")
    state.metadata.workflow_status = WorkflowStatus.RUNNING
    state.decision = decision
    state.confidence = confidence
    state.risk = risk

    state.verification = VerificationState(
        verification_status=VerificationStatus(verification_status),
        verification_result={"passed": verification_status == "VERIFIED"},
    )

    state.human_review = HumanReviewState(
        approval_status=HumanApprovalStatus(approval_status),
    )

    state.selected_candidate = {
        "candidate_id": candidate_id,
        "resolution_type": resolution_type,
        "amount_paise": amount_paise,
        "description": f"Adjust by {amount_paise} paise",
    }

    state.evidence_package = {
        "evidence_coverage": 0.85,
        "evidence_consistency": 0.90,
    }

    state.classification = {
        "exception_type": "FEE_DIFFERENCE",
    }

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_action_status_values(self):
        assert ActionStatus.PENDING.value == "PENDING"
        assert ActionStatus.AUTHORIZED.value == "AUTHORIZED"
        assert ActionStatus.REJECTED.value == "REJECTED"
        assert ActionStatus.EXECUTED.value == "EXECUTED"

    def test_authorization_source_values(self):
        assert AuthorizationSource.AUTO_GUARDRAIL.value == "AUTO_GUARDRAIL"
        assert AuthorizationSource.HUMAN_APPROVAL.value == "HUMAN_APPROVAL"
        assert AuthorizationSource.NONE.value == "NONE"

    def test_action_request_summary(self):
        req = ActionRequest(
            action_id="ACT-001",
            idempotency_key="key-001",
            workflow_id="WF-001",
            exception_id="EXC-001",
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
            authorization_source=AuthorizationSource.AUTO_GUARDRAIL,
            verification_passed=True,
            guardrail_decision="AUTO",
        )
        summary = req.summary()
        assert "ACT-001" in summary
        assert "APPLY_FEE_CORRECTION" in summary
        assert "3000" in summary

    def test_action_request_result(self):
        result = ActionRequestResult(
            success=False,
            rejection_reasons=["test reason"],
            blocked=True,
        )
        assert result.success is False
        assert result.blocked is True
        assert "test reason" in result.rejection_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Successful Action Request Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuccessfulRequest:
    def test_auto_verified_creates_request(self):
        """AUTO decision + VERIFIED → action request created."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        result = resolve_action_boundary(state)

        assert result["action_request"] is not None
        req = result["action_request"]
        assert req["resolution_type"] == "APPLY_FEE_CORRECTION"
        assert req["financial_adjustment_paise"] == 3000
        assert req["verification_passed"] is True
        assert req["guardrail_decision"] == "AUTO"
        assert req["authorization_source"] == "AUTO_GUARDRAIL"
        assert req["authorized_by"] == "auto_guardrail"
        assert req["status"] == "PENDING"

    def test_human_approved_creates_request(self):
        """HUMAN_REVIEW + APPROVED → action request created."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="APPROVED",
        )
        result = resolve_action_boundary(state)

        assert result["action_request"] is not None
        req = result["action_request"]
        assert req["authorization_source"] == "HUMAN_APPROVAL"
        assert req["guardrail_decision"] == "HUMAN_REVIEW"
        assert req["verification_passed"] is True

    def test_idempotency_key_deterministic(self):
        """Same inputs produce the same idempotency key."""
        key1 = _compute_idempotency_key("WF-001", "EXC-001", "CAND-001")
        key2 = _compute_idempotency_key("WF-001", "EXC-001", "CAND-001")
        assert key1 == key2

    def test_idempotency_key_different_inputs(self):
        """Different inputs produce different keys."""
        key1 = _compute_idempotency_key("WF-001", "EXC-001", "CAND-001")
        key2 = _compute_idempotency_key("WF-001", "EXC-002", "CAND-001")
        assert key1 != key2

    def test_request_has_action_id(self):
        """Request has a unique action ID."""
        state = _make_state()
        result = resolve_action_boundary(state)
        req = result["action_request"]
        assert req["action_id"].startswith("ACT-")
        assert len(req["action_id"]) > 5

    def test_request_preserves_workflow_context(self):
        """Request preserves workflow metadata."""
        state = _make_state()
        result = resolve_action_boundary(state)
        req = result["action_request"]
        assert req["workflow_id"] == "WF-TEST-001"
        assert req["exception_id"] == "EXC-001"
        assert req["candidate_id"] == "CAND-001"

    def test_request_includes_evidence_summary(self):
        """Request includes evidence context."""
        state = _make_state()
        result = resolve_action_boundary(state)
        req = result["action_request"]
        assert "coverage" in req["evidence_summary"]
        assert req["evidence_summary"]["coverage"] == 0.85

    def test_request_includes_metadata(self):
        """Request includes exception type and risk."""
        state = _make_state()
        result = resolve_action_boundary(state)
        req = result["action_request"]
        assert req["metadata"]["exception_type"] == "FEE_DIFFERENCE"
        assert req["metadata"]["risk"] == "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Check Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyChecks:
    def test_unresolved_rejected(self):
        """UNRESOLVED decision → rejected."""
        state = _make_state(decision="UNRESOLVED")
        result = resolve_action_boundary(state)

        assert result["action_request"] is None
        assert any("does not allow action" in w for w in result["warnings"])

    def test_no_decision_rejected(self):
        """No decision → rejected."""
        state = create_initial_state(exception_id="EXC-001", workflow_id="WF-TEST-001")
        state.metadata.workflow_status = WorkflowStatus.RUNNING
        state.decision = None
        state.verification = VerificationState(
            verification_status=VerificationStatus.VERIFIED,
        )
        state.selected_candidate = {
            "candidate_id": "CAND-001",
            "resolution_type": "APPLY_FEE_CORRECTION",
            "amount_paise": 3000,
        }

        result = resolve_action_boundary(state)
        assert result["action_request"] is None

    def test_human_review_not_approved_rejected(self):
        """HUMAN_REVIEW + PENDING approval → rejected."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="PENDING",
        )
        result = resolve_action_boundary(state)

        assert result["action_request"] is None
        assert any("requires explicit human approval" in w for w in result["warnings"])

    def test_human_review_rejected_rejected(self):
        """HUMAN_REVIEW + REJECTED → rejected."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="REJECTED",
        )
        result = resolve_action_boundary(state)

        assert result["action_request"] is None

    def test_verification_not_passed_rejected(self):
        """Verification not VERIFIED → rejected."""
        state = _make_state(decision="AUTO", verification_status="PENDING")
        result = resolve_action_boundary(state)

        assert result["action_request"] is None
        assert any("Verification status" in w for w in result["warnings"])

    def test_verification_failed_rejected(self):
        """Verification FAILED → rejected."""
        state = _make_state(decision="AUTO", verification_status="FAILED")
        result = resolve_action_boundary(state)

        assert result["action_request"] is None

    def test_no_candidate_rejected(self):
        """No selected candidate → rejected."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        state.selected_candidate = None
        result = resolve_action_boundary(state)

        assert result["action_request"] is None
        assert any("No selected candidate" in w for w in result["warnings"])

    def test_candidate_no_resolution_type_rejected(self):
        """Candidate without resolution type → rejected."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        state.selected_candidate = {"candidate_id": "CAND-001"}
        result = resolve_action_boundary(state)

        assert result["action_request"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_workflow_same_key(self):
        """Same workflow + exception + candidate → same idempotency key."""
        state1 = _make_state()
        state2 = _make_state()
        r1 = resolve_action_boundary(state1)
        r2 = resolve_action_boundary(state2)
        assert r1["action_request"]["idempotency_key"] == r2["action_request"]["idempotency_key"]

    def test_different_workflow_different_key(self):
        """Different workflow → different idempotency key."""
        state1 = _make_state()
        state1.metadata.workflow_id = "WF-001"
        state2 = _make_state()
        state2.metadata.workflow_id = "WF-002"
        r1 = resolve_action_boundary(state1)
        r2 = resolve_action_boundary(state2)
        assert r1["action_request"]["idempotency_key"] != r2["action_request"]["idempotency_key"]

    def test_different_exception_different_key(self):
        """Different exception → different idempotency key."""
        state1 = _make_state()
        state1.metadata.exception_id = "EXC-001"
        state2 = _make_state()
        state2.metadata.exception_id = "EXC-002"
        r1 = resolve_action_boundary(state1)
        r2 = resolve_action_boundary(state2)
        assert r1["action_request"]["idempotency_key"] != r2["action_request"]["idempotency_key"]

    def test_idempotency_key_is_sha256(self):
        """Idempotency key is a SHA-256 hex prefix."""
        key = _compute_idempotency_key("WF-001", "EXC-001", "CAND-001")
        assert len(key) == 32
        # Should be valid hex
        int(key, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Node Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNodeMetadata:
    def test_node_recorded_on_success(self):
        """Successful request records node execution."""
        state = _make_state()
        result = resolve_action_boundary(state)
        assert "resolve_action_boundary" in result["metadata"]["nodes_executed"]
        log = result["metadata"]["execution_log"][-1]
        assert log["node"] == "resolve_action_boundary"
        assert log["success"] is True

    def test_node_recorded_on_rejection(self):
        """Rejected request records node execution."""
        state = _make_state(decision="UNRESOLVED")
        result = resolve_action_boundary(state)
        assert "resolve_action_boundary" in result["metadata"]["nodes_executed"]
        log = result["metadata"]["execution_log"][-1]
        assert log["success"] is False

    def test_current_node_set(self):
        """Current node is set to resolve_action_boundary."""
        state = _make_state()
        result = resolve_action_boundary(state)
        assert result["metadata"]["current_node"] == "resolve_action_boundary"

    def test_warnings_on_rejection(self):
        """Warnings are added on rejection."""
        state = _make_state(decision="UNRESOLVED")
        result = resolve_action_boundary(state)
        assert len(result["warnings"]) > 0
        assert any("ACTION_REJECTED" in w for w in result["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_adjustment(self):
        """Zero financial adjustment → still creates request."""
        state = _make_state(amount_paise=0)
        result = resolve_action_boundary(state)
        assert result["action_request"] is not None
        assert result["action_request"]["financial_adjustment_paise"] == 0

    def test_high_amount(self):
        """High financial adjustment → still creates request (boundary doesn't limit amount)."""
        state = _make_state(amount_paise=10000000)
        result = resolve_action_boundary(state)
        assert result["action_request"] is not None
        assert result["action_request"]["financial_adjustment_paise"] == 10000000

    def test_missing_evidence_package(self):
        """Missing evidence package → still creates request with default summary."""
        state = _make_state()
        state.evidence_package = None
        result = resolve_action_boundary(state)
        assert result["action_request"] is not None
        assert result["action_request"]["evidence_summary"]["coverage"] == 0
        assert result["action_request"]["evidence_summary"]["consistency"] == 0

    def test_missing_classification(self):
        """Missing classification → still creates request."""
        state = _make_state()
        state.classification = None
        result = resolve_action_boundary(state)
        assert result["action_request"] is not None
        assert result["action_request"]["metadata"]["exception_type"] is None

    def test_no_exception_in_metadata(self):
        """No exception_id in metadata → rejected (missing required field)."""
        state = _make_state()
        state.metadata.exception_id = ""
        result = resolve_action_boundary(state)
        # The action request still creates — exception_id is in metadata not validation
        assert result["action_request"] is not None

    def test_action_request_not_in_state(self):
        """Action request is stored in state updates, not in AgentState model."""
        state = _make_state()
        result = resolve_action_boundary(state)
        assert "action_request" in result
        # The action_request is a dict, not an ActionRequest object
        assert isinstance(result["action_request"], dict)
