"""
Tests for AgentState (Phase 7A).

Tests:
- state creation with required fields
- state creation with all fields
- serialization (to_dict / from_dict)
- state updates (partial updates)
- workflow metadata
- human review state
- verification state
- reward state
- error state
- state summary
"""

import pytest
from datetime import datetime

from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    HumanReviewState,
    NodeStatus,
    RewardState,
    RewardStatus,
    VerificationState,
    VerificationStatus,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_metadata(
    workflow_id="WF-001",
    exception_id="EXC-001",
    case_id="CASE-001",
    status=WorkflowStatus.PENDING,
):
    return WorkflowMetadata(
        workflow_id=workflow_id,
        exception_id=exception_id,
        case_id=case_id,
        workflow_status=status,
    )


def _make_state(**kwargs):
    metadata = kwargs.pop("metadata", None) or _make_metadata()
    return AgentState(metadata=metadata, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Enum Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowStatus:
    def test_values(self):
        assert WorkflowStatus.PENDING.value == "PENDING"
        assert WorkflowStatus.RUNNING.value == "RUNNING"
        assert WorkflowStatus.WAITING_FOR_HUMAN.value == "WAITING_FOR_HUMAN"
        assert WorkflowStatus.COMPLETED.value == "COMPLETED"
        assert WorkflowStatus.FAILED.value == "FAILED"
        assert WorkflowStatus.CANCELLED.value == "CANCELLED"


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.PENDING.value == "PENDING"
        assert NodeStatus.RUNNING.value == "RUNNING"
        assert NodeStatus.COMPLETED.value == "COMPLETED"


class TestHumanApprovalStatus:
    def test_values(self):
        assert HumanApprovalStatus.NOT_REQUIRED.value == "NOT_REQUIRED"
        assert HumanApprovalStatus.PENDING.value == "PENDING"
        assert HumanApprovalStatus.APPROVED.value == "APPROVED"
        assert HumanApprovalStatus.REJECTED.value == "REJECTED"
        assert HumanApprovalStatus.RESUMED.value == "RESUMED"


class TestVerificationStatus:
    def test_values(self):
        assert VerificationStatus.NOT_REQUIRED.value == "NOT_REQUIRED"
        assert VerificationStatus.PENDING.value == "PENDING"
        assert VerificationStatus.VERIFIED.value == "VERIFIED"
        assert VerificationStatus.FAILED.value == "FAILED"


class TestRewardStatus:
    def test_values(self):
        assert RewardStatus.NOT_REQUIRED.value == "NOT_REQUIRED"
        assert RewardStatus.PENDING.value == "PENDING"
        assert RewardStatus.CALCULATED.value == "CALCULATED"
        assert RewardStatus.FAILED.value == "FAILED"


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowMetadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowMetadata:
    def test_create_minimal(self):
        meta = WorkflowMetadata(
            workflow_id="WF-001",
            exception_id="EXC-001",
        )
        assert meta.workflow_id == "WF-001"
        assert meta.exception_id == "EXC-001"
        assert meta.workflow_status == WorkflowStatus.PENDING
        assert meta.retry_count == 0
        assert meta.max_retries == 3
        assert len(meta.errors) == 0
        assert len(meta.nodes_executed) == 0

    def test_create_full(self):
        meta = WorkflowMetadata(
            workflow_id="WF-001",
            exception_id="EXC-001",
            case_id="CASE-001",
            current_node="reconcile",
            workflow_status=WorkflowStatus.RUNNING,
            retry_count=1,
            max_retries=5,
            errors=["error1"],
            nodes_executed=["node1", "node2"],
        )
        assert meta.case_id == "CASE-001"
        assert meta.current_node == "reconcile"
        assert meta.workflow_status == WorkflowStatus.RUNNING
        assert meta.retry_count == 1
        assert meta.max_retries == 5
        assert len(meta.errors) == 1
        assert len(meta.nodes_executed) == 2

    def test_timestamps(self):
        meta = WorkflowMetadata(
            workflow_id="WF-001",
            exception_id="EXC-001",
        )
        assert meta.created_at is not None
        assert meta.started_at is None
        assert meta.completed_at is None


# ─────────────────────────────────────────────────────────────────────────────
# HumanReviewState Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanReviewState:
    def test_default(self):
        state = HumanReviewState()
        assert state.approval_status == HumanApprovalStatus.NOT_REQUIRED
        assert state.assigned_reviewer is None
        assert state.reviewer_notes is None

    def test_pending_review(self):
        state = HumanReviewState(
            approval_status=HumanApprovalStatus.PENDING,
            assigned_reviewer="reviewer-001",
            review_requested_at=datetime.utcnow(),
            review_reason="High-value adjustment",
            review_priority="HIGH",
        )
        assert state.approval_status == HumanApprovalStatus.PENDING
        assert state.assigned_reviewer == "reviewer-001"
        assert state.review_reason == "High-value adjustment"

    def test_approved(self):
        state = HumanReviewState(
            approval_status=HumanApprovalStatus.APPROVED,
            review_completed_at=datetime.utcnow(),
            reviewer_notes="Looks good",
        )
        assert state.approval_status == HumanApprovalStatus.APPROVED
        assert state.reviewer_notes == "Looks good"

    def test_rejected(self):
        state = HumanReviewState(
            approval_status=HumanApprovalStatus.REJECTED,
            review_completed_at=datetime.utcnow(),
            reviewer_notes="Evidence insufficient",
        )
        assert state.approval_status == HumanApprovalStatus.REJECTED


# ─────────────────────────────────────────────────────────────────────────────
# VerificationState Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationState:
    def test_default(self):
        state = VerificationState()
        assert state.verification_status == VerificationStatus.NOT_REQUIRED
        assert state.verification_result is None
        assert len(state.verification_errors) == 0

    def test_verified(self):
        state = VerificationState(
            verification_status=VerificationStatus.VERIFIED,
            verification_result={"match": True, "amount_correct": True},
            verified_at=datetime.utcnow(),
            verified_by="system",
        )
        assert state.verification_status == VerificationStatus.VERIFIED
        assert state.verification_result["match"] is True

    def test_failed(self):
        state = VerificationState(
            verification_status=VerificationStatus.FAILED,
            verification_errors=["Amount mismatch", "Missing record"],
        )
        assert state.verification_status == VerificationStatus.FAILED
        assert len(state.verification_errors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# RewardState Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardState:
    def test_default(self):
        state = RewardState()
        assert state.reward_status == RewardStatus.NOT_REQUIRED
        assert state.reward is None
        assert state.reward_reason is None

    def test_calculated(self):
        state = RewardState(
            reward_status=RewardStatus.CALCULATED,
            reward=0.95,
            reward_reason="Resolution matched ground truth",
            reward_calculated_at=datetime.utcnow(),
        )
        assert state.reward_status == RewardStatus.CALCULATED
        assert state.reward == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# AgentState Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentStateCreation:
    def test_minimal_state(self):
        state = _make_state()
        assert state.metadata.workflow_id == "WF-001"
        assert state.metadata.exception_id == "EXC-001"
        assert state.reconciliation_result is None
        assert state.evidence_package is None
        assert state.classification is None
        assert state.decision is None
        assert state.confidence is None
        assert len(state.errors) == 0
        assert len(state.warnings) == 0

    def test_state_with_all_phases(self):
        state = _make_state(
            reconciliation_result={"status": "MATCHED"},
            evidence_package={"records": 5},
            evidence_graph={"nodes": 10},
            explanation_result={"status": "FULLY_EXPLAINED"},
            evidence_quality={"coverage": 0.95},
            classification={"type": "FEE_DIFFERENCE"},
            similar_cases={"count": 3},
            intelligence={"status": "SUPPORTED"},
            candidates={"count": 2},
            selected_candidate={"id": "CAND-001"},
            candidate_scores={"final": 0.85},
            confidence=0.85,
            risk="LOW",
            guardrail_result={"decision": "AUTO"},
            decision="AUTO",
        )
        assert state.reconciliation_result["status"] == "MATCHED"
        assert state.evidence_package["records"] == 5
        assert state.classification["type"] == "FEE_DIFFERENCE"
        assert state.confidence == 0.85
        assert state.decision == "AUTO"

    def test_state_with_human_review(self):
        state = _make_state(
            human_review=HumanReviewState(
                approval_status=HumanApprovalStatus.PENDING,
                assigned_reviewer="reviewer-001",
            ),
        )
        assert state.human_review.approval_status == HumanApprovalStatus.PENDING

    def test_state_with_verification(self):
        state = _make_state(
            verification=VerificationState(
                verification_status=VerificationStatus.VERIFIED,
            ),
        )
        assert state.verification.verification_status == VerificationStatus.VERIFIED

    def test_state_with_reward(self):
        state = _make_state(
            reward=RewardState(
                reward_status=RewardStatus.CALCULATED,
                reward=0.90,
            ),
        )
        assert state.reward.reward == 0.90


class TestAgentStateUpdates:
    def test_partial_update_metadata(self):
        state = _make_state()
        state.metadata.current_node = "reconcile"
        state.metadata.workflow_status = WorkflowStatus.RUNNING

        assert state.metadata.current_node == "reconcile"
        assert state.metadata.workflow_status == WorkflowStatus.RUNNING
        # Other fields unchanged
        assert state.metadata.retry_count == 0

    def test_partial_update_phase2(self):
        state = _make_state()
        state.reconciliation_result = {"status": "EXCEPTION", "difference": 3000}

        assert state.reconciliation_result["status"] == "EXCEPTION"
        assert state.evidence_package is None  # unchanged

    def test_partial_update_phase6(self):
        state = _make_state()
        state.confidence = 0.85
        state.risk = "LOW"
        state.decision = "AUTO"

        assert state.confidence == 0.85
        assert state.risk == "LOW"
        assert state.decision == "AUTO"

    def test_add_error(self):
        state = _make_state()
        state.errors.append("Phase 3 evidence retrieval failed")

        assert len(state.errors) == 1
        assert "Phase 3" in state.errors[0]

    def test_add_warning(self):
        state = _make_state()
        state.warnings.append("ML model confidence low")

        assert len(state.warnings) == 1

    def test_update_metadata_nodes(self):
        state = _make_state()
        state.metadata.nodes_executed.append("reconcile")
        state.metadata.nodes_executed.append("evidence")

        assert len(state.metadata.nodes_executed) == 2

    def test_update_metadata_errors(self):
        state = _make_state()
        state.metadata.errors.append("Node timeout")

        assert len(state.metadata.errors) == 1


class TestAgentStateSerialization:
    def test_to_dict(self):
        state = _make_state(
            confidence=0.85,
            decision="AUTO",
        )
        d = state.model_dump()

        assert d["metadata"]["workflow_id"] == "WF-001"
        assert d["confidence"] == 0.85
        assert d["decision"] == "AUTO"
        assert "human_review" in d
        assert "verification" in d
        assert "reward" in d

    def test_from_dict(self):
        state = _make_state(confidence=0.85, decision="AUTO")
        d = state.model_dump()
        restored = AgentState(**d)

        assert restored.metadata.workflow_id == "WF-001"
        assert restored.confidence == 0.85
        assert restored.decision == "AUTO"

    def test_roundtrip(self):
        state = _make_state(
            reconciliation_result={"status": "EXCEPTION"},
            confidence=0.85,
            decision="AUTO",
            human_review=HumanReviewState(
                approval_status=HumanApprovalStatus.PENDING,
            ),
        )
        d = state.model_dump()
        restored = AgentState(**d)

        assert restored.reconciliation_result["status"] == "EXCEPTION"
        assert restored.confidence == 0.85
        assert restored.human_review.approval_status == HumanApprovalStatus.PENDING


class TestAgentStateSummary:
    def test_summary(self):
        state = _make_state(
            confidence=0.85,
            decision="AUTO",
        )
        s = state.summary()
        assert "WF-001" in s
        assert "EXC-001" in s
        assert "AUTO" in s

    def test_summary_with_errors(self):
        state = _make_state()
        state.errors.append("error1")
        s = state.summary()
        assert "Errors: 1" in s


class TestAgentStatePhaseIsolation:
    """Verify each phase stores data independently."""

    def test_phase_data_isolation(self):
        state = _make_state(
            reconciliation_result={"phase": 2},
            evidence_package={"phase": 3},
            classification={"phase": 4},
            candidates={"phase": 5},
            guardrail_result={"phase": 6},
        )

        assert state.reconciliation_result["phase"] == 2
        assert state.evidence_package["phase"] == 3
        assert state.classification["phase"] == 4
        assert state.candidates["phase"] == 5
        assert state.guardrail_result["phase"] == 6

    def test_phase_update_isolation(self):
        state = _make_state(
            reconciliation_result={"status": "MATCHED"},
            evidence_package={"records": 5},
        )
        state.reconciliation_result["status"] = "EXCEPTION"

        assert state.reconciliation_result["status"] == "EXCEPTION"
        assert state.evidence_package["records"] == 5  # unchanged


class TestAgentStateNoBusinessLogic:
    """Verify state contains data, not business logic."""

    def test_no_methods_on_data_fields(self):
        state = _make_state()
        # State should be pure data — no business logic methods
        # on the data fields themselves
        assert isinstance(state.reconciliation_result, (dict, type(None)))
        assert isinstance(state.evidence_package, (dict, type(None)))
        assert isinstance(state.classification, (dict, type(None)))
