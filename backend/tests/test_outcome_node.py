"""
Tests for Razorpay CloseLoop Phase 7J — Outcome Recording + Reward Generation.

Tests outcome determination, reward calculation, historical storage, and ground truth isolation.
"""

import pytest
from app.agent.outcome_node import record_outcome, _determine_outcome, _calculate_reward
from app.agent.workflow import create_initial_state
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    HumanReviewState,
    VerificationStatus,
    VerificationState,
    WorkflowStatus,
)
from app.schemas.outcome import (
    HistoricalLearningRecord,
    RewardSignal,
    RewardType,
    WorkflowOutcome,
    WorkflowOutcomeRecord,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_state(
    decision="AUTO",
    verification_status="VERIFIED",
    approval_status="NOT_REQUIRED",
    workflow_status="RUNNING",
    current_node="resolve_action_boundary",
    confidence=0.85,
    risk="LOW",
    exception_type="FEE_DIFFERENCE",
    amount_paise=3000,
    errors=None,
    warnings=None,
) -> AgentState:
    state = create_initial_state(exception_id="EXC-001", workflow_id="WF-TEST-001")
    state.metadata.workflow_status = WorkflowStatus(workflow_status)
    state.metadata.current_node = current_node
    state.decision = decision
    state.confidence = confidence
    state.risk = risk
    state.errors = errors or []
    state.warnings = warnings or []

    state.verification = VerificationState(
        verification_status=VerificationStatus(verification_status),
    )
    state.human_review = HumanReviewState(
        approval_status=HumanApprovalStatus(approval_status),
    )
    state.classification = {"exception_type": exception_type}
    state.selected_candidate = {
        "candidate_id": "CAND-001",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "amount_paise": amount_paise,
    }
    state.evidence_package = {
        "evidence_coverage": 0.85,
        "evidence_consistency": 0.90,
        "supporting_evidence_ids": ["EV-001", "EV-002"],
    }
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_workflow_outcome_values(self):
        assert WorkflowOutcome.RESOLVED_AUTO.value == "RESOLVED_AUTO"
        assert WorkflowOutcome.RESOLVED_HUMAN.value == "RESOLVED_HUMAN"
        assert WorkflowOutcome.REJECTED_BY_HUMAN.value == "REJECTED_BY_HUMAN"
        assert WorkflowOutcome.VERIFICATION_FAILED.value == "VERIFICATION_FAILED"
        assert WorkflowOutcome.UNRESOLVED.value == "UNRESOLVED"
        assert WorkflowOutcome.ESCALATED.value == "ESCALATED"
        assert WorkflowOutcome.SYSTEM_ERROR.value == "SYSTEM_ERROR"

    def test_reward_type_values(self):
        assert RewardType.CORRECT_RESOLUTION.value == "CORRECT_RESOLUTION"
        assert RewardType.INCORRECT_RESOLUTION.value == "INCORRECT_RESOLUTION"
        assert RewardType.PARTIAL_CREDIT.value == "PARTIAL_CREDIT"
        assert RewardType.NO_REWARD.value == "NO_REWARD"
        assert RewardType.PENALTY.value == "PENALTY"

    def test_outcome_record_summary(self):
        record = WorkflowOutcomeRecord(
            workflow_id="WF-001",
            exception_id="EXC-001",
            decision="AUTO",
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
            outcome=WorkflowOutcome.RESOLVED_AUTO,
        )
        s = record.summary()
        assert "RESOLVED_AUTO" in s
        assert "3000" in s

    def test_learning_record_retrieval_features(self):
        record = HistoricalLearningRecord(
            workflow_id="WF-001",
            exception_id="EXC-001",
            exception_type="FEE_DIFFERENCE",
            resolution_type="APPLY_FEE_CORRECTION",
            outcome=WorkflowOutcome.RESOLVED_AUTO,
            confidence=0.85,
            risk="LOW",
            evidence_coverage=0.85,
        )
        features = record.to_retrieval_features()
        assert features["exception_type"] == "FEE_DIFFERENCE"
        assert features["confidence"] == 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Determination Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOutcomeDetermination:
    def test_auto_verified_resolved(self):
        """AUTO + VERIFIED → RESOLVED_AUTO."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        assert _determine_outcome(state) == WorkflowOutcome.RESOLVED_AUTO

    def test_human_approved_resolved(self):
        """HUMAN_REVIEW + APPROVED + VERIFIED → RESOLVED_HUMAN."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="APPROVED",
        )
        assert _determine_outcome(state) == WorkflowOutcome.RESOLVED_HUMAN

    def test_human_rejected(self):
        """REJECTED → REJECTED_BY_HUMAN."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="REJECTED",
            workflow_status="FAILED",
        )
        assert _determine_outcome(state) == WorkflowOutcome.REJECTED_BY_HUMAN

    def test_verification_failed(self):
        """VERIFICATION_FAILED → VERIFICATION_FAILED."""
        state = _make_state(
            decision="AUTO",
            verification_status="FAILED",
            workflow_status="FAILED",
        )
        assert _determine_outcome(state) == WorkflowOutcome.VERIFICATION_FAILED

    def test_unresolved(self):
        """UNRESOLVED → UNRESOLVED."""
        state = _make_state(decision="UNRESOLVED")
        assert _determine_outcome(state) == WorkflowOutcome.UNRESOLVED

    def test_escalated(self):
        """ESCALATION warning with HUMAN_REVIEW decision → ESCALATED."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="PENDING",
            warnings=["ESCALATED: low confidence"],
        )
        assert _determine_outcome(state) == WorkflowOutcome.ESCALATED

    def test_system_error(self):
        """FAILED + errors + no specific match → SYSTEM_ERROR."""
        state = _make_state(
            decision="AUTO",
            verification_status="NOT_REQUIRED",
            workflow_status="FAILED",
            errors=["Database timeout"],
        )
        assert _determine_outcome(state) == WorkflowOutcome.SYSTEM_ERROR

    def test_human_review_pending_escalated(self):
        """HUMAN_REVIEW + PENDING → ESCALATED."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="PENDING",
        )
        assert _determine_outcome(state) == WorkflowOutcome.ESCALATED


# ─────────────────────────────────────────────────────────────────────────────
# Reward Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardCalculation:
    def test_auto_resolved_reward(self):
        """RESOLVED_AUTO gets positive reward."""
        state = _make_state()
        outcome = WorkflowOutcomeRecord(
            workflow_id="WF-001",
            exception_id="EXC-001",
            decision="AUTO",
            outcome=WorkflowOutcome.RESOLVED_AUTO,
            verification_passed=True,
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
        )
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value > 0
        assert reward.reward_type == RewardType.CORRECT_RESOLUTION
        assert reward.resolution_correct is True
        assert reward.verification_bonus > 0

    def test_human_resolved_reward(self):
        """RESOLVED_HUMAN gets moderate reward."""
        state = _make_state()
        outcome = WorkflowOutcomeRecord(
            workflow_id="WF-001",
            exception_id="EXC-001",
            decision="HUMAN_REVIEW",
            outcome=WorkflowOutcome.RESOLVED_HUMAN,
            verification_passed=True,
            human_approved=True,
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
        )
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value > 0
        assert reward.human_approval_bonus > 0

    def _make_outcome(self, **overrides):
        """Helper to build a WorkflowOutcomeRecord with defaults."""
        defaults = dict(
            workflow_id="WF-001",
            exception_id="EXC-001",
            decision="AUTO",
            outcome=WorkflowOutcome.UNRESOLVED,
        )
        defaults.update(overrides)
        return WorkflowOutcomeRecord(**defaults)

    def test_rejected_reward(self):
        """REJECTED_BY_HUMAN gets partial credit."""
        state = _make_state()
        outcome = self._make_outcome(outcome=WorkflowOutcome.REJECTED_BY_HUMAN)
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value >= 0
        assert reward.reward_type == RewardType.PARTIAL_CREDIT

    def test_verification_failed_penalty(self):
        """VERIFICATION_FAILED gets penalty."""
        state = _make_state()
        outcome = self._make_outcome(outcome=WorkflowOutcome.VERIFICATION_FAILED)
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value < 0
        assert reward.reward_type == RewardType.PENALTY

    def test_unresolved_no_reward(self):
        """UNRESOLVED gets no reward."""
        state = _make_state()
        outcome = self._make_outcome(outcome=WorkflowOutcome.UNRESOLVED)
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value == 0.0
        assert reward.reward_type == RewardType.NO_REWARD

    def test_system_error_penalty(self):
        """SYSTEM_ERROR gets penalty."""
        state = _make_state()
        outcome = self._make_outcome(outcome=WorkflowOutcome.SYSTEM_ERROR)
        reward = _calculate_reward(state, outcome)
        assert reward.reward_value < 0
        assert reward.reward_type == RewardType.PENALTY

    def test_reward_clamped(self):
        """Reward is clamped to [-1.0, 1.0]."""
        state = _make_state()
        outcome = self._make_outcome(
            outcome=WorkflowOutcome.RESOLVED_AUTO,
            verification_passed=True,
            human_approved=True,
            resolution_type="APPLY_FEE_CORRECTION",
            financial_adjustment_paise=3000,
        )
        reward = _calculate_reward(state, outcome)
        assert -1.0 <= reward.reward_value <= 1.0

    def test_reward_workflow_id_preserved(self):
        """Reward preserves workflow and exception IDs."""
        state = _make_state()
        outcome = self._make_outcome()
        reward = _calculate_reward(state, outcome)
        assert reward.workflow_id == "WF-001"
        assert reward.exception_id == "EXC-001"

    def test_reward_has_reason(self):
        """Reward always has a reason."""
        state = _make_state()
        outcome = self._make_outcome()
        reward = _calculate_reward(state, outcome)
        assert reward.reward_reason is not None
        assert len(reward.reward_reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Full Outcome Node Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOutcomeNode:
    def test_auto_resolved_outcome(self):
        """AUTO + VERIFIED → complete outcome record."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        result = record_outcome(state)

        assert result["outcome"] is not None
        assert result["outcome"]["outcome"] == "RESOLVED_AUTO"
        assert result["reward"] is not None
        assert result["learning_record"] is not None
        assert result["metadata"]["current_node"] == "record_outcome"

    def test_human_approved_outcome(self):
        """HUMAN_REVIEW + APPROVED → RESOLVED_HUMAN."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="APPROVED",
        )
        result = record_outcome(state)
        assert result["outcome"]["outcome"] == "RESOLVED_HUMAN"

    def test_human_rejected_outcome(self):
        """REJECTED → REJECTED_BY_HUMAN."""
        state = _make_state(
            decision="HUMAN_REVIEW",
            verification_status="VERIFIED",
            approval_status="REJECTED",
            workflow_status="FAILED",
        )
        result = record_outcome(state)
        assert result["outcome"]["outcome"] == "REJECTED_BY_HUMAN"

    def test_unresolved_outcome(self):
        """UNRESOLVED → UNRESOLVED."""
        state = _make_state(decision="UNRESOLVED")
        result = record_outcome(state)
        assert result["outcome"]["outcome"] == "UNRESOLVED"

    def test_reward_state_updated(self):
        """Reward state is updated with calculated reward."""
        state = _make_state()
        result = record_outcome(state)
        assert result["reward_state"]["reward_status"] == "CALCULATED"
        assert result["reward_state"]["reward"] is not None

    def test_learning_record_has_evidence(self):
        """Learning record captures evidence context."""
        state = _make_state()
        result = record_outcome(state)
        lr = result["learning_record"]
        assert lr["evidence_coverage"] == 0.85
        assert lr["supporting_evidence_count"] == 2

    def test_learning_record_has_exception_type(self):
        """Learning record captures exception type."""
        state = _make_state(exception_type="REFUND_ADJUSTMENT")
        result = record_outcome(state)
        lr = result["learning_record"]
        assert lr["exception_type"] == "REFUND_ADJUSTMENT"

    def test_learning_record_has_nodes_executed(self):
        """Learning record captures nodes executed."""
        state = _make_state()
        state.metadata.nodes_executed = ["load_exception", "gather_evidence"]
        result = record_outcome(state)
        lr = result["learning_record"]
        assert "load_exception" in lr["nodes_executed"]

    def test_node_recorded(self):
        """Node execution is recorded."""
        state = _make_state()
        result = record_outcome(state)
        assert "record_outcome" in result["metadata"]["nodes_executed"]
        log = result["metadata"]["execution_log"][-1]
        assert log["node"] == "record_outcome"
        assert log["success"] is True

    def test_exception_recorded(self):
        """Error state records exception."""
        state = _make_state()
        state.errors = ["Something failed"]
        result = record_outcome(state)
        assert result["outcome"] is not None  # still succeeds


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Isolation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthIsolation:
    def _make_outcome(self, **overrides):
        """Helper to build a WorkflowOutcomeRecord with defaults."""
        defaults = dict(
            workflow_id="WF-001",
            exception_id="EXC-001",
            decision="AUTO",
            outcome=WorkflowOutcome.RESOLVED_AUTO,
        )
        defaults.update(overrides)
        return WorkflowOutcomeRecord(**defaults)

    def test_reward_has_ground_truth_fields(self):
        """Reward signal has ground truth fields for evaluation."""
        state = _make_state()
        outcome = self._make_outcome()
        reward = _calculate_reward(state, outcome)
        # Ground truth fields should be present (None by default)
        assert hasattr(reward, "ground_truth_exception_type")
        assert reward.ground_truth_exception_type is None

    def test_learning_record_has_ground_truth_fields(self):
        """Learning record has ground truth fields for evaluation."""
        state = _make_state()
        result = record_outcome(state)
        lr = result["learning_record"]
        assert "ground_truth_exception_type" in lr
        assert lr["ground_truth_exception_type"] is None

    def test_ground_truth_not_used_in_outcome_determination(self):
        """Outcome is determined without ground truth."""
        state = _make_state(decision="AUTO", verification_status="VERIFIED")
        outcome = _determine_outcome(state)
        # Outcome should be RESOLVED_AUTO based on decision/verification only
        assert outcome == WorkflowOutcome.RESOLVED_AUTO
        # No ground truth was needed

    def test_reward_does_not_affect_outcome(self):
        """Reward calculation does not change the outcome record."""
        state = _make_state()
        outcome = self._make_outcome()
        reward = _calculate_reward(state, outcome)
        # Outcome record is independent of reward
        assert outcome.outcome == WorkflowOutcome.RESOLVED_AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_candidate(self):
        """No selected candidate → outcome still recorded."""
        state = _make_state()
        state.selected_candidate = None
        result = record_outcome(state)
        assert result["outcome"] is not None
        assert result["outcome"]["financial_adjustment_paise"] == 0

    def test_no_evidence(self):
        """No evidence package → outcome still recorded."""
        state = _make_state()
        state.evidence_package = None
        result = record_outcome(state)
        assert result["outcome"] is not None

    def test_no_classification(self):
        """No classification → outcome still recorded."""
        state = _make_state()
        state.classification = None
        result = record_outcome(state)
        assert result["outcome"]["exception_type"] is None

    def test_zero_confidence(self):
        """Zero confidence → outcome still recorded."""
        state = _make_state(confidence=0.0)
        result = record_outcome(state)
        assert result["outcome"]["confidence"] == 0.0

    def test_max_confidence(self):
        """Max confidence → outcome still recorded."""
        state = _make_state(confidence=1.0)
        result = record_outcome(state)
        assert result["outcome"]["confidence"] == 1.0
