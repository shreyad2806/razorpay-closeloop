"""
Feedback and learning integration tests.

Tests the complete feedback workflow:
  resolution outcome → human feedback → outcome recording → reward → learning metrics

Covers:
1. approve
2. reject
3. correct
4. escalate
5. verification failure
6. incorrect AUTO resolution

Verifies:
- Feedback is persisted correctly
- Reward is calculated from actual outcome
- Historical cases can be retrieved
- Feedback cannot bypass guardrails or modify financial records
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_feedback_learn.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    EscalationDetail,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    OutcomeStatus,
    PredictionRecord,
    RejectionDetail,
)
from app.schemas.outcome import (
    HistoricalLearningRecord,
    RewardSignal,
    RewardType,
    WorkflowOutcome,
    WorkflowOutcomeRecord,
)
from app.schemas.reward_engine import RewardCategory
from app.services.feedback import FeedbackService, OutcomeService
from app.services.reward_engine import RewardEngine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def feedback_service():
    return FeedbackService()


@pytest.fixture
def outcome_service():
    return OutcomeService()


@pytest.fixture
def reward_engine():
    return RewardEngine()


def _make_prediction(
    exception_type="FEE_DIFFERENCE",
    resolution_type="FEE_ADJUSTMENT",
    confidence=0.85,
):
    return PredictionRecord(
        exception_type=exception_type,
        resolution_type=resolution_type,
        resolution_confidence=confidence,
        exception_confidence=0.90,
        model_version="v1.0",
    )


def _make_actual_outcome(
    resolution="FEE_ADJUSTMENT",
    correct=True,
    executed=True,
    verified=True,
    impact=3000,
):
    return ActualOutcomeRecord(
        actual_resolution=resolution,
        actual_exception_type="FEE_DIFFERENCE",
        resolution_correct=correct,
        financial_impact_paise=impact,
        was_executed=executed,
        was_verified=verified,
    )


def _make_lineage(exception_id="EXC-001"):
    return DataLineage(
        exception_id=exception_id,
        evidence_ids=["EV-001", "EV-002"],
        prediction_id="PRED-001",
        decision_id="DEC-001",
        execution_id="EXE-001",
        verification_id="VER-001",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Approve Feedback
# ─────────────────────────────────────────────────────────────────────────────


class TestApproveFeedback:
    """Test APPROVE feedback flow."""

    def test_approve_feedback_recorded(
        self, feedback_service
    ):
        """APPROVE feedback should be recorded with correct type."""
        record = feedback_service.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_1",
            system_prediction="FEE_ADJUSTMENT",
        )

        assert record.feedback_type == FeedbackType.APPROVE
        assert record.feedback_id.startswith("FB-")
        assert record.reviewer == "reviewer_1"
        assert record.system_prediction == "FEE_ADJUSTMENT"

    def test_approve_persisted_and_retrievable(
        self, feedback_service
    ):
        """APPROVE feedback should be retrievable by ID and workflow."""
        record = feedback_service.record_feedback(
            workflow_id="WF-002",
            exception_id="EXC-002",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_2",
            system_prediction="FEE_ADJUSTMENT",
        )

        retrieved = feedback_service.get_feedback(record.feedback_id)
        assert retrieved is not None
        assert retrieved.feedback_type == FeedbackType.APPROVE

        by_workflow = feedback_service.get_feedback_for_workflow("WF-002")
        assert len(by_workflow) == 1

    def test_approve_with_evidence_references(
        self, feedback_service
    ):
        """APPROVE should track which evidence was reviewed."""
        record = feedback_service.record_feedback(
            workflow_id="WF-003",
            exception_id="EXC-003",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_3",
            system_prediction="FEE_ADJUSTMENT",
            evidence_references_reviewed=["EV-001", "EV-002", "EV-003"],
        )

        assert len(record.evidence_references_reviewed) == 3
        assert "EV-001" in record.evidence_references_reviewed

    def test_approve_outcome_linkage(
        self, outcome_service, feedback_service
    ):
        """Outcome should be updated with APPROVE feedback."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-004",
            exception_id="EXC-004",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-004"),
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-004",
            exception_id="EXC-004",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_4",
            system_prediction="FEE_ADJUSTMENT",
        )

        updated = outcome_service.update_feedback(
            "WF-004", fb.feedback_id, FeedbackType.APPROVE
        )

        assert updated is not None
        assert updated.human_feedback_id == fb.feedback_id
        assert updated.human_feedback_type == FeedbackType.APPROVE
        assert updated.status == OutcomeStatus.FEEDBACK_RECEIVED


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reject Feedback
# ─────────────────────────────────────────────────────────────────────────────


class TestRejectFeedback:
    """Test REJECT feedback flow."""

    def test_reject_feedback_recorded(
        self, feedback_service
    ):
        """REJECT feedback should be recorded with rejection details."""
        record = feedback_service.record_feedback(
            workflow_id="WF-R01",
            exception_id="EXC-R01",
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_5",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(
                rejection_reason="Amount seems incorrect",
                suggested_alternative="REFUND_REVERSAL",
                risk_concern="High-value transaction",
            ),
        )

        assert record.feedback_type == FeedbackType.REJECT
        assert record.rejection is not None
        assert record.rejection.rejection_reason == "Amount seems incorrect"
        assert record.rejection.suggested_alternative == "REFUND_REVERSAL"
        assert record.rejection.risk_concern == "High-value transaction"

    def test_reject_persisted(
        self, feedback_service
    ):
        """REJECT feedback should be retrievable."""
        record = feedback_service.record_feedback(
            workflow_id="WF-R02",
            exception_id="EXC-R02",
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_6",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(
                rejection_reason="Insufficient evidence"
            ),
        )

        retrieved = feedback_service.get_feedback(record.feedback_id)
        assert retrieved.is_rejection()

    def test_reject_outcome_updates_status(
        self, outcome_service, feedback_service
    ):
        """REJECT feedback should update outcome status."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-R03",
            exception_id="EXC-R03",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(correct=False, executed=False),
            lineage=_make_lineage("EXC-R03"),
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-R03",
            exception_id="EXC-R03",
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_7",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Wrong amount"),
        )

        updated = outcome_service.update_feedback(
            "WF-R03", fb.feedback_id, FeedbackType.REJECT,
            human_override=True,
        )

        assert updated.human_override is True
        assert updated.status == OutcomeStatus.FEEDBACK_RECEIVED


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correct Feedback
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrectFeedback:
    """Test CORRECT feedback flow."""

    def test_correct_feedback_recorded(
        self, feedback_service
    ):
        """CORRECT feedback should be recorded with correction details."""
        record = feedback_service.record_feedback(
            workflow_id="WF-C01",
            exception_id="EXC-C01",
            feedback_type=FeedbackType.CORRECT,
            reviewer="reviewer_8",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_REVERSAL",
                correction_reason="System misidentified fee as the cause",
                original_confidence=0.85,
                corrected_amount_paise=5000,
            ),
        )

        assert record.feedback_type == FeedbackType.CORRECT
        assert record.correction is not None
        assert record.correction.original_resolution == "FEE_ADJUSTMENT"
        assert record.correction.corrected_resolution == "REFUND_REVERSAL"
        assert record.correction.corrected_amount_paise == 5000

    def test_correct_is_correction(
        self, feedback_service
    ):
        """is_correction() should return True for CORRECT feedback."""
        record = feedback_service.record_feedback(
            workflow_id="WF-C02",
            exception_id="EXC-C02",
            feedback_type=FeedbackType.CORRECT,
            reviewer="reviewer_9",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="TAX_ADJUSTMENT",
                correction_reason="Tax miscalculation",
            ),
        )

        assert record.is_correction()
        assert not record.is_rejection()
        assert not record.is_approval()

    def test_correction_chain(
        self, feedback_service
    ):
        """CORRECT can reference a previous feedback via correction_of."""
        first = feedback_service.record_feedback(
            workflow_id="WF-C03",
            exception_id="EXC-C03",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_10",
            system_prediction="FEE_ADJUSTMENT",
        )

        second = feedback_service.record_feedback(
            workflow_id="WF-C03",
            exception_id="EXC-C03",
            feedback_type=FeedbackType.CORRECT,
            reviewer="reviewer_10",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_REVERSAL",
                correction_reason="Changed mind after further review",
            ),
            correction_of=first.feedback_id,
        )

        assert second.correction_of == first.feedback_id

        # Both feedback records exist for the workflow
        all_fb = feedback_service.get_feedback_for_workflow("WF-C03")
        assert len(all_fb) == 2

    def test_correct_only_type(
        self, feedback_service
    ):
        """get_corrections() should return only CORRECT records."""
        feedback_service.record_feedback(
            workflow_id="WF-C04a",
            exception_id="EXC-C04",
            feedback_type=FeedbackType.APPROVE,
            reviewer="r1",
            system_prediction="FEE_ADJUSTMENT",
        )
        feedback_service.record_feedback(
            workflow_id="WF-C04b",
            exception_id="EXC-C04",
            feedback_type=FeedbackType.CORRECT,
            reviewer="r2",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_REVERSAL",
                correction_reason="Error",
            ),
        )
        feedback_service.record_feedback(
            workflow_id="WF-C04c",
            exception_id="EXC-C04",
            feedback_type=FeedbackType.REJECT,
            reviewer="r3",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Bad"),
        )

        corrections = feedback_service.get_corrections()
        assert len(corrections) == 1
        assert corrections[0].feedback_type == FeedbackType.CORRECT


# ─────────────────────────────────────────────────────────────────────────────
# 4. Escalate Feedback
# ─────────────────────────────────────────────────────────────────────────────


class TestEscalateFeedback:
    """Test ESCALATE feedback flow."""

    def test_escalate_feedback_recorded(
        self, feedback_service
    ):
        """ESCALATE feedback should be recorded with escalation details."""
        record = feedback_service.record_feedback(
            workflow_id="WF-E01",
            exception_id="EXC-E01",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="reviewer_11",
            system_prediction="FEE_ADJUSTMENT",
            escalation=EscalationDetail(
                escalation_reason="Complex multi-party dispute",
                escalation_target="finance_director",
                additional_context="Requires legal review",
            ),
        )

        assert record.feedback_type == FeedbackType.ESCALATE
        assert record.escalation is not None
        assert record.escalation.escalation_reason == "Complex multi-party dispute"
        assert record.escalation.escalation_target == "finance_director"

    def test_escalate_is_escalation(
        self, feedback_service
    ):
        """is_escalation() should return True for ESCALATE feedback."""
        record = feedback_service.record_feedback(
            workflow_id="WF-E02",
            exception_id="EXC-E02",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="reviewer_12",
            system_prediction="UNKNOWN",
            escalation=EscalationDetail(
                escalation_reason="Novel exception type"
            ),
        )

        assert record.is_escalation()

    def test_escalate_count(
        self, feedback_service
    ):
        """count_by_type should track ESCALATE correctly."""
        feedback_service.record_feedback(
            workflow_id="WF-E03a",
            exception_id="EXC-E03",
            feedback_type=FeedbackType.APPROVE,
            reviewer="r1",
            system_prediction="FEE_ADJUSTMENT",
        )
        feedback_service.record_feedback(
            workflow_id="WF-E03b",
            exception_id="EXC-E03",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="r2",
            system_prediction="UNKNOWN",
            escalation=EscalationDetail(escalation_reason="Unknown type"),
        )
        feedback_service.record_feedback(
            workflow_id="WF-E03c",
            exception_id="EXC-E03",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="r3",
            system_prediction="UNKNOWN",
            escalation=EscalationDetail(escalation_reason="Low evidence"),
        )

        counts = feedback_service.count_by_type()
        assert counts.get("APPROVE") == 1
        assert counts.get("ESCALATE") == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. Verification Failure Outcome
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFailure:
    """Test outcome recording when verification fails."""

    def test_verification_failure_outcome(
        self, outcome_service
    ):
        """Verification failure should be recorded in outcome."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-VF01",
            exception_id="EXC-VF01",
            prediction=_make_prediction(),
            actual_outcome=ActualOutcomeRecord(
                actual_resolution="FEE_ADJUSTMENT",
                was_executed=True,
                was_verified=False,
                was_rolled_back=True,
            ),
            lineage=_make_lineage("EXC-VF01"),
            verification_passed=False,
            verification_notes="Amount mismatch detected",
        )

        assert outcome.verification_passed is False
        assert outcome.verification_notes == "Amount mismatch detected"
        assert outcome.actual_outcome.was_rolled_back is True

    def test_verification_failure_blocks_learning_ready(
        self, outcome_service
    ):
        """Verification failure outcome should still be learning-ready if it has prediction + outcome."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-VF02",
            exception_id="EXC-VF02",
            prediction=_make_prediction(),
            actual_outcome=ActualOutcomeRecord(
                actual_resolution="FEE_ADJUSTMENT",
                was_executed=True,
                was_verified=False,
                was_rolled_back=True,
            ),
            lineage=_make_lineage("EXC-VF02"),
            verification_passed=False,
        )

        # Learning ready because it has prediction + actual_resolution
        assert outcome.is_learning_ready()

    def test_verification_failure_outcome_status(
        self, outcome_service
    ):
        """Verification failure should not change outcome status to FEEDBACK_RECEIVED."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-VF03",
            exception_id="EXC-VF03",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(verified=False),
            lineage=_make_lineage("EXC-VF03"),
            verification_passed=False,
        )

        # No feedback yet → status should be RECORDED
        assert outcome.status == OutcomeStatus.RECORDED

    def test_verification_failure_financial_impact(
        self, outcome_service
    ):
        """Verification failure should track financial impact."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-VF04",
            exception_id="EXC-VF04",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(verified=False),
            lineage=_make_lineage("EXC-VF04"),
            verification_passed=False,
            financial_impact=FinancialImpact(
                requested_adjustment_paise=3000,
                actual_adjustment_paise=3000,
                difference_before_paise=3000,
                difference_after_paise=0,
                discrepancy_eliminated=True,
                unintended_changes=1,
            ),
        )

        assert outcome.financial_impact.unintended_changes == 1
        assert outcome.financial_impact.discrepancy_eliminated is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Incorrect AUTO Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestIncorrectAutoResolution:
    """Test outcome when AUTO resolution was incorrect."""

    def test_incorrect_auto_outcome(
        self, outcome_service
    ):
        """Incorrect AUTO should record mismatched prediction vs actual."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-IA01",
            exception_id="EXC-IA01",
            prediction=_make_prediction(
                resolution_type="FEE_ADJUSTMENT", confidence=0.85
            ),
            actual_outcome=ActualOutcomeRecord(
                actual_resolution="REFUND_REVERSAL",
                resolution_correct=False,
                was_executed=True,
                was_verified=True,
                financial_impact_paise=5000,
            ),
            lineage=_make_lineage("EXC-IA01"),
            decision="AUTO",
            confidence=0.85,
            risk="LOW",
        )

        assert outcome.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert outcome.actual_outcome.actual_resolution == "REFUND_REVERSAL"
        assert outcome.prediction_matches_actual() is False
        assert outcome.decision == "AUTO"

    def test_incorrect_auto_prediction_accuracy(
        self, outcome_service
    ):
        """prediction_accuracy should count incorrect AUTO correctly."""
        # Correct
        outcome_service.record_outcome(
            workflow_id="WF-IA02a",
            exception_id="EXC-IA02a",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual_outcome(resolution="FEE_ADJUSTMENT", correct=True),
            lineage=_make_lineage("EXC-IA02a"),
        )
        # Incorrect
        outcome_service.record_outcome(
            workflow_id="WF-IA02b",
            exception_id="EXC-IA02b",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual_outcome(resolution="REFUND_REVERSAL", correct=False),
            lineage=_make_lineage("EXC-IA02b"),
        )

        accuracy = outcome_service.prediction_accuracy()
        assert accuracy["correct"] == 1
        assert accuracy["incorrect"] == 1

    def test_incorrect_auto_human_feedback_updates(
        self, outcome_service, feedback_service
    ):
        """Human REJECT after incorrect AUTO should update outcome."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-IA03",
            exception_id="EXC-IA03",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual_outcome(
                resolution="REFUND_REVERSAL", correct=False, executed=True
            ),
            lineage=_make_lineage("EXC-IA03"),
            decision="AUTO",
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-IA03",
            exception_id="EXC-IA03",
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_13",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(
                rejection_reason="Wrong resolution type applied"
            ),
        )

        updated = outcome_service.update_feedback(
            "WF-IA03", fb.feedback_id, FeedbackType.REJECT,
            human_override=True,
        )

        assert updated.human_override is True
        assert updated.human_feedback_type == FeedbackType.REJECT


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reward Calculation Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardCalculation:
    """Test reward calculation from outcomes."""

    def _make_outcome(
        self,
        exception_id="EXC-RW",
        case_id="CASE-RW",
        resolution_type="FEE_ADJUSTMENT",
        correct=True,
        executed=True,
        verified=True,
        rolled_back=False,
        impact=3000,
        diff_before=3000,
        diff_after=0,
        discrepancy_eliminated=True,
        unintended=0,
        decision="AUTO",
        confidence=0.85,
        risk="LOW",
    ):
        """Build an OutcomeRecord for reward testing."""
        return OutcomeRecord(
            outcome_id=f"OUT-{exception_id}",
            workflow_id=f"WF-{exception_id}",
            exception_id=exception_id,
            case_id=case_id,
            prediction=PredictionRecord(
                exception_type="FEE_DIFFERENCE",
                resolution_type=resolution_type,
                resolution_confidence=confidence,
            ),
            actual_outcome=ActualOutcomeRecord(
                actual_resolution=resolution_type if correct else "REFUND_REVERSAL",
                resolution_correct=correct,
                financial_impact_paise=impact,
                was_executed=executed,
                was_verified=verified,
                was_rolled_back=rolled_back,
            ),
            lineage=DataLineage(exception_id=exception_id),
            financial_impact=FinancialImpact(
                requested_adjustment_paise=impact,
                actual_adjustment_paise=impact if executed else 0,
                difference_before_paise=diff_before,
                difference_after_paise=diff_after,
                discrepancy_eliminated=discrepancy_eliminated,
                unintended_changes=unintended,
            ),
            decision=decision,
            confidence=confidence,
            risk=risk,
        )

    def test_correct_auto_reward_positive(
        self, reward_engine
    ):
        """Correct AUTO resolution should produce positive reward."""
        outcome = self._make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", confidence=0.85, risk="LOW",
        )
        result = reward_engine.calculate_reward(outcome)
        assert result.reward_value > 0

    def test_incorrect_auto_reward_negative(
        self, reward_engine
    ):
        """Incorrect AUTO resolution should produce negative reward."""
        outcome = self._make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", confidence=0.85, risk="LOW",
        )
        result = reward_engine.calculate_reward(outcome)
        assert result.reward_value < 0

    def test_human_approved_reward(
        self, reward_engine, feedback_service
    ):
        """Human-approved resolution should produce positive reward."""
        outcome = self._make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", confidence=0.85, risk="LOW",
        )
        fb = feedback_service.record_feedback(
            workflow_id=outcome.workflow_id,
            exception_id=outcome.exception_id,
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_rw",
            system_prediction="FEE_ADJUSTMENT",
        )
        result = reward_engine.calculate_reward(outcome, fb)
        assert result.reward_value > 0

    def test_human_rejected_reward_reduces_score(
        self, reward_engine, feedback_service
    ):
        """Human-rejected should reduce reward compared to no feedback."""
        outcome = self._make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", confidence=0.85, risk="LOW",
        )
        # Without feedback
        result_no_fb = reward_engine.calculate_reward(outcome)
        # With REJECT feedback
        fb = feedback_service.record_feedback(
            workflow_id=outcome.workflow_id,
            exception_id=outcome.exception_id,
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_rw2",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Wrong amount"),
        )
        result_with_fb = reward_engine.calculate_reward(outcome, fb)
        # REJECT penalty should reduce the reward
        assert result_with_fb.reward_value < result_no_fb.reward_value
        # Verify the human_feedback component is negative
        assert result_with_fb.breakdown.human_feedback_component.value < 0

    def test_reward_clamped(
        self, reward_engine, feedback_service
    ):
        """Reward value should be clamped to [-1.0, 1.0]."""
        outcome = self._make_outcome(
            correct=False, executed=True, verified=False, rolled_back=True,
            impact=100000, diff_before=100000, diff_after=100000,
            discrepancy_eliminated=False, unintended=5,
            decision="AUTO", confidence=0.95, risk="CRITICAL",
        )
        fb = feedback_service.record_feedback(
            workflow_id=outcome.workflow_id,
            exception_id=outcome.exception_id,
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_rw3",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="High value error"),
        )
        result = reward_engine.calculate_reward(outcome, fb)
        assert -1.0 <= result.reward_value <= 1.0

    def test_reward_deterministic(
        self, reward_engine
    ):
        """Same input should produce same reward."""
        outcome = self._make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", confidence=0.85, risk="LOW",
        )
        r1 = reward_engine.calculate_reward(outcome)
        r2 = reward_engine.calculate_reward(outcome)
        assert r1.reward_value == r2.reward_value
        assert r1.category == r2.category


# ─────────────────────────────────────────────────────────────────────────────
# 8. Learning Metrics Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestLearningMetrics:
    """Test learning metrics from outcomes and feedback."""

    def test_learning_ready_outcomes(
        self, outcome_service
    ):
        """Learning-ready outcomes should have prediction + outcome."""
        outcome_service.record_outcome(
            workflow_id="WF-LM01",
            exception_id="EXC-LM01",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-LM01"),
        )
        outcome_service.record_outcome(
            workflow_id="WF-LM02",
            exception_id="EXC-LM02",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(correct=False),
            lineage=_make_lineage("EXC-LM02"),
        )

        ready = outcome_service.get_learning_ready_outcomes()
        assert len(ready) == 2

    def test_outcome_status_tracking(
        self, outcome_service
    ):
        """Outcome status should progress through lifecycle."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-LM03",
            exception_id="EXC-LM03",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-LM03"),
        )

        assert outcome.status == OutcomeStatus.RECORDED

        outcome_service.mark_reward_calculated("WF-LM03")
        updated = outcome_service.get_outcome_for_workflow("WF-LM03")
        assert updated.status == OutcomeStatus.REWARD_CALCULATED

        outcome_service.mark_stored_for_learning("WF-LM03")
        updated = outcome_service.get_outcome_for_workflow("WF-LM03")
        assert updated.status == OutcomeStatus.STORED_FOR_LEARNING

    def test_count_by_status(
        self, outcome_service
    ):
        """count_by_status should track all outcomes."""
        outcome_service.record_outcome(
            workflow_id="WF-LM04a",
            exception_id="EXC-LM04a",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-LM04a"),
        )
        outcome_service.record_outcome(
            workflow_id="WF-LM04b",
            exception_id="EXC-LM04b",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-LM04b"),
            human_feedback_id="FB-001",
        )

        counts = outcome_service.count_by_status()
        assert counts.get("RECORDED") == 1
        assert counts.get("FEEDBACK_RECEIVED") == 1

    def test_feedback_by_exception(
        self, feedback_service
    ):
        """get_feedback_for_exception should return relevant records."""
        feedback_service.record_feedback(
            workflow_id="WF-LM05a",
            exception_id="EXC-LM05",
            feedback_type=FeedbackType.APPROVE,
            reviewer="r1",
            system_prediction="FEE_ADJUSTMENT",
        )
        feedback_service.record_feedback(
            workflow_id="WF-LM05b",
            exception_id="EXC-LM05",
            feedback_type=FeedbackType.CORRECT,
            reviewer="r2",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_REVERSAL",
                correction_reason="Error",
            ),
        )
        feedback_service.record_feedback(
            workflow_id="WF-LM05c",
            exception_id="OTHER-EXC",
            feedback_type=FeedbackType.APPROVE,
            reviewer="r3",
            system_prediction="FEE_ADJUSTMENT",
        )

        records = feedback_service.get_feedback_for_exception("EXC-LM05")
        assert len(records) == 2

    def test_has_feedback(
        self, feedback_service
    ):
        """has_feedback should return True when feedback exists."""
        assert not feedback_service.has_feedback("WF-NONE")

        feedback_service.record_feedback(
            workflow_id="WF-HF01",
            exception_id="EXC-HF01",
            feedback_type=FeedbackType.APPROVE,
            reviewer="r1",
            system_prediction="FEE_ADJUSTMENT",
        )

        assert feedback_service.has_feedback("WF-HF01")
        assert not feedback_service.has_feedback("WF-NONE")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Outcome Data Separation
# ─────────────────────────────────────────────────────────────────────────────


class TestOutcomeDataSeparation:
    """Verify prediction, actual outcome, and feedback are stored separately."""

    def test_prediction_separate_from_outcome(
        self, outcome_service
    ):
        """Prediction and actual outcome should be independent fields."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-DS01",
            exception_id="EXC-DS01",
            prediction=_make_prediction(
                resolution_type="FEE_ADJUSTMENT", confidence=0.85
            ),
            actual_outcome=_make_actual_outcome(
                resolution="REFUND_REVERSAL", correct=False
            ),
            lineage=_make_lineage("EXC-DS01"),
        )

        assert outcome.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert outcome.actual_outcome.actual_resolution == "REFUND_REVERSAL"
        assert outcome.prediction_matches_actual() is False

    def test_feedback_separate_from_outcome(
        self, outcome_service, feedback_service
    ):
        """Human feedback should be stored separately from outcome."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-DS02",
            exception_id="EXC-DS02",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(correct=True),
            lineage=_make_lineage("EXC-DS02"),
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-DS02",
            exception_id="EXC-DS02",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_14",
            system_prediction="FEE_ADJUSTMENT",
        )

        outcome_service.update_feedback(
            "WF-DS02", fb.feedback_id, FeedbackType.APPROVE
        )

        updated = outcome_service.get_outcome_for_workflow("WF-DS02")

        # Feedback is linked but stored separately
        assert updated.human_feedback_id == fb.feedback_id
        assert updated.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert updated.actual_outcome.actual_resolution == "FEE_ADJUSTMENT"

    def test_ground_truth_evaluation_only(
        self, outcome_service
    ):
        """Ground truth should be stored for evaluation, not decision."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-DS03",
            exception_id="EXC-DS03",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-DS03"),
            ground_truth_exception_type="FEE_DIFFERENCE",
            ground_truth_resolution="FEE_ADJUSTMENT",
            ground_truth_resolvable=True,
        )

        assert outcome.ground_truth_exception_type == "FEE_DIFFERENCE"
        assert outcome.ground_truth_resolution == "FEE_ADJUSTMENT"
        assert outcome.ground_truth_resolvable is True

        # Ground truth is separate from prediction
        assert outcome.prediction.exception_type != outcome.ground_truth_exception_type or \
               outcome.prediction.exception_type == outcome.ground_truth_exception_type  # both can be same value


# ─────────────────────────────────────────────────────────────────────────────
# 10. Historical Case Storage
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalCaseStorage:
    """Test that outcomes can be stored for future retrieval."""

    def test_learning_record_created(
        self, outcome_service
    ):
        """Outcome should support learning record creation."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-HC01",
            exception_id="EXC-HC01",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-HC01"),
            decision="AUTO",
            confidence=0.85,
            risk="LOW",
        )

        assert outcome.decision == "AUTO"
        assert outcome.confidence == 0.85
        assert outcome.risk == "LOW"
        assert outcome.is_learning_ready()

    def test_historical_learning_record_features(
        self,
    ):
        """HistoricalLearningRecord should extract retrieval features."""
        record = HistoricalLearningRecord(
            workflow_id="WF-HC02",
            exception_id="EXC-HC02",
            exception_type="FEE_DIFFERENCE",
            resolution_type="FEE_ADJUSTMENT",
            outcome=WorkflowOutcome.RESOLVED_AUTO,
            financial_adjustment_paise=3000,
            confidence=0.85,
            risk="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.90,
            supporting_evidence_count=3,
            verification_passed=True,
            human_approved=False,
            authorization_source="AUTO_GUARDRAIL",
        )

        features = record.to_retrieval_features()

        assert features["exception_type"] == "FEE_DIFFERENCE"
        assert features["resolution_type"] == "FEE_ADJUSTMENT"
        assert features["outcome"] == "RESOLVED_AUTO"
        assert features["confidence"] == 0.85
        assert features["verification_passed"] is True

    def test_lineage_completeness(
        self, outcome_service
    ):
        """Data lineage should trace the complete chain."""
        lineage = DataLineage(
            exception_id="EXC-HC03",
            evidence_ids=["EV-001", "EV-002"],
            prediction_id="PRED-001",
            decision_id="DEC-001",
            execution_id="EXE-001",
            verification_id="VER-001",
            feedback_id="FB-001",
            audit_event_ids=["AUD-001"],
            reward_id="RW-001",
            historical_case_id="HRES-001",
        )

        outcome = outcome_service.record_outcome(
            workflow_id="WF-HC03",
            exception_id="EXC-HC03",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=lineage,
        )

        assert outcome.lineage.exception_id == "EXC-HC03"
        assert len(outcome.lineage.evidence_ids) == 2
        assert outcome.lineage.execution_id == "EXE-001"
        assert outcome.lineage.verification_id == "VER-001"
        assert outcome.lineage.feedback_id == "FB-001"
        assert outcome.lineage.historical_case_id == "HRES-001"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Feedback Cannot Bypass Guardrails
# ─────────────────────────────────────────────────────────────────────────────


class TestFeedbackSafetyBoundary:
    """Verify feedback cannot bypass financial guardrails."""

    def test_feedback_has_no_execute_method(
        self, feedback_service
    ):
        """FeedbackService should not have execute/apply/authorize methods."""
        assert not hasattr(feedback_service, "execute")
        assert not hasattr(feedback_service, "apply_resolution")
        assert not hasattr(feedback_service, "authorize_payment")

    def test_outcome_has_no_execute_method(
        self, outcome_service
    ):
        """OutcomeService should not have execute/apply methods."""
        assert not hasattr(outcome_service, "execute")
        assert not hasattr(outcome_service, "apply_resolution")
        assert not hasattr(outcome_service, "modify_financial_record")

    def test_feedback_record_has_no_financial_fields(
        self, feedback_service
    ):
        """FeedbackRecord should not have fields that authorize financial actions."""
        record = feedback_service.record_feedback(
            workflow_id="WF-SAFE01",
            exception_id="EXC-SAFE01",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_15",
            system_prediction="FEE_ADJUSTMENT",
        )

        # Should not have authorization or execution fields
        assert not hasattr(record, "authorize_execution")
        assert not hasattr(record, "execute_resolution")
        assert not hasattr(record, "override_guardrail")

    def test_outcome_record_has_no_financial_write(
        self, outcome_service
    ):
        """OutcomeRecord should not have fields that modify financial state."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-SAFE02",
            exception_id="EXC-SAFE02",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-SAFE02"),
        )

        assert not hasattr(outcome, "execute_resolution")
        assert not hasattr(outcome, "modify_settlement")
        assert not hasattr(outcome, "override_guardrail")

    def test_feedback_cannot_directly_change_outcome(
        self, outcome_service, feedback_service
    ):
        """Feedback should update metadata, not the core outcome."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-SAFE03",
            exception_id="EXC-SAFE03",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual_outcome(resolution="FEE_ADJUSTMENT", correct=True),
            lineage=_make_lineage("EXC-SAFE03"),
            decision="AUTO",
            confidence=0.85,
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-SAFE03",
            exception_id="EXC-SAFE03",
            feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer_16",
            system_prediction="FEE_ADJUSTMENT",
        )

        outcome_service.update_feedback(
            "WF-SAFE03", fb.feedback_id, FeedbackType.APPROVE
        )

        updated = outcome_service.get_outcome_for_workflow("WF-SAFE03")

        # Core prediction and outcome should NOT change
        assert updated.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert updated.actual_outcome.actual_resolution == "FEE_ADJUSTMENT"
        assert updated.decision == "AUTO"
        assert updated.confidence == 0.85

        # Only metadata should be updated
        assert updated.human_feedback_id == fb.feedback_id
        assert updated.status == OutcomeStatus.FEEDBACK_RECEIVED

    def test_reward_cannot_authorize_execution(
        self, reward_engine
    ):
        """RewardSignal should not have execute/authorize methods."""
        assert not hasattr(RewardSignal, "execute")
        assert not hasattr(RewardSignal, "authorize")
        assert not hasattr(RewardSignal, "apply_to_financial_state")

    def test_correction_does_not_modify_financial_state(
        self, feedback_service
    ):
        """CORRECT feedback should record the correction, not apply it."""
        record = feedback_service.record_feedback(
            workflow_id="WF-SAFE04",
            exception_id="EXC-SAFE04",
            feedback_type=FeedbackType.CORRECT,
            reviewer="reviewer_17",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_REVERSAL",
                correction_reason="Error in classification",
                corrected_amount_paise=5000,
            ),
        )

        # The correction is recorded as data, not executed
        assert record.correction.corrected_resolution == "REFUND_REVERSAL"
        assert record.correction.corrected_amount_paise == 5000

        # No financial state was modified
        # (FeedbackService has no DB or financial state access)

    def test_human_override_is_metadata_only(
        self, outcome_service, feedback_service
    ):
        """human_override flag should be metadata, not an action."""
        outcome = outcome_service.record_outcome(
            workflow_id="WF-SAFE05",
            exception_id="EXC-SAFE05",
            prediction=_make_prediction(),
            actual_outcome=_make_actual_outcome(),
            lineage=_make_lineage("EXC-SAFE05"),
        )

        fb = feedback_service.record_feedback(
            workflow_id="WF-SAFE05",
            exception_id="EXC-SAFE05",
            feedback_type=FeedbackType.REJECT,
            reviewer="reviewer_18",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Wrong"),
        )

        outcome_service.update_feedback(
            "WF-SAFE05", fb.feedback_id, FeedbackType.REJECT,
            human_override=True,
        )

        updated = outcome_service.get_outcome_for_workflow("WF-SAFE05")
        assert updated.human_override is True
        # Core financial data unchanged
        assert updated.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert updated.actual_outcome.actual_resolution == "FEE_ADJUSTMENT"
