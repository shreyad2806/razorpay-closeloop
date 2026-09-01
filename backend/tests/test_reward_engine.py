"""
Tests for Phase 9B — Reward System.

Tests cover:
- Reward category determination
- Financial risk classification
- Transparent reward calculation
- All 7 reward categories
- High-value error penalty
- Verification failure penalty
- Human feedback modifiers
- Confidence modifiers
- Reproducibility
- Edge cases
"""

import pytest

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    PredictionRecord,
    RejectionDetail,
)
from app.schemas.reward_engine import (
    FinancialRiskLevel,
    RewardBreakdown,
    RewardCategory,
    RewardComponent,
    RewardConfig,
    RewardRecord,
    RewardWeights,
)
from app.services.reward_engine import (
    RewardEngine,
    classify_financial_risk,
    determine_reward_category,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_prediction(
    resolution_type: str = "FEE_ADJUSTMENT",
    confidence: float = 0.85,
    exception_type: str = "FEE_DIFFERENCE",
) -> PredictionRecord:
    return PredictionRecord(
        exception_type=exception_type,
        resolution_type=resolution_type,
        resolution_confidence=confidence,
        exception_confidence=0.9,
        model_version="xgb-v1.0",
    )


def _make_actual(
    resolution: str = "FEE_ADJUSTMENT",
    correct: bool = True,
    executed: bool = True,
    verified: bool = True,
    rolled_back: bool = False,
    impact: int = 3000,
) -> ActualOutcomeRecord:
    return ActualOutcomeRecord(
        actual_resolution=resolution,
        actual_exception_type="FEE_DIFFERENCE",
        resolution_correct=correct,
        financial_impact_paise=impact,
        was_executed=executed,
        was_verified=verified,
        was_rolled_back=rolled_back,
    )


def _make_lineage(exception_id: str = "EXC-001") -> DataLineage:
    return DataLineage(exception_id=exception_id, evidence_ids=["EVD-001"])


def _make_outcome(
    workflow_id: str = "WF-001",
    exception_id: str = "EXC-001",
    resolution: str = "FEE_ADJUSTMENT",
    correct: bool = True,
    executed: bool = True,
    verified: bool = True,
    rolled_back: bool = False,
    adjustment: int = 3000,
    decision: str = "AUTO",
    confidence: float = 0.85,
    discrepancy_eliminated: bool = True,
    unintended: int = 0,
    feedback_id: str = None,
    feedback_type: FeedbackType = None,
    ground_truth_type: str = None,
) -> OutcomeRecord:
    impact = FinancialImpact(
        requested_adjustment_paise=adjustment,
        actual_adjustment_paise=adjustment,
        difference_before_paise=adjustment,
        difference_after_paise=0 if discrepancy_eliminated else adjustment,
        discrepancy_eliminated=discrepancy_eliminated,
        unintended_changes=unintended,
    )
    return OutcomeRecord(
        outcome_id=f"OUT-{workflow_id}",
        workflow_id=workflow_id,
        exception_id=exception_id,
        prediction=_make_prediction(resolution_type=resolution, confidence=confidence),
        actual_outcome=_make_actual(
            resolution=resolution, correct=correct, executed=executed,
            verified=verified, rolled_back=rolled_back, impact=adjustment,
        ),
        financial_impact=impact,
        lineage=_make_lineage(exception_id),
        decision=decision,
        confidence=confidence,
        verification_passed=verified,
        human_feedback_id=feedback_id,
        human_feedback_type=feedback_type,
        ground_truth_exception_type=ground_truth_type,
    )


def _make_feedback(
    workflow_id: str = "WF-001",
    exception_id: str = "EXC-001",
    feedback_type: FeedbackType = FeedbackType.APPROVE,
    system_prediction: str = "FEE_ADJUSTMENT",
    correction: CorrectionDetail = None,
    rejection: RejectionDetail = None,
) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=f"FB-{workflow_id}",
        workflow_id=workflow_id,
        exception_id=exception_id,
        feedback_type=feedback_type,
        reviewer="test_reviewer",
        system_prediction=system_prediction,
        correction=correction,
        rejection=rejection,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Financial Risk Classification Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialRiskClassification:
    """Test financial risk level classification."""

    def test_negligible(self):
        assert classify_financial_risk(5000) == FinancialRiskLevel.NEGLIGIBLE  # ₹50

    def test_low(self):
        assert classify_financial_risk(50000) == FinancialRiskLevel.LOW  # ₹500

    def test_medium(self):
        assert classify_financial_risk(500000) == FinancialRiskLevel.MEDIUM  # ₹5,000

    def test_high(self):
        # HIGH range: (1,000,000, 10,000,000] i.e. ₹10,001 – ₹1,00,000
        assert classify_financial_risk(5000000) == FinancialRiskLevel.HIGH  # ₹50,000

    def test_critical(self):
        assert classify_financial_risk(50000000) == FinancialRiskLevel.CRITICAL  # ₹5,00,000

    def test_zero(self):
        assert classify_financial_risk(0) == FinancialRiskLevel.NEGLIGIBLE

    def test_boundary_negligible_low(self):
        assert classify_financial_risk(9999) == FinancialRiskLevel.NEGLIGIBLE
        assert classify_financial_risk(10000) == FinancialRiskLevel.LOW

    def test_boundary_low_medium(self):
        assert classify_financial_risk(99999) == FinancialRiskLevel.LOW
        assert classify_financial_risk(100000) == FinancialRiskLevel.MEDIUM

    def test_negative_amount(self):
        """Negative amounts classified by absolute value."""
        assert classify_financial_risk(-5000000) == FinancialRiskLevel.HIGH  # -₹50,000


# ─────────────────────────────────────────────────────────────────────────────
# Reward Category Determination Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardCategoryDetermination:
    """Test reward category determination logic."""

    def test_correct_auto_resolution(self):
        """System auto-resolved correctly."""
        outcome = _make_outcome(
            correct=True, executed=True, verified=True, decision="AUTO"
        )
        cat = determine_reward_category(outcome)
        assert cat == RewardCategory.CORRECT_AUTO_RESOLUTION

    def test_incorrect_auto_resolution(self):
        """System auto-resolved incorrectly."""
        outcome = _make_outcome(
            correct=False, executed=True, verified=True, decision="AUTO"
        )
        cat = determine_reward_category(outcome)
        assert cat == RewardCategory.INCORRECT_AUTO_RESOLUTION

    def test_high_value_error(self):
        """System auto-resolved incorrectly with high value."""
        outcome = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=50_000_000,  # ₹5,00,000
        )
        cat = determine_reward_category(outcome)
        assert cat == RewardCategory.HIGH_VALUE_ERROR

    def test_verification_failure(self):
        """Verification failed and was rolled back."""
        outcome = _make_outcome(
            executed=True, verified=False, rolled_back=True, decision="AUTO"
        )
        cat = determine_reward_category(outcome)
        assert cat == RewardCategory.VERIFICATION_FAILURE

    def test_human_confirmed(self):
        """Human approved the resolution."""
        feedback = _make_feedback(feedback_type=FeedbackType.APPROVE)
        outcome = _make_outcome(
            correct=True, executed=False, decision="HUMAN_REVIEW",
            feedback_id="FB-WF-001", feedback_type=FeedbackType.APPROVE,
        )
        cat = determine_reward_category(outcome, feedback)
        assert cat == RewardCategory.HUMAN_CONFIRMED

    def test_correct_escalation(self):
        """System escalated, escalation was correct."""
        outcome = _make_outcome(
            executed=False, decision="UNRESOLVED",
        )
        cat = determine_reward_category(outcome)
        assert cat == RewardCategory.CORRECT_ESCALATION

    def test_unnecessary_escalation(self):
        """Default path when not executed and no clear signal."""
        outcome = _make_outcome(
            executed=False, decision="HUMAN_REVIEW",
        )
        cat = determine_reward_category(outcome)
        # No feedback → falls through to default
        assert cat == RewardCategory.UNNECESSARY_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Reward Engine Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardCalculation:
    """Test transparent reward calculation."""

    def test_correct_auto_positive_reward(self):
        """Correct auto-resolution should produce positive reward."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True, decision="AUTO"
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value > 0
        assert reward.category == RewardCategory.CORRECT_AUTO_RESOLUTION
        assert reward.is_positive()

    def test_incorrect_auto_negative_reward(self):
        """Incorrect auto-resolution should produce negative reward."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=False, executed=True, verified=True, decision="AUTO"
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value < 0
        assert reward.category == RewardCategory.INCORRECT_AUTO_RESOLUTION
        assert reward.is_negative()

    def test_high_value_error_strongly_negative(self):
        """High-value error should be strongly negative."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=50_000_000,  # ₹5,00,000
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value < -0.5
        assert reward.category == RewardCategory.HIGH_VALUE_ERROR

    def test_verification_failure_penalized(self):
        """Verification failure should be penalized."""
        engine = RewardEngine()
        outcome = _make_outcome(
            executed=True, verified=False, rolled_back=True, decision="AUTO"
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value < 0
        assert reward.category == RewardCategory.VERIFICATION_FAILURE

    def test_human_approved_bonus(self):
        """Human approval should add bonus."""
        engine = RewardEngine()
        feedback = _make_feedback(feedback_type=FeedbackType.APPROVE)
        outcome = _make_outcome(
            correct=True, executed=False, decision="HUMAN_REVIEW",
            feedback_id="FB-WF-001", feedback_type=FeedbackType.APPROVE,
        )
        reward = engine.calculate_reward(outcome, feedback)
        assert reward.reward_value > 0
        assert reward.was_human_approved is True

    def test_human_reject_penalty(self):
        """Human rejection should add penalty."""
        engine = RewardEngine()
        feedback = _make_feedback(
            feedback_type=FeedbackType.REJECT,
            rejection=RejectionDetail(rejection_reason="Wrong type"),
        )
        outcome = _make_outcome(
            correct=False, executed=False, decision="HUMAN_REVIEW",
            feedback_id="FB-WF-001", feedback_type=FeedbackType.REJECT,
        )
        reward = engine.calculate_reward(outcome, feedback)
        assert reward.reward_value < 0
        assert reward.was_human_rejected is True

    def test_human_correct_penalty(self):
        """Human correction should add penalty."""
        engine = RewardEngine()
        feedback = _make_feedback(
            feedback_type=FeedbackType.CORRECT,
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_ADJUSTMENT",
                correction_reason="Wrong type",
            ),
        )
        outcome = _make_outcome(
            correct=False, executed=False, decision="HUMAN_REVIEW",
            feedback_id="FB-WF-001", feedback_type=FeedbackType.CORRECT,
        )
        reward = engine.calculate_reward(outcome, feedback)
        assert reward.reward_value < 0
        assert reward.was_human_corrected is True

    def test_high_confidence_correct_bonus(self):
        """High confidence on correct outcome → extra bonus."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", confidence=0.95,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value > 0
        assert reward.breakdown.confidence_component.value > 0

    def test_high_confidence_incorrect_penalty(self):
        """High confidence on incorrect outcome → overconfidence penalty."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", confidence=0.95,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value < 0
        assert reward.breakdown.confidence_component.value < 0

    def test_discrepancy_eliminated_bonus(self):
        """Discrepancy eliminated → bonus."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", discrepancy_eliminated=True,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.discrepancy_component.value > 0

    def test_discrepancy_remainder_penalty(self):
        """Discrepancy remains → penalty."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", discrepancy_eliminated=False,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.discrepancy_component.value < 0

    def test_unintended_changes_penalty(self):
        """Unintended changes → penalty per change."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", unintended=2,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.unintended_changes_component.value < 0

    def test_no_feedback_neutral(self):
        """No feedback → neutral human component."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True, decision="AUTO"
        )
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.human_feedback_component.value == 0.0

    def test_reward_clamped_to_minus_1(self):
        """Reward never goes below -1.0."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=False, executed=True, verified=False,
            rolled_back=True, decision="AUTO",
            adjustment=50_000_000, confidence=0.99, unintended=5,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value >= -1.0

    def test_reward_clamped_to_plus_1(self):
        """Reward never exceeds 1.0."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True, decision="AUTO",
            confidence=1.0, discrepancy_eliminated=True,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Financial Risk Weighting Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialRiskWeighting:
    """Test that financial risk amplifies penalties."""

    def test_high_risk_amplifies_incorrect_penalty(self):
        """Incorrect auto + high financial risk → extra penalty."""
        engine = RewardEngine()
        # High-value incorrect
        outcome_high = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=50_000_00,
        )
        # Low-value incorrect
        outcome_low = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=5000,
        )
        reward_high = engine.calculate_reward(outcome_high)
        reward_low = engine.calculate_reward(outcome_low)
        # High-value should be MORE negative
        assert reward_high.reward_value < reward_low.reward_value

    def test_negligible_risk_no_extra_penalty(self):
        """Negligible risk → no financial penalty."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True,
            decision="AUTO", adjustment=5000,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.financial_risk_level == FinancialRiskLevel.NEGLIGIBLE
        assert reward.breakdown.financial_risk_component.value == 0.0

    def test_critical_risk_penalty(self):
        """Critical risk → significant penalty."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=50_000_000,  # ₹5,00,000
        )
        reward = engine.calculate_reward(outcome)
        assert reward.financial_risk_level == FinancialRiskLevel.CRITICAL
        assert reward.breakdown.financial_risk_component.value < -0.3


# ─────────────────────────────────────────────────────────────────────────────
# Safety Rule Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyRules:
    """Verify the key safety rules from the spec."""

    def test_incorrect_auto_worse_than_unnecessary_escalation(self):
        """INCORRECT_AUTO_RESOLUTION must be worse than UNNECESSARY_ESCALATION."""
        engine = RewardEngine()
        # Incorrect auto
        outcome_auto = _make_outcome(
            correct=False, executed=True, verified=True, decision="AUTO",
        )
        # Unnecessary escalation (not executed, no feedback)
        outcome_esc = _make_outcome(
            executed=False, decision="HUMAN_REVIEW",
        )
        reward_auto = engine.calculate_reward(outcome_auto)
        reward_esc = engine.calculate_reward(outcome_esc)
        assert reward_auto.reward_value < reward_esc.reward_value

    def test_verification_failure_strongly_penalized(self):
        """Verification failure must be significantly negative."""
        engine = RewardEngine()
        outcome = _make_outcome(
            executed=True, verified=False, rolled_back=True, decision="AUTO"
        )
        reward = engine.calculate_reward(outcome)
        assert reward.reward_value < -0.5

    def test_high_value_error_worse_than普通的_incorrect(self):
        """HIGH_VALUE_ERROR must be worse than INCORRECT_AUTO_RESOLUTION."""
        engine = RewardEngine()
        # Normal incorrect
        outcome_normal = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=5000,
        )
        # High-value incorrect
        outcome_high = _make_outcome(
            correct=False, executed=True, verified=True,
            decision="AUTO", adjustment=50_000_000,
        )
        reward_normal = engine.calculate_reward(outcome_normal)
        reward_high = engine.calculate_reward(outcome_high)
        assert reward_high.reward_value < reward_normal.reward_value


# ─────────────────────────────────────────────────────────────────────────────
# Reward Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardExplanation:
    """Test that rewards are fully explainable."""

    def test_breakdown_has_all_components(self):
        """Every reward has a complete breakdown."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert len(reward.breakdown.all_components()) == 7

    def test_each_component_has_reason(self):
        """Every component has a reason."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        for comp in reward.breakdown.all_components():
            assert comp.reason
            assert comp.component_name

    def test_breakdown_total_matches_reward(self):
        """Breakdown total matches final reward (clamped)."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.total() == reward.reward_value

    def test_reward_has_reason(self):
        """Reward has a human-readable reason."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert reward.reward_reason
        assert len(reward.reward_reason) > 10

    def test_reward_summary(self):
        """Reward summary is readable."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        summary = reward.summary()
        assert "Reward:" in summary
        assert "Category:" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReproducibility:
    """Test that same inputs produce same rewards."""

    def test_same_inputs_same_reward(self):
        """Same outcome + same feedback → same reward."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward1 = engine.calculate_reward(outcome)
        reward2 = engine.calculate_reward(outcome)
        assert reward1.reward_value == reward2.reward_value
        assert reward1.category == reward2.category

    def test_same_config_same_reward(self):
        """Same config → same reward."""
        config = RewardConfig(policy_version="2.0.0")
        engine1 = RewardEngine(config=config)
        engine2 = RewardEngine(config=config)
        outcome = _make_outcome(correct=False, executed=True, verified=True)
        r1 = engine1.calculate_reward(outcome)
        r2 = engine2.calculate_reward(outcome)
        assert r1.reward_value == r2.reward_value

    def test_different_config_different_reward(self):
        """Different config → potentially different reward."""
        config_a = RewardConfig(
            weights=RewardWeights(
                base_rewards={"CORRECT_AUTO_RESOLUTION": 0.9}
            )
        )
        config_b = RewardConfig(
            weights=RewardWeights(
                base_rewards={"CORRECT_AUTO_RESOLUTION": 0.5}
            )
        )
        engine_a = RewardEngine(config=config_a)
        engine_b = RewardEngine(config=config_b)
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        r_a = engine_a.calculate_reward(outcome)
        r_b = engine_b.calculate_reward(outcome)
        # Different base → different total
        assert r_a.reward_value != r_b.reward_value


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardPersistence:
    """Test reward storage and retrieval."""

    def test_store_and_retrieve(self):
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert engine.get_reward(reward.reward_id) is not None

    def test_get_by_workflow(self):
        engine = RewardEngine()
        outcome = _make_outcome(workflow_id="WF-001", correct=True, executed=True)
        engine.calculate_reward(outcome)
        rewards = engine.get_rewards_for_workflow("WF-001")
        assert len(rewards) == 1

    def test_get_by_exception(self):
        engine = RewardEngine()
        outcome = _make_outcome(exception_id="EXC-001", correct=True, executed=True)
        engine.calculate_reward(outcome)
        rewards = engine.get_rewards_for_exception("EXC-001")
        assert len(rewards) == 1

    def test_category_counts(self):
        engine = RewardEngine()
        engine.calculate_reward(_make_outcome(correct=True, executed=True, verified=True))
        engine.calculate_reward(_make_outcome(correct=False, executed=True, verified=True))
        engine.calculate_reward(_make_outcome(correct=True, executed=True, verified=True))
        counts = engine.category_counts()
        assert counts["CORRECT_AUTO_RESOLUTION"] == 2
        assert counts["INCORRECT_AUTO_RESOLUTION"] == 1

    def test_average_reward(self):
        engine = RewardEngine()
        engine.calculate_reward(_make_outcome(correct=True, executed=True, verified=True))
        engine.calculate_reward(_make_outcome(correct=False, executed=True, verified=True))
        avg = engine.average_reward()
        assert -1.0 <= avg <= 1.0

    def test_average_empty(self):
        engine = RewardEngine()
        assert engine.average_reward() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_adjustment(self):
        """Zero financial adjustment — still calculates reward."""
        engine = RewardEngine()
        outcome = _make_outcome(
            correct=True, executed=True, verified=True, adjustment=0,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.financial_risk_level == FinancialRiskLevel.NEGLIGIBLE

    def test_no_confidence(self):
        """No confidence → neutral confidence component."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        outcome.confidence = None
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.confidence_component.value == 0.0

    def test_no_feedback(self):
        """No feedback → neutral human component."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown.human_feedback_component.value == 0.0

    def test_reward_record_metadata(self):
        """Reward record has all metadata."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=True, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert reward.reward_id.startswith("REW-")
        assert reward.workflow_id == "WF-001"
        assert reward.exception_id == "EXC-001"
        assert reward.policy_version == "1.0.0"
        assert reward.model_version == "xgb-v1.0"
        assert reward.calculated_at is not None

    def test_magnitude(self):
        """Magnitude returns absolute value."""
        engine = RewardEngine()
        outcome = _make_outcome(correct=False, executed=True, verified=True)
        reward = engine.calculate_reward(outcome)
        assert reward.magnitude() == abs(reward.reward_value)
