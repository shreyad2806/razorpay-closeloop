"""
Comprehensive reward calculation tests for Razorpay CloseLoop Phase 9B.

Covers all 10 scenarios from the task specification:
1. correct AUTO resolution
2. incorrect AUTO resolution
3. human confirmation
4. human rejection
5. human correction
6. unnecessary escalation
7. correct escalation
8. high-value incorrect decision
9. verification failure
10. unresolved case

Additional focus:
- Reward value is a learning signal, NOT financial authorization
- Determinism guarantee
- Boundary cases for financial risk classification
- Missing outcome information handling
- Safety: reward cannot influence financial safety decisions
"""

import pytest

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    PredictionRecord,
    RejectionDetail,
    EscalationDetail,
)
from app.schemas.reward_engine import (
    FinancialRiskLevel,
    RewardBreakdown,
    RewardCategory,
    RewardConfig,
    RewardRecord,
    RewardWeights,
)
from app.services.reward_engine import (
    RewardEngine,
    classify_financial_risk,
    determine_reward_category,
)


# ============================================================================
# HELPERS
# ============================================================================

def _outcome(
    was_executed=True,
    verification_passed=True,
    was_rolled_back=False,
    resolution_correct=True,
    adjustment_paise=5000,
    discrepancy_eliminated=True,
    difference_after=0,
    unintended_changes=0,
    decision="AUTO",
    confidence=0.85,
    requested_adjustment_paise=5000,
    difference_before=5000,
    exception_id="EXC-001",
    case_id="CASE-001",
    human_feedback_type=None,
):
    """Build an OutcomeRecord fixture."""
    return OutcomeRecord(
        outcome_id="OUT-001",
        workflow_id="WF-001",
        exception_id=exception_id,
        case_id=case_id,
        prediction=PredictionRecord(
            resolution_type="FEE_ADJUSTMENT",
            model_version="v1.0",
        ),
        actual_outcome=ActualOutcomeRecord(
            actual_resolution="FEE_ADJUSTMENT",
            resolution_correct=resolution_correct,
            was_executed=was_executed,
            was_verified=verification_passed,
            was_rolled_back=was_rolled_back,
        ),
        human_feedback_type=human_feedback_type,
        verification_passed=verification_passed,
        financial_impact=FinancialImpact(
            requested_adjustment_paise=requested_adjustment_paise,
            actual_adjustment_paise=adjustment_paise,
            difference_before_paise=difference_before,
            difference_after_paise=difference_after,
            discrepancy_eliminated=discrepancy_eliminated,
            unintended_changes=unintended_changes,
        ),
        lineage={"exception_id": exception_id, "evidence_ids": []},
        decision=decision,
        confidence=confidence,
        ground_truth_exception_type="FEE_DIFFERENCE",
        ground_truth_resolution="FEE_ADJUSTMENT",
    )


def _feedback(
    feedback_type=FeedbackType.APPROVE,
    exception_id="EXC-001",
    case_id="CASE-001",
):
    """Build a FeedbackRecord fixture."""
    base = dict(
        feedback_id="FB-001",
        workflow_id="WF-001",
        exception_id=exception_id,
        case_id=case_id,
        feedback_type=feedback_type,
        reviewer="reviewer-01",
        system_prediction="FEE_ADJUSTMENT",
        financial_adjustment_paise=5000,
    )
    if feedback_type == FeedbackType.CORRECT:
        base["correction"] = CorrectionDetail(
            original_resolution="FEE_ADJUSTMENT",
            corrected_resolution="REFUND_ADJUSTMENT",
            correction_reason="Wrong adjustment type",
        )
    elif feedback_type == FeedbackType.REJECT:
        base["rejection"] = RejectionDetail(
            rejection_reason="Insufficient evidence",
        )
    elif feedback_type == FeedbackType.ESCALATE:
        base["escalation"] = EscalationDetail(
            escalation_reason="Complex case needs senior review",
        )
    return FeedbackRecord(**base)


# ============================================================================
# 1. CORRECT AUTO RESOLUTION
# ============================================================================

class TestCorrectAutoResolution:
    """Tests for correct auto-resolution reward."""

    def test_reward_positive(self):
        """Correct AUTO → positive reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, discrepancy_eliminated=True,
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_value > 0
        assert record.category == RewardCategory.CORRECT_AUTO_RESOLUTION

    def test_verification_bonus(self):
        """Correct AUTO with verification → additional bonus."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, discrepancy_eliminated=True,
        )
        record = engine.calculate_reward(outcome)
        assert record.breakdown.verification_component.value > 0

    def test_discrepancy_eliminated_bonus(self):
        """Correct AUTO with discrepancy eliminated → bonus."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, discrepancy_eliminated=True,
            difference_after=0,
        )
        record = engine.calculate_reward(outcome)
        assert record.breakdown.discrepancy_component.value > 0

    def test_auto_flag_set(self):
        """was_auto_resolved should be True."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.was_auto_resolved is True


# ============================================================================
# 2. INCORRECT AUTO RESOLUTION
# ============================================================================

class TestIncorrectAutoResolution:
    """Tests for incorrect auto-resolution reward."""

    def test_reward_negative(self):
        """Incorrect AUTO → negative reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_value < 0
        assert record.category == RewardCategory.INCORRECT_AUTO_RESOLUTION

    def test_overconfidence_penalty(self):
        """High confidence + incorrect → extra penalty (overconfidence)."""
        engine = RewardEngine()
        outcome_high = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=5000, confidence=0.95,
        )
        outcome_low = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=5000, confidence=0.50,
        )
        record_high = engine.calculate_reward(outcome_high)
        record_low = engine.calculate_reward(outcome_low)
        # Higher confidence when wrong → more negative
        assert record_high.reward_value < record_low.reward_value

    def test_financial_risk_penalty(self):
        """Incorrect AUTO + high financial adjustment → financial risk penalty."""
        engine = RewardEngine()
        # 150,000 paise = ₹1,500 → MEDIUM (10K-100K range)
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=150_000,
        )
        record = engine.calculate_reward(outcome)
        assert record.financial_risk_level == FinancialRiskLevel.MEDIUM
        assert record.breakdown.financial_risk_component.value < 0


# ============================================================================
# 3. HUMAN CONFIRMATION
# ============================================================================

class TestHumanConfirmation:
    """Tests for human-confirmed reward."""

    def test_reward_positive(self):
        """Human approved (not auto-resolved) → HUMAN_CONFIRMED."""
        engine = RewardEngine()
        # HUMAN_CONFIRMED requires: feedback=APPROVE + NOT auto-resolved
        # Rule 4 (CORRECT_AUTO_RESOLUTION) fires first if was_executed=True
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.APPROVE,
            adjustment_paise=0, requested_adjustment_paise=0,
            discrepancy_eliminated=False, difference_before=0, difference_after=0,
        )
        feedback = _feedback(feedback_type=FeedbackType.APPROVE)
        record = engine.calculate_reward(outcome, feedback)
        assert record.reward_value > 0
        assert record.category == RewardCategory.HUMAN_CONFIRMED
        assert record.was_human_approved is True

    def test_approve_bonus(self):
        """Approve feedback → human_feedback_component positive."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.APPROVE,
            adjustment_paise=0, requested_adjustment_paise=0,
            discrepancy_eliminated=False, difference_before=0, difference_after=0,
        )
        feedback = _feedback(feedback_type=FeedbackType.APPROVE)
        record = engine.calculate_reward(outcome, feedback)
        assert record.breakdown.human_feedback_component.value > 0


# ============================================================================
# 4. HUMAN REJECTION
# ============================================================================

class TestHumanRejection:
    """Tests for human-rejected reward."""

    def test_reward_negative(self):
        """Human rejected → negative reward (system was wrong)."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.REJECT,
        )
        feedback = _feedback(feedback_type=FeedbackType.REJECT)
        record = engine.calculate_reward(outcome, feedback)
        assert record.reward_value < 0
        assert record.category == RewardCategory.INCORRECT_AUTO_RESOLUTION
        assert record.was_human_rejected is True

    def test_reject_penalty(self):
        """Reject feedback → human_feedback_component negative."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.REJECT,
        )
        feedback = _feedback(feedback_type=FeedbackType.REJECT)
        record = engine.calculate_reward(outcome, feedback)
        assert record.breakdown.human_feedback_component.value < 0


# ============================================================================
# 5. HUMAN CORRECTION
# ============================================================================

class TestHumanCorrection:
    """Tests for human-corrected reward."""

    def test_reward_negative(self):
        """Human corrected → negative reward (system was wrong)."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.CORRECT,
        )
        feedback = _feedback(feedback_type=FeedbackType.CORRECT)
        record = engine.calculate_reward(outcome, feedback)
        assert record.reward_value < 0
        assert record.category == RewardCategory.INCORRECT_AUTO_RESOLUTION
        assert record.was_human_corrected is True

    def test_correct_penalty(self):
        """Correct feedback → human_feedback_component negative."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.CORRECT,
        )
        feedback = _feedback(feedback_type=FeedbackType.CORRECT)
        record = engine.calculate_reward(outcome, feedback)
        assert record.breakdown.human_feedback_component.value < 0


# ============================================================================
# 6. UNNECESSARY ESCALATION
# ============================================================================

class TestUnnecessaryEscalation:
    """Tests for unnecessary escalation reward."""

    def test_reward_negative(self):
        """Unnecessary escalation → negative reward."""
        engine = RewardEngine()
        # UNNECESSARY_ESCALATION: escalated but prediction was wrong
        # (rule 5: UNRESOLVED + resolution_correct=False → UNNECESSARY_ESCALATION)
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=False, decision="UNRESOLVED",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_value < 0
        assert record.category == RewardCategory.UNNECESSARY_ESCALATION

    def test_escalation_flag(self):
        """was_escalated flag should be True."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="UNRESOLVED",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.was_escalated is True


# ============================================================================
# 7. CORRECT ESCALATION
# ============================================================================

class TestCorrectEscalation:
    """Tests for correct escalation reward."""

    def test_reward_positive(self):
        """Correct escalation → positive reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="UNRESOLVED",
            adjustment_paise=5000,
            human_feedback_type=FeedbackType.ESCALATE,
        )
        feedback = _feedback(feedback_type=FeedbackType.ESCALATE)
        record = engine.calculate_reward(outcome, feedback)
        assert record.reward_value > 0
        assert record.category == RewardCategory.CORRECT_ESCALATION

    def test_escalate_feedback_neutral(self):
        """Escalate feedback → human_feedback_component is neutral (0)."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="UNRESOLVED",
            adjustment_paise=5000,
            human_feedback_type=FeedbackType.ESCALATE,
        )
        feedback = _feedback(feedback_type=FeedbackType.ESCALATE)
        record = engine.calculate_reward(outcome, feedback)
        assert record.breakdown.human_feedback_component.value == 0.0


# ============================================================================
# 8. HIGH-VALUE INCORRECT DECISION
# ============================================================================

class TestHighValueIncorrectDecision:
    """Tests for high-value incorrect decision reward."""

    def test_reward_strongly_negative(self):
        """High-value incorrect AUTO → strongly negative reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=100_000_000,  # ₹10,00,000 > threshold
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_value < -0.8
        assert record.category == RewardCategory.HIGH_VALUE_ERROR

    def test_high_value_worse_than_regular_incorrect(self):
        """HIGH_VALUE_ERROR → more negative than INCORRECT_AUTO."""
        engine = RewardEngine()
        hv_outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=100_000_000,
        )
        inc_outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=5000,
        )
        hv_record = engine.calculate_reward(hv_outcome)
        inc_record = engine.calculate_reward(inc_outcome)
        assert hv_record.reward_value < inc_record.reward_value

    def test_amplified_financial_risk_penalty(self):
        """HIGH_VALUE_ERROR + HIGH risk → amplified penalty (1.5x)."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=100_000_000,  # CRITICAL risk
        )
        record = engine.calculate_reward(outcome)
        # Amplified financial risk penalty
        assert record.breakdown.financial_risk_component.value < -0.3


# ============================================================================
# 9. VERIFICATION FAILURE
# ============================================================================

class TestVerificationFailure:
    """Tests for verification failure reward."""

    def test_reward_strongly_negative(self):
        """Verification failure + rollback → strongly negative."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=False,
            was_rolled_back=True, resolution_correct=None,
            decision="AUTO", adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_value < -0.5
        assert record.category == RewardCategory.VERIFICATION_FAILURE

    def test_verification_penalty(self):
        """Verification failure → negative verification component."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=False,
            was_rolled_back=True, resolution_correct=None,
            decision="AUTO", adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.breakdown.verification_component.value < 0


# ============================================================================
# 10. UNRESOLVED CASE
# ============================================================================

class TestUnresolvedCase:
    """Tests for unresolved case reward."""

    def test_unresolved_with_correct_escalation(self):
        """UNRESOLVED + correct escalation → positive."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=True, decision="UNRESOLVED",
            adjustment_paise=5000,
        )
        feedback = _feedback(feedback_type=FeedbackType.ESCALATE)
        record = engine.calculate_reward(outcome, feedback)
        assert record.category == RewardCategory.CORRECT_ESCALATION

    def test_unresolved_without_feedback(self):
        """UNRESOLVED without feedback, wrong prediction → unnecessary escalation."""
        engine = RewardEngine()
        # With resolution_correct=False, UNRESOLVED → UNNECESSARY_ESCALATION
        outcome = _outcome(
            was_executed=False, verification_passed=False,
            resolution_correct=False, decision="UNRESOLVED",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.category == RewardCategory.UNNECESSARY_ESCALATION


# ============================================================================
# REWARD VALUE IS LEARNING SIGNAL ONLY
# ============================================================================

class TestRewardIsLearningSignalOnly:
    """
    CRITICAL SAFETY TEST: Reward is an evaluation signal.
    It must NOT directly authorize financial execution.
    """

    def test_reward_does_not_execute_financial_actions(self):
        """RewardRecord has no execute/apply/commit methods."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        # Verify reward is a data record, not an action executor
        assert hasattr(record, 'reward_value')
        assert hasattr(record, 'category')
        assert not hasattr(record, 'execute')
        assert not hasattr(record, 'apply')
        assert not hasattr(record, 'commit')
        assert not hasattr(record, 'authorize')

    def test_positive_reward_does_not_bypass_guardrails(self):
        """A high positive reward cannot change a guardrail decision."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        # Reward record has no reference to guardrail or Phase 6
        assert not hasattr(record, 'guardrail_result')
        assert not hasattr(record, 'override_guardrail')
        assert not hasattr(record, 'bypass_safety')

    def test_reward_value_clamped(self):
        """Reward value is always in [-1.0, 1.0]."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert -1.0 <= record.reward_value <= 1.0

    def test_reward_record_is_recommendation_only(self):
        """Reward record must not contain financial authorization fields."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        # No financial authorization fields
        assert not hasattr(record, 'authorization_token')
        assert not hasattr(record, 'financial_approval')


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Reward calculation must be deterministic."""

    def test_same_inputs_same_reward(self):
        """Same outcome + same feedback → same reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        feedback = _feedback(feedback_type=FeedbackType.APPROVE)
        r1 = engine.calculate_reward(outcome, feedback)
        r2 = engine.calculate_reward(outcome, feedback)
        assert r1.reward_value == r2.reward_value
        assert r1.category == r2.category

    def test_same_inputs_same_breakdown(self):
        """Same inputs → identical breakdown components."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=5000,
        )
        r1 = engine.calculate_reward(outcome)
        r2 = engine.calculate_reward(outcome)
        assert r1.breakdown.base_reward.value == r2.breakdown.base_reward.value
        assert r1.breakdown.verification_component.value == r2.breakdown.verification_component.value
        assert r1.breakdown.financial_risk_component.value == r2.breakdown.financial_risk_component.value

    def test_different_config_different_reward(self):
        """Different config → different reward."""
        config_a = RewardConfig(weights=RewardWeights(
            base_rewards={"CORRECT_AUTO_RESOLUTION": 0.8},
        ))
        config_b = RewardConfig(weights=RewardWeights(
            base_rewards={"CORRECT_AUTO_RESOLUTION": 0.3},
        ))
        engine_a = RewardEngine(config=config_a)
        engine_b = RewardEngine(config=config_b)
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        r_a = engine_a.calculate_reward(outcome)
        r_b = engine_b.calculate_reward(outcome)
        assert r_a.reward_value != r_b.reward_value


# ============================================================================
# BOUNDARY CASES
# ============================================================================

class TestBoundaryCases:
    """Boundary cases for reward calculation."""

    def test_zero_adjustment(self):
        """Zero financial adjustment → no financial risk penalty."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=0, requested_adjustment_paise=0,
            discrepancy_eliminated=True, difference_before=0,
            difference_after=0,
        )
        record = engine.calculate_reward(outcome)
        assert record.financial_risk_level == FinancialRiskLevel.NEGLIGIBLE
        assert record.breakdown.financial_risk_component.value == 0.0

    def test_no_confidence(self):
        """No confidence → confidence component is neutral."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, confidence=None,
        )
        record = engine.calculate_reward(outcome)
        assert record.breakdown.confidence_component.value == 0.0

    def test_no_feedback(self):
        """No feedback → human_feedback_component is neutral."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome, feedback=None)
        assert record.breakdown.human_feedback_component.value == 0.0

    def test_unintended_changes_penalty(self):
        """Unintended changes → penalty per change."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, unintended_changes=3,
        )
        record = engine.calculate_reward(outcome)
        assert record.breakdown.unintended_changes_component.value < 0
        assert record.unintended_changes == 3

    def test_reward_record_metadata(self):
        """Reward record has correct metadata."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        record = engine.calculate_reward(outcome)
        assert record.reward_id.startswith("REW-")
        assert record.workflow_id == "WF-001"
        assert record.exception_id == "EXC-001"
        assert record.policy_version == "1.0.0"
        assert record.calculated_at is not None

    def test_breakdown_total_matches_reward(self):
        """Breakdown total matches the final reward value."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000,
        )
        feedback = _feedback(feedback_type=FeedbackType.APPROVE)
        record = engine.calculate_reward(outcome, feedback)
        breakdown_total = record.breakdown.total()
        assert record.reward_value == breakdown_total


# ============================================================================
# FINANCIAL RISK CLASSIFICATION BOUNDARIES
# ============================================================================

class TestFinancialRiskBoundaries:
    """Test exact boundary values for financial risk classification."""

    def test_negligible_below_10k(self):
        """< 10,000 paise → NEGLIGIBLE."""
        assert classify_financial_risk(9999) == FinancialRiskLevel.NEGLIGIBLE

    def test_low_at_10k(self):
        """10,000 paise → LOW."""
        assert classify_financial_risk(10000) == FinancialRiskLevel.LOW

    def test_low_below_100k(self):
        """99,999 paise → LOW."""
        assert classify_financial_risk(99999) == FinancialRiskLevel.LOW

    def test_medium_at_100k(self):
        """100,000 paise → MEDIUM."""
        assert classify_financial_risk(100000) == FinancialRiskLevel.MEDIUM

    def test_medium_below_1M(self):
        """999,999 paise → MEDIUM."""
        assert classify_financial_risk(999999) == FinancialRiskLevel.MEDIUM

    def test_high_at_1M(self):
        """1,000,000 paise → HIGH."""
        assert classify_financial_risk(1000000) == FinancialRiskLevel.HIGH

    def test_critical_above_threshold(self):
        """Above high_value_threshold → CRITICAL."""
        assert classify_financial_risk(100_000_001) == FinancialRiskLevel.CRITICAL

    def test_negative_amount_uses_abs(self):
        """Negative adjustment uses absolute value."""
        assert classify_financial_risk(-50000) == FinancialRiskLevel.LOW


# ============================================================================
# REWARD RANKING ORDERING
# ============================================================================

class TestRewardRanking:
    """Verify correct ordering of reward categories from best to worst."""

    def test_correct_auto_best(self):
        """CORRECT_AUTO_RESOLUTION should produce the highest reward."""
        engine = RewardEngine()
        outcome_correct = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="AUTO",
            adjustment_paise=5000, discrepancy_eliminated=True,
        )
        outcome_confirmed = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=True, decision="HUMAN_REVIEW",
            human_feedback_type=FeedbackType.APPROVE,
        )
        feedback = _feedback(feedback_type=FeedbackType.APPROVE)
        r_correct = engine.calculate_reward(outcome_correct)
        r_confirmed = engine.calculate_reward(outcome_confirmed, feedback)
        assert r_correct.reward_value >= r_confirmed.reward_value

    def test_high_value_error_worst(self):
        """HIGH_VALUE_ERROR should produce a strongly negative reward."""
        engine = RewardEngine()
        outcome = _outcome(
            was_executed=True, verification_passed=True,
            resolution_correct=False, decision="AUTO",
            adjustment_paise=100_000_000,
        )
        record = engine.calculate_reward(outcome)
        assert record.category == RewardCategory.HIGH_VALUE_ERROR
        assert record.reward_value < -0.8
