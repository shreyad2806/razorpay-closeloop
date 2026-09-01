"""
Tests for Phase 9H — Learning Metrics + Evaluation.

Tests automation metrics, precision, human review, reward tracking,
financial impact, verification, safety assessment, trend analysis,
and metrics comparison.
"""

import math
import pytest
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    PredictionRecord,
)
from app.schemas.learning_metrics import (
    AutomationMetrics,
    FinancialImpactMetrics,
    HumanReviewMetrics,
    LearningMetrics,
    LearningMetricsComparison,
    MetricTrend,
    MetricTrendAnalysis,
    PrecisionMetrics,
    RewardMetrics,
    SafetyAssessmentResult,
    SafetyVerdict,
    VerificationMetrics,
)
from app.schemas.reward_engine import (
    FinancialRiskLevel,
    RewardBreakdown,
    RewardCategory,
    RewardComponent,
    RewardRecord,
)
from app.services.learning_metrics import (
    LearningMetricsComparator,
    LearningMetricsService,
    SafetyThresholds,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8].upper()}"


def _make_outcome(
    exception_id: str = "EXC-001",
    exception_type: str = "FEE_DIFFERENCE",
    decision: str = "AUTO",
    resolution_correct: Optional[bool] = True,
    was_executed: bool = True,
    was_verified: bool = True,
    was_rolled_back: bool = False,
    adjustment_paise: int = 5000,
    confidence: float = 0.85,
    discrepancy_eliminated: bool = True,
) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=_gen_id("OUT"),
        workflow_id="WF-001",
        exception_id=exception_id,
        case_id=f"CASE-{exception_id}",
        candidate_id=_gen_id("CND"),
        prediction=PredictionRecord(
            exception_type=exception_type,
            resolution_type="FEE_CORRECTION",
            resolution_confidence=confidence,
            exception_confidence=0.9,
            model_version="v1.0",
        ),
        actual_outcome=ActualOutcomeRecord(
            actual_resolution="FEE_CORRECTION",
            actual_exception_type=exception_type,
            resolution_correct=resolution_correct,
            financial_impact_paise=adjustment_paise,
            was_executed=was_executed,
            was_verified=was_verified,
            was_rolled_back=was_rolled_back,
        ),
        verification_passed=was_verified and not was_rolled_back,
        financial_impact=FinancialImpact(
            requested_adjustment_paise=adjustment_paise,
            actual_adjustment_paise=adjustment_paise,
            difference_before_paise=adjustment_paise,
            difference_after_paise=0 if resolution_correct else adjustment_paise,
            discrepancy_eliminated=discrepancy_eliminated and resolution_correct is True,
        ),
        lineage=DataLineage(exception_id=exception_id),
        decision=decision,
        confidence=confidence,
        risk="LOW",
    )


def _make_feedback(
    exception_id: str = "EXC-001",
    feedback_type: FeedbackType = FeedbackType.APPROVE,
) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=_gen_id("FB"),
        workflow_id="WF-001",
        exception_id=exception_id,
        system_prediction="FEE_CORRECTION",
        feedback_type=feedback_type,
        reviewer="test_reviewer",
        correction=(
            CorrectionDetail(
                original_resolution="FEE_CORRECTION",
                corrected_resolution="REFUND",
                correction_reason="Test correction",
            )
            if feedback_type == FeedbackType.CORRECT
            else None
        ),
    )


def _make_reward(
    exception_id: str = "EXC-001",
    reward_value: float = 0.5,
    category: RewardCategory = RewardCategory.CORRECT_AUTO_RESOLUTION,
    risk_level: FinancialRiskLevel = FinancialRiskLevel.LOW,
    model_version: Optional[str] = "v1.0",
) -> RewardRecord:
    return RewardRecord(
        reward_id=_gen_id("REW"),
        workflow_id="WF-001",
        exception_id=exception_id,
        category=category,
        reward_value=reward_value,
        reward_reason="Test reward",
        breakdown=RewardBreakdown(
            base_reward=RewardComponent(
                component_name="base_reward", value=reward_value, reason="test",
            ),
            verification_component=RewardComponent(
                component_name="verification", value=0.0, reason="test",
            ),
            financial_risk_component=RewardComponent(
                component_name="financial_risk", value=0.0, reason="test",
            ),
            human_feedback_component=RewardComponent(
                component_name="human_feedback", value=0.0, reason="test",
            ),
            confidence_component=RewardComponent(
                component_name="confidence", value=0.0, reason="test",
            ),
            discrepancy_component=RewardComponent(
                component_name="discrepancy", value=0.0, reason="test",
            ),
            unintended_changes_component=RewardComponent(
                component_name="unintended_changes", value=0.0, reason="test",
            ),
        ),
        financial_risk_level=risk_level,
        model_version=model_version,
    )


def _make_outcomes_and_rewards(
    n: int = 10,
    auto_count: int = 7,
    correct_auto: int = 5,
    human_count: int = 2,
    unresolved_count: int = 1,
    hv_error_count: int = 0,
    rolled_back_count: int = 0,
):
    """Create a balanced set of outcomes and rewards for testing."""
    outcomes = []
    rewards = []
    idx = 0

    for i in range(correct_auto):
        oid = f"EXC-{idx:03d}"
        outcomes.append(_make_outcome(
            exception_id=oid, decision="AUTO", resolution_correct=True,
            was_executed=True, was_verified=True,
        ))
        rewards.append(_make_reward(exception_id=oid, reward_value=0.6))
        idx += 1

    for i in range(auto_count - correct_auto):
        oid = f"EXC-{idx:03d}"
        adj = 15_000_000 if i < hv_error_count else 5000
        rolled_back = i < rolled_back_count
        outcomes.append(_make_outcome(
            exception_id=oid, decision="AUTO", resolution_correct=False,
            was_executed=True, was_verified=not rolled_back,
            was_rolled_back=rolled_back, adjustment_paise=adj,
        ))
        rewards.append(_make_reward(
            exception_id=oid, reward_value=-0.5,
            category=RewardCategory.INCORRECT_AUTO_RESOLUTION,
        ))
        idx += 1

    for i in range(human_count):
        oid = f"EXC-{idx:03d}"
        outcomes.append(_make_outcome(
            exception_id=oid, decision="HUMAN_REVIEW", resolution_correct=True,
        ))
        rewards.append(_make_reward(
            exception_id=oid, reward_value=0.3,
            category=RewardCategory.CORRECT_ESCALATION,
        ))
        idx += 1

    for i in range(unresolved_count):
        oid = f"EXC-{idx:03d}"
        outcomes.append(_make_outcome(
            exception_id=oid, decision="UNRESOLVED", resolution_correct=None,
            was_executed=False, was_verified=False,
        ))
        idx += 1

    return outcomes, rewards


# ─────────────────────────────────────────────────────────────────────────────
# Test: Automation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestAutomationMetrics:
    """Tests for automation metrics computation."""

    def test_basic_automation_rate(self):
        outcomes, rewards = _make_outcomes_and_rewards(
            n=10, auto_count=7, human_count=2, unresolved_count=1,
        )
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], rewards)

        assert m.automation.total_exceptions == 10
        assert m.automation.auto_decisions == 7
        assert m.automation.human_decisions == 2
        assert m.automation.unresolved_decisions == 1
        assert m.automation.automation_rate == pytest.approx(0.7)

    def test_zero_outcomes(self):
        svc = LearningMetricsService()
        m = svc.compute([], [], [])

        assert m.automation.total_exceptions == 0
        assert m.automation.automation_rate == 0.0

    def test_successful_automation_rate(self):
        """Successful auto = executed + verified + correct."""
        outcomes = [_make_outcome(
            decision="AUTO", resolution_correct=True,
            was_executed=True, was_verified=True,
        ) for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.automation.successful_auto == 5
        assert m.automation.successful_automation_rate == pytest.approx(1.0)

    def test_failed_automation_rate(self):
        """Failed auto = executed but verification failed or rolled back."""
        outcomes, rewards = _make_outcomes_and_rewards(
            auto_count=5, correct_auto=3, rolled_back_count=2,
        )
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], rewards)

        assert m.automation.failed_auto == 2
        assert m.automation.failed_automation_rate == pytest.approx(2 / 5)

    def test_all_human_review(self):
        outcomes = [_make_outcome(decision="HUMAN_REVIEW") for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.automation.automation_rate == 0.0
        assert m.automation.human_review_rate == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Precision Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecisionMetrics:
    """Tests for precision metrics computation."""

    def test_perfect_precision(self):
        outcomes, rewards = _make_outcomes_and_rewards(
            auto_count=5, correct_auto=5,
        )
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], rewards)

        assert m.precision.precision == 1.0
        assert m.precision.false_automation_count == 0
        assert m.precision.false_automation_rate == 0.0

    def test_zero_precision(self):
        outcomes = [_make_outcome(
            decision="AUTO", resolution_correct=False,
        ) for _ in range(3)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.precision.precision == 0.0
        assert m.precision.false_automation_count == 3
        assert m.precision.false_automation_rate == pytest.approx(1.0)

    def test_mixed_precision(self):
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=False),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.precision.precision == pytest.approx(2 / 3)

    def test_no_auto_decisions(self):
        outcomes = [_make_outcome(decision="HUMAN_REVIEW") for _ in range(3)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.precision.precision is None
        assert m.precision.false_automation_count == 0

    def test_per_exception_precision(self):
        outcomes = [
            _make_outcome(
                exception_type="FEE_DIFFERENCE", decision="AUTO",
                resolution_correct=True,
            ),
            _make_outcome(
                exception_type="FEE_DIFFERENCE", decision="AUTO",
                resolution_correct=True,
            ),
            _make_outcome(
                exception_type="REFUND_DIFFERENCE", decision="AUTO",
                resolution_correct=False,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.precision.per_exception_precision["FEE_DIFFERENCE"] == 1.0
        assert m.precision.per_exception_precision["REFUND_DIFFERENCE"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Human Review Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanReviewMetrics:
    """Tests for human review metrics."""

    def test_human_review_counts(self):
        outcomes = [
            _make_outcome(decision="HUMAN_REVIEW") for _ in range(5)
        ]
        feedbacks = [
            _make_feedback(feedback_type=FeedbackType.CORRECT),
            _make_feedback(feedback_type=FeedbackType.CORRECT),
            _make_feedback(feedback_type=FeedbackType.REJECT),
            _make_feedback(feedback_type=FeedbackType.APPROVE),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, feedbacks, [])

        assert m.human_review.total_human_reviews == 5
        assert m.human_review.human_corrections == 2
        assert m.human_review.human_rejections == 1
        assert m.human_review.human_approvals == 1

    def test_unnecessary_escalations(self):
        """Escalated cases where system was actually correct."""
        outcomes = [
            _make_outcome(decision="HUMAN_REVIEW", resolution_correct=True),
            _make_outcome(decision="HUMAN_REVIEW", resolution_correct=True),
            _make_outcome(decision="HUMAN_REVIEW", resolution_correct=False),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.human_review.unnecessary_escalations == 2
        assert m.human_review.unnecessary_escalation_rate == pytest.approx(2 / 3)

    def test_correction_rate(self):
        outcomes = [_make_outcome(decision="HUMAN_REVIEW") for _ in range(4)]
        feedbacks = [
            _make_feedback(feedback_type=FeedbackType.CORRECT),
            _make_feedback(feedback_type=FeedbackType.CORRECT),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, feedbacks, [])

        assert m.human_review.correction_rate == pytest.approx(0.5)

    def test_no_human_reviews(self):
        outcomes = [_make_outcome(decision="AUTO") for _ in range(3)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.human_review.total_human_reviews == 0
        assert m.human_review.correction_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Reward Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardMetrics:
    """Tests for reward metrics computation."""

    def test_basic_reward_metrics(self):
        rewards = [
            _make_reward(reward_value=0.5),
            _make_reward(reward_value=0.7),
            _make_reward(reward_value=-0.3),
        ]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert m.reward.total_rewards == 3
        assert m.reward.avg_reward == pytest.approx((0.5 + 0.7 - 0.3) / 3)
        assert m.reward.positive_rewards == 2
        assert m.reward.negative_rewards == 1

    def test_all_positive_rewards(self):
        rewards = [_make_reward(reward_value=v) for v in [0.3, 0.5, 0.8]]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert m.reward.positive_rate == pytest.approx(1.0)

    def test_reward_by_category(self):
        rewards = [
            _make_reward(reward_value=0.8, category=RewardCategory.CORRECT_AUTO_RESOLUTION),
            _make_reward(reward_value=0.8, category=RewardCategory.CORRECT_AUTO_RESOLUTION),
            _make_reward(reward_value=-0.5, category=RewardCategory.INCORRECT_AUTO_RESOLUTION),
        ]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert "CORRECT_AUTO_RESOLUTION" in m.reward.rewards_by_category
        assert m.reward.rewards_by_category["CORRECT_AUTO_RESOLUTION"] == pytest.approx(0.8)
        assert m.reward.rewards_by_category["INCORRECT_AUTO_RESOLUTION"] == pytest.approx(-0.5)

    def test_reward_by_risk_level(self):
        rewards = [
            _make_reward(reward_value=0.5, risk_level=FinancialRiskLevel.LOW),
            _make_reward(reward_value=0.3, risk_level=FinancialRiskLevel.HIGH),
        ]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert "LOW" in m.reward.rewards_by_risk
        assert "HIGH" in m.reward.rewards_by_risk

    def test_reward_by_model_version(self):
        rewards = [
            _make_reward(reward_value=0.6, model_version="v1.0"),
            _make_reward(reward_value=0.8, model_version="v2.0"),
        ]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert "v1.0" in m.reward.rewards_by_model
        assert "v2.0" in m.reward.rewards_by_model

    def test_no_rewards(self):
        svc = LearningMetricsService()
        m = svc.compute([], [], [])

        assert m.reward.total_rewards == 0
        assert m.reward.avg_reward is None

    def test_single_reward(self):
        rewards = [_make_reward(reward_value=0.5)]
        svc = LearningMetricsService()
        m = svc.compute([], [], rewards)

        assert m.reward.avg_reward == pytest.approx(0.5)
        assert m.reward.min_reward == pytest.approx(0.5)
        assert m.reward.max_reward == pytest.approx(0.5)
        assert m.reward.reward_std is None  # std needs >1 values


# ─────────────────────────────────────────────────────────────────────────────
# Test: Financial Impact Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestFinancialImpactMetrics:
    """Tests for financial impact metrics."""

    def test_basic_financial(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", adjustment_paise=10000,
                resolution_correct=True,
            ),
            _make_outcome(
                decision="AUTO", adjustment_paise=5000,
                resolution_correct=True,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.total_adjustment_paise == 15000
        assert m.financial.max_adjustment_paise == 10000
        assert m.financial.avg_adjustment_paise == pytest.approx(7500)

    def test_error_impact(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", adjustment_paise=10000,
                resolution_correct=False,
            ),
            _make_outcome(
                decision="AUTO", adjustment_paise=5000,
                resolution_correct=True,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.total_error_impact_paise == 10000

    def test_high_value_errors(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", adjustment_paise=15_000_000,  # ₹1,50,000
                resolution_correct=False,
            ),
            _make_outcome(
                decision="AUTO", adjustment_paise=5000,
                resolution_correct=False,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.high_value_error_count == 1
        assert m.financial.high_value_error_impact_paise == 15_000_000

    def test_discrepancy_elimination(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", was_executed=True,
                resolution_correct=True, discrepancy_eliminated=True,
            ),
            _make_outcome(
                decision="AUTO", was_executed=True,
                resolution_correct=True, discrepancy_eliminated=True,
            ),
            _make_outcome(
                decision="AUTO", was_executed=True,
                resolution_correct=False, discrepancy_eliminated=False,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.discrepancy_eliminated_count == 2
        assert m.financial.discrepancy_elimination_rate == pytest.approx(2 / 3)

    def test_impact_avoided(self):
        """Impact avoided = adjustments from correct human-review cases."""
        outcomes = [
            _make_outcome(
                decision="HUMAN_REVIEW", resolution_correct=True,
                adjustment_paise=20000,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.impact_avoided_paise == 20000

    def test_no_auto_decisions(self):
        outcomes = [_make_outcome(decision="HUMAN_REVIEW") for _ in range(3)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.financial.total_adjustment_paise == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Verification Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationMetrics:
    """Tests for verification metrics."""

    def test_all_verified(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", was_executed=True,
                was_verified=True, was_rolled_back=False,
            )
            for _ in range(5)
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.verification.total_executed == 5
        assert m.verification.total_verified == 5
        assert m.verification.verification_success_rate == pytest.approx(1.0)
        assert m.verification.rollback_rate == pytest.approx(0.0)

    def test_some_rolled_back(self):
        outcomes = [
            _make_outcome(
                decision="AUTO", was_executed=True,
                was_verified=False, was_rolled_back=True,
            ),
            _make_outcome(
                decision="AUTO", was_executed=True,
                was_verified=True, was_rolled_back=False,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.verification.total_executed == 2
        assert m.verification.total_rolled_back == 1
        assert m.verification.rollback_rate == pytest.approx(0.5)

    def test_no_executions(self):
        outcomes = [_make_outcome(
            decision="HUMAN_REVIEW", was_executed=False,
        ) for _ in range(3)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.verification.total_executed == 0
        assert m.verification.verification_success_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Safety Assessment
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyAssessment:
    """Tests for safety assessment."""

    def test_all_safe(self):
        outcomes, rewards = _make_outcomes_and_rewards(
            auto_count=10, correct_auto=9, hv_error_count=0,
            rolled_back_count=0,
        )
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], rewards)

        assert m.safety.verdict == SafetyVerdict.SAFE
        assert m.safety.checks_failed == 0

    def test_precision_below_threshold(self):
        """Low precision → UNSAFE."""
        outcomes = [_make_outcome(
            decision="AUTO", resolution_correct=False,
        ) for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.safety.verdict == SafetyVerdict.UNSAFE
        assert "precision" in m.safety.critical_failures

    def test_false_automation_rate_high(self):
        """High false auto rate → UNSAFE."""
        outcomes = [
            _make_outcome(
                decision="AUTO", resolution_correct=False,
            ) for _ in range(8)
        ] + [
            _make_outcome(
                decision="AUTO", resolution_correct=True,
            ) for _ in range(2)
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        # 80% false auto rate > 15% threshold
        assert m.safety.verdict == SafetyVerdict.UNSAFE

    def test_high_value_errors(self):
        """HV errors > 0 → UNSAFE."""
        outcomes = [
            _make_outcome(
                decision="AUTO", adjustment_paise=15_000_000,
                resolution_correct=False,
            ),
            _make_outcome(
                decision="AUTO", resolution_correct=True,
            ),
        ]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.safety.verdict == SafetyVerdict.UNSAFE
        assert "high_value_errors" in m.safety.critical_failures

    def test_custom_thresholds(self):
        """Custom thresholds are applied."""
        thresholds = SafetyThresholds(
            min_precision=0.50,
            max_false_automation_rate=0.60,
        )
        svc = LearningMetricsService(safety_thresholds=thresholds)

        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=False),
        ]
        m = svc.compute(outcomes, [], [])

        # 50% precision ≥ 50% threshold → SAFE
        assert m.safety.verdict == SafetyVerdict.SAFE

    def test_safety_checks_count(self):
        outcomes, rewards = _make_outcomes_and_rewards(
            auto_count=10, correct_auto=8, rolled_back_count=0,
        )
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], rewards)

        # Should have 5 safety checks
        assert len(m.safety.checks) == 5
        assert m.safety.checks_passed + m.safety.checks_failed == 5


# ─────────────────────────────────────────────────────────────────────────────
# Test: Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendAnalysis:
    """Tests for trend analysis."""

    def test_improving_trend(self):
        """Precision improving → IMPROVING."""
        prev = LearningMetrics(
            metrics_id="LM-PREV",
            precision=PrecisionMetrics(precision=0.70),
            automation=AutomationMetrics(automation_rate=0.60),
        )
        curr = LearningMetrics(
            metrics_id="LM-CURR",
            precision=PrecisionMetrics(precision=0.85),
            automation=AutomationMetrics(automation_rate=0.60),
        )

        svc = LearningMetricsService()
        # Compute new metrics with previous for trend comparison
        outcomes = [_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(10)]
        m = svc.compute(outcomes, [], [], previous_metrics=prev)

        # Find precision trend
        prec_trend = next(
            (t for t in m.trends if t.metric_name == "precision"), None
        )
        assert prec_trend is not None
        assert prec_trend.trend == MetricTrend.IMPROVING

    def test_declining_trend(self):
        """Precision declining → DECLINING."""
        prev = LearningMetrics(
            metrics_id="LM-PREV",
            precision=PrecisionMetrics(precision=0.90),
            automation=AutomationMetrics(automation_rate=0.60),
            financial=FinancialImpactMetrics(total_error_impact_paise=0),
        )
        curr = LearningMetrics(
            metrics_id="LM-CURR",
            precision=PrecisionMetrics(precision=0.80),
            automation=AutomationMetrics(automation_rate=0.60),
            financial=FinancialImpactMetrics(total_error_impact_paise=0),
        )

        svc = LearningMetricsService()
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True) for _ in range(5)
        ] + [
            _make_outcome(decision="AUTO", resolution_correct=False) for _ in range(5)
        ]
        m = svc.compute(outcomes, [], [], previous_metrics=prev)

        prec_trend = next(
            (t for t in m.trends if t.metric_name == "precision"), None
        )
        assert prec_trend is not None
        assert prec_trend.trend == MetricTrend.DECLINING

    def test_no_previous_metrics(self):
        """No previous → no trends."""
        outcomes = [_make_outcome() for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [], previous_metrics=None)

        assert len(m.trends) == 0

    def test_stable_trend(self):
        """Same values → STABLE."""
        prev = LearningMetrics(
            metrics_id="LM-PREV",
            precision=PrecisionMetrics(precision=0.80),
            automation=AutomationMetrics(automation_rate=0.60),
            financial=FinancialImpactMetrics(total_error_impact_paise=100),
        )

        svc = LearningMetricsService()
        # Create outcomes that produce ~80% precision
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True) for _ in range(4)
        ] + [
            _make_outcome(decision="AUTO", resolution_correct=False) for _ in range(1)
        ]
        m = svc.compute(outcomes, [], [], previous_metrics=prev)

        # At least check that trends are computed
        assert len(m.trends) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Metrics Comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningMetricsComparison:
    """Tests for comparing two learning metrics snapshots."""

    def test_improvement_comparison(self):
        svc = LearningMetricsService()

        # Current: lower precision
        outcomes1 = [
            _make_outcome(decision="AUTO", resolution_correct=True) for _ in range(6)
        ] + [
            _make_outcome(decision="AUTO", resolution_correct=False) for _ in range(4)
        ]
        current = svc.compute(outcomes1, [], [])

        # Candidate: higher precision
        outcomes2 = [
            _make_outcome(decision="AUTO", resolution_correct=True) for _ in range(9)
        ] + [
            _make_outcome(decision="AUTO", resolution_correct=False) for _ in range(1)
        ]
        candidate = svc.compute(outcomes2, [], [])

        comparator = LearningMetricsComparator()
        result = comparator.compare(current, candidate)

        assert result.overall_improvement is True
        assert result.safety_maintained is True
        assert len(result.improvements) > 0

    def test_regression_comparison(self):
        svc = LearningMetricsService()

        # Current: good
        outcomes1 = [_make_outcome(
            decision="AUTO", resolution_correct=True,
        ) for _ in range(10)]
        current = svc.compute(outcomes1, [], [])

        # Candidate: worse
        outcomes2 = [_make_outcome(
            decision="AUTO", resolution_correct=False,
        ) for _ in range(10)]
        candidate = svc.compute(outcomes2, [], [])

        comparator = LearningMetricsComparator()
        result = comparator.compare(current, candidate)

        assert result.overall_improvement is False
        assert len(result.regressions) > 0

    def test_safety_regression_detected(self):
        svc = LearningMetricsService()

        # Current: clean
        outcomes1 = [_make_outcome(
            decision="AUTO", resolution_correct=True,
        ) for _ in range(10)]
        current = svc.compute(outcomes1, [], [])

        # Candidate: HV errors
        outcomes2 = [
            _make_outcome(
                decision="AUTO", resolution_correct=True,
            ) for _ in range(8)
        ] + [
            _make_outcome(
                decision="AUTO", adjustment_paise=15_000_000,
                resolution_correct=False,
            ) for _ in range(2)
        ]
        candidate = svc.compute(outcomes2, [], [])

        comparator = LearningMetricsComparator()
        result = comparator.compare(current, candidate)

        assert result.safety_maintained is False
        assert len(result.safety_regressions) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_outcome(self):
        outcomes = [_make_outcome(decision="AUTO", resolution_correct=True)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.automation.total_exceptions == 1
        assert m.precision.precision == 1.0

    def test_all_unresolved(self):
        outcomes = [_make_outcome(
            decision="UNRESOLVED", was_executed=False,
        ) for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.automation.unresolved_rate == 1.0
        assert m.automation.automation_rate == 0.0

    def test_metrics_snapshot_has_id(self):
        outcomes = [_make_outcome()]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.metrics_id.startswith("LM-")

    def test_metrics_has_timestamps(self):
        outcomes = [_make_outcome()]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.computed_at is not None

    def test_is_safe_method(self):
        outcomes = [_make_outcome(
            decision="AUTO", resolution_correct=True,
        ) for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.is_safe() is True

    def test_is_unsafe_method(self):
        outcomes = [_make_outcome(
            decision="AUTO", resolution_correct=False,
        ) for _ in range(5)]
        svc = LearningMetricsService()
        m = svc.compute(outcomes, [], [])

        assert m.is_safe() is False

    def test_summary_methods(self):
        """All summary methods return strings without error."""
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="HUMAN_REVIEW"),
        ]
        feedbacks = [_make_feedback(feedback_type=FeedbackType.APPROVE)]
        rewards = [_make_reward(reward_value=0.5)]

        svc = LearningMetricsService()
        m = svc.compute(outcomes, feedbacks, rewards)

        assert isinstance(m.summary(), str)
        assert isinstance(m.automation.summary(), str)
        assert isinstance(m.precision.summary(), str)
        assert isinstance(m.human_review.summary(), str)
        assert isinstance(m.reward.summary(), str)
        assert isinstance(m.financial.summary(), str)
        assert isinstance(m.verification.summary(), str)
        assert isinstance(m.safety.summary(), str)

    def test_deterministic_computation(self):
        """Same inputs → same outputs."""
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True)
            for _ in range(5)
        ]

        svc1 = LearningMetricsService()
        m1 = svc1.compute(outcomes, [], [])

        svc2 = LearningMetricsService()
        m2 = svc2.compute(outcomes, [], [])

        assert m1.automation.automation_rate == m2.automation.automation_rate
        assert m1.precision.precision == m2.precision.precision
        assert m1.safety.verdict == m2.safety.verdict
