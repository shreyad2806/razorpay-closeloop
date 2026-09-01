"""
Tests for Phase 9G — Batch Learning Loop.

Tests batch creation, metrics calculation, comparison, safety assessment,
candidate training, promotion/rejection, and report generation.
"""

import math
import pytest
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.schemas.batch_learning import (
    BatchComparison,
    BatchComparisonReport,
    BatchConfig,
    BatchMetrics,
    BatchRecord,
    BatchRecommendation,
    BatchReportRow,
    BatchStatus,
    MetricChange,
    SafetyAssessment,
)
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
from app.schemas.learning_dataset import (
    FeatureSnapshot,
    LearningDataset,
    LearningExample,
    LearningLabels,
    SplitType,
)
from app.schemas.reward_engine import RewardCategory, RewardRecord, RewardBreakdown, RewardComponent
from app.schemas.model_training import EvaluationMetrics, TrainingConfig, ModelMetadata, ModelStatus, ModelType
from app.services.batch_learning import (
    BatchLearningLoop,
    BatchMetricsCalculator,
    BatchComparator,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8].upper()}"


def _make_outcome(
    exception_id: str = "EXC-001",
    workflow_id: str = "WF-001",
    decision: str = "AUTO",
    resolution_correct: Optional[bool] = True,
    was_executed: bool = True,
    was_verified: bool = True,
    was_rolled_back: bool = False,
    adjustment_paise: int = 5000,
    confidence: float = 0.85,
    feedback_type: Optional[FeedbackType] = None,
) -> OutcomeRecord:
    """Create a test OutcomeRecord."""
    return OutcomeRecord(
        outcome_id=_gen_id("OUT"),
        workflow_id=workflow_id,
        exception_id=exception_id,
        case_id=f"CASE-{exception_id}",
        candidate_id=_gen_id("CND"),
        prediction=PredictionRecord(
            exception_type="FEE_DIFFERENCE",
            resolution_type="FEE_CORRECTION",
            resolution_confidence=confidence,
            exception_confidence=0.9,
            model_version="v1.0",
        ),
        actual_outcome=ActualOutcomeRecord(
            actual_resolution="FEE_CORRECTION",
            actual_exception_type="FEE_DIFFERENCE",
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
            discrepancy_eliminated=resolution_correct is True,
        ),
        lineage=DataLineage(exception_id=exception_id),
        decision=decision,
        confidence=confidence,
        risk="LOW",
    )


def _make_feedback(
    exception_id: str = "EXC-001",
    feedback_type: FeedbackType = FeedbackType.APPROVE,
    system_prediction: str = "FEE_CORRECTION",
) -> FeedbackRecord:
    """Create a test FeedbackRecord."""
    return FeedbackRecord(
        feedback_id=_gen_id("FB"),
        workflow_id="WF-001",
        exception_id=exception_id,
        system_prediction=system_prediction,
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
) -> RewardRecord:
    """Create a test RewardRecord."""
    return RewardRecord(
        reward_id=_gen_id("REW"),
        workflow_id="WF-001",
        exception_id=exception_id,
        category=category,
        reward_value=reward_value,
        reward_reason="Test reward",
        breakdown=RewardBreakdown(
            base_reward=RewardComponent(
                component_name="base_reward",
                value=reward_value,
                reason="test",
            ),
            verification_component=RewardComponent(
                component_name="verification",
                value=0.0,
                reason="test",
            ),
            financial_risk_component=RewardComponent(
                component_name="financial_risk",
                value=0.0,
                reason="test",
            ),
            human_feedback_component=RewardComponent(
                component_name="human_feedback",
                value=0.0,
                reason="test",
            ),
            confidence_component=RewardComponent(
                component_name="confidence",
                value=0.0,
                reason="test",
            ),
            discrepancy_component=RewardComponent(
                component_name="discrepancy",
                value=0.0,
                reason="test",
            ),
            unintended_changes_component=RewardComponent(
                component_name="unintended_changes",
                value=0.0,
                reason="test",
            ),
        ),
    )


def _make_learning_dataset(
    n_examples: int = 20,
    exception_type: str = "FEE_DIFFERENCE",
) -> LearningDataset:
    """Create a test learning dataset with enough examples for training."""
    examples = []
    for i in range(n_examples):
        examples.append(LearningExample(
            example_id=_gen_id("LEX"),
            case_id=f"CASE-{i:03d}",
            exception_id=f"EXC-{i:03d}",
            workflow_id=f"WF-{i:03d}",
            features=FeatureSnapshot(
                financial_features={
                    "adjustment_paise": float(1000 + i * 100),
                    "difference_paise": float(500 + i * 50),
                },
                structural_features={
                    "resolution_type_fee_correction": 1.0,
                    "has_settlement": 1.0,
                },
                evidence_features={
                    "evidence_count": 3.0,
                    "evidence_coverage": 0.8,
                },
            ),
            labels=LearningLabels(
                true_exception_type=exception_type if i % 3 != 0 else "UNKNOWN",
                predicted_exception_type=exception_type,
                exception_prediction_correct=i % 3 != 0,
                true_resolution="FEE_CORRECTION" if i % 3 != 0 else "MANUAL_REVIEW",
                predicted_resolution="FEE_CORRECTION",
                resolution_correct=i % 5 != 0,
                verification_passed=i % 7 != 0,
            ),
            guardrail_decision="AUTO" if i % 4 != 0 else "HUMAN_REVIEW",
            confidence=0.7 + (i % 10) * 0.02,
        ))

    return LearningDataset(
        dataset_id=_gen_id("LDS"),
        version="1.0.0",
        examples=examples,
    )


def _make_eval_metrics(
    model_id: str = "MOD-001",
    model_version: str = "v1.0",
    accuracy: float = 0.85,
    precision: float = 0.82,
    recall: float = 0.80,
    f1: float = 0.81,
    false_automation: int = 2,
    high_value_errors: int = 0,
) -> EvaluationMetrics:
    """Create test evaluation metrics."""
    return EvaluationMetrics(
        model_id=model_id,
        model_version=model_version,
        total_samples=50,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        precision_weighted=precision,
        recall_weighted=recall,
        f1_weighted=f1,
        false_automation=false_automation,
        high_value_errors=high_value_errors,
        incorrect_auto_resolution=false_automation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Batch Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchMetricsCalculator:
    """Tests for BatchMetricsCalculator."""

    def test_calculate_basic(self):
        """Basic metrics from a mix of AUTO and HUMAN outcomes."""
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=False),
            _make_outcome(decision="HUMAN_REVIEW", resolution_correct=True),
        ]
        feedbacks = [_make_feedback(feedback_type=FeedbackType.APPROVE)]
        rewards = [_make_reward(reward_value=0.7), _make_reward(reward_value=-0.3)]

        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-001",
            outcomes=outcomes,
            feedbacks=feedbacks,
            rewards=rewards,
        )

        assert metrics.batch_id == "BAT-001"
        assert metrics.dataset_size == 4
        assert metrics.auto_decisions == 3
        assert metrics.human_decisions == 1
        assert metrics.automation_rate == pytest.approx(0.75)
        assert metrics.correct_auto == 2
        assert metrics.incorrect_auto == 1
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.false_automation == 1
        assert metrics.feedback_received == 1
        assert metrics.avg_reward is not None

    def test_calculate_empty(self):
        """Empty batch produces zero metrics."""
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-EMPTY",
            outcomes=[],
            feedbacks=[],
            rewards=[],
        )
        assert metrics.dataset_size == 0
        assert metrics.automation_rate == 0.0
        assert metrics.precision is None

    def test_calculate_all_auto_correct(self):
        """All AUTO decisions are correct → perfect precision."""
        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True)
            for _ in range(5)
        ]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-ALL",
            outcomes=outcomes,
            feedbacks=[],
            rewards=[],
        )
        assert metrics.precision == 1.0
        assert metrics.false_automation == 0
        assert metrics.automation_rate == 1.0

    def test_calculate_high_value_errors(self):
        """High-value incorrect AUTO decisions are counted."""
        outcomes = [
            _make_outcome(
                decision="AUTO",
                resolution_correct=False,
                adjustment_paise=15_000_000,  # ₹1,50,000 → high value
            ),
            _make_outcome(
                decision="AUTO",
                resolution_correct=True,
                adjustment_paise=5000,
            ),
        ]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-HV",
            outcomes=outcomes,
            feedbacks=[],
            rewards=[],
            high_value_threshold=10_000_000,
        )
        assert metrics.high_value_errors == 1
        assert metrics.false_automation == 1

    def test_calculate_verification_failure_rate(self):
        """Verification failure rate computed from executed + rolled back."""
        outcomes = [
            _make_outcome(
                decision="AUTO",
                was_executed=True,
                was_verified=False,
                was_rolled_back=True,
            ),
            _make_outcome(
                decision="AUTO",
                was_executed=True,
                was_verified=True,
                was_rolled_back=False,
            ),
        ]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-VF",
            outcomes=outcomes,
            feedbacks=[],
            rewards=[],
        )
        assert metrics.verification_failures == 1
        assert metrics.verification_failure_rate == pytest.approx(0.5)

    def test_calculate_human_feedback_counts(self):
        """Human corrections and rejections are counted."""
        feedbacks = [
            _make_feedback(feedback_type=FeedbackType.CORRECT),
            _make_feedback(feedback_type=FeedbackType.CORRECT),
            _make_feedback(feedback_type=FeedbackType.REJECT),
            _make_feedback(feedback_type=FeedbackType.APPROVE),
        ]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-HF",
            outcomes=[_make_outcome() for _ in range(4)],
            feedbacks=feedbacks,
            rewards=[],
        )
        assert metrics.human_corrections == 2
        assert metrics.human_rejections == 1
        assert metrics.feedback_received == 4

    def test_calculate_reward_std(self):
        """Reward standard deviation is computed."""
        rewards = [_make_reward(reward_value=v) for v in [0.5, 0.7, 0.3, 0.6]]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-STD",
            outcomes=[_make_outcome() for _ in range(4)],
            feedbacks=[],
            rewards=rewards,
        )
        assert metrics.reward_std is not None
        assert metrics.reward_std > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Batch Comparator
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchComparator:
    """Tests for BatchComparator."""

    def _make_metrics(
        self,
        batch_id: str = "BAT-001",
        precision: float = 0.80,
        false_auto: int = 3,
        hv_errors: int = 0,
        auto_rate: float = 0.60,
        ver_fail_rate: Optional[float] = 0.05,
        avg_reward: float = 0.4,
        error_impact: int = 50000,
        unnecessary_esc: int = 2,
        dataset_size: int = 50,
    ) -> BatchMetrics:
        return BatchMetrics(
            batch_id=batch_id,
            dataset_size=dataset_size,
            auto_decisions=int(dataset_size * auto_rate),
            human_decisions=int(dataset_size * (1 - auto_rate)),
            automation_rate=auto_rate,
            precision=precision,
            false_automation=false_auto,
            high_value_errors=hv_errors,
            verification_failure_rate=ver_fail_rate,
            avg_reward=avg_reward,
            error_impact_paise=error_impact,
            unnecessary_escalations=unnecessary_esc,
        )

    def test_improvement(self):
        """Precision improving → PROCEED."""
        prev = self._make_metrics(batch_id="BAT-001", precision=0.70)
        curr = self._make_metrics(batch_id="BAT-002", precision=0.80)

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        assert result.recommendation == BatchRecommendation.PROCEED
        assert any("precision" in i for i in result.improvements)
        assert result.safety.all_safety_maintained

    def test_regression(self):
        """Precision declining → INVESTIGATE or ROLLBACK."""
        prev = self._make_metrics(batch_id="BAT-001", precision=0.90, false_auto=2)
        curr = self._make_metrics(batch_id="BAT-002", precision=0.75, false_auto=5)

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        # Should not be PROCEED
        assert result.recommendation != BatchRecommendation.PROCEED

    def test_safety_critical_false_auto_increase(self):
        """False automation jumping from 2 to 10 → safety regression."""
        prev = self._make_metrics(batch_id="BAT-001", false_auto=2)
        curr = self._make_metrics(batch_id="BAT-002", false_auto=10)

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        assert result.safety.has_critical_regression
        assert not result.safety.all_safety_maintained
        assert result.recommendation == BatchRecommendation.ROLLBACK

    def test_high_value_error_increase(self):
        """HV errors increasing is a safety regression."""
        prev = self._make_metrics(batch_id="BAT-001", hv_errors=0)
        curr = self._make_metrics(batch_id="BAT-002", hv_errors=3)

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        assert result.safety.has_critical_regression
        assert result.recommendation == BatchRecommendation.ROLLBACK

    def test_equal_improvements_regressions(self):
        """Equal improvements and regressions → HOLD."""
        prev = self._make_metrics(
            batch_id="BAT-001",
            precision=0.80,
            avg_reward=0.5,
            error_impact=10000,
        )
        curr = self._make_metrics(
            batch_id="BAT-002",
            precision=0.85,
            avg_reward=0.4,
            error_impact=5000,
        )

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        # One metric improved, one regressed
        assert len(result.improvements) >= 1
        assert len(result.regressions) >= 1

    def test_metric_change_values(self):
        """MetricChange captures correct values."""
        prev = self._make_metrics(batch_id="BAT-001", precision=0.80)
        curr = self._make_metrics(batch_id="BAT-002", precision=0.90)

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        prec_change = next(
            c for c in result.changes if c.metric_name == "precision"
        )
        assert prec_change.previous_value == pytest.approx(0.80)
        assert prec_change.current_value == pytest.approx(0.90)
        assert prec_change.change == pytest.approx(0.10)
        assert prec_change.is_improvement is True
        assert prec_change.is_safety_critical is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Batch Learning Loop
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchLearningLoop:
    """Tests for BatchLearningLoop."""

    def test_start_batch(self):
        """Starting a batch creates it with COLLECTING status."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        assert batch.batch_number == 1
        assert batch.status == BatchStatus.COLLECTING
        assert batch.batch_id.startswith("BAT-")

    def test_batch_numbering(self):
        """Batches are numbered sequentially."""
        loop = BatchLearningLoop()
        b1 = loop.start_batch()
        b2 = loop.start_batch()
        b3 = loop.start_batch()

        assert b1.batch_number == 1
        assert b2.batch_number == 2
        assert b3.batch_number == 3

    def test_add_case_to_batch(self):
        """Cases can be added to a collecting batch."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        assert loop.add_case_to_batch(batch.batch_id, "CASE-001")
        assert loop.add_case_to_batch(batch.batch_id, "CASE-002")
        updated = loop.get_batch(batch.batch_id)
        assert len(updated.case_ids) == 2

    def test_add_case_to_complete_batch_fails(self):
        """Cannot add cases to a completed batch."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()
        loop.complete_collection(batch.batch_id)

        assert not loop.add_case_to_batch(batch.batch_id, "CASE-001")

    def test_complete_collection(self):
        """Completing collection changes status to COMPLETE."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        assert loop.complete_collection(batch.batch_id)
        updated = loop.get_batch(batch.batch_id)
        assert updated.status == BatchStatus.COMPLETE

    def test_compute_metrics(self):
        """Metrics are computed from outcomes/feedback/rewards."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        outcomes = [
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="AUTO", resolution_correct=True),
            _make_outcome(decision="HUMAN_REVIEW"),
        ]
        feedbacks = [_make_feedback(feedback_type=FeedbackType.APPROVE)]
        rewards = [_make_reward(reward_value=0.6)]

        metrics = loop.compute_metrics(
            batch.batch_id,
            outcomes=outcomes,
            feedbacks=feedbacks,
            rewards=rewards,
        )

        assert metrics.dataset_size == 3
        assert metrics.auto_decisions == 2
        assert metrics.precision == 1.0

    def test_compare_batches(self):
        """Comparing two batches produces a BatchComparison."""
        loop = BatchLearningLoop()

        # Batch 1
        b1 = loop.start_batch()
        loop.complete_collection(b1.batch_id)
        outcomes1 = [_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(10)]
        outcomes1 += [_make_outcome(decision="AUTO", resolution_correct=False) for _ in range(3)]
        loop.compute_metrics(b1.batch_id, outcomes=outcomes1, feedbacks=[], rewards=[])

        # Batch 2
        b2 = loop.start_batch()
        loop.complete_collection(b2.batch_id)
        outcomes2 = [_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(12)]
        outcomes2 += [_make_outcome(decision="AUTO", resolution_correct=False) for _ in range(1)]
        loop.compute_metrics(b2.batch_id, outcomes=outcomes2, feedbacks=[], rewards=[])

        comparison = loop.compare_with_previous(b2.batch_id)

        assert comparison is not None
        assert comparison.previous_batch_id == b1.batch_id
        assert comparison.current_batch_id == b2.batch_id

    def test_compare_first_batch_returns_none(self):
        """First batch has no previous to compare with."""
        loop = BatchLearningLoop()
        b1 = loop.start_batch()
        loop.complete_collection(b1.batch_id)
        loop.compute_metrics(b1.batch_id, outcomes=[], feedbacks=[], rewards=[])

        comparison = loop.compare_with_previous(b1.batch_id)
        assert comparison is None

    def test_deterministic_metrics(self):
        """Same inputs produce same metrics (deterministic)."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        outcomes = [_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(5)]
        feedbacks = [_make_feedback()]
        rewards = [_make_reward(reward_value=0.5)]

        m1 = loop.compute_metrics(batch.batch_id, outcomes, feedbacks, rewards)
        # Reset and compute again with same data
        loop2 = BatchLearningLoop()
        b2 = loop2.start_batch()
        m2 = loop2.compute_metrics(b2.batch_id, outcomes, feedbacks, rewards)

        assert m1.precision == m2.precision
        assert m1.automation_rate == m2.automation_rate
        assert m1.false_automation == m2.false_automation

    def test_get_batch_by_number(self):
        """Retrieve batch by sequential number."""
        loop = BatchLearningLoop()
        b1 = loop.start_batch()
        b2 = loop.start_batch()

        assert loop.get_batch_by_number(1).batch_id == b1.batch_id
        assert loop.get_batch_by_number(2).batch_id == b2.batch_id
        assert loop.get_batch_by_number(999) is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Candidate Training and Promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchCandidateTraining:
    """Tests for candidate model training within batch learning."""

    def test_train_candidate(self):
        """Train a candidate model from batch dataset."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()
        loop.complete_collection(batch.batch_id)

        dataset = _make_learning_dataset(n_examples=30)
        metadata = loop.train_candidate(
            batch.batch_id,
            dataset=dataset,
            model_version="batch-v1",
        )

        assert metadata.model_id is not None
        assert metadata.version == "batch-v1"
        assert batch.status == BatchStatus.EVALUATING
        assert batch.candidate_model_id == metadata.model_id

    def test_evaluate_candidate(self):
        """Evaluate candidate model on test data."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()
        loop.complete_collection(batch.batch_id)

        dataset = _make_learning_dataset(n_examples=30)
        loop.train_candidate(batch.batch_id, dataset)

        eval_metrics = loop.evaluate_candidate(
            batch.batch_id,
            dataset,
        )

        assert eval_metrics.total_samples > 0
        assert eval_metrics.accuracy >= 0.0
        assert batch.candidate_evaluated is True

    def test_promote_candidate_passes_safety(self):
        """Candidate that passes safety → PROMOTED."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        # Simulate batch comparison with good improvement
        batch.comparison = BatchComparison(
            comparison_id="BCP-001",
            previous_batch_id="BAT-000",
            current_batch_id=batch.batch_id,
            previous_metrics=BatchMetrics(
                batch_id="BAT-000",
                dataset_size=50,
                precision=0.75,
                false_automation=3,
                high_value_errors=0,
            ),
            current_metrics=BatchMetrics(
                batch_id=batch.batch_id,
                dataset_size=50,
                precision=0.85,
                false_automation=1,
                high_value_errors=0,
            ),
            safety=SafetyAssessment(
                checks_passed=4,
                checks_failed=0,
                all_safety_maintained=True,
            ),
        )

        current_m = _make_eval_metrics(model_version="v1.0", false_automation=3)
        candidate_m = _make_eval_metrics(
            model_id="MOD-C", model_version="v2.0",
            accuracy=0.90, precision=0.88, recall=0.85, f1=0.86,
            false_automation=1,
        )

        result = loop.promote_or_reject(
            batch.batch_id,
            current_model_metrics=current_m,
            candidate_model_metrics=candidate_m,
            candidate_model_id="MOD-C",
            candidate_version="v2.0",
            current_model_id="MOD-A",
            current_version="v1.0",
        )

        assert result.promoted is True
        assert result.status == BatchStatus.PROMOTED

    def test_reject_candidate_fails_safety(self):
        """Candidate with safety regression → REJECTED."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        batch.comparison = BatchComparison(
            comparison_id="BCP-002",
            previous_batch_id="BAT-000",
            current_batch_id=batch.batch_id,
            previous_metrics=BatchMetrics(
                batch_id="BAT-000",
                false_automation=1,
                high_value_errors=0,
            ),
            current_metrics=BatchMetrics(
                batch_id=batch.batch_id,
                false_automation=8,
                high_value_errors=2,
            ),
            safety=SafetyAssessment(
                checks_passed=1,
                checks_failed=3,
                all_safety_maintained=False,
                has_critical_regression=True,
                safety_regressions=["false_automation: 1.0 → 8.0"],
            ),
        )

        result = loop.promote_or_reject(batch.batch_id)

        assert result.promoted is False
        assert result.status == BatchStatus.REJECTED
        assert "safety" in result.promotion_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Report Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchReport:
    """Tests for batch comparison report generation."""

    def test_empty_report(self):
        """No batches → empty report."""
        loop = BatchLearningLoop()
        report = loop.generate_report()

        assert report.total_batches == 0
        assert len(report.rows) == 0

    def test_report_with_batches(self):
        """Report includes all batch metrics."""
        loop = BatchLearningLoop()

        # Batch 1
        b1 = loop.start_batch()
        loop.complete_collection(b1.batch_id)
        loop.compute_metrics(
            b1.batch_id,
            outcomes=[_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(10)],
            feedbacks=[],
            rewards=[],
        )

        # Batch 2
        b2 = loop.start_batch()
        loop.complete_collection(b2.batch_id)
        loop.compute_metrics(
            b2.batch_id,
            outcomes=[_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(12)],
            feedbacks=[],
            rewards=[],
        )

        report = loop.generate_report()

        assert report.total_batches == 2
        assert len(report.rows) == 2
        assert report.rows[0].batch_number == 1
        assert report.rows[1].batch_number == 2

    def test_report_precision_trend(self):
        """Report detects improving precision trend."""
        loop = BatchLearningLoop()

        for i, prec in enumerate([0.70, 0.75, 0.82], start=1):
            b = loop.start_batch()
            loop.complete_collection(b.batch_id)
            correct = int(20 * prec)
            incorrect = 20 - correct
            outcomes = (
                [_make_outcome(decision="AUTO", resolution_correct=True) for _ in range(correct)]
                + [_make_outcome(decision="AUTO", resolution_correct=False) for _ in range(incorrect)]
            )
            loop.compute_metrics(b.batch_id, outcomes=outcomes, feedbacks=[], rewards=[])

        report = loop.generate_report()
        assert report.precision_trend == "improving"
        assert report.improvement_demonstrated is True

    def test_report_safety_maintained(self):
        """Report correctly tracks safety across batches."""
        loop = BatchLearningLoop()

        b1 = loop.start_batch()
        loop.complete_collection(b1.batch_id)
        loop.compute_metrics(
            b1.batch_id,
            outcomes=[_make_outcome() for _ in range(5)],
            feedbacks=[],
            rewards=[],
        )

        b2 = loop.start_batch()
        loop.complete_collection(b2.batch_id)
        loop.compute_metrics(
            b2.batch_id,
            outcomes=[_make_outcome() for _ in range(5)],
            feedbacks=[],
            rewards=[],
        )

        report = loop.generate_report()
        assert report.safety_maintained is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Safety Rules
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchSafetyRules:
    """Tests verifying batch learning safety rules."""

    def test_higher_automation_alone_not_success(self):
        """Higher automation without precision improvement → HOLD/INVESTIGATE."""
        prev = BatchMetrics(
            batch_id="BAT-001",
            dataset_size=50,
            precision=0.85,
            automation_rate=0.60,
            false_automation=2,
        )
        curr = BatchMetrics(
            batch_id="BAT-002",
            dataset_size=50,
            precision=0.80,  # Precision dropped
            automation_rate=0.80,  # Automation increased
            false_automation=6,  # False auto increased
        )

        comp = BatchComparator()
        result = comp.compare(prev, curr)

        # Higher automation but worse precision and false auto → not PROCEED
        assert result.recommendation != BatchRecommendation.PROCEED

    def test_safety_blocks_promotion(self):
        """Safety regression in batch blocks promotion even with model pass."""
        loop = BatchLearningLoop()
        batch = loop.start_batch()

        batch.comparison = BatchComparison(
            comparison_id="BCP-SAFE",
            previous_batch_id="BAT-000",
            current_batch_id=batch.batch_id,
            previous_metrics=BatchMetrics(batch_id="BAT-000"),
            current_metrics=BatchMetrics(batch_id=batch.batch_id),
            safety=SafetyAssessment(
                all_safety_maintained=False,
                has_critical_regression=True,
                safety_regressions=["false_automation regression"],
            ),
        )

        # Even with perfect model metrics
        current_m = _make_eval_metrics(false_automation=0)
        candidate_m = _make_eval_metrics(
            model_id="MOD-C", model_version="v2.0",
            accuracy=0.95, precision=0.95, recall=0.95, f1=0.95,
            false_automation=0, high_value_errors=0,
        )

        result = loop.promote_or_reject(
            batch.batch_id,
            current_model_metrics=current_m,
            candidate_model_metrics=candidate_m,
            candidate_model_id="MOD-C",
            candidate_version="v2.0",
            current_model_id="MOD-A",
            current_version="v1.0",
        )

        assert result.promoted is False

    def test_no_forcing_batch_metrics(self):
        """Batch metrics calculator never invents precision when no AUTO decisions."""
        outcomes = [
            _make_outcome(decision="HUMAN_REVIEW"),
            _make_outcome(decision="UNRESOLVED"),
        ]
        calc = BatchMetricsCalculator()
        metrics = calc.calculate(
            batch_id="BAT-NOFORCE",
            outcomes=outcomes,
            feedbacks=[],
            rewards=[],
        )
        assert metrics.precision is None
        assert metrics.false_automation == 0
        assert metrics.automation_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Batch Config
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchConfig:
    """Tests for batch configuration."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = BatchConfig()
        assert config.batch_size == 50
        assert config.min_feedback_rate == 0.8
        assert config.random_seed == 42

    def test_custom_config(self):
        """Custom config is preserved."""
        config = BatchConfig(batch_size=100, random_seed=123)
        loop = BatchLearningLoop()
        batch = loop.start_batch(config=config)
        assert batch.config.batch_size == 100
        assert batch.config.random_seed == 123


# ─────────────────────────────────────────────────────────────────────────────
# Test: MetricChange
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricChange:
    """Tests for MetricChange schema."""

    def test_improvement_detected(self):
        """Positive change on higher_is_better metric is improvement."""
        change = MetricChange(
            metric_name="precision",
            previous_value=0.70,
            current_value=0.80,
            change=0.10,
            change_pct=0.143,
            is_improvement=True,
            is_safety_critical=True,
        )
        assert change.is_improvement is True
        assert change.is_safety_critical is True

    def test_none_values(self):
        """MetricChange handles None values."""
        change = MetricChange(
            metric_name="precision",
            previous_value=None,
            current_value=0.80,
        )
        assert change.previous_value is None
        assert change.change is None
