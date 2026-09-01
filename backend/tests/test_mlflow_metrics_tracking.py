"""
Tests for Razorpay CloseLoop Phase 10C — MLflow Metrics Tracking.

Tests metrics conversion, logging, and consistency with Phase 9 definitions.
"""

import pytest

from app.schemas.learning_metrics import (
    AutomationMetrics,
    FinancialImpactMetrics,
    HumanReviewMetrics,
    LearningMetrics,
    PrecisionMetrics,
    RewardMetrics,
    SafetyAssessmentResult,
    VerificationMetrics,
)
from app.schemas.mlflow_tracking import MetricsSnapshot
from app.schemas.model_training import EvaluationMetrics
from app.services.mlflow_metrics_builder import (
    EvaluationMetricsBuilder,
    LearningMetricsBuilder,
    MLflowMetricsBuilder,
)
from app.services.mlflow_tracking import MLflowTrackingService, MLflowConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def service() -> MLflowTrackingService:
    return MLflowTrackingService(config=MLflowConfig(tracking_uri="file:./test_mlruns"))


@pytest.fixture
def eval_metrics() -> EvaluationMetrics:
    """Sample EvaluationMetrics from Phase 9E."""
    return EvaluationMetrics(
        model_id="MOD-TEST",
        model_version="1.0.0",
        evaluated_on="test",
        total_samples=100,
        accuracy=0.85,
        precision_macro=0.83,
        recall_macro=0.82,
        f1_macro=0.825,
        precision_weighted=0.86,
        recall_weighted=0.85,
        f1_weighted=0.855,
        per_class_precision={"FEE_DIFFERENCE": 0.90, "EXACT_MATCH": 0.95},
        per_class_recall={"FEE_DIFFERENCE": 0.85, "EXACT_MATCH": 0.92},
        per_class_f1={"FEE_DIFFERENCE": 0.875, "EXACT_MATCH": 0.935},
        per_class_support={"FEE_DIFFERENCE": 40, "EXACT_MATCH": 60},
        confusion_matrix=[[34, 6], [3, 57]],
        confusion_labels=["FEE_DIFFERENCE", "EXACT_MATCH"],
        incorrect_auto_resolution=3,
        high_value_errors=0,
        unknown_case_errors=1,
        novel_pattern_errors=0,
        verification_failure_rate=0.02,
        false_automation=3,
        resolution_accuracy=0.87,
    )


@pytest.fixture
def learning_metrics() -> LearningMetrics:
    """Sample LearningMetrics from Phase 9H."""
    return LearningMetrics(
        metrics_id="LM-TEST",
        automation=AutomationMetrics(
            total_exceptions=100,
            eligible_exceptions=80,
            auto_decisions=30,
            human_decisions=40,
            unresolved_decisions=30,
            automation_rate=0.375,
            human_review_rate=0.40,
            unresolved_rate=0.30,
            successful_auto=28,
            successful_automation_rate=0.35,
            failed_auto=2,
            failed_automation_rate=0.067,
        ),
        precision=PrecisionMetrics(
            correct_auto=28,
            incorrect_auto=2,
            total_auto_with_outcome=30,
            precision=0.933,
            false_automation_count=2,
            false_automation_rate=0.067,
        ),
        human_review=HumanReviewMetrics(
            total_human_reviews=40,
            human_corrections=5,
            human_rejections=3,
            human_approvals=30,
            human_escalations=2,
            correction_rate=0.125,
            unnecessary_escalations=8,
            unnecessary_escalation_rate=0.20,
        ),
        reward=RewardMetrics(
            total_rewards=30,
            avg_reward=0.62,
            median_reward=0.70,
            reward_std=0.15,
            min_reward=-0.5,
            max_reward=0.8,
            positive_rewards=25,
            negative_rewards=5,
            neutral_rewards=0,
            positive_rate=0.833,
        ),
        financial=FinancialImpactMetrics(
            total_adjustment_paise=500000,
            avg_adjustment_paise=16667.0,
            max_adjustment_paise=100000,
            total_error_impact_paise=20000,
            avg_error_impact_paise=10000.0,
            high_value_error_count=0,
            high_value_error_impact_paise=0,
            impact_avoided_paise=300000,
            discrepancy_eliminated_count=25,
            discrepancy_elimination_rate=0.833,
        ),
        verification=VerificationMetrics(
            total_executed=30,
            total_verified=28,
            total_rolled_back=2,
            total_verification_failed=0,
            verification_success_rate=0.933,
            rollback_rate=0.067,
        ),
        safety=SafetyAssessmentResult(verdict="SAFE"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MetricsSnapshot Expanded Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsSnapshotExpanded:
    """Test that MetricsSnapshot has all required fields."""

    def test_all_field_categories_exist(self):
        snap = MetricsSnapshot(run_id="RUN-123")
        # Classification
        assert hasattr(snap, "accuracy")
        assert hasattr(snap, "per_class_precision")
        assert hasattr(snap, "confusion_matrix")
        # Safety
        assert hasattr(snap, "false_automation")
        assert hasattr(snap, "high_value_errors")
        assert hasattr(snap, "verification_failure_rate")
        assert hasattr(snap, "unsafe_decision_rate")
        assert hasattr(snap, "unresolved_rate")
        # Resolution
        assert hasattr(snap, "resolution_accuracy")
        assert hasattr(snap, "candidate_selection_accuracy")
        # Automation
        assert hasattr(snap, "auto_decisions")
        assert hasattr(snap, "automation_rate")
        # Human review
        assert hasattr(snap, "total_human_reviews")
        assert hasattr(snap, "correction_rate")
        # Precision
        assert hasattr(snap, "correct_auto")
        assert hasattr(snap, "precision")
        # Verification
        assert hasattr(snap, "total_executed")
        assert hasattr(snap, "verification_success_rate")
        # Financial
        assert hasattr(snap, "total_adjustment_paise")
        assert hasattr(snap, "total_error_impact_paise")
        assert hasattr(snap, "high_value_error_impact_paise")
        # Reward
        assert hasattr(snap, "avg_reward")
        assert hasattr(snap, "positive_reward_rate")

    def test_to_mlflow_dict_all_categories(self):
        snap = MetricsSnapshot(
            run_id="RUN-123",
            accuracy=0.85,
            false_automation=2,
            auto_decisions=30,
            automation_rate=0.375,
            total_human_reviews=40,
            total_adjustment_paise=500000,
            avg_reward=0.62,
            total_executed=30,
            verification_success_rate=0.933,
        )
        d = snap.to_mlflow_dict()
        assert d["accuracy"] == 0.85
        assert d["false_automation"] == 2.0
        assert d["auto_decisions"] == 30.0
        assert d["automation_rate"] == 0.375
        assert d["total_human_reviews"] == 40.0
        assert d["total_adjustment_paise"] == 500000.0
        assert d["avg_reward"] == 0.62
        assert d["total_executed"] == 30.0
        assert d["verification_success_rate"] == 0.933


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationMetricsBuilder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluationMetricsBuilder:
    """Test EvaluationMetrics → MetricsSnapshot conversion."""

    def test_basic_conversion(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        assert snap.run_id == "RUN-1"
        assert snap.accuracy == pytest.approx(0.85)
        assert snap.precision_macro == pytest.approx(0.83)
        assert snap.recall_macro == pytest.approx(0.82)
        assert snap.f1_macro == pytest.approx(0.825)
        assert snap.total_samples == 100

    def test_per_class_metrics(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        assert snap.per_class_precision["FEE_DIFFERENCE"] == pytest.approx(0.90)
        assert snap.per_class_recall["EXACT_MATCH"] == pytest.approx(0.92)
        assert snap.per_class_f1["FEE_DIFFERENCE"] == pytest.approx(0.875)
        assert snap.per_class_support["FEE_DIFFERENCE"] == 40

    def test_confusion_matrix(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        assert snap.confusion_matrix == [[34, 6], [3, 57]]
        assert snap.confusion_labels == ["FEE_DIFFERENCE", "EXACT_MATCH"]

    def test_safety_metrics(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        assert snap.false_automation == 3
        assert snap.high_value_errors == 0
        assert snap.unknown_case_errors == 1
        assert snap.verification_failure_rate == pytest.approx(0.02)
        assert snap.incorrect_auto_resolution == 3
        assert snap.resolution_accuracy == pytest.approx(0.87)

    def test_false_automation_rate_derived(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        # 3 / 100 = 0.03
        assert snap.false_automation_rate == pytest.approx(0.03)

    def test_to_mlflow_dict(self, eval_metrics: EvaluationMetrics):
        snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-1")
        d = snap.to_mlflow_dict()
        assert "accuracy" in d
        assert "per_class_precision.FEE_DIFFERENCE" not in d  # Nested dicts not flattened
        assert d["accuracy"] == pytest.approx(0.85)
        assert d["false_automation"] == 3.0


# ─────────────────────────────────────────────────────────────────────────────
# LearningMetricsBuilder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLearningMetricsBuilder:
    """Test LearningMetrics → MetricsSnapshot conversion."""

    def test_automation_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.auto_decisions == 30
        assert snap.human_decisions == 40
        assert snap.unresolved_decisions == 30
        assert snap.automation_rate == pytest.approx(0.375)
        assert snap.human_review_rate == pytest.approx(0.40)
        assert snap.successful_auto == 28
        assert snap.successful_automation_rate == pytest.approx(0.35)
        assert snap.failed_auto == 2
        assert snap.failed_automation_rate == pytest.approx(0.067)

    def test_precision_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.correct_auto == 28
        assert snap.incorrect_auto == 2
        assert snap.precision == pytest.approx(0.933)
        assert snap.false_automation == 2
        assert snap.false_automation_rate == pytest.approx(0.067)

    def test_human_review_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.total_human_reviews == 40
        assert snap.human_corrections == 5
        assert snap.human_rejections == 3
        assert snap.human_approvals == 30
        assert snap.correction_rate == pytest.approx(0.125)
        assert snap.unnecessary_escalations == 8

    def test_verification_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.total_executed == 30
        assert snap.total_verified == 28
        assert snap.total_rolled_back == 2
        assert snap.verification_success_rate == pytest.approx(0.933)
        assert snap.rollback_rate == pytest.approx(0.067)

    def test_financial_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.total_adjustment_paise == 500000
        assert snap.avg_adjustment_paise == pytest.approx(16667.0)
        assert snap.max_adjustment_paise == 100000
        assert snap.total_error_impact_paise == 20000
        assert snap.high_value_errors == 0
        assert snap.high_value_error_impact_paise == 0
        assert snap.impact_avoided_paise == 300000
        assert snap.discrepancy_eliminated_count == 25
        assert snap.discrepancy_elimination_rate == pytest.approx(0.833)

    def test_reward_metrics(self, learning_metrics: LearningMetrics):
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-2")
        assert snap.avg_reward == pytest.approx(0.62)
        assert snap.median_reward == pytest.approx(0.70)
        assert snap.reward_std == pytest.approx(0.15)
        assert snap.positive_rewards == 25
        assert snap.negative_rewards == 5
        assert snap.positive_reward_rate == pytest.approx(0.833)


# ─────────────────────────────────────────────────────────────────────────────
# MLflowMetricsBuilder Combined Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLflowMetricsBuilderCombined:
    """Test combined metrics building."""

    def test_combined_metrics(
        self, eval_metrics: EvaluationMetrics, learning_metrics: LearningMetrics
    ):
        snap = MLflowMetricsBuilder.build_combined(eval_metrics, learning_metrics, "RUN-3")
        # Should have classification from eval_metrics
        assert snap.accuracy == pytest.approx(0.85)
        assert snap.per_class_precision["FEE_DIFFERENCE"] == pytest.approx(0.90)
        # Should have automation from learning_metrics
        assert snap.automation_rate == pytest.approx(0.375)
        assert snap.human_review_rate == pytest.approx(0.40)
        # Should have financial from learning_metrics
        assert snap.total_adjustment_paise == 500000
        # Should have reward from learning_metrics
        assert snap.avg_reward == pytest.approx(0.62)
        # Should have verification from learning_metrics
        assert snap.total_executed == 30

    def test_eval_only(self, eval_metrics: EvaluationMetrics):
        snap = MLflowMetricsBuilder.build_combined(eval_metrics, None, "RUN-4")
        assert snap.accuracy == pytest.approx(0.85)
        assert snap.automation_rate is None  # No learning metrics

    def test_learning_only(self, learning_metrics: LearningMetrics):
        snap = MLflowMetricsBuilder.build_combined(None, learning_metrics, "RUN-5")
        assert snap.automation_rate == pytest.approx(0.375)
        assert snap.accuracy is None  # No eval metrics

    def test_neither(self):
        snap = MLflowMetricsBuilder.build_combined(None, None, "RUN-6")
        assert snap.run_id == "RUN-6"
        assert snap.accuracy is None


# ─────────────────────────────────────────────────────────────────────────────
# Service Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsServiceIntegration:
    """Test that tracking service correctly logs metrics."""

    def test_log_evaluation_metrics(self, service: MLflowTrackingService, eval_metrics: EvaluationMetrics):
        service.create_experiment("eval-log")
        run = service.create_run(
            experiment_name="eval-log",
            model_type="test", model_name="test", model_version="1.0.0", algorithm="xgboost",
        )
        service.log_evaluation_metrics(run.run_id, eval_metrics)
        history = service.get_metrics_history(run.run_id)
        assert len(history) == 1
        assert history[0]["accuracy"] == pytest.approx(0.85)
        assert history[0]["false_automation"] == 3.0

    def test_log_learning_metrics(self, service: MLflowTrackingService, learning_metrics: LearningMetrics):
        service.create_experiment("learn-log")
        run = service.create_run(
            experiment_name="learn-log",
            model_type="test", model_name="test", model_version="1.0.0", algorithm="xgboost",
        )
        service.log_learning_metrics(run.run_id, learning_metrics)
        history = service.get_metrics_history(run.run_id)
        assert len(history) == 1
        assert history[0]["automation_rate"] == pytest.approx(0.375)
        assert history[0]["total_adjustment_paise"] == 500000.0

    def test_log_combined_metrics(
        self, service: MLflowTrackingService, eval_metrics: EvaluationMetrics, learning_metrics: LearningMetrics
    ):
        service.create_experiment("combined-log")
        run = service.create_run(
            experiment_name="combined-log",
            model_type="test", model_name="test", model_version="1.0.0", algorithm="xgboost",
        )
        service.log_combined_metrics(run.run_id, eval_metrics, learning_metrics)
        history = service.get_metrics_history(run.run_id)
        assert len(history) == 1
        # Classification from eval
        assert history[0]["accuracy"] == pytest.approx(0.85)
        # Automation from learning
        assert history[0]["automation_rate"] == pytest.approx(0.375)

    def test_log_metrics_nonexistent_run(self, service: MLflowTrackingService, eval_metrics: EvaluationMetrics):
        with pytest.raises(ValueError, match="not found"):
            service.log_evaluation_metrics("RUN-FAKE", eval_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricConsistency:
    """Verify metrics are consistent with Phase 9 definitions."""

    def test_precision_formula_matches_phase9(
        self, eval_metrics: EvaluationMetrics, learning_metrics: LearningMetrics
    ):
        """Precision = correct_auto / total_auto_with_outcome (from PrecisionMetrics)."""
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-C")
        # Phase 9 definition: precision = 28 / 30 = 0.933...
        assert snap.precision == pytest.approx(0.933, abs=0.01)

    def test_automation_rate_formula_matches_phase9(self, learning_metrics: LearningMetrics):
        """Automation rate = auto_decisions / eligible_exceptions (from AutomationMetrics)."""
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-C")
        # Phase 9 definition: 30 / 80 = 0.375
        assert snap.automation_rate == pytest.approx(0.375)

    def test_verification_rate_formula_matches_phase9(self, learning_metrics: LearningMetrics):
        """Verification success rate = total_verified / total_executed."""
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-C")
        # Phase 9 definition: 28 / 30 = 0.933...
        assert snap.verification_success_rate == pytest.approx(0.933, abs=0.01)

    def test_correction_rate_formula_matches_phase9(self, learning_metrics: LearningMetrics):
        """Correction rate = human_corrections / total_human_reviews."""
        snap = LearningMetricsBuilder.build(learning_metrics, "RUN-C")
        # Phase 9 definition: 5 / 40 = 0.125
        assert snap.correction_rate == pytest.approx(0.125)

    def test_false_automation_rate_consistent(
        self, eval_metrics: EvaluationMetrics, learning_metrics: LearningMetrics
    ):
        """Both sources should report consistent false automation."""
        eval_snap = EvaluationMetricsBuilder.build(eval_metrics, "RUN-E")
        learn_snap = LearningMetricsBuilder.build(learning_metrics, "RUN-L")
        # Both should be defined and positive
        assert eval_snap.false_automation_rate is not None
        assert learn_snap.false_automation_rate is not None
        assert eval_snap.false_automation_rate > 0
        assert learn_snap.false_automation_rate > 0


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsEdgeCases:
    """Test edge cases in metrics conversion."""

    def test_empty_eval_metrics(self):
        eval_m = EvaluationMetrics(model_id="M1", model_version="1.0")
        snap = EvaluationMetricsBuilder.build(eval_m, "RUN-E1")
        assert snap.accuracy == 0.0
        assert snap.false_automation == 0
        assert snap.total_samples == 0

    def test_empty_learning_metrics(self):
        learn_m = LearningMetrics(metrics_id="LM1")
        snap = LearningMetricsBuilder.build(learn_m, "RUN-L1")
        assert snap.auto_decisions == 0
        assert snap.automation_rate == 0.0
        assert snap.total_adjustment_paise == 0

    def test_high_safety_metrics(self):
        """High false automation should be properly captured."""
        learn_m = LearningMetrics(
            metrics_id="LM-SAFE",
            precision=PrecisionMetrics(
                correct_auto=5,
                incorrect_auto=25,
                total_auto_with_outcome=30,
                precision=0.167,
                false_automation_count=25,
                false_automation_rate=0.833,
            ),
        )
        snap = LearningMetricsBuilder.build(learn_m, "RUN-SAFE")
        assert snap.false_automation == 25
        assert snap.false_automation_rate == pytest.approx(0.833, abs=0.01)
        assert snap.precision == pytest.approx(0.167, abs=0.01)

    def test_zero_total_samples_eval(self):
        eval_m = EvaluationMetrics(
            model_id="M1", model_version="1.0",
            total_samples=0,
            false_automation=0,
        )
        snap = EvaluationMetricsBuilder.build(eval_m, "RUN-Z")
        assert snap.false_automation_rate is None  # 0 / 0 = undefined


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Snapshots / History Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleSnapshots:
    """Test logging multiple snapshots over time."""

    def test_multiple_eval_snapshots(self, service: MLflowTrackingService, eval_metrics: EvaluationMetrics):
        service.create_experiment("multi-eval")
        run = service.create_run(
            experiment_name="multi-eval",
            model_type="test", model_name="test", model_version="1.0.0", algorithm="xgboost",
        )
        for i in range(3):
            eval_m = eval_metrics.model_copy(update={"accuracy": 0.80 + i * 0.05})
            service.log_evaluation_metrics(run.run_id, eval_m)
        history = service.get_metrics_history(run.run_id)
        assert len(history) == 3
        assert history[0]["accuracy"] == pytest.approx(0.80)
        assert history[2]["accuracy"] == pytest.approx(0.90)

    def test_combined_then_learning(self, service: MLflowTrackingService, eval_metrics: EvaluationMetrics, learning_metrics: LearningMetrics):
        service.create_experiment("seq")
        run = service.create_run(
            experiment_name="seq",
            model_type="test", model_name="test", model_version="1.0.0", algorithm="xgboost",
        )
        service.log_evaluation_metrics(run.run_id, eval_metrics)
        service.log_learning_metrics(run.run_id, learning_metrics)
        history = service.get_metrics_history(run.run_id)
        assert len(history) == 2
        # First: classification metrics
        assert history[0]["accuracy"] == pytest.approx(0.85)
        assert "automation_rate" not in history[0] or history[0].get("automation_rate") is None
        # Second: automation metrics
        assert history[1]["automation_rate"] == pytest.approx(0.375)
