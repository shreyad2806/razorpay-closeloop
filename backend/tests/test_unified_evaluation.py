"""
Tests for Razorpay CloseLoop Phase 10F — Unified Model Evaluation.

Tests the evaluation pipeline, metric comparison, safety checks,
and MLflow integration.
"""

import pytest

from app.schemas.model_training import EvaluationMetrics
from app.schemas.unified_evaluation import (
    EvaluationVerdict,
    MetricComparison,
    SafetyRegressionCheck,
    SafetyRegressionSeverity,
    UnifiedEvaluationReport,
)
from app.services.mlflow_tracking import MLflowTrackingService, MLflowConfig
from app.services.unified_evaluation import (
    EvaluationThresholds,
    UnifiedEvaluationService,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def service() -> UnifiedEvaluationService:
    return UnifiedEvaluationService()


@pytest.fixture
def tracking_service() -> MLflowTrackingService:
    return MLflowTrackingService(config=MLflowConfig(tracking_uri="file:./test_mlruns"))


@pytest.fixture
def current_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id="MOD-CURRENT",
        model_version="1.0.0",
        total_samples=100,
        accuracy=0.80,
        precision_macro=0.78,
        recall_macro=0.76,
        f1_macro=0.77,
        false_automation=5,
        high_value_errors=1,
        unknown_case_errors=2,
        incorrect_auto_resolution=5,
        verification_failure_rate=0.05,
        resolution_accuracy=0.82,
    )


@pytest.fixture
def candidate_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id="MOD-CANDIDATE",
        model_version="2.0.0",
        total_samples=100,
        accuracy=0.85,
        precision_macro=0.83,
        recall_macro=0.81,
        f1_macro=0.82,
        false_automation=3,
        high_value_errors=0,
        unknown_case_errors=1,
        incorrect_auto_resolution=3,
        verification_failure_rate=0.03,
        resolution_accuracy=0.87,
    )


@pytest.fixture
def worse_candidate_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id="MOD-WORSE",
        model_version="3.0.0",
        total_samples=100,
        accuracy=0.75,
        precision_macro=0.72,
        recall_macro=0.70,
        f1_macro=0.71,
        false_automation=10,
        high_value_errors=3,
        unknown_case_errors=5,
        incorrect_auto_resolution=10,
        verification_failure_rate=0.15,
        resolution_accuracy=0.70,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifiedEvaluationReportSchema:
    """Test UnifiedEvaluationReport schema."""

    def test_basic_creation(self):
        report = UnifiedEvaluationReport(
            report_id="EVAL-123",
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
        )
        assert report.report_id == "EVAL-123"
        assert report.verdict == EvaluationVerdict.DEFER
        assert report.all_safety_passed is True

    def test_summary(self):
        report = UnifiedEvaluationReport(
            report_id="EVAL-1",
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
            current_model_version="1.0.0",
            verdict=EvaluationVerdict.PROMOTE,
            total_improvements=5,
            total_regressions=1,
        )
        s = report.summary()
        assert "2.0.0" in s
        assert "1.0.0" in s
        assert "PROMOTE" in s
        assert "5" in s

    def test_to_report_dict(self):
        report = UnifiedEvaluationReport(
            report_id="EVAL-1",
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
            current_model_version="1.0.0",
            candidate_accuracy=0.85,
            verdict=EvaluationVerdict.PROMOTE,
        )
        d = report.to_report_dict()
        assert d["report_id"] == "EVAL-1"
        assert d["candidate_model"]["version"] == "2.0.0"
        assert d["metrics"]["classification"]["candidate_accuracy"] == 0.85
        assert d["verdict"]["verdict"] == "PROMOTE"


class TestMetricComparisonSchema:
    """Test MetricComparison schema."""

    def test_basic(self):
        comp = MetricComparison(
            metric_name="accuracy",
            current_value=0.80,
            candidate_value=0.85,
            change=0.05,
            is_improvement=True,
        )
        assert comp.metric_name == "accuracy"
        assert comp.change == 0.05
        assert comp.is_improvement is True


class TestSafetyRegressionCheckSchema:
    """Test SafetyRegressionCheck schema."""

    def test_passed(self):
        check = SafetyRegressionCheck(
            metric_name="false_automation",
            current_value=5.0,
            candidate_value=3.0,
            passed=True,
            severity=SafetyRegressionSeverity.NONE,
        )
        assert check.passed is True

    def test_failed_critical(self):
        check = SafetyRegressionCheck(
            metric_name="false_automation",
            current_value=0.0,
            candidate_value=10.0,
            passed=False,
            severity=SafetyRegressionSeverity.CRITICAL,
        )
        assert check.passed is False
        assert check.severity == SafetyRegressionSeverity.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluationPipeline:
    """Test the complete evaluation pipeline."""

    def test_improving_candidate(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            current_model_id="MOD-CURRENT",
            current_model_version="1.0.0",
            candidate_model_id="MOD-CANDIDATE",
            candidate_model_version="2.0.0",
        )
        assert report.verdict == EvaluationVerdict.PROMOTE
        assert report.all_safety_passed is True
        assert report.total_improvements > 0
        assert report.promotion_eligible is True

    def test_worse_candidate(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        worse_candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=worse_candidate_metrics,
            current_model_id="MOD-CURRENT",
            current_model_version="1.0.0",
            candidate_model_id="MOD-WORSE",
            candidate_model_version="3.0.0",
        )
        assert report.verdict in (EvaluationVerdict.REJECT, EvaluationVerdict.DEFER)
        assert report.all_safety_passed is False

    def test_no_current_model(
        self, service: UnifiedEvaluationService,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=None,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-FIRST",
            candidate_model_version="1.0.0",
        )
        assert report.verdict == EvaluationVerdict.PROMOTE
        assert report.promotion_eligible is True

    def test_metric_comparisons_populated(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
        )
        assert len(report.comparisons) > 0
        # Check that accuracy comparison exists
        acc_comp = next(c for c in report.comparisons if c.metric_name == "accuracy")
        assert acc_comp.current_value == 0.80
        assert acc_comp.candidate_value == 0.85
        assert acc_comp.is_improvement is True

    def test_safety_checks_populated(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
        )
        assert len(report.safety_checks) > 0
        # False automation check should pass (3 <= 5)
        fa_check = next(s for s in report.safety_checks if s.metric_name == "false_automation")
        assert fa_check.passed is True

    def test_report_lineage(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            current_model_id="MOD-CURRENT",
            current_model_version="1.0.0",
            candidate_model_id="MOD-CANDIDATE",
            candidate_model_version="2.0.0",
            dataset_version="1.0.0",
            feature_schema_version="1.0.0",
            mlflow_run_id="RUN-123",
        )
        assert report.current_model_id == "MOD-CURRENT"
        assert report.current_model_version == "1.0.0"
        assert report.candidate_model_id == "MOD-CANDIDATE"
        assert report.candidate_model_version == "2.0.0"
        assert report.dataset_version == "1.0.0"
        assert report.feature_schema_version == "1.0.0"
        assert report.mlflow_run_id == "RUN-123"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Check Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyChecks:
    """Test safety regression detection."""

    def test_false_automation_increase_blocked(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
    ):
        # Candidate has MORE false automation
        bad_candidate = current_metrics.model_copy(
            update={"false_automation": 20, "accuracy": 0.90}
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=bad_candidate,
            candidate_model_id="MOD-BAD",
            candidate_model_version="2.0.0",
        )
        fa_check = next(s for s in report.safety_checks if s.metric_name == "false_automation")
        assert fa_check.passed is False
        assert fa_check.severity == SafetyRegressionSeverity.CRITICAL

    def test_high_value_error_increase_blocked(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
    ):
        bad_candidate = current_metrics.model_copy(
            update={"high_value_errors": 5, "accuracy": 0.90}
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=bad_candidate,
            candidate_model_id="MOD-BAD",
            candidate_model_version="2.0.0",
        )
        hv_check = next(s for s in report.safety_checks if s.metric_name == "high_value_errors")
        assert hv_check.passed is False

    def test_low_accuracy_blocked(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
    ):
        bad_candidate = current_metrics.model_copy(
            update={"accuracy": 0.30, "f1_macro": 0.25}
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=bad_candidate,
            candidate_model_id="MOD-BAD",
            candidate_model_version="2.0.0",
        )
        acc_check = next(s for s in report.safety_checks if s.metric_name == "accuracy")
        assert acc_check.passed is False

    def test_custom_thresholds(self):
        strict_service = UnifiedEvaluationService(
            thresholds=EvaluationThresholds(
                max_false_automation_increase=0,
                min_accuracy=0.90,
            )
        )
        current = EvaluationMetrics(
            model_id="M1", model_version="1.0",
            accuracy=0.85, f1_macro=0.80, false_automation=2,
        )
        candidate = EvaluationMetrics(
            model_id="M2", model_version="2.0",
            accuracy=0.88, f1_macro=0.83, false_automation=2,
        )
        report = strict_service.evaluate(
            current_metrics=current,
            candidate_metrics=candidate,
            candidate_model_id="M2",
            candidate_model_version="2.0",
        )
        # 0.88 < 0.90 → accuracy check fails
        acc_check = next(s for s in report.safety_checks if s.metric_name == "accuracy")
        assert acc_check.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Verdict Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerdict:
    """Test verdict determination."""

    def test_promote_on_improvement(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
        )
        assert report.verdict == EvaluationVerdict.PROMOTE
        assert report.promotion_eligible is True

    def test_reject_on_critical_safety(
        self, service: UnifiedEvaluationService,
        current_metrics: EvaluationMetrics,
    ):
        # Massive false automation increase
        bad = current_metrics.model_copy(
            update={"false_automation": 50, "accuracy": 0.95}
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=bad,
            candidate_model_id="MOD-BAD",
            candidate_model_version="2.0.0",
        )
        assert report.verdict == EvaluationVerdict.REJECT
        assert report.promotion_eligible is False

    def test_defer_on_equal(
        self, service: UnifiedEvaluationService,
    ):
        # Same metrics
        m = EvaluationMetrics(
            model_id="M1", model_version="1.0",
            accuracy=0.80, f1_macro=0.77, false_automation=5,
        )
        report = service.evaluate(
            current_metrics=m,
            candidate_metrics=m.model_copy(update={"model_id": "M2", "model_version": "2.0"}),
            candidate_model_id="M2",
            candidate_model_version="2.0",
        )
        assert report.verdict == EvaluationVerdict.DEFER
        assert report.promotion_eligible is False

    def test_first_model_promotes(
        self, service: UnifiedEvaluationService,
        candidate_metrics: EvaluationMetrics,
    ):
        report = service.evaluate(
            current_metrics=None,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-FIRST",
            candidate_model_version="1.0.0",
        )
        assert report.verdict == EvaluationVerdict.PROMOTE


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLflowIntegration:
    """Test logging evaluation reports to MLflow."""

    def test_log_to_mlflow(
        self, service: UnifiedEvaluationService,
        tracking_service: MLflowTrackingService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        tracking_service.create_experiment("eval-test")
        run = tracking_service.create_run(
            experiment_name="eval-test",
            model_type="test", model_name="test", model_version="2.0.0", algorithm="xgboost",
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-CANDIDATE",
            candidate_model_version="2.0.0",
            mlflow_run_id=run.run_id,
        )
        service.log_to_mlflow(report, tracking_service, run.run_id)

        # Verify metrics were logged
        history = tracking_service.get_metrics_history(run.run_id)
        assert len(history) == 1
        assert "eval.candidate_accuracy" in history[0]

        # Verify artifact was logged
        artifacts = tracking_service.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert "evaluation_report" in artifacts[0].artifact_name

    def test_log_with_explicit_run_id(
        self, service: UnifiedEvaluationService,
        tracking_service: MLflowTrackingService,
        current_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
    ):
        tracking_service.create_experiment("eval-explicit")
        run = tracking_service.create_run(
            experiment_name="eval-explicit",
            model_type="test", model_name="test", model_version="2.0.0", algorithm="xgboost",
        )
        report = service.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id="MOD-1",
            candidate_model_version="2.0.0",
        )
        # Pass run_id explicitly
        service.log_to_mlflow(report, tracking_service, run.run_id)
        history = tracking_service.get_metrics_history(run.run_id)
        assert len(history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestThresholdConfiguration:
    """Test configurable evaluation thresholds."""

    def test_default_thresholds(self):
        t = EvaluationThresholds()
        assert t.max_false_automation_increase == 0
        assert t.max_high_value_error_increase == 0
        assert t.min_accuracy == 0.50
        assert t.min_f1_macro == 0.40

    def test_custom_thresholds(self):
        t = EvaluationThresholds(
            max_false_automation_increase=5,
            min_accuracy=0.90,
        )
        assert t.max_false_automation_increase == 5
        assert t.min_accuracy == 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Enum Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEnums:
    """Test enum values."""

    def test_evaluation_verdict(self):
        assert EvaluationVerdict.PROMOTE.value == "PROMOTE"
        assert EvaluationVerdict.REJECT.value == "REJECT"
        assert EvaluationVerdict.DEFER.value == "DEFER"

    def test_safety_severity(self):
        assert SafetyRegressionSeverity.NONE.value == "NONE"
        assert SafetyRegressionSeverity.CRITICAL.value == "CRITICAL"
