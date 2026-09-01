"""
Tests for Razorpay CloseLoop Phase 10A — MLflow Experiment Tracking.

Tests MLflow setup, experiment creation, run tracking, and metadata persistence.
"""

import pytest

from app.schemas.mlflow_tracking import (
    EnvironmentMetadata,
    ExperimentSummary,
    ExperimentType,
    MLflowConfig,
    MLflowRunMetadata,
    MetricsSnapshot,
    RunStatus,
)
from app.services.mlflow_tracking import MLflowTrackingService, collect_environment_metadata


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> MLflowConfig:
    """Default MLflow config for testing."""
    return MLflowConfig(
        tracking_uri="file:./test_mlruns",
        experiment_name="test-experiment",
        tags_prefix="test.",
    )


@pytest.fixture
def service(config: MLflowConfig) -> MLflowTrackingService:
    """Fresh MLflow tracking service."""
    return MLflowTrackingService(config=config)


@pytest.fixture
def service_no_mlflow() -> MLflowTrackingService:
    """Service with invalid MLflow URI (forces in-memory fallback)."""
    return MLflowTrackingService(
        config=MLflowConfig(tracking_uri="file:./nonexistent_path_12345")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLflowConfig:
    """Test MLflow configuration."""

    def test_default_config(self):
        config = MLflowConfig()
        assert config.tracking_uri == "file:./mlruns"
        assert config.experiment_name == "razorpay-closeloop"
        assert config.tags_prefix == "closeloop."

    def test_custom_config(self):
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="my-experiment",
            artifact_root="/tmp/artifacts",
            tags_prefix="custom.",
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "my-experiment"
        assert config.artifact_root == "/tmp/artifacts"
        assert config.tags_prefix == "custom."

    def test_config_frozen(self):
        config = MLflowConfig()
        with pytest.raises(Exception):
            config.tracking_uri = "http://changed"

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-server:5000")
        config = MLflowConfig()
        # Config respects its own default; env override happens in service init
        assert config.tracking_uri == "file:./mlruns"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentType:
    """Test experiment type enum."""

    def test_exception_classification(self):
        assert ExperimentType.EXCEPTION_CLASSIFICATION.value == "exception_classification"

    def test_resolution_prediction(self):
        assert ExperimentType.RESOLUTION_PREDICTION.value == "resolution_prediction"

    def test_feedback_learning(self):
        assert ExperimentType.FEEDBACK_LEARNING.value == "feedback_learning"

    def test_policy_comparison(self):
        assert ExperimentType.POLICY_COMPARISON.value == "policy_comparison"


# ─────────────────────────────────────────────────────────────────────────────
# Run Status Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunStatus:
    """Test run status enum."""

    def test_all_statuses(self):
        statuses = [s.value for s in RunStatus]
        assert "RUNNING" in statuses
        assert "COMPLETED" in statuses
        assert "FAILED" in statuses
        assert "KILLED" in statuses


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Creation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentCreation:
    """Test experiment creation and management."""

    def test_create_experiment(self, service: MLflowTrackingService):
        exp_id = service.create_experiment("classification")
        assert exp_id is not None
        assert exp_id.startswith("EXP-")

    def test_create_experiment_with_type(self, service: MLflowTrackingService):
        exp_id = service.create_experiment(
            "resolution",
            experiment_type=ExperimentType.RESOLUTION_PREDICTION,
        )
        assert exp_id is not None

    def test_create_experiment_with_tags(self, service: MLflowTrackingService):
        exp_id = service.create_experiment(
            "tagged",
            tags={"owner": "ml-team", "purpose": "testing"},
        )
        assert exp_id is not None

    def test_get_existing_experiment(self, service: MLflowTrackingService):
        exp_id1 = service.create_experiment("existing")
        exp_id2 = service.create_experiment("existing")
        # Same experiment ID returned
        assert exp_id1 == exp_id2

    def test_get_experiment_id(self, service: MLflowTrackingService):
        exp_id = service.create_experiment("lookup")
        found = service.get_experiment_id("lookup")
        assert found == exp_id

    def test_get_nonexistent_experiment(self, service: MLflowTrackingService):
        found = service.get_experiment_id("no-such-experiment")
        assert found is None

    def test_list_experiments_empty(self, service: MLflowTrackingService):
        experiments = service.list_experiments()
        assert experiments == {}

    def test_list_experiments(self, service: MLflowTrackingService):
        service.create_experiment("exp1")
        service.create_experiment("exp2")
        experiments = service.list_experiments()
        assert len(experiments) == 2

    def test_experiment_prefix(self, service: MLflowTrackingService):
        exp_id = service.create_experiment("prefixed")
        # Should be stored with prefix
        experiments = service.list_experiments()
        assert any("prefixed" in k for k in experiments.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Run Creation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunCreation:
    """Test run creation and lifecycle."""

    def test_create_run(self, service: MLflowTrackingService):
        service.create_experiment("test")
        run = service.create_run(
            experiment_name="test",
            model_type="exception_classifier",
            model_name="xgb-classifier",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        assert run.run_id is not None
        assert run.run_id.startswith("RUN-")
        assert run.status == RunStatus.RUNNING
        assert run.model_type == "exception_classifier"
        assert run.model_name == "xgb-classifier"
        assert run.model_version == "1.0.0"
        assert run.algorithm == "xgboost"

    def test_create_run_with_details(self, service: MLflowTrackingService):
        service.create_experiment("detailed")
        run = service.create_run(
            experiment_name="detailed",
            model_type="resolution_predictor",
            model_name="lr-classifier",
            model_version="2.1.0",
            algorithm="logistic_regression",
            run_name="batch-1-training",
            dataset_id="DS-001",
            dataset_version="1.2.0",
            feature_schema_version="1.0.0",
            training_examples=500,
            validation_examples=100,
            test_examples=100,
            total_examples=700,
            feature_count=12,
            random_seed=42,
            label_classes=["FEE_DIFFERENCE", "EXACT_MATCH"],
        )
        assert run.run_name == "batch-1-training"
        assert run.dataset_id == "DS-001"
        assert run.dataset_version == "1.2.0"
        assert run.feature_schema_version == "1.0.0"
        assert run.training_examples == 500
        assert run.validation_examples == 100
        assert run.test_examples == 100
        assert run.total_examples == 700
        assert run.feature_count == 12
        assert run.random_seed == 42
        assert run.label_classes == ["FEE_DIFFERENCE", "EXACT_MATCH"]

    def test_create_run_with_hyperparameters(self, service: MLflowTrackingService):
        service.create_experiment("hyper")
        run = service.create_run(
            experiment_name="hyper",
            model_type="exception_classifier",
            model_name="xgb",
            model_version="1.0.0",
            algorithm="xgboost",
            hyperparameters={"max_depth": 6, "learning_rate": 0.1},
            training_config={"train_ratio": 0.7},
        )
        assert run.hyperparameters == {"max_depth": 6, "learning_rate": 0.1}
        assert run.training_config == {"train_ratio": 0.7}

    def test_create_run_with_algorithm_params(self, service: MLflowTrackingService):
        service.create_experiment("algo-params")
        run = service.create_run(
            experiment_name="algo-params",
            model_type="exception_classifier",
            model_name="xgb",
            model_version="1.0.0",
            algorithm="xgboost",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            class_weight="balanced",
            early_stopping_rounds=15,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        assert run.n_estimators == 200
        assert run.max_depth == 8
        assert run.learning_rate == 0.05
        assert run.subsample == 0.8
        assert run.colsample_bytree == 0.9
        assert run.class_weight == "balanced"
        assert run.early_stopping_rounds == 15
        assert run.train_ratio == 0.7
        assert run.val_ratio == 0.15
        assert run.test_ratio == 0.15

    def test_create_run_generates_unique_ids(self, service: MLflowTrackingService):
        service.create_experiment("unique")
        runs = []
        for _ in range(5):
            run = service.create_run(
                experiment_name="unique",
                model_type="test",
                model_name="test",
                model_version="1.0.0",
                algorithm="test",
            )
            runs.append(run)
        run_ids = [r.run_id for r in runs]
        assert len(set(run_ids)) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Run Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunLifecycle:
    """Test run completion and retrieval."""

    def test_complete_run(self, service: MLflowTrackingService):
        service.create_experiment("lifecycle")
        run = service.create_run(
            experiment_name="lifecycle",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        completed = service.complete_run(run.run_id, status=RunStatus.COMPLETED)
        assert completed.status == RunStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.duration_seconds is not None
        assert completed.duration_seconds >= 0

    def test_fail_run(self, service: MLflowTrackingService):
        service.create_experiment("fail")
        run = service.create_run(
            experiment_name="fail",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        failed = service.complete_run(
            run.run_id, status=RunStatus.FAILED, error_message="Out of memory"
        )
        assert failed.status == RunStatus.FAILED
        assert failed.error_message == "Out of memory"

    def test_complete_nonexistent_run(self, service: MLflowTrackingService):
        with pytest.raises(ValueError, match="not found"):
            service.complete_run("RUN-FAKE-123")

    def test_get_run(self, service: MLflowTrackingService):
        service.create_experiment("get")
        run = service.create_run(
            experiment_name="get",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        fetched = service.get_run(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id

    def test_get_nonexistent_run(self, service: MLflowTrackingService):
        fetched = service.get_run("RUN-FAKE-999")
        assert fetched is None

    def test_list_runs_empty(self, service: MLflowTrackingService):
        runs = service.list_runs()
        assert runs == []

    def test_list_runs(self, service: MLflowTrackingService):
        service.create_experiment("list")
        service.create_run(
            experiment_name="list",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        service.create_run(
            experiment_name="list",
            model_type="test",
            model_name="test2",
            model_version="2.0.0",
            algorithm="decision_tree",
        )
        runs = service.list_runs()
        assert len(runs) == 2

    def test_list_runs_filtered(self, service: MLflowTrackingService):
        service.create_experiment("filter1")
        service.create_experiment("filter2")
        service.create_run(
            experiment_name="filter1",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        service.create_run(
            experiment_name="filter2",
            model_type="test",
            model_name="test2",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        runs = service.list_runs(experiment_name="filter1")
        assert len(runs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Logging Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsLogging:
    """Test metric logging and retrieval."""

    def test_log_metrics(self, service: MLflowTrackingService):
        service.create_experiment("metrics")
        run = service.create_run(
            experiment_name="metrics",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        snapshot = MetricsSnapshot(
            run_id=run.run_id,
            accuracy=0.85,
            precision_macro=0.83,
            recall_macro=0.82,
            f1_macro=0.825,
        )
        service.log_metrics(run.run_id, snapshot)

        history = service.get_metrics_history(run.run_id)
        assert len(history) == 1
        assert history[0]["accuracy"] == 0.85
        assert history[0]["precision_macro"] == 0.83

    def test_log_safety_metrics(self, service: MLflowTrackingService):
        service.create_experiment("safety")
        run = service.create_run(
            experiment_name="safety",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        snapshot = MetricsSnapshot(
            run_id=run.run_id,
            accuracy=0.90,
            false_automation=2,
            high_value_errors=0,
            unknown_case_errors=1,
            verification_failure_rate=0.05,
        )
        service.log_metrics(run.run_id, snapshot)

        history = service.get_metrics_history(run.run_id)
        assert history[0]["false_automation"] == 2
        assert history[0]["high_value_errors"] == 0
        assert history[0]["verification_failure_rate"] == 0.05

    def test_log_custom_metrics(self, service: MLflowTrackingService):
        service.create_experiment("custom")
        run = service.create_run(
            experiment_name="custom",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        snapshot = MetricsSnapshot(
            run_id=run.run_id,
            accuracy=0.85,
            custom_metrics={"reward_avg": 0.6, "automation_rate": 0.4},
        )
        service.log_metrics(run.run_id, snapshot)

        history = service.get_metrics_history(run.run_id)
        assert history[0]["reward_avg"] == 0.6
        assert history[0]["automation_rate"] == 0.4

    def test_multiple_metric_snapshots(self, service: MLflowTrackingService):
        service.create_experiment("multi")
        run = service.create_run(
            experiment_name="multi",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        for i in range(3):
            snapshot = MetricsSnapshot(
                run_id=run.run_id,
                accuracy=0.80 + i * 0.05,
            )
            service.log_metrics(run.run_id, snapshot)

        history = service.get_metrics_history(run.run_id)
        assert len(history) == 3
        assert history[0]["accuracy"] == pytest.approx(0.80)
        assert history[2]["accuracy"] == pytest.approx(0.90)

    def test_log_metrics_nonexistent_run(self, service: MLflowTrackingService):
        snapshot = MetricsSnapshot(run_id="RUN-FAKE", accuracy=0.5)
        with pytest.raises(ValueError, match="not found"):
            service.log_metrics("RUN-FAKE", snapshot)

    def test_log_params(self, service: MLflowTrackingService):
        service.create_experiment("params")
        run = service.create_run(
            experiment_name="params",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        # Should not raise
        service.log_params(run.run_id, {"key1": "val1", "key2": "val2"})

    def test_log_tag(self, service: MLflowTrackingService):
        service.create_experiment("tags")
        run = service.create_run(
            experiment_name="tags",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        # Should not raise
        service.log_tag(run.run_id, "closeloop.model_id", "MOD-123")


# ─────────────────────────────────────────────────────────────────────────────
# Model-to-Run Linking Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRunLinking:
    """Test linking models to training runs."""

    def test_link_model_to_run(self, service: MLflowTrackingService):
        service.create_experiment("link")
        run = service.create_run(
            experiment_name="link",
            model_type="exception_classifier",
            model_name="xgb",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        service.link_model_to_run(run.run_id, "MOD-123", "1.0.0")
        # Should not raise

    def test_link_nonexistent_run(self, service: MLflowTrackingService):
        with pytest.raises(ValueError, match="not found"):
            service.link_model_to_run("RUN-FAKE", "MOD-123", "1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentSummary:
    """Test experiment summary generation."""

    def test_empty_experiment(self, service: MLflowTrackingService):
        service.create_experiment("empty-summary")
        summary = service.get_experiment_summary("empty-summary")
        assert summary.run_count == 0
        assert summary.completed_run_count == 0

    def test_experiment_with_runs(self, service: MLflowTrackingService):
        service.create_experiment("summary-runs")
        # Create and complete a run
        run = service.create_run(
            experiment_name="summary-runs",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        snapshot = MetricsSnapshot(run_id=run.run_id, accuracy=0.85)
        service.log_metrics(run.run_id, snapshot)
        service.complete_run(run.run_id, RunStatus.COMPLETED)

        summary = service.get_experiment_summary("summary-runs")
        assert summary.run_count == 1
        assert summary.completed_run_count == 1
        assert summary.best_run_id == run.run_id
        assert summary.best_metric == "accuracy"
        assert summary.best_metric_value == pytest.approx(0.85)

    def test_experiment_with_mixed_statuses(self, service: MLflowTrackingService):
        service.create_experiment("mixed")
        # Completed run
        r1 = service.create_run(
            experiment_name="mixed",
            model_type="test", model_name="t1", model_version="1.0.0", algorithm="xgboost",
        )
        service.complete_run(r1.run_id, RunStatus.COMPLETED)
        # Failed run
        r2 = service.create_run(
            experiment_name="mixed",
            model_type="test", model_name="t2", model_version="2.0.0", algorithm="xgboost",
        )
        service.complete_run(r2.run_id, RunStatus.FAILED, error_message="OOM")

        summary = service.get_experiment_summary("mixed")
        assert summary.run_count == 2
        assert summary.completed_run_count == 1
        assert summary.failed_run_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# MetricsSnapshot Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsSnapshot:
    """Test MetricsSnapshot schema."""

    def test_to_mlflow_dict(self):
        snapshot = MetricsSnapshot(
            run_id="RUN-123",
            accuracy=0.85,
            f1_macro=0.82,
            false_automation=2,
        )
        d = snapshot.to_mlflow_dict()
        assert d["accuracy"] == 0.85
        assert d["f1_macro"] == 0.82
        assert d["false_automation"] == 2
        assert "run_id" not in d
        assert "logged_at" not in d

    def test_to_mlflow_dict_excludes_none(self):
        snapshot = MetricsSnapshot(
            run_id="RUN-123",
            accuracy=0.85,
            precision_macro=None,
        )
        d = snapshot.to_mlflow_dict()
        assert "accuracy" in d
        assert "precision_macro" not in d

    def test_to_mlflow_dict_with_custom(self):
        snapshot = MetricsSnapshot(
            run_id="RUN-123",
            accuracy=0.85,
            custom_metrics={"my_metric": 1.23},
        )
        d = snapshot.to_mlflow_dict()
        assert d["my_metric"] == 1.23


# ─────────────────────────────────────────────────────────────────────────────
# MLflowRunMetadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLflowRunMetadata:
    """Test run metadata schema."""

    def test_summary(self):
        metadata = MLflowRunMetadata(
            run_id="RUN-ABCDEFGHIJKLM",
            experiment_name="test-exp",
            model_type="test",
            model_name="xgb",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        summary = metadata.summary()
        assert "RUN-ABCD" in summary
        assert "xgb" in summary
        assert "1.0.0" in summary

    def test_defaults(self):
        metadata = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="test-exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="test",
        )
        assert metadata.status == RunStatus.RUNNING
        assert metadata.environment == "local"
        assert metadata.training_config == {}
        assert metadata.hyperparameters == {}


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentSummary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentSummarySchema:
    """Test ExperimentSummary schema."""

    def test_summary(self):
        summary = ExperimentSummary(
            experiment_name="test",
            experiment_type=ExperimentType.EXCEPTION_CLASSIFICATION,
            run_count=5,
            completed_run_count=4,
            best_run_id="RUN-BEST1234",
        )
        s = summary.summary()
        assert "test" in s
        assert "5" in s

    def test_summary_no_best(self):
        summary = ExperimentSummary(
            experiment_name="empty",
            experiment_type=ExperimentType.EXCEPTION_CLASSIFICATION,
        )
        s = summary.summary()
        assert "none" in s


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInMemoryFallback:
    """Test behavior when MLflow server is unavailable."""

    def test_fallback_still_works(self):
        service = MLflowTrackingService(
            config=MLflowConfig(tracking_uri="file:./nonexistent_12345")
        )
        # Even without MLflow, operations should succeed (in-memory)
        exp_id = service.create_experiment("fallback-test")
        assert exp_id is not None

    def test_fallback_runs(self):
        service = MLflowTrackingService(
            config=MLflowConfig(tracking_uri="file:./nonexistent_12345")
        )
        service.create_experiment("fallback")
        run = service.create_run(
            experiment_name="fallback",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        assert run.run_id is not None
        service.complete_run(run.run_id, RunStatus.COMPLETED)
        assert service.get_run(run.run_id).status == RunStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    """Test that repeated operations are idempotent."""

    def test_duplicate_experiment_creation(self, service: MLflowTrackingService):
        id1 = service.create_experiment("idem")
        id2 = service.create_experiment("idem")
        assert id1 == id2

    def test_duplicate_get_experiment(self, service: MLflowTrackingService):
        service.create_experiment("idem2")
        id1 = service.get_experiment_id("idem2")
        id2 = service.get_experiment_id("idem2")
        assert id1 == id2


# ─────────────────────────────────────────────────────────────────────────────
# Environment Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvironmentMetadata:
    """Test environment metadata collection and schema."""

    def test_collect_environment_metadata(self):
        env = collect_environment_metadata()
        assert env.python_version is not None
        assert env.platform is not None
        assert env.collected_at is not None

    def test_python_version_format(self):
        env = collect_environment_metadata()
        parts = env.python_version.split(".")
        assert len(parts) == 3
        assert parts[0].isdigit()

    def test_git_info_collected(self):
        env = collect_environment_metadata()
        # Git info may or may not be available depending on environment
        # Just verify the fields exist and are the right type
        if env.git_commit is not None:
            assert isinstance(env.git_commit, str)
            assert len(env.git_commit) > 0
        if env.git_branch is not None:
            assert isinstance(env.git_branch, str)

    def test_package_versions(self):
        env = collect_environment_metadata()
        # NumPy should be available since it's a dependency
        assert env.numpy_version is not None
        # XGBoost should be available
        assert env.xgboost_version is not None

    def test_to_params_dict(self):
        env = EnvironmentMetadata(
            python_version="3.12.10",
            platform="Windows 10",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            numpy_version="2.5.2",
            key_dependency_versions={"fastapi": "0.141.1"},
        )
        d = env.to_params_dict()
        assert d["env.python_version"] == "3.12.10"
        assert d["env.platform"] == "Windows 10"
        assert d["env.git_commit"] == "abc123"
        assert d["env.git_branch"] == "main"
        assert d["env.git_dirty"] == "False"
        assert d["env.numpy_version"] == "2.5.2"
        assert d["env.dep.fastapi"] == "0.141.1"
        assert "collected_at" not in d

    def test_to_params_dict_excludes_none(self):
        env = EnvironmentMetadata(
            python_version="3.12.10",
            platform="Linux",
        )
        d = env.to_params_dict()
        assert "env.git_commit" not in d
        assert "env.hostname" not in d
        assert "env.collected_at" not in d


# ─────────────────────────────────────────────────────────────────────────────
# Run Metadata — New Fields Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMetadataNewFields:
    """Test new fields on MLflowRunMetadata for parameter tracking."""

    def test_split_fields(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            training_examples=700,
            validation_examples=150,
            test_examples=150,
            total_examples=1000,
        )
        assert meta.training_examples == 700
        assert meta.validation_examples == 150
        assert meta.test_examples == 150
        assert meta.total_examples == 1000

    def test_label_classes(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            label_classes=["A", "B", "C"],
        )
        assert meta.label_classes == ["A", "B", "C"]

    def test_environment_metadata(self):
        env = EnvironmentMetadata(python_version="3.12", platform="Linux")
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            environment_metadata=env,
        )
        assert meta.environment_metadata is not None
        assert meta.environment_metadata.python_version == "3.12"

    def test_algorithm_specific_fields(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            class_weight="balanced",
            early_stopping_rounds=15,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_features=50,
        )
        assert meta.n_estimators == 200
        assert meta.max_depth == 8
        assert meta.learning_rate == 0.05
        assert meta.subsample == 0.8
        assert meta.colsample_bytree == 0.9
        assert meta.class_weight == "balanced"
        assert meta.early_stopping_rounds == 15
        assert meta.train_ratio == 0.7
        assert meta.val_ratio == 0.15
        assert meta.test_ratio == 0.15
        assert meta.max_features == 50

    def test_target_label_version(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            target_label_version="2.0.0",
        )
        assert meta.target_label_version == "2.0.0"

    def test_new_fields_default_to_none(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        assert meta.validation_examples == 0
        assert meta.test_examples == 0
        assert meta.total_examples == 0
        assert meta.label_classes == []
        assert meta.target_label_version is None
        assert meta.n_estimators is None
        assert meta.max_depth is None
        assert meta.learning_rate is None
        assert meta.subsample is None
        assert meta.colsample_bytree is None
        assert meta.class_weight is None
        assert meta.environment_metadata is None


# ─────────────────────────────────────────────────────────────────────────────
# get_all_params Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetAllParams:
    """Test comprehensive parameter collection."""

    def test_basic_params(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="exception_classifier",
            model_name="xgb",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        params = meta.get_all_params()
        assert params["model_type"] == "exception_classifier"
        assert params["model_name"] == "xgb"
        assert params["model_version"] == "1.0.0"
        assert params["algorithm"] == "xgboost"
        assert params["training_examples"] == "0"
        assert params["feature_count"] == "0"
        assert params["environment"] == "local"

    def test_dataset_params(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            dataset_id="DS-001",
            dataset_version="1.2.0",
            feature_schema_version="1.0.0",
            training_examples=500,
            validation_examples=100,
            test_examples=100,
            total_examples=700,
            feature_count=12,
            label_classes=["A", "B"],
            target_label_version="2.0.0",
        )
        params = meta.get_all_params()
        assert params["dataset_id"] == "DS-001"
        assert params["dataset_version"] == "1.2.0"
        assert params["feature_schema_version"] == "1.0.0"
        assert params["training_examples"] == "500"
        assert params["validation_examples"] == "100"
        assert params["test_examples"] == "100"
        assert params["total_examples"] == "700"
        assert params["feature_count"] == "12"
        assert params["label_classes"] == "A,B"
        assert params["target_label_version"] == "2.0.0"

    def test_hyperparameters(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            hyperparameters={"max_depth": 6, "learning_rate": 0.1},
        )
        params = meta.get_all_params()
        assert params["hp.max_depth"] == "6"
        assert params["hp.learning_rate"] == "0.1"

    def test_algorithm_specific_params(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            class_weight="balanced",
            early_stopping_rounds=15,
            random_seed=42,
        )
        params = meta.get_all_params()
        assert params["n_estimators"] == "200"
        assert params["max_depth"] == "8"
        assert params["learning_rate"] == "0.05"
        assert params["subsample"] == "0.8"
        assert params["colsample_bytree"] == "0.9"
        assert params["class_weight"] == "balanced"
        assert params["early_stopping_rounds"] == "15"
        assert params["random_seed"] == "42"

    def test_split_params(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_features=50,
        )
        params = meta.get_all_params()
        assert params["train_ratio"] == "0.7"
        assert params["val_ratio"] == "0.15"
        assert params["test_ratio"] == "0.15"
        assert params["max_features"] == "50"

    def test_environment_metadata_in_params(self):
        env = EnvironmentMetadata(
            python_version="3.12.10",
            platform="Windows",
            numpy_version="2.5.2",
        )
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            environment_metadata=env,
        )
        params = meta.get_all_params()
        assert params["env.python_version"] == "3.12.10"
        assert params["env.platform"] == "Windows"
        assert params["env.numpy_version"] == "2.5.2"

    def test_none_fields_excluded(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
        )
        params = meta.get_all_params()
        assert "dataset_id" not in params
        assert "dataset_version" not in params
        assert "n_estimators" not in params
        assert "max_depth" not in params
        assert "env.python_version" not in params

    def test_training_config_non_hyperparameter(self):
        meta = MLflowRunMetadata(
            run_id="RUN-123",
            experiment_name="exp",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            training_config={"custom_setting": "value", "max_depth": 6},
            hyperparameters={"max_depth": 6},
        )
        params = meta.get_all_params()
        # max_depth is in hyperparameters, so it's logged as hp.max_depth
        # training_config.max_depth should NOT also appear as config.max_depth
        assert "config.custom_setting" in params
        assert params["config.custom_setting"] == "value"


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Logging Service Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParameterLoggingService:
    """Test log_all_parameters and get_parameters methods."""

    def test_get_parameters(self, service: MLflowTrackingService):
        service.create_experiment("get-params")
        run = service.create_run(
            experiment_name="get-params",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            training_examples=500,
            feature_count=12,
        )
        params = service.get_parameters(run.run_id)
        assert isinstance(params, dict)
        assert params["training_examples"] == "500"
        assert params["feature_count"] == "12"

    def test_log_all_parameters(self, service: MLflowTrackingService):
        service.create_experiment("log-params")
        run = service.create_run(
            experiment_name="log-params",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            n_estimators=200,
        )
        # Should not raise
        service.log_all_parameters(run.run_id)

    def test_log_all_parameters_nonexistent(self, service: MLflowTrackingService):
        with pytest.raises(ValueError, match="not found"):
            service.log_all_parameters("RUN-FAKE")

    def test_get_parameters_nonexistent(self, service: MLflowTrackingService):
        with pytest.raises(ValueError, match="not found"):
            service.get_parameters("RUN-FAKE")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReproducibility:
    """Test that runs contain enough info for reproducibility."""

    def test_run_has_all_reproducibility_fields(self, service: MLflowTrackingService):
        service.create_experiment("repro")
        run = service.create_run(
            experiment_name="repro",
            model_type="exception_classifier",
            model_name="xgb-classifier",
            model_version="1.0.0",
            algorithm="xgboost",
            dataset_version="1.2.0",
            feature_schema_version="1.0.0",
            training_examples=500,
            validation_examples=100,
            test_examples=100,
            feature_count=12,
            hyperparameters={"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
            random_seed=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        params = service.get_parameters(run.run_id)

        # Verify all reproducibility-critical params are present
        assert params["algorithm"] == "xgboost"
        assert params["random_seed"] == "42"
        assert params["dataset_version"] == "1.2.0"
        assert params["feature_schema_version"] == "1.0.0"
        assert params["training_examples"] == "500"
        assert params["n_estimators"] == "100"
        assert params["max_depth"] == "6"
        assert params["learning_rate"] == "0.1"
        assert params["train_ratio"] == "0.7"

    def test_environment_metadata_in_run(self, service: MLflowTrackingService):
        service.create_experiment("env-repro")
        run = service.create_run(
            experiment_name="env-repro",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            collect_env=True,
        )
        assert run.environment_metadata is not None
        assert run.environment_metadata.python_version is not None
        assert run.environment_metadata.platform is not None

    def test_collect_env_false(self, service: MLflowTrackingService):
        service.create_experiment("no-env")
        run = service.create_run(
            experiment_name="no-env",
            model_type="test",
            model_name="test",
            model_version="1.0.0",
            algorithm="xgboost",
            collect_env=False,
        )
        assert run.environment_metadata is None


# ─────────────────────────────────────────────────────────────────────────────
# Repeated Run Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRepeatedRuns:
    """Test that repeated runs create separate IDs."""

    def test_separate_runs_separate_ids(self, service: MLflowTrackingService):
        service.create_experiment("repeat")
        ids = set()
        for i in range(5):
            run = service.create_run(
                experiment_name="repeat",
                model_type="test",
                model_name="test",
                model_version=f"{i}.0.0",
                algorithm="xgboost",
            )
            ids.add(run.run_id)
        assert len(ids) == 5
