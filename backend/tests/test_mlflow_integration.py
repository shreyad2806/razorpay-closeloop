"""
Tests for Razorpay CloseLoop Phase 10J — Complete MLflow Integration.

Verifies the end-to-end lifecycle:
  Training → MLflow Tracking → Evaluation → Model Registry →
  Prediction Lineage → Experiment Comparison → Outcome → Learning
"""

import pytest
from datetime import datetime, timezone

from app.schemas.mlflow_integration import (
    IntegrationPhase,
    IntegrationStatus,
    MLflowIntegrationSummary,
    MLflowLifecycleRecord,
    MLflowLifecycleStep,
    PredictAndTrackRequest,
    PredictAndTrackResult,
    TrainAndTrackRequest,
    TrainAndTrackResult,
)
from app.schemas.mlflow_tracking import (
    MetricsSnapshot,
    MLflowRunMetadata,
    RunStatus,
)
from app.schemas.mlflow_model_registry import ModelLifecycleState
from app.schemas.model_training import EvaluationMetrics
from app.services.mlflow_integration import MLflowIntegration  # noqa: F811


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def integration() -> MLflowIntegration:
    return MLflowIntegration()


@pytest.fixture
def train_request() -> TrainAndTrackRequest:
    return TrainAndTrackRequest(
        model_name="XGBoost Classifier",
        model_version="1.0.0",
        experiment_name="razorpay-closeloop.classification",
        algorithm="xgboost",
        dataset_id="dataset-001",
        dataset_version="v3",
        feature_schema_version="fv2",
        hyperparameters={"n_estimators": 100, "max_depth": 6},
    )


@pytest.fixture
def eval_metrics_good() -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id="model-001",
        model_version="1.0.0",
        accuracy=0.90,
        precision_macro=0.88,
        recall_macro=0.85,
        f1_macro=0.86,
        total_samples=100,
        false_automation=1,
        high_value_errors=0,
        incorrect_auto_resolution=1,
        verification_failure_rate=0.02,
        resolution_accuracy=0.87,
    )


@pytest.fixture
def eval_metrics_baseline() -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id="model-000",
        model_version="0.9.0",
        accuracy=0.85,
        precision_macro=0.83,
        recall_macro=0.80,
        f1_macro=0.81,
        total_samples=100,
        false_automation=2,
        high_value_errors=0,
        incorrect_auto_resolution=2,
        verification_failure_rate=0.03,
        resolution_accuracy=0.82,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle Record Schema
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycleRecordSchema:
    def test_step_creation(self):
        step = MLflowLifecycleStep(
            phase=IntegrationPhase.TRAINING,
            status=IntegrationStatus.COMPLETED,
            timestamp=datetime.now(timezone.utc),
            details={"model_id": "m-1"},
        )
        assert step.phase == IntegrationPhase.TRAINING
        assert step.status == IntegrationStatus.COMPLETED

    def test_record_get_step(self):
        record = MLflowLifecycleRecord(
            record_id="LCY-001",
            model_id="m-1",
            model_version="1.0",
            steps=[
                MLflowLifecycleStep(
                    phase=IntegrationPhase.TRAINING,
                    status=IntegrationStatus.COMPLETED,
                ),
                MLflowLifecycleStep(
                    phase=IntegrationPhase.TRACKING,
                    status=IntegrationStatus.COMPLETED,
                ),
            ],
        )
        step = record.get_step(IntegrationPhase.TRAINING)
        assert step is not None
        assert step.status == IntegrationStatus.COMPLETED
        assert record.get_step(IntegrationPhase.EVALUATION) is None

    def test_record_summary(self):
        record = MLflowLifecycleRecord(
            record_id="LCY-001",
            model_id="m-1",
            model_version="1.0",
            mlflow_run_id="run-ABCDEF123456",
        )
        s = record.summary()
        assert "1.0" in s
        # summary() truncates run_id to 8 chars: run-ABCD
        assert "run-ABCD" in s


# ─────────────────────────────────────────────────────────────────────────────
# Train and Track
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainAndTrack:
    def test_basic_train_and_track(self, integration, train_request):
        result = integration.train_and_track(train_request)
        assert result.lifecycle_record_id.startswith("LCY-")
        assert result.model_id.startswith("MOD-")
        assert result.model_version == "1.0.0"
        assert result.mlflow_run_id.startswith("RUN-")
        assert result.parameters_logged is True

    def test_experiment_created(self, integration, train_request):
        integration.train_and_track(train_request)
        experiments = list(integration.tracking.list_experiments().keys())
        # Experiment is stored with tags_prefix (closeloop.)
        assert any("classification" in e for e in experiments)

    def test_run_created(self, integration, train_request):
        result = integration.train_and_track(train_request)
        run = integration.tracking.get_run(result.mlflow_run_id)
        assert run is not None
        assert run.model_name == "XGBoost Classifier"
        assert run.algorithm == "xgboost"
        assert run.dataset_version == "v3"

    def test_version_registry_recorded(self, integration, train_request):
        result = integration.train_and_track(train_request)
        meta = integration.version_registry.get_version(result.model_id)
        assert meta is not None
        assert meta.mlflow_run_id == result.mlflow_run_id
        assert meta.dataset_version == "v3"
        assert meta.feature_schema_version == "fv2"

    def test_registry_candidate(self, integration, train_request):
        result = integration.train_and_track(train_request)
        entry = integration.registry.get_model(result.model_id)
        assert entry is not None
        assert entry.state == ModelLifecycleState.CANDIDATE

    def test_lifecycle_record_created(self, integration, train_request):
        result = integration.train_and_track(train_request)
        rec = integration.get_lifecycle_record(result.lifecycle_record_id)
        assert rec is not None
        assert rec.model_id == result.model_id
        assert len(rec.steps) == 2
        assert rec.steps[0].phase == IntegrationPhase.TRAINING
        assert rec.steps[1].phase == IntegrationPhase.TRACKING

    def test_lifecycle_by_model_lookup(self, integration, train_request):
        result = integration.train_and_track(train_request)
        rec = integration.get_lifecycle_by_model(result.model_id)
        assert rec is not None
        assert rec.mlflow_run_id == result.mlflow_run_id

    def test_custom_model_id(self, integration, train_request):
        result = integration.train_and_track(train_request, model_id="custom-id")
        assert result.model_id == "custom-id"


# ─────────────────────────────────────────────────────────────────────────────
# Log Training Metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestLogTrainingMetrics:
    def test_log_metrics(self, integration, train_request, eval_metrics_good):
        result = integration.train_and_track(train_request)
        snapshot = MetricsSnapshot(
            run_id=result.mlflow_run_id,
            accuracy=eval_metrics_good.accuracy,
            f1_macro=eval_metrics_good.f1_macro,
            false_automation=eval_metrics_good.false_automation,
        )
        integration.log_training_metrics(result.model_id, snapshot)

        history = integration.tracking.get_metrics_history(result.mlflow_run_id)
        assert len(history) >= 1
        assert history[-1].get("accuracy") == 0.90

    def test_log_metrics_updates_lifecycle(self, integration, train_request):
        result = integration.train_and_track(train_request)
        snapshot = MetricsSnapshot(
            run_id=result.mlflow_run_id,
            accuracy=0.88,
        )
        integration.log_training_metrics(result.model_id, snapshot)

        rec = integration.get_lifecycle_by_model(result.model_id)
        assert rec is not None
        eval_steps = [s for s in rec.steps if s.phase == IntegrationPhase.EVALUATION]
        assert len(eval_steps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate and Track
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateAndTrack:
    def test_evaluate_no_current(self, integration, train_request, eval_metrics_good):
        result = integration.train_and_track(train_request)
        output = integration.evaluate_and_track(
            model_id=result.model_id,
            evaluation_metrics=eval_metrics_good,
        )
        report = output["report"]
        assert report.candidate_model_id == result.model_id
        assert report.all_safety_passed is True

    def test_evaluate_with_current(
        self, integration, train_request, eval_metrics_good, eval_metrics_baseline
    ):
        result = integration.train_and_track(train_request)
        output = integration.evaluate_and_track(
            model_id=result.model_id,
            evaluation_metrics=eval_metrics_good,
            current_model_id="baseline-model",
            current_metrics=eval_metrics_baseline,
        )
        report = output["report"]
        assert report.total_improvements > 0

    def test_evaluation_logged_to_mlflow(
        self, integration, train_request, eval_metrics_good
    ):
        result = integration.train_and_track(train_request)
        integration.evaluate_and_track(
            model_id=result.model_id,
            evaluation_metrics=eval_metrics_good,
        )
        artifacts = integration.tracking.get_artifacts(result.mlflow_run_id)
        eval_artifacts = [
            a for a in artifacts
            if "evaluation" in a.artifact_name.lower()
        ]
        assert len(eval_artifacts) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Promote Model
# ─────────────────────────────────────────────────────────────────────────────


class TestPromoteModel:
    def test_full_lifecycle_to_production(self, integration, train_request):
        result = integration.train_and_track(train_request)
        output = integration.promote_model(
            model_id=result.model_id,
            evaluation_verdict="PROMOTE",
            accuracy=0.90,
            f1_macro=0.88,
            false_automation=0,
            high_value_errors=0,
            reason="All safety checks passed",
        )
        assert output["new_state"] == "PRODUCTION"

        entry = integration.registry.get_model(result.model_id)
        assert entry.state == ModelLifecycleState.PRODUCTION

    def test_validation_failure(self, integration, train_request):
        result = integration.train_and_track(train_request)
        # Register with metrics that fail validation (low accuracy)
        entry = integration.registry.get_model(result.model_id)
        entry.accuracy = 0.30  # Below min_accuracy
        entry.f1_macro = 0.25  # Below min_f1_macro

        output = integration.promote_model(model_id=result.model_id)
        assert output["new_state"] == "ARCHIVED"

    def test_promotion_updates_lifecycle(self, integration, train_request):
        result = integration.train_and_track(train_request)
        integration.promote_model(
            model_id=result.model_id,
            evaluation_verdict="PROMOTE",
            accuracy=0.90,
            f1_macro=0.88,
        )
        rec = integration.get_lifecycle_by_model(result.model_id)
        assert rec is not None
        reg_steps = [s for s in rec.steps if s.phase == IntegrationPhase.REGISTRY]
        assert len(reg_steps) >= 1
        assert rec.registry_state == "PRODUCTION"

    def test_promotion_updates_version_registry(self, integration, train_request):
        result = integration.train_and_track(train_request)
        integration.promote_model(
            model_id=result.model_id,
            evaluation_verdict="PROMOTE",
            accuracy=0.90,
            f1_macro=0.88,
        )
        meta = integration.version_registry.get_version(result.model_id)
        assert meta is not None
        from app.schemas.model_version_metadata import ModelVersionStatus
        assert meta.status == ModelVersionStatus.ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# Predict and Track
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictAndTrack:
    def test_basic_prediction(self, integration, train_request):
        result = integration.train_and_track(train_request)
        pred = integration.predict_and_track(PredictAndTrackRequest(
            exception_id="exc-001",
            model_id=result.model_id,
            prediction="FEE_DIFFERENCE",
            confidence=0.92,
            policy_version="policy-v2",
        ))
        assert pred.result_id.startswith("RESULT-")
        assert pred.model_version == "1.0.0"
        assert pred.prediction == "FEE_DIFFERENCE"
        assert pred.confidence == 0.92

    def test_prediction_lineage(self, integration, train_request):
        result = integration.train_and_track(train_request)
        pred = integration.predict_and_track(PredictAndTrackRequest(
            exception_id="exc-001",
            model_id=result.model_id,
            prediction="FEE_DIFFERENCE",
            confidence=0.92,
        ))
        # Verify full lineage chain
        chain = integration.result_lineage.build_lineage_chain(pred.result_id)
        assert chain is not None
        assert chain.mlflow_run_id == result.mlflow_run_id
        assert chain.dataset_version == "v3"
        assert chain.algorithm == "xgboost"
        assert chain.prediction == "FEE_DIFFERENCE"

    def test_prediction_updates_lifecycle(self, integration, train_request):
        result = integration.train_and_track(train_request)
        integration.predict_and_track(PredictAndTrackRequest(
            exception_id="exc-001",
            model_id=result.model_id,
        ))
        rec = integration.get_lifecycle_by_model(result.model_id)
        pred_steps = [s for s in rec.steps if s.phase == IntegrationPhase.PREDICTION]
        assert len(pred_steps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Compare Model Runs
# ─────────────────────────────────────────────────────────────────────────────


class TestCompareRuns:
    def test_compare_two_runs(self, integration):
        # Create two runs with different metrics
        meta1 = integration.tracking.create_run(
            experiment_name="test-exp",
            model_type="classifier",
            model_name="XGB",
            model_version="v1.0",
            algorithm="xgboost",
        )
        meta2 = integration.tracking.create_run(
            experiment_name="test-exp",
            model_type="classifier",
            model_name="XGB",
            model_version="v2.0",
            algorithm="xgboost",
        )

        # Log different metrics
        snap1 = MetricsSnapshot(run_id=meta1.run_id, accuracy=0.85, f1_macro=0.83)
        snap2 = MetricsSnapshot(run_id=meta2.run_id, accuracy=0.92, f1_macro=0.90)
        integration.tracking.log_metrics(meta1.run_id, snap1)
        integration.tracking.log_metrics(meta2.run_id, snap2)

        result = integration.compare_model_runs([meta1.run_id, meta2.run_id])
        assert "comparison" in result
        assert result["run_count"] == 2

    def test_compare_insufficient_runs(self, integration):
        result = integration.compare_model_runs(["run-only-one"])
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────────────


class TestAudit:
    def test_audit_exception(self, integration, train_request):
        result = integration.train_and_track(train_request)
        integration.predict_and_track(PredictAndTrackRequest(
            exception_id="exc-001",
            model_id=result.model_id,
            prediction="FEE_DIFFERENCE",
            confidence=0.92,
        ))
        audit = integration.audit_exception("exc-001")
        assert audit is not None
        assert audit.model_version == "1.0.0"
        assert audit.mlflow_run_id == result.mlflow_run_id
        assert audit.dataset_version == "v3"

    def test_audit_exception_not_found(self, integration):
        assert integration.audit_exception("nonexistent") is None

    def test_audit_model(self, integration, train_request):
        result = integration.train_and_track(train_request)
        audit = integration.audit_model(result.model_id)
        assert audit is not None
        assert "model" in audit
        assert "registry" in audit
        assert "lifecycle" in audit

    def test_audit_model_not_found(self, integration):
        assert integration.audit_model("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration Summary
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegrationSummary:
    def test_empty_summary(self, integration):
        summary = integration.get_integration_summary()
        assert summary.total_lifecycle_records == 0
        assert summary.safety_boundary == "ENFORCED"

    def test_populated_summary(self, integration, train_request):
        integration.train_and_track(train_request)
        integration.train_and_track(train_request)  # Second model
        summary = integration.get_integration_summary()
        assert summary.total_lifecycle_records == 2
        assert summary.total_models_tracked == 2
        assert summary.registry_models == 2


# ─────────────────────────────────────────────────────────────────────────────
# Complete End-to-End Lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestCompleteLifecycle:
    def test_training_to_prediction(
        self, integration, train_request, eval_metrics_good
    ):
        """Test the complete flow: train → evaluate → promote → predict → audit."""
        # 1. Train and track
        train_result = integration.train_and_track(train_request)
        assert train_result.parameters_logged

        # 2. Log metrics
        snapshot = MetricsSnapshot(
            run_id=train_result.mlflow_run_id,
            accuracy=0.90,
            f1_macro=0.88,
            false_automation=1,
            high_value_errors=0,
        )
        integration.log_training_metrics(train_result.model_id, snapshot)

        # 3. Evaluate
        eval_output = integration.evaluate_and_track(
            model_id=train_result.model_id,
            evaluation_metrics=eval_metrics_good,
        )
        assert eval_output["report"].all_safety_passed

        # 4. Promote
        promo_output = integration.promote_model(
            model_id=train_result.model_id,
            evaluation_verdict="PROMOTE",
            accuracy=0.90,
            f1_macro=0.88,
        )
        assert promo_output["new_state"] == "PRODUCTION"

        # 5. Predict
        pred_result = integration.predict_and_track(PredictAndTrackRequest(
            exception_id="exc-001",
            model_id=train_result.model_id,
            prediction="FEE_DIFFERENCE",
            confidence=0.92,
            policy_version="policy-v2",
        ))
        assert pred_result.result_id.startswith("RESULT-")

        # 6. Audit
        audit = integration.audit_exception("exc-001")
        assert audit is not None
        assert audit.model_version == "1.0.0"
        assert audit.mlflow_run_id == train_result.mlflow_run_id
        assert audit.prediction == "FEE_DIFFERENCE"

        # 7. Full lineage chain
        chain = integration.result_lineage.build_lineage_chain(pred_result.result_id)
        assert chain is not None
        assert chain.mlflow_run_id == train_result.mlflow_run_id
        assert chain.dataset_version == "v3"
        assert chain.algorithm == "xgboost"
        assert chain.prediction == "FEE_DIFFERENCE"

        # 8. Lifecycle record complete
        rec = integration.get_lifecycle_by_model(train_result.model_id)
        assert rec is not None
        assert rec.registry_state == "PRODUCTION"
        assert len(rec.steps) >= 4  # training + tracking + evaluation + registry + prediction

    def test_phase_6_not_bypassed(
        self, integration, train_request, eval_metrics_good
    ):
        """Verify MLflow integration never bypasses Phase 6."""
        result = integration.train_and_track(train_request)

        # Run evaluation — it uses safety checks
        eval_output = integration.evaluate_and_track(
            model_id=result.model_id,
            evaluation_metrics=eval_metrics_good,
        )
        report = eval_output["report"]
        # Safety checks must have been run
        assert len(report.safety_checks) > 0
        # All safety checks are recorded
        for check in report.safety_checks:
            assert hasattr(check, 'passed')
            assert hasattr(check, 'severity')

    def test_two_models_promote_second(
        self, integration, train_request, eval_metrics_good, eval_metrics_baseline
    ):
        """Train two models, promote first, then evaluate second against it."""
        # Train first model
        result1 = integration.train_and_track(train_request)
        integration.promote_model(
            model_id=result1.model_id,
            evaluation_verdict="PROMOTE",
            accuracy=eval_metrics_baseline.accuracy,
            f1_macro=eval_metrics_baseline.f1_macro,
        )

        # Train second model
        result2 = integration.train_and_track(train_request)
        eval_output = integration.evaluate_and_track(
            model_id=result2.model_id,
            evaluation_metrics=eval_metrics_good,
            current_model_id=result1.model_id,
            current_metrics=eval_metrics_baseline,
        )
        report = eval_output["report"]
        assert report.total_improvements > 0


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_integration_observational(self, integration):
        """Integration service cannot authorize execution."""
        assert not hasattr(integration, 'execute_refund')
        assert not hasattr(integration, 'modify_settlement')
        assert not hasattr(integration, 'authorize_payment')

    def test_safety_boundary_enforced(self, integration):
        summary = integration.get_integration_summary()
        assert summary.safety_boundary == "ENFORCED"

    def test_registry_cannot_bypass_phase6(self, integration, train_request):
        """Registry lifecycle management does not bypass Phase 6."""
        result = integration.train_and_track(train_request)
        # Even if we call promote, it goes through gates
        output = integration.promote_model(
            model_id=result.model_id,
            evaluation_verdict="REJECT",  # Bad verdict
            accuracy=0.30,  # Low accuracy
            f1_macro=0.25,
            false_automation=5,
            high_value_errors=2,
        )
        assert output["new_state"] in ("ARCHIVED", "CANDIDATE")  # NOT PRODUCTION
