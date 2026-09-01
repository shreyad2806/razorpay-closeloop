"""
Tests for Razorpay CloseLoop Phase 10E — Model Version Metadata.

Tests model version registration, current/candidate tracking,
MLflow run linkage, and prediction lineage.
"""

import pytest

from app.schemas.model_version_metadata import (
    ModelVersionMetadata,
    ModelVersionStatus,
    ModelVersionSummary,
    PredictionLineage,
)
from app.services.model_version_registry import ModelVersionRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ModelVersionRegistry:
    return ModelVersionRegistry()


@pytest.fixture
def model_v1_args():
    return dict(
        model_id="MOD-001",
        model_name="xgb-classifier",
        model_version="1.0.0",
        mlflow_run_id="RUN-abc123",
        mlflow_experiment_name="razorpay-closeloop.classification",
        dataset_id="DS-001",
        dataset_version="1.0.0",
        feature_schema_version="1.0.0",
        algorithm="xgboost",
        training_config={"algorithm": "xgboost", "seed": 42},
        hyperparameters={"max_depth": 6, "learning_rate": 0.1},
        random_seed=42,
        feature_count=12,
        feature_names=["f1", "f2", "f3"],
        label_classes=["FEE_DIFF", "EXACT_MATCH"],
        training_examples=500,
        accuracy=0.85,
        f1_macro=0.82,
        false_automation=2,
        high_value_errors=0,
    )


@pytest.fixture
def model_v2_args():
    return dict(
        model_id="MOD-002",
        model_name="xgb-classifier",
        model_version="2.0.0",
        mlflow_run_id="RUN-def456",
        mlflow_experiment_name="razorpay-closeloop.classification",
        dataset_id="DS-001",
        dataset_version="1.1.0",
        feature_schema_version="1.1.0",
        algorithm="xgboost",
        training_config={"algorithm": "xgboost", "seed": 42},
        hyperparameters={"max_depth": 8, "learning_rate": 0.05},
        random_seed=42,
        feature_count=15,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        label_classes=["FEE_DIFF", "EXACT_MATCH", "REFUND"],
        training_examples=800,
        accuracy=0.88,
        f1_macro=0.86,
        false_automation=1,
        high_value_errors=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelVersionMetadataSchema:
    """Test ModelVersionMetadata schema."""

    def test_basic_creation(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="test",
            model_version="1.0.0",
        )
        assert meta.model_id == "MOD-1"
        assert meta.status == ModelVersionStatus.CANDIDATE
        assert meta.trained_at is None

    def test_with_mlflow_linkage(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="test",
            model_version="1.0.0",
            mlflow_run_id="RUN-123",
            mlflow_experiment_name="test-exp",
        )
        assert meta.mlflow_run_id == "RUN-123"
        assert meta.mlflow_experiment_name == "test-exp"

    def test_with_dataset_lineage(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="test",
            model_version="1.0.0",
            dataset_id="DS-1",
            dataset_version="2.0.0",
            feature_schema_version="1.5.0",
            label_schema_version="1.0.0",
        )
        assert meta.dataset_version == "2.0.0"
        assert meta.feature_schema_version == "1.5.0"
        assert meta.label_schema_version == "1.0.0"

    def test_with_metrics(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="test",
            model_version="1.0.0",
            accuracy=0.85,
            f1_macro=0.82,
            false_automation=2,
            high_value_errors=0,
        )
        assert meta.accuracy == 0.85
        assert meta.f1_macro == 0.82

    def test_summary(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="xgb-classifier",
            model_version="1.0.0",
            mlflow_run_id="RUN-abc123def456",
            status=ModelVersionStatus.ACTIVE,
            dataset_version="1.0.0",
        )
        s = meta.summary()
        assert "xgb-classifier" in s
        assert "1.0.0" in s
        assert "ACTIVE" in s
        assert "RUN-abc1" in s

    def test_get_lineage_dict(self):
        meta = ModelVersionMetadata(
            model_id="MOD-1",
            model_name="test",
            model_version="1.0.0",
            mlflow_run_id="RUN-123",
            dataset_version="2.0.0",
            feature_schema_version="1.0.0",
            accuracy=0.85,
        )
        lineage = meta.get_lineage_dict()
        assert lineage["model_id"] == "MOD-1"
        assert lineage["mlflow_run_id"] == "RUN-123"
        assert lineage["dataset_version"] == "2.0.0"
        assert lineage["accuracy"] == 0.85


class TestPredictionLineageSchema:
    """Test PredictionLineage schema."""

    def test_basic_creation(self):
        lineage = PredictionLineage(
            prediction_id="PRED-1",
            model_id="MOD-1",
            model_version="1.0.0",
        )
        assert lineage.prediction_id == "PRED-1"
        assert lineage.predicted_at is not None

    def test_with_mlflow_linkage(self):
        lineage = PredictionLineage(
            prediction_id="PRED-1",
            model_id="MOD-1",
            model_version="1.0.0",
            mlflow_run_id="RUN-123",
            dataset_version="1.0.0",
            feature_schema_version="1.0.0",
            algorithm="xgboost",
            prediction="FEE_DIFF",
            confidence=0.92,
        )
        assert lineage.mlflow_run_id == "RUN-123"
        assert lineage.dataset_version == "1.0.0"
        assert lineage.prediction == "FEE_DIFF"
        assert lineage.confidence == 0.92

    def test_summary(self):
        lineage = PredictionLineage(
            prediction_id="PRED-12345678",
            model_id="MOD-1",
            model_version="2.0.0",
            mlflow_run_id="RUN-abc123",
        )
        s = lineage.summary()
        assert "2.0.0" in s
        assert "RUN-abc1" in s


class TestModelVersionSummarySchema:
    """Test ModelVersionSummary schema."""

    def test_basic_creation(self):
        summary = ModelVersionSummary()
        assert summary.total_versions == 0
        assert summary.active_model is None

    def test_summary(self):
        summary = ModelVersionSummary(
            total_versions=5,
            active_count=1,
            candidate_count=1,
            retired_count=2,
            rejected_count=1,
        )
        s = summary.summary()
        assert "5" in s
        assert "1" in s


# ─────────────────────────────────────────────────────────────────────────────
# Registry: Registration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionRegistration:
    """Test model version registration."""

    def test_register_version(self, registry: ModelVersionRegistry, model_v1_args):
        meta = registry.register_version(**model_v1_args)
        assert meta.model_id == "MOD-001"
        assert meta.status == ModelVersionStatus.CANDIDATE

    def test_get_version(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        found = registry.get_version("MOD-001")
        assert found is not None
        assert found.model_version == "1.0.0"

    def test_get_nonexistent_version(self, registry: ModelVersionRegistry):
        found = registry.get_version("MOD-NONE")
        assert found is None

    def test_register_multiple_versions(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        versions = registry.list_versions()
        assert len(versions) == 2

    def test_get_version_by_mlflow_run(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        found = registry.get_version_by_mlflow_run("RUN-abc123")
        assert found is not None
        assert found.model_id == "MOD-001"

    def test_get_version_by_mlflow_run_not_found(self, registry: ModelVersionRegistry):
        found = registry.get_version_by_mlflow_run("RUN-nonexistent")
        assert found is None

    def test_list_versions_by_status(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        candidates = registry.list_versions(status=ModelVersionStatus.CANDIDATE)
        assert len(candidates) == 1
        active = registry.list_versions(status=ModelVersionStatus.ACTIVE)
        assert len(active) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Registry: Current/Candidate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCurrentCandidateTracking:
    """Test current and candidate model identification."""

    def test_no_active_initially(self, registry: ModelVersionRegistry):
        assert registry.active_model_id is None
        assert registry.get_active_model() is None

    def test_no_candidate_initially(self, registry: ModelVersionRegistry):
        assert registry.candidate_model_id is None
        assert registry.get_candidate_model() is None

    def test_promote_to_active(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        meta = registry.promote_to_active("MOD-001")
        assert meta.status == ModelVersionStatus.ACTIVE
        assert meta.promoted_at is not None
        assert registry.active_model_id == "MOD-001"
        assert registry.get_active_model() is not None

    def test_promote_retires_previous(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        registry.promote_to_active("MOD-001")
        registry.promote_to_active("MOD-002")
        assert registry.active_model_id == "MOD-002"
        old = registry.get_version("MOD-001")
        assert old.status == ModelVersionStatus.RETIRED
        assert old.retired_at is not None

    def test_promote_nonexistent(self, registry: ModelVersionRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.promote_to_active("MOD-NONE")

    def test_set_candidate(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        meta = registry.set_candidate("MOD-001")
        assert meta.status == ModelVersionStatus.CANDIDATE
        assert registry.candidate_model_id == "MOD-001"
        assert registry.get_candidate_model() is not None

    def test_set_candidate_nonexistent(self, registry: ModelVersionRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.set_candidate("MOD-NONE")

    def test_reject_candidate(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        registry.set_candidate("MOD-001")
        meta = registry.reject_candidate("MOD-001")
        assert meta.status == ModelVersionStatus.REJECTED
        assert registry.candidate_model_id is None

    def test_reject_nonexistent(self, registry: ModelVersionRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.reject_candidate("MOD-NONE")

    def test_promote_clears_candidate(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        registry.set_candidate("MOD-001")
        assert registry.candidate_model_id == "MOD-001"
        registry.promote_to_active("MOD-001")
        assert registry.candidate_model_id is None


# ─────────────────────────────────────────────────────────────────────────────
# Registry: Rollback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRollback:
    """Test model rollback."""

    def test_rollback_no_active(self, registry: ModelVersionRegistry):
        result = registry.rollback()
        assert result is None

    def test_rollback_no_retired(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        registry.promote_to_active("MOD-001")
        result = registry.rollback()
        assert result is None

    def test_rollback_to_previous(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        registry.promote_to_active("MOD-001")
        registry.promote_to_active("MOD-002")
        result = registry.rollback()
        assert result is not None
        assert result.model_id == "MOD-001"
        assert result.status == ModelVersionStatus.ACTIVE
        assert registry.active_model_id == "MOD-001"
        old = registry.get_version("MOD-002")
        assert old.status == ModelVersionStatus.RETIRED


# ─────────────────────────────────────────────────────────────────────────────
# Registry: Prediction Lineage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictionLineage:
    """Test prediction lineage tracking."""

    def test_record_prediction(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        lineage = registry.record_prediction(
            prediction_id="PRED-1",
            model_id="MOD-001",
            prediction="FEE_DIFF",
            confidence=0.92,
        )
        assert lineage.model_version == "1.0.0"
        assert lineage.mlflow_run_id == "RUN-abc123"
        assert lineage.dataset_version == "1.0.0"
        assert lineage.feature_schema_version == "1.0.0"
        assert lineage.algorithm == "xgboost"
        assert lineage.prediction == "FEE_DIFF"
        assert lineage.confidence == 0.92

    def test_record_prediction_nonexistent_model(self, registry: ModelVersionRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.record_prediction("PRED-1", "MOD-NONE")

    def test_get_prediction_lineage(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        registry.record_prediction("PRED-1", "MOD-001")
        lineage = registry.get_prediction_lineage("PRED-1")
        assert lineage is not None
        assert lineage.model_id == "MOD-001"

    def test_get_prediction_lineage_not_found(self, registry: ModelVersionRegistry):
        lineage = registry.get_prediction_lineage("PRED-NONE")
        assert lineage is None

    def test_get_predictions_by_model(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        registry.record_prediction("PRED-1", "MOD-001", prediction="A")
        registry.record_prediction("PRED-2", "MOD-001", prediction="B")
        predictions = registry.get_predictions_by_model("MOD-001")
        assert len(predictions) == 2

    def test_get_predictions_by_model_empty(self, registry: ModelVersionRegistry):
        predictions = registry.get_predictions_by_model("MOD-NONE")
        assert predictions == []


# ─────────────────────────────────────────────────────────────────────────────
# Registry: Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSummary:
    """Test model version summary."""

    def test_empty_summary(self, registry: ModelVersionRegistry):
        summary = registry.get_summary()
        assert summary.total_versions == 0
        assert summary.active_count == 0

    def test_summary_with_versions(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        registry.promote_to_active("MOD-001")
        registry.set_candidate("MOD-002")
        summary = registry.get_summary()
        assert summary.total_versions == 2
        assert summary.active_count == 1
        assert summary.candidate_count == 1
        assert summary.active_model is not None
        assert summary.candidate_model is not None

    def test_model_lineage(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        registry.record_prediction("PRED-1", "MOD-001")
        lineage = registry.get_model_lineage("MOD-001")
        assert lineage is not None
        assert lineage["model"]["model_id"] == "MOD-001"
        assert lineage["predictions_count"] == 1

    def test_model_lineage_not_found(self, registry: ModelVersionRegistry):
        lineage = registry.get_model_lineage("MOD-NONE")
        assert lineage is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration: MLflow Run Linkage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLflowRunLinkage:
    """Test that MLflow run IDs are properly linked."""

    def test_run_id_in_version_metadata(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        meta = registry.get_version("MOD-001")
        assert meta.mlflow_run_id == "RUN-abc123"
        assert meta.mlflow_experiment_name == "razorpay-closeloop.classification"

    def test_run_id_in_prediction_lineage(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        registry.record_prediction("PRED-1", "MOD-001")
        lineage = registry.get_prediction_lineage("PRED-1")
        assert lineage.mlflow_run_id == "RUN-abc123"

    def test_lookup_by_run_id(self, registry: ModelVersionRegistry, model_v1_args):
        registry.register_version(**model_v1_args)
        found = registry.get_version_by_mlflow_run("RUN-abc123")
        assert found is not None
        assert found.model_id == "MOD-001"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Dataset/Feature Lineage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetFeatureLineage:
    """Test dataset and feature version lineage."""

    def test_dataset_version_in_metadata(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        meta = registry.get_version("MOD-001")
        assert meta.dataset_version == "1.0.0"
        assert meta.feature_schema_version == "1.0.0"

    def test_dataset_version_in_prediction(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        registry.register_version(**model_v1_args)
        registry.record_prediction("PRED-1", "MOD-001")
        lineage = registry.get_prediction_lineage("PRED-1")
        assert lineage.dataset_version == "1.0.0"
        assert lineage.feature_schema_version == "1.0.0"

    def test_different_versions_per_model(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        meta1 = registry.get_version("MOD-001")
        meta2 = registry.get_version("MOD-002")
        assert meta1.dataset_version == "1.0.0"
        assert meta2.dataset_version == "1.1.0"
        assert meta1.feature_schema_version == "1.0.0"
        assert meta2.feature_schema_version == "1.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Full Lineage Chain Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullLineageChain:
    """Test the complete prediction → model → MLflow → dataset → features chain."""

    def test_complete_chain(
        self, registry: ModelVersionRegistry, model_v1_args
    ):
        # Register model with all lineage
        registry.register_version(**model_v1_args)
        registry.promote_to_active("MOD-001")

        # Record a prediction
        lineage = registry.record_prediction(
            prediction_id="PRED-FULL",
            model_id="MOD-001",
            prediction="FEE_DIFF",
            confidence=0.95,
        )

        # Verify complete chain
        assert lineage.model_id == "MOD-001"
        assert lineage.model_version == "1.0.0"
        assert lineage.mlflow_run_id == "RUN-abc123"
        assert lineage.dataset_version == "1.0.0"
        assert lineage.feature_schema_version == "1.0.0"
        assert lineage.algorithm == "xgboost"

        # Verify we can look up the model from the prediction
        model_meta = registry.get_version(lineage.model_id)
        assert model_meta is not None
        assert model_meta.mlflow_run_id == lineage.mlflow_run_id
        assert model_meta.dataset_version == lineage.dataset_version

        # Verify full model lineage includes the prediction
        full = registry.get_model_lineage("MOD-001")
        assert full["predictions_count"] == 1
        assert full["predictions"][0]["prediction_id"] == "PRED-FULL"

    def test_prediction_after_promotion(
        self, registry: ModelVersionRegistry, model_v1_args, model_v2_args
    ):
        # Register v1, promote, then v2, promote
        registry.register_version(**model_v1_args)
        registry.register_version(**model_v2_args)
        registry.promote_to_active("MOD-001")
        registry.promote_to_active("MOD-002")

        # Record predictions from both
        registry.record_prediction("PRED-V1", "MOD-001", prediction="A")
        registry.record_prediction("PRED-V2", "MOD-002", prediction="B")

        # Each prediction traces to its own model version
        p1 = registry.get_prediction_lineage("PRED-V1")
        p2 = registry.get_prediction_lineage("PRED-V2")
        assert p1.model_version == "1.0.0"
        assert p2.model_version == "2.0.0"
        assert p1.mlflow_run_id != p2.mlflow_run_id


# ─────────────────────────────────────────────────────────────────────────────
# Enum Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelVersionStatus:
    """Test ModelVersionStatus enum."""

    def test_all_statuses(self):
        statuses = [s.value for s in ModelVersionStatus]
        assert "CANDIDATE" in statuses
        assert "ACTIVE" in statuses
        assert "RETIRED" in statuses
        assert "REJECTED" in statuses
