"""
Tests for Razorpay CloseLoop Phase 10D — MLflow Artifact Tracking.

Tests artifact types, storage, lineage, and content integrity.
"""

import json
import pytest

from app.schemas.mlflow_tracking import (
    ArtifactLineage,
    ArtifactMetadata,
    ArtifactType,
    MLflowConfig,
)
from app.services.mlflow_tracking import MLflowTrackingService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def service() -> MLflowTrackingService:
    return MLflowTrackingService(config=MLflowConfig(tracking_uri="file:./test_mlruns"))


@pytest.fixture
def run_with_experiment(service: MLflowTrackingService):
    service.create_experiment("artifact-test")
    return service.create_run(
        experiment_name="artifact-test",
        model_type="exception_classifier",
        model_name="xgb-classifier",
        model_version="1.0.0",
        algorithm="xgboost",
        dataset_version="2.0.0",
        feature_schema_version="1.5.0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ArtifactType Enum Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactType:
    """Test artifact type enum values."""

    def test_all_types(self):
        types = [t.value for t in ArtifactType]
        assert "model" in types
        assert "evaluation_report" in types
        assert "confusion_matrix" in types
        assert "classification_report" in types
        assert "safety_report" in types
        assert "automation_report" in types
        assert "resolution_report" in types
        assert "dataset_metadata" in types
        assert "feature_schema" in types
        assert "label_schema" in types
        assert "training_config" in types
        assert "training_summary" in types
        assert "evaluation_summary" in types
        assert "dataset_statistics" in types
        assert "custom" in types

    def test_model_type(self):
        assert ArtifactType.MODEL.value == "model"


# ─────────────────────────────────────────────────────────────────────────────
# ArtifactMetadata Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactMetadataSchema:
    """Test ArtifactMetadata schema."""

    def test_basic_creation(self):
        meta = ArtifactMetadata(
            artifact_id="ART-123",
            run_id="RUN-456",
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
        )
        assert meta.artifact_id == "ART-123"
        assert meta.run_id == "RUN-456"
        assert meta.artifact_type == ArtifactType.MODEL
        assert meta.artifact_name == "model.pkl"
        assert meta.logged_at is not None

    def test_lineage_fields(self):
        meta = ArtifactMetadata(
            artifact_id="ART-123",
            run_id="RUN-456",
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            model_version="1.0.0",
            dataset_version="2.0.0",
            feature_schema_version="1.5.0",
        )
        assert meta.model_version == "1.0.0"
        assert meta.dataset_version == "2.0.0"
        assert meta.feature_schema_version == "1.5.0"

    def test_content_metadata(self):
        meta = ArtifactMetadata(
            artifact_id="ART-123",
            run_id="RUN-456",
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name="eval.json",
            content_type="application/json",
            size_bytes=1024,
            checksum="abc123",
        )
        assert meta.content_type == "application/json"
        assert meta.size_bytes == 1024
        assert meta.checksum == "abc123"

    def test_summary(self):
        meta = ArtifactMetadata(
            artifact_id="ART-123",
            run_id="RUN-456",
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            model_version="1.0.0",
        )
        s = meta.summary()
        assert "model.pkl" in s
        assert "model" in s
        assert "1.0.0" in s


# ─────────────────────────────────────────────────────────────────────────────
# ArtifactLineage Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactLineageSchema:
    """Test ArtifactLineage schema."""

    def test_basic_creation(self):
        lineage = ArtifactLineage(run_id="RUN-123")
        assert lineage.run_id == "RUN-123"
        assert lineage.artifacts == []

    def test_with_artifacts(self):
        art = ArtifactMetadata(
            artifact_id="ART-1",
            run_id="RUN-123",
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
        )
        lineage = ArtifactLineage(
            run_id="RUN-123",
            model_version="1.0.0",
            dataset_version="2.0.0",
            feature_schema_version="1.5.0",
            artifacts=[art],
        )
        assert len(lineage.artifacts) == 1
        assert lineage.model_version == "1.0.0"

    def test_summary(self):
        lineage = ArtifactLineage(
            run_id="RUN-123",
            model_version="1.0.0",
            dataset_version="2.0.0",
            artifacts=[],
        )
        s = lineage.summary()
        assert "1.0.0" in s
        assert "2.0.0" in s


# ─────────────────────────────────────────────────────────────────────────────
# Log Artifact Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLogArtifact:
    """Test logging artifacts to runs."""

    def test_log_binary_artifact(self, service: MLflowTrackingService, run_with_experiment):
        content = b"model bytes here"
        artifact = service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=content,
            description="Test model",
        )
        assert artifact.artifact_id.startswith("ART-")
        assert artifact.run_id == run_with_experiment.run_id
        assert artifact.artifact_type == ArtifactType.MODEL
        assert artifact.artifact_name == "model.pkl"
        assert artifact.size_bytes == len(content)
        assert artifact.description == "Test model"
        assert artifact.checksum is not None
        assert len(artifact.checksum) == 64  # SHA-256 hex

    def test_log_json_artifact(self, service: MLflowTrackingService, run_with_experiment):
        data = {"accuracy": 0.85, "f1": 0.82, "confusion": [[10, 2], [3, 15]]}
        artifact = service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name="evaluation.json",
            data=data,
            description="Evaluation results",
        )
        assert artifact.content_type == "application/json"
        assert artifact.size_bytes is not None
        assert artifact.size_bytes > 0

    def test_log_text_artifact(self, service: MLflowTrackingService, run_with_experiment):
        text = "Classification Report\\n==================\\nPrecision: 0.85"
        artifact = service.log_text_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.CLASSIFICATION_REPORT,
            artifact_name="classification.txt",
            text=text,
        )
        assert artifact.content_type == "text/plain"
        assert artifact.size_bytes == len(text.encode("utf-8"))

    def test_log_model_artifact(self, service: MLflowTrackingService, run_with_experiment):
        model_bytes = b"serialized xgboost model"
        artifact = service.log_model_artifact(
            run_id=run_with_experiment.run_id,
            model_bytes=model_bytes,
            artifact_name="classifier.pkl",
        )
        assert artifact.artifact_type == ArtifactType.MODEL
        assert artifact.artifact_name == "classifier.pkl"
        assert artifact.size_bytes == len(model_bytes)

    def test_log_artifact_nonexistent_run(self, service: MLflowTrackingService):
        with pytest.raises(ValueError, match="not found"):
            service.log_artifact(
                run_id="RUN-FAKE",
                artifact_type=ArtifactType.MODEL,
                artifact_name="model.pkl",
                content=b"data",
            )

    def test_artifact_lineage_populated(self, service: MLflowTrackingService, run_with_experiment):
        service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=b"model",
        )
        lineage = service.get_artifact_lineage(run_with_experiment.run_id)
        assert lineage is not None
        assert lineage.model_version == "1.0.0"
        assert lineage.dataset_version == "2.0.0"
        assert lineage.feature_schema_version == "1.5.0"
        assert len(lineage.artifacts) == 1

    def test_artifact_checksum_integrity(self, service: MLflowTrackingService, run_with_experiment):
        content = b"important model data for integrity check"
        artifact = service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=content,
        )
        import hashlib
        expected = hashlib.sha256(content).hexdigest()
        assert artifact.checksum == expected


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Retrieval Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactRetrieval:
    """Test artifact retrieval and filtering."""

    def test_get_artifacts_empty(self, service: MLflowTrackingService, run_with_experiment):
        artifacts = service.get_artifacts(run_with_experiment.run_id)
        assert artifacts == []

    def test_get_artifacts(self, service: MLflowTrackingService, run_with_experiment):
        service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=b"model",
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name="eval.json",
            data={"acc": 0.85},
        )
        artifacts = service.get_artifacts(run_with_experiment.run_id)
        assert len(artifacts) == 2

    def test_get_artifacts_by_type(self, service: MLflowTrackingService, run_with_experiment):
        service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=b"model",
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name="eval.json",
            data={"acc": 0.85},
        )
        models = service.get_artifacts_by_type(run_with_experiment.run_id, ArtifactType.MODEL)
        assert len(models) == 1
        assert models[0].artifact_type == ArtifactType.MODEL

        evals = service.get_artifacts_by_type(run_with_experiment.run_id, ArtifactType.EVALUATION_REPORT)
        assert len(evals) == 1

    def test_get_artifacts_nonexistent_run(self, service: MLflowTrackingService):
        artifacts = service.get_artifacts("RUN-FAKE")
        assert artifacts == []

    def test_get_artifact_lineage_nonexistent(self, service: MLflowTrackingService):
        lineage = service.get_artifact_lineage("RUN-FAKE")
        assert lineage is None


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Lineage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactLineage:
    """Test artifact lineage tracking."""

    def test_lineage_completeness(self, service: MLflowTrackingService, run_with_experiment):
        # Log multiple artifacts
        service.log_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name="model.pkl",
            content=b"model",
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name="eval.json",
            data={"acc": 0.85},
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.CONFUSION_MATRIX,
            artifact_name="confusion.json",
            data={"matrix": [[10, 2], [3, 15]]},
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.SAFETY_REPORT,
            artifact_name="safety.json",
            data={"false_auto": 2, "hv_errors": 0},
        )
        service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=ArtifactType.DATASET_METADATA,
            artifact_name="dataset.json",
            data={"version": "2.0.0", "size": 1000},
        )
        lineage = service.get_artifact_lineage(run_with_experiment.run_id)
        assert lineage is not None
        assert len(lineage.artifacts) == 5
        artifact_types = {a.artifact_type for a in lineage.artifacts}
        assert ArtifactType.MODEL in artifact_types
        assert ArtifactType.EVALUATION_REPORT in artifact_types
        assert ArtifactType.CONFUSION_MATRIX in artifact_types
        assert ArtifactType.SAFETY_REPORT in artifact_types
        assert ArtifactType.DATASET_METADATA in artifact_types

    def test_lineage_versions(self, service: MLflowTrackingService, run_with_experiment):
        lineage = service.get_artifact_lineage(run_with_experiment.run_id)
        assert lineage.model_version == "1.0.0"
        assert lineage.dataset_version == "2.0.0"
        assert lineage.feature_schema_version == "1.5.0"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Type Coverage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactTypeCoverage:
    """Test all artifact types can be logged."""

    @pytest.mark.parametrize("artifact_type", [
        ArtifactType.MODEL,
        ArtifactType.EVALUATION_REPORT,
        ArtifactType.CONFUSION_MATRIX,
        ArtifactType.CLASSIFICATION_REPORT,
        ArtifactType.SAFETY_REPORT,
        ArtifactType.AUTOMATION_REPORT,
        ArtifactType.RESOLUTION_REPORT,
        ArtifactType.DATASET_METADATA,
        ArtifactType.FEATURE_SCHEMA,
        ArtifactType.LABEL_SCHEMA,
        ArtifactType.TRAINING_CONFIG,
        ArtifactType.TRAINING_SUMMARY,
        ArtifactType.EVALUATION_SUMMARY,
        ArtifactType.DATASET_STATISTICS,
        ArtifactType.CUSTOM,
    ])
    def test_log_each_type(self, service: MLflowTrackingService, run_with_experiment, artifact_type):
        data = {"type": artifact_type.value, "content": "test data"}
        artifact = service.log_json_artifact(
            run_id=run_with_experiment.run_id,
            artifact_type=artifact_type,
            artifact_name=f"{artifact_type.value}.json",
            data=data,
        )
        assert artifact.artifact_type == artifact_type
        # Verify retrieval
        found = service.get_artifacts_by_type(run_with_experiment.run_id, artifact_type)
        assert len(found) == 1
