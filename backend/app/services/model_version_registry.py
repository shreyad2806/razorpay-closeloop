"""
Model Version Registry service for Razorpay CloseLoop Phase 10E.

Connects MLflow runs with the existing Phase 9 model-versioning system.

Goal: Know exactly which model produced every prediction.

Architecture:
  ModelVersionRegistry maintains:
  - All model versions with their MLflow linkage
  - Current (ACTIVE) model identification
  - Candidate model identification
  - Prediction lineage tracking

Safety principle:
  Model version metadata is OBSERVATIONAL ONLY.
  It records provenance but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.model_version_metadata import (
    ModelVersionMetadata,
    ModelVersionStatus,
    ModelVersionSummary,
    PredictionLineage,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


class ModelVersionRegistry:
    """Registry linking model versions to MLflow runs.

    Maintains complete provenance:
    - model version → MLflow run
    - model version → dataset version
    - model version → feature version
    - prediction → model version → MLflow run
    """

    def __init__(self) -> None:
        self._versions: Dict[str, ModelVersionMetadata] = {}  # model_id → metadata
        self._predictions: Dict[str, PredictionLineage] = {}  # prediction_id → lineage
        self._active_model_id: Optional[str] = None
        self._candidate_model_id: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────
    # Model Version Management
    # ─────────────────────────────────────────────────────────────────────

    def register_version(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        label_schema_version: Optional[str] = None,
        algorithm: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        feature_count: int = 0,
        feature_names: Optional[List[str]] = None,
        label_classes: Optional[List[str]] = None,
        training_examples: int = 0,
        training_duration_seconds: Optional[float] = None,
        accuracy: Optional[float] = None,
        f1_macro: Optional[float] = None,
        precision_macro: Optional[float] = None,
        false_automation: Optional[int] = None,
        high_value_errors: Optional[int] = None,
        policy_version: Optional[str] = None,
    ) -> ModelVersionMetadata:
        """Register a new model version in the registry."""
        metadata = ModelVersionMetadata(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            status=ModelVersionStatus.CANDIDATE,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=mlflow_experiment_name,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            label_schema_version=label_schema_version,
            algorithm=algorithm,
            training_config=training_config or {},
            hyperparameters=hyperparameters or {},
            random_seed=random_seed,
            feature_count=feature_count,
            feature_names=feature_names or [],
            label_classes=label_classes or [],
            training_examples=training_examples,
            training_duration_seconds=training_duration_seconds,
            accuracy=accuracy,
            f1_macro=f1_macro,
            precision_macro=precision_macro,
            false_automation=false_automation,
            high_value_errors=high_value_errors,
            trained_at=datetime.utcnow(),
            policy_version=policy_version,
        )
        self._versions[model_id] = metadata
        return metadata

    def get_version(self, model_id: str) -> Optional[ModelVersionMetadata]:
        """Get model version metadata by ID."""
        return self._versions.get(model_id)

    def get_version_by_mlflow_run(
        self, mlflow_run_id: str
    ) -> Optional[ModelVersionMetadata]:
        """Find model version by MLflow run ID."""
        for meta in self._versions.values():
            if meta.mlflow_run_id == mlflow_run_id:
                return meta
        return None

    def list_versions(
        self, status: Optional[ModelVersionStatus] = None
    ) -> List[ModelVersionMetadata]:
        """List all model versions, optionally filtered by status."""
        versions = list(self._versions.values())
        if status is not None:
            versions = [v for v in versions if v.status == status]
        return versions

    # ─────────────────────────────────────────────────────────────────────
    # Current / Candidate Model
    # ─────────────────────────────────────────────────────────────────────

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def candidate_model_id(self) -> Optional[str]:
        return self._candidate_model_id

    def get_active_model(self) -> Optional[ModelVersionMetadata]:
        """Get the current production model metadata."""
        if self._active_model_id:
            return self._versions.get(self._active_model_id)
        return None

    def get_candidate_model(self) -> Optional[ModelVersionMetadata]:
        """Get the current candidate model metadata."""
        if self._candidate_model_id:
            return self._versions.get(self._candidate_model_id)
        return None

    def promote_to_active(self, model_id: str) -> ModelVersionMetadata:
        """Promote a model version to ACTIVE status.

        Retires the previous active model if one exists.
        """
        metadata = self._versions.get(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        # Retire previous active
        if self._active_model_id and self._active_model_id in self._versions:
            old = self._versions[self._active_model_id]
            old.status = ModelVersionStatus.RETIRED
            old.retired_at = datetime.utcnow()

        # Promote new
        metadata.status = ModelVersionStatus.ACTIVE
        metadata.promoted_at = datetime.utcnow()
        self._active_model_id = model_id

        # Clear candidate if it was the one promoted
        if self._candidate_model_id == model_id:
            self._candidate_model_id = None

        return metadata

    def set_candidate(self, model_id: str) -> ModelVersionMetadata:
        """Set a model version as the current candidate."""
        metadata = self._versions.get(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        # Mark previous candidate back to CANDIDATE if needed
        if (
            self._candidate_model_id
            and self._candidate_model_id in self._versions
            and self._candidate_model_id != model_id
        ):
            old = self._versions[self._candidate_model_id]
            if old.status == ModelVersionStatus.CANDIDATE:
                pass  # Already candidate, no change needed

        metadata.status = ModelVersionStatus.CANDIDATE
        self._candidate_model_id = model_id
        return metadata

    def reject_candidate(self, model_id: str) -> ModelVersionMetadata:
        """Reject a candidate model."""
        metadata = self._versions.get(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        metadata.status = ModelVersionStatus.REJECTED
        if self._candidate_model_id == model_id:
            self._candidate_model_id = None

        return metadata

    def rollback(self, reason: str = "manual rollback") -> Optional[ModelVersionMetadata]:
        """Rollback to the most recently retired model.

        Returns the restored model metadata, or None if no rollback target.
        """
        if self._active_model_id is None:
            return None

        current = self._versions.get(self._active_model_id)
        if current is None:
            return None

        # Find most recently retired model
        retired = [
            v for v in self._versions.values()
            if v.status == ModelVersionStatus.RETIRED and v.retired_at is not None
        ]
        if not retired:
            return None

        retired.sort(key=lambda v: v.retired_at, reverse=True)
        target = retired[0]

        # Retire current
        current.status = ModelVersionStatus.RETIRED
        current.retired_at = datetime.utcnow()

        # Restore target
        target.status = ModelVersionStatus.ACTIVE
        target.promoted_at = datetime.utcnow()
        self._active_model_id = target.model_id

        return target

    # ─────────────────────────────────────────────────────────────────────
    # Prediction Lineage
    # ─────────────────────────────────────────────────────────────────────

    def record_prediction(
        self,
        prediction_id: str,
        model_id: str,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> PredictionLineage:
        """Record a prediction with full lineage traceability.

        Links the prediction to the exact model version, MLflow run,
        dataset version, and feature version.
        """
        metadata = self._versions.get(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        lineage = PredictionLineage(
            prediction_id=prediction_id,
            model_id=metadata.model_id,
            model_version=metadata.model_version,
            mlflow_run_id=metadata.mlflow_run_id,
            dataset_version=metadata.dataset_version,
            feature_schema_version=metadata.feature_schema_version,
            algorithm=metadata.algorithm,
            prediction=prediction,
            confidence=confidence,
        )
        self._predictions[prediction_id] = lineage
        return lineage

    def get_prediction_lineage(
        self, prediction_id: str
    ) -> Optional[PredictionLineage]:
        """Get the lineage for a specific prediction."""
        return self._predictions.get(prediction_id)

    def get_predictions_by_model(
        self, model_id: str
    ) -> List[PredictionLineage]:
        """Get all predictions made by a specific model version."""
        return [
            p for p in self._predictions.values()
            if p.model_id == model_id
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    def get_summary(self) -> ModelVersionSummary:
        """Get a summary of all model versions."""
        all_versions = list(self._versions.values())
        return ModelVersionSummary(
            total_versions=len(all_versions),
            active_count=sum(1 for v in all_versions if v.status == ModelVersionStatus.ACTIVE),
            candidate_count=sum(1 for v in all_versions if v.status == ModelVersionStatus.CANDIDATE),
            retired_count=sum(1 for v in all_versions if v.status == ModelVersionStatus.RETIRED),
            rejected_count=sum(1 for v in all_versions if v.status == ModelVersionStatus.REJECTED),
            active_model=self.get_active_model(),
            candidate_model=self.get_candidate_model(),
        )

    def get_model_lineage(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get complete lineage for a model version including its predictions."""
        metadata = self._versions.get(model_id)
        if metadata is None:
            return None

        predictions = self.get_predictions_by_model(model_id)
        return {
            "model": metadata.get_lineage_dict(),
            "predictions_count": len(predictions),
            "predictions": [p.model_dump() for p in predictions],
        }
