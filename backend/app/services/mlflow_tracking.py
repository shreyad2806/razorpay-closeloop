"""
MLflow Experiment Tracking service for Razorpay CloseLoop Phase 10A.

Implements MLflow integration for model development reproducibility.

Milestone: We can prove which model produced which result.

Architecture:
  This service wraps MLflow's tracking client to provide:
  - Experiment creation and management
  - Run creation and lifecycle
  - Metric and parameter logging
  - Run metadata management

Safety principle:
  MLflow tracking is OBSERVATIONAL ONLY.
  It never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import platform
import sys
from datetime import datetime, timezone

import hashlib
import json
import tempfile

from app.schemas.mlflow_tracking import (
    ArtifactLineage,
    ArtifactMetadata,
    ArtifactType,
    EnvironmentMetadata,
    ExperimentSummary,
    ExperimentType,
    MLflowConfig,
    MLflowRunMetadata,
    MetricsSnapshot,
    RunStatus,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


def collect_environment_metadata() -> EnvironmentMetadata:
    """Collect reproducibility metadata from the current environment.

    Gathers: Python version, platform, git state, package versions.
    Falls back gracefully if any information is unavailable.
    """
    # Python and platform
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_platform = f"{platform.system()} {platform.release()}"
    hostname = platform.node() or None

    # Git info
    git_commit = None
    git_branch = None
    git_dirty = None
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_branch = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_dirty = len(result.stdout.strip()) > 0
    except Exception:
        pass

    # Package versions
    def _get_version(module_name: str) -> Optional[str]:
        try:
            mod = __import__(module_name)
            return getattr(mod, "__version__", None)
        except ImportError:
            return None

    mlflow_ver = _get_version("mlflow")
    numpy_ver = _get_version("numpy")
    pandas_ver = _get_version("pandas")
    sklearn_ver = None
    try:
        import sklearn
        sklearn_ver = sklearn.__version__
    except Exception:
        pass
    xgboost_ver = _get_version("xgboost")
    torch_ver = _get_version("torch")

    return EnvironmentMetadata(
        python_version=python_version,
        platform=os_platform,
        hostname=hostname,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        mlflow_version=mlflow_ver,
        numpy_version=numpy_ver,
        pandas_version=pandas_ver,
        sklearn_version=sklearn_ver,
        xgboost_version=xgboost_ver,
        torch_version=torch_ver,
        collected_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Tracking Service
# ─────────────────────────────────────────────────────────────────────────────


class MLflowTrackingService:
    """Service for MLflow experiment tracking.

    Manages experiments, runs, and metrics logging.
    Uses MLflow's built-in tracking where possible.

    Falls back to in-memory tracking when MLflow server is unavailable,
    ensuring the system can still operate for local development.
    """

    def __init__(self, config: Optional[MLflowConfig] = None) -> None:
        self._config = config or MLflowConfig()
        self._client = None
        self._experiments: Dict[str, str] = {}  # name → mlflow_experiment_id
        self._runs: Dict[str, MLflowRunMetadata] = {}  # run_id → metadata
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}  # run_id → [metrics]
        self._artifacts: Dict[str, List[ArtifactMetadata]] = {}  # run_id → [artifacts]
        self._mlflow_available = False

        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking client."""
        try:
            import mlflow

            # Set tracking URI from config or environment
            tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI", self._config.tracking_uri
            )
            mlflow.set_tracking_uri(tracking_uri)
            self._client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            self._mlflow_available = True
        except Exception:
            # Fall back to in-memory tracking
            self._mlflow_available = False

    @property
    def is_mlflow_available(self) -> bool:
        return self._mlflow_available

    @property
    def config(self) -> MLflowConfig:
        return self._config

    # ─────────────────────────────────────────────────────────────────────
    # Experiment Management
    # ─────────────────────────────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType = ExperimentType.EXCEPTION_CLASSIFICATION,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create or get an MLflow experiment.

        Returns the MLflow experiment ID.
        """
        prefixed_name = f"{self._config.tags_prefix}{name}"

        if prefixed_name in self._experiments:
            return self._experiments[prefixed_name]

        experiment_id = None
        if self._mlflow_available and self._client is not None:
            try:
                # Try to get existing experiment
                experiment = self._client.get_experiment_by_name(prefixed_name)
                if experiment is not None:
                    experiment_id = experiment.experiment_id
                else:
                    # Create new experiment
                    mlflow_tags = {f"{self._config.tags_prefix}type": experiment_type.value}
                    if tags:
                        mlflow_tags.update({f"{self._config.tags_prefix}{k}": v for k, v in tags.items()})
                    experiment_id = self._client.create_experiment(
                        prefixed_name, tags=mlflow_tags
                    )
            except Exception:
                experiment_id = _gen_id("EXP")
        else:
            experiment_id = _gen_id("EXP")

        self._experiments[prefixed_name] = experiment_id
        return experiment_id

    def get_experiment_id(self, name: str) -> Optional[str]:
        """Get experiment ID by name."""
        prefixed_name = f"{self._config.tags_prefix}{name}"
        return self._experiments.get(prefixed_name)

    def list_experiments(self) -> Dict[str, str]:
        """List all tracked experiments."""
        return dict(self._experiments)

    # ─────────────────────────────────────────────────────────────────────
    # Run Management
    # ─────────────────────────────────────────────────────────────────────

    def create_run(
        self,
        experiment_name: str,
        model_type: str,
        model_name: str,
        model_version: str,
        algorithm: str,
        run_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        training_examples: int = 0,
        validation_examples: int = 0,
        test_examples: int = 0,
        total_examples: int = 0,
        feature_count: int = 0,
        label_classes: Optional[List[str]] = None,
        target_label_version: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        class_weight: Optional[str] = None,
        early_stopping_rounds: Optional[int] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        max_features: Optional[int] = None,
        collect_env: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> MLflowRunMetadata:
        """Create a new training run with comprehensive parameter tracking.

        Automatically collects environment metadata (git, Python, deps) for
        reproducibility.  Logs all parameters to both in-memory store and
        MLflow when available.

        Returns RunMetadata with run_id for later metric logging.
        """
        prefixed_exp_name = f"{self._config.tags_prefix}{experiment_name}"
        experiment_id = self._experiments.get(prefixed_exp_name)

        # Collect environment metadata for reproducibility
        env_meta: Optional[EnvironmentMetadata] = None
        if collect_env:
            try:
                env_meta = collect_environment_metadata()
            except Exception:
                env_meta = None

        mlflow_run_id = None
        if self._mlflow_available and self._client is not None:
            try:
                import mlflow

                # Create MLflow run
                mlflow_run = self._client.create_run(
                    experiment_id=experiment_id or "0",
                    run_name=run_name,
                    tags=tags or {},
                )
                mlflow_run_id = mlflow_run.info.run_id

                # Build a temporary metadata to get all params, then log them
                _tmp = MLflowRunMetadata(
                    run_id=mlflow_run_id,
                    experiment_name=experiment_name,
                    model_type=model_type,
                    model_name=model_name,
                    model_version=model_version,
                    algorithm=algorithm,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    feature_schema_version=feature_schema_version,
                    training_examples=training_examples,
                    validation_examples=validation_examples,
                    test_examples=test_examples,
                    total_examples=total_examples,
                    feature_count=feature_count,
                    label_classes=label_classes or [],
                    target_label_version=target_label_version,
                    training_config=training_config or {},
                    hyperparameters=hyperparameters or {},
                    random_seed=random_seed,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    class_weight=class_weight,
                    early_stopping_rounds=early_stopping_rounds,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    max_features=max_features,
                    environment_metadata=env_meta,
                    environment=os.environ.get("ENVIRONMENT", "local"),
                )
                all_params = _tmp.get_all_params()
                for key, value in all_params.items():
                    self._client.log_param(mlflow_run_id, key, value)
            except Exception:
                mlflow_run_id = _gen_id("RUN")
        else:
            mlflow_run_id = _gen_id("RUN")

        metadata = MLflowRunMetadata(
            run_id=mlflow_run_id,
            run_name=run_name,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            model_type=model_type,
            model_name=model_name,
            model_version=model_version,
            algorithm=algorithm,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            training_examples=training_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            total_examples=total_examples,
            feature_count=feature_count,
            label_classes=label_classes or [],
            target_label_version=target_label_version,
            training_config=training_config or {},
            hyperparameters=hyperparameters or {},
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight=class_weight,
            early_stopping_rounds=early_stopping_rounds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            max_features=max_features,
            environment_metadata=env_meta,
            started_at=datetime.now(timezone.utc),
            status=RunStatus.RUNNING,
            environment=os.environ.get("ENVIRONMENT", "local"),
        )

        self._runs[mlflow_run_id] = metadata
        self._metrics[mlflow_run_id] = []

        return metadata

    def complete_run(
        self,
        run_id: str,
        status: RunStatus = RunStatus.COMPLETED,
        error_message: Optional[str] = None,
    ) -> MLflowRunMetadata:
        """Mark a run as completed."""
        metadata = self._runs.get(run_id)
        if metadata is None:
            raise ValueError(f"Run {run_id} not found")

        metadata.status = status
        metadata.completed_at = datetime.now(timezone.utc)
        metadata.error_message = error_message

        if metadata.started_at:
            metadata.duration_seconds = (
                metadata.completed_at - metadata.started_at
            ).total_seconds()

        if self._mlflow_available and self._client is not None:
            try:
                import mlflow

                # Map our status to MLflow status
                mlflow_status = {
                    RunStatus.COMPLETED: "FINISHED",
                    RunStatus.FAILED: "FAILED",
                    RunStatus.KILLED: "KILLED",
                }.get(status, "RUNNING")
                self._client.set_terminated(run_id, status=mlflow_status)
            except Exception:
                pass

        return metadata

    def get_run(self, run_id: str) -> Optional[MLflowRunMetadata]:
        """Get run metadata by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self, experiment_name: Optional[str] = None
    ) -> List[MLflowRunMetadata]:
        """List runs, optionally filtered by experiment."""
        runs = list(self._runs.values())
        if experiment_name:
            prefixed = f"{self._config.tags_prefix}{experiment_name}"
            runs = [r for r in runs if r.experiment_name == prefixed or r.experiment_name == experiment_name]
        return runs

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: Create Run from TrainingConfig + ModelMetadata
    # ─────────────────────────────────────────────────────────────────────

    def create_run_from_training(
        self,
        experiment_name: str,
        model_metadata: Any,  # ModelMetadata from model_training.py
        run_name: Optional[str] = None,
        validation_examples: int = 0,
        test_examples: int = 0,
        label_classes: Optional[List[str]] = None,
        collect_env: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> MLflowRunMetadata:
        """Create an MLflow run directly from a ModelMetadata object.

        Extracts algorithm-specific parameters from the TrainingConfig
        embedded in the metadata.
        """
        config = model_metadata.config
        hp = config.hyperparameters

        return self.create_run(
            experiment_name=experiment_name,
            model_type=model_metadata.model_type.value,
            model_name=model_metadata.model_name,
            model_version=model_metadata.version,
            algorithm=config.algorithm,
            run_name=run_name,
            dataset_id=model_metadata.dataset_id,
            dataset_version=model_metadata.dataset_version,
            feature_schema_version=model_metadata.feature_schema_version,
            training_examples=model_metadata.training_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            total_examples=model_metadata.training_examples + validation_examples + test_examples,
            feature_count=model_metadata.feature_count,
            label_classes=label_classes or model_metadata.label_classes,
            training_config=config.model_dump(),
            hyperparameters=hp,
            random_seed=config.random_seed,
            n_estimators=hp.get("n_estimators"),
            max_depth=hp.get("max_depth"),
            learning_rate=hp.get("learning_rate"),
            subsample=hp.get("subsample"),
            colsample_bytree=hp.get("colsample_bytree"),
            class_weight=config.class_weight,
            early_stopping_rounds=config.early_stopping_rounds,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            max_features=config.max_features,
            collect_env=collect_env,
            tags=tags,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Parameter Logging
    # ─────────────────────────────────────────────────────────────────────

    def log_all_parameters(self, run_id: str) -> None:
        """Log the full parameter set from the stored run metadata.

        Uses MLflowRunMetadata.get_all_params() to produce a comprehensive
        parameter dictionary and logs every entry to MLflow.
        """
        metadata = self._runs.get(run_id)
        if metadata is None:
            raise ValueError(f"Run {run_id} not found")

        all_params = metadata.get_all_params()

        if self._mlflow_available and self._client is not None:
            for key, value in all_params.items():
                try:
                    self._client.log_param(run_id, key, value)
                except Exception:
                    pass  # Param may already exist; MLflow throws on overwrite

    def get_parameters(self, run_id: str) -> Dict[str, str]:
        """Get all tracked parameters for a run as a dict."""
        metadata = self._runs.get(run_id)
        if metadata is None:
            raise ValueError(f"Run {run_id} not found")
        return metadata.get_all_params()

    # ─────────────────────────────────────────────────────────────────────
    # Metric Logging
    # ─────────────────────────────────────────────────────────────────────

    def log_metrics(self, run_id: str, snapshot: MetricsSnapshot) -> None:
        """Log a metrics snapshot to a run.

        Logs both standard ML metrics and safety-critical metrics.
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        metrics_dict = snapshot.to_mlflow_dict()

        # Store in memory
        self._metrics.setdefault(run_id, [])
        self._metrics[run_id].append(metrics_dict)

        # Log to MLflow
        if self._mlflow_available and self._client is not None:
            try:
                for key, value in metrics_dict.items():
                    self._client.log_metric(run_id, key, value)
            except Exception:
                pass

    def log_params(self, run_id: str, params: Dict[str, str]) -> None:
        """Log parameters to a run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        if self._mlflow_available and self._client is not None:
            try:
                for key, value in params.items():
                    self._client.log_param(run_id, key, value)
            except Exception:
                pass

    def log_tag(self, run_id: str, key: str, value: str) -> None:
        """Log a tag to a run."""
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        if self._mlflow_available and self._client is not None:
            try:
                self._client.set_tag(run_id, key, value)
            except Exception:
                pass

    def get_metrics_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get the metrics history for a run."""
        return list(self._metrics.get(run_id, []))

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: Log EvaluationMetrics / LearningMetrics Directly
    # ─────────────────────────────────────────────────────────────────────

    def log_evaluation_metrics(
        self, run_id: str, eval_metrics: Any  # EvaluationMetrics
    ) -> None:
        """Log EvaluationMetrics (Phase 9E) directly to a run.

        Converts to MetricsSnapshot using EvaluationMetricsBuilder
        and logs all classification + safety metrics.
        """
        from app.services.mlflow_metrics_builder import EvaluationMetricsBuilder
        snapshot = EvaluationMetricsBuilder.build(eval_metrics, run_id)
        self.log_metrics(run_id, snapshot)

    def log_learning_metrics(
        self, run_id: str, learning_metrics: Any  # LearningMetrics
    ) -> None:
        """Log LearningMetrics (Phase 9H) directly to a run.

        Converts to MetricsSnapshot using LearningMetricsBuilder
        and logs automation + human review + financial + reward + verification metrics.
        """
        from app.services.mlflow_metrics_builder import LearningMetricsBuilder
        snapshot = LearningMetricsBuilder.build(learning_metrics, run_id)
        self.log_metrics(run_id, snapshot)

    def log_combined_metrics(
        self,
        run_id: str,
        eval_metrics: Any = None,  # EvaluationMetrics (optional)
        learning_metrics: Any = None,  # LearningMetrics (optional)
    ) -> None:
        """Log both EvaluationMetrics and LearningMetrics to a run.

        Builds a merged MetricsSnapshot that includes classification,
        safety, automation, human review, financial, reward, and
        verification metrics.
        """
        from app.services.mlflow_metrics_builder import MLflowMetricsBuilder
        snapshot = MLflowMetricsBuilder.build_combined(
            eval_metrics, learning_metrics, run_id
        )
        self.log_metrics(run_id, snapshot)

    # ─────────────────────────────────────────────────────────────────────
    # Run-to-Model Linking
    # ─────────────────────────────────────────────────────────────────────

    def link_model_to_run(
        self, run_id: str, model_id: str, model_version: str
    ) -> None:
        """Link a model to its training run.

        Stores the model reference as a tag for provenance.
        """
        metadata = self._runs.get(run_id)
        if metadata is None:
            raise ValueError(f"Run {run_id} not found")

        self.log_tag(run_id, f"{self._config.tags_prefix}model_id", model_id)
        self.log_tag(run_id, f"{self._config.tags_prefix}model_version", model_version)

    # ─────────────────────────────────────────────────────────────────────
    # Artifact Logging
    # ─────────────────────────────────────────────────────────────────────

    def log_artifact(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        artifact_name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        description: str = "",
        model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Log an artifact (binary content) to a run.

        Stores content to MLflow artifact store when available,
        and always keeps in-memory metadata for lineage tracking.
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        run_meta = self._runs[run_id]
        artifact_id = _gen_id("ART")
        artifact_path = f"{artifact_type.value}/{artifact_name}"
        checksum = hashlib.sha256(content).hexdigest()

        # Log to MLflow
        if self._mlflow_available and self._client is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{artifact_name}") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                self._client.log_artifact(run_id, tmp_path, artifact_path=artifact_type.value)
                import os
                os.unlink(tmp_path)
            except Exception:
                pass

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            artifact_path=artifact_path,
            description=description,
            model_version=model_version or run_meta.model_version,
            dataset_version=dataset_version or run_meta.dataset_version,
            feature_schema_version=feature_schema_version or run_meta.feature_schema_version,
            content_type=content_type,
            size_bytes=len(content),
            checksum=checksum,
        )

        self._artifacts.setdefault(run_id, [])
        self._artifacts[run_id].append(metadata)

        return metadata

    def log_json_artifact(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        artifact_name: str,
        data: Dict[str, Any],
        description: str = "",
        model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Log a JSON-serializable artifact to a run."""
        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self.log_artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            content=content,
            content_type="application/json",
            description=description,
            model_version=model_version,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
        )

    def log_text_artifact(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        artifact_name: str,
        text: str,
        description: str = "",
        model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Log a text artifact to a run."""
        content = text.encode("utf-8")
        return self.log_artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            content=content,
            content_type="text/plain",
            description=description,
            model_version=model_version,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
        )

    def log_model_artifact(
        self,
        run_id: str,
        model_bytes: bytes,
        artifact_name: str = "model",
        description: str = "Trained model artifact",
        model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Log a model artifact (serialized model bytes) to a run."""
        return self.log_artifact(
            run_id=run_id,
            artifact_type=ArtifactType.MODEL,
            artifact_name=artifact_name,
            content=model_bytes,
            content_type="application/octet-stream",
            description=description,
            model_version=model_version,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
        )

    def get_artifacts(self, run_id: str) -> List[ArtifactMetadata]:
        """Get all artifacts for a run."""
        return list(self._artifacts.get(run_id, []))

    def get_artifacts_by_type(
        self, run_id: str, artifact_type: ArtifactType
    ) -> List[ArtifactMetadata]:
        """Get artifacts filtered by type."""
        return [
            a for a in self._artifacts.get(run_id, [])
            if a.artifact_type == artifact_type
        ]

    def get_artifact_lineage(self, run_id: str) -> Optional[ArtifactLineage]:
        """Get complete artifact lineage for a run."""
        metadata = self._runs.get(run_id)
        if metadata is None:
            return None

        return ArtifactLineage(
            run_id=run_id,
            model_id=metadata.model_name,
            model_version=metadata.model_version,
            dataset_id=metadata.dataset_id,
            dataset_version=metadata.dataset_version,
            feature_schema_version=metadata.feature_schema_version,
            label_schema_version=metadata.target_label_version,
            artifacts=self.get_artifacts(run_id),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Experiment Summary
    # ─────────────────────────────────────────────────────────────────────

    def get_experiment_summary(
        self,
        experiment_name: str,
        experiment_type: ExperimentType = ExperimentType.EXCEPTION_CLASSIFICATION,
    ) -> ExperimentSummary:
        """Get a summary of an experiment."""
        runs = self.list_runs(experiment_name)
        completed = [r for r in runs if r.status == RunStatus.COMPLETED]
        failed = [r for r in runs if r.status == RunStatus.FAILED]
        active = [r for r in runs if r.status == RunStatus.RUNNING]

        best_run_id = None
        best_metric_name = None
        best_metric_value = None

        # Find best run by accuracy if available
        for run in completed:
            history = self.get_metrics_history(run.run_id)
            if history:
                latest = history[-1]
                acc = latest.get("accuracy")
                if acc is not None and (best_metric_value is None or acc > best_metric_value):
                    best_run_id = run.run_id
                    best_metric_name = "accuracy"
                    best_metric_value = acc

        prefixed = f"{self._config.tags_prefix}{experiment_name}"
        return ExperimentSummary(
            experiment_name=experiment_name,
            experiment_id=self._experiments.get(prefixed),
            experiment_type=experiment_type,
            run_count=len(runs),
            active_run_count=len(active),
            completed_run_count=len(completed),
            failed_run_count=len(failed),
            best_run_id=best_run_id,
            best_metric=best_metric_name,
            best_metric_value=best_metric_value,
        )
