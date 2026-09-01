"""
MLflow Integration service for Razorpay CloseLoop Phase 10J.

Unified integration that ties together all Phase 10 components into a
single coherent pipeline:

  Training → MLflow Tracking → Evaluation → Model Registry →
  Prediction Lineage → Experiment Comparison

Wires into:
  Phase 9: ModelTrainer, ModelRegistry, PromotionService
  Phase 8: Execution provenance
  Phase 6: Safety boundary (never bypassed)

Safety principle:
  Integration orchestrates the lifecycle.
  Phase 6 guardrails remain the final safety authority.
  MLflow records provenance — it never authorizes execution.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

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
    ArtifactType,
    MetricsSnapshot,
    MLflowRunMetadata,
    RunStatus,
)
from app.schemas.mlflow_model_registry import (
    ModelLifecycleState,
    RegistryModelEntry,
)
from app.schemas.model_training import EvaluationMetrics
from app.services.mlflow_tracking import MLflowTrackingService
from app.services.mlflow_model_registry import MLflowModelRegistry
from app.services.model_version_registry import ModelVersionRegistry
from app.services.result_lineage import ModelResultLineage
from app.services.experiment_comparison import ExperimentComparisonService
from app.services.unified_evaluation import UnifiedEvaluationService


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


class MLflowIntegration:
    """Unified MLflow integration service.

    Ties together:
    - MLflowTrackingService: experiment + run management
    - MLflowModelRegistry: lifecycle management
    - ModelVersionRegistry: model version provenance
    - ModelResultLineage: prediction traceability
    - ExperimentComparisonService: multi-dimensional comparison
    - UnifiedEvaluationService: evaluation pipeline

    Each component is used for its specific purpose.
    No component replaces another.
    No component bypasses Phase 6.
    """

    def __init__(
        self,
        tracking_service: Optional[MLflowTrackingService] = None,
        model_registry: Optional[MLflowModelRegistry] = None,
        version_registry: Optional[ModelVersionRegistry] = None,
        result_lineage: Optional[ModelResultLineage] = None,
        comparison_service: Optional[ExperimentComparisonService] = None,
        evaluation_service: Optional[UnifiedEvaluationService] = None,
    ) -> None:
        self._tracking = tracking_service or MLflowTrackingService()
        self._registry = model_registry or MLflowModelRegistry()
        self._versions = version_registry or ModelVersionRegistry()
        self._lineage = result_lineage or ModelResultLineage(
            registry=self._versions,
        )
        self._comparison = comparison_service or ExperimentComparisonService()
        self._evaluation = evaluation_service or UnifiedEvaluationService()

        self._lifecycle_records: Dict[str, MLflowLifecycleRecord] = {}
        self._lifecycle_by_model: Dict[str, str] = {}  # model_id → record_id

    # ─────────────────────────────────────────────────────────────────────
    # Accessors (for tests and external use)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def tracking(self) -> MLflowTrackingService:
        return self._tracking

    @property
    def registry(self) -> MLflowModelRegistry:
        return self._registry

    @property
    def version_registry(self) -> ModelVersionRegistry:
        return self._versions

    @property
    def result_lineage(self) -> ModelResultLineage:
        return self._lineage

    @property
    def comparison(self) -> ExperimentComparisonService:
        return self._comparison

    @property
    def evaluation(self) -> UnifiedEvaluationService:
        return self._evaluation

    # ─────────────────────────────────────────────────────────────────────
    # 1. Train and Track
    # ─────────────────────────────────────────────────────────────────────

    def train_and_track(
        self,
        request: TrainAndTrackRequest,
        model_id: Optional[str] = None,
    ) -> TrainAndTrackResult:
        """Train a model and set up full MLflow tracking.

        Steps:
        1. Create MLflow experiment (if needed)
        2. Create MLflow run with parameters
        3. Record model in MLflow version registry
        4. Create lifecycle record
        5. Register candidate in MLflow model registry

        Does NOT train the model — that's Phase 9's job.
        This sets up the tracking infrastructure.
        """
        mid = model_id or _gen_id("MOD")
        record_id = _gen_id("LCY")

        # Step 1: Experiment
        experiment_id = self._tracking.create_experiment(request.experiment_name)

        # Step 2: MLflow run
        run_meta = self._tracking.create_run(
            experiment_name=request.experiment_name,
            model_type="classifier",
            model_name=request.model_name,
            model_version=request.model_version,
            algorithm=request.algorithm,
            dataset_id=request.dataset_id,
            dataset_version=request.dataset_version,
            feature_schema_version=request.feature_schema_version,
            hyperparameters=request.hyperparameters,
            training_config=request.training_config,
        )

        # Step 3: Model version metadata
        self._versions.register_version(
            model_id=mid,
            model_name=request.model_name,
            model_version=request.model_version,
            mlflow_run_id=run_meta.run_id,
            mlflow_experiment_name=request.experiment_name,
            dataset_id=request.dataset_id,
            dataset_version=request.dataset_version,
            feature_schema_version=request.feature_schema_version,
            algorithm=request.algorithm,
            hyperparameters=request.hyperparameters,
        )

        # Step 4: Lifecycle record
        record = MLflowLifecycleRecord(
            record_id=record_id,
            model_id=mid,
            model_version=request.model_version,
            mlflow_experiment_name=request.experiment_name,
            mlflow_run_id=run_meta.run_id,
            dataset_id=request.dataset_id,
            dataset_version=request.dataset_version,
            feature_schema_version=request.feature_schema_version,
            steps=[
                MLflowLifecycleStep(
                    phase=IntegrationPhase.TRAINING,
                    status=IntegrationStatus.COMPLETED,
                    timestamp=datetime.now(timezone.utc),
                    details={"model_id": mid, "run_id": run_meta.run_id},
                ),
                MLflowLifecycleStep(
                    phase=IntegrationPhase.TRACKING,
                    status=IntegrationStatus.COMPLETED,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "experiment": request.experiment_name,
                        "parameters_logged": True,
                    },
                ),
            ],
            current_phase=IntegrationPhase.TRACKING,
            created_at=datetime.now(timezone.utc),
        )
        self._lifecycle_records[record_id] = record
        self._lifecycle_by_model[mid] = record_id

        # Step 5: Register as CANDIDATE in MLflow registry
        self._registry.register_candidate(
            model_id=mid,
            model_name=request.model_name,
            model_version=request.model_version,
            mlflow_run_id=run_meta.run_id,
            mlflow_experiment_name=request.experiment_name,
            dataset_version=request.dataset_version,
            feature_schema_version=request.feature_schema_version,
            algorithm=request.algorithm,
        )

        return TrainAndTrackResult(
            lifecycle_record_id=record_id,
            model_id=mid,
            model_version=request.model_version,
            mlflow_run_id=run_meta.run_id,
            mlflow_experiment=request.experiment_name,
            registry_model_id=mid,
            parameters_logged=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2. Log Metrics After Training
    # ─────────────────────────────────────────────────────────────────────

    def log_training_metrics(
        self,
        model_id: str,
        metrics: MetricsSnapshot,
    ) -> None:
        """Log training metrics to the MLflow run associated with a model."""
        version_meta = self._versions.get_version(model_id)
        if version_meta and version_meta.mlflow_run_id:
            self._tracking.log_metrics(version_meta.mlflow_run_id, metrics)

            # Update lifecycle record
            record_id = self._lifecycle_by_model.get(model_id)
            if record_id:
                rec = self._lifecycle_records.get(record_id)
                if rec:
                    rec.steps.append(MLflowLifecycleStep(
                        phase=IntegrationPhase.EVALUATION,
                        status=IntegrationStatus.COMPLETED,
                        timestamp=datetime.now(timezone.utc),
                        details={"metrics_logged": True},
                    ))

    # ─────────────────────────────────────────────────────────────────────
    # 3. Evaluate and Log to MLflow
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_and_track(
        self,
        model_id: str,
        evaluation_metrics: EvaluationMetrics,
        current_model_id: Optional[str] = None,
        current_metrics: Optional[EvaluationMetrics] = None,
    ) -> Dict[str, Any]:
        """Evaluate a candidate and log results to MLflow.

        Uses UnifiedEvaluationService for safety checks.
        Logs everything to MLflow for experiment comparison.
        """
        version_meta = self._versions.get_version(model_id)
        run_id = version_meta.mlflow_run_id if version_meta else None

        # Run unified evaluation
        report = self._evaluation.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=evaluation_metrics,
            current_model_id=current_model_id,
            current_model_version=(
                self._versions.get_version(current_model_id).model_version
                if current_model_id and self._versions.get_version(current_model_id)
                else None
            ),
            candidate_model_id=model_id,
            candidate_model_version=version_meta.model_version if version_meta else None,
            dataset_version=version_meta.dataset_version if version_meta else None,
            feature_schema_version=version_meta.feature_schema_version if version_meta else None,
            mlflow_run_id=run_id,
        )

        # Log evaluation report as MLflow artifact
        if run_id:
            self._tracking.log_json_artifact(
                run_id=run_id,
                artifact_type=ArtifactType.EVALUATION_REPORT,
                artifact_name=f"evaluation_{model_id}",
                data=report.to_report_dict(),
                description=f"Unified evaluation report for {model_id}",
            )

            # Log safety metrics
            safety_snapshot = MetricsSnapshot(
                run_id=run_id,
                false_automation=evaluation_metrics.false_automation,
                high_value_errors=evaluation_metrics.high_value_errors,
                unknown_case_errors=evaluation_metrics.unknown_case_errors,
                incorrect_auto_resolution=evaluation_metrics.incorrect_auto_resolution,
                verification_failure_rate=evaluation_metrics.verification_failure_rate,
                resolution_accuracy=evaluation_metrics.resolution_accuracy,
            )
            self._tracking.log_metrics(run_id, safety_snapshot)

        # Update lifecycle record
        record_id = self._lifecycle_by_model.get(model_id)
        if record_id:
            rec = self._lifecycle_records.get(record_id)
            if rec:
                rec.steps.append(MLflowLifecycleStep(
                    phase=IntegrationPhase.EVALUATION,
                    status=IntegrationStatus.COMPLETED,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "verdict": report.verdict.value,
                        "improvements": report.total_improvements,
                        "regressions": report.total_regressions,
                        "all_safety_passed": report.all_safety_passed,
                    },
                ))

        return {
            "report": report,
            "mlflow_run_id": run_id,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 4. Promote Model
    # ─────────────────────────────────────────────────────────────────────

    def promote_model(
        self,
        model_id: str,
        evaluation_verdict: str = "PROMOTE",
        accuracy: Optional[float] = None,
        f1_macro: Optional[float] = None,
        false_automation: Optional[int] = None,
        high_value_errors: Optional[int] = None,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Promote a model through the MLflow registry lifecycle.

        Steps:
        1. CANDIDATE → VALIDATION (validation gate)
        2. VALIDATION → PRODUCTION (promotion gate)

        Also updates the Phase 9 version registry.
        """
        version_meta = self._versions.get_version(model_id)

        # Step 1: Validation gate
        val_transition = self._registry.validate_candidate(model_id)

        # Step 2: Promotion gate (only if validation passed)
        if val_transition.to_state == ModelLifecycleState.VALIDATION:
            prod_transition = self._registry.promote_to_production(
                model_id=model_id,
                evaluation_verdict=evaluation_verdict,
                accuracy=accuracy,
                f1_macro=f1_macro,
                false_automation=false_automation,
                high_value_errors=high_value_errors,
                reason=reason,
            )
        else:
            prod_transition = val_transition

        # Update version registry status
        if version_meta:
            if prod_transition.to_state == ModelLifecycleState.PRODUCTION:
                self._versions.promote_to_active(model_id)
            elif prod_transition.to_state == ModelLifecycleState.ARCHIVED:
                self._versions.reject_candidate(model_id)

        # Update lifecycle record
        record_id = self._lifecycle_by_model.get(model_id)
        if record_id:
            rec = self._lifecycle_records.get(record_id)
            if rec:
                rec.registry_state = prod_transition.to_state.value
                rec.steps.append(MLflowLifecycleStep(
                    phase=IntegrationPhase.REGISTRY,
                    status=(
                        IntegrationStatus.COMPLETED
                        if prod_transition.to_state == ModelLifecycleState.PRODUCTION
                        else IntegrationStatus.FAILED
                    ),
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "from_state": prod_transition.from_state.value,
                        "to_state": prod_transition.to_state.value,
                        "reason": reason,
                    },
                ))
                if prod_transition.to_state == ModelLifecycleState.PRODUCTION:
                    rec.current_phase = IntegrationPhase.REGISTRY

        return {
            "validation": val_transition,
            "promotion": prod_transition,
            "new_state": prod_transition.to_state.value,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 5. Record Prediction with Lineage
    # ─────────────────────────────────────────────────────────────────────

    def predict_and_track(
        self, request: PredictAndTrackRequest
    ) -> PredictAndTrackResult:
        """Record a prediction with full lineage traceability."""
        version_meta = self._versions.get_version(request.model_id)

        result = self._lineage.record_result(
            model_name=version_meta.model_name if version_meta else "unknown",
            model_version=version_meta.model_version if version_meta else "unknown",
            model_id=request.model_id,
            exception_id=request.exception_id,
            workflow_id=request.workflow_id,
            prediction=request.prediction,
            confidence=request.confidence,
            policy_version=request.policy_version,
        )

        # Update lifecycle record
        record_id = self._lifecycle_by_model.get(request.model_id)
        if record_id:
            rec = self._lifecycle_records.get(record_id)
            if rec:
                rec.steps.append(MLflowLifecycleStep(
                    phase=IntegrationPhase.PREDICTION,
                    status=IntegrationStatus.COMPLETED,
                    timestamp=datetime.now(timezone.utc),
                    details={"result_id": result.result_id},
                ))

        return PredictAndTrackResult(
            result_id=result.result_id,
            model_id=request.model_id,
            model_version=version_meta.model_version if version_meta else "unknown",
            mlflow_run_id=version_meta.mlflow_run_id if version_meta else None,
            prediction=request.prediction,
            confidence=request.confidence,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 6. Compare Experiments
    # ─────────────────────────────────────────────────────────────────────

    def compare_model_runs(
        self,
        run_ids: List[str],
        strategy: str = "safety_first",
    ) -> Dict[str, Any]:
        """Compare multiple MLflow runs using the comparison service."""
        from app.schemas.experiment_comparison import RankingStrategy

        runs = []
        for run_id in run_ids:
            meta = self._tracking.get_run(run_id)
            if meta is None:
                continue

            # Get latest metrics
            history = self._tracking.get_metrics_history(run_id)
            metrics_snapshot = None
            if history:
                latest = history[-1]
                metrics_snapshot = MetricsSnapshot(
                    run_id=run_id,
                    accuracy=latest.get("accuracy"),
                    precision_macro=latest.get("precision_macro"),
                    recall_macro=latest.get("recall_macro"),
                    f1_macro=latest.get("f1_macro"),
                    false_automation=int(latest.get("false_automation", 0)),
                    high_value_errors=int(latest.get("high_value_errors", 0)),
                    verification_failure_rate=latest.get("verification_failure_rate"),
                    automation_rate=latest.get("automation_rate"),
                    human_review_rate=latest.get("human_review_rate"),
                    resolution_accuracy=latest.get("resolution_accuracy"),
                    avg_reward=latest.get("avg_reward"),
                )

            comparison_run = ExperimentComparisonService.prepare_run(meta, metrics_snapshot)
            runs.append(comparison_run)

        if len(runs) < 2:
            return {"error": "Need at least 2 valid runs to compare"}

        strategy_enum = RankingStrategy(strategy)
        comparison = self._comparison.compare(runs, strategy_enum)

        return {
            "comparison": comparison,
            "run_count": len(runs),
            "strategy": strategy,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 7. Audit Trace
    # ─────────────────────────────────────────────────────────────────────

    def audit_exception(self, exception_id: str) -> Optional[Dict[str, Any]]:
        """Produce a complete audit trace for an exception.

        Chain: Exception → Model → MLflow Run → Dataset → Features
        """
        return self._lineage.audit_exception(exception_id)

    def audit_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Produce a complete audit trace for a model."""
        version_meta = self._versions.get_version(model_id)
        if version_meta is None:
            return None

        lifecycle_record_id = self._lifecycle_by_model.get(model_id)
        lifecycle_record = (
            self._lifecycle_records.get(lifecycle_record_id)
            if lifecycle_record_id
            else None
        )

        return {
            "model": version_meta.get_lineage_dict(),
            "registry": self._registry.get_model(model_id),
            "lifecycle": lifecycle_record.model_dump() if lifecycle_record else None,
            "predictions_count": len(
                self._versions.get_predictions_by_model(model_id)
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # 8. Summary
    # ─────────────────────────────────────────────────────────────────────

    def get_lifecycle_record(
        self, record_id: str
    ) -> Optional[MLflowLifecycleRecord]:
        """Get a lifecycle record by ID."""
        return self._lifecycle_records.get(record_id)

    def get_lifecycle_by_model(
        self, model_id: str
    ) -> Optional[MLflowLifecycleRecord]:
        """Get the lifecycle record for a model."""
        record_id = self._lifecycle_by_model.get(model_id)
        if record_id:
            return self._lifecycle_records.get(record_id)
        return None

    def get_integration_summary(self) -> MLflowIntegrationSummary:
        """Get a summary of the complete integration state."""
        records = list(self._lifecycle_records.values())
        completed = sum(1 for r in records if r.all_completed)
        failed = sum(1 for r in records if r.any_failed)
        in_progress = len(records) - completed - failed

        prod = self._registry.get_production_model()

        return MLflowIntegrationSummary(
            total_lifecycle_records=len(records),
            completed_records=completed,
            failed_records=failed,
            in_progress_records=in_progress,
            total_models_tracked=len(self._versions.list_versions()),
            total_experiments=len(self._tracking.list_experiments()),
            registry_models=len(self._registry.list_models()),
            production_model=prod.model_version if prod else None,
            safety_boundary="ENFORCED",
        )
