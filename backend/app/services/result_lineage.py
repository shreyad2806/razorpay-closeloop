"""
Result Lineage service for Razorpay CloseLoop Phase 10H.

Provides end-to-end traceability from any model prediction or resolution
result back to the exact model, MLflow run, dataset, and features.

Chain:  Result → Model Version → MLflow Run → Parameters → Metrics → Artifacts → Dataset → Features

Safety principle:
  Result lineage is OBSERVATIONAL ONLY.
  It records provenance but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.result_lineage import (
    AuditResponse,
    ModelLineageChain,
    ResultRecord,
    ResultType,
)
from app.schemas.model_version_metadata import ModelVersionMetadata
from app.services.model_version_registry import ModelVersionRegistry


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


class ModelResultLineage:
    """End-to-end traceability from result → model → MLflow → dataset → features.

    Maintains:
    - All ResultRecords keyed by result_id
    - ResultRecords keyed by exception_id for audit lookup
    - Links to ModelVersionRegistry for model provenance
    """

    def __init__(self, registry: Optional[ModelVersionRegistry] = None) -> None:
        self._results: Dict[str, ResultRecord] = {}
        self._results_by_exception: Dict[str, List[str]] = {}  # exception_id → [result_ids]
        self._registry = registry

    # ─────────────────────────────────────────────────────────────────────
    # Recording Results
    # ─────────────────────────────────────────────────────────────────────

    def record_result(
        self,
        model_name: str,
        model_version: str,
        result_type: ResultType = ResultType.CLASSIFICATION,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        model_id: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        label_schema_version: Optional[str] = None,
        algorithm: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        model_accuracy: Optional[float] = None,
        model_f1_macro: Optional[float] = None,
        model_precision_macro: Optional[float] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        prediction_raw: Optional[Dict[str, Any]] = None,
        policy_version: Optional[str] = None,
        result_id: Optional[str] = None,
    ) -> ResultRecord:
        """Record a model-generated result with full lineage.

        If model_id is provided and registry is available, the model metadata
        will be automatically filled from the registry.
        """
        rid = result_id or _gen_id("RESULT")

        # Try to enrich from registry if we have a model_id
        if model_id and self._registry:
            meta = self._registry.get_version(model_id)
            if meta:
                mlflow_run_id = mlflow_run_id or meta.mlflow_run_id
                mlflow_experiment_name = mlflow_experiment_name or meta.mlflow_experiment_name
                dataset_id = dataset_id or meta.dataset_id
                dataset_version = dataset_version or meta.dataset_version
                feature_schema_version = feature_schema_version or meta.feature_schema_version
                label_schema_version = label_schema_version or meta.label_schema_version
                algorithm = algorithm or meta.algorithm
                training_config = training_config or meta.training_config
                model_accuracy = model_accuracy or meta.accuracy
                model_f1_macro = model_f1_macro or meta.f1_macro
                model_precision_macro = model_precision_macro or meta.precision_macro
                policy_version = policy_version or meta.policy_version

        record = ResultRecord(
            result_id=rid,
            result_type=result_type,
            workflow_id=workflow_id,
            exception_id=exception_id,
            model_name=model_name,
            model_version=model_version,
            model_id=model_id,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=mlflow_experiment_name,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            label_schema_version=label_schema_version,
            algorithm=algorithm,
            training_config=training_config or {},
            model_accuracy=model_accuracy,
            model_f1_macro=model_f1_macro,
            model_precision_macro=model_precision_macro,
            prediction=prediction,
            confidence=confidence,
            prediction_raw=prediction_raw,
            policy_version=policy_version,
        )

        self._results[rid] = record

        if exception_id:
            self._results_by_exception.setdefault(exception_id, []).append(rid)

        return record

    # ─────────────────────────────────────────────────────────────────────
    # Lookup
    # ─────────────────────────────────────────────────────────────────────

    def get_result(self, result_id: str) -> Optional[ResultRecord]:
        """Get a result record by ID."""
        return self._results.get(result_id)

    def get_results_for_exception(self, exception_id: str) -> List[ResultRecord]:
        """Get all result records for a specific exception."""
        rids = self._results_by_exception.get(exception_id, [])
        return [self._results[rid] for rid in rids if rid in self._results]

    def get_results_by_model(
        self, model_name: str, model_version: Optional[str] = None
    ) -> List[ResultRecord]:
        """Get all results produced by a specific model (optionally filtered by version)."""
        results = [
            r for r in self._results.values()
            if r.model_name == model_name
        ]
        if model_version:
            results = [r for r in results if r.model_version == model_version]
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Lineage Chain
    # ─────────────────────────────────────────────────────────────────────

    def build_lineage_chain(self, result_id: str) -> Optional[ModelLineageChain]:
        """Build the complete lineage chain for a result.

        Chain: Result → Model Version → MLflow Run → Dataset → Features
        """
        record = self._results.get(result_id)
        if record is None:
            return None

        # Enrich from registry if available
        mlflow_run_status = None
        mlflow_run_params: Dict[str, str] = {}
        training_examples = None
        feature_count = None
        feature_names: List[str] = []
        label_classes: List[str] = []
        model_status = None
        trained_at = None

        if record.model_id and self._registry:
            meta = self._registry.get_version(record.model_id)
            if meta:
                model_status = meta.status.value if meta.status else None
                training_examples = meta.training_examples or None
                feature_count = meta.feature_count or None
                feature_names = meta.feature_names or []
                label_classes = meta.label_classes or []
                trained_at = meta.trained_at

        chain = ModelLineageChain(
            result_id=record.result_id,
            result_type=record.result_type.value,
            exception_id=record.exception_id,
            workflow_id=record.workflow_id,
            model_id=record.model_id,
            model_name=record.model_name,
            model_version=record.model_version,
            model_status=model_status,
            mlflow_run_id=record.mlflow_run_id,
            mlflow_experiment_name=record.mlflow_experiment_name,
            dataset_id=record.dataset_id,
            dataset_version=record.dataset_version,
            feature_schema_version=record.feature_schema_version,
            label_schema_version=record.label_schema_version,
            algorithm=record.algorithm,
            training_config=record.training_config,
            training_examples=training_examples,
            feature_count=feature_count,
            feature_names=feature_names,
            label_classes=label_classes,
            model_accuracy=record.model_accuracy,
            model_f1_macro=record.model_f1_macro,
            model_precision_macro=record.model_precision_macro,
            prediction=record.prediction,
            confidence=record.confidence,
            predicted_at=record.predicted_at,
            policy_version=record.policy_version,
            mlflow_run_status=mlflow_run_status,
            mlflow_run_params=mlflow_run_params,
        )
        return chain

    # ─────────────────────────────────────────────────────────────────────
    # Audit Response
    # ─────────────────────────────────────────────────────────────────────

    def audit_exception(self, exception_id: str) -> Optional[AuditResponse]:
        """Produce an audit response for a given exception.

        Answers:
        - Which model classified it?
        - Which version?
        - Which MLflow run?
        - Which dataset trained it?
        - Which features were used?
        - What was the model confidence?
        - What were the model's validation metrics?
        - Which policy version was active?
        """
        results = self.get_results_for_exception(exception_id)
        if not results:
            return None

        # Use the most recent result
        record = max(results, key=lambda r: r.predicted_at)

        trained_at = None
        training_examples = None
        feature_count = None
        if record.model_id and self._registry:
            meta = self._registry.get_version(record.model_id)
            if meta:
                trained_at = meta.trained_at
                training_examples = meta.training_examples or None
                feature_count = meta.feature_count or None

        chain = self.build_lineage_chain(record.result_id)

        return AuditResponse(
            exception_id=exception_id,
            result_id=record.result_id,
            model_name=record.model_name,
            model_version=record.model_version,
            model_id=record.model_id,
            mlflow_run_id=record.mlflow_run_id,
            mlflow_experiment_name=record.mlflow_experiment_name,
            dataset_version=record.dataset_version,
            feature_schema_version=record.feature_schema_version,
            algorithm=record.algorithm,
            training_examples=training_examples,
            prediction=record.prediction,
            confidence=record.confidence,
            model_accuracy=record.model_accuracy,
            model_f1_macro=record.model_f1_macro,
            policy_version=record.policy_version,
            predicted_at=record.predicted_at,
            trained_at=trained_at,
            lineage_chain=chain,
        )

    def audit_result(self, result_id: str) -> Optional[AuditResponse]:
        """Produce an audit response for a specific result ID."""
        record = self._results.get(result_id)
        if record is None:
            return None

        trained_at = None
        training_examples = None
        if record.model_id and self._registry:
            meta = self._registry.get_version(record.model_id)
            if meta:
                trained_at = meta.trained_at
                training_examples = meta.training_examples or None

        chain = self.build_lineage_chain(result_id)

        return AuditResponse(
            exception_id=record.exception_id,
            result_id=record.result_id,
            model_name=record.model_name,
            model_version=record.model_version,
            model_id=record.model_id,
            mlflow_run_id=record.mlflow_run_id,
            mlflow_experiment_name=record.mlflow_experiment_name,
            dataset_version=record.dataset_version,
            feature_schema_version=record.feature_schema_version,
            algorithm=record.algorithm,
            training_examples=training_examples,
            prediction=record.prediction,
            confidence=record.confidence,
            model_accuracy=record.model_accuracy,
            model_f1_macro=record.model_f1_macro,
            policy_version=record.policy_version,
            predicted_at=record.predicted_at,
            trained_at=trained_at,
            lineage_chain=chain,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Historical Result Preservation
    # ─────────────────────────────────────────────────────────────────────

    def get_historical_result(self, result_id: str) -> Optional[ResultRecord]:
        """Get a historical result preserving its original model metadata.

        Historical results MUST retain their original model metadata.
        If production moves from v3 → v4, old results must still reference v3.
        """
        return self._results.get(result_id)

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked results."""
        results = list(self._results.values())
        models_used: Dict[str, int] = {}
        for r in results:
            key = f"{r.model_name} v{r.model_version}"
            models_used[key] = models_used.get(key, 0) + 1

        return {
            "total_results": len(results),
            "total_exceptions": len(self._results_by_exception),
            "results_by_type": {
                rt.value: sum(1 for r in results if r.result_type == rt)
                for rt in ResultType
            },
            "models_used": models_used,
            "has_registry": self._registry is not None,
        }
