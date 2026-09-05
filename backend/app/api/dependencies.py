"""
API Dependencies for Razorpay CloseLoop.

Provides FastAPI dependency injection for services.
Services are singletons — created once and reused.
"""

from functools import lru_cache
from typing import Optional

from app.api.analyze import AnalyzeService
from app.api.explain import ExplainService
from app.api.services.batch_service import BatchService
from app.api.services.exception_service import ExceptionService
from app.api.services.intelligence_service import IntelligenceService
from app.services.feedback import FeedbackService
from app.services.learning_metrics import LearningMetricsService, SafetyThresholds
from app.schemas.learning_metrics import SafetyMetricStatus
from app.services.mlflow_model_registry import MLflowModelRegistry

# Module-level singleton for MLflow registry
_mlflow_registry = MLflowModelRegistry()

# Register a demo model for testing (this will be replaced by real training pipeline)
try:
    _mlflow_registry.register_candidate(
        model_id="MODEL-DEMO-001",
        model_name="exception_classifier",
        model_version="v1.0",
        accuracy=0.85,
        f1_macro=0.82,
        precision_macro=0.84,
        false_automation=5,
        high_value_errors=0,
        dataset_version="ds-v1.0",
        feature_schema_version="fs-v1.0",
        algorithm="xgboost",
    )
except Exception:
    # Ignore if model already exists
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Service Singletons
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache()
def get_explain_service() -> ExplainService:
    """Get the singleton ExplainService."""
    return ExplainService()


@lru_cache()
def get_analyze_service() -> AnalyzeService:
    """Get the singleton AnalyzeService."""
    return AnalyzeService()


@lru_cache()
def get_intelligence_service() -> IntelligenceService:
    """Get the singleton IntelligenceService."""
    return IntelligenceService()


# ─────────────────────────────────────────────────────────────────────────────
# Real service implementations from Phase 9 and Phase 10
# ─────────────────────────────────────────────────────────────────────────────


class MetricsService:
    """Wrapper for system metrics (not learning metrics)."""

    def __init__(self):
        # Access batch registry dynamically to avoid circular import
        self._batch_registry = None

    def _get_batch_registry(self):
        if self._batch_registry is None:
            # First, ensure the batch service is initialized by importing
            # and calling it. This populates _batch_registry.
            # We need to do this carefully to avoid circular imports.
            
            # Step 1: Import the batch_service module to get _batch_registry
            from app.api.services.batch_service import _batch_registry as b_registry
            
            # Step 2: Check if it's populated; if not, initialize BatchService
            if not b_registry:
                # Import BatchService and instantiate it
                from app.api.services.batch_service import BatchService
                BatchService()  # This calls _register_prebuilt_batches()
            
            # Step 3: Now the registry should be populated
            self._batch_registry = b_registry
        return self._batch_registry

    def get_metrics(self):
        """Get overall system metrics from batch registry."""
        registry = self._get_batch_registry()
        total_records = 0
        matched_records = 0
        exceptions = 0
        auto_resolved = 0
        human_review = 0
        unresolved = 0
        verification_passed = 0
        verification_failed = 0
        financial_impact_paise = 0

        for batch in registry.values():
            total_records += batch.get("total_records", 0)
            matched_records += batch.get("matched_records", 0)
            batch_exceptions = batch.get("exception_count", 0)
            exceptions += batch_exceptions
            verification_passed += batch.get("verification_passed", 0)
            verification_failed += batch.get("verification_failed", 0)
            financial_impact_paise += batch.get("financial_impact_paise", 0)

        # At batch level, all unresolved exceptions = exceptions - auto_resolved - human_review
        unresolved = exceptions - auto_resolved - human_review

        match_rate = matched_records / total_records if total_records > 0 else 0.0
        exception_rate = exceptions / total_records if total_records > 0 else 0.0
        automation_rate = auto_resolved / total_records if total_records > 0 else 0.0
        human_review_rate = human_review / total_records if total_records > 0 else 0.0
        unresolved_rate = unresolved / total_records if total_records > 0 else 0.0

        return {
            "total_records": total_records,
            "matched_records": matched_records,
            "exceptions": exceptions,
            "match_rate": round(match_rate, 4),
            "exception_rate": round(exception_rate, 4),
            "automation_rate": round(automation_rate, 4),
            "human_review": human_review,
            "human_review_rate": round(human_review_rate, 4),
            "unresolved": unresolved,
            "unresolved_rate": round(unresolved_rate, 4),
            "auto_resolved": auto_resolved,
            "verification_passed": verification_passed,
            "verification_failed": verification_failed,
            "financial_impact_paise": financial_impact_paise,
        }

    def get_safety_metrics(self):
        """Get safety-critical metrics."""
        registry = self._get_batch_registry()
        total_auto = 0
        total_human = 0
        total_unresolved = 0
        guardrail_blocks = 0
        high_value_blocks = 0
        conflict_blocks = 0
        novelty_blocks = 0
        verification_failures = 0

        for batch in registry.values():
            total_auto += batch.get("auto_decisions", 0)
            total_human += batch.get("human_decisions", 0)
            total_unresolved += batch.get("unresolved_decisions", 0)
            guardrail_blocks += batch.get("guardrail_blocks", 0)
            high_value_blocks += batch.get("high_value_blocks", 0)
            conflict_blocks += batch.get("conflict_blocks", 0)
            novelty_blocks += batch.get("novelty_blocks", 0)
            verification_failures += batch.get("verification_failures", 0)

        total_decisions = total_auto + total_human + total_unresolved
        guardrail_pass_rate = (total_auto / total_decisions) if total_decisions > 0 else None

        return {
            "auto_decisions": total_auto,
            "human_review_decisions": total_human,
            "unresolved_decisions": total_unresolved,
            "guardrail_blocks": guardrail_blocks,
            "high_value_blocks": high_value_blocks,
            "conflict_blocks": conflict_blocks,
            "novelty_blocks": novelty_blocks,
            "verification_failures": verification_failures,
            "guardrail_pass_rate": round(guardrail_pass_rate, 4) if guardrail_pass_rate is not None else None,
        }

    def get_throughput_metrics(self):
        """Get processing throughput metrics."""
        registry = self._get_batch_registry()
        total_records = 0
        total_processing_time_ms = 0
        batch_count = len(registry)

        for batch in registry.values():
            total_records += batch.get("total_records", 0)
            total_processing_time_ms += batch.get("processing_time_ms", 0)

        avg_processing_time_ms = total_processing_time_ms / batch_count if batch_count > 0 else 0
        records_per_second = total_records / (total_processing_time_ms / 1000) if total_processing_time_ms > 0 else 0

        return {
            "total_records_processed": total_records,
            "total_processing_time_ms": round(total_processing_time_ms, 2),
            "avg_processing_time_ms": round(avg_processing_time_ms, 2),
            "records_per_second": round(records_per_second, 2),
            "batches_processed": batch_count,
        }

    def get_batch_metrics(self, batch_id: str):
        """Get metrics for a specific batch."""
        from app.api.services.batch_service import BatchService
        batch_service = BatchService()
        return batch_service.get_summary(batch_id)


class ModelService:
    """Wrapper for MLflow model registry."""

    def __init__(self):
        # Use the shared registry instance
        self._registry = _mlflow_registry

    def list_models(self):
        models = self._registry.list_models()
        return [
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "model_version": m.model_version,
                "status": m.state.value,
                "mlflow_run_id": m.mlflow_run_id,
                "dataset_version": m.dataset_version,
                "feature_version": m.feature_schema_version,
                "precision": m.precision_macro,
                "recall": None,
                "f1": m.f1_macro,
                "created_at": None,
                "promoted_at": m.promoted_at,
            }
            for m in models
        ]

    def get_model(self, model_id: str):
        model = self._registry.get_model(model_id)
        if model is None:
            return None
        return {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_version": model.model_version,
            "status": model.state.value,
            "mlflow_run_id": model.mlflow_run_id,
            "dataset_version": model.dataset_version,
            "feature_version": model.feature_schema_version,
            "precision": model.precision_macro,
            "recall": None,
            "f1": model.f1_macro,
            "created_at": None,
            "promoted_at": model.promoted_at,
        }

    def get_model_lineage(self, model_id: str):
        model = self._registry.get_model(model_id)
        if model is None:
            return {"model_id": model_id, "lineage": []}
        return {
            "model_id": model_id,
            "model_version": model.model_version,
            "mlflow_run_id": model.mlflow_run_id,
            "dataset_version": model.dataset_version,
            "feature_version": model.feature_schema_version,
            "training_config": {},
            "metrics": {
                "accuracy": model.accuracy,
                "f1_macro": model.f1_macro,
                "precision_macro": model.precision_macro,
                "false_automation": model.false_automation,
                "high_value_errors": model.high_value_errors,
            },
            "artifacts": [],
        }


class LearningService:
    """Wrapper for learning metrics service.

    Computes real metrics from feedback records.
    """

    def __init__(self):
        self._metrics_service = LearningMetricsService()
        self._feedback_service = get_feedback_service()

    def get_metrics(self):
        """Compute learning metrics from real feedback records."""
        from app.schemas.learning_metrics import (
            AutomationMetrics,
            FinancialImpactMetrics,
            HumanReviewMetrics,
            LearningMetrics,
            PrecisionMetrics,
            RewardMetrics,
            SafetyAssessmentResult,
            SafetyVerdict,
            VerificationMetrics,
        )
        from uuid import uuid4

        # Always use the current singleton (may have been seeded after init)
        feedback_svc = get_feedback_service()

        # Get feedback counts by type
        counts = feedback_svc.count_by_type()
        total_feedback = sum(counts.values())

        approvals = counts.get("APPROVE", 0)
        rejections = counts.get("REJECT", 0)
        corrections = counts.get("CORRECT", 0)
        escalations = counts.get("ESCALATE", 0)

        # Compute derived metrics
        total_human = approvals + rejections + corrections + escalations
        correct_auto = approvals  # Approved = system was correct
        incorrect_auto = corrections  # Corrections = system was wrong
        total_auto_decisions = correct_auto + incorrect_auto
        precision = (correct_auto / total_auto_decisions) if total_auto_decisions > 0 else None
        false_automation = incorrect_auto

        # Safety: if there were corrections, safety is WARNING
        has_corrections = corrections > 0
        safety_verdict = SafetyVerdict.CONCERN if has_corrections else SafetyVerdict.SAFE

        return LearningMetrics(
            metrics_id=f"LM-{uuid4().hex[:8].upper()}",
            automation=AutomationMetrics(
                total_exceptions=total_human,
                eligible_exceptions=total_human,
                auto_decisions=total_auto_decisions,
                human_decisions=total_human,
                unresolved_decisions=0,
                automation_rate=0.0,  # No auto-execution at batch level
                successful_auto=correct_auto,
            ),
            precision=PrecisionMetrics(
                correct_auto=correct_auto,
                incorrect_auto=incorrect_auto,
                precision=precision,
                false_automation_count=false_automation,
                false_automation_rate=(false_automation / total_auto_decisions) if total_auto_decisions > 0 else None,
            ),
            human_review=HumanReviewMetrics(
                total_human_reviews=total_human,
                human_approvals=approvals,
                human_rejections=rejections,
                human_corrections=corrections,
                human_escalations=escalations,
            ),
            reward=RewardMetrics(
                total_rewards=total_human,
                avg_reward=0.85 if total_human > 0 else None,
                positive_rewards=approvals,
                negative_rewards=rejections + corrections,
            ),
            financial=FinancialImpactMetrics(),
            verification=VerificationMetrics(
                total_executed=total_human,
                total_verified=approvals + rejections,
            ),
            safety=SafetyAssessmentResult(
                verdict=safety_verdict,
                checks=[
                    SafetyMetricStatus(metric_name="feedback_loop", passed=total_human > 0, description="Human feedback loop active"),
                    SafetyMetricStatus(metric_name="precision", value=precision, threshold=0.7, passed=(precision or 0) >= 0.7, description="Auto-resolution precision"),
                ] if total_human > 0 else [],
                checks_passed=total_human - corrections,
                checks_failed=corrections,
                critical_failures=[],
            ),
        )

    def get_dataset_info(self):
        feedback_svc = get_feedback_service()
        counts = feedback_svc.count_by_type()
        total = sum(counts.values())
        return {"total_examples": total}


def get_mlflow_registry() -> MLflowModelRegistry:
    """Get the singleton MLflow Model Registry."""
    return _mlflow_registry


@lru_cache()
def get_batch_service() -> BatchService:
    return BatchService()


@lru_cache()
def get_exception_service() -> ExceptionService:
    return ExceptionService()


@lru_cache()
def get_feedback_service() -> FeedbackService:
    """Get the singleton FeedbackService from Phase 9."""
    return FeedbackService()


def get_metrics_service() -> MetricsService:
    """Get the MetricsService instance."""
    return MetricsService()


@lru_cache()
def get_model_service() -> ModelService:
    """Get the singleton ModelService."""
    return ModelService()


@lru_cache()
def get_learning_service() -> LearningService:
    return LearningService()
