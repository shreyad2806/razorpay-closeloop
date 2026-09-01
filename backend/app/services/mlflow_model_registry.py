"""
MLflow Model Registry service for Razorpay CloseLoop Phase 10G.

Implements a controlled lifecycle for model versions:
  CANDIDATE → VALIDATION → PRODUCTION → ARCHIVED

Safety principle:
  Model Registry controls model lifecycle.
  Phase 6 controls financial safety.
  Registry MUST NOT bypass Phase 6.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.mlflow_model_registry import (
    LifecycleTransition,
    ModelLifecycleState,
    PromotionGateConfig,
    RegistryModelEntry,
    RegistrySummary,
    ValidationGateConfig,
    is_valid_transition,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


class MLflowModelRegistry:
    """Manages model lifecycle through controlled state transitions.

    Lifecycle: CANDIDATE → VALIDATION → PRODUCTION → ARCHIVED

    Every transition is recorded for audit.
    """

    def __init__(
        self,
        validation_config: Optional[ValidationGateConfig] = None,
        promotion_config: Optional[PromotionGateConfig] = None,
    ) -> None:
        self._models: Dict[str, RegistryModelEntry] = {}  # model_id → entry
        self._transitions: List[LifecycleTransition] = []
        self._production_model_id: Optional[str] = None
        self._validation_config = validation_config or ValidationGateConfig()
        self._promotion_config = promotion_config or PromotionGateConfig()

    @property
    def validation_config(self) -> ValidationGateConfig:
        return self._validation_config

    @property
    def promotion_config(self) -> PromotionGateConfig:
        return self._promotion_config

    # ─────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────

    def register_candidate(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        algorithm: Optional[str] = None,
        accuracy: Optional[float] = None,
        f1_macro: Optional[float] = None,
        precision_macro: Optional[float] = None,
        false_automation: Optional[int] = None,
        high_value_errors: Optional[int] = None,
        previous_model_id: Optional[str] = None,
        previous_model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegistryModelEntry:
        """Register a new model as CANDIDATE."""
        entry = RegistryModelEntry(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            state=ModelLifecycleState.CANDIDATE,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=mlflow_experiment_name,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            algorithm=algorithm,
            accuracy=accuracy,
            f1_macro=f1_macro,
            precision_macro=precision_macro,
            false_automation=false_automation,
            high_value_errors=high_value_errors,
            previous_model_id=previous_model_id,
            previous_model_version=previous_model_version,
            metadata=metadata or {},
        )
        self._models[model_id] = entry
        return entry

    def get_model(self, model_id: str) -> Optional[RegistryModelEntry]:
        """Get a model entry by ID."""
        return self._models.get(model_id)

    def list_models(
        self, state: Optional[ModelLifecycleState] = None
    ) -> List[RegistryModelEntry]:
        """List all models, optionally filtered by state."""
        models = list(self._models.values())
        if state is not None:
            models = [m for m in models if m.state == state]
        return models

    # ─────────────────────────────────────────────────────────────────────
    # Validation Gate (CANDIDATE → VALIDATION)
    # ─────────────────────────────────────────────────────────────────────

    def validate_candidate(
        self,
        model_id: str,
        training_succeeded: bool = True,
        evaluation_metrics_exist: bool = True,
        safety_checks_passed: bool = True,
        reason: str = "",
    ) -> LifecycleTransition:
        """Run the validation gate and transition CANDIDATE → VALIDATION.

        Validation requires:
        - Training succeeded
        - Evaluation metrics exist
        - Safety checks passed
        - Model meets minimum quality thresholds
        """
        entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found")

        if entry.state != ModelLifecycleState.CANDIDATE:
            raise ValueError(
                f"Model {model_id} is {entry.state.value}, not CANDIDATE"
            )

        # Run validation checks
        config = self._validation_config
        checks_passed = True
        failure_reasons: List[str] = []

        if config.require_training_success and not training_succeeded:
            checks_passed = False
            failure_reasons.append("training_failed")

        if config.require_evaluation_metrics and not evaluation_metrics_exist:
            checks_passed = False
            failure_reasons.append("no_evaluation_metrics")

        if config.require_safety_checks and not safety_checks_passed:
            checks_passed = False
            failure_reasons.append("safety_checks_failed")

        if entry.accuracy is not None and entry.accuracy < config.min_accuracy:
            checks_passed = False
            failure_reasons.append(f"accuracy_{entry.accuracy:.2f}_below_{config.min_accuracy}")

        if entry.f1_macro is not None and entry.f1_macro < config.min_f1_macro:
            checks_passed = False
            failure_reasons.append(f"f1_{entry.f1_macro:.2f}_below_{config.min_f1_macro}")

        if entry.false_automation is not None and entry.false_automation > config.max_false_automation:
            checks_passed = False
            failure_reasons.append(f"false_auto_{entry.false_automation}_above_{config.max_false_automation}")

        if entry.high_value_errors is not None and entry.high_value_errors > config.max_high_value_errors:
            checks_passed = False
            failure_reasons.append(f"hv_errors_{entry.high_value_errors}_above_{config.max_high_value_errors}")

        if not checks_passed:
            # Transition to ARCHIVED (rejected at validation)
            return self._transition(
                model_id,
                ModelLifecycleState.ARCHIVED,
                reason=f"Validation failed: {'; '.join(failure_reasons)}",
            )

        # Transition to VALIDATION
        return self._transition(
            model_id,
            ModelLifecycleState.VALIDATION,
            reason=reason or "Validation gate passed",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Promotion Gate (VALIDATION → PRODUCTION)
    # ─────────────────────────────────────────────────────────────────────

    def promote_to_production(
        self,
        model_id: str,
        evaluation_verdict: str = "PROMOTE",
        accuracy: Optional[float] = None,
        f1_macro: Optional[float] = None,
        false_automation: Optional[int] = None,
        high_value_errors: Optional[int] = None,
        reason: str = "",
    ) -> LifecycleTransition:
        """Run the promotion gate and transition VALIDATION → PRODUCTION.

        Promotion requires:
        - Model is in VALIDATION state
        - Unified evaluation verdict is PROMOTE
        - Model meets production quality thresholds
        """
        entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found")

        if entry.state != ModelLifecycleState.VALIDATION:
            raise ValueError(
                f"Model {model_id} is {entry.state.value}, not VALIDATION"
            )

        # Run promotion checks
        config = self._promotion_config
        checks_passed = True
        failure_reasons: List[str] = []

        if config.require_unified_evaluation and evaluation_verdict != config.evaluation_verdict_must_be:
            checks_passed = False
            failure_reasons.append(f"verdict_{evaluation_verdict}_not_{config.evaluation_verdict_must_be}")

        # Use provided metrics or fall back to entry metrics
        acc = accuracy if accuracy is not None else entry.accuracy
        f1 = f1_macro if f1_macro is not None else entry.f1_macro
        fa = false_automation if false_automation is not None else entry.false_automation
        hv = high_value_errors if high_value_errors is not None else entry.high_value_errors

        if acc is not None and acc < config.min_accuracy:
            checks_passed = False
            failure_reasons.append(f"accuracy_{acc:.2f}_below_{config.min_accuracy}")

        if f1 is not None and f1 < config.min_f1_macro:
            checks_passed = False
            failure_reasons.append(f"f1_{f1:.2f}_below_{config.min_f1_macro}")

        if fa is not None and fa > config.max_false_automation:
            checks_passed = False
            failure_reasons.append(f"false_auto_{fa}_above_{config.max_false_automation}")

        if hv is not None and hv > config.max_high_value_errors:
            checks_passed = False
            failure_reasons.append(f"hv_errors_{hv}_above_{config.max_high_value_errors}")

        if not checks_passed:
            # Return to CANDIDATE (rejected at promotion)
            return self._transition(
                model_id,
                ModelLifecycleState.CANDIDATE,
                reason=f"Promotion failed: {'; '.join(failure_reasons)}",
            )

        # Retire current production model
        if self._production_model_id and self._production_model_id in self._models:
            old = self._models[self._production_model_id]
            if old.state == ModelLifecycleState.PRODUCTION:
                self._transition(
                    self._production_model_id,
                    ModelLifecycleState.ARCHIVED,
                    reason=f"Replaced by v{entry.model_version}",
                )

        # Transition to PRODUCTION
        transition = self._transition(
            model_id,
            ModelLifecycleState.PRODUCTION,
            reason=reason or "Promotion gate passed",
        )
        self._production_model_id = model_id
        return transition

    # ─────────────────────────────────────────────────────────────────────
    # Archive
    # ─────────────────────────────────────────────────────────────────────

    def archive_model(
        self, model_id: str, reason: str = "manual archive"
    ) -> LifecycleTransition:
        """Archive a model (move to ARCHIVED state)."""
        entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found")

        if entry.state == ModelLifecycleState.ARCHIVED:
            raise ValueError(f"Model {model_id} is already ARCHIVED")

        transition = self._transition(
            model_id, ModelLifecycleState.ARCHIVED, reason=reason
        )

        if self._production_model_id == model_id:
            self._production_model_id = None

        return transition

    # ─────────────────────────────────────────────────────────────────────
    # Rollback
    # ─────────────────────────────────────────────────────────────────────

    def rollback_to_previous(
        self, reason: str = "rollback"
    ) -> Optional[LifecycleTransition]:
        """Rollback from current production to the previous model.

        Returns the transition, or None if no rollback target.
        """
        if self._production_model_id is None:
            return None

        current = self._models.get(self._production_model_id)
        if current is None:
            return None

        # Find the model this version replaced
        if current.previous_model_id and current.previous_model_id in self._models:
            target = self._models[current.previous_model_id]
            if target.state == ModelLifecycleState.ARCHIVED:
                # Archive current, restore target
                self._transition(
                    self._production_model_id,
                    ModelLifecycleState.ARCHIVED,
                    reason=f"Rolled back: {reason}",
                )
                transition = self._transition(
                    current.previous_model_id,
                    ModelLifecycleState.PRODUCTION,
                    reason=f"Restored via rollback: {reason}",
                )
                self._production_model_id = current.previous_model_id
                return transition

        return None

    # ─────────────────────────────────────────────────────────────────────
    # Production Model
    # ─────────────────────────────────────────────────────────────────────

    @property
    def production_model_id(self) -> Optional[str]:
        return self._production_model_id

    def get_production_model(self) -> Optional[RegistryModelEntry]:
        """Get the current production model."""
        if self._production_model_id:
            return self._models.get(self._production_model_id)
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Transitions & Summary
    # ─────────────────────────────────────────────────────────────────────

    def _transition(
        self,
        model_id: str,
        to_state: ModelLifecycleState,
        reason: str = "",
        performed_by: str = "system",
    ) -> LifecycleTransition:
        """Perform a lifecycle state transition."""
        entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found")

        from_state = entry.state

        if not is_valid_transition(from_state, to_state):
            raise ValueError(
                f"Invalid transition: {from_state.value} → {to_state.value}"
            )

        # Apply transition
        entry.state = to_state
        now = datetime.utcnow()

        if to_state == ModelLifecycleState.VALIDATION:
            entry.validated_at = now
        elif to_state == ModelLifecycleState.PRODUCTION:
            entry.promoted_at = now
        elif to_state == ModelLifecycleState.ARCHIVED:
            entry.archived_at = now

        transition = LifecycleTransition(
            transition_id=_gen_id("TRANS"),
            model_id=model_id,
            model_version=entry.model_version,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            performed_by=performed_by,
            timestamp=now,
        )
        self._transitions.append(transition)
        return transition

    def get_transitions(
        self, model_id: Optional[str] = None
    ) -> List[LifecycleTransition]:
        """Get transition history, optionally filtered by model."""
        if model_id:
            return [t for t in self._transitions if t.model_id == model_id]
        return list(self._transitions)

    def get_summary(self) -> RegistrySummary:
        """Get a summary of the registry state."""
        all_models = list(self._models.values())
        prod = self.get_production_model()
        last_promotion = prod.promoted_at if prod else None
        return RegistrySummary(
            total_models=len(all_models),
            candidate_count=sum(1 for m in all_models if m.state == ModelLifecycleState.CANDIDATE),
            validation_count=sum(1 for m in all_models if m.state == ModelLifecycleState.VALIDATION),
            production_count=sum(1 for m in all_models if m.state == ModelLifecycleState.PRODUCTION),
            archived_count=sum(1 for m in all_models if m.state == ModelLifecycleState.ARCHIVED),
            production_model=prod,
            total_transitions=len(self._transitions),
            last_promotion_at=last_promotion,
        )
