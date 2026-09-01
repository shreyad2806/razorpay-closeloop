"""
Model Promotion service for Razorpay CloseLoop Phase 9F.

Implements controlled model promotion with explicit safety criteria.

Safety principle:
  Promotion requires passing explicit safety criteria.
  Higher accuracy alone is NOT sufficient.
  Safety-critical metrics must NOT regress.
  Rollback is explicit and auditable.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.model_training import (
    EvaluationMetrics,
    ModelMetadata,
    ModelStatus,
)
from app.schemas.model_promotion import (
    GateCheck,
    PromotionDecision,
    PromotionGateResult,
    PromotionGateStatus,
    PromotionRecord,
    PromotionThresholds,
    RollbackRecord,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate
# ─────────────────────────────────────────────────────────────────────────────


class PromotionGate:
    """Evaluates whether a candidate model passes promotion criteria.

    Safety rules:
    - Candidate must meet minimum precision, recall, F1, accuracy
    - False automation must not exceed maximum count
    - False automation increase must not exceed allowed percentage
    - High-value errors must not increase (if configured)
    - Verification failure rate must not exceed threshold
    - Unknown case errors must not exceed maximum
    """

    def __init__(self, thresholds: Optional[PromotionThresholds] = None) -> None:
        self.thresholds = thresholds or PromotionThresholds()

    def evaluate(
        self,
        current_metrics: Optional[EvaluationMetrics],
        candidate_metrics: EvaluationMetrics,
        candidate_model_id: str,
        candidate_version: str,
        current_model_id: Optional[str] = None,
        current_version: Optional[str] = None,
    ) -> PromotionGateResult:
        """Evaluate a candidate model against promotion criteria.

        Args:
            current_metrics: Metrics of the current active model (None if no active model).
            candidate_metrics: Metrics of the candidate model.
            candidate_model_id: Candidate model ID.
            candidate_version: Candidate version string.
            current_model_id: Current model ID.
            current_version: Current version string.

        Returns:
            PromotionGateResult with checks and decision.
        """
        checks: List[GateCheck] = []
        t = self.thresholds

        # 1. Minimum precision
        checks.append(GateCheck(
            check_name="min_precision",
            status=PromotionGateStatus.PASSED if candidate_metrics.precision_macro >= t.min_precision else PromotionGateStatus.FAILED,
            current_value=current_metrics.precision_macro if current_metrics else 0.0,
            candidate_value=candidate_metrics.precision_macro,
            threshold=t.min_precision,
            description=f"Precision {candidate_metrics.precision_macro:.1%} vs min {t.min_precision:.1%}",
        ))

        # 2. Minimum recall
        checks.append(GateCheck(
            check_name="min_recall",
            status=PromotionGateStatus.PASSED if candidate_metrics.recall_macro >= t.min_recall else PromotionGateStatus.FAILED,
            current_value=current_metrics.recall_macro if current_metrics else 0.0,
            candidate_value=candidate_metrics.recall_macro,
            threshold=t.min_recall,
            description=f"Recall {candidate_metrics.recall_macro:.1%} vs min {t.min_recall:.1%}",
        ))

        # 3. Minimum F1
        checks.append(GateCheck(
            check_name="min_f1",
            status=PromotionGateStatus.PASSED if candidate_metrics.f1_macro >= t.min_f1 else PromotionGateStatus.FAILED,
            current_value=current_metrics.f1_macro if current_metrics else 0.0,
            candidate_value=candidate_metrics.f1_macro,
            threshold=t.min_f1,
            description=f"F1 {candidate_metrics.f1_macro:.1%} vs min {t.min_f1:.1%}",
        ))

        # 4. Minimum accuracy
        checks.append(GateCheck(
            check_name="min_accuracy",
            status=PromotionGateStatus.PASSED if candidate_metrics.accuracy >= t.min_accuracy else PromotionGateStatus.FAILED,
            current_value=current_metrics.accuracy if current_metrics else 0.0,
            candidate_value=candidate_metrics.accuracy,
            threshold=t.min_accuracy,
            description=f"Accuracy {candidate_metrics.accuracy:.1%} vs min {t.min_accuracy:.1%}",
        ))

        # 5. Maximum false automation
        checks.append(GateCheck(
            check_name="max_false_automation",
            status=PromotionGateStatus.PASSED if candidate_metrics.false_automation <= t.max_false_automation else PromotionGateStatus.FAILED,
            current_value=float(current_metrics.false_automation) if current_metrics else 0.0,
            candidate_value=float(candidate_metrics.false_automation),
            threshold=float(t.max_false_automation),
            description=f"False automation {candidate_metrics.false_automation} vs max {t.max_false_automation}",
        ))

        # 6. False automation increase (if current exists)
        if current_metrics and current_metrics.false_automation > 0:
            increase_pct = (
                (candidate_metrics.false_automation - current_metrics.false_automation)
                / current_metrics.false_automation
                * 100
            )
            max_allowed = t.max_false_auto_increase_pct
            checks.append(GateCheck(
                check_name="false_auto_increase",
                status=PromotionGateStatus.PASSED if increase_pct <= max_allowed else PromotionGateStatus.FAILED,
                current_value=float(current_metrics.false_automation),
                candidate_value=float(candidate_metrics.false_automation),
                threshold=max_allowed,
                description=f"False auto increase: {increase_pct:.1f}% vs max {max_allowed:.1f}%",
            ))
        elif current_metrics and current_metrics.false_automation == 0:
            # From zero — any false automation is a regression
            checks.append(GateCheck(
                check_name="false_auto_from_zero",
                status=PromotionGateStatus.PASSED if candidate_metrics.false_automation == 0 else PromotionGateStatus.FAILED,
                current_value=0.0,
                candidate_value=float(candidate_metrics.false_automation),
                threshold=0.0,
                description=f"False automation from zero: {candidate_metrics.false_automation}",
            ))

        # 7. High-value errors
        if t.no_hv_error_increase and current_metrics:
            checks.append(GateCheck(
                check_name="no_hv_error_increase",
                status=PromotionGateStatus.PASSED if candidate_metrics.high_value_errors <= current_metrics.high_value_errors else PromotionGateStatus.FAILED,
                current_value=float(current_metrics.high_value_errors),
                candidate_value=float(candidate_metrics.high_value_errors),
                threshold=float(current_metrics.high_value_errors),
                description=f"HV errors: {current_metrics.high_value_errors} → {candidate_metrics.high_value_errors}",
            ))
        else:
            checks.append(GateCheck(
                check_name="max_hv_errors",
                status=PromotionGateStatus.PASSED if candidate_metrics.high_value_errors <= t.max_high_value_errors else PromotionGateStatus.FAILED,
                current_value=float(current_metrics.high_value_errors) if current_metrics else 0.0,
                candidate_value=float(candidate_metrics.high_value_errors),
                threshold=float(t.max_high_value_errors),
                description=f"HV errors {candidate_metrics.high_value_errors} vs max {t.max_high_value_errors}",
            ))

        # 8. Verification failure rate
        if candidate_metrics.verification_failure_rate is not None:
            checks.append(GateCheck(
                check_name="max_verification_failure_rate",
                status=PromotionGateStatus.PASSED if candidate_metrics.verification_failure_rate <= t.max_verification_failure_rate else PromotionGateStatus.FAILED,
                current_value=current_metrics.verification_failure_rate if current_metrics and current_metrics.verification_failure_rate is not None else 0.0,
                candidate_value=candidate_metrics.verification_failure_rate,
                threshold=t.max_verification_failure_rate,
                description=f"Ver fail rate {candidate_metrics.verification_failure_rate:.1%} vs max {t.max_verification_failure_rate:.1%}",
            ))

        # 9. Unknown case errors
        checks.append(GateCheck(
            check_name="max_unknown_case_errors",
            status=PromotionGateStatus.PASSED if candidate_metrics.unknown_case_errors <= t.max_unknown_case_errors else PromotionGateStatus.FAILED,
            current_value=float(current_metrics.unknown_case_errors) if current_metrics else 0.0,
            candidate_value=float(candidate_metrics.unknown_case_errors),
            threshold=float(t.max_unknown_case_errors),
            description=f"Unknown errors {candidate_metrics.unknown_case_errors} vs max {t.max_unknown_case_errors}",
        ))

        # Determine result
        failed = [c.check_name for c in checks if c.status == PromotionGateStatus.FAILED]
        all_passed = len(failed) == 0

        gate_id = _gen_id("GATE")
        return PromotionGateResult(
            gate_id=gate_id,
            candidate_model_id=candidate_model_id,
            candidate_version=candidate_version,
            current_model_id=current_model_id,
            current_version=current_version,
            checks=checks,
            all_passed=all_passed,
            failed_checks=failed,
            decision=PromotionDecision.PROMOTED if all_passed else PromotionDecision.REJECTED,
            decision_reason=(
                "All safety checks passed"
                if all_passed
                else f"Failed: {', '.join(failed)}"
            ),
            thresholds=self.thresholds,
            evaluated_at=datetime.utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Manages model lifecycle: promotion, rollback, and audit.

    Maintains:
    - current active model
    - candidate models
    - promotion history
    - rollback history
    """

    def __init__(
        self,
        gate: Optional[PromotionGate] = None,
    ) -> None:
        self._gate = gate or PromotionGate()
        self._models: Dict[str, ModelMetadata] = {}
        self._promotion_history: List[PromotionRecord] = []
        self._rollback_history: List[RollbackRecord] = []
        self._active_model_id: Optional[str] = None

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get the currently active model."""
        if self._active_model_id:
            return self._models.get(self._active_model_id)
        return None

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a model in the registry."""
        self._models[metadata.model_id] = metadata

    def promote(
        self,
        candidate_id: str,
        current_metrics: Optional[EvaluationMetrics],
        candidate_metrics: EvaluationMetrics,
        performed_by: str = "system",
    ) -> PromotionRecord:
        """Attempt to promote a candidate model.

        Runs the promotion gate. If passed, promotes the candidate.
        Returns the promotion record regardless of outcome.
        """
        candidate = self._models.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate model {candidate_id} not found")

        current = self._models.get(self._active_model_id) if self._active_model_id else None

        # Run gate
        gate_result = self._gate.evaluate(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            candidate_model_id=candidate_id,
            candidate_version=candidate.version,
            current_model_id=self._active_model_id,
            current_version=current.version if current else None,
        )

        # Build metrics summaries
        old_summary = None
        if current_metrics:
            old_summary = {
                "accuracy": current_metrics.accuracy,
                "precision_macro": current_metrics.precision_macro,
                "recall_macro": current_metrics.recall_macro,
                "f1_macro": current_metrics.f1_macro,
                "false_automation": float(current_metrics.false_automation),
                "high_value_errors": float(current_metrics.high_value_errors),
            }
        new_summary = {
            "accuracy": candidate_metrics.accuracy,
            "precision_macro": candidate_metrics.precision_macro,
            "recall_macro": candidate_metrics.recall_macro,
            "f1_macro": candidate_metrics.f1_macro,
            "false_automation": float(candidate_metrics.false_automation),
            "high_value_errors": float(candidate_metrics.high_value_errors),
        }

        record = PromotionRecord(
            record_id=_gen_id("PROM"),
            gate_id=gate_result.gate_id,
            old_model_id=self._active_model_id,
            old_version=current.version if current else None,
            new_model_id=candidate_id,
            new_version=candidate.version,
            decision=gate_result.decision,
            reason=gate_result.decision_reason,
            old_metrics_summary=old_summary,
            new_metrics_summary=new_summary,
            thresholds=gate_result.thresholds,
            performed_by=performed_by,
            dataset_version=candidate.dataset_version,
            feature_schema_version=candidate.feature_schema_version,
        )

        # Apply decision
        if gate_result.decision == PromotionDecision.PROMOTED:
            # Retire old active model
            if self._active_model_id and self._active_model_id in self._models:
                old = self._models[self._active_model_id]
                old.status = ModelStatus.RETIRED
                old.retired_at = datetime.utcnow()

            # Promote candidate
            candidate.status = ModelStatus.ACTIVE
            candidate.promoted_at = datetime.utcnow()
            self._active_model_id = candidate_id
        else:
            candidate.status = ModelStatus.REJECTED

        self._promotion_history.append(record)
        return record

    def rollback(
        self,
        reason: str,
        performed_by: str = "system",
    ) -> Optional[RollbackRecord]:
        """Rollback to the previous model.

        Returns None if there is no previous model to rollback to.
        """
        if self._active_model_id is None:
            return None

        current = self._models.get(self._active_model_id)
        if current is None:
            return None

        # Find the most recent successful promotion to get the previous model
        previous_model_id = None
        for rec in reversed(self._promotion_history):
            if (
                rec.decision == PromotionDecision.PROMOTED
                and rec.old_model_id is not None
            ):
                previous_model_id = rec.old_model_id
                break

        if previous_model_id is None:
            return None

        previous = self._models.get(previous_model_id)
        if previous is None:
            return None

        # Perform rollback
        current.status = ModelStatus.RETIRED
        current.retired_at = datetime.utcnow()

        previous.status = ModelStatus.ACTIVE
        previous.promoted_at = datetime.utcnow()  # Re-promote
        self._active_model_id = previous_model_id

        record = RollbackRecord(
            record_id=_gen_id("RB"),
            rolled_back_from_id=self._active_model_id,
            rolled_back_from_version=current.version,
            restored_to_id=previous_model_id,
            restored_to_version=previous.version,
            reason=reason,
            performed_by=performed_by,
        )

        self._rollback_history.append(record)
        return record

    def get_promotion_history(self) -> List[PromotionRecord]:
        return list(self._promotion_history)

    def get_rollback_history(self) -> List[RollbackRecord]:
        return list(self._rollback_history)

    def get_all_models(self) -> List[ModelMetadata]:
        return list(self._models.values())

    def get_models_by_status(self, status: ModelStatus) -> List[ModelMetadata]:
        return [m for m in self._models.values() if m.status == status]
