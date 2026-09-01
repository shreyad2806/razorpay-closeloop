"""
MLflow Metrics Builder for Razorpay CloseLoop Phase 10C.

Converts existing Phase 9 metric objects into MLflow MetricsSnapshot format.

Metric definitions are CONSISTENT with:
- EvaluationMetrics (model_training.py) — classification + safety
- LearningMetrics (learning_metrics.py) — automation, precision, human review, reward, financial, verification

Safety principle:
  This builder is OBSERVATIONAL ONLY.
  It records metrics but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from typing import Optional

from app.schemas.learning_metrics import (
    AutomationMetrics,
    FinancialImpactMetrics,
    HumanReviewMetrics,
    LearningMetrics,
    PrecisionMetrics,
    RewardMetrics,
    VerificationMetrics,
)
from app.schemas.mlflow_tracking import MetricsSnapshot
from app.schemas.model_training import EvaluationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# MetricsBuilder — Convert EvaluationMetrics → MetricsSnapshot
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationMetricsBuilder:
    """Converts EvaluationMetrics (Phase 9E) into MetricsSnapshot for MLflow."""

    @staticmethod
    def build(
        eval_metrics: EvaluationMetrics,
        run_id: str,
    ) -> MetricsSnapshot:
        """Build a MetricsSnapshot from EvaluationMetrics.

        Uses the SAME metric definitions as EvaluationMetrics:
        - accuracy, precision, recall, F1 (macro + weighted)
        - per-class precision, recall, F1, support
        - confusion matrix
        - false_automation, high_value_errors, unknown_case_errors
        - verification_failure_rate
        - resolution_accuracy
        """
        # Classification metrics — direct mapping
        # Resolution metrics
        return MetricsSnapshot(
            run_id=run_id,
            # Classification
            accuracy=eval_metrics.accuracy,
            precision_macro=eval_metrics.precision_macro,
            recall_macro=eval_metrics.recall_macro,
            f1_macro=eval_metrics.f1_macro,
            precision_weighted=eval_metrics.precision_weighted,
            recall_weighted=eval_metrics.recall_weighted,
            f1_weighted=eval_metrics.f1_weighted,
            total_samples=eval_metrics.total_samples,
            # Per-class
            per_class_precision=eval_metrics.per_class_precision,
            per_class_recall=eval_metrics.per_class_recall,
            per_class_f1=eval_metrics.per_class_f1,
            per_class_support=eval_metrics.per_class_support,
            confusion_matrix=eval_metrics.confusion_matrix,
            confusion_labels=eval_metrics.confusion_labels,
            # Safety-critical
            false_automation=eval_metrics.false_automation,
            high_value_errors=eval_metrics.high_value_errors,
            unknown_case_errors=eval_metrics.unknown_case_errors,
            novel_pattern_errors=eval_metrics.novel_pattern_errors,
            verification_failure_rate=eval_metrics.verification_failure_rate,
            incorrect_auto_resolution=eval_metrics.incorrect_auto_resolution,
            # Resolution
            resolution_accuracy=eval_metrics.resolution_accuracy,
            # Derived safety rates
            false_automation_rate=(
                eval_metrics.false_automation / eval_metrics.total_samples
                if eval_metrics.total_samples > 0
                else None
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MetricsBuilder — Convert LearningMetrics → MetricsSnapshot
# ─────────────────────────────────────────────────────────────────────────────


class LearningMetricsBuilder:
    """Converts LearningMetrics (Phase 9H) into MetricsSnapshot for MLflow."""

    @staticmethod
    def build(
        learning_metrics: LearningMetrics,
        run_id: str,
    ) -> MetricsSnapshot:
        """Build a MetricsSnapshot from LearningMetrics.

        Uses the SAME metric definitions as each sub-component:
        - AutomationMetrics → automation fields
        - PrecisionMetrics → precision fields
        - HumanReviewMetrics → human review fields
        - RewardMetrics → reward fields
        - FinancialImpactMetrics → financial fields
        - VerificationMetrics → verification fields
        """
        auto = learning_metrics.automation
        prec = learning_metrics.precision
        human = learning_metrics.human_review
        reward = learning_metrics.reward
        fin = learning_metrics.financial
        ver = learning_metrics.verification

        return MetricsSnapshot(
            run_id=run_id,
            # ── Automation ──────────────────────────────────────────────
            auto_decisions=auto.auto_decisions,
            human_decisions=auto.human_decisions,
            unresolved_decisions=auto.unresolved_decisions,
            automation_rate=auto.automation_rate,
            human_review_rate=auto.human_review_rate,
            successful_auto=auto.successful_auto,
            successful_automation_rate=auto.successful_automation_rate,
            failed_auto=auto.failed_auto,
            failed_automation_rate=auto.failed_automation_rate,
            # ── Precision ───────────────────────────────────────────────
            correct_auto=prec.correct_auto,
            incorrect_auto=prec.incorrect_auto,
            precision=prec.precision,
            false_automation=prec.false_automation_count,
            false_automation_rate=prec.false_automation_rate,
            # ── Safety (derived from precision + automation) ────────────
            unsafe_decision_rate=(
                prec.false_automation_rate
                if prec.false_automation_rate is not None
                else None
            ),
            # ── Human Review ────────────────────────────────────────────
            total_human_reviews=human.total_human_reviews,
            human_corrections=human.human_corrections,
            human_rejections=human.human_rejections,
            human_approvals=human.human_approvals,
            correction_rate=human.correction_rate,
            unnecessary_escalations=human.unnecessary_escalations,
            # ── Verification ────────────────────────────────────────────
            total_executed=ver.total_executed,
            total_verified=ver.total_verified,
            total_rolled_back=ver.total_rolled_back,
            verification_success_rate=ver.verification_success_rate,
            verification_failure_rate=(
                ver.rollback_rate if ver.rollback_rate is not None else None
            ),
            rollback_rate=ver.rollback_rate,
            # ── Financial ───────────────────────────────────────────────
            total_adjustment_paise=fin.total_adjustment_paise,
            avg_adjustment_paise=fin.avg_adjustment_paise,
            max_adjustment_paise=fin.max_adjustment_paise,
            total_error_impact_paise=fin.total_error_impact_paise,
            high_value_errors=fin.high_value_error_count,
            high_value_error_impact_paise=fin.high_value_error_impact_paise,
            impact_avoided_paise=fin.impact_avoided_paise,
            discrepancy_eliminated_count=fin.discrepancy_eliminated_count,
            discrepancy_elimination_rate=fin.discrepancy_elimination_rate,
            # ── Reward ──────────────────────────────────────────────────
            avg_reward=reward.avg_reward,
            median_reward=reward.median_reward,
            reward_std=reward.reward_std,
            positive_rewards=reward.positive_rewards,
            negative_rewards=reward.negative_rewards,
            positive_reward_rate=reward.positive_rate,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Builder — from EvaluationMetrics + LearningMetrics
# ─────────────────────────────────────────────────────────────────────────────


class MLflowMetricsBuilder:
    """Unified builder that combines all metric sources into one MetricsSnapshot.

    Merges EvaluationMetrics (classification + safety) with
    LearningMetrics (automation + human review + financial + reward + verification)
    into a single comprehensive snapshot for MLflow logging.
    """

    @staticmethod
    def build_from_evaluation(
        eval_metrics: EvaluationMetrics,
        run_id: str,
    ) -> MetricsSnapshot:
        """Build from EvaluationMetrics only."""
        return EvaluationMetricsBuilder.build(eval_metrics, run_id)

    @staticmethod
    def build_from_learning(
        learning_metrics: LearningMetrics,
        run_id: str,
    ) -> MetricsSnapshot:
        """Build from LearningMetrics only."""
        return LearningMetricsBuilder.build(learning_metrics, run_id)

    @staticmethod
    def build_combined(
        eval_metrics: Optional[EvaluationMetrics],
        learning_metrics: Optional[LearningMetrics],
        run_id: str,
    ) -> MetricsSnapshot:
        """Build a combined MetricsSnapshot from both metric sources.

        When both are provided, EvaluationMetrics provides classification
        and per-class metrics, while LearningMetrics provides automation,
        human review, financial, reward, and verification metrics.

        Fields from LearningMetrics take precedence for overlapping safety
        metrics since they represent production/system-level measurements.
        """
        if eval_metrics is not None and learning_metrics is not None:
            eval_snap = EvaluationMetricsBuilder.build(eval_metrics, run_id)
            learn_snap = LearningMetricsBuilder.build(learning_metrics, run_id)

            # Merge: LearningMetrics fields override EvaluationMetrics
            # for overlapping safety/automation metrics, but EvaluationMetrics
            # provides the richer classification + per-class data.
            eval_dict = eval_snap.model_dump()
            learn_dict = learn_snap.model_dump()

            # Start with evaluation (classification + per-class)
            merged = eval_dict.copy()

            # Override with learning metrics for all non-classification fields.
            # Skip empty dicts/lists (they mean 'no data', not 'override').
            for key, value in learn_dict.items():
                if key in ("run_id", "logged_at", "custom_metrics"):
                    continue
                if value is None:
                    continue
                # Skip empty containers — they shouldn't override populated data
                if isinstance(value, (dict, list)) and len(value) == 0:
                    continue
                merged[key] = value

            # Merge custom metrics
            merged["custom_metrics"] = {
                **eval_dict.get("custom_metrics", {}),
                **learn_dict.get("custom_metrics", {}),
            }

            return MetricsSnapshot(**merged)

        elif eval_metrics is not None:
            return EvaluationMetricsBuilder.build(eval_metrics, run_id)
        elif learning_metrics is not None:
            return LearningMetricsBuilder.build(learning_metrics, run_id)
        else:
            return MetricsSnapshot(run_id=run_id)
