"""
Batch Learning Loop service for Razorpay CloseLoop Phase 9G.

Implements iterative batch learning across successive batches,
demonstrating measurable improvement without sacrificing safety.

Safety principle:
  Improvement in automation rate alone is NOT success.
  The candidate must preserve safety thresholds across batches.
  Phase 6 hard safety constraints remain mandatory.
"""

import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from app.schemas.batch_learning import (
    BatchComparison,
    BatchComparisonReport,
    BatchConfig,
    BatchMetrics,
    BatchRecord,
    BatchRecommendation,
    BatchReportRow,
    BatchStatus,
    MetricChange,
    SafetyAssessment,
)
from app.schemas.feedback import (
    FeedbackRecord,
    FeedbackType,
    OutcomeRecord,
)
from app.schemas.learning_dataset import (
    LearningDataset,
    LearningExample,
    SplitType,
)
from app.schemas.model_promotion import PromotionThresholds
from app.schemas.model_training import (
    EvaluationMetrics,
    ModelMetadata,
    TrainingConfig,
)
from app.schemas.reward_engine import RewardRecord
from app.services.model_promotion import PromotionGate
from app.services.model_training import ModelEvaluator, ModelTrainer


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────

class BatchMetricsCalculator:
    """Computes metrics for a batch from its outcomes, feedback, and rewards."""

    def calculate(
        self,
        batch_id: str,
        outcomes: List[OutcomeRecord],
        feedbacks: List[FeedbackRecord],
        rewards: List[RewardRecord],
        high_value_threshold: int = 10_000_000,
        model_version: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> BatchMetrics:
        """Calculate comprehensive metrics for a batch."""
        total = len(outcomes)
        if total == 0:
            return BatchMetrics(batch_id=batch_id, dataset_size=0)

        # Decision counts
        auto = sum(1 for o in outcomes if o.decision == "AUTO")
        human = sum(1 for o in outcomes if o.decision == "HUMAN_REVIEW")
        unresolved = sum(1 for o in outcomes if o.decision == "UNRESOLVED")
        automation_rate = auto / total if total > 0 else 0.0

        # Feedback
        feedback_count = len(feedbacks)
        feedback_rate = feedback_count / total if total > 0 else 0.0

        # Precision — only AUTO decisions with known correctness
        auto_outcomes = [o for o in outcomes if o.decision == "AUTO"]
        correct_auto = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.resolution_correct is True
        )
        incorrect_auto = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.resolution_correct is False
        )
        precision = (
            correct_auto / (correct_auto + incorrect_auto)
            if (correct_auto + incorrect_auto) > 0
            else None
        )

        # False automation = incorrect AUTO decisions
        false_automation = incorrect_auto

        # High-value errors
        high_value_errors = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.resolution_correct is False
            and abs(o.financial_impact.actual_adjustment_paise) >= high_value_threshold
        )

        # Verification failures
        auto_executed = sum(
            1 for o in auto_outcomes if o.actual_outcome.was_executed
        )
        auto_verified = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.was_executed and o.verification_passed
        )
        verification_failures = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.was_executed
            and o.actual_outcome.was_rolled_back
        )
        verification_failure_rate = (
            verification_failures / auto_executed
            if auto_executed > 0
            else None
        )

        # Human review quality
        human_corrections = sum(
            1 for f in feedbacks if f.feedback_type == FeedbackType.CORRECT
        )
        human_rejections = sum(
            1 for f in feedbacks if f.feedback_type == FeedbackType.REJECT
        )

        # Unnecessary escalations — escalated but actually correct
        unnecessary_escalations = sum(
            1 for o in outcomes
            if o.decision in ("HUMAN_REVIEW", "UNRESOLVED")
            and o.actual_outcome.resolution_correct is True
        )

        # Financial metrics
        total_impact = sum(
            o.financial_impact.actual_adjustment_paise for o in auto_outcomes
        )
        error_impact = sum(
            o.financial_impact.actual_adjustment_paise
            for o in auto_outcomes
            if o.actual_outcome.resolution_correct is False
        )

        # Reward metrics
        reward_values = [r.reward_value for r in rewards]
        avg_reward = (
            sum(reward_values) / len(reward_values)
            if reward_values
            else None
        )
        reward_std = (
            float(np.std(reward_values))
            if len(reward_values) > 1
            else None
        )

        return BatchMetrics(
            batch_id=batch_id,
            dataset_size=total,
            feedback_received=feedback_count,
            feedback_rate=feedback_rate,
            auto_decisions=auto,
            human_decisions=human,
            unresolved_decisions=unresolved,
            automation_rate=automation_rate,
            correct_auto=correct_auto,
            incorrect_auto=incorrect_auto,
            precision=precision,
            false_automation=false_automation,
            high_value_errors=high_value_errors,
            verification_failures=verification_failures,
            verification_failure_rate=verification_failure_rate,
            human_corrections=human_corrections,
            human_rejections=human_rejections,
            unnecessary_escalations=unnecessary_escalations,
            total_financial_impact_paise=total_impact,
            error_impact_paise=error_impact,
            avg_reward=avg_reward,
            reward_std=reward_std,
            model_version=model_version,
            policy_version=policy_version,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Comparator
# ─────────────────────────────────────────────────────────────────────────────

class BatchComparator:
    """Compares two consecutive batch metrics with safety assessment."""

    # Safety thresholds for batch comparison
    PRECISION_MIN_THRESHOLD = 0.70
    FALSE_AUTO_MAX_INCREASE_RATIO = 1.2  # Max 20% increase
    HV_ERROR_NO_INCREASE = True
    VER_FAIL_MAX_INCREASE_RATIO = 1.10   # Max 10% increase

    def compare(
        self,
        previous: BatchMetrics,
        current: BatchMetrics,
    ) -> BatchComparison:
        """Compare two batch metrics."""
        changes: List[MetricChange] = []
        improvements: List[str] = []
        regressions: List[str] = []
        safety_regressions: List[str] = []

        # 1. Precision
        changes.append(self._compare_metric(
            "precision", previous.precision, current.precision,
            higher_is_better=True, safety_critical=True,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 2. False automation (lower is better)
        changes.append(self._compare_metric(
            "false_automation",
            float(previous.false_automation),
            float(current.false_automation),
            higher_is_better=False, safety_critical=True,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 3. High-value errors (lower is better, safety critical)
        changes.append(self._compare_metric(
            "high_value_errors",
            float(previous.high_value_errors),
            float(current.high_value_errors),
            higher_is_better=False, safety_critical=True,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 4. Automation rate (informational)
        changes.append(self._compare_metric(
            "automation_rate", previous.automation_rate, current.automation_rate,
            higher_is_better=True, safety_critical=False,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 5. Human review rate (lower is better)
        prev_hr = (
            previous.human_decisions / previous.dataset_size
            if previous.dataset_size > 0 else 0.0
        )
        curr_hr = (
            current.human_decisions / current.dataset_size
            if current.dataset_size > 0 else 0.0
        )
        changes.append(self._compare_metric(
            "human_review_rate", prev_hr, curr_hr,
            higher_is_better=False, safety_critical=False,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 6. Verification failure rate
        changes.append(self._compare_metric(
            "verification_failure_rate",
            previous.verification_failure_rate,
            current.verification_failure_rate,
            higher_is_better=False, safety_critical=True,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 7. Average reward
        changes.append(self._compare_metric(
            "avg_reward", previous.avg_reward, current.avg_reward,
            higher_is_better=True, safety_critical=False,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 8. Error impact (lower is better)
        changes.append(self._compare_metric(
            "error_impact_paise",
            float(previous.error_impact_paise),
            float(current.error_impact_paise),
            higher_is_better=False, safety_critical=False,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # 9. Unnecessary escalations (lower is better)
        changes.append(self._compare_metric(
            "unnecessary_escalations",
            float(previous.unnecessary_escalations),
            float(current.unnecessary_escalations),
            higher_is_better=False, safety_critical=False,
            improvements=improvements, regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Safety assessment
        safety = self._assess_safety(previous, current, safety_regressions)

        # Recommendation
        recommendation, reason = self._determine_recommendation(
            improvements, regressions, safety,
        )

        return BatchComparison(
            comparison_id=_gen_id("BCP"),
            previous_batch_id=previous.batch_id,
            current_batch_id=current.batch_id,
            previous_metrics=previous,
            current_metrics=current,
            changes=changes,
            safety=safety,
            improvements=improvements,
            regressions=regressions,
            recommendation=recommendation,
            recommendation_reason=reason,
        )

    def _compare_metric(
        self,
        name: str,
        previous: Optional[float],
        current: Optional[float],
        higher_is_better: bool,
        safety_critical: bool,
        improvements: List[str],
        regressions: List[str],
        safety_regressions: List[str],
    ) -> MetricChange:
        """Compare a single metric and categorize the change."""
        if previous is None and current is None:
            return MetricChange(
                metric_name=name,
                previous_value=None,
                current_value=None,
                is_safety_critical=safety_critical,
            )

        change = None
        change_pct = None
        is_improvement = None

        if previous is not None and current is not None:
            change = current - previous
            if previous != 0:
                change_pct = change / abs(previous)

            if change > 0:
                is_improvement = higher_is_better
            elif change < 0:
                is_improvement = not higher_is_better
            else:
                is_improvement = True  # No change is neutral-positive

            if is_improvement and change != 0:
                improvements.append(f"{name}: {previous:.3f} → {current:.3f}")
            elif not is_improvement and change != 0:
                regressions.append(f"{name}: {previous:.3f} → {current:.3f}")
                if safety_critical:
                    safety_regressions.append(
                        f"{name}: {previous:.3f} → {current:.3f}"
                    )
        elif previous is None and current is not None:
            is_improvement = None  # Can't compare
        elif previous is not None and current is None:
            is_improvement = None

        return MetricChange(
            metric_name=name,
            previous_value=previous,
            current_value=current,
            change=change,
            change_pct=change_pct,
            is_improvement=is_improvement,
            is_safety_critical=safety_critical,
        )

    def _assess_safety(
        self,
        previous: BatchMetrics,
        current: BatchMetrics,
        safety_regressions: List[str],
    ) -> SafetyAssessment:
        """Assess safety across the batch comparison."""
        checks_passed = 0
        checks_failed = 0

        # Check precision
        if current.precision is not None:
            if current.precision >= self.PRECISION_MIN_THRESHOLD:
                checks_passed += 1
            else:
                checks_failed += 1

        # Check false automation increase
        if previous.false_automation > 0:
            ratio = (
                current.false_automation / previous.false_automation
            )
            if ratio <= self.FALSE_AUTO_MAX_INCREASE_RATIO:
                checks_passed += 1
            else:
                checks_failed += 1
        elif current.false_automation == 0:
            checks_passed += 1
        else:
            checks_failed += 1  # Introduced false auto from zero

        # Check HV errors
        if not self.HV_ERROR_NO_INCREASE:
            checks_passed += 1
        elif current.high_value_errors <= previous.high_value_errors:
            checks_passed += 1
        else:
            checks_failed += 1

        # Check verification failure rate
        if (
            previous.verification_failure_rate is not None
            and current.verification_failure_rate is not None
        ):
            if current.verification_failure_rate <= previous.verification_failure_rate * self.VER_FAIL_MAX_INCREASE_RATIO:
                checks_passed += 1
            else:
                checks_failed += 1
        else:
            checks_passed += 1  # Can't compare, assume OK

        has_critical = len(safety_regressions) > 0

        return SafetyAssessment(
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            safety_regressions=safety_regressions,
            has_critical_regression=has_critical,
            all_safety_maintained=checks_failed == 0,
        )

    def _determine_recommendation(
        self,
        improvements: List[str],
        regressions: List[str],
        safety: SafetyAssessment,
    ) -> Tuple[BatchRecommendation, str]:
        """Determine batch comparison recommendation."""
        if safety.has_critical_regression:
            return BatchRecommendation.ROLLBACK, (
                f"Critical safety regression detected: "
                f"{len(safety.safety_regressions)} safety issues"
            )

        if not safety.all_safety_maintained:
            return BatchRecommendation.INVESTIGATE, (
                f"Safety concerns detected: {safety.checks_failed} checks failed"
            )

        if len(improvements) > len(regressions):
            return BatchRecommendation.PROCEED, (
                f"Improvement demonstrated: {len(improvements)} improvements, "
                f"{len(regressions)} regressions"
            )
        elif len(improvements) == len(regressions):
            return BatchRecommendation.HOLD, (
                "Equal improvements and regressions — no clear improvement"
            )
        else:
            return BatchRecommendation.INVESTIGATE, (
                f"Regressions outweigh improvements: "
                f"{len(regressions)} regressions vs {len(improvements)} improvements"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Learning Loop
# ─────────────────────────────────────────────────────────────────────────────

class BatchLearningLoop:
    """Manages iterative batch learning across successive batches.

    Lifecycle:
    1. Start batch → COLLECTING
    2. Collect outcomes/feedback → COMPLETE
    3. Compute metrics → complete metrics
    4. Train candidate model → TRAINING
    5. Evaluate candidate → EVALUATING
    6. Compare with previous batch → COMPARING
    7. Decide: promote or reject → PROMOTED / REJECTED

    Safety principle:
    - Batch improvement in automation alone is NOT success.
    - Safety thresholds must be maintained.
    - Phase 6 hard safety constraints remain mandatory.
    """

    def __init__(self) -> None:
        self._batches: Dict[str, BatchRecord] = {}
        self._batches_by_number: Dict[int, str] = {}
        self._next_batch_number = 1
        self._metrics_calculator = BatchMetricsCalculator()
        self._comparator = BatchComparator()
        self._trainer = ModelTrainer()
        self._evaluator = ModelEvaluator()
        self._promotion_gate = PromotionGate()

    def start_batch(
        self,
        config: Optional[BatchConfig] = None,
    ) -> BatchRecord:
        """Start a new learning batch."""
        config = config or BatchConfig()
        batch_id = _gen_id("BAT")
        batch_number = self._next_batch_number
        self._next_batch_number += 1

        batch = BatchRecord(
            batch_id=batch_id,
            batch_number=batch_number,
            config=config,
            status=BatchStatus.COLLECTING,
            started_at=datetime.utcnow(),
        )

        self._batches[batch_id] = batch
        self._batches_by_number[batch_number] = batch_id
        return batch

    def add_case_to_batch(
        self,
        batch_id: str,
        case_id: str,
    ) -> bool:
        """Add a case to a collecting batch."""
        batch = self._batches.get(batch_id)
        if not batch or batch.status != BatchStatus.COLLECTING:
            return False
        if case_id not in batch.case_ids:
            batch.case_ids.append(case_id)
        return True

    def complete_collection(
        self,
        batch_id: str,
    ) -> bool:
        """Mark batch as COMPLETE (collection done)."""
        batch = self._batches.get(batch_id)
        if not batch or batch.status != BatchStatus.COLLECTING:
            return False
        batch.status = BatchStatus.COMPLETE
        batch.completed_at = datetime.utcnow()
        return True

    def compute_metrics(
        self,
        batch_id: str,
        outcomes: List[OutcomeRecord],
        feedbacks: List[FeedbackRecord],
        rewards: List[RewardRecord],
        high_value_threshold: int = 10_000_000,
        model_version: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> BatchMetrics:
        """Compute metrics for a completed batch."""
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")

        metrics = self._metrics_calculator.calculate(
            batch_id=batch_id,
            outcomes=outcomes,
            feedbacks=feedbacks,
            rewards=rewards,
            high_value_threshold=high_value_threshold,
            model_version=model_version,
            policy_version=policy_version,
        )
        batch.metrics = metrics
        return metrics

    def train_candidate(
        self,
        batch_id: str,
        dataset: LearningDataset,
        config: Optional[TrainingConfig] = None,
        model_version: Optional[str] = None,
    ) -> ModelMetadata:
        """Train a candidate model from the batch's learning dataset."""
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")

        batch.status = BatchStatus.TRAINING
        config = config or TrainingConfig(algorithm=batch.config.model_algorithm)

        mv = model_version or f"{batch.config.model_version_prefix}-v{batch.batch_number}"
        start = time.time()

        metadata = self._trainer.train(
            dataset=dataset,
            config=config,
            model_version=mv,
            model_name=f"batch_{batch.batch_number}_candidate",
        )

        duration = time.time() - start
        batch.candidate_model_id = metadata.model_id
        batch.candidate_model_version = metadata.version
        batch.status = BatchStatus.EVALUATING

        # Update metrics with model version and duration
        if batch.metrics:
            batch.metrics.candidate_model_version = metadata.version
            batch.metrics.training_duration_seconds = duration

        return metadata

    def evaluate_candidate(
        self,
        batch_id: str,
        dataset: LearningDataset,
        split: SplitType = SplitType.TEST,
        high_value_threshold: int = 10_000_000,
    ) -> EvaluationMetrics:
        """Evaluate the candidate model."""
        batch = self._batches.get(batch_id)
        if not batch or not batch.candidate_model_id:
            raise ValueError(f"Batch {batch_id} has no candidate model")

        eval_metrics = self._evaluator.evaluate(
            trainer=self._trainer,
            model_id=batch.candidate_model_id,
            dataset=dataset,
            split=split,
            high_value_threshold=high_value_threshold,
        )

        batch.candidate_evaluated = True
        return eval_metrics

    def compare_with_previous(
        self,
        batch_id: str,
    ) -> Optional[BatchComparison]:
        """Compare this batch's metrics with the previous batch."""
        batch = self._batches.get(batch_id)
        if not batch or not batch.metrics:
            raise ValueError(f"Batch {batch_id} has no metrics")

        batch.status = BatchStatus.COMPARING

        # Find previous batch
        prev_number = batch.batch_number - 1
        if prev_number < 1:
            # First batch — no comparison possible
            return None

        prev_batch_id = self._batches_by_number.get(prev_number)
        if not prev_batch_id:
            return None

        prev_batch = self._batches.get(prev_batch_id)
        if not prev_batch or not prev_batch.metrics:
            return None

        comparison = self._comparator.compare(
            previous=prev_batch.metrics,
            current=batch.metrics,
        )
        batch.comparison = comparison
        return comparison

    def promote_or_reject(
        self,
        batch_id: str,
        current_model_metrics: Optional[EvaluationMetrics] = None,
        candidate_model_metrics: Optional[EvaluationMetrics] = None,
        candidate_model_id: Optional[str] = None,
        candidate_version: Optional[str] = None,
        current_model_id: Optional[str] = None,
        current_version: Optional[str] = None,
    ) -> BatchRecord:
        """Decide whether to promote or reject the candidate model.

        Uses the promotion gate for model-level checks AND
        the batch comparison for batch-level safety.
        """
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")

        # Check batch comparison for safety
        batch_safe = True
        batch_reason = ""

        if batch.comparison:
            if batch.comparison.safety.has_critical_regression:
                batch_safe = False
                batch_reason = (
                    f"Batch comparison has critical safety regression: "
                    f"{batch.comparison.safety.safety_regressions}"
                )
            elif not batch.comparison.safety.all_safety_maintained:
                batch_safe = False
                batch_reason = (
                    f"Batch comparison has safety concerns: "
                    f"{batch.comparison.safety.checks_failed} checks failed"
                )

        # Check model-level promotion gate
        model_can_promote = True
        model_reason = ""

        if (
            current_model_metrics is not None
            and candidate_model_metrics is not None
            and candidate_model_id is not None
            and candidate_version is not None
        ):
            gate_result = self._promotion_gate.evaluate(
                current_metrics=current_model_metrics,
                candidate_metrics=candidate_model_metrics,
                candidate_model_id=candidate_model_id,
                candidate_version=candidate_version,
                current_model_id=current_model_id,
                current_version=current_version,
            )
            if not gate_result.all_passed:
                model_can_promote = False
                model_reason = (
                    f"Model gate failed: {gate_result.failed_checks}"
                )
            batch.candidate_comparison = gate_result.model_dump()

        # Final decision
        if batch_safe and model_can_promote:
            batch.promoted = True
            batch.promotion_reason = "All safety checks passed"
            batch.status = BatchStatus.PROMOTED
        else:
            batch.promoted = False
            reasons = []
            if not batch_safe:
                reasons.append(f"Batch safety: {batch_reason}")
            if not model_can_promote:
                reasons.append(f"Model gate: {model_reason}")
            batch.promotion_reason = "; ".join(reasons) if reasons else "Unknown"
            batch.status = BatchStatus.REJECTED

        return batch

    def get_batch(self, batch_id: str) -> Optional[BatchRecord]:
        return self._batches.get(batch_id)

    def get_batch_by_number(self, batch_number: int) -> Optional[BatchRecord]:
        bid = self._batches_by_number.get(batch_number)
        return self._batches.get(bid) if bid else None

    def get_all_batches(self) -> List[BatchRecord]:
        return sorted(
            self._batches.values(),
            key=lambda b: b.batch_number,
        )

    def generate_report(self) -> BatchComparisonReport:
        """Generate a batch comparison report across all batches."""
        batches = self.get_all_batches()
        if not batches:
            return BatchComparisonReport(
                report_id=_gen_id("RPT"),
                total_batches=0,
            )

        rows: List[BatchReportRow] = []
        total_cases = 0

        for batch in batches:
            m = batch.metrics
            total_cases += len(batch.case_ids)

            rows.append(BatchReportRow(
                batch_number=batch.batch_number,
                batch_id=batch.batch_id,
                dataset_size=m.dataset_size if m else 0,
                precision=m.precision if m else None,
                false_automation=m.false_automation if m else 0,
                automation_rate=m.automation_rate if m else 0.0,
                human_review_rate=(
                    m.human_decisions / m.dataset_size
                    if m and m.dataset_size > 0 else 0.0
                ),
                unresolved_rate=(
                    m.unresolved_decisions / m.dataset_size
                    if m and m.dataset_size > 0 else 0.0
                ),
                verification_failure_rate=m.verification_failure_rate if m else None,
                avg_reward=m.avg_reward if m else None,
                total_error_impact_paise=m.error_impact_paise if m else 0,
                model_version=m.model_version if m else None,
                policy_version=m.policy_version if m else None,
                promoted=batch.promoted,
            ))

        # Trend analysis
        precision_vals = [r.precision for r in rows if r.precision is not None]
        reward_vals = [r.avg_reward for r in rows if r.avg_reward is not None]
        auto_vals = [r.automation_rate for r in rows]

        precision_trend = self._compute_trend(precision_vals)
        automation_trend = self._compute_trend(auto_vals)
        reward_trend = self._compute_trend(reward_vals)

        safety_maintained = all(
            b.comparison is None
            or b.comparison.safety.all_safety_maintained
            for b in batches
            if b.batch_number > 1
        )

        improvement_demonstrated = (
            len(precision_vals) >= 2 and precision_vals[-1] > precision_vals[0]
        ) or (
            len(reward_vals) >= 2 and reward_vals[-1] > reward_vals[0]
        )

        return BatchComparisonReport(
            report_id=_gen_id("RPT"),
            total_batches=len(batches),
            total_cases=total_cases,
            rows=rows,
            precision_trend=precision_trend,
            automation_trend=automation_trend,
            safety_trend="maintained" if safety_maintained else "degraded",
            reward_trend=reward_trend,
            improvement_demonstrated=improvement_demonstrated,
            safety_maintained=safety_maintained,
        )

    def _compute_trend(self, values: List[float]) -> Optional[str]:
        """Compute trend from a list of values."""
        if len(values) < 2:
            return None
        if values[-1] > values[0] * 1.01:
            return "improving"
        elif values[-1] < values[0] * 0.99:
            return "declining"
        else:
            return "stable"
