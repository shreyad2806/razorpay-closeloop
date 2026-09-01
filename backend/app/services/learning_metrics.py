"""
Learning Metrics service for Razorpay CloseLoop Phase 9H.

Computes comprehensive metrics from outcomes, feedback, rewards,
and verification results.

Safety principle:
  Learning metrics measure improvement.
  They must NEVER authorize execution or bypass Phase 6 guardrails.
  High automation rate alone is NOT success — safety must be maintained.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from app.schemas.feedback import (
    FeedbackRecord,
    FeedbackType,
    OutcomeRecord,
)
from app.schemas.learning_metrics import (
    AutomationMetrics,
    CoreMetric,
    FinancialImpactMetrics,
    HumanReviewMetrics,
    LearningMetrics,
    LearningMetricsComparison,
    MetricComparisonEntry,
    MetricTrend,
    MetricTrendAnalysis,
    PrecisionMetrics,
    RewardMetrics,
    SafetyAssessmentResult,
    SafetyMetricStatus,
    SafetyVerdict,
    VerificationMetrics,
)
from app.schemas.reward_engine import RewardCategory, RewardRecord


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Thresholds (configurable)
# ─────────────────────────────────────────────────────────────────────────────

class SafetyThresholds:
    """Configurable safety thresholds for metric assessment."""

    def __init__(
        self,
        min_precision: float = 0.70,
        max_false_automation_rate: float = 0.15,
        max_verification_failure_rate: float = 0.10,
        max_high_value_errors: int = 0,
        max_unresolved_rate: float = 0.30,
        high_value_threshold_paise: int = 10_000_000,
    ) -> None:
        self.min_precision = min_precision
        self.max_false_automation_rate = max_false_automation_rate
        self.max_verification_failure_rate = max_verification_failure_rate
        self.max_high_value_errors = max_high_value_errors
        self.max_unresolved_rate = max_unresolved_rate
        self.high_value_threshold_paise = high_value_threshold_paise


# ─────────────────────────────────────────────────────────────────────────────
# Core Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────


class LearningMetricsService:
    """Computes comprehensive learning metrics.

    Takes outcomes, feedbacks, and rewards as input.
    Produces a complete LearningMetrics snapshot.
    """

    def __init__(
        self,
        safety_thresholds: Optional[SafetyThresholds] = None,
    ) -> None:
        self.thresholds = safety_thresholds or SafetyThresholds()

    def compute(
        self,
        outcomes: List[OutcomeRecord],
        feedbacks: List[FeedbackRecord],
        rewards: List[RewardRecord],
        source_type: str = "overall",
        source_id: Optional[str] = None,
        previous_metrics: Optional[LearningMetrics] = None,
    ) -> LearningMetrics:
        """Compute complete learning metrics from raw data."""
        metrics_id = _gen_id("LM")

        automation = self._compute_automation(outcomes)
        precision = self._compute_precision(outcomes)
        human_review = self._compute_human_review(outcomes, feedbacks)
        reward_metrics = self._compute_reward_metrics(rewards, outcomes)
        financial = self._compute_financial(outcomes)
        verification = self._compute_verification(outcomes)
        safety = self._assess_safety(
            automation, precision, verification, financial,
        )
        trends = self._compute_trends(previous_metrics, automation, precision, reward_metrics, financial)

        return LearningMetrics(
            metrics_id=metrics_id,
            automation=automation,
            precision=precision,
            human_review=human_review,
            reward=reward_metrics,
            financial=financial,
            verification=verification,
            safety=safety,
            trends=trends,
            source_type=source_type,
            source_id=source_id,
        )

    # ── Automation ───────────────────────────────────────────────────────

    def _compute_automation(self, outcomes: List[OutcomeRecord]) -> AutomationMetrics:
        total = len(outcomes)
        if total == 0:
            return AutomationMetrics()

        auto = sum(1 for o in outcomes if o.decision == "AUTO")
        human = sum(1 for o in outcomes if o.decision == "HUMAN_REVIEW")
        unresolved = sum(1 for o in outcomes if o.decision == "UNRESOLVED")

        # Eligible = total minus explicitly blocked/unsafe
        eligible = total  # All exceptions are eligible for evaluation

        # Successful auto = executed + verified + correct
        auto_outcomes = [o for o in outcomes if o.decision == "AUTO"]
        successful = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.was_executed
            and o.verification_passed
            and o.actual_outcome.resolution_correct is True
        )

        # Failed auto
        failed = sum(
            1 for o in auto_outcomes
            if o.actual_outcome.was_executed
            and (not o.verification_passed or o.actual_outcome.was_rolled_back)
        )

        return AutomationMetrics(
            total_exceptions=total,
            eligible_exceptions=eligible,
            auto_decisions=auto,
            human_decisions=human,
            unresolved_decisions=unresolved,
            automation_rate=auto / eligible if eligible > 0 else 0.0,
            human_review_rate=human / total if total > 0 else 0.0,
            unresolved_rate=unresolved / total if total > 0 else 0.0,
            successful_auto=successful,
            successful_automation_rate=successful / eligible if eligible > 0 else 0.0,
            failed_auto=failed,
            failed_automation_rate=failed / auto if auto > 0 else 0.0,
        )

    # ── Precision ────────────────────────────────────────────────────────

    def _compute_precision(self, outcomes: List[OutcomeRecord]) -> PrecisionMetrics:
        auto_outcomes = [
            o for o in outcomes
            if o.decision == "AUTO" and o.actual_outcome.resolution_correct is not None
        ]

        correct = sum(1 for o in auto_outcomes if o.actual_outcome.resolution_correct is True)
        incorrect = sum(1 for o in auto_outcomes if o.actual_outcome.resolution_correct is False)
        total_with_outcome = len(auto_outcomes)

        precision = (
            correct / total_with_outcome if total_with_outcome > 0 else None
        )
        false_auto_rate = (
            incorrect / total_with_outcome if total_with_outcome > 0 else None
        )

        # Per-exception precision
        per_exc_correct: Dict[str, int] = {}
        per_exc_incorrect: Dict[str, int] = {}
        per_exc_precision: Dict[str, Optional[float]] = {}

        for o in auto_outcomes:
            exc_type = o.prediction.exception_type or "UNKNOWN"
            if o.actual_outcome.resolution_correct is True:
                per_exc_correct[exc_type] = per_exc_correct.get(exc_type, 0) + 1
            elif o.actual_outcome.resolution_correct is False:
                per_exc_incorrect[exc_type] = per_exc_incorrect.get(exc_type, 0) + 1

        for exc_type in set(list(per_exc_correct.keys()) + list(per_exc_incorrect.keys())):
            c = per_exc_correct.get(exc_type, 0)
            inc = per_exc_incorrect.get(exc_type, 0)
            per_exc_precision[exc_type] = c / (c + inc) if (c + inc) > 0 else None

        return PrecisionMetrics(
            correct_auto=correct,
            incorrect_auto=incorrect,
            total_auto_with_outcome=total_with_outcome,
            precision=precision,
            false_automation_count=incorrect,
            false_automation_rate=false_auto_rate,
            per_exception_precision=per_exc_precision,
            per_exception_correct=per_exc_correct,
            per_exception_incorrect=per_exc_incorrect,
        )

    # ── Human Review ─────────────────────────────────────────────────────

    def _compute_human_review(
        self,
        outcomes: List[OutcomeRecord],
        feedbacks: List[FeedbackRecord],
    ) -> HumanReviewMetrics:
        human_outcomes = [
            o for o in outcomes if o.decision == "HUMAN_REVIEW"
        ]
        total_reviews = len(human_outcomes)

        # From feedback records
        corrections = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.CORRECT)
        rejections = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.REJECT)
        approvals = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.APPROVE)
        escalations = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.ESCALATE)

        # Unnecessary escalations = human review where system was actually correct
        unnecessary = sum(
            1 for o in human_outcomes
            if o.actual_outcome.resolution_correct is True
        )

        correction_rate = (
            corrections / total_reviews if total_reviews > 0 else None
        )
        unnecessary_rate = (
            unnecessary / total_reviews if total_reviews > 0 else None
        )

        return HumanReviewMetrics(
            total_human_reviews=total_reviews,
            human_corrections=corrections,
            human_rejections=rejections,
            human_approvals=approvals,
            human_escalations=escalations,
            correction_rate=correction_rate,
            unnecessary_escalations=unnecessary,
            unnecessary_escalation_rate=unnecessary_rate,
        )

    # ── Reward ───────────────────────────────────────────────────────────

    def _compute_reward_metrics(
        self,
        rewards: List[RewardRecord],
        outcomes: List[OutcomeRecord],
    ) -> RewardMetrics:
        if not rewards:
            return RewardMetrics()

        values = [r.reward_value for r in rewards]
        arr = np.array(values)

        positive = sum(1 for v in values if v > 0)
        negative = sum(1 for v in values if v < 0)
        neutral = sum(1 for v in values if v == 0.0)

        # By category
        by_category: Dict[str, List[float]] = {}
        for r in rewards:
            cat = r.category.value
            by_category.setdefault(cat, []).append(r.reward_value)
        avg_by_category = {
            k: float(np.mean(v)) for k, v in by_category.items()
        }

        # By exception type (from outcomes)
        outcome_by_exc: Dict[str, List[float]] = {}
        reward_by_id = {r.reward_id: r for r in rewards}
        for o in outcomes:
            # Match rewards to outcomes via exception_id
            exc_type = o.prediction.exception_type or "UNKNOWN"
            for r in rewards:
                if r.exception_id == o.exception_id:
                    outcome_by_exc.setdefault(exc_type, []).append(r.reward_value)
        avg_by_exception = {
            k: float(np.mean(v)) for k, v in outcome_by_exc.items()
        }

        # By risk level
        by_risk: Dict[str, List[float]] = {}
        for r in rewards:
            risk = r.financial_risk_level.value
            by_risk.setdefault(risk, []).append(r.reward_value)
        avg_by_risk = {k: float(np.mean(v)) for k, v in by_risk.items()}

        # By model version
        by_model: Dict[str, List[float]] = {}
        for r in rewards:
            mv = r.model_version or "unknown"
            by_model.setdefault(mv, []).append(r.reward_value)
        avg_by_model = {k: float(np.mean(v)) for k, v in by_model.items()}

        return RewardMetrics(
            total_rewards=len(rewards),
            avg_reward=float(np.mean(arr)),
            median_reward=float(np.median(arr)),
            reward_std=float(np.std(arr)) if len(arr) > 1 else None,
            min_reward=float(np.min(arr)),
            max_reward=float(np.max(arr)),
            positive_rewards=positive,
            negative_rewards=negative,
            neutral_rewards=neutral,
            positive_rate=positive / len(values) if values else None,
            rewards_by_category=avg_by_category,
            rewards_by_exception_type=avg_by_exception,
            rewards_by_risk=avg_by_risk,
            rewards_by_model=avg_by_model,
        )

    # ── Financial Impact ─────────────────────────────────────────────────

    def _compute_financial(self, outcomes: List[OutcomeRecord]) -> FinancialImpactMetrics:
        auto_outcomes = [o for o in outcomes if o.decision == "AUTO"]

        if not auto_outcomes:
            # Still compute impact avoided from human-review cases
            human_correct = [
                o for o in outcomes
                if o.decision == "HUMAN_REVIEW"
                and o.actual_outcome.resolution_correct is True
            ]
            avoided = sum(
                abs(o.financial_impact.actual_adjustment_paise) for o in human_correct
            )
            return FinancialImpactMetrics(impact_avoided_paise=avoided)

        adjustments = [
            o.financial_impact.actual_adjustment_paise for o in auto_outcomes
        ]
        abs_adjustments = [abs(a) for a in adjustments]

        # Error impact (incorrect AUTO)
        error_outcomes = [
            o for o in auto_outcomes
            if o.actual_outcome.resolution_correct is False
        ]
        error_impacts = [
            abs(o.financial_impact.actual_adjustment_paise) for o in error_outcomes
        ]

        # High-value errors
        hv_threshold = self.thresholds.high_value_threshold_paise
        hv_errors = [
            o for o in error_outcomes
            if abs(o.financial_impact.actual_adjustment_paise) >= hv_threshold
        ]
        hv_impact = sum(
            abs(o.financial_impact.actual_adjustment_paise) for o in hv_errors
        )

        # Discrepancy elimination
        executed = [o for o in auto_outcomes if o.actual_outcome.was_executed]
        eliminated = sum(
            1 for o in executed
            if o.financial_impact.discrepancy_eliminated
        )

        # Impact avoided (from HUMAN_REVIEW cases that were correct escalations)
        # Approximation: sum of adjustments from human-reviewed correct cases
        human_correct = [
            o for o in outcomes
            if o.decision == "HUMAN_REVIEW"
            and o.actual_outcome.resolution_correct is True
        ]
        avoided = sum(
            abs(o.financial_impact.actual_adjustment_paise) for o in human_correct
        )

        return FinancialImpactMetrics(
            total_adjustment_paise=sum(abs_adjustments),
            avg_adjustment_paise=float(np.mean(abs_adjustments)) if abs_adjustments else None,
            max_adjustment_paise=max(abs_adjustments) if abs_adjustments else 0,
            total_error_impact_paise=sum(error_impacts),
            avg_error_impact_paise=float(np.mean(error_impacts)) if error_impacts else None,
            high_value_error_count=len(hv_errors),
            high_value_error_impact_paise=hv_impact,
            impact_avoided_paise=avoided,
            discrepancy_eliminated_count=eliminated,
            discrepancy_elimination_rate=(
                eliminated / len(executed) if executed else None
            ),
        )

    # ── Verification ─────────────────────────────────────────────────────

    def _compute_verification(self, outcomes: List[OutcomeRecord]) -> VerificationMetrics:
        executed = [o for o in outcomes if o.actual_outcome.was_executed]

        if not executed:
            return VerificationMetrics()

        verified = sum(1 for o in executed if o.verification_passed)
        rolled_back = sum(1 for o in executed if o.actual_outcome.was_rolled_back)
        ver_failed = sum(
            1 for o in executed
            if not o.verification_passed and not o.actual_outcome.was_rolled_back
        )

        return VerificationMetrics(
            total_executed=len(executed),
            total_verified=verified,
            total_rolled_back=rolled_back,
            total_verification_failed=ver_failed,
            verification_success_rate=verified / len(executed) if executed else None,
            rollback_rate=rolled_back / len(executed) if executed else None,
        )

    # ── Safety Assessment ────────────────────────────────────────────────

    def _assess_safety(
        self,
        automation: AutomationMetrics,
        precision: PrecisionMetrics,
        verification: VerificationMetrics,
        financial: FinancialImpactMetrics,
    ) -> SafetyAssessmentResult:
        t = self.thresholds
        checks: List[SafetyMetricStatus] = []
        failed_names: List[str] = []

        # 1. Precision
        prec_val = precision.precision if precision.precision is not None else 1.0
        prec_passed = prec_val >= t.min_precision
        checks.append(SafetyMetricStatus(
            metric_name="precision",
            value=precision.precision,
            threshold=t.min_precision,
            passed=prec_passed,
            description=f"Precision {prec_val:.1%} vs min {t.min_precision:.1%}",
        ))
        if not prec_passed:
            failed_names.append("precision")

        # 2. False automation rate
        far_val = (
            precision.false_automation_rate
            if precision.false_automation_rate is not None
            else 0.0
        )
        far_passed = far_val <= t.max_false_automation_rate
        checks.append(SafetyMetricStatus(
            metric_name="false_automation_rate",
            value=far_val,
            threshold=t.max_false_automation_rate,
            passed=far_passed,
            description=f"False auto rate {far_val:.1%} vs max {t.max_false_automation_rate:.1%}",
        ))
        if not far_passed:
            failed_names.append("false_automation_rate")

        # 3. Verification failure rate
        vfr_val = (
            verification.rollback_rate
            if verification.rollback_rate is not None
            else 0.0
        )
        vfr_passed = vfr_val <= t.max_verification_failure_rate
        checks.append(SafetyMetricStatus(
            metric_name="verification_failure_rate",
            value=vfr_val,
            threshold=t.max_verification_failure_rate,
            passed=vfr_passed,
            description=f"Rollback rate {vfr_val:.1%} vs max {t.max_verification_failure_rate:.1%}",
        ))
        if not vfr_passed:
            failed_names.append("verification_failure_rate")

        # 4. High-value errors
        hv_passed = financial.high_value_error_count <= t.max_high_value_errors
        checks.append(SafetyMetricStatus(
            metric_name="high_value_errors",
            value=float(financial.high_value_error_count),
            threshold=float(t.max_high_value_errors),
            passed=hv_passed,
            description=(
                f"HV errors {financial.high_value_error_count} vs max {t.max_high_value_errors}"
            ),
        ))
        if not hv_passed:
            failed_names.append("high_value_errors")

        # 5. Unresolved rate
        unr_val = automation.unresolved_rate
        unr_passed = unr_val <= t.max_unresolved_rate
        checks.append(SafetyMetricStatus(
            metric_name="unresolved_rate",
            value=unr_val,
            threshold=t.max_unresolved_rate,
            passed=unr_passed,
            description=f"Unresolved rate {unr_val:.1%} vs max {t.max_unresolved_rate:.1%}",
        ))
        if not unr_passed:
            failed_names.append("unresolved_rate")

        # Determine verdict
        passed = sum(1 for c in checks if c.passed)
        failed = sum(1 for c in checks if not c.passed)

        if failed == 0:
            verdict = SafetyVerdict.SAFE
        elif any(n in ("precision", "false_automation_rate", "high_value_errors") for n in failed_names):
            verdict = SafetyVerdict.UNSAFE
        else:
            verdict = SafetyVerdict.CONCERN

        return SafetyAssessmentResult(
            verdict=verdict,
            checks=checks,
            checks_passed=passed,
            checks_failed=failed,
            critical_failures=failed_names,
        )

    # ── Trend Analysis ───────────────────────────────────────────────────

    def _compute_trends(
        self,
        previous: Optional[LearningMetrics],
        automation: AutomationMetrics,
        precision: PrecisionMetrics,
        reward: RewardMetrics,
        financial: FinancialImpactMetrics,
    ) -> List[MetricTrendAnalysis]:
        if previous is None:
            return []

        trends: List[MetricTrendAnalysis] = []

        # Automation rate trend
        trends.append(self._analyze_trend(
            "automation_rate",
            [previous.automation.automation_rate, automation.automation_rate],
            higher_is_better=False,  # Higher automation is not always better
            is_safety_critical=False,
        ))

        # Precision trend
        prev_prec = previous.precision.precision
        curr_prec = precision.precision
        if prev_prec is not None and curr_prec is not None:
            trends.append(self._analyze_trend(
                "precision",
                [prev_prec, curr_prec],
                higher_is_better=True,
                is_safety_critical=True,
            ))

        # False automation trend
        prev_fa = previous.precision.false_automation_count
        curr_fa = precision.false_automation_count
        trends.append(self._analyze_trend(
            "false_automation",
            [float(prev_fa), float(curr_fa)],
            higher_is_better=False,
            is_safety_critical=True,
        ))

        # Reward trend
        if previous.reward.avg_reward is not None and reward.avg_reward is not None:
            trends.append(self._analyze_trend(
                "avg_reward",
                [previous.reward.avg_reward, reward.avg_reward],
                higher_is_better=True,
                is_safety_critical=False,
            ))

        # Error impact trend
        prev_ei = float(previous.financial.total_error_impact_paise)
        curr_ei = float(financial.total_error_impact_paise)
        trends.append(self._analyze_trend(
            "error_impact",
            [prev_ei, curr_ei],
            higher_is_better=False,
            is_safety_critical=False,
        ))

        return trends

    def _analyze_trend(
        self,
        name: str,
        values: List[float],
        higher_is_better: bool,
        is_safety_critical: bool,
    ) -> MetricTrendAnalysis:
        if len(values) < 2:
            return MetricTrendAnalysis(
                metric_name=name,
                trend=MetricTrend.INSUFFICIENT_DATA,
                values=values,
                is_safety_critical=is_safety_critical,
            )

        first = values[0]
        last = values[-1]
        change = last - first

        if first == 0:
            pct = None
        else:
            pct = change / abs(first) if first != 0 else None

        if change == 0:
            trend = MetricTrend.STABLE
        elif (change > 0) == higher_is_better:
            trend = MetricTrend.IMPROVING
        else:
            trend = MetricTrend.DECLINING

        return MetricTrendAnalysis(
            metric_name=name,
            trend=trend,
            values=values,
            change_from_first=change,
            change_from_previous=values[-1] - values[-2] if len(values) >= 2 else None,
            is_safety_critical=is_safety_critical,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Comparator
# ─────────────────────────────────────────────────────────────────────────────


class LearningMetricsComparator:
    """Compares two LearningMetrics snapshots."""

    SAFETY_CRITICAL_METRICS = {
        "precision",
        "false_automation_rate",
        "high_value_errors",
        "verification_failure_rate",
    }

    def compare(
        self,
        current: LearningMetrics,
        candidate: LearningMetrics,
        current_label: str = "current",
        candidate_label: str = "candidate",
    ) -> LearningMetricsComparison:
        """Compare two sets of learning metrics."""
        entries: List[MetricComparisonEntry] = []
        improvements: List[str] = []
        regressions: List[str] = []
        safety_regressions: List[str] = []

        # Precision
        entries.append(self._compare_single(
            "precision",
            current.precision.precision,
            candidate.precision.precision,
            higher_is_better=True,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # False automation rate
        entries.append(self._compare_single(
            "false_automation_rate",
            current.precision.false_automation_rate,
            candidate.precision.false_automation_rate,
            higher_is_better=False,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Automation rate
        entries.append(self._compare_single(
            "automation_rate",
            current.automation.automation_rate,
            candidate.automation.automation_rate,
            higher_is_better=True,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Successful automation rate
        entries.append(self._compare_single(
            "successful_automation_rate",
            current.automation.successful_automation_rate,
            candidate.automation.successful_automation_rate,
            higher_is_better=True,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # High-value errors
        entries.append(self._compare_single(
            "high_value_errors",
            float(current.financial.high_value_error_count),
            float(candidate.financial.high_value_error_count),
            higher_is_better=False,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Avg reward
        entries.append(self._compare_single(
            "avg_reward",
            current.reward.avg_reward,
            candidate.reward.avg_reward,
            higher_is_better=True,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Error impact
        entries.append(self._compare_single(
            "error_impact",
            float(current.financial.total_error_impact_paise),
            float(candidate.financial.total_error_impact_paise),
            higher_is_better=False,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Verification success rate
        entries.append(self._compare_single(
            "verification_success_rate",
            current.verification.verification_success_rate,
            candidate.verification.verification_success_rate,
            higher_is_better=True,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # Unresolved rate
        entries.append(self._compare_single(
            "unresolved_rate",
            current.automation.unresolved_rate,
            candidate.automation.unresolved_rate,
            higher_is_better=False,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        # False automation count
        entries.append(self._compare_single(
            "false_automation_count",
            float(current.precision.false_automation_count),
            float(candidate.precision.false_automation_count),
            higher_is_better=False,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
        ))

        overall_improvement = len(improvements) > len(regressions)
        safety_maintained = len(safety_regressions) == 0

        return LearningMetricsComparison(
            comparison_id=_gen_id("LMC"),
            current_label=current_label,
            candidate_label=candidate_label,
            entries=entries,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
            overall_improvement=overall_improvement,
            safety_maintained=safety_maintained,
        )

    def _compare_single(
        self,
        name: str,
        current_val: Optional[float],
        candidate_val: Optional[float],
        higher_is_better: bool,
        improvements: List[str],
        regressions: List[str],
        safety_regressions: List[str],
    ) -> MetricComparisonEntry:
        is_safety = name in self.SAFETY_CRITICAL_METRICS

        if current_val is None and candidate_val is None:
            return MetricComparisonEntry(
                metric_name=name, is_safety_critical=is_safety,
            )

        if current_val is None or candidate_val is None:
            return MetricComparisonEntry(
                metric_name=name,
                current_value=current_val,
                candidate_value=candidate_val,
                is_safety_critical=is_safety,
            )

        change = candidate_val - current_val
        is_improvement = None

        if change > 0:
            is_improvement = higher_is_better
        elif change < 0:
            is_improvement = not higher_is_better
        else:
            is_improvement = True  # No change is neutral

        if is_improvement and change != 0:
            improvements.append(f"{name}: {current_val:.4f} → {candidate_val:.4f}")
        elif not is_improvement and change != 0:
            msg = f"{name}: {current_val:.4f} → {candidate_val:.4f}"
            regressions.append(msg)
            if is_safety:
                safety_regressions.append(msg)

        return MetricComparisonEntry(
            metric_name=name,
            current_value=current_val,
            candidate_value=candidate_val,
            change=change,
            is_improvement=is_improvement,
            is_safety_critical=is_safety,
        )
