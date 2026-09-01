"""
Learning Metrics schemas for Razorpay CloseLoop Phase 9H.

Defines comprehensive metrics for measuring whether the system is
becoming more automated WITHOUT becoming less safe.

Safety principle:
  Learning metrics measure improvement.
  They must NEVER authorize execution or bypass Phase 6 guardrails.
  High automation rate alone is NOT success — safety must be maintained.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Metric Enums
# ─────────────────────────────────────────────────────────────────────────────

class MetricTrend(str, Enum):
    """Trend direction for a metric."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SafetyVerdict(str, Enum):
    """Overall safety verdict."""
    SAFE = "SAFE"
    CONCERN = "CONCERN"
    UNSAFE = "UNSAFE"


class MetricDimension(str, Enum):
    """Dimensions for metric breakdown."""
    OVERALL = "OVERALL"
    EXCEPTION_TYPE = "EXCEPTION_TYPE"
    RISK_CATEGORY = "RISK_CATEGORY"
    MODEL_VERSION = "MODEL_VERSION"
    POLICY_VERSION = "POLICY_VERSION"
    BATCH = "BATCH"
    DECISION = "DECISION"


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Definitions
# ─────────────────────────────────────────────────────────────────────────────

class CoreMetric(BaseModel):
    """A single core metric with metadata."""
    name: str = Field(..., description="Metric name")
    value: Optional[float] = Field(default=None, description="Metric value")
    count: int = Field(default=0, description="Sample count for this metric")
    is_safety_critical: bool = Field(
        default=False, description="Whether this metric is safety-critical",
    )
    is_perfect_good: bool = Field(
        default=True,
        description="Whether higher values are always better (False = lower is better)",
    )
    description: str = Field(default="", description="Human-readable description")

    def summary(self) -> str:
        v = f"{self.value:.4f}" if self.value is not None else "N/A"
        return f"{self.name}: {v} (n={self.count})"


# ─────────────────────────────────────────────────────────────────────────────
# Automation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class AutomationMetrics(BaseModel):
    """Metrics measuring automation behavior."""
    total_exceptions: int = Field(default=0, description="Total exceptions processed")
    eligible_exceptions: int = Field(
        default=0,
        description="Exceptions eligible for automation (excludes blocked/unknown)",
    )

    # Decision distribution
    auto_decisions: int = Field(default=0, description="AUTO decisions")
    human_decisions: int = Field(default=0, description="HUMAN_REVIEW decisions")
    unresolved_decisions: int = Field(default=0, description="UNRESOLVED decisions")

    # Rates
    automation_rate: float = Field(
        default=0.0,
        description="auto_decisions / eligible_exceptions (safe automation only)",
    )
    human_review_rate: float = Field(
        default=0.0,
        description="human_decisions / total_exceptions",
    )
    unresolved_rate: float = Field(
        default=0.0,
        description="unresolved_decisions / total_exceptions",
    )

    # Successful automation (verified + correct)
    successful_auto: int = Field(
        default=0,
        description="AUTO decisions that were executed, verified, and correct",
    )
    successful_automation_rate: float = Field(
        default=0.0,
        description="successful_auto / eligible_exceptions",
    )

    # Failed automation
    failed_auto: int = Field(
        default=0,
        description="AUTO decisions that failed execution or verification",
    )
    failed_automation_rate: float = Field(
        default=0.0,
        description="failed_auto / auto_decisions",
    )

    def summary(self) -> str:
        return (
            f"Auto: {self.automation_rate:.1%} | "
            f"Human: {self.human_review_rate:.1%} | "
            f"Unresolved: {self.unresolved_rate:.1%} | "
            f"Successful auto: {self.successful_automation_rate:.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Precision Metrics
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionMetrics(BaseModel):
    """Metrics measuring correctness of automated decisions."""
    # Core precision
    correct_auto: int = Field(default=0, description="Correct AUTO decisions")
    incorrect_auto: int = Field(default=0, description="Incorrect AUTO decisions")
    total_auto_with_outcome: int = Field(
        default=0, description="AUTO decisions with known outcome",
    )
    precision: Optional[float] = Field(
        default=None,
        description="correct_auto / total_auto_with_outcome",
    )

    # False automation (safety-critical)
    false_automation_count: int = Field(
        default=0,
        description="Incorrect AUTO decisions (alias for safety tracking)",
    )
    false_automation_rate: Optional[float] = Field(
        default=None,
        description="incorrect_auto / total_auto_with_outcome",
    )

    # Per-class precision
    per_exception_precision: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Precision per exception type",
    )
    per_exception_correct: Dict[str, int] = Field(
        default_factory=dict,
        description="Correct count per exception type",
    )
    per_exception_incorrect: Dict[str, int] = Field(
        default_factory=dict,
        description="Incorrect count per exception type",
    )

    def summary(self) -> str:
        prec = f"{self.precision:.1%}" if self.precision is not None else "N/A"
        far = (
            f"{self.false_automation_rate:.1%}"
            if self.false_automation_rate is not None
            else "N/A"
        )
        return f"Precision: {prec} | False auto rate: {far} | Count: {self.false_automation_count}"


# ─────────────────────────────────────────────────────────────────────────────
# Human Review Metrics
# ─────────────────────────────────────────────────────────────────────────────

class HumanReviewMetrics(BaseModel):
    """Metrics measuring human review effectiveness."""
    total_human_reviews: int = Field(
        default=0, description="Total cases sent to human review",
    )
    human_corrections: int = Field(
        default=0, description="Cases where human corrected the system",
    )
    human_rejections: int = Field(
        default=0, description="Cases where human rejected the resolution",
    )
    human_approvals: int = Field(
        default=0, description="Cases where human approved the system",
    )
    human_escalations: int = Field(
        default=0, description="Cases where human escalated further",
    )

    # Quality signals
    correction_rate: Optional[float] = Field(
        default=None,
        description="human_corrections / total_human_reviews",
    )
    unnecessary_escalations: int = Field(
        default=0,
        description="Escalated cases that turned out correct (system was right)",
    )
    unnecessary_escalation_rate: Optional[float] = Field(
        default=None,
        description="unnecessary_escalations / total_human_reviews",
    )

    def summary(self) -> str:
        cr = f"{self.correction_rate:.1%}" if self.correction_rate is not None else "N/A"
        return (
            f"Reviews: {self.total_human_reviews} | "
            f"Corrections: {self.human_corrections} ({cr}) | "
            f"Unnecessary esc: {self.unnecessary_escalations}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reward Metrics
# ─────────────────────────────────────────────────────────────────────────────

class RewardMetrics(BaseModel):
    """Metrics measuring reward distribution."""
    total_rewards: int = Field(default=0, description="Total rewards computed")
    avg_reward: Optional[float] = Field(default=None, description="Mean reward")
    median_reward: Optional[float] = Field(default=None, description="Median reward")
    reward_std: Optional[float] = Field(default=None, description="Reward std dev")
    min_reward: Optional[float] = Field(default=None, description="Minimum reward")
    max_reward: Optional[float] = Field(default=None, description="Maximum reward")

    # Positive/negative breakdown
    positive_rewards: int = Field(default=0, description="Positive reward count")
    negative_rewards: int = Field(default=0, description="Negative reward count")
    neutral_rewards: int = Field(default=0, description="Zero reward count")
    positive_rate: Optional[float] = Field(
        default=None, description="positive / total",
    )

    # By category
    rewards_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Average reward per RewardCategory",
    )

    # By exception type
    rewards_by_exception_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Average reward per exception type",
    )

    # By risk level
    rewards_by_risk: Dict[str, float] = Field(
        default_factory=dict,
        description="Average reward per risk level",
    )

    # By model version
    rewards_by_model: Dict[str, float] = Field(
        default_factory=dict,
        description="Average reward per model version",
    )

    def summary(self) -> str:
        avg = f"{self.avg_reward:.3f}" if self.avg_reward is not None else "N/A"
        return (
            f"Avg reward: {avg} | "
            f"Positive: {self.positive_rewards} | "
            f"Negative: {self.negative_rewards}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Financial Impact Metrics
# ─────────────────────────────────────────────────────────────────────────────

class FinancialImpactMetrics(BaseModel):
    """Metrics measuring financial impact of automated resolutions."""
    total_adjustment_paise: int = Field(
        default=0, description="Total financial adjustment from AUTO decisions",
    )
    avg_adjustment_paise: Optional[float] = Field(
        default=None, description="Average adjustment per AUTO decision",
    )
    max_adjustment_paise: int = Field(
        default=0, description="Maximum single adjustment",
    )

    # Error impact
    total_error_impact_paise: int = Field(
        default=0,
        description="Financial impact of incorrect AUTO decisions",
    )
    avg_error_impact_paise: Optional[float] = Field(
        default=None, description="Average error impact",
    )

    # High-value errors
    high_value_error_count: int = Field(
        default=0, description="Incorrect AUTO above high-value threshold",
    )
    high_value_error_impact_paise: int = Field(
        default=0, description="Total impact of high-value errors",
    )

    # Impact avoided through escalation
    impact_avoided_paise: int = Field(
        default=0,
        description="Financial impact that would have occurred if escalated cases were auto-resolved incorrectly",
    )

    # Discrepancy resolution
    discrepancy_eliminated_count: int = Field(
        default=0, description="Cases where discrepancy was eliminated",
    )
    discrepancy_elimination_rate: Optional[float] = Field(
        default=None,
        description="discrepancy_eliminated / total_auto_executed",
    )

    def summary(self) -> str:
        total = f"₹{self.total_adjustment_paise // 100}"
        errors = f"₹{self.total_error_impact_paise // 100}"
        return (
            f"Total adjustment: {total} | "
            f"Error impact: {errors} | "
            f"HV errors: {self.high_value_error_count}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification Metrics
# ─────────────────────────────────────────────────────────────────────────────

class VerificationMetrics(BaseModel):
    """Metrics measuring verification behavior."""
    total_executed: int = Field(default=0, description="Total executed resolutions")
    total_verified: int = Field(default=0, description="Total verified resolutions")
    total_rolled_back: int = Field(default=0, description="Total rolled-back resolutions")
    total_verification_failed: int = Field(
        default=0, description="Total verification failures (not counting rollback)",
    )

    verification_success_rate: Optional[float] = Field(
        default=None,
        description="total_verified / total_executed",
    )
    rollback_rate: Optional[float] = Field(
        default=None,
        description="total_rolled_back / total_executed",
    )

    def summary(self) -> str:
        vsr = (
            f"{self.verification_success_rate:.1%}"
            if self.verification_success_rate is not None
            else "N/A"
        )
        rr = (
            f"{self.rollback_rate:.1%}"
            if self.rollback_rate is not None
            else "N/A"
        )
        return (
            f"Executed: {self.total_executed} | "
            f"Verified: {vsr} | "
            f"Rollback: {rr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety Assessment
# ─────────────────────────────────────────────────────────────────────────────

class SafetyMetricStatus(BaseModel):
    """Status of a single safety metric."""
    metric_name: str = Field(..., description="Metric name")
    value: Optional[float] = Field(default=None, description="Current value")
    threshold: Optional[float] = Field(default=None, description="Safety threshold")
    passed: bool = Field(default=True, description="Whether safety check passed")
    description: str = Field(default="", description="What this check means")


class SafetyAssessmentResult(BaseModel):
    """Overall safety assessment across all safety-critical metrics."""
    verdict: SafetyVerdict = Field(
        default=SafetyVerdict.SAFE, description="Overall safety verdict",
    )
    checks: List[SafetyMetricStatus] = Field(
        default_factory=list, description="Individual safety checks",
    )
    checks_passed: int = Field(default=0, description="Checks that passed")
    checks_failed: int = Field(default=0, description="Checks that failed")
    critical_failures: List[str] = Field(
        default_factory=list,
        description="Names of critical safety failures",
    )

    def summary(self) -> str:
        return (
            f"Verdict: {self.verdict.value} | "
            f"Passed: {self.checks_passed} | "
            f"Failed: {self.checks_failed}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────

class MetricTrendAnalysis(BaseModel):
    """Trend analysis for a single metric across time/batches."""
    metric_name: str = Field(..., description="Metric name")
    trend: MetricTrend = Field(
        default=MetricTrend.INSUFFICIENT_DATA, description="Trend direction",
    )
    values: List[float] = Field(
        default_factory=list, description="Historical values (oldest to newest)",
    )
    change_from_first: Optional[float] = Field(
        default=None, description="Change from first to last value",
    )
    change_from_previous: Optional[float] = Field(
        default=None, description="Change from second-to-last to last value",
    )
    is_safety_critical: bool = Field(
        default=False, description="Whether declining trend is a safety concern",
    )

    def summary(self) -> str:
        first = self.values[0] if self.values else None
        last = self.values[-1] if self.values else None
        return (
            f"{self.metric_name}: {self.trend.value} "
            f"({first:.3f} → {last:.3f})"
            if first is not None and last is not None
            else f"{self.metric_name}: {self.trend.value}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

class MetricComparisonEntry(BaseModel):
    """A single metric in a comparison."""
    metric_name: str = Field(..., description="Metric name")
    current_value: Optional[float] = Field(default=None, description="Current value")
    candidate_value: Optional[float] = Field(default=None, description="Candidate value")
    change: Optional[float] = Field(default=None, description="Absolute change")
    is_improvement: Optional[bool] = Field(default=None, description="Is improvement")
    is_safety_critical: bool = Field(default=False, description="Safety critical?")


class LearningMetricsComparison(BaseModel):
    """Comparison between two sets of learning metrics."""
    comparison_id: str = Field(..., description="Unique comparison ID")
    current_label: str = Field(..., description="Label for current metrics")
    candidate_label: str = Field(..., description="Label for candidate metrics")

    entries: List[MetricComparisonEntry] = Field(
        default_factory=list, description="Metric comparisons",
    )

    improvements: List[str] = Field(
        default_factory=list, description="Improved metrics",
    )
    regressions: List[str] = Field(
        default_factory=list, description="Regressed metrics",
    )
    safety_regressions: List[str] = Field(
        default_factory=list, description="Safety-critical regressions",
    )

    overall_improvement: bool = Field(
        default=False, description="Net improvement demonstrated",
    )
    safety_maintained: bool = Field(
        default=True, description="Safety maintained in comparison",
    )

    compared_at: datetime = Field(
        default_factory=datetime.utcnow, description="When comparison was made",
    )

    def summary(self) -> str:
        return (
            f"Comparison: {self.current_label} → {self.candidate_label} | "
            f"Improvements: {len(self.improvements)} | "
            f"Regressions: {len(self.regressions)} | "
            f"Safety: {'OK' if self.safety_maintained else 'BREACH'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Complete Learning Metrics
# ─────────────────────────────────────────────────────────────────────────────

class LearningMetrics(BaseModel):
    """Complete learning metrics snapshot.

    Captures the full picture of system performance at a point in time.
    Designed to measure whether the system is becoming more automated
    WITHOUT becoming less safe.
    """
    metrics_id: str = Field(..., description="Unique metrics snapshot ID")

    # Core metric groups
    automation: AutomationMetrics = Field(
        default_factory=AutomationMetrics,
        description="Automation behavior metrics",
    )
    precision: PrecisionMetrics = Field(
        default_factory=PrecisionMetrics,
        description="Correctness metrics",
    )
    human_review: HumanReviewMetrics = Field(
        default_factory=HumanReviewMetrics,
        description="Human review metrics",
    )
    reward: RewardMetrics = Field(
        default_factory=RewardMetrics,
        description="Reward distribution metrics",
    )
    financial: FinancialImpactMetrics = Field(
        default_factory=FinancialImpactMetrics,
        description="Financial impact metrics",
    )
    verification: VerificationMetrics = Field(
        default_factory=VerificationMetrics,
        description="Verification behavior metrics",
    )
    safety: SafetyAssessmentResult = Field(
        default_factory=SafetyAssessmentResult,
        description="Safety assessment",
    )

    # Trend analysis
    trends: List[MetricTrendAnalysis] = Field(
        default_factory=list, description="Trend analyses for key metrics",
    )

    # Source metadata
    source_type: str = Field(
        default="overall",
        description="Source: overall, batch, model_version, policy_version",
    )
    source_id: Optional[str] = Field(
        default=None, description="Source identifier (batch_id, model_id, etc.)",
    )

    # Timestamps
    computed_at: datetime = Field(
        default_factory=datetime.utcnow, description="When metrics were computed",
    )
    period_start: Optional[datetime] = Field(
        default=None, description="Start of the metrics period",
    )
    period_end: Optional[datetime] = Field(
        default=None, description="End of the metrics period",
    )

    def summary(self) -> str:
        return (
            f"Metrics: {self.metrics_id} | "
            f"{self.automation.summary()} | "
            f"{self.precision.summary()} | "
            f"{self.safety.summary()}"
        )

    def is_safe(self) -> bool:
        """Check if the system is in a safe state."""
        return self.safety.verdict == SafetyVerdict.SAFE
