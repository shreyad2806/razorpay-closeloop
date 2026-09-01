"""
Batch Learning schemas for Razorpay CloseLoop Phase 9G.

Defines batch records, batch metrics, and batch comparison structures
for iterative learning across successive batches.

Safety principle:
  Batch improvement in automation rate alone is NOT success.
  The candidate must preserve safety thresholds across batches.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Batch Enums
# ─────────────────────────────────────────────────────────────────────────────

class BatchStatus(str, Enum):
    """Status of a learning batch."""
    COLLECTING = "COLLECTING"       # Feedback still being collected
    COMPLETE = "COMPLETE"           # All feedback collected
    TRAINING = "TRAINING"           # Training candidate model
    EVALUATING = "EVALUATING"       # Evaluating candidate
    COMPARING = "COMPARING"         # Comparing against previous batch
    PROMOTED = "PROMOTED"           # Candidate model promoted
    REJECTED = "REJECTED"           # Candidate model rejected


class BatchRecommendation(str, Enum):
    """Recommendation from batch comparison."""
    PROCEED = "PROCEED"             # Improvement demonstrated, continue
    ROLLBACK = "ROLLBACK"           # Regression detected, rollback
    INVESTIGATE = "INVESTIGATE"     # Mixed results, needs investigation
    HOLD = "HOLD"                   # No significant change, hold current


# ─────────────────────────────────────────────────────────────────────────────
# Batch Configuration
# ─────────────────────────────────────────────────────────────────────────────

class BatchConfig(BaseModel):
    """Configuration for a learning batch."""
    batch_size: int = Field(
        default=50,
        ge=1,
        description="Number of cases to collect in this batch",
    )
    min_feedback_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of cases requiring feedback before training",
    )
    model_algorithm: str = Field(
        default="xgboost",
        description="Algorithm to use for candidate training",
    )
    model_version_prefix: str = Field(
        default="batch",
        description="Prefix for model version naming",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Metrics
# ─────────────────────────────────────────────────────────────────────────────

class BatchMetrics(BaseModel):
    """Computed metrics for a single batch.

    Captures the full picture of what happened during this batch:
    dataset, model, policy, safety, and financial outcomes.
    """
    batch_id: str = Field(..., description="Batch identifier")

    # Dataset metrics
    dataset_size: int = Field(default=0, description="Total cases in batch")
    feedback_received: int = Field(default=0, description="Cases with feedback")
    feedback_rate: float = Field(default=0.0, description="feedback / dataset_size")

    # Decision metrics
    auto_decisions: int = Field(default=0, description="AUTO decisions in batch")
    human_decisions: int = Field(default=0, description="HUMAN_REVIEW decisions")
    unresolved_decisions: int = Field(default=0, description="UNRESOLVED decisions")
    automation_rate: float = Field(default=0.0, description="auto / total decisions")

    # Precision metrics
    correct_auto: int = Field(default=0, description="Correct AUTO decisions")
    incorrect_auto: int = Field(default=0, description="Incorrect AUTO decisions")
    precision: Optional[float] = Field(
        default=None,
        description="correct_auto / (correct_auto + incorrect_auto)",
    )

    # Safety metrics
    false_automation: int = Field(default=0, description="Incorrect AUTO decisions")
    high_value_errors: int = Field(default=0, description="Errors above HV threshold")
    verification_failures: int = Field(default=0, description="Verification failures")
    verification_failure_rate: Optional[float] = Field(
        default=None,
        description="ver_failures / auto_executed",
    )

    # Human review quality
    human_corrections: int = Field(default=0, description="Cases human corrected")
    human_rejections: int = Field(default=0, description="Cases human rejected")
    unnecessary_escalations: int = Field(
        default=0,
        description="Escalations that turned out unnecessary",
    )

    # Financial metrics
    total_financial_impact_paise: int = Field(
        default=0, description="Total financial impact",
    )
    error_impact_paise: int = Field(
        default=0,
        description="Financial impact of incorrect AUTO decisions",
    )
    avg_reward: Optional[float] = Field(
        default=None, description="Average reward in batch",
    )
    reward_std: Optional[float] = Field(
        default=None, description="Reward standard deviation",
    )

    # Model information
    model_version: Optional[str] = Field(
        default=None, description="Model version used",
    )
    policy_version: Optional[str] = Field(
        default=None, description="Policy version used",
    )
    candidate_model_version: Optional[str] = Field(
        default=None, description="Candidate model version (if trained)",
    )

    # Metadata
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When batch started collecting",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When batch completed",
    )
    training_duration_seconds: Optional[float] = Field(
        default=None, description="How long candidate training took",
    )

    def summary(self) -> str:
        prec = f"{self.precision:.1%}" if self.precision is not None else "N/A"
        vfr = (
            f"{self.verification_failure_rate:.1%}"
            if self.verification_failure_rate is not None
            else "N/A"
        )
        return (
            f"Batch {self.batch_id} | "
            f"Size: {self.dataset_size} | "
            f"Auto: {self.automation_rate:.1%} | "
            f"Precision: {prec} | "
            f"False auto: {self.false_automation} | "
            f"Ver fail: {vfr} | "
            f"Avg reward: {self.avg_reward or 'N/A'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Comparison
# ─────────────────────────────────────────────────────────────────────────────

class MetricChange(BaseModel):
    """A single metric change between two batches."""
    metric_name: str = Field(..., description="Metric name")
    previous_value: Optional[float] = Field(..., description="Previous batch value")
    current_value: Optional[float] = Field(..., description="Current batch value")
    change: Optional[float] = Field(
        default=None, description="Absolute change (current - previous)",
    )
    change_pct: Optional[float] = Field(
        default=None, description="Percentage change",
    )
    is_improvement: Optional[bool] = Field(
        default=None, description="Whether this is an improvement",
    )
    is_safety_critical: bool = Field(
        default=False, description="Whether this metric is safety-critical",
    )


class SafetyAssessment(BaseModel):
    """Safety assessment of a batch comparison."""
    checks_passed: int = Field(default=0, description="Safety checks that passed")
    checks_failed: int = Field(default=0, description="Safety checks that failed")
    safety_regressions: List[str] = Field(
        default_factory=list,
        description="Safety-critical regressions detected",
    )
    has_critical_regression: bool = Field(
        default=False,
        description="Whether any critical safety regression exists",
    )
    all_safety_maintained: bool = Field(
        default=True,
        description="Whether all safety thresholds are maintained",
    )


class BatchComparison(BaseModel):
    """Comparison between two consecutive batches."""
    comparison_id: str = Field(..., description="Unique comparison ID")
    previous_batch_id: str = Field(..., description="Previous batch ID")
    current_batch_id: str = Field(..., description="Current batch ID")

    # Previous batch metrics
    previous_metrics: BatchMetrics = Field(
        ..., description="Previous batch metrics",
    )

    # Current batch metrics
    current_metrics: BatchMetrics = Field(
        ..., description="Current batch metrics",
    )

    # Detailed changes
    changes: List[MetricChange] = Field(
        default_factory=list, description="Metric changes",
    )

    # Safety
    safety: SafetyAssessment = Field(
        default_factory=SafetyAssessment,
        description="Safety assessment",
    )

    # Improvements and regressions
    improvements: List[str] = Field(
        default_factory=list, description="Metrics that improved",
    )
    regressions: List[str] = Field(
        default_factory=list, description="Metrics that regressed",
    )

    # Recommendation
    recommendation: BatchRecommendation = Field(
        default=BatchRecommendation.HOLD,
        description="Batch comparison recommendation",
    )
    recommendation_reason: str = Field(
        default="", description="Why this recommendation",
    )

    compared_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When comparison was made",
    )

    def summary(self) -> str:
        return (
            f"Comparison: {self.previous_batch_id} → {self.current_batch_id} | "
            f"Recommendation: {self.recommendation.value} | "
            f"Improvements: {len(self.improvements)} | "
            f"Regressions: {len(self.regressions)} | "
            f"Safety: {'PASS' if self.safety.all_safety_maintained else 'FAIL'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Record
# ─────────────────────────────────────────────────────────────────────────────

class BatchRecord(BaseModel):
    """Complete record of a learning batch.

    Captures the batch lifecycle:
    COLLECTING → COMPLETE → TRAINING → EVALUATING → COMPARING → PROMOTED/REJECTED
    """
    batch_id: str = Field(..., description="Unique batch identifier")
    batch_number: int = Field(..., description="Sequential batch number (1, 2, 3, ...)")

    # Configuration
    config: BatchConfig = Field(
        default_factory=BatchConfig,
        description="Batch configuration",
    )

    # Status
    status: BatchStatus = Field(
        default=BatchStatus.COLLECTING,
        description="Batch status",
    )

    # Metrics
    metrics: Optional[BatchMetrics] = Field(
        default=None, description="Computed batch metrics",
    )

    # Comparison (set after batch N compares with batch N-1)
    comparison: Optional[BatchComparison] = Field(
        default=None, description="Comparison with previous batch",
    )

    # Candidate model
    candidate_model_id: Optional[str] = Field(
        default=None, description="Trained candidate model ID",
    )
    candidate_model_version: Optional[str] = Field(
        default=None, description="Candidate model version",
    )
    candidate_evaluated: bool = Field(
        default=False, description="Whether candidate was evaluated",
    )
    candidate_comparison: Optional[dict] = Field(
        default=None, description="Model comparison result (CandidateModelComparison dict)",
    )

    # Promotion
    promoted: bool = Field(
        default=False, description="Whether candidate was promoted",
    )
    promotion_reason: Optional[str] = Field(
        default=None, description="Why candidate was promoted/rejected",
    )

    # Timestamps
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When batch started",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When batch completed all phases",
    )

    # Case IDs collected in this batch
    case_ids: List[str] = Field(
        default_factory=list, description="Exception/case IDs in this batch",
    )

    def summary(self) -> str:
        return (
            f"Batch #{self.batch_number} ({self.batch_id}) | "
            f"Status: {self.status.value} | "
            f"Cases: {len(self.case_ids)} | "
            f"Promoted: {self.promoted}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Comparison Report
# ─────────────────────────────────────────────────────────────────────────────

class BatchReportRow(BaseModel):
    """A single row in the batch comparison report table."""
    batch_number: int = Field(..., description="Batch number")
    batch_id: str = Field(..., description="Batch ID")
    dataset_size: int = Field(default=0, description="Dataset size")
    precision: Optional[float] = Field(default=None, description="Precision")
    false_automation: int = Field(default=0, description="False automation count")
    automation_rate: float = Field(default=0.0, description="Automation rate")
    human_review_rate: float = Field(default=0.0, description="Human review rate")
    unresolved_rate: float = Field(default=0.0, description="Unresolved rate")
    verification_failure_rate: Optional[float] = Field(
        default=None, description="Verification failure rate",
    )
    avg_reward: Optional[float] = Field(default=None, description="Average reward")
    total_error_impact_paise: int = Field(
        default=0, description="Total error impact",
    )
    model_version: Optional[str] = Field(
        default=None, description="Model version",
    )
    policy_version: Optional[str] = Field(
        default=None, description="Policy version",
    )
    promoted: bool = Field(default=False, description="Whether promoted")


class BatchComparisonReport(BaseModel):
    """Complete report across all batches."""
    report_id: str = Field(..., description="Unique report ID")
    total_batches: int = Field(default=0, description="Total batches")
    total_cases: int = Field(default=0, description="Total cases across all batches")

    # Report table
    rows: List[BatchReportRow] = Field(
        default_factory=list, description="Per-batch metrics rows",
    )

    # Overall trends
    precision_trend: Optional[str] = Field(
        default=None,
        description="Overall precision trend: improving/stable/declining",
    )
    automation_trend: Optional[str] = Field(
        default=None,
        description="Overall automation trend",
    )
    safety_trend: Optional[str] = Field(
        default=None,
        description="Overall safety trend: maintained/degraded",
    )
    reward_trend: Optional[str] = Field(
        default=None,
        description="Overall reward trend",
    )

    # Summary
    improvement_demonstrated: bool = Field(
        default=False,
        description="Whether measurable improvement was demonstrated",
    )
    safety_maintained: bool = Field(
        default=True,
        description="Whether safety was maintained across all batches",
    )

    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When report was generated",
    )

    def summary(self) -> str:
        return (
            f"Report: {self.total_batches} batches | "
            f"Cases: {self.total_cases} | "
            f"Improvement: {self.improvement_demonstrated} | "
            f"Safety: {'MAINTAINED' if self.safety_maintained else 'DEGRADED'}"
        )
