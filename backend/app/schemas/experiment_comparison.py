"""
Experiment Comparison schemas for Razorpay CloseLoop Phase 10I.

Defines multi-dimensional comparison and ranking of MLflow training runs.

Goal: Make it easy to compare multiple model experiments and answer:
- Which model is best?
- Which model is safest?
- Which model improved automation?
- Which model reduced false automation?
- Which model performed best on rare exception types?
- Which dataset produced the best result?

IMPORTANT:
  Highest accuracy does NOT automatically mean best model.
  Multi-dimensional evaluation is required.

Safety principle:
  Comparison is OBSERVATIONAL ONLY.
  It never authorizes execution or bypasses Phase 6 guardrails.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class ComparisonDimension(str, Enum):
    """Dimensions along which experiments can be compared."""
    SAFETY = "safety"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUTOMATION_RATE = "automation_rate"
    FALSE_AUTOMATION = "false_automation"
    RESOLUTION_ACCURACY = "resolution_accuracy"
    FINANCIAL_IMPACT = "financial_impact"
    REWARD = "reward"
    VERIFICATION = "verification"
    HUMAN_EFFICIENCY = "human_efficiency"
    HIGH_VALUE_SAFETY = "high_value_safety"
    OVERALL = "overall"


class RankingStrategy(str, Enum):
    """Strategy for ranking experiments."""
    SAFETY_FIRST = "safety_first"
    BALANCED = "balanced"
    ACCURACY_FOCUSED = "accuracy_focused"
    AUTOMATION_FOCUSED = "automation_focused"
    CUSTOM = "custom"


class RunPosition(str, Enum):
    """Position of a run in a ranking."""
    BEST = "BEST"
    SECOND = "SECOND"
    THIRD = "THIRD"
    TIED = "TIED"
    WORST = "WORST"
    MIDDLE = "MIDDLE"


# ─────────────────────────────────────────────────────────────────────────────
# Run Data for Comparison
# ─────────────────────────────────────────────────────────────────────────────


class ComparisonRun(BaseModel):
    """A single run's data prepared for comparison."""
    run_id: str = Field(..., description="MLflow run ID")
    run_name: Optional[str] = Field(None, description="Human-readable run name")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="Training algorithm")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    training_examples: int = Field(default=0, description="Training examples")
    feature_count: int = Field(default=0, description="Feature count")

    # Hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    n_estimators: Optional[int] = Field(None, description="Estimator count")
    max_depth: Optional[int] = Field(None, description="Max depth")
    learning_rate: Optional[float] = Field(None, description="Learning rate")

    # Classification metrics
    accuracy: Optional[float] = Field(None, description="Accuracy")
    precision_macro: Optional[float] = Field(None, description="Precision macro")
    recall_macro: Optional[float] = Field(None, description="Recall macro")
    f1_macro: Optional[float] = Field(None, description="F1 macro")

    # Safety metrics
    false_automation: Optional[int] = Field(None, description="False automation count")
    high_value_errors: Optional[int] = Field(None, description="High-value errors")
    verification_failure_rate: Optional[float] = Field(None, description="Verification failure rate")
    unsafe_decision_rate: Optional[float] = Field(None, description="Unsafe decision rate")

    # Automation metrics
    automation_rate: Optional[float] = Field(None, description="Automation rate")
    human_review_rate: Optional[float] = Field(None, description="Human review rate")
    unresolved_rate: Optional[float] = Field(None, description="Unresolved rate")

    # Resolution metrics
    resolution_accuracy: Optional[float] = Field(None, description="Resolution accuracy")

    # Financial metrics
    total_adjustment_paise: int = Field(default=0, description="Total adjustment (paise)")
    total_error_impact_paise: int = Field(default=0, description="Error impact (paise)")
    high_value_error_impact_paise: int = Field(default=0, description="HV error impact (paise)")

    # Reward metrics
    avg_reward: Optional[float] = Field(None, description="Average reward")

    # Per-class metrics (for rare exception analysis)
    per_class_f1: Dict[str, float] = Field(default_factory=dict, description="Per-class F1")

    # Timestamps
    started_at: Optional[datetime] = Field(None, description="Run start time")
    completed_at: Optional[datetime] = Field(None, description="Run completion time")

    # Raw metrics snapshot for custom access
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Raw metrics dict")


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Comparison
# ─────────────────────────────────────────────────────────────────────────────


class MetricDiff(BaseModel):
    """Difference in a single metric between two runs."""
    metric_name: str = Field(..., description="Metric name")
    metric_display: str = Field(default="", description="Human-readable metric name")
    higher_is_better: bool = Field(default=True, description="Whether higher is better")
    is_safety_critical: bool = Field(default=False, description="Safety-critical metric")

    run_a_value: Optional[float] = Field(None, description="Run A value")
    run_b_value: Optional[float] = Field(None, description="Run B value")
    absolute_diff: Optional[float] = Field(None, description="B - A")
    percentage_diff: Optional[float] = Field(None, description="Percentage diff")

    winner: Optional[str] = Field(None, description="Which run wins this metric (run_id)")
    is_tied: bool = Field(default=False, description="Whether values are equivalent")


class PairwiseComparison(BaseModel):
    """Pairwise comparison between two runs."""
    run_a_id: str = Field(..., description="Run A ID")
    run_b_id: str = Field(..., description="Run B ID")
    run_a_name: str = Field(default="", description="Run A name")
    run_b_name: str = Field(default="", description="Run B name")

    metric_diffs: List[MetricDiff] = Field(default_factory=list, description="Metric differences")
    run_a_wins: int = Field(default=0, description="Metrics where A wins")
    run_b_wins: int = Field(default=0, description="Metrics where B wins")
    ties: int = Field(default=0, description="Tied metrics")

    parameter_diffs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Parameter differences"
    )

    safety_notes: List[str] = Field(
        default_factory=list, description="Safety-relevant observations"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dimension Score
# ─────────────────────────────────────────────────────────────────────────────


class DimensionScore(BaseModel):
    """Score of a run on a single comparison dimension."""
    dimension: ComparisonDimension = Field(..., description="Dimension")
    score: float = Field(default=0.0, description="Score (0-1 normalized)")
    rank: int = Field(default=0, description="Rank within this dimension (1=best)")
    position: RunPosition = Field(default=RunPosition.MIDDLE, description="Position label")
    contributing_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Metrics contributing to score"
    )
    explanation: str = Field(default="", description="Why this score")


# ─────────────────────────────────────────────────────────────────────────────
# Run Ranking
# ─────────────────────────────────────────────────────────────────────────────


class RunRanking(BaseModel):
    """Ranking of a single run across all dimensions."""
    run_id: str = Field(..., description="Run ID")
    run_name: str = Field(default="", description="Run name")
    model_name: str = Field(default="", description="Model name")
    model_version: str = Field(default="", description="Model version")

    # Dimension scores
    dimension_scores: List[DimensionScore] = Field(
        default_factory=list, description="Scores per dimension"
    )

    # Aggregate
    overall_score: float = Field(default=0.0, description="Weighted overall score")
    overall_rank: int = Field(default=0, description="Overall rank")

    # Safety summary
    safety_score: float = Field(default=0.0, description="Safety dimension score")
    safety_rank: int = Field(default=0, description="Safety rank")
    has_safety_regression: bool = Field(default=False, description="Any safety regression")
    safety_issues: List[str] = Field(default_factory=list, description="Safety issues found")


# ─────────────────────────────────────────────────────────────────────────────
# Complete Experiment Comparison
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentComparison(BaseModel):
    """Complete experiment comparison report.

    Contains pairwise comparisons, dimension rankings, and overall ranking.
    """
    comparison_id: str = Field(..., description="Unique comparison ID")
    comparison_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When comparison was made"
    )
    strategy: RankingStrategy = Field(
        default=RankingStrategy.SAFETY_FIRST, description="Ranking strategy"
    )

    # Run data
    runs: List[ComparisonRun] = Field(default_factory=list, description="Runs being compared")
    run_count: int = Field(default=0, description="Number of runs")

    # Pairwise comparisons
    pairwise_comparisons: List[PairwiseComparison] = Field(
        default_factory=list, description="Pairwise metric diffs"
    )

    # Dimension rankings
    rankings: List[RunRanking] = Field(default_factory=list, description="Run rankings")

    # Answer questions
    safest_run_id: Optional[str] = Field(None, description="Safest run")
    most_accurate_run_id: Optional[str] = Field(None, description="Most accurate run")
    best_automation_run_id: Optional[str] = Field(None, description="Best automation run")
    best_resolution_run_id: Optional[str] = Field(None, description="Best resolution run")
    best_reward_run_id: Optional[str] = Field(None, description="Best reward run")
    best_overall_run_id: Optional[str] = Field(None, description="Best overall run")

    # Rare exception analysis
    best_per_class: Dict[str, str] = Field(
        default_factory=dict, description="Best run per exception class"
    )

    # Dataset analysis
    best_per_dataset: Dict[str, str] = Field(
        default_factory=dict, description="Best run per dataset version"
    )

    # Safety summary
    all_runs_safe: bool = Field(default=True, description="Whether all runs are safe")
    safety_warnings: List[str] = Field(
        default_factory=list, description="Safety warnings"
    )

    # Insights
    insights: List[str] = Field(
        default_factory=list, description="Key comparison insights"
    )

    def summary(self) -> str:
        best = self.best_overall_run_id
        return (
            f"Comparison: {self.run_count} runs | "
            f"Strategy: {self.strategy.value} | "
            f"Best: {best[:8] + '...' if best else 'none'} | "
            f"Safety: {'PASS' if self.all_runs_safe else 'WARN'}"
        )
