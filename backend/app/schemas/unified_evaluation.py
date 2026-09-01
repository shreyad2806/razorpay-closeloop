"""
Unified Evaluation schemas for Razorpay CloseLoop Phase 10F.

Defines the complete evaluation report that compares current vs candidate models
across classification, safety, automation, financial, and resolution metrics.

MLflow should provide one place to compare model quality and financial safety.

Safety principle:
  Evaluation results are OBSERVATIONAL ONLY.
  They never authorize execution or bypass Phase 6 guardrails.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Enums
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationVerdict(str, Enum):
    """Verdict from the unified evaluation."""
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    DEFER = "DEFER"


class SafetyRegressionSeverity(str, Enum):
    """Severity of a safety regression."""
    NONE = "NONE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Metric Comparison Entry
# ─────────────────────────────────────────────────────────────────────────────


class MetricComparison(BaseModel):
    """A single metric comparison between current and candidate."""
    metric_name: str = Field(..., description="Metric name")
    current_value: Optional[float] = Field(None, description="Current model value")
    candidate_value: Optional[float] = Field(None, description="Candidate model value")
    change: Optional[float] = Field(None, description="Absolute change (candidate - current)")
    change_pct: Optional[float] = Field(None, description="Percentage change")
    is_improvement: bool = Field(default=False, description="Whether change is an improvement")
    is_safety_critical: bool = Field(default=False, description="Whether metric is safety-critical")
    higher_is_better: bool = Field(default=True, description="Whether higher values are better")


# ─────────────────────────────────────────────────────────────────────────────
# Safety Regression Check
# ─────────────────────────────────────────────────────────────────────────────


class SafetyRegressionCheck(BaseModel):
    """A single safety regression check."""
    metric_name: str = Field(..., description="Metric name")
    current_value: Optional[float] = Field(None, description="Current model value")
    candidate_value: Optional[float] = Field(None, description="Candidate model value")
    passed: bool = Field(default=True, description="Whether check passed")
    severity: SafetyRegressionSeverity = Field(
        default=SafetyRegressionSeverity.NONE,
        description="Severity if failed",
    )
    description: str = Field(default="", description="Check description")


# ─────────────────────────────────────────────────────────────────────────────
# Unified Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────


class UnifiedEvaluationReport(BaseModel):
    """Complete evaluation report comparing current vs candidate model.

    Contains all metrics, comparisons, safety checks, and promotion verdict.
    Designed to be logged as an MLflow artifact for experiment comparison.
    """
    report_id: str = Field(..., description="Unique report identifier")
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When evaluation was performed",
    )

    # Model identification
    current_model_id: Optional[str] = Field(None, description="Current model ID")
    current_model_version: Optional[str] = Field(None, description="Current model version")
    candidate_model_id: str = Field(..., description="Candidate model ID")
    candidate_model_version: str = Field(..., description="Candidate model version")

    # Dataset lineage
    dataset_version: Optional[str] = Field(None, description="Evaluation dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID for candidate")

    # ── Classification Metrics ──────────────────────────────────────────
    current_accuracy: Optional[float] = Field(None, description="Current accuracy")
    candidate_accuracy: Optional[float] = Field(None, description="Candidate accuracy")
    current_f1_macro: Optional[float] = Field(None, description="Current F1 macro")
    candidate_f1_macro: Optional[float] = Field(None, description="Candidate F1 macro")
    current_precision_macro: Optional[float] = Field(None, description="Current precision macro")
    candidate_precision_macro: Optional[float] = Field(None, description="Candidate precision macro")
    current_recall_macro: Optional[float] = Field(None, description="Current recall macro")
    candidate_recall_macro: Optional[float] = Field(None, description="Candidate recall macro")

    # ── Safety Metrics ──────────────────────────────────────────────────
    current_false_automation: int = Field(default=0, description="Current false automation")
    candidate_false_automation: int = Field(default=0, description="Candidate false automation")
    current_high_value_errors: int = Field(default=0, description="Current HV errors")
    candidate_high_value_errors: int = Field(default=0, description="Candidate HV errors")
    current_verification_failure_rate: Optional[float] = Field(
        None, description="Current verification failure rate"
    )
    candidate_verification_failure_rate: Optional[float] = Field(
        None, description="Candidate verification failure rate"
    )

    # ── Automation Metrics ──────────────────────────────────────────────
    current_automation_rate: Optional[float] = Field(None, description="Current automation rate")
    candidate_automation_rate: Optional[float] = Field(None, description="Candidate automation rate")
    current_human_review_rate: Optional[float] = Field(None, description="Current human review rate")
    candidate_human_review_rate: Optional[float] = Field(None, description="Candidate human review rate")

    # ── Resolution Metrics ──────────────────────────────────────────────
    current_resolution_accuracy: Optional[float] = Field(None, description="Current resolution accuracy")
    candidate_resolution_accuracy: Optional[float] = Field(None, description="Candidate resolution accuracy")

    # ── Financial Metrics ───────────────────────────────────────────────
    current_total_adjustment_paise: int = Field(default=0, description="Current total adjustment")
    candidate_total_adjustment_paise: int = Field(default=0, description="Candidate total adjustment")
    current_error_impact_paise: int = Field(default=0, description="Current error impact")
    candidate_error_impact_paise: int = Field(default=0, description="Candidate error impact")

    # ── Reward Metrics ──────────────────────────────────────────────────
    current_avg_reward: Optional[float] = Field(None, description="Current avg reward")
    candidate_avg_reward: Optional[float] = Field(None, description="Candidate avg reward")

    # ── Metric Comparisons ──────────────────────────────────────────────
    comparisons: List[MetricComparison] = Field(
        default_factory=list, description="Detailed metric comparisons"
    )
    improvements: List[str] = Field(
        default_factory=list, description="Metrics that improved"
    )
    regressions: List[str] = Field(
        default_factory=list, description="Metrics that regressed"
    )

    # ── Safety Checks ───────────────────────────────────────────────────
    safety_checks: List[SafetyRegressionCheck] = Field(
        default_factory=list, description="Safety regression checks"
    )
    all_safety_passed: bool = Field(
        default=True, description="Whether all safety checks passed"
    )
    safety_regression_severity: SafetyRegressionSeverity = Field(
        default=SafetyRegressionSeverity.NONE,
        description="Maximum severity of safety regressions",
    )

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict: EvaluationVerdict = Field(
        default=EvaluationVerdict.DEFER,
        description="Promotion verdict",
    )
    verdict_reason: str = Field(
        default="", description="Explanation of the verdict"
    )
    promotion_eligible: bool = Field(
        default=False, description="Whether candidate is eligible for promotion"
    )

    # ── Summary Stats ───────────────────────────────────────────────────
    total_improvements: int = Field(default=0, description="Count of improvements")
    total_regressions: int = Field(default=0, description="Count of regressions")
    safety_checks_passed: int = Field(default=0, description="Safety checks passed")
    safety_checks_failed: int = Field(default=0, description="Safety checks failed")

    def summary(self) -> str:
        return (
            f"Evaluation: {self.candidate_model_version} vs {self.current_model_version or 'none'} | "
            f"Verdict: {self.verdict.value} | "
            f"Improvements: {self.total_improvements} | "
            f"Regressions: {self.total_regressions} | "
            f"Safety: {'PASS' if self.all_safety_passed else 'FAIL'}"
        )

    def to_report_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for MLflow artifact logging."""
        return {
            "report_id": self.report_id,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "current_model": {
                "model_id": self.current_model_id,
                "version": self.current_model_version,
            },
            "candidate_model": {
                "model_id": self.candidate_model_id,
                "version": self.candidate_model_version,
                "mlflow_run_id": self.mlflow_run_id,
            },
            "dataset": {
                "version": self.dataset_version,
                "feature_schema_version": self.feature_schema_version,
            },
            "metrics": {
                "classification": {
                    "current_accuracy": self.current_accuracy,
                    "candidate_accuracy": self.candidate_accuracy,
                    "current_f1_macro": self.current_f1_macro,
                    "candidate_f1_macro": self.candidate_f1_macro,
                },
                "safety": {
                    "current_false_automation": self.current_false_automation,
                    "candidate_false_automation": self.candidate_false_automation,
                    "current_high_value_errors": self.current_high_value_errors,
                    "candidate_high_value_errors": self.candidate_high_value_errors,
                },
                "automation": {
                    "current_automation_rate": self.current_automation_rate,
                    "candidate_automation_rate": self.candidate_automation_rate,
                },
                "resolution": {
                    "current_resolution_accuracy": self.current_resolution_accuracy,
                    "candidate_resolution_accuracy": self.candidate_resolution_accuracy,
                },
                "financial": {
                    "current_total_adjustment_paise": self.current_total_adjustment_paise,
                    "candidate_total_adjustment_paise": self.candidate_total_adjustment_paise,
                },
            },
            "comparisons": [c.model_dump() for c in self.comparisons],
            "safety_checks": [s.model_dump() for s in self.safety_checks],
            "verdict": {
                "verdict": self.verdict.value,
                "reason": self.verdict_reason,
                "promotion_eligible": self.promotion_eligible,
            },
            "summary": {
                "improvements": self.total_improvements,
                "regressions": self.total_regressions,
                "safety_passed": self.safety_checks_passed,
                "safety_failed": self.safety_checks_failed,
                "safety_severity": self.safety_regression_severity.value,
            },
        }
