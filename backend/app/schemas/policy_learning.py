"""
Policy Learning schemas for Razorpay CloseLoop Phase 9D.

Defines versioned policies, decision logs, and policy metrics.

Safety principle:
  A learned policy is a CANDIDATE.
  It is NOT automatically trusted.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Policy Enums
# ─────────────────────────────────────────────────────────────────────────────


class PolicyStatus(str, Enum):
    """Status of a policy version."""
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class PolicyPromotionDecision(str, Enum):
    """Decision on policy promotion."""
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    DEFER = "DEFER"


# ─────────────────────────────────────────────────────────────────────────────
# Policy Definition
# ─────────────────────────────────────────────────────────────────────────────


class PolicyThresholds(BaseModel):
    """Configurable thresholds for a policy version."""
    min_confidence_for_auto: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Minimum confidence for AUTO decision",
    )
    max_exposure_for_auto_paise: int = Field(
        default=25000, ge=0,
        description="Maximum exposure in paise for AUTO",
    )
    min_evidence_coverage_for_auto: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum evidence coverage for AUTO",
    )
    min_margin_for_auto: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Minimum margin over second candidate for AUTO",
    )
    allowed_risk_for_auto: List[str] = Field(
        default=["LOW"], description="Risk levels allowed for AUTO",
    )
    high_value_threshold_paise: int = Field(
        default=100000, ge=0,
        description="Above this → forced HUMAN_REVIEW",
    )
    blocked_exception_types: List[str] = Field(
        default_factory=list,
        description="Exception types that cannot auto-resolve",
    )


class PolicyDefinition(BaseModel):
    """A versioned automation policy."""
    policy_id: str = Field(..., description="Unique policy identifier")
    version: str = Field(..., description="Policy version string")
    status: PolicyStatus = Field(
        default=PolicyStatus.CANDIDATE, description="Policy status"
    )
    thresholds: PolicyThresholds = Field(
        default_factory=PolicyThresholds, description="Policy thresholds"
    )
    applicable_categories: List[str] = Field(
        default_factory=list,
        description="Exception categories this policy covers (empty = all)",
    )
    risk_limits: List[str] = Field(
        default_factory=lambda: ["LOW"],
        description="Maximum risk levels allowed",
    )
    confidence_requirements: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-category confidence requirements",
    )
    description: str = Field(
        default="", description="Human-readable policy description"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When policy was created"
    )
    created_by: str = Field(
        default="system", description="Who/what created this policy"
    )
    promoted_at: Optional[datetime] = Field(
        default=None, description="When policy was promoted to active"
    )
    retired_at: Optional[datetime] = Field(
        default=None, description="When policy was retired"
    )

    def summary(self) -> str:
        return (
            f"Policy: {self.policy_id} v{self.version} | "
            f"Status: {self.status.value} | "
            f"Auto threshold: {self.thresholds.min_confidence_for_auto:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Decision Log
# ─────────────────────────────────────────────────────────────────────────────


class PolicyDecisionLogEntry(BaseModel):
    """A single decision log entry under a specific policy."""
    log_id: str = Field(..., description="Unique log entry ID")
    policy_id: str = Field(..., description="Policy ID")
    policy_version: str = Field(..., description="Policy version at decision time")
    exception_id: str = Field(..., description="Exception identifier")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    candidate_id: Optional[str] = Field(
        default=None, description="Selected candidate ID"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence at decision")
    risk: str = Field(default="LOW", description="Risk category at decision")
    decision: str = Field(
        ..., description="AUTO, HUMAN_REVIEW, or UNRESOLVED"
    )
    reason_codes: List[str] = Field(
        default_factory=list, description="Decision reason codes"
    )
    resolution_type: Optional[str] = Field(
        default=None, description="Resolution type proposed"
    )
    financial_adjustment_paise: int = Field(
        default=0, description="Proposed adjustment in paise"
    )
    # Outcome (filled later)
    outcome_correct: Optional[bool] = Field(
        default=None, description="Whether the resolution was correct"
    )
    outcome_executed: bool = Field(
        default=False, description="Whether resolution was executed"
    )
    outcome_verified: bool = Field(
        default=False, description="Whether verification passed"
    )
    outcome_rolled_back: bool = Field(
        default=False, description="Whether execution was rolled back"
    )
    outcome_reward: Optional[float] = Field(
        default=None, description="Reward value if calculated"
    )
    human_feedback: Optional[str] = Field(
        default=None, description="APPROVE, REJECT, CORRECT, ESCALATE"
    )
    logged_at: datetime = Field(
        default_factory=datetime.utcnow, description="When decision was logged"
    )
    outcome_recorded_at: Optional[datetime] = Field(
        default=None, description="When outcome was recorded"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Metrics
# ─────────────────────────────────────────────────────────────────────────────


class PolicyMetrics(BaseModel):
    """Computed metrics for a policy over its decision log."""
    policy_id: str = Field(..., description="Policy ID")
    policy_version: str = Field(..., description="Policy version")
    total_decisions: int = Field(default=0, description="Total decisions under this policy")
    auto_decisions: int = Field(default=0, description="AUTO decisions")
    human_decisions: int = Field(default=0, description="HUMAN_REVIEW decisions")
    unresolved_decisions: int = Field(default=0, description="UNRESOLVED decisions")

    # Rates
    automation_rate: float = Field(default=0.0, description="auto / total")
    human_review_rate: float = Field(default=0.0, description="human / total")
    unresolved_rate: float = Field(default=0.0, description="unresolved / total")

    # Quality (requires outcomes)
    decisions_with_outcomes: int = Field(default=0, description="Decisions that have outcomes")
    correct_auto: int = Field(default=0, description="AUTO decisions that were correct")
    incorrect_auto: int = Field(default=0, description="AUTO decisions that were incorrect")
    precision: Optional[float] = Field(
        default=None, description="correct_auto / (correct_auto + incorrect_auto)"
    )
    false_automation: int = Field(
        default=0, description="Incorrect AUTO decisions (quality failures)"
    )

    # Verification
    auto_executed: int = Field(default=0, description="AUTO decisions that were executed")
    auto_verified: int = Field(default=0, description="Executed AUTO that verified")
    auto_rolled_back: int = Field(default=0, description="Executed AUTO that rolled back")
    verification_failure_rate: Optional[float] = Field(
        default=None, description="rollbacks / auto_executed"
    )

    # Financial
    total_exposure_paise: int = Field(
        default=0, description="Total financial exposure from AUTO decisions"
    )
    total_error_impact_paise: int = Field(
        default=0, description="Financial impact of incorrect AUTO decisions"
    )
    avg_reward: Optional[float] = Field(
        default=None, description="Average reward across decisions with rewards"
    )

    # High-value errors
    high_value_errors: int = Field(
        default=0, description="Incorrect AUTO decisions above high-value threshold"
    )

    computed_at: datetime = Field(
        default_factory=datetime.utcnow, description="When metrics were computed"
    )

    def summary(self) -> str:
        prec = f"{self.precision:.1%}" if self.precision is not None else "N/A"
        vfr = f"{self.verification_failure_rate:.1%}" if self.verification_failure_rate is not None else "N/A"
        return (
            f"Policy {self.policy_id} v{self.policy_version} | "
            f"Auto: {self.automation_rate:.1%} | "
            f"Precision: {prec} | "
            f"False auto: {self.false_automation} | "
            f"Ver fail: {vfr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Comparison
# ─────────────────────────────────────────────────────────────────────────────


class SafetyRegression(BaseModel):
    """A detected safety regression between policies."""
    metric_name: str = Field(..., description="Metric that regressed")
    current_value: float = Field(..., description="Current policy value")
    candidate_value: float = Field(..., description="Candidate policy value")
    threshold: float = Field(..., description="Acceptable threshold")
    severity: str = Field(
        default="critical", description="warning or critical"
    )
    description: str = Field(..., description="What the regression means")


class PolicyComparison(BaseModel):
    """Comparison between current and candidate policy metrics."""
    current_policy_id: str = Field(...)
    current_version: str = Field(...)
    candidate_policy_id: str = Field(...)
    candidate_version: str = Field(...)

    current_metrics: PolicyMetrics = Field(...)
    candidate_metrics: PolicyMetrics = Field(...)

    # Improvements
    improvements: List[str] = Field(
        default_factory=list, description="Metrics that improved"
    )
    regressions: List[str] = Field(
        default_factory=list, description="Metrics that regressed"
    )

    # Safety
    safety_regressions: List[SafetyRegression] = Field(
        default_factory=list, description="Safety-critical regressions"
    )
    has_safety_regression: bool = Field(
        default=False, description="Whether any safety regression exists"
    )

    # Decision
    recommendation: PolicyPromotionDecision = Field(
        ..., description="PROMOTE, REJECT, or DEFER"
    )
    recommendation_reason: str = Field(
        ..., description="Why this recommendation was made"
    )

    compared_at: datetime = Field(
        default_factory=datetime.utcnow, description="When comparison was made"
    )

    def summary(self) -> str:
        return (
            f"Comparison: {self.current_version} vs {self.candidate_version} | "
            f"Recommendation: {self.recommendation.value} | "
            f"Improvements: {len(self.improvements)} | "
            f"Regressions: {len(self.regressions)} | "
            f"Safety issues: {len(self.safety_regressions)}"
        )
