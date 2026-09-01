"""
Learning Dataset schemas for Razorpay CloseLoop Phase 9C.

Defines the learning example structure connecting:
exception → features → evidence → prediction → resolution → guardrail →
execution → verification → feedback → actual resolution → reward

Key safety principle:
  Feature snapshots preserve the features that existed at decision time.
  Do not recompute historical features using future information.
  Avoid temporal leakage.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Constants
# ─────────────────────────────────────────────────────────────────────────────

LEARNING_DATASET_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Snapshot
# ─────────────────────────────────────────────────────────────────────────────


class FeatureSnapshot(BaseModel):
    """Features as they existed at decision time.

    CRITICAL: These are frozen at the moment the guardrail decision was made.
    They must NOT be recomputed from later information.
    This prevents temporal leakage.
    """
    financial_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Financial features at decision time",
    )
    structural_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Structural features at decision time",
    )
    evidence_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Evidence features at decision time",
    )
    temporal_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Temporal features at decision time",
    )
    merchant_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Merchant features at decision time",
    )
    historical_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Historical features at decision time",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Feature schema version used at decision time",
    )
    captured_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this snapshot was captured (decision time)",
    )

    def to_flat_dict(self) -> Dict[str, float]:
        """Flatten all feature categories into a single dict."""
        result = {}
        for category_dict in [
            self.financial_features,
            self.structural_features,
            self.evidence_features,
            self.temporal_features,
            self.merchant_features,
            self.historical_features,
        ]:
            result.update(category_dict)
        return result

    def feature_count(self) -> int:
        return len(self.to_flat_dict())

    def has_missing_features(self, required: List[str]) -> List[str]:
        """Check which required features are missing."""
        flat = self.to_flat_dict()
        return [f for f in required if f not in flat]


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────


class LearningLabels(BaseModel):
    """Labels for a learning example.

    Each label captures a different aspect of the outcome.
    These are ground-truth signals for offline learning ONLY.
    """
    # Exception classification labels
    true_exception_type: Optional[str] = Field(
        default=None, description="Ground-truth exception type"
    )
    predicted_exception_type: Optional[str] = Field(
        default=None, description="What the system predicted"
    )
    exception_prediction_correct: Optional[bool] = Field(
        default=None, description="Whether exception prediction was correct"
    )

    # Resolution labels
    true_resolution: Optional[str] = Field(
        default=None, description="Ground-truth resolution type"
    )
    predicted_resolution: Optional[str] = Field(
        default=None, description="What the system predicted as resolution"
    )
    resolution_correct: Optional[bool] = Field(
        default=None, description="Whether resolution prediction was correct"
    )

    # Escalation labels
    escalation_correct: Optional[bool] = Field(
        default=None, description="Whether escalation was the right call"
    )

    # Verification labels
    verification_passed: bool = Field(
        default=False, description="Whether verification passed"
    )

    # Human correction labels
    human_corrected: bool = Field(
        default=False, description="Whether human corrected the system"
    )
    human_rejected: bool = Field(
        default=False, description="Whether human rejected the resolution"
    )

    # Financial labels
    discrepancy_eliminated: bool = Field(
        default=False, description="Whether discrepancy was eliminated"
    )
    unintended_changes: int = Field(
        default=0, description="Number of unintended financial changes"
    )

    # Resolvability (evaluation only)
    resolvable: Optional[bool] = Field(
        default=None, description="Whether case was truly resolvable"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Learning Example
# ─────────────────────────────────────────────────────────────────────────────


class LearningExample(BaseModel):
    """A single learning example for offline model training/evaluation.

    Connects:
      exception → feature snapshot → labels → reward → lineage

    Feature snapshot is frozen at decision time.
    Labels capture ground truth for offline evaluation.
    """
    # Identity
    example_id: str = Field(..., description="Unique learning example ID")
    case_id: str = Field(..., description="Source case identifier")
    exception_id: str = Field(..., description="Source exception identifier")
    workflow_id: str = Field(..., description="Source workflow identifier")

    # Feature snapshot (frozen at decision time)
    features: FeatureSnapshot = Field(
        ..., description="Features as they existed at decision time"
    )

    # Labels
    labels: LearningLabels = Field(
        ..., description="Ground-truth labels for learning"
    )

    # Reward
    reward_value: Optional[float] = Field(
        default=None, description="Reward value from Phase 9B (-1.0 to 1.0)"
    )
    reward_category: Optional[str] = Field(
        default=None, description="Reward category from Phase 9B"
    )

    # Decision context (what the system decided)
    guardrail_decision: Optional[str] = Field(
        default=None, description="Guardrail decision: AUTO, HUMAN_REVIEW, UNRESOLVED"
    )
    confidence: Optional[float] = Field(
        default=None, description="System confidence at decision time"
    )
    risk: Optional[str] = Field(
        default=None, description="Risk category at decision time"
    )

    # Timestamps
    decision_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the guardrail decision was made",
    )
    outcome_time: Optional[datetime] = Field(
        default=None, description="When the final outcome was recorded"
    )

    # Data lineage
    lineage_exception_id: Optional[str] = Field(
        default=None, description="Reference to source exception"
    )
    lineage_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence record IDs"
    )
    lineage_execution_id: Optional[str] = Field(
        default=None, description="Execution record ID"
    )
    lineage_verification_id: Optional[str] = Field(
        default=None, description="Verification record ID"
    )
    lineage_feedback_id: Optional[str] = Field(
        default=None, description="Feedback record ID"
    )
    lineage_reward_id: Optional[str] = Field(
        default=None, description="Reward record ID"
    )

    def summary(self) -> str:
        return (
            f"Example: {self.example_id} | "
            f"Case: {self.case_id} | "
            f"Decision: {self.guardrail_decision} | "
            f"Correct: {self.labels.resolution_correct} | "
            f"Reward: {self.reward_value}"
        )

    def is_valid(self) -> bool:
        """Check if this example has minimum required data for learning."""
        has_features = self.features.feature_count() > 0
        has_any_label = (
            self.labels.true_exception_type is not None
            or self.labels.true_resolution is not None
            or self.labels.resolution_correct is not None
        )
        return has_features and has_any_label


# ─────────────────────────────────────────────────────────────────────────────
# Data Split
# ─────────────────────────────────────────────────────────────────────────────


class SplitType(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataSplit(BaseModel):
    """A dataset split with metadata."""
    split_type: SplitType = Field(..., description="Split type")
    example_ids: List[str] = Field(
        default_factory=list, description="IDs of examples in this split"
    )
    example_count: int = Field(default=0, description="Number of examples")
    label_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count per exception type"
    )
    split_strategy: str = Field(
        default="temporal", description="How the split was created"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When split was created"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quality Report
# ─────────────────────────────────────────────────────────────────────────────


class QualityIssue(str, Enum):
    """Types of data quality issues."""
    MISSING_FEATURES = "missing_features"
    MISSING_LABELS = "missing_labels"
    DUPLICATE_EXAMPLES = "duplicate_examples"
    CONTRADICTORY_LABELS = "contradictory_labels"
    INVALID_FINANCIAL_VALUES = "invalid_financial_values"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_OUTCOME = "missing_outcome"
    MISSING_VERIFICATION = "missing_verification"
    TEMPORAL_LEAKAGE = "temporal_leakage"


class QualityIssueRecord(BaseModel):
    """A single quality issue detected."""
    issue_type: QualityIssue = Field(..., description="Type of issue")
    example_id: Optional[str] = Field(default=None, description="Affected example")
    description: str = Field(..., description="What the issue is")
    severity: str = Field(
        default="warning", description="severity: warning, error, critical"
    )


class QualityReport(BaseModel):
    """Data quality report for the learning dataset."""
    total_examples: int = Field(default=0, description="Total examples checked")
    valid_examples: int = Field(default=0, description="Examples passing all checks")
    issues: List[QualityIssueRecord] = Field(
        default_factory=list, description="Detected issues"
    )
    issues_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count by issue type"
    )
    quality_score: float = Field(
        default=0.0, description="Overall quality score (0.0 to 1.0)"
    )
    checked_at: datetime = Field(
        default_factory=datetime.utcnow, description="When check was performed"
    )

    def has_critical_issues(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    def summary(self) -> str:
        return (
            f"Quality: {self.quality_score:.2f} | "
            f"Valid: {self.valid_examples}/{self.total_examples} | "
            f"Issues: {len(self.issues)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Learning Dataset
# ─────────────────────────────────────────────────────────────────────────────


class LearningDataset(BaseModel):
    """Complete learning dataset with examples, splits, and quality report."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    version: str = Field(
        default=LEARNING_DATASET_VERSION, description="Dataset version"
    )
    feature_schema_version: str = Field(
        default="1.0.0", description="Feature schema version used"
    )

    # Examples
    examples: List[LearningExample] = Field(
        default_factory=list, description="Learning examples"
    )

    # Splits
    splits: Dict[str, DataSplit] = Field(
        default_factory=dict, description="Train/validation/test splits"
    )

    # Quality
    quality_report: Optional[QualityReport] = Field(
        default=None, description="Data quality report"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When dataset was created"
    )
    split_seed: int = Field(
        default=42, description="Random seed for reproducible splits"
    )

    def get_examples_by_split(self, split_type: SplitType) -> List[LearningExample]:
        """Get examples for a specific split."""
        split = self.splits.get(split_type.value)
        if not split:
            return []
        id_set = set(split.example_ids)
        return [e for e in self.examples if e.example_id in id_set]

    def example_count(self) -> int:
        return len(self.examples)

    def split_summary(self) -> Dict[str, int]:
        """Count examples per split."""
        return {k: v.example_count for k, v in self.splits.items()}
