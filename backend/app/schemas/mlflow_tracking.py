"""
MLflow Experiment Tracking schemas for Razorpay CloseLoop Phase 10A.

Defines the configuration, experiment structure, and run tracking
for model development reproducibility and auditability.

Milestone: We can prove which model produced which result.

Safety principle:
  MLflow tracking is OBSERVATIONAL ONLY.
  It records what happened but never influences financial decisions.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking.

    All environment-specific values are configurable.
    Defaults use a local development setup.
    """
    tracking_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking URI (file path or HTTP server)",
    )
    experiment_name: str = Field(
        default="razorpay-closeloop",
        description="Default MLflow experiment name",
    )
    artifact_root: Optional[str] = Field(
        default=None,
        description="Artifact storage root (None = default from tracking URI)",
    )
    tags_prefix: str = Field(
        default="closeloop.",
        description="Prefix for custom MLflow tags",
    )

    class Config:
        frozen = True


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Types
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentType(str, Enum):
    """Types of MLflow experiments."""
    EXCEPTION_CLASSIFICATION = "exception_classification"
    RESOLUTION_PREDICTION = "resolution_prediction"
    FEEDBACK_LEARNING = "feedback_learning"
    POLICY_COMPARISON = "policy_comparison"


# ─────────────────────────────────────────────────────────────────────────────
# Environment Metadata
# ─────────────────────────────────────────────────────────────────────────────


class EnvironmentMetadata(BaseModel):
    """Reproducibility metadata captured at training time.

    Records everything needed to reproduce an experiment:
    git state, Python version, dependency versions, platform.
    """
    python_version: str = Field(default="unknown", description="Python version")
    platform: str = Field(default="unknown", description="OS platform")
    hostname: Optional[str] = Field(None, description="Machine hostname")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    git_branch: Optional[str] = Field(None, description="Git branch name")
    git_dirty: Optional[bool] = Field(None, description="Whether repo has uncommitted changes")
    mlflow_version: Optional[str] = Field(None, description="MLflow version")
    numpy_version: Optional[str] = Field(None, description="NumPy version")
    pandas_version: Optional[str] = Field(None, description="Pandas version")
    sklearn_version: Optional[str] = Field(None, description="Scikit-learn version")
    xgboost_version: Optional[str] = Field(None, description="XGBoost version")
    torch_version: Optional[str] = Field(None, description="PyTorch version (if available)")
    key_dependency_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Other notable dependency versions",
    )
    collected_at: Optional[datetime] = Field(
        None, description="When metadata was collected"
    )

    def to_params_dict(self) -> Dict[str, str]:
        """Convert to flat dict for MLflow parameter logging."""
        d: Dict[str, str] = {}
        for k, v in self.model_dump().items():
            if k == "collected_at":
                continue
            if v is not None:
                d[f"env.{k}"] = str(v)
        for dep, ver in self.key_dependency_versions.items():
            d[f"env.dep.{dep}"] = ver
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Run Metadata
# ─────────────────────────────────────────────────────────────────────────────


class RunStatus(str, Enum):
    """Status of an MLflow run."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class MLflowRunMetadata(BaseModel):
    """Metadata captured for an MLflow training run.

    Records the full provenance chain:
    experiment → run → model → dataset → config → metrics
    """
    run_id: str = Field(..., description="MLflow run ID")
    run_name: Optional[str] = Field(None, description="Human-readable run name")
    experiment_name: str = Field(..., description="MLflow experiment name")
    experiment_id: Optional[str] = Field(None, description="MLflow experiment ID")

    # Model provenance
    model_type: str = Field(..., description="Type of model trained")
    model_name: str = Field(..., description="Human-readable model name")
    model_version: str = Field(..., description="Model version string")
    algorithm: str = Field(..., description="Training algorithm used")

    # Dataset provenance
    dataset_id: Optional[str] = Field(None, description="Training dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    training_examples: int = Field(default=0, description="Number of training examples")
    validation_examples: int = Field(default=0, description="Number of validation examples")
    test_examples: int = Field(default=0, description="Number of test examples")
    total_examples: int = Field(default=0, description="Total dataset size")
    feature_count: int = Field(default=0, description="Number of features")
    label_classes: List[str] = Field(
        default_factory=list, description="Label classes in the dataset"
    )
    target_label_version: Optional[str] = Field(
        None, description="Version of the target label schema"
    )

    # Training config (model parameters)
    training_config: Dict[str, Any] = Field(
        default_factory=dict, description="Full training configuration"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    random_seed: Optional[int] = Field(None, description="Random seed used")

    # Algorithm-specific parameters
    n_estimators: Optional[int] = Field(None, description="Number of estimators (ensemble)")
    max_depth: Optional[int] = Field(None, description="Max tree depth")
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    subsample: Optional[float] = Field(None, description="Subsample ratio")
    colsample_bytree: Optional[float] = Field(None, description="Feature subsample ratio")
    class_weight: Optional[str] = Field(None, description="Class weighting scheme")
    early_stopping_rounds: Optional[int] = Field(
        None, description="Early stopping rounds"
    )

    # Split configuration
    train_ratio: Optional[float] = Field(None, description="Training split ratio")
    val_ratio: Optional[float] = Field(None, description="Validation split ratio")
    test_ratio: Optional[float] = Field(None, description="Test split ratio")
    max_features: Optional[int] = Field(None, description="Max features used")

    # Environment / reproducibility
    environment_metadata: Optional[EnvironmentMetadata] = Field(
        None, description="Reproducibility metadata"
    )

    # Timing
    started_at: Optional[datetime] = Field(None, description="Run start time")
    completed_at: Optional[datetime] = Field(None, description="Run completion time")
    duration_seconds: Optional[float] = Field(None, description="Run duration")

    # Status
    status: RunStatus = Field(default=RunStatus.RUNNING, description="Run status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Environment
    environment: str = Field(default="local", description="Environment name")

    def summary(self) -> str:
        return (
            f"Run: {self.run_id[:8]}... | "
            f"Model: {self.model_name} v{self.model_version} | "
            f"Algorithm: {self.algorithm} | "
            f"Status: {self.status.value}"
        )

    def get_all_params(self) -> Dict[str, str]:
        """Collect all parameters into a flat dict for MLflow logging.

        Returns a comprehensive dictionary of every tracked parameter.
        """
        params: Dict[str, str] = {}

        # Model provenance
        params["model_type"] = self.model_type
        params["model_name"] = self.model_name
        params["model_version"] = self.model_version
        params["algorithm"] = self.algorithm

        # Dataset provenance
        if self.dataset_id:
            params["dataset_id"] = self.dataset_id
        if self.dataset_version:
            params["dataset_version"] = self.dataset_version
        if self.feature_schema_version:
            params["feature_schema_version"] = self.feature_schema_version
        params["training_examples"] = str(self.training_examples)
        params["validation_examples"] = str(self.validation_examples)
        params["test_examples"] = str(self.test_examples)
        params["total_examples"] = str(self.total_examples)
        params["feature_count"] = str(self.feature_count)
        if self.label_classes:
            params["label_classes"] = ",".join(self.label_classes)
        if self.target_label_version:
            params["target_label_version"] = self.target_label_version

        # Hyperparameters
        for k, v in self.hyperparameters.items():
            params[f"hp.{k}"] = str(v)

        # Algorithm-specific params
        if self.n_estimators is not None:
            params["n_estimators"] = str(self.n_estimators)
        if self.max_depth is not None:
            params["max_depth"] = str(self.max_depth)
        if self.learning_rate is not None:
            params["learning_rate"] = str(self.learning_rate)
        if self.subsample is not None:
            params["subsample"] = str(self.subsample)
        if self.colsample_bytree is not None:
            params["colsample_bytree"] = str(self.colsample_bytree)
        if self.class_weight is not None:
            params["class_weight"] = self.class_weight
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = str(self.early_stopping_rounds)

        # Random seed
        if self.random_seed is not None:
            params["random_seed"] = str(self.random_seed)

        # Split config
        if self.train_ratio is not None:
            params["train_ratio"] = str(self.train_ratio)
        if self.val_ratio is not None:
            params["val_ratio"] = str(self.val_ratio)
        if self.test_ratio is not None:
            params["test_ratio"] = str(self.test_ratio)
        if self.max_features is not None:
            params["max_features"] = str(self.max_features)

        # Training config (non-hyperparameter settings)
        for k, v in self.training_config.items():
            if k not in self.hyperparameters:
                params[f"config.{k}"] = str(v)

        # Environment
        params["environment"] = self.environment

        # Environment metadata
        if self.environment_metadata:
            params.update(self.environment_metadata.to_params_dict())

        return params


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Snapshot
# ─────────────────────────────────────────────────────────────────────────────


class MetricsSnapshot(BaseModel):
    """Complete metrics snapshot for an MLflow training run.

    Captures ALL metric categories using the same definitions as Phase 9.
    Metric names and formulas are consistent with:
    - EvaluationMetrics (model_training.py)
    - LearningMetrics (learning_metrics.py)
    
    Safety principle:
      Metrics are OBSERVATIONAL ONLY.
      They never authorize execution or bypass Phase 6 guardrails.
    """
    run_id: str = Field(..., description="MLflow run ID")

    # ── Classification Metrics (consistent with EvaluationMetrics) ──────
    accuracy: Optional[float] = Field(None, description="Overall accuracy")
    precision_macro: Optional[float] = Field(None, description="Macro-averaged precision")
    recall_macro: Optional[float] = Field(None, description="Macro-averaged recall")
    f1_macro: Optional[float] = Field(None, description="Macro-averaged F1")
    precision_weighted: Optional[float] = Field(None, description="Weighted precision")
    recall_weighted: Optional[float] = Field(None, description="Weighted recall")
    f1_weighted: Optional[float] = Field(None, description="Weighted F1")
    total_samples: int = Field(default=0, description="Total samples evaluated")

    # Per-class metrics
    per_class_precision: Dict[str, float] = Field(
        default_factory=dict, description="Per-class precision"
    )
    per_class_recall: Dict[str, float] = Field(
        default_factory=dict, description="Per-class recall"
    )
    per_class_f1: Dict[str, float] = Field(
        default_factory=dict, description="Per-class F1"
    )
    per_class_support: Dict[str, int] = Field(
        default_factory=dict, description="Per-class sample count"
    )
    confusion_matrix: List[List[int]] = Field(
        default_factory=list, description="Confusion matrix"
    )
    confusion_labels: List[str] = Field(
        default_factory=list, description="Confusion matrix labels"
    )

    # ── Safety-Critical Metrics (consistent with EvaluationMetrics + LearningMetrics) ──
    false_automation: Optional[int] = Field(None, description="False automation count")
    false_automation_rate: Optional[float] = Field(None, description="False automation rate")
    high_value_errors: Optional[int] = Field(None, description="High-value error count")
    high_value_error_rate: Optional[float] = Field(None, description="High-value error rate")
    unknown_case_errors: Optional[int] = Field(None, description="Unknown case errors")
    novel_pattern_errors: Optional[int] = Field(None, description="Novel pattern errors")
    verification_failure_rate: Optional[float] = Field(
        None, description="Verification failure rate"
    )
    unsafe_decision_rate: Optional[float] = Field(None, description="Unsafe decision rate")
    unresolved_rate: Optional[float] = Field(None, description="Unresolved rate")
    incorrect_auto_resolution: int = Field(
        default=0, description="Incorrect AUTO decisions"
    )

    # ── Resolution Metrics ─────────────────────────────────────────────
    resolution_accuracy: Optional[float] = Field(None, description="Resolution accuracy")
    candidate_selection_accuracy: Optional[float] = Field(
        None, description="Candidate selection accuracy"
    )
    correct_resolution_rate: Optional[float] = Field(
        None, description="Correct resolution rate"
    )

    # ── Automation Metrics (consistent with AutomationMetrics) ──────────
    auto_decisions: int = Field(default=0, description="AUTO decisions")
    human_decisions: int = Field(default=0, description="HUMAN_REVIEW decisions")
    unresolved_decisions: int = Field(default=0, description="UNRESOLVED decisions")
    automation_rate: Optional[float] = Field(None, description="Automation rate")
    human_review_rate: Optional[float] = Field(None, description="Human review rate")
    successful_auto: int = Field(default=0, description="Successful AUTO decisions")
    successful_automation_rate: Optional[float] = Field(
        None, description="Successful automation rate"
    )
    failed_auto: int = Field(default=0, description="Failed AUTO decisions")
    failed_automation_rate: Optional[float] = Field(
        None, description="Failed automation rate"
    )

    # ── Human Review Metrics (consistent with HumanReviewMetrics) ───────
    total_human_reviews: int = Field(default=0, description="Total human reviews")
    human_corrections: int = Field(default=0, description="Human corrections")
    human_rejections: int = Field(default=0, description="Human rejections")
    human_approvals: int = Field(default=0, description="Human approvals")
    correction_rate: Optional[float] = Field(None, description="Correction rate")
    unnecessary_escalations: int = Field(default=0, description="Unnecessary escalations")

    # ── Precision Metrics (consistent with PrecisionMetrics) ────────────
    correct_auto: int = Field(default=0, description="Correct AUTO decisions")
    incorrect_auto: int = Field(default=0, description="Incorrect AUTO decisions")
    precision: Optional[float] = Field(None, description="Precision (correct/total auto)")

    # ── Verification Metrics (consistent with VerificationMetrics) ──────
    total_executed: int = Field(default=0, description="Total executed resolutions")
    total_verified: int = Field(default=0, description="Total verified resolutions")
    total_rolled_back: int = Field(default=0, description="Total rolled-back resolutions")
    verification_success_rate: Optional[float] = Field(
        None, description="Verification success rate"
    )
    rollback_rate: Optional[float] = Field(None, description="Rollback rate")

    # ── Financial Metrics (consistent with FinancialImpactMetrics) ──────
    total_adjustment_paise: int = Field(
        default=0, description="Total financial adjustment in paise"
    )
    avg_adjustment_paise: Optional[float] = Field(
        None, description="Average adjustment per AUTO decision"
    )
    max_adjustment_paise: int = Field(
        default=0, description="Maximum single adjustment"
    )
    total_error_impact_paise: int = Field(
        default=0, description="Financial impact of incorrect AUTO"
    )
    high_value_error_impact_paise: int = Field(
        default=0, description="Total impact of high-value errors"
    )
    impact_avoided_paise: int = Field(
        default=0, description="Financial impact avoided through escalation"
    )
    discrepancy_eliminated_count: int = Field(
        default=0, description="Cases where discrepancy was eliminated"
    )
    discrepancy_elimination_rate: Optional[float] = Field(
        None, description="Discrepancy elimination rate"
    )

    # ── Reward Metrics (consistent with RewardMetrics) ──────────────────
    avg_reward: Optional[float] = Field(None, description="Average reward")
    median_reward: Optional[float] = Field(None, description="Median reward")
    reward_std: Optional[float] = Field(None, description="Reward std dev")
    positive_rewards: int = Field(default=0, description="Positive reward count")
    negative_rewards: int = Field(default=0, description="Negative reward count")
    positive_reward_rate: Optional[float] = Field(
        None, description="Positive reward rate"
    )

    # ── Custom Metrics ─────────────────────────────────────────────────
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Additional custom metrics"
    )

    logged_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When metrics were logged",
    )

    def to_mlflow_dict(self) -> Dict[str, float]:
        """Convert to flat dict for MLflow logging (only non-None numeric values).

        Includes all metric categories for comprehensive experiment comparison.
        """
        d: Dict[str, float] = {}
        for k, v in self.model_dump().items():
            if k in ("run_id", "logged_at", "custom_metrics"):
                continue
            if v is not None and isinstance(v, (int, float)):
                d[k] = float(v)
        d.update(self.custom_metrics)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Types
# ─────────────────────────────────────────────────────────────────────────────


class ArtifactType(str, Enum):
    """Types of artifacts tracked per training run."""
    MODEL = "model"
    EVALUATION_REPORT = "evaluation_report"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    SAFETY_REPORT = "safety_report"
    AUTOMATION_REPORT = "automation_report"
    RESOLUTION_REPORT = "resolution_report"
    DATASET_METADATA = "dataset_metadata"
    FEATURE_SCHEMA = "feature_schema"
    LABEL_SCHEMA = "label_schema"
    TRAINING_CONFIG = "training_config"
    TRAINING_SUMMARY = "training_summary"
    EVALUATION_SUMMARY = "evaluation_summary"
    DATASET_STATISTICS = "dataset_statistics"
    CUSTOM = "custom"


class ArtifactMetadata(BaseModel):
    """Metadata for a single artifact logged to an MLflow run.

    Every artifact is traceable to:
    run_id → model_version → dataset_version → feature_version
    """
    artifact_id: str = Field(..., description="Unique artifact identifier")
    run_id: str = Field(..., description="MLflow run ID this artifact belongs to")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    artifact_name: str = Field(..., description="Human-readable artifact name")
    artifact_path: Optional[str] = Field(None, description="Path in MLflow artifact store")
    description: str = Field(default="", description="What this artifact contains")

    # Lineage
    model_version: Optional[str] = Field(None, description="Model version this artifact belongs to")
    dataset_version: Optional[str] = Field(None, description="Dataset version used")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")

    # Content metadata
    content_type: str = Field(default="application/octet-stream", description="MIME type")
    size_bytes: Optional[int] = Field(None, description="Artifact size in bytes")
    checksum: Optional[str] = Field(None, description="Content checksum for integrity")

    # Timestamps
    logged_at: datetime = Field(
        default_factory=datetime.utcnow, description="When artifact was logged"
    )

    def summary(self) -> str:
        return (
            f"Artifact: {self.artifact_name} | "
            f"Type: {self.artifact_type.value} | "
            f"Run: {self.run_id[:8]}... | "
            f"Model: v{self.model_version or '?'}"
        )


class ArtifactLineage(BaseModel):
    """Complete lineage chain for artifact traceability."""
    run_id: str = Field(..., description="MLflow run ID")
    model_id: Optional[str] = Field(None, description="Model ID")
    model_version: Optional[str] = Field(None, description="Model version")
    dataset_id: Optional[str] = Field(None, description="Dataset ID")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    feature_schema_version: Optional[str] = Field(None, description="Feature schema version")
    label_schema_version: Optional[str] = Field(None, description="Label schema version")
    artifacts: List[ArtifactMetadata] = Field(
        default_factory=list, description="All artifacts for this lineage"
    )

    def summary(self) -> str:
        return (
            f"Lineage: run={self.run_id[:8]}... | "
            f"model=v{self.model_version or '?'} | "
            f"dataset=v{self.dataset_version or '?'} | "
            f"artifacts={len(self.artifacts)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Summary
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentSummary(BaseModel):
    """Summary of an MLflow experiment."""
    experiment_name: str = Field(..., description="Experiment name")
    experiment_id: Optional[str] = Field(None, description="MLflow experiment ID")
    experiment_type: ExperimentType = Field(
        ..., description="Type of experiment"
    )
    run_count: int = Field(default=0, description="Total runs in experiment")
    active_run_count: int = Field(default=0, description="Currently running")
    completed_run_count: int = Field(default=0, description="Completed runs")
    failed_run_count: int = Field(default=0, description="Failed runs")
    best_run_id: Optional[str] = Field(None, description="Run with best metrics")
    best_metric: Optional[str] = Field(None, description="Metric used for best")
    best_metric_value: Optional[float] = Field(None, description="Best metric value")

    def summary(self) -> str:
        return (
            f"Experiment: {self.experiment_name} | "
            f"Runs: {self.run_count} | "
            f"Completed: {self.completed_run_count} | "
            f"Best: {self.best_run_id[:8] if self.best_run_id else 'none'}"
        )
