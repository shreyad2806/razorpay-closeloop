"""
ML dataset contract for Razorpay CloseLoop Phase 4.

Defines the training/evaluation dataset structure connecting:
- case → deterministic reconciliation result → evidence-derived features → ground-truth label

Ground truth labels may be used ONLY by training/evaluation.
Production inference must not receive the true label.

Feature schema version tracks which features were used for training.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.enums import ExceptionType


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema Version
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Categories
# ─────────────────────────────────────────────────────────────────────────────


class FeatureCategory(str, Enum):
    """Categories of ML features."""

    FINANCIAL = "financial"
    STRUCTURAL = "structural"
    EVIDENCE = "evidence"
    TEMPORAL = "temporal"
    MERCHANT = "merchant"
    HISTORICAL = "historical"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema Definition
# ─────────────────────────────────────────────────────────────────────────────


class FeatureDefinition(BaseModel):
    """Definition of a single ML feature."""

    name: str = Field(..., description="Unique feature name")
    category: FeatureCategory = Field(..., description="Feature category")
    dtype: str = Field(..., description="Data type: int, float, bool")
    description: str = Field(..., description="Human-readable description")
    min_value: Optional[float] = Field(None, description="Minimum expected value")
    max_value: Optional[float] = Field(None, description="Maximum expected value")
    default_value: Optional[float] = Field(None, description="Default when missing")


class FeatureSchema(BaseModel):
    """
    Versioned feature schema defining all ML features.

    This schema MUST be checked when training or loading a model.
    If the schema version changes, the model must be retrained.
    """

    version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Schema version — must match model training version",
    )
    features: List[FeatureDefinition] = Field(
        ..., description="Ordered list of feature definitions"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this schema was created",
    )

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return [f.name for f in self.features]

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.features)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Vector
# ─────────────────────────────────────────────────────────────────────────────


class FeatureVector(BaseModel):
    """
    Numeric feature vector for a single ML sample.

    All values are numeric (int or float).
    Boolean indicators are stored as 0.0/1.0.
    """

    features: Dict[str, float] = Field(
        ..., description="Feature name → numeric value mapping"
    )
    schema_version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Schema version used to generate these features",
    )

    def to_list(self, feature_names: List[str]) -> List[float]:
        """Convert to ordered numeric list matching feature schema order."""
        return [self.features.get(name, 0.0) for name in feature_names]


# ─────────────────────────────────────────────────────────────────────────────
# ML Labels
# ─────────────────────────────────────────────────────────────────────────────


class MLLabels(BaseModel):
    """
    Ground-truth labels for a single ML sample.

    These are used ONLY for training and evaluation.
    Production inference must NOT receive these fields.
    """

    true_exception_type: ExceptionType = Field(
        ..., description="The actual exception category (ground truth)"
    )
    true_resolution: Optional[str] = Field(
        None, description="The correct resolution (ground truth)"
    )
    resolvable: bool = Field(
        ..., description="Whether this case has a deterministic resolution"
    )
    risk_category: Optional[str] = Field(
        None, description="Risk level assigned to this case"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ML Sample
# ─────────────────────────────────────────────────────────────────────────────


class MLSample(BaseModel):
    """
    Complete ML sample connecting case, features, and labels.

    Structure:
        case_id → deterministic result → evidence-derived features → ground-truth label

    The feature vector must NOT contain any label fields.
    """

    # Case identification
    case_id: str = Field(..., description="Unique case identifier")
    payment_id: str = Field(..., description="Reference to Payment")
    merchant_id: Optional[str] = Field(None, description="Reference to Merchant")
    batch_id: Optional[str] = Field(None, description="Source batch identifier")

    # Financial context (non-leaking)
    expected_amount: int = Field(..., description="Expected settlement in paise")
    actual_amount: int = Field(..., description="Actual settlement in paise")
    difference: int = Field(..., description="expected - actual in paise")
    payment_amount: int = Field(..., description="Original payment amount in paise")

    # Feature vector (deterministic, evidence-derived)
    features: FeatureVector = Field(
        ..., description="Numeric feature vector for ML model"
    )

    # Labels (ONLY for training/evaluation — never for production inference)
    labels: MLLabels = Field(
        ..., description="Ground-truth labels for training/evaluation"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this sample was created",
    )

    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.features.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Split
# ─────────────────────────────────────────────────────────────────────────────


class SplitType(str, Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetSplit(BaseModel):
    """
    Metadata for a dataset split.

    Defines the reproducible split strategy:
    - Batch-level separation where appropriate
    - Deterministic random seeds
    - No case overlap across incompatible splits
    """

    split_type: SplitType = Field(..., description="Train, validation, or test")
    batch_ids: List[str] = Field(
        ..., description="Batch IDs included in this split"
    )
    case_count: int = Field(..., description="Number of cases in this split")
    seed: int = Field(..., description="Random seed used for this split")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this split was created",
    )

    # Label distribution
    label_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each exception type in this split",
    )


class DatasetManifest(BaseModel):
    """
    Complete dataset manifest describing the full dataset and its splits.

    Ensures reproducibility and documents the split strategy.
    """

    version: str = Field(default=FEATURE_SCHEMA_VERSION, description="Dataset version")
    feature_schema_version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Feature schema version used",
    )
    total_samples: int = Field(..., description="Total number of ML samples")
    splits: List[DatasetSplit] = Field(
        ..., description="Train/validation/test splits"
    )
    label_classes: List[str] = Field(
        ..., description="All possible label values (ExceptionType values)"
    )
    feature_count: int = Field(..., description="Number of features per sample")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this manifest was created",
    )

    def get_split(self, split_type: SplitType) -> Optional[DatasetSplit]:
        """Get a specific split by type."""
        for split in self.splits:
            if split.split_type == split_type:
                return split
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Protection Constants
# ─────────────────────────────────────────────────────────────────────────────

# Fields that MUST NEVER appear in the feature vector.
# These are ground-truth labels or evaluation metadata.
LEAKED_FIELDS = frozenset([
    "true_exception_type",
    "true_resolution",
    "resolvable",
    "risk_category",
])
