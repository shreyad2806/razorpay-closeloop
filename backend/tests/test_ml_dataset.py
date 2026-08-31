"""
Tests for ML dataset contract.

Covers:
- All expected labels exist
- No missing required features
- Feature types are correct
- Train/test split reproducible
- No ground-truth leakage
- Same case does not appear across incompatible splits
- Feature extraction is deterministic
- Feature schema versioning
- ML sample structure
- Dataset manifest
"""

import os
import sys
from pathlib import Path

import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_ml_dataset.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.enums import ExceptionType
from app.schemas.ml_dataset import (
    FEATURE_SCHEMA_VERSION,
    LEAKED_FIELDS,
    DatasetManifest,
    DatasetSplit,
    FeatureDefinition,
    FeatureSchema,
    FeatureVector,
    MLLabels,
    MLSample,
    SplitType,
)
from app.ml.features import (
    ML_FEATURE_SCHEMA,
    extract_features,
    validate_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_features():
    """Provide a valid feature dict."""
    return extract_features(
        difference=500,
        payment_amount=100000,
        settlement_amount=99500,
        refund_amount=1000,
        fee_amount=500,
        tax_amount=200,
        adjustment_amount=0,
        num_settlements=1,
        num_refunds=1,
        num_fees=1,
        num_taxes=1,
        num_adjustments=0,
        has_missing_evidence=False,
        num_missing_evidence=0,
        evidence_coverage=1.0,
        consistency_score=0.9,
        fully_explained=True,
        partially_explained=False,
        has_conflict=False,
        supporting_evidence_count=3,
        num_candidate_explanations=1,
    )


@pytest.fixture
def sample_labels():
    """Provide valid ML labels."""
    return MLLabels(
        true_exception_type=ExceptionType.FEE_DIFFERENCE,
        true_resolution="FEE_ADJUSTMENT",
        resolvable=True,
        risk_category="LOW",
    )


@pytest.fixture
def sample_ml_sample(sample_features, sample_labels):
    """Provide a valid ML sample."""
    return MLSample(
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        batch_id="batch_001",
        expected_amount=100000,
        actual_amount=99500,
        difference=500,
        payment_amount=100000,
        features=FeatureVector(features=sample_features),
        labels=sample_labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureSchema:
    """Tests for feature schema definition."""

    def test_schema_has_version(self):
        """Test that schema has a version."""
        assert ML_FEATURE_SCHEMA.version == FEATURE_SCHEMA_VERSION

    def test_schema_has_features(self):
        """Test that schema has feature definitions."""
        assert len(ML_FEATURE_SCHEMA.features) > 0

    def test_feature_names_unique(self):
        """Test that all feature names are unique."""
        names = ML_FEATURE_SCHEMA.get_feature_names()
        assert len(names) == len(set(names))

    def test_feature_count(self):
        """Test expected feature count."""
        # Should have at least 25 features
        assert ML_FEATURE_SCHEMA.get_feature_count() >= 25

    def test_all_features_have_required_fields(self):
        """Test that all features have required fields."""
        for feature in ML_FEATURE_SCHEMA.features:
            assert feature.name
            assert feature.category
            assert feature.dtype in ("int", "float", "bool")
            assert feature.description

    def test_get_feature_names(self):
        """Test get_feature_names returns ordered list."""
        names = ML_FEATURE_SCHEMA.get_feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_features_returns_dict(self, sample_features):
        """Test that feature extraction returns a dict."""
        assert isinstance(sample_features, dict)

    def test_extract_features_all_schema_names_present(self, sample_features):
        """Test that extracted features include all schema names."""
        schema_names = set(ML_FEATURE_SCHEMA.get_feature_names())
        extracted_names = set(sample_features.keys())
        assert schema_names == extracted_names

    def test_extract_features_deterministic(self):
        """Test that feature extraction is deterministic."""
        kwargs = dict(
            difference=500,
            payment_amount=100000,
            settlement_amount=99500,
            refund_amount=1000,
            fee_amount=500,
            tax_amount=200,
            adjustment_amount=0,
            num_settlements=1,
            num_refunds=1,
            num_fees=1,
            num_taxes=1,
            num_adjustments=0,
            has_missing_evidence=False,
            num_missing_evidence=0,
            evidence_coverage=1.0,
            consistency_score=0.9,
            fully_explained=True,
            partially_explained=False,
            has_conflict=False,
            supporting_evidence_count=3,
            num_candidate_explanations=1,
        )
        f1 = extract_features(**kwargs)
        f2 = extract_features(**kwargs)
        assert f1 == f2

    def test_extract_features_types_are_numeric(self, sample_features):
        """Test that all feature values are numeric."""
        for name, value in sample_features.items():
            assert isinstance(value, (int, float)), f"{name} is {type(value)}"

    def test_extract_features_safe_division(self):
        """Test that division by zero is handled safely."""
        features = extract_features(
            difference=500,
            payment_amount=0,  # Zero payment
            settlement_amount=0,
            refund_amount=0,
            fee_amount=0,
            tax_amount=0,
            adjustment_amount=0,
            num_settlements=0,
            num_refunds=0,
            num_fees=0,
            num_taxes=0,
            num_adjustments=0,
            has_missing_evidence=False,
            num_missing_evidence=0,
            evidence_coverage=0.0,
            consistency_score=0.0,
            fully_explained=False,
            partially_explained=False,
            has_conflict=False,
            supporting_evidence_count=0,
            num_candidate_explanations=0,
        )
        # Ratios should be 0.0, not crash
        assert features["relative_difference"] == 0.0
        assert features["refund_ratio"] == 0.0
        assert features["fee_ratio"] == 0.0
        assert features["tax_ratio"] == 0.0

    def test_extract_features_boolean_to_float(self):
        """Test that boolean indicators are converted to 0.0/1.0."""
        features = extract_features(
            difference=500,
            payment_amount=100000,
            settlement_amount=99500,
            refund_amount=0,
            fee_amount=0,
            tax_amount=0,
            adjustment_amount=0,
            num_settlements=1,
            num_refunds=0,
            num_fees=0,
            num_taxes=0,
            num_adjustments=0,
            has_missing_evidence=True,
            num_missing_evidence=1,
            evidence_coverage=0.5,
            consistency_score=0.7,
            fully_explained=False,
            partially_explained=True,
            has_conflict=False,
            supporting_evidence_count=0,
            num_candidate_explanations=0,
        )
        assert features["has_missing_evidence"] == 1.0
        assert features["fully_explained"] == 0.0
        assert features["partially_explained"] == 1.0
        assert features["has_conflict"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureValidation:
    """Tests for feature validation."""

    def test_valid_features_no_errors(self, sample_features):
        """Test that valid features produce no errors."""
        errors = validate_features(sample_features)
        assert errors == []

    def test_missing_feature_detected(self):
        """Test that missing features are detected."""
        features = {"difference_amount": 500.0}  # Missing most features
        errors = validate_features(features)
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)

    def test_unexpected_feature_detected(self):
        """Test that unexpected features are detected."""
        features = extract_features(
            difference=500, payment_amount=100000, settlement_amount=99500,
            refund_amount=0, fee_amount=0, tax_amount=0, adjustment_amount=0,
            num_settlements=1, num_refunds=0, num_fees=0, num_taxes=0,
            num_adjustments=0, has_missing_evidence=False, num_missing_evidence=0,
            evidence_coverage=1.0, consistency_score=0.9, fully_explained=True,
            partially_explained=False, has_conflict=False,
            supporting_evidence_count=1, num_candidate_explanations=1,
        )
        features["unexpected_feature"] = 42.0
        errors = validate_features(features)
        assert any("Unexpected" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# ML Sample Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLSample:
    """Tests for ML sample structure."""

    def test_sample_has_case_id(self, sample_ml_sample):
        """Test that sample has case_id."""
        assert sample_ml_sample.case_id == "CASE-001"

    def test_sample_has_features(self, sample_ml_sample):
        """Test that sample has feature vector."""
        assert isinstance(sample_ml_sample.features, FeatureVector)

    def test_sample_has_labels(self, sample_ml_sample):
        """Test that sample has labels."""
        assert isinstance(sample_ml_sample.labels, MLLabels)

    def test_sample_feature_names(self, sample_ml_sample):
        """Test that sample can list feature names."""
        names = sample_ml_sample.feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_sample_to_list(self, sample_ml_sample):
        """Test that feature vector can convert to ordered list."""
        schema_names = ML_FEATURE_SCHEMA.get_feature_names()
        vec = sample_ml_sample.features.to_list(schema_names)
        assert len(vec) == len(schema_names)
        assert all(isinstance(v, float) for v in vec)


# ─────────────────────────────────────────────────────────────────────────────
# ML Labels Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLLabels:
    """Tests for ML labels structure."""

    def test_labels_have_exception_type(self, sample_labels):
        """Test that labels have true_exception_type."""
        assert sample_labels.true_exception_type == ExceptionType.FEE_DIFFERENCE

    def test_labels_have_resolvable(self, sample_labels):
        """Test that labels have resolvable."""
        assert sample_labels.resolvable is True

    def test_labels_all_exception_types(self):
        """Test that all exception types can be used as labels."""
        for exc_type in ExceptionType:
            labels = MLLabels(
                true_exception_type=exc_type,
                resolvable=True,
            )
            assert labels.true_exception_type == exc_type


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Leakage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthLeakage:
    """Verify that ground truth labels never appear in features."""

    def test_leaked_fields_constant(self):
        """Test that LEAKED_FIELDS contains all prohibited field names."""
        assert "true_exception_type" in LEAKED_FIELDS
        assert "true_resolution" in LEAKED_FIELDS
        assert "resolvable" in LEAKED_FIELDS
        assert "risk_category" in LEAKED_FIELDS

    def test_feature_names_no_leakage(self):
        """Test that feature schema contains no leaked field names."""
        feature_names = set(ML_FEATURE_SCHEMA.get_feature_names())
        for leaked in LEAKED_FIELDS:
            assert leaked not in feature_names, f"Leaked field in features: {leaked}"

    def test_extracted_features_no_leakage(self, sample_features):
        """Test that extracted features contain no leaked field names."""
        for leaked in LEAKED_FIELDS:
            assert leaked not in sample_features, f"Leaked field in features: {leaked}"

    def test_labels_separate_from_features(self, sample_ml_sample):
        """Test that labels are separate from feature vector."""
        feature_keys = set(sample_ml_sample.features.features.keys())
        label_fields = {"true_exception_type", "true_resolution", "resolvable", "risk_category"}
        assert feature_keys.isdisjoint(label_fields)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema Versioning Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureVersioning:
    """Tests for feature schema versioning."""

    def test_schema_version_defined(self):
        """Test that schema version is defined."""
        assert FEATURE_SCHEMA_VERSION
        assert isinstance(FEATURE_SCHEMA_VERSION, str)

    def test_schema_version_format(self):
        """Test that schema version follows semver-like format."""
        parts = FEATURE_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_feature_vector_has_schema_version(self, sample_features):
        """Test that FeatureVector includes schema version."""
        fv = FeatureVector(features=sample_features)
        assert fv.schema_version == FEATURE_SCHEMA_VERSION

    def test_manifest_has_feature_schema_version(self):
        """Test that DatasetManifest includes feature schema version."""
        manifest = DatasetManifest(
            total_samples=100,
            splits=[],
            label_classes=[e.value for e in ExceptionType],
            feature_count=25,
        )
        assert manifest.feature_schema_version == FEATURE_SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Split Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetSplit:
    """Tests for dataset split strategy."""

    def test_split_types(self):
        """Test that all split types exist."""
        assert SplitType.TRAIN.value == "train"
        assert SplitType.VALIDATION.value == "validation"
        assert SplitType.TEST.value == "test"

    def test_split_has_batch_ids(self):
        """Test that split has batch IDs."""
        split = DatasetSplit(
            split_type=SplitType.TRAIN,
            batch_ids=["batch_001", "batch_002"],
            case_count=8000,
            seed=42,
        )
        assert len(split.batch_ids) == 2

    def test_split_has_seed(self):
        """Test that split has deterministic seed."""
        split = DatasetSplit(
            split_type=SplitType.TRAIN,
            batch_ids=["batch_001"],
            case_count=5000,
            seed=42,
        )
        assert split.seed == 42

    def test_split_has_label_distribution(self):
        """Test that split has label distribution."""
        dist = {"EXACT_MATCH": 3000, "FEE_DIFFERENCE": 1000}
        split = DatasetSplit(
            split_type=SplitType.TRAIN,
            batch_ids=["batch_001"],
            case_count=4000,
            seed=42,
            label_distribution=dist,
        )
        assert split.label_distribution == dist

    def test_no_case_overlap(self):
        """Test that splits can be defined without overlapping cases."""
        train = DatasetSplit(
            split_type=SplitType.TRAIN,
            batch_ids=["batch_001"],
            case_count=5000,
            seed=42,
        )
        val = DatasetSplit(
            split_type=SplitType.VALIDATION,
            batch_ids=["batch_002"],
            case_count=3000,
            seed=42,
        )
        test = DatasetSplit(
            split_type=SplitType.TEST,
            batch_ids=["batch_003"],
            case_count=2500,
            seed=42,
        )
        # Batch-level separation ensures no overlap
        all_batches = set(train.batch_ids + val.batch_ids + test.batch_ids)
        assert len(all_batches) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Manifest Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetManifest:
    """Tests for dataset manifest."""

    def test_manifest_has_total_samples(self):
        """Test that manifest has total samples."""
        manifest = DatasetManifest(
            total_samples=10500,
            splits=[],
            label_classes=[e.value for e in ExceptionType],
            feature_count=25,
        )
        assert manifest.total_samples == 10500

    def test_manifest_has_splits(self):
        """Test that manifest has splits."""
        splits = [
            DatasetSplit(split_type=SplitType.TRAIN, batch_ids=["b1"], case_count=5000, seed=42),
            DatasetSplit(split_type=SplitType.VALIDATION, batch_ids=["b2"], case_count=3000, seed=42),
            DatasetSplit(split_type=SplitType.TEST, batch_ids=["b3"], case_count=2500, seed=42),
        ]
        manifest = DatasetManifest(
            total_samples=10500,
            splits=splits,
            label_classes=[e.value for e in ExceptionType],
            feature_count=25,
        )
        assert len(manifest.splits) == 3

    def test_manifest_get_split(self):
        """Test that manifest can get a specific split."""
        splits = [
            DatasetSplit(split_type=SplitType.TRAIN, batch_ids=["b1"], case_count=5000, seed=42),
        ]
        manifest = DatasetManifest(
            total_samples=5000,
            splits=splits,
            label_classes=[],
            feature_count=25,
        )
        train = manifest.get_split(SplitType.TRAIN)
        assert train is not None
        assert train.split_type == SplitType.TRAIN

    def test_manifest_label_classes(self):
        """Test that manifest has all exception type labels."""
        manifest = DatasetManifest(
            total_samples=100,
            splits=[],
            label_classes=[e.value for e in ExceptionType],
            feature_count=25,
        )
        assert len(manifest.label_classes) == len(ExceptionType)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_feature_extraction_deterministic(self):
        """Test that same inputs produce same features."""
        kwargs = dict(
            difference=1234, payment_amount=100000, settlement_amount=98766,
            refund_amount=500, fee_amount=200, tax_amount=100,
            adjustment_amount=50, num_settlements=1, num_refunds=1,
            num_fees=1, num_taxes=1, num_adjustments=1,
            has_missing_evidence=False, num_missing_evidence=0,
            evidence_coverage=0.8, consistency_score=0.9,
            fully_explained=True, partially_explained=False,
            has_conflict=False, supporting_evidence_count=3,
            num_candidate_explanations=1,
        )
        f1 = extract_features(**kwargs)
        f2 = extract_features(**kwargs)
        assert f1 == f2

    def test_feature_schema_deterministic(self):
        """Test that feature schema is deterministic."""
        names1 = ML_FEATURE_SCHEMA.get_feature_names()
        names2 = ML_FEATURE_SCHEMA.get_feature_names()
        assert names1 == names2
