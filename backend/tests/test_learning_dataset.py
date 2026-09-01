"""
Tests for Phase 9C — Learning Dataset Construction.

Tests cover:
- Feature snapshot correctness
- Label correctness
- Feature/label separation
- Leakage detection
- Duplicate detection
- Missing outcome
- Contradictory labels
- Temporal split
- Data lineage
- Quality checks
"""

import pytest
from datetime import datetime, timedelta

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    PredictionRecord,
)
from app.schemas.learning_dataset import (
    DataSplit,
    FeatureSnapshot,
    LearningDataset,
    LearningExample,
    LearningLabels,
    QualityIssue,
    QualityReport,
    SplitType,
)
from app.schemas.reward_engine import RewardCategory, RewardRecord
from app.services.learning_dataset import (
    FeatureSnapshotBuilder,
    LabelBuilder,
    LearningDatasetBuilder,
    LearningExampleBuilder,
    LeakageDetector,
    QualityChecker,
    SplitStrategy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_prediction(
    resolution_type: str = "FEE_ADJUSTMENT",
    confidence: float = 0.85,
    exception_type: str = "FEE_DIFFERENCE",
) -> PredictionRecord:
    return PredictionRecord(
        exception_type=exception_type,
        resolution_type=resolution_type,
        resolution_confidence=confidence,
        exception_confidence=0.9,
        model_version="xgb-v1.0",
    )


def _make_actual(
    resolution: str = "FEE_ADJUSTMENT",
    correct: bool = True,
    executed: bool = True,
    verified: bool = True,
    rolled_back: bool = False,
    impact: int = 3000,
) -> ActualOutcomeRecord:
    return ActualOutcomeRecord(
        actual_resolution=resolution,
        actual_exception_type="FEE_DIFFERENCE",
        resolution_correct=correct,
        financial_impact_paise=impact,
        was_executed=executed,
        was_verified=verified,
        was_rolled_back=rolled_back,
    )


def _make_lineage(exception_id: str = "EXC-001") -> DataLineage:
    return DataLineage(
        exception_id=exception_id,
        evidence_ids=["EVD-001"],
        execution_id="EXEC-001",
        verification_id="VER-001",
    )


def _make_outcome(
    workflow_id: str = "WF-001",
    exception_id: str = "EXC-001",
    case_id: str = "CASE-001",
    resolution: str = "FEE_ADJUSTMENT",
    correct: bool = True,
    executed: bool = True,
    verified: bool = True,
    rolled_back: bool = False,
    adjustment: int = 3000,
    decision: str = "AUTO",
    confidence: float = 0.85,
    discrepancy_eliminated: bool = True,
    unintended: int = 0,
    feedback_id: str = None,
    feedback_type: FeedbackType = None,
    human_override: bool = False,
    ground_truth_type: str = "FEE_DIFFERENCE",
    ground_truth_resolution: str = "FEE_ADJUSTMENT",
    created_at: datetime = None,
) -> OutcomeRecord:
    impact = FinancialImpact(
        requested_adjustment_paise=adjustment,
        actual_adjustment_paise=adjustment,
        difference_before_paise=adjustment,
        difference_after_paise=0 if discrepancy_eliminated else adjustment,
        discrepancy_eliminated=discrepancy_eliminated,
        unintended_changes=unintended,
    )
    return OutcomeRecord(
        outcome_id=f"OUT-{workflow_id}",
        workflow_id=workflow_id,
        exception_id=exception_id,
        case_id=case_id,
        prediction=_make_prediction(resolution_type=resolution, confidence=confidence),
        actual_outcome=_make_actual(
            resolution=resolution, correct=correct, executed=executed,
            verified=verified, rolled_back=rolled_back, impact=adjustment,
        ),
        financial_impact=impact,
        lineage=_make_lineage(exception_id),
        decision=decision,
        confidence=confidence,
        verification_passed=verified,
        human_feedback_id=feedback_id,
        human_feedback_type=feedback_type,
        human_override=human_override,
        ground_truth_exception_type=ground_truth_type,
        ground_truth_resolution=ground_truth_resolution,
        created_at=created_at or datetime.utcnow(),
    )


def _make_feedback(
    workflow_id: str = "WF-001",
    exception_id: str = "EXC-001",
    feedback_type: FeedbackType = FeedbackType.APPROVE,
) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=f"FB-{workflow_id}",
        workflow_id=workflow_id,
        exception_id=exception_id,
        feedback_type=feedback_type,
        reviewer="test_reviewer",
        system_prediction="FEE_ADJUSTMENT",
    )


def _make_reward(
    workflow_id: str = "WF-001",
    reward_value: float = 0.8,
    category: RewardCategory = RewardCategory.CORRECT_AUTO_RESOLUTION,
) -> RewardRecord:
    from app.schemas.reward_engine import RewardBreakdown, RewardComponent
    breakdown = RewardBreakdown(
        base_reward=RewardComponent(
            component_name="base_reward", value=0.8, reason="test"
        ),
        verification_component=RewardComponent(
            component_name="verification", value=0.1, reason="test"
        ),
        financial_risk_component=RewardComponent(
            component_name="financial_risk", value=0.0, reason="test"
        ),
        human_feedback_component=RewardComponent(
            component_name="human_feedback", value=0.0, reason="test"
        ),
        confidence_component=RewardComponent(
            component_name="confidence", value=0.0, reason="test"
        ),
        discrepancy_component=RewardComponent(
            component_name="discrepancy", value=0.1, reason="test"
        ),
        unintended_changes_component=RewardComponent(
            component_name="unintended_changes", value=0.0, reason="test"
        ),
    )
    return RewardRecord(
        reward_id=f"REW-{workflow_id}",
        workflow_id=workflow_id,
        exception_id="EXC-001",
        category=category,
        reward_value=reward_value,
        reward_reason="test",
        breakdown=breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Snapshot Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureSnapshot:
    """Test feature snapshot building and correctness."""

    def test_snapshot_from_outcome(self):
        """Feature snapshot captures features at decision time."""
        outcome = _make_outcome(adjustment=5000)
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        assert snapshot.feature_count() > 0
        assert snapshot.financial_features["requested_adjustment_paise"] == 5000.0
        assert snapshot.financial_features["actual_adjustment_paise"] == 5000.0
        assert snapshot.financial_features["discrepancy_eliminated"] == 1.0

    def test_snapshot_has_all_categories(self):
        """Snapshot has financial, structural, evidence, temporal."""
        outcome = _make_outcome()
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        assert len(snapshot.financial_features) > 0
        assert len(snapshot.structural_features) > 0
        assert len(snapshot.evidence_features) > 0
        assert len(snapshot.temporal_features) > 0

    def test_snapshot_from_feedback(self):
        """Feedback augments snapshot with feedback-type features."""
        outcome = _make_outcome()
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        feedback = _make_feedback(feedback_type=FeedbackType.APPROVE)
        augmented = FeatureSnapshotBuilder.from_feedback(snapshot, feedback)
        assert augmented.structural_features["feedback_type_approve"] == 1.0
        assert augmented.structural_features["feedback_type_reject"] == 0.0
        assert augmented.feature_count() > snapshot.feature_count()

    def test_snapshot_to_flat_dict(self):
        """Flat dict contains all features."""
        outcome = _make_outcome()
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        flat = snapshot.to_flat_dict()
        assert "requested_adjustment_paise" in flat
        assert "confidence" in flat

    def test_snapshot_frozen_at_decision_time(self):
        """Snapshot timestamp matches outcome creation time."""
        now = datetime.utcnow()
        outcome = _make_outcome(created_at=now)
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        assert snapshot.captured_at == now

    def test_snapshot_no_leaked_fields(self):
        """Snapshot does not contain ground-truth fields as features."""
        outcome = _make_outcome()
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        flat = snapshot.to_flat_dict()
        assert "true_exception_type" not in flat
        assert "true_resolution" not in flat
        assert "resolvable" not in flat
        assert "risk_category" not in flat

    def test_snapshot_schema_version(self):
        """Snapshot records feature schema version."""
        outcome = _make_outcome()
        snapshot = FeatureSnapshotBuilder.from_outcome(outcome)
        assert snapshot.schema_version == "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Label Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLabels:
    """Test label building correctness."""

    def test_labels_from_outcome(self):
        """Labels capture all outcome dimensions."""
        outcome = _make_outcome(correct=True, verified=True)
        labels = LabelBuilder.from_outcome(outcome)
        assert labels.true_exception_type == "FEE_DIFFERENCE"
        assert labels.true_resolution == "FEE_ADJUSTMENT"
        assert labels.resolution_correct is True
        assert labels.verification_passed is True

    def test_labels_separate_prediction_from_truth(self):
        """Labels keep prediction and ground truth separate."""
        outcome = _make_outcome(
            resolution="FEE_ADJUSTMENT",
            correct=False,
            ground_truth_type="REFUND_ADJUSTMENT",
            ground_truth_resolution="REFUND_ADJUSTMENT",
        )
        labels = LabelBuilder.from_outcome(outcome)
        assert labels.predicted_resolution == "FEE_ADJUSTMENT"
        assert labels.true_resolution == "REFUND_ADJUSTMENT"
        assert labels.resolution_correct is False

    def test_labels_human_correction(self):
        """Labels capture human correction."""
        outcome = _make_outcome(
            correct=False,
            feedback_type=FeedbackType.CORRECT,
            human_override=True,
        )
        labels = LabelBuilder.from_outcome(outcome)
        assert labels.human_corrected is True

    def test_labels_human_rejection(self):
        """Labels capture human rejection."""
        outcome = _make_outcome(
            correct=False,
            feedback_type=FeedbackType.REJECT,
        )
        labels = LabelBuilder.from_outcome(outcome)
        assert labels.human_rejected is True

    def test_labels_discrepancy_status(self):
        """Labels capture discrepancy elimination."""
        outcome = _make_outcome(discrepancy_eliminated=True, adjustment=5000)
        labels = LabelBuilder.from_outcome(outcome)
        assert labels.discrepancy_eliminated is True


# ─────────────────────────────────────────────────────────────────────────────
# Learning Example Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLearningExample:
    """Test learning example building."""

    def test_example_from_records(self):
        """Full example from outcome + feedback + reward."""
        outcome = _make_outcome()
        feedback = _make_feedback()
        reward = _make_reward()
        example = LearningExampleBuilder.from_records(outcome, feedback, reward)
        assert example.features.feature_count() > 0
        assert example.labels.true_exception_type == "FEE_DIFFERENCE"
        assert example.reward_value == 0.8
        assert example.reward_category == "CORRECT_AUTO_RESOLUTION"

    def test_example_has_lineage(self):
        """Example references source data."""
        outcome = _make_outcome()
        example = LearningExampleBuilder.from_records(outcome)
        assert example.lineage_exception_id == "EXC-001"
        assert "EVD-001" in example.lineage_evidence_ids
        assert example.lineage_execution_id == "EXEC-001"

    def test_example_valid(self):
        """Example with features + labels is valid."""
        outcome = _make_outcome()
        example = LearningExampleBuilder.from_records(outcome)
        assert example.is_valid() is True

    def test_example_without_feedback(self):
        """Example can be built without feedback."""
        outcome = _make_outcome()
        example = LearningExampleBuilder.from_records(outcome, feedback=None)
        assert example.features.feature_count() > 0

    def test_example_preserves_decision_context(self):
        """Example preserves guardrail decision and confidence."""
        outcome = _make_outcome(decision="AUTO", confidence=0.92)
        example = LearningExampleBuilder.from_records(outcome)
        assert example.guardrail_decision == "AUTO"
        assert example.confidence == 0.92

    def test_example_summary(self):
        """Example summary is readable."""
        outcome = _make_outcome()
        example = LearningExampleBuilder.from_records(outcome)
        summary = example.summary()
        assert "Example:" in summary
        assert "LEX-" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Quality Check Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityChecks:
    """Test quality checking logic."""

    def test_valid_examples_pass(self):
        """Well-formed examples pass quality checks."""
        examples = []
        for i in range(3):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            examples.append(LearningExampleBuilder.from_records(outcome))
        report = QualityChecker.check(examples)
        assert report.quality_score == 1.0
        assert len(report.issues) == 0

    def test_duplicate_ids_detected(self):
        """Duplicate example IDs are detected."""
        outcome = _make_outcome()
        ex1 = LearningExampleBuilder.from_records(outcome)
        # Create a second with same ID
        ex2 = LearningExampleBuilder.from_records(outcome)
        ex2.example_id = ex1.example_id  # Force duplicate
        report = QualityChecker.check([ex1, ex2])
        assert report.issues_by_type.get("duplicate_examples", 0) >= 1

    def test_missing_features_detected(self):
        """Examples with no features are flagged."""
        ex = LearningExample(
            example_id="LEX-001",
            case_id="CASE-001",
            exception_id="EXC-001",
            workflow_id="WF-001",
            features=FeatureSnapshot(),
            labels=LearningLabels(
                true_exception_type="FEE_DIFFERENCE",
                resolution_correct=True,
            ),
        )
        report = QualityChecker.check([ex])
        assert "missing_features" in report.issues_by_type

    def test_missing_labels_detected(self):
        """Examples with no labels are flagged."""
        outcome = _make_outcome()
        outcome.ground_truth_exception_type = None
        outcome.ground_truth_resolution = None
        outcome.actual_outcome.resolution_correct = None
        example = LearningExampleBuilder.from_records(outcome)
        report = QualityChecker.check([example])
        assert "missing_labels" in report.issues_by_type

    def test_contradictory_labels_detected(self):
        """Contradictory labels (correct + human corrected) are flagged."""
        outcome = _make_outcome(
            correct=True,
            feedback_type=FeedbackType.CORRECT,
            human_override=True,
        )
        example = LearningExampleBuilder.from_records(outcome)
        report = QualityChecker.check([example])
        assert "contradictory_labels" in report.issues_by_type

    def test_quality_score_calculation(self):
        """Quality score reflects proportion of valid examples."""
        good = LearningExampleBuilder.from_records(_make_outcome(workflow_id="WF-001", exception_id="EXC-001", case_id="CASE-001"))
        bad = LearningExample(
            example_id="LEX-BAD",
            case_id="CASE-BAD",
            exception_id="EXC-BAD",
            workflow_id="WF-BAD",
            features=FeatureSnapshot(),
            labels=LearningLabels(),
        )
        report = QualityChecker.check([good, bad])
        assert report.quality_score == 0.5

    def test_missing_evidence_warning(self):
        """Examples without evidence references get a warning."""
        outcome = _make_outcome()
        outcome.lineage.evidence_ids = []
        example = LearningExampleBuilder.from_records(outcome)
        report = QualityChecker.check([example])
        assert "missing_evidence" in report.issues_by_type


# ─────────────────────────────────────────────────────────────────────────────
# Split Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSplits:
    """Test train/validation/test splitting."""

    def test_temporal_split_proportions(self):
        """Temporal split respects ratios."""
        examples = []
        base_time = datetime(2024, 1, 1)
        for i in range(100):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                created_at=base_time + timedelta(hours=i),
            )
            examples.append(LearningExampleBuilder.from_records(outcome))

        splits = SplitStrategy.temporal_split(examples, 0.7, 0.15, 0.15)
        assert splits["train"].example_count == 70
        assert splits["validation"].example_count == 15
        assert splits["test"].example_count == 15

    def test_temporal_split_no_overlap(self):
        """No example appears in multiple splits."""
        examples = []
        base_time = datetime(2024, 1, 1)
        for i in range(50):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                created_at=base_time + timedelta(hours=i),
            )
            examples.append(LearningExampleBuilder.from_records(outcome))

        splits = SplitStrategy.temporal_split(examples)
        all_ids = []
        for split in splits.values():
            all_ids.extend(split.example_ids)
        assert len(all_ids) == 50  # No duplicates

    def test_random_split_reproducible(self):
        """Same seed produces same split."""
        examples = []
        for i in range(50):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            examples.append(LearningExampleBuilder.from_records(outcome))

        splits1 = SplitStrategy.random_split(examples, seed=42)
        splits2 = SplitStrategy.random_split(examples, seed=42)
        assert splits1["train"].example_ids == splits2["train"].example_ids

    def test_split_label_distribution(self):
        """Splits record label distribution."""
        examples = []
        for i in range(30):
            gt_type = "FEE_DIFFERENCE" if i % 2 == 0 else "REFUND_ADJUSTMENT"
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                ground_truth_type=gt_type,
            )
            examples.append(LearningExampleBuilder.from_records(outcome))

        splits = SplitStrategy.temporal_split(examples, 0.6, 0.2, 0.2)
        for split in splits.values():
            total_dist = sum(split.label_distribution.values())
            assert total_dist == split.example_count

    def test_split_strategy_recorded(self):
        """Splits record their strategy."""
        examples = []
        for i in range(10):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            examples.append(LearningExampleBuilder.from_records(outcome))

        splits = SplitStrategy.temporal_split(examples)
        assert splits["train"].split_strategy == "temporal"

        splits_rand = SplitStrategy.random_split(examples, seed=42)
        assert "random" in splits_rand["train"].split_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Detection Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLeakageDetection:
    """Test temporal leakage and contamination detection."""

    def test_no_overlap_passes(self):
        """Non-overlapping splits pass leakage check."""
        train = {"A", "B", "C"}
        val = {"D", "E"}
        test = {"F", "G"}
        issues = LeakageDetector.check_no_case_overlap(train, val, test)
        assert len(issues) == 0

    def test_overlap_detected(self):
        """Overlapping splits are detected."""
        train = {"A", "B", "C"}
        val = {"C", "D"}  # C is in both
        test = {"F", "G"}
        issues = LeakageDetector.check_no_case_overlap(train, val, test)
        assert len(issues) >= 1
        assert any("train-val" in i.description for i in issues)

    def test_feature_leakage_detected(self):
        """Leaked fields in features are detected."""
        features = FeatureSnapshot(
            financial_features={"true_exception_type": 1.0, "requested_adjustment": 5000},
        )
        issues = LeakageDetector.check_feature_leakage(features)
        assert len(issues) >= 1
        assert any("true_exception_type" in i.description for i in issues)

    def test_no_leakage_passes(self):
        """Clean features pass leakage check."""
        features = FeatureSnapshot(
            financial_features={"requested_adjustment_paise": 5000.0},
            structural_features={"has_resolution": 1.0},
        )
        issues = LeakageDetector.check_feature_leakage(features)
        assert len(issues) == 0

    def test_temporal_ordering_check(self):
        """Test cases earlier than train are flagged."""
        base_time = datetime(2024, 1, 1)
        examples = [
            LearningExample(
                example_id="LEX-1", case_id="C1", exception_id="E1", workflow_id="W1",
                features=FeatureSnapshot(financial_features={"x": 1.0}),
                labels=LearningLabels(true_exception_type="A"),
                decision_time=base_time,
            ),
            LearningExample(
                example_id="LEX-2", case_id="C2", exception_id="E2", workflow_id="W2",
                features=FeatureSnapshot(financial_features={"x": 2.0}),
                labels=LearningLabels(true_exception_type="A"),
                decision_time=base_time + timedelta(days=10),
            ),
        ]
        # Train: LEX-2 (later), Test: LEX-1 (earlier) → leakage
        issues = LeakageDetector.check_temporal_ordering(
            examples, train_ids={"LEX-2"}, test_ids={"LEX-1"}
        )
        assert len(issues) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetBuilder:
    """Test the complete dataset builder pipeline."""

    def test_build_dataset(self):
        """Full pipeline: outcomes → examples → quality → splits → dataset."""
        builder = LearningDatasetBuilder()
        base_time = datetime(2024, 1, 1)
        for i in range(20):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                created_at=base_time + timedelta(hours=i),
            )
            builder.add_example(outcome)

        dataset = builder.build_dataset(split_strategy="temporal")
        assert dataset.example_count() == 20
        assert dataset.quality_report is not None
        assert dataset.quality_report.quality_score == 1.0
        assert "train" in dataset.splits
        assert "validation" in dataset.splits
        assert "test" in dataset.splits

    def test_dataset_get_examples_by_split(self):
        """Retrieve examples by split type."""
        builder = LearningDatasetBuilder()
        base_time = datetime(2024, 1, 1)
        for i in range(10):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                created_at=base_time + timedelta(hours=i),
            )
            builder.add_example(outcome)

        dataset = builder.build_dataset(split_strategy="temporal")
        train_ex = dataset.get_examples_by_split(SplitType.TRAIN)
        assert len(train_ex) > 0
        assert all(e.guardrail_decision is not None for e in train_ex)

    def test_batch_add(self):
        """Batch add from outcomes."""
        builder = LearningDatasetBuilder()
        outcomes = [
            _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            for i in range(5)
        ]
        examples = builder.add_examples_batch(outcomes)
        assert len(examples) == 5

    def test_dataset_with_feedback_and_reward(self):
        """Dataset with feedback and reward data."""
        builder = LearningDatasetBuilder()
        for i in range(5):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            feedback = _make_feedback(workflow_id=f"WF-{i:03d}")
            reward = _make_reward(workflow_id=f"WF-{i:03d}")
            builder.add_example(outcome, feedback, reward)

        dataset = builder.build_dataset()
        assert dataset.example_count() == 5
        for ex in dataset.examples:
            assert ex.reward_value is not None

    def test_dataset_idempotent(self):
        """Building dataset twice produces same structure."""
        builder = LearningDatasetBuilder()
        for i in range(10):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
            )
            builder.add_example(outcome)

        ds1 = builder.build_dataset()
        ds2 = builder.build_dataset()
        assert ds1.example_count() == ds2.example_count()
        assert ds1.quality_report.quality_score == ds2.quality_report.quality_score

    def test_dataset_split_summary(self):
        """Split summary reports counts."""
        builder = LearningDatasetBuilder()
        base_time = datetime(2024, 1, 1)
        for i in range(30):
            outcome = _make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                created_at=base_time + timedelta(hours=i),
            )
            builder.add_example(outcome)

        dataset = builder.build_dataset(split_strategy="temporal")
        summary = dataset.split_summary()
        assert sum(summary.values()) == 30


# ─────────────────────────────────────────────────────────────────────────────
# Lineage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataLineage:
    """Test data lineage traceability."""

    def test_lineage_from_outcome(self):
        """Example references source exception, evidence, execution."""
        outcome = _make_outcome()
        example = LearningExampleBuilder.from_records(outcome)
        assert example.lineage_exception_id == "EXC-001"
        assert "EVD-001" in example.lineage_evidence_ids
        assert example.lineage_execution_id == "EXEC-001"
        assert example.lineage_verification_id == "VER-001"

    def test_lineage_with_feedback(self):
        """Example references feedback when available."""
        outcome = _make_outcome()
        feedback = _make_feedback()
        example = LearningExampleBuilder.from_records(outcome, feedback)
        assert example.lineage_feedback_id == "FB-WF-001"

    def test_lineage_with_reward(self):
        """Example references reward when available."""
        outcome = _make_outcome()
        reward = _make_reward()
        example = LearningExampleBuilder.from_records(outcome, reward=reward)
        assert example.lineage_reward_id == "REW-WF-001"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataset(self):
        """Empty dataset builds successfully."""
        builder = LearningDatasetBuilder()
        dataset = builder.build_dataset()
        assert dataset.example_count() == 0
        assert dataset.quality_report.total_examples == 0

    def test_single_example(self):
        """Single example works."""
        builder = LearningDatasetBuilder()
        builder.add_example(_make_outcome(workflow_id="WF-001", exception_id="EXC-001", case_id="CASE-001"))
        dataset = builder.build_dataset()
        assert dataset.example_count() == 1

    def test_all_same_label(self):
        """All examples with same label still split correctly."""
        builder = LearningDatasetBuilder()
        base_time = datetime(2024, 1, 1)
        for i in range(15):
            builder.add_example(_make_outcome(
                workflow_id=f"WF-{i:03d}",
                exception_id=f"EXC-{i:03d}",
                case_id=f"CASE-{i:03d}",
                ground_truth_type="FEE_DIFFERENCE",
                created_at=base_time + timedelta(hours=i),
            ))
        dataset = builder.build_dataset(split_strategy="temporal")
        assert dataset.example_count() == 15
        train_ex = dataset.get_examples_by_split(SplitType.TRAIN)
        assert all(e.labels.true_exception_type == "FEE_DIFFERENCE" for e in train_ex)
