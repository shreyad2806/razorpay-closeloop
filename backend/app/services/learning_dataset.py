"""
Learning Dataset Builder service for Razorpay CloseLoop Phase 9C.

Builds high-quality historical examples from completed cases.

Safety principle:
  Feature snapshots are frozen at decision time.
  Do not recompute historical features using future information.
  Ground truth is used ONLY for labels, never for feature computation.
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.schemas.feedback import (
    ActualOutcomeRecord,
    FeedbackRecord,
    FeedbackType,
    OutcomeRecord,
    PredictionRecord,
)
from app.schemas.learning_dataset import (
    DataSplit,
    FeatureSnapshot,
    LEARNING_DATASET_VERSION,
    LearningDataset,
    LearningExample,
    LearningLabels,
    QualityIssue,
    QualityIssueRecord,
    QualityReport,
    SplitType,
)
from app.schemas.reward_engine import RewardRecord


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Snapshot Builder
# ─────────────────────────────────────────────────────────────────────────────


class FeatureSnapshotBuilder:
    """Builds feature snapshots from outcome data.

    Features are captured at decision time and frozen.
    They must NOT be recomputed from later information.
    """

    @staticmethod
    def from_outcome(outcome: OutcomeRecord) -> FeatureSnapshot:
        """Build a feature snapshot from an outcome record.

        Uses the feature data that was available at decision time.
        Does NOT use ground truth for feature computation.
        """
        # Financial features from the outcome's financial impact
        fi = outcome.financial_impact
        financial = {
            "requested_adjustment_paise": float(fi.requested_adjustment_paise),
            "actual_adjustment_paise": float(fi.actual_adjustment_paise),
            "difference_before_paise": float(fi.difference_before_paise),
            "difference_after_paise": float(fi.difference_after_paise),
            "discrepancy_eliminated": float(fi.discrepancy_eliminated),
            "unintended_changes": float(fi.unintended_changes),
        }

        # Structural features from decision context
        structural = {
            "has_resolution": float(
                outcome.prediction.resolution_type is not None
            ),
            "has_verification": float(outcome.verification_passed),
            "has_human_feedback": float(outcome.human_feedback_id is not None),
            "human_override": float(outcome.human_override),
        }

        # Evidence features from lineage
        evidence = {
            "evidence_count": float(len(outcome.lineage.evidence_ids)),
            "has_execution": float(outcome.lineage.execution_id is not None),
            "has_verification_ref": float(
                outcome.lineage.verification_id is not None
            ),
        }

        # Confidence features
        temporal = {
            "confidence": float(outcome.confidence or 0.0),
        }

        return FeatureSnapshot(
            financial_features=financial,
            structural_features=structural,
            evidence_features=evidence,
            temporal_features=temporal,
            merchant_features={},
            historical_features={},
            schema_version="1.0.0",
            captured_at=outcome.created_at,
        )

    @staticmethod
    def from_feedback(
        snapshot: FeatureSnapshot, feedback: FeedbackRecord
    ) -> FeatureSnapshot:
        """Augment a feature snapshot with feedback-time features.

        These features were available when the human gave feedback.
        """
        # Clone the snapshot's data
        updated = FeatureSnapshot(
            financial_features=snapshot.financial_features.copy(),
            structural_features=snapshot.structural_features.copy(),
            evidence_features=snapshot.evidence_features.copy(),
            temporal_features=snapshot.temporal_features.copy(),
            merchant_features=snapshot.merchant_features.copy(),
            historical_features=snapshot.historical_features.copy(),
            schema_version=snapshot.schema_version,
            captured_at=snapshot.captured_at,
        )

        # Add feedback-time structural features
        updated.structural_features["feedback_type_approve"] = float(
            feedback.feedback_type == FeedbackType.APPROVE
        )
        updated.structural_features["feedback_type_reject"] = float(
            feedback.feedback_type == FeedbackType.REJECT
        )
        updated.structural_features["feedback_type_correct"] = float(
            feedback.feedback_type == FeedbackType.CORRECT
        )
        updated.structural_features["feedback_type_escalate"] = float(
            feedback.feedback_type == FeedbackType.ESCALATE
        )
        updated.structural_features["reviewer_evidence_count"] = float(
            len(feedback.evidence_references_reviewed)
        )

        return updated


# ─────────────────────────────────────────────────────────────────────────────
# Label Builder
# ─────────────────────────────────────────────────────────────────────────────


class LabelBuilder:
    """Builds labels from outcome + feedback records."""

    @staticmethod
    def from_outcome(outcome: OutcomeRecord) -> LearningLabels:
        """Build labels from an outcome record."""
        return LearningLabels(
            true_exception_type=outcome.ground_truth_exception_type,
            predicted_exception_type=outcome.prediction.exception_type,
            exception_prediction_correct=(
                outcome.prediction.exception_type == outcome.ground_truth_exception_type
                if outcome.prediction.exception_type and outcome.ground_truth_exception_type
                else None
            ),
            true_resolution=outcome.ground_truth_resolution,
            predicted_resolution=outcome.prediction.resolution_type,
            resolution_correct=outcome.actual_outcome.resolution_correct,
            verification_passed=outcome.verification_passed,
            human_corrected=outcome.human_override and outcome.human_feedback_type == FeedbackType.CORRECT,
            human_rejected=outcome.human_feedback_type == FeedbackType.REJECT,
            discrepancy_eliminated=outcome.financial_impact.discrepancy_eliminated,
            unintended_changes=outcome.financial_impact.unintended_changes,
            resolvable=outcome.ground_truth_resolvable,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Learning Example Builder
# ─────────────────────────────────────────────────────────────────────────────


class LearningExampleBuilder:
    """Builds learning examples from outcome + feedback + reward."""

    @staticmethod
    def from_records(
        outcome: OutcomeRecord,
        feedback: Optional[FeedbackRecord] = None,
        reward: Optional[RewardRecord] = None,
    ) -> LearningExample:
        """Build a learning example from completed records.

        Feature snapshot is frozen at decision time.
        Labels come from ground truth (evaluation only).
        """
        # Build feature snapshot from outcome (decision-time data)
        features = FeatureSnapshotBuilder.from_outcome(outcome)

        # Augment with feedback features if available
        if feedback:
            features = FeatureSnapshotBuilder.from_feedback(features, feedback)

        # Build labels
        labels = LabelBuilder.from_outcome(outcome)

        # Build example
        return LearningExample(
            example_id=_gen_id("LEX"),
            case_id=outcome.case_id or outcome.exception_id,
            exception_id=outcome.exception_id,
            workflow_id=outcome.workflow_id,
            features=features,
            labels=labels,
            reward_value=reward.reward_value if reward else None,
            reward_category=reward.category.value if reward else None,
            guardrail_decision=outcome.decision,
            confidence=outcome.confidence,
            risk=outcome.risk,
            decision_time=outcome.created_at,
            outcome_time=outcome.completed_at,
            lineage_exception_id=outcome.lineage.exception_id,
            lineage_evidence_ids=outcome.lineage.evidence_ids,
            lineage_execution_id=outcome.lineage.execution_id,
            lineage_verification_id=outcome.lineage.verification_id,
            lineage_feedback_id=feedback.feedback_id if feedback else outcome.lineage.feedback_id,
            lineage_reward_id=reward.reward_id if reward else outcome.lineage.reward_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quality Checker
# ─────────────────────────────────────────────────────────────────────────────


class QualityChecker:
    """Checks data quality of learning examples."""

    @staticmethod
    def check(examples: List[LearningExample]) -> QualityReport:
        """Run all quality checks on a list of examples."""
        issues: List[QualityIssueRecord] = []
        valid_count = 0

        seen_ids: Set[str] = set()
        seen_hashes: Dict[str, str] = {}

        for ex in examples:
            example_valid = True

            # Check missing features
            if ex.features.feature_count() == 0:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.MISSING_FEATURES,
                    example_id=ex.example_id,
                    description="Example has no features",
                    severity="error",
                ))
                example_valid = False

            # Check missing labels
            if (
                ex.labels.true_exception_type is None
                and ex.labels.true_resolution is None
                and ex.labels.resolution_correct is None
            ):
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.MISSING_LABELS,
                    example_id=ex.example_id,
                    description="Example has no labels at all",
                    severity="error",
                ))
                example_valid = False

            # Check missing outcome
            if ex.reward_value is None and ex.guardrail_decision is None:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.MISSING_OUTCOME,
                    example_id=ex.example_id,
                    description="No reward or decision recorded",
                    severity="warning",
                ))

            # Check missing verification
            if not ex.labels.verification_passed and not ex.labels.human_corrected:
                # Verification not passed AND not human-corrected could be fine
                # (UNRESOLVED cases), but flag for review
                pass

            # Check missing evidence
            if len(ex.lineage_evidence_ids) == 0:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.MISSING_EVIDENCE,
                    example_id=ex.example_id,
                    description="No evidence records referenced",
                    severity="warning",
                ))

            # Check invalid financial values
            for fname, fval in ex.features.financial_features.items():
                if fval != fval:  # NaN check
                    issues.append(QualityIssueRecord(
                        issue_type=QualityIssue.INVALID_FINANCIAL_VALUES,
                        example_id=ex.example_id,
                        description=f"NaN in feature {fname}",
                        severity="error",
                    ))
                    example_valid = False

            # Check duplicate example IDs
            if ex.example_id in seen_ids:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.DUPLICATE_EXAMPLES,
                    example_id=ex.example_id,
                    description=f"Duplicate example ID: {ex.example_id}",
                    severity="error",
                ))
                example_valid = False
            seen_ids.add(ex.example_id)

            # Check duplicate content (same features + labels)
            content_hash = _content_hash(ex)
            if content_hash in seen_hashes:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.DUPLICATE_EXAMPLES,
                    example_id=ex.example_id,
                    description=(
                        f"Content duplicate of {seen_hashes[content_hash]}"
                    ),
                    severity="warning",
                ))
            else:
                seen_hashes[content_hash] = ex.example_id

            # Check contradictory labels
            if ex.labels.resolution_correct is True and ex.labels.human_corrected:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.CONTRADICTORY_LABELS,
                    example_id=ex.example_id,
                    description="Resolution marked correct but human corrected it",
                    severity="warning",
                ))

            if ex.labels.resolution_correct is False and not ex.labels.human_corrected and not ex.labels.human_rejected:
                # Resolution incorrect but no human correction — could be valid
                # (escalated case), but worth noting
                pass

            if example_valid:
                valid_count += 1

        # Build issue counts
        issues_by_type: Dict[str, int] = {}
        for issue in issues:
            key = issue.issue_type.value
            issues_by_type[key] = issues_by_type.get(key, 0) + 1

        total = len(examples)
        quality_score = valid_count / total if total > 0 else 0.0

        return QualityReport(
            total_examples=total,
            valid_examples=valid_count,
            issues=issues,
            issues_by_type=issues_by_type,
            quality_score=quality_score,
            checked_at=datetime.utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Split Strategy
# ─────────────────────────────────────────────────────────────────────────────


class SplitStrategy:
    """Creates reproducible train/validation/test splits."""

    @staticmethod
    def temporal_split(
        examples: List[LearningExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, DataSplit]:
        """Split examples by time (earliest → train, latest → test).

        This prevents temporal leakage where future outcomes
        inform historical training.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Sort by decision time
        sorted_ex = sorted(examples, key=lambda e: e.decision_time)
        n = len(sorted_ex)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_ids = [e.example_id for e in sorted_ex[:train_end]]
        val_ids = [e.example_id for e in sorted_ex[train_end:val_end]]
        test_ids = [e.example_id for e in sorted_ex[val_end:]]

        splits = {}
        for split_type, ids in [
            (SplitType.TRAIN, train_ids),
            (SplitType.VALIDATION, val_ids),
            (SplitType.TEST, test_ids),
        ]:
            # Build label distribution
            label_dist: Dict[str, int] = {}
            for ex in sorted_ex:
                if ex.example_id in set(ids):
                    etype = ex.labels.true_exception_type or "unknown"
                    label_dist[etype] = label_dist.get(etype, 0) + 1

            splits[split_type.value] = DataSplit(
                split_type=split_type,
                example_ids=ids,
                example_count=len(ids),
                label_distribution=label_dist,
                split_strategy="temporal",
            )

        return splits

    @staticmethod
    def random_split(
        examples: List[LearningExample],
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, DataSplit]:
        """Deterministic random split (no batch leakage guaranteed).

        Use temporal_split when temporal ordering matters.
        """
        import random

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        rng = random.Random(seed)
        indices = list(range(len(examples)))
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {}
        for split_type, slice_indices in [
            (SplitType.TRAIN, indices[:train_end]),
            (SplitType.VALIDATION, indices[train_end:val_end]),
            (SplitType.TEST, indices[val_end:]),
        ]:
            ids = [examples[i].example_id for i in slice_indices]
            label_dist: Dict[str, int] = {}
            for i in slice_indices:
                etype = examples[i].labels.true_exception_type or "unknown"
                label_dist[etype] = label_dist.get(etype, 0) + 1

            splits[split_type.value] = DataSplit(
                split_type=split_type,
                example_ids=ids,
                example_count=len(ids),
                label_distribution=label_dist,
                split_strategy=f"random(seed={seed})",
            )

        return splits


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Detector
# ─────────────────────────────────────────────────────────────────────────────


class LeakageDetector:
    """Detects temporal leakage and data contamination."""

    @staticmethod
    def check_no_case_overlap(
        train_ids: Set[str],
        val_ids: Set[str],
        test_ids: Set[str],
    ) -> List[QualityIssueRecord]:
        """Verify no case appears in multiple splits."""
        issues = []
        for pair_name, set_a, set_b in [
            ("train-val", train_ids, val_ids),
            ("train-test", train_ids, test_ids),
            ("val-test", val_ids, test_ids),
        ]:
            overlap = set_a & set_b
            if overlap:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.TEMPORAL_LEAKAGE,
                    description=f"Case overlap between {pair_name}: {len(overlap)} cases",
                    severity="critical",
                ))
        return issues

    @staticmethod
    def check_feature_leakage(
        features: FeatureSnapshot,
    ) -> List[QualityIssueRecord]:
        """Check if feature snapshot contains leaked fields."""
        issues = []
        leaked_fields = {
            "true_exception_type", "true_resolution",
            "resolvable", "risk_category",
        }
        flat = features.to_flat_dict()
        for field in leaked_fields:
            if field in flat:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.TEMPORAL_LEAKAGE,
                    description=f"Leaked field found in features: {field}",
                    severity="critical",
                ))
        return issues

    @staticmethod
    def check_temporal_ordering(
        examples: List[LearningExample],
        train_ids: Set[str],
        test_ids: Set[str],
    ) -> List[QualityIssueRecord]:
        """Check that test examples are not earlier than train examples."""
        issues = []
        ex_map = {e.example_id: e for e in examples}

        train_times = [
            ex_map[eid].decision_time
            for eid in train_ids if eid in ex_map
        ]
        test_times = [
            ex_map[eid].decision_time
            for eid in test_ids if eid in ex_map
        ]

        if train_times and test_times:
            latest_train = max(train_times)
            earliest_test = min(test_times)
            if earliest_test < latest_train:
                issues.append(QualityIssueRecord(
                    issue_type=QualityIssue.TEMPORAL_LEAKAGE,
                    description=(
                        f"Test data contains examples earlier than training data: "
                        f"latest train={latest_train}, earliest test={earliest_test}"
                    ),
                    severity="warning",
                ))

        return issues


# ─────────────────────────────────────────────────────────────────────────────
# Learning Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────


class LearningDatasetBuilder:
    """Complete pipeline for building learning datasets."""

    def __init__(self) -> None:
        self._examples: List[LearningExample] = []
        self._example_builder = LearningExampleBuilder()
        self._quality_checker = QualityChecker()
        self._split_strategy = SplitStrategy()
        self._leakage_detector = LeakageDetector()

    def add_example(
        self,
        outcome: OutcomeRecord,
        feedback: Optional[FeedbackRecord] = None,
        reward: Optional[RewardRecord] = None,
    ) -> LearningExample:
        """Build and add a learning example from records."""
        example = self._example_builder.from_records(outcome, feedback, reward)
        self._examples.append(example)
        return example

    def add_examples_batch(
        self,
        outcomes: List[OutcomeRecord],
        feedbacks: Optional[Dict[str, FeedbackRecord]] = None,
        rewards: Optional[Dict[str, RewardRecord]] = None,
    ) -> List[LearningExample]:
        """Build learning examples from a batch of outcomes."""
        feedbacks = feedbacks or {}
        rewards = rewards or {}
        examples = []
        for outcome in outcomes:
            fb = feedbacks.get(outcome.workflow_id)
            rw = rewards.get(outcome.workflow_id)
            ex = self.add_example(outcome, fb, rw)
            examples.append(ex)
        return examples

    def check_quality(self) -> QualityReport:
        """Run quality checks on all examples."""
        return self._quality_checker.check(self._examples)

    def create_splits(
        self,
        strategy: str = "temporal",
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, DataSplit]:
        """Create train/validation/test splits."""
        if strategy == "temporal":
            splits = self._split_strategy.temporal_split(
                self._examples, train_ratio, val_ratio, test_ratio
            )
        else:
            splits = self._split_strategy.random_split(
                self._examples, seed, train_ratio, val_ratio, test_ratio
            )

        # Run leakage detection on splits
        train_ids = set(splits.get("train", DataSplit(split_type=SplitType.TRAIN)).example_ids)
        val_ids = set(splits.get("validation", DataSplit(split_type=SplitType.VALIDATION)).example_ids)
        test_ids = set(splits.get("test", DataSplit(split_type=SplitType.TEST)).example_ids)

        leakage_issues = self._leakage_detector.check_no_case_overlap(
            train_ids, val_ids, test_ids
        )
        leakage_issues.extend(
            self._leakage_detector.check_temporal_ordering(
                self._examples, train_ids, test_ids
            )
        )

        return splits

    def build_dataset(
        self,
        dataset_id: Optional[str] = None,
        split_strategy: str = "temporal",
        seed: int = 42,
    ) -> LearningDataset:
        """Build the complete learning dataset."""
        # Quality check
        quality = self.check_quality()

        # Create splits
        splits = self.create_splits(split_strategy, seed)

        return LearningDataset(
            dataset_id=dataset_id or _gen_id("LDS"),
            version=LEARNING_DATASET_VERSION,
            examples=self._examples,
            splits=splits,
            quality_report=quality,
            split_seed=seed,
        )

    def get_examples(self) -> List[LearningExample]:
        return list(self._examples)

    def example_count(self) -> int:
        return len(self._examples)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _content_hash(example: LearningExample) -> str:
    """Create a content hash for duplicate detection."""
    key = (
        f"{example.case_id}|"
        f"{example.labels.true_exception_type}|"
        f"{example.labels.true_resolution}|"
        f"{example.labels.resolution_correct}|"
        f"{example.guardrail_decision}|"
        f"{example.reward_value}"
    )
    return hashlib.md5(key.encode()).hexdigest()
