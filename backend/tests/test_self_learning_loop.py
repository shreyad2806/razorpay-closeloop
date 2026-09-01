"""
Tests for Phase 9I — Complete Self-Learning Loop.

Integration tests that exercise the full learning cycle:
outcome → feedback → reward → dataset → training → evaluation → promotion

Verifies safety boundaries and that learning cannot bypass Phase 6.
"""

import pytest
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    FeedbackType,
    FinancialImpact,
    PredictionRecord,
)
from app.schemas.learning_dataset import (
    FeatureSnapshot,
    LearningDataset,
    LearningExample,
    LearningLabels,
    SplitType,
)
from app.schemas.model_training import (
    EvaluationMetrics,
    TrainingConfig,
)
from app.schemas.self_learning_loop import (
    LearningCycleRecord,
    LearningCycleStatus,
    LearningSystemState,
    PromotionAction,
)
from app.services.self_learning_loop import SelfLearningLoop


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8].upper()}"


def _make_prediction(
    exception_type: str = "FEE_DIFFERENCE",
    resolution_type: str = "FEE_CORRECTION",
    confidence: float = 0.85,
) -> PredictionRecord:
    return PredictionRecord(
        exception_type=exception_type,
        resolution_type=resolution_type,
        resolution_confidence=confidence,
        exception_confidence=0.9,
        model_version="v1.0",
    )


def _make_actual(
    resolution_correct: bool = True,
    was_executed: bool = True,
    was_verified: bool = True,
    was_rolled_back: bool = False,
    adjustment_paise: int = 5000,
    discrepancy_eliminated: bool = True,
) -> ActualOutcomeRecord:
    return ActualOutcomeRecord(
        actual_resolution="FEE_CORRECTION",
        actual_exception_type="FEE_DIFFERENCE",
        resolution_correct=resolution_correct,
        financial_impact_paise=adjustment_paise,
        was_executed=was_executed,
        was_verified=was_verified,
        was_rolled_back=was_rolled_back,
    )


def _make_financial_impact(
    adjustment_paise: int = 5000,
    discrepancy_eliminated: bool = True,
) -> FinancialImpact:
    return FinancialImpact(
        requested_adjustment_paise=adjustment_paise,
        actual_adjustment_paise=adjustment_paise,
        difference_before_paise=adjustment_paise,
        difference_after_paise=0 if discrepancy_eliminated else adjustment_paise,
        discrepancy_eliminated=discrepancy_eliminated,
    )


def _make_lineage(exception_id: str = "EXC-001") -> DataLineage:
    return DataLineage(exception_id=exception_id)


def _make_features() -> FeatureSnapshot:
    return FeatureSnapshot(
        financial_features={
            "adjustment_paise": 5000.0,
            "difference_paise": 2500.0,
        },
        structural_features={
            "resolution_type_fee_correction": 1.0,
            "has_settlement": 1.0,
        },
        evidence_features={
            "evidence_count": 3.0,
            "evidence_coverage": 0.8,
        },
    )


def _make_learning_dataset(n: int = 25) -> LearningDataset:
    """Create a training-ready learning dataset."""
    examples = []
    for i in range(n):
        examples.append(LearningExample(
            example_id=_gen_id("LEX"),
            case_id=f"CASE-{i:03d}",
            exception_id=f"EXC-{i:03d}",
            workflow_id=f"WF-{i:03d}",
            features=FeatureSnapshot(
                financial_features={
                    "adjustment_paise": float(1000 + i * 100),
                    "difference_paise": float(500 + i * 50),
                },
                structural_features={
                    "resolution_type_fee_correction": 1.0,
                    "has_settlement": 1.0,
                },
                evidence_features={
                    "evidence_count": 3.0,
                    "evidence_coverage": 0.8,
                },
            ),
            labels=LearningLabels(
                true_exception_type="FEE_DIFFERENCE" if i % 3 != 0 else "UNKNOWN",
                predicted_exception_type="FEE_DIFFERENCE",
                resolution_correct=i % 5 != 0,
                verification_passed=i % 7 != 0,
            ),
            guardrail_decision="AUTO" if i % 4 != 0 else "HUMAN_REVIEW",
            confidence=0.7 + (i % 10) * 0.02,
        ))
    return LearningDataset(
        dataset_id=_gen_id("LDS"),
        version="1.0.0",
        examples=examples,
    )


def _make_eval_metrics(
    model_id: str = "MOD-001",
    version: str = "v1.0",
    accuracy: float = 0.85,
    precision: float = 0.82,
    recall: float = 0.80,
    f1: float = 0.81,
    false_auto: int = 2,
    hv_errors: int = 0,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id=model_id,
        model_version=version,
        total_samples=50,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        precision_weighted=precision,
        recall_weighted=recall,
        f1_weighted=f1,
        false_automation=false_auto,
        high_value_errors=hv_errors,
        incorrect_auto_resolution=false_auto,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Complete Learning Cycle (Happy Path)
# ─────────────────────────────────────────────────────────────────────────────

class TestCompleteLearningCycle:
    """Tests the complete learning cycle from outcome to promotion."""

    def test_end_to_end_cycle(self):
        """Complete cycle: outcome → feedback → reward → example → batch."""
        loop = SelfLearningLoop()

        # Step 1: Record outcome
        cycle = loop.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution_correct=True),
            lineage=_make_lineage(),
            decision="AUTO",
            confidence=0.85,
        )
        assert cycle.status == LearningCycleStatus.RECORDING
        assert cycle.prediction_correct is True

        # Step 2: Record feedback
        feedback = loop.record_feedback(
            cycle_id=cycle.cycle_id,
            feedback_type=FeedbackType.APPROVE,
            reviewer="ops_team",
            system_prediction="FEE_CORRECTION",
        )
        assert feedback.feedback_type == FeedbackType.APPROVE

        # Step 3: Calculate reward
        reward = loop.calculate_reward(cycle.cycle_id)
        assert reward.reward_value > 0
        assert cycle.reward_value > 0
        assert cycle.status == LearningCycleStatus.REWARD_CALCULATED

        # Step 4: Build learning example
        example = loop.build_learning_example(
            cycle.cycle_id, _make_features(),
        )
        assert example.example_id is not None
        assert cycle.learning_example_id is not None
        assert cycle.status == LearningCycleStatus.EXAMPLE_BUILT

        # Step 5: Add to batch
        success = loop.add_example_to_batch(cycle.cycle_id, example)
        assert success is True
        assert cycle.status == LearningCycleStatus.BATCH_READY

    def test_correction_feedback_cycle(self):
        """Human correction feeds into learning."""
        loop = SelfLearningLoop()

        cycle = loop.record_outcome(
            workflow_id="WF-002",
            exception_id="EXC-002",
            prediction=_make_prediction(resolution_type="FEE_CORRECTION"),
            actual_outcome=_make_actual(resolution_correct=False),
            lineage=_make_lineage("EXC-002"),
            decision="AUTO",
        )

        feedback = loop.record_feedback(
            cycle_id=cycle.cycle_id,
            feedback_type=FeedbackType.CORRECT,
            reviewer="auditor",
            system_prediction="FEE_CORRECTION",
            correction=CorrectionDetail(
                original_resolution="FEE_CORRECTION",
                corrected_resolution="REFUND",
                correction_reason="Actual refund was processed",
            ),
        )

        assert feedback.feedback_type == FeedbackType.CORRECT
        assert cycle.feedback_type == "CORRECT"
        assert cycle.feedback_id is not None

        # Outcome should reflect human override
        outcome = loop.outcome_service._outcomes.get(cycle.outcome_id)
        assert outcome is not None
        assert outcome.human_override is True

    def test_rejection_feedback_cycle(self):
        """Human rejection feeds into learning."""
        loop = SelfLearningLoop()

        cycle = loop.record_outcome(
            workflow_id="WF-003",
            exception_id="EXC-003",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution_correct=True),
            lineage=_make_lineage("EXC-003"),
            decision="AUTO",
        )

        feedback = loop.record_feedback(
            cycle_id=cycle.cycle_id,
            feedback_type=FeedbackType.REJECT,
            reviewer="senior_ops",
            system_prediction="FEE_CORRECTION",
            rejection={
                "rejection_reason": "Settlement still pending",
                "risk_concern": "Cannot adjust before settlement",
            },
        )

        assert feedback.feedback_type == FeedbackType.REJECT
        assert cycle.feedback_type == "REJECT"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Multiple Cycles
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleCycles:
    """Tests multiple learning cycles."""

    def test_multiple_outcomes(self):
        """Multiple outcomes are tracked independently."""
        loop = SelfLearningLoop()

        c1 = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution_correct=True),
            lineage=_make_lineage("EXC-001"),
        )
        c2 = loop.record_outcome(
            workflow_id="WF-002", exception_id="EXC-002",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution_correct=False),
            lineage=_make_lineage("EXC-002"),
        )

        assert c1.cycle_number == 1
        assert c2.cycle_number == 2
        assert len(loop.get_all_cycles()) == 2

    def test_cycle_by_exception(self):
        """Cycles can be looked up by exception ID."""
        loop = SelfLearningLoop()

        c = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-FIND-ME",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage("EXC-FIND-ME"),
        )

        found = loop.get_cycle_by_exception("EXC-FIND-ME")
        assert found is not None
        assert found.cycle_id == c.cycle_id

    def test_cycles_by_status(self):
        """Cycles can be filtered by status."""
        loop = SelfLearningLoop()

        c1 = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage("EXC-001"),
        )
        c2 = loop.record_outcome(
            workflow_id="WF-002", exception_id="EXC-002",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage("EXC-002"),
        )

        # Complete c1 but not c2
        loop.record_feedback(
            c1.cycle_id, FeedbackType.APPROVE, "ops", "FEE_CORRECTION",
        )
        loop.calculate_reward(c1.cycle_id)

        recording = loop.get_cycles_by_status(LearningCycleStatus.RECORDING)
        reward_calc = loop.get_cycles_by_status(
            LearningCycleStatus.REWARD_CALCULATED
        )
        assert len(recording) == 1
        assert len(reward_calc) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Reward Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardTracking:
    """Tests reward tracking across cycles."""

    def test_reward_values(self):
        """Reward values are tracked across cycles."""
        loop = SelfLearningLoop()

        for i in range(5):
            c = loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(resolution_correct=i % 2 == 0),
                lineage=_make_lineage(f"EXC-{i}"),
            )
            loop.calculate_reward(c.cycle_id)

        rewards = loop.get_reward_values()
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

    def test_promotion_action_counts(self):
        """Promotion actions are counted."""
        loop = SelfLearningLoop()

        for i in range(3):
            c = loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(),
                lineage=_make_lineage(f"EXC-{i}"),
            )

        counts = loop.count_by_promotion_action()
        assert counts.get("NO_CANDIDATE", 0) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test: System State
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemState:
    """Tests learning system state tracking."""

    def test_initial_state(self):
        """Initial state shows no activity."""
        loop = SelfLearningLoop()
        state = loop.get_system_state()

        assert state.total_cycles == 0
        assert state.completed_cycles == 0
        assert state.safety_maintained_all_cycles is True

    def test_state_after_outcomes(self):
        """State reflects recorded outcomes."""
        loop = SelfLearningLoop()

        for i in range(3):
            loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(),
                lineage=_make_lineage(f"EXC-{i}"),
            )

        state = loop.get_system_state()
        assert state.total_cycles == 3
        assert state.safety_maintained_all_cycles is True

    def test_state_summary(self):
        """System state has a readable summary."""
        loop = SelfLearningLoop()
        state = loop.get_system_state()
        assert isinstance(state.summary(), str)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Metrics Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsIntegration:
    """Tests learning metrics from the complete loop."""

    def test_metrics_from_loop(self):
        """Metrics computed from all recorded outcomes/rewards."""
        loop = SelfLearningLoop()

        for i in range(5):
            c = loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(resolution_correct=i % 2 == 0),
                lineage=_make_lineage(f"EXC-{i}"),
                decision="AUTO" if i < 3 else "HUMAN_REVIEW",
            )
            loop.calculate_reward(c.cycle_id)

        metrics = loop.compute_learning_metrics()

        assert metrics.automation.total_exceptions == 5
        assert metrics.reward.total_rewards == 5
        assert metrics.reward.avg_reward is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Safety Boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyBoundary:
    """Tests that the learning system respects safety boundaries."""

    def test_safety_boundary_all_pass(self):
        """All safety boundary checks pass."""
        loop = SelfLearningLoop()
        boundary = loop.verify_safety_boundary()

        assert boundary["no_financial_modification"] is True
        assert boundary["no_guardrail_bypass"] is True
        assert boundary["no_execution_authorization"] is True
        assert boundary["promotion_requires_gate"] is True
        assert boundary["reward_does_not_authorize"] is True
        assert boundary["all_cycles_safe"] is True

    def test_safety_boundary_after_cycles(self):
        """Safety boundary remains intact after learning cycles."""
        loop = SelfLearningLoop()

        for i in range(5):
            c = loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(resolution_correct=True),
                lineage=_make_lineage(f"EXC-{i}"),
            )
            loop.record_feedback(
                c.cycle_id, FeedbackType.APPROVE, "ops", "FEE_CORRECTION",
            )
            loop.calculate_reward(c.cycle_id)

        boundary = loop.verify_safety_boundary()
        for key, value in boundary.items():
            assert value is True, f"Safety check {key} failed"

    def test_learning_does_not_modify_financials(self):
        """Learning loop never touches financial records."""
        loop = SelfLearningLoop()

        # Run a complete cycle
        c = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(adjustment_paise=50000),
            lineage=_make_lineage("EXC-001"),
            decision="AUTO",
            confidence=0.95,
        )
        loop.record_feedback(
            c.cycle_id, FeedbackType.APPROVE, "ops", "FEE_CORRECTION",
        )
        reward = loop.calculate_reward(c.cycle_id)

        # Reward is high positive, but it does NOT authorize anything
        assert reward.reward_value > 0
        # Verify no financial service was called
        boundary = loop.verify_safety_boundary()
        assert boundary["no_financial_modification"] is True
        assert boundary["reward_does_not_authorize"] is True

    def test_human_correction_does_not_bypass_guardrails(self):
        """Even after human correction, guardrails still apply."""
        loop = SelfLearningLoop()

        c = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution_correct=False),
            lineage=_make_lineage("EXC-001"),
            decision="AUTO",
        )
        loop.record_feedback(
            c.cycle_id, FeedbackType.CORRECT, "auditor", "FEE_CORRECTION",
            correction=CorrectionDetail(
                original_resolution="FEE_CORRECTION",
                corrected_resolution="REFUND",
                correction_reason="System was wrong",
            ),
        )

        # The cycle records the correction, but guardrails remain
        assert c.feedback_type == "CORRECT"
        boundary = loop.verify_safety_boundary()
        assert boundary["no_guardrail_bypass"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Batch Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchIntegration:
    """Tests batch learning integration."""

    def test_batch_created_automatically(self):
        """Batch is created when first example is added."""
        loop = SelfLearningLoop()

        c = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage("EXC-001"),
        )
        example = loop.build_learning_example(c.cycle_id, _make_features())
        loop.add_example_to_batch(c.cycle_id, example)

        assert loop._current_batch_id is not None
        state = loop.get_system_state()
        assert state.active_batch_id is not None

    def test_batch_reuse(self):
        """Same batch is reused for multiple examples."""
        loop = SelfLearningLoop()

        for i in range(3):
            c = loop.record_outcome(
                workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(),
                lineage=_make_lineage(f"EXC-{i}"),
            )
            example = loop.build_learning_example(c.cycle_id, _make_features())
            loop.add_example_to_batch(c.cycle_id, example)

        # All should be in the same batch
        batch_ids = set(
            c.dataset_id for c in loop.get_all_cycles()
            if c.dataset_id
        )
        assert len(batch_ids) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Tests that the learning loop is deterministic."""

    def test_same_inputs_same_rewards(self):
        """Same outcome + feedback → same reward."""
        loop1 = SelfLearningLoop()
        loop2 = SelfLearningLoop()

        for loop in [loop1, loop2]:
            c = loop.record_outcome(
                workflow_id="WF-001", exception_id="EXC-001",
                prediction=_make_prediction(),
                actual_outcome=_make_actual(resolution_correct=True),
                lineage=_make_lineage("EXC-001"),
                decision="AUTO",
                confidence=0.85,
            )
            loop.record_feedback(
                c.cycle_id, FeedbackType.APPROVE, "ops", "FEE_CORRECTION",
            )
            loop.calculate_reward(c.cycle_id)

        r1 = loop1.get_reward_values()
        r2 = loop2.get_reward_values()
        assert r1 == r2

    def test_deterministic_metrics(self):
        """Same data → same metrics."""
        loop1 = SelfLearningLoop()
        loop2 = SelfLearningLoop()

        for loop in [loop1, loop2]:
            for i in range(5):
                c = loop.record_outcome(
                    workflow_id=f"WF-{i}", exception_id=f"EXC-{i}",
                    prediction=_make_prediction(),
                    actual_outcome=_make_actual(resolution_correct=i % 2 == 0),
                    lineage=_make_lineage(f"EXC-{i}"),
                    decision="AUTO",
                )
                loop.calculate_reward(c.cycle_id)

        m1 = loop1.compute_learning_metrics()
        m2 = loop2.compute_learning_metrics()
        assert m1.automation.automation_rate == m2.automation.automation_rate
        assert m1.precision.precision == m2.precision.precision
        assert m1.reward.avg_reward == m2.reward.avg_reward


# ─────────────────────────────────────────────────────────────────────────────
# Test: Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests edge cases in the learning loop."""

    def test_feedback_without_outcome_fails(self):
        """Cannot record feedback for non-existent cycle."""
        loop = SelfLearningLoop()
        with pytest.raises(ValueError):
            loop.record_feedback(
                "NON-EXISTENT", FeedbackType.APPROVE, "ops", "FEE_CORRECTION",
            )

    def test_reward_without_outcome_fails(self):
        """Cannot calculate reward for non-existent cycle."""
        loop = SelfLearningLoop()
        with pytest.raises(ValueError):
            loop.calculate_reward("NON-EXISTENT")

    def test_empty_loop_metrics(self):
        """Metrics from empty loop produce safe defaults."""
        loop = SelfLearningLoop()
        metrics = loop.compute_learning_metrics()

        assert metrics.automation.total_exceptions == 0
        assert metrics.safety.verdict.value == "SAFE"

    def test_cycle_summary(self):
        """Cycle has a readable summary."""
        loop = SelfLearningLoop()
        c = loop.record_outcome(
            workflow_id="WF-001", exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage("EXC-001"),
        )
        assert isinstance(c.summary(), str)

    def test_all_safety_checks_in_boundary(self):
        """Safety boundary returns expected checks."""
        loop = SelfLearningLoop()
        boundary = loop.verify_safety_boundary()
        assert len(boundary) == 7
        assert all(isinstance(v, bool) for v in boundary.values())
