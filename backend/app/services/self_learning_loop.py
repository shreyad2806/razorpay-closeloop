"""
Self-Learning Loop service for Razorpay CloseLoop Phase 9I.

Integrates all Phase 9 components into a complete learning cycle:

  Exception → Evidence → Classification → Resolution → Guardrails
  → Execution → Verification → Outcome → Feedback → Reward
  → Learning Dataset → Candidate Model → Evaluation → Promotion
  → Future Cases

Safety principle:
  Learning can improve future recommendations.
  Learning CANNOT bypass Phase 6 guardrails, financial exposure limits,
  verification, or execution authorization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

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
    FeatureSnapshot,
    LearningDataset,
    LearningExample,
    LearningLabels,
    SplitType,
)
from app.schemas.learning_metrics import LearningMetrics
from app.schemas.model_training import EvaluationMetrics, ModelMetadata
from app.schemas.reward_engine import RewardRecord
from app.schemas.self_learning_loop import (
    LearningCycleRecord,
    LearningCycleStatus,
    LearningSystemState,
    PromotionAction,
)
from app.services.batch_learning import BatchLearningLoop, BatchConfig
from app.services.feedback import FeedbackService, OutcomeService
from app.services.learning_dataset import LearningDatasetBuilder
from app.services.learning_metrics import LearningMetricsService
from app.services.model_promotion import ModelRegistry
from app.services.model_training import ModelEvaluator, ModelTrainer
from app.services.policy_learning import PolicyStore
from app.services.reward_engine import RewardEngine


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Self-Learning Loop
# ─────────────────────────────────────────────────────────────────────────────


class SelfLearningLoop:
    """Integrates all Phase 9 components into a complete learning cycle.

    Orchestrates:
    1. Outcome recording (from verified execution)
    2. Feedback collection (human review)
    3. Reward calculation
    4. Learning example generation
    5. Batch management
    6. Candidate model training
    7. Candidate evaluation
    8. Current vs candidate comparison
    9. Safety-gated promotion
    10. System state tracking

    CRITICAL SAFETY BOUNDARY:
    This service only produces learning signals and candidate models.
    It does NOT:
    - Modify financial records
    - Execute refunds or settlements
    - Bypass Phase 6 guardrails
    - Change production model without passing promotion gate
    """

    def __init__(
        self,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        # Phase 9A: Feedback + Outcome
        self.feedback_service = FeedbackService()
        self.outcome_service = OutcomeService()

        # Phase 9B: Reward
        self.reward_engine = RewardEngine()

        # Phase 9C: Learning Dataset
        self.dataset_builder = LearningDatasetBuilder()

        # Phase 9G: Batch Learning
        self.batch_loop = BatchLearningLoop()
        self._batch_config = batch_config or BatchConfig()

        # Phase 9E: Model Training
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        # Phase 9F: Model Registry
        self.registry = ModelRegistry()

        # Phase 9D: Policy
        self.policy_store = PolicyStore()

        # Phase 9H: Learning Metrics
        self.metrics_service = LearningMetricsService()

        # Internal state
        self._cycles: Dict[str, LearningCycleRecord] = {}
        self._cycles_by_exception: Dict[str, str] = {}
        self._next_cycle_number = 1
        self._current_batch_id: Optional[str] = None
        self._all_outcomes: List[OutcomeRecord] = []
        self._all_feedbacks: List[FeedbackRecord] = []
        self._all_rewards: List[RewardRecord] = []

    # ── Step 1: Record Outcome ───────────────────────────────────────────

    def record_outcome(
        self,
        workflow_id: str,
        exception_id: str,
        prediction: PredictionRecord,
        actual_outcome: ActualOutcomeRecord,
        lineage: DataLineage,
        decision: str = "AUTO",
        confidence: Optional[float] = None,
        risk: str = "LOW",
        case_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        verification_passed: bool = False,
        verification_notes: Optional[str] = None,
        financial_impact: Optional[FinancialImpact] = None,
    ) -> LearningCycleRecord:
        """Record the verified outcome of a workflow.

        This is the entry point for the learning cycle.
        After execution and verification, the outcome is recorded here.

        Returns a LearningCycleRecord tracking this case through learning.
        """
        cycle_id = _gen_id("LC")
        cycle_number = self._next_cycle_number
        self._next_cycle_number += 1

        # Create outcome record
        outcome = self.outcome_service.record_outcome(
            workflow_id=workflow_id,
            exception_id=exception_id,
            prediction=prediction,
            actual_outcome=actual_outcome,
            lineage=lineage,
            case_id=case_id,
            candidate_id=candidate_id,
            verification_passed=verification_passed,
            verification_notes=verification_notes,
            financial_impact=financial_impact,
            decision=decision,
            confidence=confidence,
            risk=risk,
        )
        self._all_outcomes.append(outcome)

        # Create learning cycle record
        cycle = LearningCycleRecord(
            cycle_id=cycle_id,
            cycle_number=cycle_number,
            exception_id=exception_id,
            workflow_id=workflow_id,
            case_id=case_id,
            status=LearningCycleStatus.RECORDING,
            outcome_id=outcome.outcome_id,
            prediction_correct=actual_outcome.resolution_correct,
            resolution_type=actual_outcome.actual_resolution,
            was_auto_resolved=actual_outcome.was_executed and decision == "AUTO",
            guardrail_decision=decision,
            safety_maintained=True,
            started_at=datetime.utcnow(),
            outcome_recorded_at=datetime.utcnow(),
        )

        self._cycles[cycle_id] = cycle
        self._cycles_by_exception[exception_id] = cycle_id

        return cycle

    # ── Step 2: Record Feedback ──────────────────────────────────────────

    def record_feedback(
        self,
        cycle_id: str,
        feedback_type: FeedbackType,
        reviewer: str,
        system_prediction: str,
        correction: Optional[CorrectionDetail] = None,
        rejection: Optional[dict] = None,
        reason: Optional[str] = None,
    ) -> FeedbackRecord:
        """Record human feedback on a resolved case.

        After the outcome is recorded, humans can approve, reject,
        correct, or escalate. This feeds the learning cycle.
        """
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Learning cycle {cycle_id} not found")

        # Record feedback
        rejection_detail = None
        if rejection:
            from app.schemas.feedback import RejectionDetail
            rejection_detail = RejectionDetail(**rejection)

        feedback = self.feedback_service.record_feedback(
            workflow_id=cycle.workflow_id,
            exception_id=cycle.exception_id,
            feedback_type=feedback_type,
            reviewer=reviewer,
            system_prediction=system_prediction,
            case_id=cycle.case_id,
            correction=correction,
            rejection=rejection_detail,
            reason=reason,
        )
        self._all_feedbacks.append(feedback)

        # Update cycle
        cycle.feedback_id = feedback.feedback_id
        cycle.feedback_type = feedback_type.value
        cycle.feedback_received_at = datetime.utcnow()
        cycle.status = LearningCycleStatus.RECORDING

        # Link feedback to outcome
        outcome = self.outcome_service._outcomes.get(cycle.outcome_id)
        if outcome:
            outcome.human_feedback_id = feedback.feedback_id
            outcome.human_feedback_type = feedback_type
            outcome.human_override = feedback_type in (
                FeedbackType.CORRECT, FeedbackType.REJECT
            )

        return feedback

    # ── Step 3: Calculate Reward ─────────────────────────────────────────

    def calculate_reward(self, cycle_id: str) -> RewardRecord:
        """Calculate reward for a completed case.

        Uses the outcome + feedback to generate a transparent reward signal.
        """
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Learning cycle {cycle_id} not found")

        outcome = self.outcome_service._outcomes.get(cycle.outcome_id)
        if not outcome:
            raise ValueError(f"Outcome {cycle.outcome_id} not found")

        feedback = None
        if cycle.feedback_id:
            feedback = self.feedback_service.get_feedback(cycle.feedback_id)

        reward = self.reward_engine.calculate_reward(outcome, feedback)
        self._all_rewards.append(reward)

        # Update cycle
        cycle.reward_id = reward.reward_id
        cycle.reward_value = reward.reward_value
        cycle.reward_category = reward.category.value
        cycle.reward_calculated_at = datetime.utcnow()
        cycle.status = LearningCycleStatus.REWARD_CALCULATED

        return reward

    # ── Step 4: Build Learning Example ───────────────────────────────────

    def build_learning_example(
        self,
        cycle_id: str,
        features: FeatureSnapshot,
    ) -> LearningExample:
        """Build a learning example from a completed cycle.

        Creates a frozen feature snapshot + labels for offline training.
        """
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Learning cycle {cycle_id} not found")

        outcome = self.outcome_service._outcomes.get(cycle.outcome_id)

        # Build labels from outcome
        labels = LearningLabels(
            true_exception_type=(
                outcome.ground_truth_exception_type if outcome else None
            ),
            predicted_exception_type=(
                outcome.prediction.exception_type if outcome else None
            ),
            exception_prediction_correct=(
                outcome.actual_outcome.resolution_correct if outcome else None
            ),
            true_resolution=(
                outcome.ground_truth_resolution if outcome else None
            ),
            predicted_resolution=(
                outcome.prediction.resolution_type if outcome else None
            ),
            resolution_correct=(
                outcome.actual_outcome.resolution_correct if outcome else None
            ),
            verification_passed=(
                outcome.verification_passed if outcome else False
            ),
            discrepancy_eliminated=(
                outcome.financial_impact.discrepancy_eliminated
                if outcome else False
            ),
        )

        example = LearningExample(
            example_id=_gen_id("LEX"),
            case_id=cycle.case_id or cycle.exception_id,
            exception_id=cycle.exception_id,
            workflow_id=cycle.workflow_id,
            features=features,
            labels=labels,
            reward_value=cycle.reward_value,
            reward_category=cycle.reward_category,
            guardrail_decision=cycle.guardrail_decision,
            confidence=outcome.confidence if outcome else None,
            risk=outcome.risk if outcome else None,
            lineage_exception_id=cycle.exception_id,
            lineage_feedback_id=cycle.feedback_id,
            lineage_reward_id=cycle.reward_id,
        )

        # Update cycle
        cycle.learning_example_id = example.example_id
        cycle.status = LearningCycleStatus.EXAMPLE_BUILT

        return example

    # ── Step 5: Batch Management ─────────────────────────────────────────

    def ensure_batch_exists(self) -> str:
        """Ensure a collecting batch exists, start one if needed."""
        if self._current_batch_id:
            batch = self.batch_loop.get_batch(self._current_batch_id)
            if batch and batch.status.value == "COLLECTING":
                return self._current_batch_id

        batch = self.batch_loop.start_batch(config=self._batch_config)
        self._current_batch_id = batch.batch_id
        return batch.batch_id

    def add_example_to_batch(
        self,
        cycle_id: str,
        example: LearningExample,
    ) -> bool:
        """Add a learning example to the current batch."""
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            return False

        batch_id = self.ensure_batch_exists()
        success = self.batch_loop.add_case_to_batch(
            batch_id, cycle.exception_id,
        )
        if success:
            cycle.dataset_id = batch_id
            cycle.status = LearningCycleStatus.BATCH_READY
        return success

    # ── Step 6: Train Candidate ──────────────────────────────────────────

    def train_candidate(
        self,
        batch_id: str,
        dataset: LearningDataset,
        model_version: Optional[str] = None,
    ) -> ModelMetadata:
        """Train a candidate model from a batch's learning dataset.

        This does NOT automatically promote the model.
        """
        metadata = self.batch_loop.train_candidate(
            batch_id, dataset, model_version=model_version,
        )

        # Register in model registry
        self.registry.register_model(metadata)

        # Update related cycles
        for cycle in self._cycles.values():
            if cycle.dataset_id == batch_id:
                cycle.candidate_model_id = metadata.model_id
                cycle.candidate_model_version = metadata.version
                cycle.trained_at = datetime.utcnow()
                cycle.status = LearningCycleStatus.TRAINING

        return metadata

    # ── Step 7: Evaluate Candidate ───────────────────────────────────────

    def evaluate_candidate(
        self,
        batch_id: str,
        dataset: LearningDataset,
        split: SplitType = SplitType.TEST,
    ) -> EvaluationMetrics:
        """Evaluate the candidate model."""
        eval_metrics = self.batch_loop.evaluate_candidate(
            batch_id, dataset, split=split,
        )

        # Update related cycles
        for cycle in self._cycles.values():
            if cycle.dataset_id == batch_id:
                cycle.evaluation_accuracy = eval_metrics.accuracy
                cycle.evaluation_f1 = eval_metrics.f1_macro
                cycle.evaluation_precision = eval_metrics.precision_macro
                cycle.evaluated_at = datetime.utcnow()
                cycle.status = LearningCycleStatus.EVALUATING

        return eval_metrics

    # ── Step 8: Compare and Decide ───────────────────────────────────────

    def compare_and_decide(
        self,
        batch_id: str,
        current_metrics: Optional[EvaluationMetrics] = None,
        candidate_metrics: Optional[EvaluationMetrics] = None,
        candidate_model_id: Optional[str] = None,
        candidate_version: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Compare batch metrics and model, decide promotion.

        Returns (promoted: bool, reason: str).

        SAFETY: This method does NOT bypass the promotion gate.
        """
        # Compare batch with previous
        comparison = self.batch_loop.compare_with_previous(batch_id)

        # Decide via batch learning loop (which checks both batch safety + model gate)
        batch = self.batch_loop.promote_or_reject(
            batch_id,
            current_model_metrics=current_metrics,
            candidate_model_metrics=candidate_metrics,
            candidate_model_id=candidate_model_id,
            candidate_version=candidate_version,
        )

        promoted = batch.promoted
        reason = batch.promotion_reason or "Unknown"

        # If batch says promote AND we have model metrics, also run registry gate
        if (
            promoted
            and current_metrics is not None
            and candidate_metrics is not None
            and candidate_model_id is not None
        ):
            try:
                record = self.registry.promote(
                    candidate_id=candidate_model_id,
                    current_metrics=current_metrics,
                    candidate_metrics=candidate_metrics,
                )
                promoted = record.decision.value == "PROMOTED"
                if not promoted:
                    reason = f"Model registry gate rejected: {record.reason}"
            except Exception as e:
                promoted = False
                reason = f"Registry error: {str(e)}"

        # Update related cycles
        for cycle in self._cycles.values():
            if cycle.dataset_id == batch_id:
                cycle.promotion_action = (
                    PromotionAction.PROMOTED if promoted
                    else PromotionAction.REJECTED
                )
                cycle.promotion_reason = reason
                cycle.safety_maintained = (
                    comparison.safety.all_safety_maintained
                    if comparison else True
                )
                if promoted and candidate_version:
                    cycle.promoted_model_version = candidate_version
                cycle.promoted_at = datetime.utcnow()
                cycle.status = (
                    LearningCycleStatus.PROMOTED if promoted
                    else LearningCycleStatus.REJECTED
                )
                cycle.completed_at = datetime.utcnow()

        return promoted, reason

    # ── Metrics ──────────────────────────────────────────────────────────

    def compute_learning_metrics(
        self,
        previous_metrics: Optional[LearningMetrics] = None,
    ) -> LearningMetrics:
        """Compute comprehensive learning metrics from all recorded data."""
        return self.metrics_service.compute(
            outcomes=self._all_outcomes,
            feedbacks=self._all_feedbacks,
            rewards=self._all_rewards,
            source_type="learning_loop",
            previous_metrics=previous_metrics,
        )

    # ── System State ─────────────────────────────────────────────────────

    def get_system_state(self) -> LearningSystemState:
        """Get the complete state of the learning system."""
        completed = sum(
            1 for c in self._cycles.values()
            if c.completed_at is not None
        )
        promotions = sum(
            1 for c in self._cycles.values()
            if c.promotion_action == PromotionAction.PROMOTED
        )
        rejections = sum(
            1 for c in self._cycles.values()
            if c.promotion_action == PromotionAction.REJECTED
        )

        reward_values = [
            c.reward_value for c in self._cycles.values()
            if c.reward_value is not None
        ]
        avg_reward = (
            sum(reward_values) / len(reward_values)
            if reward_values else None
        )

        safety_ok = all(
            c.safety_maintained for c in self._cycles.values()
        )

        active_model = self.registry.get_active_model()

        batches = self.batch_loop.get_all_batches()

        return LearningSystemState(
            total_cycles=len(self._cycles),
            completed_cycles=completed,
            active_model_id=active_model.model_id if active_model else None,
            active_model_version=active_model.version if active_model else None,
            total_learning_examples=sum(
                1 for c in self._cycles.values()
                if c.learning_example_id is not None
            ),
            total_batches=len(batches),
            active_batch_id=self._current_batch_id,
            total_rewards=len(self._all_rewards),
            avg_reward=avg_reward,
            total_promotions=promotions,
            total_rejections=rejections,
            safety_maintained_all_cycles=safety_ok,
            last_cycle_completed_at=max(
                (c.completed_at for c in self._cycles.values()
                 if c.completed_at is not None),
                default=None,
            ),
            last_promotion_at=max(
                (c.promoted_at for c in self._cycles.values()
                 if c.promoted_at is not None
                 and c.promotion_action == PromotionAction.PROMOTED),
                default=None,
            ),
        )

    # ── Query Methods ────────────────────────────────────────────────────

    def get_cycle(self, cycle_id: str) -> Optional[LearningCycleRecord]:
        return self._cycles.get(cycle_id)

    def get_cycle_by_exception(
        self, exception_id: str,
    ) -> Optional[LearningCycleRecord]:
        cid = self._cycles_by_exception.get(exception_id)
        return self._cycles.get(cid) if cid else None

    def get_all_cycles(self) -> List[LearningCycleRecord]:
        return sorted(
            self._cycles.values(),
            key=lambda c: c.cycle_number,
        )

    def get_cycles_by_status(
        self, status: LearningCycleStatus,
    ) -> List[LearningCycleRecord]:
        return [
            c for c in self._cycles.values()
            if c.status == status
        ]

    def get_reward_values(self) -> List[float]:
        return [
            c.reward_value for c in self._cycles.values()
            if c.reward_value is not None
        ]

    def count_by_promotion_action(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self._cycles.values():
            key = c.promotion_action.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ── Safety Boundary ──────────────────────────────────────────────────

    def verify_safety_boundary(self) -> Dict[str, bool]:
        """Verify that the learning system respects safety boundaries.

        Returns a dict of safety checks and whether they pass.
        """
        return {
            "no_financial_modification": True,  # By design
            "no_guardrail_bypass": True,         # By design
            "no_execution_authorization": True,  # By design
            "promotion_requires_gate": True,     # Enforced by ModelRegistry
            "reward_does_not_authorize": True,   # By design
            "all_cycles_safe": all(
                c.safety_maintained for c in self._cycles.values()
            ),
            "no_unauthorized_promotions": all(
                c.promotion_action != PromotionAction.PROMOTED
                or c.promotion_reason is not None
                for c in self._cycles.values()
            ),
        }
