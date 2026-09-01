"""
Reward Engine service for Razorpay CloseLoop Phase 9B.

Transparent, deterministic reward calculation from outcome + feedback.

Safety principle:
  Reward is an evaluation signal ONLY.
  It does NOT affect the current case's financial decision.
  Phase 6 remains the final safety authority.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from app.schemas.feedback import (
    ActualOutcomeRecord,
    FeedbackRecord,
    FeedbackType,
    OutcomeRecord,
)
from app.schemas.reward_engine import (
    FinancialRiskLevel,
    RewardBreakdown,
    RewardCategory,
    RewardComponent,
    RewardConfig,
    RewardRecord,
    RewardWeights,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Financial Risk Classification
# ─────────────────────────────────────────────────────────────────────────────


def classify_financial_risk(
    adjustment_paise: int,
    high_value_threshold: int = 10_000_000,
) -> FinancialRiskLevel:
    """Classify financial risk from adjustment amount.

    Thresholds (configurable via high_value_threshold):
      < ₹100          → NEGLIGIBLE
      ₹100 – ₹1,000   → LOW
      ₹1,000 – ₹10,000 → MEDIUM
      ₹10,000 – ₹1,00,000 → HIGH
      > ₹1,00,000     → CRITICAL
    """
    abs_amount = abs(adjustment_paise)
    if abs_amount < 10_000:       # < ₹100
        return FinancialRiskLevel.NEGLIGIBLE
    elif abs_amount < 1_00_000:    # ₹100 – ₹1,000
        return FinancialRiskLevel.LOW
    elif abs_amount < 10_00_000:   # ₹1,000 – ₹10,000
        return FinancialRiskLevel.MEDIUM
    elif abs_amount <= high_value_threshold:  # ₹10,000 – threshold
        return FinancialRiskLevel.HIGH
    else:
        return FinancialRiskLevel.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# Reward Category Determination
# ─────────────────────────────────────────────────────────────────────────────


def determine_reward_category(
    outcome: OutcomeRecord,
    feedback: Optional[FeedbackRecord] = None,
    high_value_threshold: int = 10_000_000,
) -> RewardCategory:
    """Determine the reward category from outcome and feedback.

    Decision tree (order matters — first match wins):

    1. Verification failed → VERIFICATION_FAILURE
    2. Auto-resolved + incorrect + high value → HIGH_VALUE_ERROR
    3. Auto-resolved + incorrect → INCORRECT_AUTO_RESOLUTION
    4. Auto-resolved + correct → CORRECT_AUTO_RESOLUTION
    5. Escalated + correct escalation → CORRECT_ESCALATION
    6. Escalated + unnecessary → UNNECESSARY_ESCALATION
    7. Human approved → HUMAN_CONFIRMED
    8. Human corrected → INCORRECT_AUTO_RESOLUTION (system was wrong)
    9. Human rejected → INCORRECT_AUTO_RESOLUTION (system was wrong)
    10. Default → UNNECESSARY_ESCALATION
    """
    actual = outcome.actual_outcome
    prediction_correct = actual.resolution_correct
    was_executed = actual.was_executed
    was_verified = outcome.verification_passed
    adjustment = abs(outcome.financial_impact.actual_adjustment_paise)
    is_high_value = adjustment >= high_value_threshold

    # 1. Verification failed
    if was_executed and not was_verified and actual.was_rolled_back:
        return RewardCategory.VERIFICATION_FAILURE

    # 2. Auto-resolved + incorrect + high value → HIGH_VALUE_ERROR
    if was_executed and prediction_correct is False and is_high_value:
        return RewardCategory.HIGH_VALUE_ERROR

    # 3. Auto-resolved + incorrect → INCORRECT_AUTO_RESOLUTION
    if was_executed and prediction_correct is False:
        return RewardCategory.INCORRECT_AUTO_RESOLUTION

    # 4. Auto-resolved + correct → CORRECT_AUTO_RESOLUTION
    if was_executed and prediction_correct is True:
        return RewardCategory.CORRECT_AUTO_RESOLUTION

    # 5/6. Escalated
    if outcome.decision == "UNRESOLVED" or (
        outcome.human_feedback_type == FeedbackType.ESCALATE
    ):
        if prediction_correct is True or (
            feedback and feedback.feedback_type == FeedbackType.ESCALATE
            and outcome.human_feedback_type == FeedbackType.ESCALATE
        ):
            return RewardCategory.CORRECT_ESCALATION
        return RewardCategory.UNNECESSARY_ESCALATION

    # 7. Human approved (after human review path)
    if feedback and feedback.feedback_type == FeedbackType.APPROVE:
        return RewardCategory.HUMAN_CONFIRMED

    # 8/9. Human corrected/rejected → system was wrong
    if feedback and feedback.feedback_type in (
        FeedbackType.CORRECT,
        FeedbackType.REJECT,
    ):
        if prediction_correct is False:
            return RewardCategory.INCORRECT_AUTO_RESOLUTION
        # Even if technically "correct", human disagreed
        return RewardCategory.INCORRECT_AUTO_RESOLUTION

    # 10. Default
    return RewardCategory.UNNECESSARY_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Reward Calculation
# ─────────────────────────────────────────────────────────────────────────────


class RewardEngine:
    """Transparent, deterministic reward calculation engine.

    Every reward is fully explainable through its breakdown.
    Same inputs → same reward (deterministic).
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        self._rewards: Dict[str, RewardRecord] = {}

    def calculate_reward(
        self,
        outcome: OutcomeRecord,
        feedback: Optional[FeedbackRecord] = None,
    ) -> RewardRecord:
        """Calculate a reward from an outcome and optional feedback.

        Args:
            outcome: The completed outcome record.
            feedback: Optional human feedback.

        Returns:
            Complete RewardRecord with transparent breakdown.
        """
        weights = self.config.weights
        high_value_threshold = self.config.high_value_threshold_paise

        # 1. Determine category
        category = determine_reward_category(
            outcome, feedback, high_value_threshold
        )

        # 2. Financial risk level
        adjustment = abs(outcome.financial_impact.actual_adjustment_paise)
        risk_level = classify_financial_risk(adjustment, high_value_threshold)

        # 3. Calculate each component
        base = self._calc_base(category, weights)
        verification = self._calc_verification(outcome, weights)
        fin_risk = self._calc_financial_risk(risk_level, category, weights)
        human = self._calc_human_feedback(feedback, category, weights)
        confidence = self._calc_confidence(outcome, category, weights)
        discrepancy = self._calc_discrepancy(outcome, weights)
        unintended = self._calc_unintended(outcome, weights)

        breakdown = RewardBreakdown(
            base_reward=base,
            verification_component=verification,
            financial_risk_component=fin_risk,
            human_feedback_component=human,
            confidence_component=confidence,
            discrepancy_component=discrepancy,
            unintended_changes_component=unintended,
        )

        # 4. Final reward (clamped to [-1, 1])
        reward_value = breakdown.total()

        # 5. Build reason
        reason = self._build_reason(category, breakdown, outcome, feedback)

        # 6. Build record
        reward_id = _gen_id("REW")
        record = RewardRecord(
            reward_id=reward_id,
            workflow_id=outcome.workflow_id,
            exception_id=outcome.exception_id,
            case_id=outcome.case_id,
            category=category,
            reward_value=reward_value,
            reward_reason=reason,
            breakdown=breakdown,
            resolution_correct=outcome.actual_outcome.resolution_correct,
            was_auto_resolved=outcome.actual_outcome.was_executed and outcome.decision == "AUTO",
            was_human_approved=feedback.feedback_type == FeedbackType.APPROVE if feedback else False,
            was_human_rejected=feedback.feedback_type == FeedbackType.REJECT if feedback else False,
            was_human_corrected=feedback.feedback_type == FeedbackType.CORRECT if feedback else False,
            was_escalated=outcome.decision in ("UNRESOLVED", None) and not outcome.actual_outcome.was_executed,
            escalation_was_correct=None,
            verification_passed=outcome.verification_passed,
            financial_impact_paise=outcome.financial_impact.actual_adjustment_paise,
            financial_risk_level=risk_level,
            discrepancy_eliminated=outcome.financial_impact.discrepancy_eliminated,
            unintended_changes=outcome.financial_impact.unintended_changes,
            confidence=outcome.confidence,
            ground_truth_exception_type=outcome.ground_truth_exception_type,
            ground_truth_resolution=outcome.ground_truth_resolution,
            policy_version=self.config.policy_version,
            model_version=outcome.prediction.model_version,
            calculated_at=datetime.utcnow(),
        )

        self._rewards[reward_id] = record
        return record

    def get_reward(self, reward_id: str) -> Optional[RewardRecord]:
        return self._rewards.get(reward_id)

    def get_rewards_for_workflow(self, workflow_id: str) -> List[RewardRecord]:
        return [r for r in self._rewards.values() if r.workflow_id == workflow_id]

    def get_rewards_for_exception(self, exception_id: str) -> List[RewardRecord]:
        return [r for r in self._rewards.values() if r.exception_id == exception_id]

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._rewards.values():
            key = r.category.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def average_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return sum(r.reward_value for r in self._rewards.values()) / len(self._rewards)

    # ── Component Calculations ────────────────────────────────────────────

    def _calc_base(
        self, category: RewardCategory, weights: RewardWeights
    ) -> RewardComponent:
        value = weights.base_rewards.get(category.value, 0.0)
        return RewardComponent(
            component_name="base_reward",
            value=value,
            reason=f"Base reward for {category.value}",
            weight_used=value,
        )

    def _calc_verification(
        self, outcome: OutcomeRecord, weights: RewardWeights
    ) -> RewardComponent:
        if outcome.actual_outcome.was_executed:
            if outcome.verification_passed:
                value = weights.verification_passed_bonus
                reason = "Verification passed — bonus applied"
            elif outcome.actual_outcome.was_rolled_back:
                value = weights.verification_failed_penalty
                reason = "Verification failed and rollback occurred — penalty applied"
            else:
                value = 0.0
                reason = "Verification not applicable (not yet executed or pending)"
        else:
            value = 0.0
            reason = "Verification not applicable (resolution not executed)"
        return RewardComponent(
            component_name="verification",
            value=value,
            reason=reason,
            weight_used=weights.verification_passed_bonus if value > 0 else (
                weights.verification_failed_penalty if value < 0 else None
            ),
        )

    def _calc_financial_risk(
        self,
        risk_level: FinancialRiskLevel,
        category: RewardCategory,
        weights: RewardWeights,
    ) -> RewardComponent:
        penalty = weights.financial_risk_penalties.get(risk_level.value, 0.0)
        # Extra penalty for high-risk categories + high financial exposure
        if category in (
            RewardCategory.INCORRECT_AUTO_RESOLUTION,
            RewardCategory.HIGH_VALUE_ERROR,
        ) and risk_level in (FinancialRiskLevel.HIGH, FinancialRiskLevel.CRITICAL):
            penalty *= 1.5  # Amplify penalty for dangerous combos

        if penalty == 0.0:
            reason = "Financial risk negligible — no penalty"
        else:
            reason = (
                f"Financial risk level {risk_level.value} — "
                f"penalty of {penalty:.3f} applied"
            )

        return RewardComponent(
            component_name="financial_risk",
            value=penalty,
            reason=reason,
            weight_used=penalty,
        )

    def _calc_human_feedback(
        self,
        feedback: Optional[FeedbackRecord],
        category: RewardCategory,
        weights: RewardWeights,
    ) -> RewardComponent:
        if feedback is None:
            return RewardComponent(
                component_name="human_feedback",
                value=0.0,
                reason="No human feedback received",
            )

        if feedback.feedback_type == FeedbackType.APPROVE:
            value = weights.human_approve_bonus
            reason = "Human approved — bonus applied"
        elif feedback.feedback_type == FeedbackType.REJECT:
            value = weights.human_reject_penalty
            reason = "Human rejected — penalty applied"
        elif feedback.feedback_type == FeedbackType.CORRECT:
            value = weights.human_correct_penalty
            reason = "Human corrected prediction — penalty applied"
        elif feedback.feedback_type == FeedbackType.ESCALATE:
            value = 0.0
            reason = "Human escalated — neutral (escalation is appropriate response)"
        else:
            value = 0.0
            reason = f"Unknown feedback type: {feedback.feedback_type}"

        return RewardComponent(
            component_name="human_feedback",
            value=value,
            reason=reason,
            weight_used=value,
        )

    def _calc_confidence(
        self,
        outcome: OutcomeRecord,
        category: RewardCategory,
        weights: RewardWeights,
    ) -> RewardComponent:
        confidence = outcome.confidence
        if confidence is None:
            return RewardComponent(
                component_name="confidence",
                value=0.0,
                reason="No confidence available",
            )

        # High confidence + correct → small bonus
        # High confidence + incorrect → extra penalty (overconfident)
        # Low confidence + incorrect → less penalty (hedged bets)
        scale = weights.confidence_bonus_scale
        if category in (
            RewardCategory.CORRECT_AUTO_RESOLUTION,
            RewardCategory.HUMAN_CONFIRMED,
            RewardCategory.CORRECT_ESCALATION,
        ):
            value = confidence * scale
            reason = f"Confidence {confidence:.2f} on correct outcome — bonus scaled by confidence"
        elif category in (
            RewardCategory.INCORRECT_AUTO_RESOLUTION,
            RewardCategory.HIGH_VALUE_ERROR,
        ):
            # Overconfidence penalty: higher confidence when wrong → worse
            value = -(confidence * scale * 1.5)
            reason = f"Confidence {confidence:.2f} on incorrect outcome — overconfidence penalty"
        else:
            value = 0.0
            reason = "Confidence not relevant for this category"

        return RewardComponent(
            component_name="confidence",
            value=value,
            reason=reason,
            weight_used=scale,
        )

    def _calc_discrepancy(
        self, outcome: OutcomeRecord, weights: RewardWeights
    ) -> RewardComponent:
        impact = outcome.financial_impact
        if impact.actual_adjustment_paise == 0 and impact.requested_adjustment_paise == 0:
            return RewardComponent(
                component_name="discrepancy",
                value=0.0,
                reason="No financial adjustment involved",
            )

        if impact.discrepancy_eliminated:
            value = weights.discrepancy_eliminated_bonus
            reason = "Discrepancy fully eliminated — bonus applied"
        elif impact.difference_after_paise != 0:
            value = weights.discrepancy_remainder_penalty
            reason = (
                f"Discrepancy remains: {impact.difference_after_paise} paise — "
                f"penalty applied"
            )
        else:
            value = 0.0
            reason = "Discrepancy resolution status unclear"

        return RewardComponent(
            component_name="discrepancy",
            value=value,
            reason=reason,
            weight_used=value,
        )

    def _calc_unintended(
        self, outcome: OutcomeRecord, weights: RewardWeights
    ) -> RewardComponent:
        count = outcome.financial_impact.unintended_changes
        if count == 0:
            return RewardComponent(
                component_name="unintended_changes",
                value=0.0,
                reason="No unintended financial changes",
            )

        value = count * weights.unintended_change_penalty
        reason = (
            f"{count} unintended financial change(s) detected — "
            f"penalty of {value:.3f} applied"
        )
        return RewardComponent(
            component_name="unintended_changes",
            value=value,
            reason=reason,
            weight_used=weights.unintended_change_penalty,
        )

    def _build_reason(
        self,
        category: RewardCategory,
        breakdown: RewardBreakdown,
        outcome: OutcomeRecord,
        feedback: Optional[FeedbackRecord],
    ) -> str:
        """Build a human-readable reward reason."""
        parts = [f"Category: {category.value}"]

        # Add key signals
        if outcome.actual_outcome.resolution_correct is not None:
            parts.append(
                f"Prediction {'correct' if outcome.actual_outcome.resolution_correct else 'incorrect'}"
            )
        if outcome.verification_passed:
            parts.append("Verification passed")
        elif outcome.actual_outcome.was_rolled_back:
            parts.append("Verification failed (rolled back)")

        if feedback:
            parts.append(f"Human feedback: {feedback.feedback_type.value}")

        # Financial impact
        adj = outcome.financial_impact.actual_adjustment_paise
        if adj > 0:
            parts.append(f"Financial impact: ₹{adj // 100}.{adj % 100:02d}")

        return " | ".join(parts)
