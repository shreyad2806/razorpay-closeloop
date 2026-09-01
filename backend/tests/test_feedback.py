"""
Tests for Phase 9A — Human Feedback + Outcome Recording.

Tests cover:
- Feedback record creation (APPROVE, REJECT, CORRECT, ESCALATE)
- Outcome record creation with separated prediction/actual/feedback
- Data lineage traceability
- Service operations
- Edge cases
"""

import pytest

from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    EscalationDetail,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    OutcomeStatus,
    PredictionRecord,
    RejectionDetail,
)
from app.services.feedback import FeedbackService, OutcomeService


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
    impact: int = 3000,
) -> ActualOutcomeRecord:
    return ActualOutcomeRecord(
        actual_resolution=resolution,
        actual_exception_type="FEE_DIFFERENCE",
        resolution_correct=correct,
        financial_impact_paise=impact,
        was_executed=executed,
        was_verified=verified,
        was_rolled_back=False,
    )


def _make_lineage(
    exception_id: str = "EXC-001",
    evidence_ids=None,
) -> DataLineage:
    return DataLineage(
        exception_id=exception_id,
        evidence_ids=evidence_ids or ["EVD-001"],
        prediction_id="PRED-001",
        decision_id="DEC-001",
        execution_id="EXEC-001",
        verification_id="VER-001",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeedbackSchemas:
    """Test feedback schema creation and validation."""

    def test_approve_feedback(self):
        """APPROVE feedback — no extra detail needed."""
        fb = FeedbackRecord(
            feedback_id="FB-001",
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="ops_user_1",
            system_prediction="FEE_ADJUSTMENT",
            system_confidence=0.9,
            financial_adjustment_paise=3000,
        )
        assert fb.feedback_type == FeedbackType.APPROVE
        assert fb.is_approval()
        assert not fb.is_correction()
        assert not fb.is_rejection()
        assert not fb.is_escalation()
        assert fb.correction is None
        assert fb.rejection is None
        assert fb.escalation is None

    def test_reject_feedback(self):
        """REJECT feedback with reason."""
        fb = FeedbackRecord(
            feedback_id="FB-002",
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.REJECT,
            reviewer="ops_user_2",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(
                rejection_reason="Fee correction amount is incorrect",
                suggested_alternative="REFUND_ADJUSTMENT",
                risk_concern="Amount exceeds evidence",
            ),
        )
        assert fb.is_rejection()
        assert fb.rejection.rejection_reason == "Fee correction amount is incorrect"
        assert fb.rejection.suggested_alternative == "REFUND_ADJUSTMENT"

    def test_correct_feedback(self):
        """CORRECT feedback with correction details."""
        fb = FeedbackRecord(
            feedback_id="FB-003",
            workflow_id="WF-002",
            exception_id="EXC-002",
            feedback_type=FeedbackType.CORRECT,
            reviewer="auditor_1",
            system_prediction="FEE_ADJUSTMENT",
            system_confidence=0.75,
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_ADJUSTMENT",
                correction_reason="The discrepancy is due to a missing refund, not a fee",
                original_confidence=0.75,
                corrected_amount_paise=5000,
            ),
        )
        assert fb.is_correction()
        assert fb.correction.corrected_resolution == "REFUND_ADJUSTMENT"
        assert fb.correction.corrected_amount_paise == 5000

    def test_escalate_feedback(self):
        """ESCALATE feedback with escalation details."""
        fb = FeedbackRecord(
            feedback_id="FB-004",
            workflow_id="WF-003",
            exception_id="EXC-003",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="ops_user_1",
            system_prediction="UNKNOWN_UNRESOLVED",
            escalation=EscalationDetail(
                escalation_reason="Complex multi-adjustment case needs senior review",
                escalation_target="finance_team_lead",
                additional_context="Multiple conflicting settlements detected",
            ),
        )
        assert fb.is_escalation()
        assert fb.escalation.escalation_target == "finance_team_lead"

    def test_correction_of_previous(self):
        """Feedback correcting a previous feedback record."""
        fb = FeedbackRecord(
            feedback_id="FB-005",
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.CORRECT,
            reviewer="senior_auditor",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="TAX_ADJUSTMENT",
                correction_reason="Actually a tax miscalculation",
            ),
            correction_of="FB-003",
        )
        assert fb.correction_of == "FB-003"

    def test_feedback_summary(self):
        """Feedback summary output."""
        fb = FeedbackRecord(
            feedback_id="FB-006",
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.CORRECT,
            reviewer="auditor",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_ADJUSTMENT",
                correction_reason="Wrong type",
            ),
        )
        summary = fb.summary()
        assert "CORRECT" in summary
        assert "REFUND_ADJUSTMENT" in summary

    def test_feedback_timestamp(self):
        """Feedback auto-generates timestamp."""
        fb = FeedbackRecord(
            feedback_id="FB-007",
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="user",
            system_prediction="NO_ACTION",
        )
        assert fb.created_at is not None


class TestOutcomeSchemas:
    """Test outcome schema creation and validation."""

    def test_outcome_record_creation(self):
        """Full outcome record with all components."""
        outcome = OutcomeRecord(
            outcome_id="OUT-001",
            workflow_id="WF-001",
            exception_id="EXC-001",
            case_id="CASE-001",
            candidate_id="CAND-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
            verification_passed=True,
            decision="AUTO",
            confidence=0.92,
            risk="LOW",
        )
        assert outcome.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert outcome.actual_outcome.actual_resolution == "FEE_ADJUSTMENT"
        assert outcome.verification_passed is True
        assert outcome.status == OutcomeStatus.RECORDED

    def test_outcome_with_feedback(self):
        """Outcome that has received human feedback."""
        outcome = OutcomeRecord(
            outcome_id="OUT-002",
            workflow_id="WF-002",
            exception_id="EXC-002",
            prediction=_make_prediction(resolution_type="REFUND_ADJUSTMENT"),
            actual_outcome=_make_actual(resolution="REFUND_ADJUSTMENT", correct=True),
            lineage=_make_lineage(exception_id="EXC-002"),
            human_feedback_id="FB-001",
            human_feedback_type=FeedbackType.APPROVE,
            human_override=False,
        )
        assert outcome.status == OutcomeStatus.FEEDBACK_RECEIVED
        assert outcome.human_feedback_type == FeedbackType.APPROVE
        assert outcome.human_override is False

    def test_outcome_with_human_override(self):
        """Outcome where human overrode system prediction."""
        outcome = OutcomeRecord(
            outcome_id="OUT-003",
            workflow_id="WF-003",
            exception_id="EXC-003",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT", confidence=0.7),
            actual_outcome=_make_actual(
                resolution="TAX_ADJUSTMENT", correct=False, impact=8000
            ),
            lineage=_make_lineage(exception_id="EXC-003"),
            human_feedback_id="FB-002",
            human_feedback_type=FeedbackType.CORRECT,
            human_override=True,
        )
        assert outcome.human_override is True
        assert outcome.prediction_matches_actual() is False

    def test_prediction_matches_actual(self):
        """Check prediction vs actual comparison."""
        outcome = OutcomeRecord(
            outcome_id="OUT-004",
            workflow_id="WF-004",
            exception_id="EXC-004",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual(resolution="FEE_ADJUSTMENT", correct=True),
            lineage=_make_lineage(exception_id="EXC-004"),
        )
        assert outcome.prediction_matches_actual() is True

    def test_prediction_mismatches_actual(self):
        """Prediction does not match actual."""
        outcome = OutcomeRecord(
            outcome_id="OUT-005",
            workflow_id="WF-005",
            exception_id="EXC-005",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual(resolution="REFUND_ADJUSTMENT", correct=False),
            lineage=_make_lineage(exception_id="EXC-005"),
        )
        assert outcome.prediction_matches_actual() is False

    def test_prediction_no_actual(self):
        """Prediction exists but no actual resolution yet."""
        outcome = OutcomeRecord(
            outcome_id="OUT-006",
            workflow_id="WF-006",
            exception_id="EXC-006",
            prediction=_make_prediction(),
            actual_outcome=ActualOutcomeRecord(
                was_executed=False,
                was_verified=False,
            ),
            lineage=_make_lineage(exception_id="EXC-006"),
        )
        assert outcome.prediction_matches_actual() is None

    def test_learning_ready(self):
        """Outcome is learning-ready when prediction + actual exist."""
        outcome = OutcomeRecord(
            outcome_id="OUT-007",
            workflow_id="WF-007",
            exception_id="EXC-007",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(exception_id="EXC-007"),
        )
        assert outcome.is_learning_ready() is True

    def test_not_learning_ready_without_outcome(self):
        """Outcome not ready when actual resolution is missing."""
        outcome = OutcomeRecord(
            outcome_id="OUT-008",
            workflow_id="WF-008",
            exception_id="EXC-008",
            prediction=_make_prediction(),
            actual_outcome=ActualOutcomeRecord(was_executed=False),
            lineage=_make_lineage(exception_id="EXC-008"),
        )
        assert outcome.is_learning_ready() is False

    def test_financial_impact_separation(self):
        """Financial impact is separate from prediction/outcome."""
        impact = FinancialImpact(
            requested_adjustment_paise=5000,
            actual_adjustment_paise=5000,
            difference_before_paise=5000,
            difference_after_paise=0,
            discrepancy_eliminated=True,
            unintended_changes=0,
        )
        outcome = OutcomeRecord(
            outcome_id="OUT-009",
            workflow_id="WF-009",
            exception_id="EXC-009",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(impact=5000),
            financial_impact=impact,
            lineage=_make_lineage(exception_id="EXC-009"),
        )
        assert outcome.financial_impact.discrepancy_eliminated is True
        assert outcome.financial_impact.actual_adjustment_paise == 5000

    def test_outcome_summary(self):
        """Outcome summary output."""
        outcome = OutcomeRecord(
            outcome_id="OUT-010",
            workflow_id="WF-010",
            exception_id="EXC-010",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(exception_id="EXC-010"),
        )
        summary = outcome.summary()
        assert "OUT-010" in summary
        assert "FEE_ADJUSTMENT" in summary

    def test_ground_truth_isolation(self):
        """Ground truth fields exist but are clearly evaluation-only."""
        outcome = OutcomeRecord(
            outcome_id="OUT-011",
            workflow_id="WF-011",
            exception_id="EXC-011",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(correct=False),
            lineage=_make_lineage(exception_id="EXC-011"),
            ground_truth_exception_type="FEE_DIFFERENCE",
            ground_truth_resolution="FEE_ADJUSTMENT",
            ground_truth_resolvable=True,
        )
        assert outcome.ground_truth_exception_type == "FEE_DIFFERENCE"
        assert outcome.ground_truth_resolution == "FEE_ADJUSTMENT"
        assert outcome.ground_truth_resolvable is True
        # Ground truth exists but actual_outcome says different
        assert outcome.actual_outcome.resolution_correct is False


# ─────────────────────────────────────────────────────────────────────────────
# FeedbackService Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeedbackService:
    """Test FeedbackService operations."""

    def test_record_approve(self):
        """Record APPROVE feedback."""
        svc = FeedbackService()
        fb = svc.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="user1",
            system_prediction="FEE_ADJUSTMENT",
        )
        assert fb.feedback_id.startswith("FB-")
        assert fb.feedback_type == FeedbackType.APPROVE
        assert fb.reviewer == "user1"

    def test_record_reject(self):
        """Record REJECT feedback."""
        svc = FeedbackService()
        fb = svc.record_feedback(
            workflow_id="WF-002",
            exception_id="EXC-002",
            feedback_type=FeedbackType.REJECT,
            reviewer="user2",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(
                rejection_reason="Amount is wrong",
                risk_concern="Exceeds evidence",
            ),
        )
        assert fb.is_rejection()
        assert svc.get_feedback(fb.feedback_id) is not None

    def test_record_correct(self):
        """Record CORRECT feedback."""
        svc = FeedbackService()
        fb = svc.record_feedback(
            workflow_id="WF-003",
            exception_id="EXC-003",
            feedback_type=FeedbackType.CORRECT,
            reviewer="auditor1",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="REFUND_ADJUSTMENT",
                correction_reason="Wrong type",
            ),
        )
        assert fb.is_correction()
        corrections = svc.get_corrections()
        assert len(corrections) == 1

    def test_record_escalate(self):
        """Record ESCALATE feedback."""
        svc = FeedbackService()
        fb = svc.record_feedback(
            workflow_id="WF-004",
            exception_id="EXC-004",
            feedback_type=FeedbackType.ESCALATE,
            reviewer="user1",
            system_prediction="UNKNOWN_UNRESOLVED",
            escalation=EscalationDetail(
                escalation_reason="Complex case",
                escalation_target="senior_team",
            ),
        )
        assert fb.is_escalation()

    def test_get_feedback_for_workflow(self):
        """Retrieve all feedback for a workflow."""
        svc = FeedbackService()
        svc.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="u1",
            system_prediction="FEE_ADJUSTMENT",
        )
        svc.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.CORRECT,
            reviewer="u2",
            system_prediction="FEE_ADJUSTMENT",
            correction=CorrectionDetail(
                original_resolution="FEE_ADJUSTMENT",
                corrected_resolution="TAX_ADJUSTMENT",
                correction_reason="Wrong",
            ),
        )
        feedbacks = svc.get_feedback_for_workflow("WF-001")
        assert len(feedbacks) == 2

    def test_get_feedback_for_exception(self):
        """Retrieve feedback by exception ID."""
        svc = FeedbackService()
        svc.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="u1",
            system_prediction="FEE_ADJUSTMENT",
        )
        svc.record_feedback(
            workflow_id="WF-002",
            exception_id="EXC-001",
            feedback_type=FeedbackType.REJECT,
            reviewer="u2",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Wrong"),
        )
        feedbacks = svc.get_feedback_for_exception("EXC-001")
        assert len(feedbacks) == 2

    def test_count_by_type(self):
        """Count feedback by type."""
        svc = FeedbackService()
        for _ in range(3):
            svc.record_feedback(
                workflow_id="WF-1",
                exception_id="EXC-1",
                feedback_type=FeedbackType.APPROVE,
                reviewer="u",
                system_prediction="FEE_ADJUSTMENT",
            )
        svc.record_feedback(
            workflow_id="WF-2",
            exception_id="EXC-2",
            feedback_type=FeedbackType.REJECT,
            reviewer="u",
            system_prediction="FEE_ADJUSTMENT",
            rejection=RejectionDetail(rejection_reason="Wrong"),
        )
        counts = svc.count_by_type()
        assert counts["APPROVE"] == 3
        assert counts["REJECT"] == 1

    def test_has_feedback(self):
        """Check if workflow has feedback."""
        svc = FeedbackService()
        assert svc.has_feedback("WF-001") is False
        svc.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type=FeedbackType.APPROVE,
            reviewer="u",
            system_prediction="FEE_ADJUSTMENT",
        )
        assert svc.has_feedback("WF-001") is True

    def test_missing_feedback(self):
        """Get non-existent feedback returns None."""
        svc = FeedbackService()
        assert svc.get_feedback("FB-NONEXISTENT") is None

    def test_invalid_feedback_type(self):
        """Invalid feedback type raises error."""
        svc = FeedbackService()
        with pytest.raises(Exception):
            svc.record_feedback(
                workflow_id="WF-001",
                exception_id="EXC-001",
                feedback_type="INVALID",
                reviewer="u",
                system_prediction="FEE_ADJUSTMENT",
            )


# ─────────────────────────────────────────────────────────────────────────────
# OutcomeService Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOutcomeService:
    """Test OutcomeService operations."""

    def test_record_outcome(self):
        """Record a complete outcome."""
        svc = OutcomeService()
        outcome = svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
            decision="AUTO",
            confidence=0.92,
            risk="LOW",
        )
        assert outcome.outcome_id.startswith("OUT-")
        assert outcome.prediction.resolution_type == "FEE_ADJUSTMENT"
        assert outcome.status == OutcomeStatus.RECORDED

    def test_get_outcome_for_workflow(self):
        """Retrieve outcome by workflow ID."""
        svc = OutcomeService()
        outcome = svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        found = svc.get_outcome_for_workflow("WF-001")
        assert found is not None
        assert found.outcome_id == outcome.outcome_id

    def test_get_outcome_for_exception(self):
        """Retrieve outcome by exception ID."""
        svc = OutcomeService()
        outcome = svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        found = svc.get_outcome_for_exception("EXC-001")
        assert found is not None

    def test_update_feedback(self):
        """Update outcome with received feedback."""
        svc = OutcomeService()
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        updated = svc.update_feedback(
            workflow_id="WF-001",
            feedback_id="FB-001",
            feedback_type=FeedbackType.APPROVE,
        )
        assert updated is not None
        assert updated.status == OutcomeStatus.FEEDBACK_RECEIVED
        assert updated.human_feedback_id == "FB-001"

    def test_update_feedback_nonexistent(self):
        """Update feedback for non-existent workflow returns None."""
        svc = OutcomeService()
        result = svc.update_feedback(
            workflow_id="WF-NONE",
            feedback_id="FB-001",
            feedback_type=FeedbackType.APPROVE,
        )
        assert result is None

    def test_mark_reward_calculated(self):
        """Mark outcome as reward calculated."""
        svc = OutcomeService()
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        updated = svc.mark_reward_calculated("WF-001")
        assert updated.status == OutcomeStatus.REWARD_CALCULATED

    def test_mark_stored_for_learning(self):
        """Mark outcome as stored for learning."""
        svc = OutcomeService()
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        updated = svc.mark_stored_for_learning("WF-001")
        assert updated.status == OutcomeStatus.STORED_FOR_LEARNING

    def test_learning_ready_outcomes(self):
        """Get outcomes ready for learning."""
        svc = OutcomeService()
        # Ready
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        # Not ready (no actual resolution)
        svc.record_outcome(
            workflow_id="WF-002",
            exception_id="EXC-002",
            prediction=_make_prediction(),
            actual_outcome=ActualOutcomeRecord(was_executed=False),
            lineage=_make_lineage(exception_id="EXC-002"),
        )
        ready = svc.get_learning_ready_outcomes()
        assert len(ready) == 1

    def test_prediction_accuracy(self):
        """Calculate prediction accuracy."""
        svc = OutcomeService()
        # Correct
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual(resolution="FEE_ADJUSTMENT", correct=True),
            lineage=_make_lineage(),
        )
        # Incorrect
        svc.record_outcome(
            workflow_id="WF-002",
            exception_id="EXC-002",
            prediction=_make_prediction(resolution_type="FEE_ADJUSTMENT"),
            actual_outcome=_make_actual(resolution="REFUND_ADJUSTMENT", correct=False),
            lineage=_make_lineage(exception_id="EXC-002"),
        )
        accuracy = svc.prediction_accuracy()
        assert accuracy["correct"] == 1
        assert accuracy["incorrect"] == 1

    def test_count_by_status(self):
        """Count outcomes by status."""
        svc = OutcomeService()
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        counts = svc.count_by_status()
        assert counts["RECORDED"] == 1

    def test_duplicate_workflow_id(self):
        """Second outcome for same workflow overwrites mapping."""
        svc = OutcomeService()
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=_make_lineage(),
        )
        svc.record_outcome(
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(resolution="TAX_ADJUSTMENT"),
            lineage=_make_lineage(),
        )
        found = svc.get_outcome_for_workflow("WF-001")
        assert found is not None


# ─────────────────────────────────────────────────────────────────────────────
# Lineage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataLineage:
    """Test data lineage completeness."""

    def test_complete_lineage(self):
        """Full lineage with all references."""
        lineage = DataLineage(
            exception_id="EXC-001",
            evidence_ids=["EVD-001", "EVD-002"],
            prediction_id="PRED-001",
            decision_id="DEC-001",
            execution_id="EXEC-001",
            verification_id="VER-001",
            feedback_id="FB-001",
            audit_event_ids=["AUD-001", "AUD-002"],
            reward_id="REW-001",
            historical_case_id="HC-001",
        )
        assert lineage.exception_id == "EXC-001"
        assert len(lineage.evidence_ids) == 2
        assert lineage.execution_id == "EXEC-001"
        assert lineage.feedback_id == "FB-001"
        assert len(lineage.audit_event_ids) == 2

    def test_minimal_lineage(self):
        """Minimal lineage with only exception ID."""
        lineage = DataLineage(exception_id="EXC-001")
        assert lineage.exception_id == "EXC-001"
        assert lineage.evidence_ids == []
        assert lineage.prediction_id is None
        assert lineage.execution_id is None

    def test_outcome_references_lineage(self):
        """Outcome record references full lineage."""
        lineage = DataLineage(
            exception_id="EXC-001",
            evidence_ids=["EVD-001"],
            execution_id="EXEC-001",
            verification_id="VER-001",
        )
        outcome = OutcomeRecord(
            outcome_id="OUT-001",
            workflow_id="WF-001",
            exception_id="EXC-001",
            prediction=_make_prediction(),
            actual_outcome=_make_actual(),
            lineage=lineage,
        )
        assert outcome.lineage.exception_id == "EXC-001"
        assert "EVD-001" in outcome.lineage.evidence_ids
        assert outcome.lineage.execution_id == "EXEC-001"
