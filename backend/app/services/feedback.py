"""
Feedback and Outcome services for Razorpay CloseLoop Phase 9A.

Implements:
  - FeedbackService: records and retrieves human feedback
  - OutcomeService: builds and records outcome records

Key safety principle:
  Learning may improve future recommendations.
  Learning must NEVER directly weaken or bypass Phase 6 guardrails.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

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


def _gen_id(prefix: str) -> str:
    """Generate a prefixed unique ID."""
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Service
# ─────────────────────────────────────────────────────────────────────────────


class FeedbackService:
    """Records and retrieves human feedback on resolutions.

    Feedback is immutable once created.
    Corrections create new records referencing the original.
    """

    def __init__(self) -> None:
        self._feedback: Dict[str, FeedbackRecord] = {}
        self._by_workflow: Dict[str, List[str]] = {}

    def record_feedback(
        self,
        workflow_id: str,
        exception_id: str,
        feedback_type: FeedbackType,
        reviewer: str,
        system_prediction: str,
        case_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        system_confidence: Optional[float] = None,
        financial_adjustment_paise: int = 0,
        correction: Optional[CorrectionDetail] = None,
        rejection: Optional[RejectionDetail] = None,
        escalation: Optional[EscalationDetail] = None,
        reason: Optional[str] = None,
        evidence_references_reviewed: Optional[List[str]] = None,
        correction_of: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        model_version: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> FeedbackRecord:
        """Record human feedback on a resolution.

        Args:
            workflow_id: Workflow being reviewed.
            exception_id: Exception being reviewed.
            feedback_type: APPROVE, REJECT, CORRECT, or ESCALATE.
            reviewer: Who provided the feedback.
            system_prediction: What the system predicted.
            case_id: Optional case ID.
            candidate_id: Optional candidate ID.
            system_confidence: Optional system confidence.
            financial_adjustment_paise: Proposed adjustment.
            correction: Correction details (for CORRECT).
            rejection: Rejection details (for REJECT).
            escalation: Escalation details (for ESCALATE).
            reason: Generic reason.
            evidence_references_reviewed: Evidence IDs the reviewer checked.
            correction_of: Previous feedback ID this supersedes.
            reviewer_role: Role of the reviewer.
            model_version: ML model version at time of review.
            policy_version: Guardrail policy version at time of review.

        Returns:
            Created FeedbackRecord.
        """
        feedback_id = _gen_id("FB")

        record = FeedbackRecord(
            feedback_id=feedback_id,
            workflow_id=workflow_id,
            exception_id=exception_id,
            case_id=case_id,
            candidate_id=candidate_id,
            feedback_type=feedback_type,
            reviewer=reviewer,
            reviewer_role=reviewer_role,
            system_prediction=system_prediction,
            system_confidence=system_confidence,
            financial_adjustment_paise=financial_adjustment_paise,
            correction=correction,
            rejection=rejection,
            escalation=escalation,
            reason=reason,
            evidence_references_reviewed=evidence_references_reviewed or [],
            correction_of=correction_of,
            created_at=datetime.utcnow(),
            model_version=model_version,
            policy_version=policy_version,
        )

        self._feedback[feedback_id] = record
        self._by_workflow.setdefault(workflow_id, []).append(feedback_id)
        return record

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Retrieve a feedback record by ID."""
        return self._feedback.get(feedback_id)

    def get_feedback_for_workflow(self, workflow_id: str) -> List[FeedbackRecord]:
        """Retrieve all feedback records for a workflow."""
        ids = self._by_workflow.get(workflow_id, [])
        return [self._feedback[fid] for fid in ids if fid in self._feedback]

    def get_feedback_for_exception(self, exception_id: str) -> List[FeedbackRecord]:
        """Retrieve all feedback records for an exception."""
        return [
            r for r in self._feedback.values()
            if r.exception_id == exception_id
        ]

    def count_by_type(self) -> Dict[str, int]:
        """Count feedback records by type."""
        counts: Dict[str, int] = {}
        for record in self._feedback.values():
            key = record.feedback_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_corrections(self) -> List[FeedbackRecord]:
        """Get all correction feedback records."""
        return [
            r for r in self._feedback.values()
            if r.feedback_type == FeedbackType.CORRECT
        ]

    def has_feedback(self, workflow_id: str) -> bool:
        """Check if a workflow has any feedback."""
        return len(self._by_workflow.get(workflow_id, [])) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Service
# ─────────────────────────────────────────────────────────────────────────────


class OutcomeService:
    """Builds and records outcome records.

    Separates prediction, actual outcome, human feedback, and verification.
    """

    def __init__(self) -> None:
        self._outcomes: Dict[str, OutcomeRecord] = {}
        self._by_workflow: Dict[str, str] = {}
        self._by_exception: Dict[str, str] = {}

    def record_outcome(
        self,
        workflow_id: str,
        exception_id: str,
        prediction: PredictionRecord,
        actual_outcome: ActualOutcomeRecord,
        lineage: DataLineage,
        case_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        human_feedback_id: Optional[str] = None,
        human_feedback_type: Optional[FeedbackType] = None,
        human_override: bool = False,
        verification_passed: bool = False,
        verification_notes: Optional[str] = None,
        financial_impact: Optional[FinancialImpact] = None,
        decision: Optional[str] = None,
        confidence: Optional[float] = None,
        risk: Optional[str] = None,
        nodes_executed: Optional[List[str]] = None,
        completed_at: Optional[datetime] = None,
        feedback_received_at: Optional[datetime] = None,
        ground_truth_exception_type: Optional[str] = None,
        ground_truth_resolution: Optional[str] = None,
        ground_truth_resolvable: Optional[bool] = None,
    ) -> OutcomeRecord:
        """Record the complete outcome of a workflow.

        Args:
            workflow_id: Workflow identifier.
            exception_id: Exception identifier.
            prediction: What the system predicted.
            actual_outcome: What actually happened.
            lineage: Full data lineage traceability.
            case_id: Optional case ID.
            candidate_id: Optional candidate ID.
            human_feedback_id: Optional feedback record ID.
            human_feedback_type: Type of feedback received.
            human_override: Whether human overrode the prediction.
            verification_passed: Whether verification passed.
            verification_notes: Verification details.
            financial_impact: Financial impact details.
            decision: Guardrail decision.
            confidence: Final confidence.
            risk: Risk category.
            nodes_executed: Nodes that ran.
            completed_at: When workflow completed.
            feedback_received_at: When feedback was received.
            ground_truth_exception_type: Evaluation only.
            ground_truth_resolution: Evaluation only.
            ground_truth_resolvable: Evaluation only.

        Returns:
            Created OutcomeRecord.
        """
        outcome_id = _gen_id("OUT")

        status = OutcomeStatus.RECORDED
        if human_feedback_id:
            status = OutcomeStatus.FEEDBACK_RECEIVED

        record = OutcomeRecord(
            outcome_id=outcome_id,
            workflow_id=workflow_id,
            exception_id=exception_id,
            case_id=case_id,
            candidate_id=candidate_id,
            prediction=prediction,
            actual_outcome=actual_outcome,
            human_feedback_id=human_feedback_id,
            human_feedback_type=human_feedback_type,
            human_override=human_override,
            verification_passed=verification_passed,
            verification_notes=verification_notes,
            financial_impact=financial_impact or FinancialImpact(),
            lineage=lineage,
            status=status,
            ground_truth_exception_type=ground_truth_exception_type,
            ground_truth_resolution=ground_truth_resolution,
            ground_truth_resolvable=ground_truth_resolvable,
            decision=decision,
            confidence=confidence,
            risk=risk,
            nodes_executed=nodes_executed or [],
            created_at=datetime.utcnow(),
            completed_at=completed_at,
            feedback_received_at=feedback_received_at,
        )

        self._outcomes[outcome_id] = record
        self._by_workflow[workflow_id] = outcome_id
        self._by_exception[exception_id] = outcome_id
        return record

    def get_outcome(self, outcome_id: str) -> Optional[OutcomeRecord]:
        """Retrieve an outcome record by ID."""
        return self._outcomes.get(outcome_id)

    def get_outcome_for_workflow(self, workflow_id: str) -> Optional[OutcomeRecord]:
        """Retrieve the outcome for a workflow."""
        oid = self._by_workflow.get(workflow_id)
        if oid:
            return self._outcomes.get(oid)
        return None

    def get_outcome_for_exception(self, exception_id: str) -> Optional[OutcomeRecord]:
        """Retrieve the outcome for an exception."""
        oid = self._by_exception.get(exception_id)
        if oid:
            return self._outcomes.get(oid)
        return None

    def update_feedback(
        self,
        workflow_id: str,
        feedback_id: str,
        feedback_type: FeedbackType,
        human_override: bool = False,
        feedback_received_at: Optional[datetime] = None,
    ) -> Optional[OutcomeRecord]:
        """Update an outcome with received feedback."""
        oid = self._by_workflow.get(workflow_id)
        if not oid or oid not in self._outcomes:
            return None
        record = self._outcomes[oid]
        record.human_feedback_id = feedback_id
        record.human_feedback_type = feedback_type
        record.human_override = human_override
        record.status = OutcomeStatus.FEEDBACK_RECEIVED
        record.feedback_received_at = feedback_received_at or datetime.utcnow()
        return record

    def mark_reward_calculated(self, workflow_id: str) -> Optional[OutcomeRecord]:
        """Mark outcome as having reward calculated."""
        oid = self._by_workflow.get(workflow_id)
        if not oid or oid not in self._outcomes:
            return None
        record = self._outcomes[oid]
        record.status = OutcomeStatus.REWARD_CALCULATED
        return record

    def mark_stored_for_learning(self, workflow_id: str) -> Optional[OutcomeRecord]:
        """Mark outcome as stored for future learning."""
        oid = self._by_workflow.get(workflow_id)
        if not oid or oid not in self._outcomes:
            return None
        record = self._outcomes[oid]
        record.status = OutcomeStatus.STORED_FOR_LEARNING
        return record

    def get_learning_ready_outcomes(self) -> List[OutcomeRecord]:
        """Get all outcomes ready for learning."""
        return [
            r for r in self._outcomes.values()
            if r.is_learning_ready()
        ]

    def prediction_accuracy(self) -> Dict[str, int]:
        """Count correct vs incorrect predictions."""
        correct = 0
        incorrect = 0
        unknown = 0
        for record in self._outcomes.values():
            result = record.prediction_matches_actual()
            if result is True:
                correct += 1
            elif result is False:
                incorrect += 1
            else:
                unknown += 1
        return {"correct": correct, "incorrect": incorrect, "unknown": unknown}

    def count_by_status(self) -> Dict[str, int]:
        """Count outcomes by status."""
        counts: Dict[str, int] = {}
        for record in self._outcomes.values():
            key = record.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts
