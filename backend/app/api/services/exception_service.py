"""
Exception Service for Razorpay CloseLoop Phase 13.4.

Provides exception management by delegating to existing services:
- FinancialDataAdapter for data loading and search
- FeedbackService for human feedback recording
- OutcomeService for outcome recording

Does NOT duplicate business logic.
Does NOT bypass Phase 6 guardrails.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.feedback import FeedbackService, OutcomeService
from app.schemas.feedback import (
    ActualOutcomeRecord,
    CorrectionDetail,
    DataLineage,
    EscalationDetail,
    FeedbackType as Phase9FeedbackType,
    PredictionRecord,
    RejectionDetail,
)
from app.schemas.enums import ExceptionType, RiskCategory


# In-memory exception registry
_exception_registry: Dict[str, Dict[str, Any]] = {}


class ExceptionService:
    """
    Service for managing exceptions.

    Loads data from the FinancialDataAdapter and delegates
    feedback/outcome operations to Phase 9 services.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent.parent / "data")
        self._data_dir = data_dir
        self._feedback_service = FeedbackService()
        self._outcome_service = OutcomeService()

    def list_exceptions(
        self,
        limit: int = 50,
        offset: int = 0,
        exception_type: Optional[str] = None,
        status: Optional[str] = None,
        risk_category: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List exceptions from loaded batch data.

        Supports filtering by exception type, status, risk category.
        """
        cases = self._load_cases(batch_id)
        results = []

        for case in cases:
            # Build exception summary from case data
            exc = self._case_to_exception(case)

            # Apply filters
            if exception_type and exc.get("exception_type") != exception_type:
                continue
            if status and exc.get("status") != status:
                continue
            if risk_category and exc.get("risk_category") != risk_category:
                continue

            results.append(exc)

        return results[offset:offset + limit]

    def get_exception(self, exception_id: str, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed exception information.

        Returns exception with financial discrepancy, classification, risk.
        """
        cases = self._load_cases(batch_id)

        for case in cases:
            if case.get("case_id") == exception_id:
                exc = self._case_to_exception(case)
                exc["detail"] = True
                return exc

        # Check registered exceptions (from API operations)
        return _exception_registry.get(exception_id)

    def resolve_exception(
        self,
        exception_id: str,
        resolution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Submit a resolution PROPOSAL for an exception.

        CRITICAL #1 FIX: This endpoint does NOT declare a resolution safe.
        It records a proposal that must go through the guardrail pipeline.
        The client CANNOT force AUTO, verification_passed, or a final outcome.

        Flow:
        - Client proposes resolution
        - Server records the proposal
        - Resolution is pending guardrail evaluation and verification
        - Server-computed decision, verification, and outcome are authoritative
        """
        # Verify exception exists
        exc = self.get_exception(exception_id)
        if exc is None:
            return {"error": f"Exception '{exception_id}' not found"}

        resolution_type = resolution.get("resolution_type", "UNKNOWN")
        adjustment_paise = resolution.get("adjustment_paise", 0)
        reason = resolution.get("reason", "")
        candidate_id = resolution.get("candidate_id")

        # Record the proposal as an outcome (PHASE 9 learning data)
        workflow_id = f"WF-{exception_id}"
        outcome = self._outcome_service.record_outcome(
            workflow_id=workflow_id,
            exception_id=exception_id,
            prediction=PredictionRecord(
                resolution_type=resolution_type,
            ),
            actual_outcome=ActualOutcomeRecord(
                financial_impact_paise=adjustment_paise,
            ),
            lineage=DataLineage(
                exception_id=exception_id,
            ),
            case_id=exc.get("case_id"),
        )

        # Register the PROPOSAL — status is PENDING, not RESOLVED.
        # The client must NOT be able to skip guardrails/verification.
        _exception_registry[exception_id] = {
            **exc,
            "status": "PENDING",  # NOT RESOLVED — guardrails must evaluate first
            "resolution_type": resolution_type,
            "adjustment_paise": adjustment_paise,
            "resolution_reason": reason,
            "candidate_id": candidate_id,
            "proposal_submitted_at": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
        }

        # CRITICAL #1 FIX: Server does NOT claim guardrail_decision=AUTO.
        # The client-submitted proposal is PENDING guardrail evaluation.
        return {
            "exception_id": exception_id,
            "resolution_type": resolution_type,
            "status": "PENDING",  # Proposal submitted, guardrails not yet evaluated
            "adjustment_paise": adjustment_paise,
            "guardrail_decision": None,  # NOT YET COMPUTED by server
            "execution_result": None,  # NOT YET EXECUTED
            "verification_result": None,  # NOT YET VERIFIED
            "workflow_id": workflow_id,
            "message": "Resolution proposal submitted. Pending guardrail evaluation and verification.",
        }

    def approve_exception(
        self,
        exception_id: str,
        approval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Approve a resolution through the human approval workflow.

        Records feedback through Phase 9 FeedbackService.
        """
        exc = self.get_exception(exception_id)
        if exc is None:
            return {"error": f"Exception '{exception_id}' not found"}

        approved_by = approval.get("approved_by", "unknown")
        comments = approval.get("comments", "")
        workflow_id = exc.get("workflow_id", f"WF-{exception_id}")

        # Record feedback
        fb = self._feedback_service.record_feedback(
            workflow_id=workflow_id,
            exception_id=exception_id,
            feedback_type=Phase9FeedbackType.APPROVE,
            reviewer=approved_by,
            system_prediction=exc.get("resolution_type", "UNKNOWN"),
            case_id=exc.get("case_id"),
        )

        # Update exception status
        _exception_registry[exception_id] = {
            **exc,
            "status": "APPROVED",
            "approved_by": approved_by,
            "approved_at": datetime.utcnow().isoformat(),
            "feedback_id": fb.feedback_id,
        }

        return {
            "exception_id": exception_id,
            "status": "APPROVED",
            "approved_by": approved_by,
            "feedback_id": fb.feedback_id,
        }

    def reject_exception(
        self,
        exception_id: str,
        rejection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reject a resolution through the feedback mechanism.

        Records rejection through Phase 9 FeedbackService.
        """
        exc = self.get_exception(exception_id)
        if exc is None:
            return {"error": f"Exception '{exception_id}' not found"}

        rejected_by = rejection.get("rejected_by", "unknown")
        reason = rejection.get("reason", "No reason provided")
        workflow_id = exc.get("workflow_id", f"WF-{exception_id}")

        # Record feedback
        fb = self._feedback_service.record_feedback(
            workflow_id=workflow_id,
            exception_id=exception_id,
            feedback_type=Phase9FeedbackType.REJECT,
            reviewer=rejected_by,
            system_prediction=exc.get("resolution_type", "UNKNOWN"),
            case_id=exc.get("case_id"),
            rejection=RejectionDetail(
                rejection_reason=reason,
            ),
        )

        # Update exception status
        _exception_registry[exception_id] = {
            **exc,
            "status": "REJECTED",
            "rejected_by": rejected_by,
            "rejection_reason": reason,
            "rejected_at": datetime.utcnow().isoformat(),
            "feedback_id": fb.feedback_id,
        }

        return {
            "exception_id": exception_id,
            "status": "REJECTED",
            "rejected_by": rejected_by,
            "reason": reason,
            "feedback_id": fb.feedback_id,
        }

    def escalate_exception(
        self,
        exception_id: str,
        reason: str,
        escalated_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Escalate an exception for human review.

        Records escalation through Phase 9 FeedbackService.
        """
        exc = self.get_exception(exception_id)
        if exc is None:
            return {"error": f"Exception '{exception_id}' not found"}

        workflow_id = exc.get("workflow_id", f"WF-{exception_id}")

        # Record feedback
        fb = self._feedback_service.record_feedback(
            workflow_id=workflow_id,
            exception_id=exception_id,
            feedback_type=Phase9FeedbackType.ESCALATE,
            reviewer=escalated_by or "system",
            system_prediction=exc.get("resolution_type", "UNKNOWN"),
            case_id=exc.get("case_id"),
            escalation=EscalationDetail(
                escalation_reason=reason,
            ),
        )

        # Update exception status
        _exception_registry[exception_id] = {
            **exc,
            "status": "ESCALATED",
            "escalated_by": escalated_by,
            "escalation_reason": reason,
            "escalated_at": datetime.utcnow().isoformat(),
            "feedback_id": fb.feedback_id,
        }

        return {
            "exception_id": exception_id,
            "status": "ESCALATED",
            "reason": reason,
            "escalated_by": escalated_by,
            "feedback_id": fb.feedback_id,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_cases(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load cases from batch data."""
        if batch_id:
            batch_dirs = [batch_id]
        else:
            # Find all batch directories
            batch_dirs = []
            if os.path.isdir(self._data_dir):
                for d in os.listdir(self._data_dir):
                    gen_dir = os.path.join(self._data_dir, d, "generated")
                    if os.path.isdir(gen_dir):
                        batch_dirs.append(d)

        all_cases = []
        for bd in batch_dirs:
            cases_path = os.path.join(self._data_dir, bd, "generated", "cases.json")
            if os.path.isfile(cases_path):
                import json
                with open(cases_path, "r") as f:
                    cases = json.load(f)
                if isinstance(cases, list):
                    for c in cases:
                        c["_batch_id"] = bd
                    all_cases.extend(cases)

        return all_cases

    def _case_to_exception(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a case record to an exception summary."""
        case_id = case.get("case_id", "")
        scenario = case.get("scenario", "UNKNOWN")

        # Map scenario to ExceptionType
        try:
            exc_type = ExceptionType(scenario)
        except ValueError:
            exc_type = ExceptionType.UNKNOWN

        # Map risk
        risk = case.get("risk_category", "LOW")
        try:
            risk_cat = RiskCategory(risk)
        except ValueError:
            risk_cat = RiskCategory.LOW

        # Check if there's a registered override
        reg = _exception_registry.get(case_id, {})

        return {
            "exception_id": case_id,
            "case_id": case_id,
            "merchant_id": case.get("merchant_id", ""),
            "payment_id": case.get("payment_id", ""),
            "exception_type": exc_type.value,
            "expected_amount_paise": case.get("expected_amount", 0),
            "actual_amount_paise": case.get("actual_amount", 0),
            "difference_paise": case.get("difference", 0),
            "risk_category": risk_cat.value,
            "status": reg.get("status", "PENDING"),
            "classification_confidence": case.get("confidence"),
            "resolvable": case.get("resolvable", False),
            "batch_id": case.get("_batch_id", ""),
            "created_at": case.get("observation_timestamp"),
        }
