"""
Human-in-the-Loop approval for Razorpay CloseLoop Phase 7G.

Allows the LangGraph workflow to:
1. Pause safely
2. Persist state
3. Wait for human input
4. Resume after decision

Security:
- Validate workflow ID
- Validate current status
- Prevent double approval
- Prevent approval after rejection
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Human Decision
# ─────────────────────────────────────────────────────────────────────────────


class HumanDecision:
    """Represents a human reviewer's decision."""

    def __init__(
        self,
        workflow_id: str,
        decision: str,
        reviewer_id: str = "unknown",
        reason: str = "",
    ):
        self.workflow_id = workflow_id
        self.decision = decision  # "APPROVED" or "REJECTED"
        self.reviewer_id = reviewer_id
        self.reason = reason
        self.timestamp = datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────
# Human Approval Node
# ─────────────────────────────────────────────────────────────────────────────


def _record_node(
    state: AgentState,
    node_name: str,
    success: bool,
    error: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    elapsed_ms = None
    if start_time:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    log_entry = {
        "node": node_name,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms else None,
        "error": error,
    }

    metadata = state.metadata.model_dump()
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + [node_name]
    metadata["execution_log"] = list(state.metadata.execution_log) + [log_entry]

    if error:
        metadata["errors"] = list(state.metadata.errors) + [error]

    return {"metadata": metadata}


def pause_for_human_approval(state: AgentState) -> Dict[str, Any]:
    """Pause workflow and wait for human approval.

    This node:
    1. Records the case details for human review
    2. Sets status to WAITING_FOR_HUMAN
    3. Persists state (via checkpoint)
    4. Returns — workflow pauses here

    The workflow resumes when human_decision() is called.
    """
    start_time = time.perf_counter()
    node_name = "pause_for_human_approval"

    try:
        # Build review package for human
        review_package = _build_review_package(state)

        # Set human review state
        human_state = {
            "approval_status": HumanApprovalStatus.PENDING.value,
            "assigned_reviewer": None,
            "review_requested_at": datetime.utcnow().isoformat(),
            "review_reason": state.guardrail_result.get("primary_reason", "Guardrail flagged")
            if state.guardrail_result
            else "Unknown reason",
            "review_priority": "HIGH" if state.risk == "HIGH" else "MEDIUM",
        }

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["human_review"] = human_state
        updates["metadata"]["workflow_status"] = WorkflowStatus.WAITING_FOR_HUMAN.value
        updates["metadata"]["current_node"] = node_name

        # Store review package for human consumption
        updates["_review_package"] = review_package

        return updates

    except Exception as e:
        return _record_node(state, node_name, success=False, error=str(e), start_time=start_time)


def _build_review_package(state: AgentState) -> Dict[str, Any]:
    """Build a complete review package for human reviewers."""
    candidate = state.selected_candidate or {}
    guardrail = state.guardrail_result or {}

    return {
        "workflow_id": state.metadata.workflow_id,
        "exception_id": state.metadata.exception_id,
        "case_id": state.metadata.case_id,
        "candidate": candidate,
        "proposed_resolution": candidate.get("resolution_type"),
        "financial_adjustment_paise": candidate.get("amount_paise", 0),
        "evidence_summary": {
            "coverage": (state.evidence_package or {}).get("evidence_coverage", 0),
            "consistency": (state.evidence_package or {}).get("evidence_consistency", 0),
        },
        "guardrail_decision": guardrail.get("decision"),
        "guardrail_reason": guardrail.get("primary_reason"),
        "confidence": state.confidence,
        "risk": state.risk,
        "classification": (state.classification or {}).get("exception_type"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human Decision Processing
# ─────────────────────────────────────────────────────────────────────────────


def process_human_decision(
    state: AgentState,
    decision: HumanDecision,
) -> Dict[str, Any]:
    """Process a human decision and resume workflow.

    Security validations:
    - Workflow ID must match
    - Workflow must be in WAITING_FOR_HUMAN status
    - Approval status must be PENDING
    - No double approval
    - No approval after rejection

    Args:
        state: Current agent state
        decision: Human reviewer's decision

    Returns:
        State updates

    Raises:
        ValueError: If validation fails
    """
    # Validate workflow ID
    if decision.workflow_id != state.metadata.workflow_id:
        raise ValueError(
            f"Workflow ID mismatch: expected {state.metadata.workflow_id}, "
            f"got {decision.workflow_id}"
        )

    # Validate workflow status
    if state.metadata.workflow_status != WorkflowStatus.WAITING_FOR_HUMAN:
        raise ValueError(
            f"Workflow not waiting for human: current status is "
            f"{state.metadata.workflow_status.value}"
        )

    # Validate approval status
    if state.human_review.approval_status != HumanApprovalStatus.PENDING:
        raise ValueError(
            f"Approval not pending: current status is "
            f"{state.human_review.approval_status.value}"
        )

    # Process decision
    if decision.decision == "APPROVED":
        return _process_approval(state, decision)
    elif decision.decision == "REJECTED":
        return _process_rejection(state, decision)
    else:
        raise ValueError(f"Invalid decision: {decision.decision}")


def _process_approval(
    state: AgentState,
    decision: HumanDecision,
) -> Dict[str, Any]:
    """Process human approval — continue to verification."""
    human_state = state.human_review.model_dump()
    human_state["approval_status"] = HumanApprovalStatus.APPROVED.value
    human_state["review_completed_at"] = datetime.utcnow().isoformat()
    human_state["reviewer_notes"] = decision.reason
    human_state["assigned_reviewer"] = decision.reviewer_id

    metadata = state.metadata.model_dump()
    metadata["workflow_status"] = WorkflowStatus.RUNNING.value
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + ["human_decision"]
    metadata["execution_log"] = list(state.metadata.execution_log) + [{
        "node": "human_decision",
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "error": None,
        "decision": "APPROVED",
        "reviewer": decision.reviewer_id,
    }]

    return {
        "human_review": human_state,
        "metadata": metadata,
    }


def _process_rejection(
    state: AgentState,
    decision: HumanDecision,
) -> Dict[str, Any]:
    """Process human rejection — escalate."""
    human_state = state.human_review.model_dump()
    human_state["approval_status"] = HumanApprovalStatus.REJECTED.value
    human_state["review_completed_at"] = datetime.utcnow().isoformat()
    human_state["reviewer_notes"] = decision.reason
    human_state["assigned_reviewer"] = decision.reviewer_id

    metadata = state.metadata.model_dump()
    metadata["workflow_status"] = WorkflowStatus.FAILED.value
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + ["human_decision"]
    metadata["execution_log"] = list(state.metadata.execution_log) + [{
        "node": "human_decision",
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "error": None,
        "decision": "REJECTED",
        "reviewer": decision.reviewer_id,
    }]

    warnings = list(state.warnings) + [f"REJECTED by {decision.reviewer_id}: {decision.reason}"]

    return {
        "human_review": human_state,
        "metadata": metadata,
        "decision": "REJECTED",
        "warnings": warnings,
    }
