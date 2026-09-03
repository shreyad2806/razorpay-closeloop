"""
Resolve / Action Boundary Node for Razorpay CloseLoop Phase 7I.

This is the BOUNDARY between the recommendation pipeline and future execution.

CRITICAL SAFETY RULE:
- This node does NOT execute financial actions
- It only produces an action request for a future execution service
- Real execution belongs to a future guarded agent/action layer

The node verifies:
1. Guardrail decision allows action
2. Verification passed
3. Authorization exists
4. Candidate is unchanged

If any condition fails → ESCALATE / UNRESOLVED
"""

import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.action_request import (
    ActionRequest,
    ActionRequestResult,
    ActionStatus,
    AuthorizationSource,
)
from app.schemas.agent_state import AgentState, WorkflowStatus


# ─────────────────────────────────────────────────────────────────────────────
# Action Boundary Node
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


def resolve_action_boundary(state: AgentState) -> Dict[str, Any]:
    """Resolve / action boundary node.

    Verifies that all conditions allow action, then produces an action request.

    This node does NOT:
    - Execute refunds
    - Modify settlements
    - Call financial APIs
    - Perform monetary actions

    It only produces an ActionRequest for a future execution service.

    Safety checks:
    1. Guardrail decision allows action (AUTO or human-approved)
    2. Verification passed
    3. Authorization exists
    4. Candidate is unchanged
    """
    start_time = time.perf_counter()
    node_name = "resolve_action_boundary"

    try:
        # ── Safety Check 1: Guardrail decision allows action ──
        decision = state.decision
        if decision not in ("AUTO", "HUMAN_REVIEW"):
            return _reject(state, node_name, start_time, [
                f"Guardrail decision '{decision}' does not allow action"
            ])

        # For HUMAN_REVIEW, require explicit human approval
        if decision == "HUMAN_REVIEW":
            from app.schemas.agent_state import HumanApprovalStatus
            if state.human_review.approval_status != HumanApprovalStatus.APPROVED:
                return _reject(state, node_name, start_time, [
                    "HUMAN_REVIEW decision requires explicit human approval"
                ])

        # ── Safety Check 2: Verification passed ──
        verification = state.verification
        verification_status = verification.verification_status.value
        if verification_status != "VERIFIED":
            return _reject(state, node_name, start_time, [
                f"Verification status is '{verification_status}', expected 'VERIFIED'"
            ])

        # ── Safety Check 3: Authorization exists ──
        candidate = state.selected_candidate
        if not candidate:
            return _reject(state, node_name, start_time, [
                "No selected candidate"
            ])

        # ── Safety Check 4: Candidate has resolution info ──
        resolution_type = candidate.get("resolution_type")
        if not resolution_type:
            return _reject(state, node_name, start_time, [
                "Candidate has no resolution type"
            ])

        # ── Build Action Request ──
        auth_source = AuthorizationSource.AUTO_GUARDRAIL
        authorized_by = "auto_guardrail"
        if decision == "HUMAN_REVIEW":
            auth_source = AuthorizationSource.HUMAN_APPROVAL
            authorized_by = state.human_review.assigned_reviewer or "human_reviewer"

        # Idempotency key: deterministic from workflow + candidate
        idempotency_key = _compute_idempotency_key(
            state.metadata.workflow_id,
            state.metadata.exception_id,
            candidate.get("candidate_id", ""),
        )

        evidence = state.evidence_package or {}
        verification_result = verification.verification_result or {}

        action_request = ActionRequest(
            action_id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
            idempotency_key=idempotency_key,
            workflow_id=state.metadata.workflow_id,
            exception_id=state.metadata.exception_id,
            case_id=state.metadata.case_id,
            candidate_id=candidate.get("candidate_id"),
            resolution_type=resolution_type,
            financial_adjustment_paise=candidate.get("amount_paise", 0),
            financial_adjustment_description=candidate.get("description"),
            authorization_source=auth_source,
            authorized_by=authorized_by,
            authorization_timestamp=datetime.utcnow(),
            verification_passed=(verification_status == "VERIFIED"),
            verification_action=verification_status,
            guardrail_decision=decision,
            guardrail_confidence=state.confidence,
            status=ActionStatus.PENDING,
            evidence_summary={
                "coverage": evidence.get("evidence_coverage", 0),
                "consistency": evidence.get("evidence_consistency", 0),
            },
            metadata={
                "exception_type": (state.classification or {}).get("exception_type"),
                "risk": state.risk,
            },
        )

        # ── Record success ──
        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["action_request"] = action_request.model_dump(mode="json")
        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        return _reject(state, node_name, start_time, [f"Unexpected error: {str(e)}"])


def _reject(
    state: AgentState,
    node_name: str,
    start_time: float,
    reasons: list,
) -> Dict[str, Any]:
    """Reject the action request — ESCALATE."""
    updates = _record_node(state, node_name, success=False, error="; ".join(reasons), start_time=start_time)
    updates["metadata"]["current_node"] = node_name
    updates["action_request"] = None
    warnings = list(state.warnings) + [f"ACTION_REJECTED: {'; '.join(reasons)}"]
    updates["warnings"] = warnings
    return updates


def _compute_idempotency_key(
    workflow_id: str,
    exception_id: str,
    candidate_id: str,
) -> str:
    """Compute a deterministic idempotency key."""
    raw = f"{workflow_id}:{exception_id}:{candidate_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
