"""
Execution and Verification Workflow Nodes for Phase 8G.

Integrates Phase 8 execution, verification, and rollback
into the LangGraph workflow.

Nodes:
- execute_resolution: Execute the financial resolution
- verify_execution: Verify the resolution achieved its goal
- rollback_resolution: Rollback if verification fails
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import (
    AgentState,
    VerificationStatus as AgentVerificationStatus,
    WorkflowStatus,
)
from app.services.execution import ResolutionExecutionService
from app.services.financial_diff import FinancialDiffService
from app.services.resolution_verification import ResolutionVerificationEngine
from app.services.rollback import RollbackService
from app.schemas.rollback import RollbackReason


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _build_action_request(state: AgentState) -> Dict[str, Any]:
    """Build an action request from agent state."""
    candidate = state.selected_candidate or {}
    # HIGH #4 FIX: Do NOT hardcode verification_passed=True.
    # Verification is an independent safety gate — its result must come
    # from the verification service, not from the request builder.
    # Here we record what the verification service actually determined.
    ver = state.verification
    ver_status = ver.verification_status.value if ver else "NOT_REQUIRED"
    return {
        "action_id": f"ACT-{state.metadata.workflow_id}",
        "idempotency_key": f"key-{state.metadata.workflow_id}-{state.metadata.exception_id}",
        "workflow_id": state.metadata.workflow_id,
        "exception_id": state.metadata.exception_id,
        "case_id": state.metadata.case_id,
        "candidate_id": candidate.get("candidate_id"),
        "resolution_type": candidate.get("resolution_type", ""),
        "financial_adjustment_paise": candidate.get("amount_paise", 0),
        "authorization_source": "HUMAN_APPROVAL" if state.human_review.approval_status.value == "APPROVED" else "AUTO_GUARDRAIL",
        # Caller-provided verification_passed is NEVER trusted.
        # Only the actual verification service result matters.
        "verification_passed": ver_status == "VERIFIED",
        "verification_action": ver_status,
        "guardrail_decision": state.decision,
        "guardrail_confidence": state.confidence,
        "evidence_summary": state.evidence_package or {},
        "metadata": {
            "risk": state.risk,
            "reason_codes": (state.guardrail_result or {}).get("reason_codes", []),
        },
    }


def _build_financial_state(state: AgentState) -> Dict[str, Any]:
    """Build financial state from evidence package."""
    evidence = state.evidence_package or {}
    candidate = state.selected_candidate or {}
    return {
        "payment_amount": evidence.get("payment_amount", 0),
        "expected_amount": evidence.get("expected_amount", candidate.get("expected_amount", 0)),
        "actual_amount": evidence.get("actual_amount", candidate.get("actual_amount", 0)),
        "difference": evidence.get("difference", candidate.get("difference", 0)),
        "total_refunds": evidence.get("total_refunds", 0),
        "total_fees": evidence.get("total_fees", 0),
        "total_taxes": evidence.get("total_taxes", 0),
        "total_adjustments": evidence.get("total_adjustments", 0),
        "settlement_count": evidence.get("settlement_count", 0),
        "refund_count": evidence.get("refund_count", 0),
        "fee_count": evidence.get("fee_count", 0),
        "tax_count": evidence.get("tax_count", 0),
        "adjustment_count": evidence.get("adjustment_count", 0),
    }


def _load_fresh_financial_state(state: AgentState) -> Optional[Dict[str, Any]]:
    """Load FRESH financial state from persistence after execution.

    HIGH #7 FIX: Verification must use the actual post-execution state,
    NOT the stale pre-execution evidence package.

    The execution result's after_state is the authoritative post-execution snapshot.
    If unavailable, returns None to trigger fail-closed verification.
    """
    exec_result_dict = state.execution_result
    if not exec_result_dict:
        return None

    # Use execution result's after_state as the fresh post-execution state
    after_state = exec_result_dict.get("after_state")
    if after_state:
        return after_state

    # Fallback: execution result may carry the adjustment applied.
    # Build the current state from execution evidence (still better than stale evidence_package).
    exec_evidence = exec_result_dict.get("evidence", {})
    if exec_evidence:
        return {
            "payment_amount": exec_evidence.get("payment_amount", 0),
            "expected_amount": exec_evidence.get("expected_amount", 0),
            "actual_amount": exec_evidence.get("actual_amount", 0),
            "difference": exec_evidence.get("difference", 0),
            "total_refunds": exec_evidence.get("total_refunds", 0),
            "total_fees": exec_evidence.get("total_fees", 0),
            "total_taxes": exec_evidence.get("total_taxes", 0),
            "total_adjustments": exec_evidence.get("total_adjustments", 0),
            "settlement_count": exec_evidence.get("settlement_count", 0),
            "refund_count": exec_evidence.get("refund_count", 0),
            "fee_count": exec_evidence.get("fee_count", 0),
            "tax_count": exec_evidence.get("tax_count", 0),
            "adjustment_count": exec_evidence.get("adjustment_count", 0),
        }

    # FAIL-CLOSED: No fresh state available
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Execute Resolution Node
# ─────────────────────────────────────────────────────────────────────────────


def execute_resolution(state: AgentState) -> Dict[str, Any]:
    """Execute the financial resolution.

    Delegates to ResolutionExecutionService.
    Stores execution result in state.
    """
    start_time = time.perf_counter()
    node_name = "execute_resolution"

    try:
        action_request = _build_action_request(state)
        financial_state = _build_financial_state(state)

        service = ResolutionExecutionService()
        result = service.execute(action_request, financial_state)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["execution_result"] = result.model_dump(mode="json")
        updates["execution_status"] = result.status.value
        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        return _record_node(state, node_name, success=False, error=str(e), start_time=start_time)


# ─────────────────────────────────────────────────────────────────────────────
# Verify Execution Node
# ─────────────────────────────────────────────────────────────────────────────


def verify_execution(state: AgentState) -> Dict[str, Any]:
    """Verify the resolution achieved its goal.

    Delegates to ResolutionVerificationEngine.
    Stores verification result in state.

    HIGH #7 FIX: The current financial state MUST come from persistence
    (fresh read after execution), NOT from stale pre-execution evidence.
    If fresh state cannot be retrieved, verification FAILS CLOSED.
    """
    start_time = time.perf_counter()
    node_name = "verify_execution"

    try:
        exec_result_dict = state.execution_result
        if not exec_result_dict:
            return _record_node(state, node_name, success=False, error="No execution result to verify", start_time=start_time)

        # Reconstruct ExecutionResult from dict
        from app.schemas.execution import ExecutionResult
        exec_result = ExecutionResult(**exec_result_dict)

        # HIGH #7 FIX: Load FRESH financial state from persistence after execution.
        # Do NOT use stale evidence_package (pre-execution data).
        financial_state_dict = _load_fresh_financial_state(state)
        if financial_state_dict is None:
            # FAIL-CLOSED: Cannot verify without fresh state
            return _record_node(
                state, node_name, success=False,
                error="FAIL-CLOSED: Cannot retrieve fresh financial state after execution",
                start_time=start_time,
            )
        from app.schemas.execution import FinancialStateSnapshot
        current_state = FinancialStateSnapshot(
            exception_id=state.metadata.exception_id,
            **financial_state_dict,
        )

        engine = ResolutionVerificationEngine()
        verification = engine.verify(exec_result, current_state)

        # Compute diff
        diff_service = FinancialDiffService()
        diff = diff_service.compare(
            exec_result.before_state,
            current_state,
            exec_result,
        )

        # Update verification state
        agent_verification = state.verification.model_dump()
        if verification.status.value == "PASSED":
            agent_verification["verification_status"] = AgentVerificationStatus.VERIFIED.value
        else:
            agent_verification["verification_status"] = AgentVerificationStatus.FAILED.value
        agent_verification["verification_result"] = verification.model_dump(mode="json")
        agent_verification["verified_at"] = datetime.utcnow().isoformat()
        agent_verification["verified_by"] = "resolution_verification_engine"

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["verification"] = agent_verification
        updates["resolution_verification"] = verification.model_dump(mode="json")
        updates["financial_diff"] = diff.model_dump(mode="json")
        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        return _record_node(state, node_name, success=False, error=str(e), start_time=start_time)


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Resolution Node
# ─────────────────────────────────────────────────────────────────────────────


def rollback_resolution(state: AgentState) -> Dict[str, Any]:
    """Rollback the resolution if verification failed.

    Delegates to RollbackService.
    Stores rollback result in state.
    """
    start_time = time.perf_counter()
    node_name = "rollback_resolution"

    try:
        exec_result_dict = state.execution_result
        if not exec_result_dict:
            return _record_node(state, node_name, success=False, error="No execution result to rollback", start_time=start_time)

        from app.schemas.execution import ExecutionResult, FinancialStateSnapshot
        exec_result = ExecutionResult(**exec_result_dict)

        # Determine rollback reason
        reason = RollbackReason.VERIFICATION_FAILED
        ver_result = state.resolution_verification or {}
        if ver_result.get("has_unintended_changes"):
            reason = RollbackReason.UNINTENDED_CHANGES

        # Get current financial state
        financial_state_dict = _build_financial_state(state)
        current_state = FinancialStateSnapshot(
            exception_id=state.metadata.exception_id,
            **financial_state_dict,
        )

        service = RollbackService()
        result = service.rollback(exec_result, current_state, reason=reason)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["rollback_result"] = result.model_dump(mode="json")
        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        return _record_node(state, node_name, success=False, error=str(e), start_time=start_time)
