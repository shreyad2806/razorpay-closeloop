"""
Guardrail node for Razorpay CloseLoop Phase 7E.

Integrates the Phase 6 Guardrail Engine into the LangGraph workflow.

This node is a HARD BOUNDARY.
LangGraph must NEVER bypass guardrails.

The node delegates to the existing GuardrailEngine.
It does NOT recreate guardrail logic.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import AgentState, WorkflowStatus
from app.schemas.resolution_engine import ResolutionEngineResult
from app.services.guardrail_engine import GuardrailEngine


# ─────────────────────────────────────────────────────────────────────────────
# Node Result Helper
# ─────────────────────────────────────────────────────────────────────────────


def _record_node(
    state: AgentState,
    node_name: str,
    success: bool,
    error: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Record node execution and return state updates."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail Node
# ─────────────────────────────────────────────────────────────────────────────


def apply_guardrails(state: AgentState) -> Dict[str, Any]:
    """Apply Phase 6 guardrails to the selected resolution.

    Delegates to GuardrailEngine.

    This is a HARD BOUNDARY:
    - LangGraph must NEVER bypass guardrails
    - LangGraph must NEVER override guardrail decisions
    - LangGraph must NEVER convert HUMAN_REVIEW to AUTO
    - LangGraph must NEVER convert UNRESOLVED to AUTO

    Stores:
    - guardrail_result
    - decision (from guardrails, NOT from LangGraph)
    - confidence
    - risk
    """
    start_time = time.perf_counter()
    node_name = "apply_guardrails"

    # Validate prerequisites
    if not state.selected_candidate:
        return _fail_node(state, node_name, "No selected candidate for guardrails", start_time)

    try:
        # Build a ResolutionEngineResult for the guardrail engine
        engine_result = _build_engine_result_for_guardrails(state)

        # Delegate to Phase 6 GuardrailEngine - REAL implementation
        guardrail_engine = GuardrailEngine()
        guardrail_result_obj = guardrail_engine.evaluate(engine_result)

        # Convert to dict for state storage
        guardrail_result = {
            "decision": guardrail_result_obj.decision.value,
            "confidence": guardrail_result_obj.confidence,
            "risk_category": guardrail_result_obj.risk_category,
            "reason_codes": guardrail_result_obj.reason_codes,
            "primary_reason": guardrail_result_obj.primary_reason,
            "passed_gates": guardrail_result_obj.passed_gates,
            "failed_gates": guardrail_result_obj.failed_gates,
            "financial_exposure_paise": guardrail_result_obj.financial_exposure_paise,
            "evidence_coverage": guardrail_result_obj.evidence_coverage,
            "evidence_consistency": guardrail_result_obj.evidence_consistency,
            "is_novel": guardrail_result_obj.is_novel,
            "has_conflict": guardrail_result_obj.has_conflict,
            "system_healthy": guardrail_result_obj.system_healthy,
        }

        updates = _record_node(state, node_name, success=True, start_time=start_time)

        # Store guardrail result
        updates["guardrail_result"] = guardrail_result

        # CRITICAL: Decision comes FROM guardrails, not from LangGraph
        # The guardrail engine is the authority
        updates["decision"] = guardrail_result["decision"]
        updates["confidence"] = guardrail_result["confidence"]
        updates["risk"] = guardrail_result["risk_category"]

        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        # FAIL-CLOSED: unexpected error → UNRESOLVED, never AUTO
        updates = _record_node(
            state, node_name,
            success=False,
            error=f"Guardrail engine error: {str(e)}",
            start_time=start_time,
        )
        updates["decision"] = "UNRESOLVED"
        updates["metadata"]["current_node"] = node_name
        return updates


def _build_engine_result_for_guardrails(state: AgentState) -> ResolutionEngineResult:
    """Build a ResolutionEngineResult for the guardrail engine from agent state."""
    candidate = state.selected_candidate or {}
    scores = state.candidate_scores or {}
    best_score = scores.get("best_score", 0.0) if scores else 0.0
    evidence_pkg = state.evidence_package or {}
    classification = state.classification or {}

    reconciliation = state.reconciliation_result or {}
    return ResolutionEngineResult(
        exception_id=state.metadata.exception_id or "",
        case_id=state.metadata.case_id or "",
        payment_id=reconciliation.get("payment_id"),
        merchant_id=reconciliation.get("merchant_id"),
        expected_amount=evidence_pkg.get("expected_amount", 0),
        actual_amount=evidence_pkg.get("actual_amount", 0),
        difference=evidence_pkg.get("difference", 0),
        status="RECOMMENDED",  # Candidate selected, so recommended
        selected_resolution=candidate.get("resolution_type"),
        # HIGH #3 FIX: The exposure guard reads adjustment from selected_candidate.
        # Since we have a candidate dict (not ResolutionProposal), we store the
        # adjustment amount in evidence so the guardrail engine can compute exposure.
        selected_candidate=None,  # Not a ResolutionProposal — keep None
        selected_score=None,  # Not needed for guardrails — exposure uses proposed_adjustment_paise
        ranked_candidates=[],
        candidate_scores=[],
        confidence=best_score,
        confidence_factors={},
        risk_category=state.risk or "LOW",
        risk_factors=[],
        explainability=None,
        rejection_reasons=[],
        deterministic_exception_type=classification.get("exception_type", "UNKNOWN"),
        ml_exception_type=classification.get("ml_exception_type"),
        classification_agreement=classification.get("agreement", True),
        evidence_explanation_status=evidence_pkg.get("explanation_status", ""),
        evidence_coverage=evidence_pkg.get("evidence_coverage", 0.0),
        evidence_consistency=evidence_pkg.get("evidence_consistency", 0.0),
        # HIGH #8 FIX: Use None as sentinel for unknown state.
        # False means verified-safe; None means insufficient information.
        # The decision matrix must treat None/unknown as HUMAN_REVIEW.
        has_conflict=evidence_pkg.get("has_conflict", None),
        is_novel=evidence_pkg.get("is_novel", None),
        missing_evidence=evidence_pkg.get("missing_evidence", []),
        # HIGH #3 FIX: Pass adjustment amount so exposure guard evaluates real exposure.
        proposed_adjustment_paise=abs(candidate.get("amount_paise", 0)),
    )


def _fail_node(
    state: AgentState,
    node_name: str,
    error_msg: str,
    start_time: float,
) -> Dict[str, Any]:
    """Create failure state update — FAIL-CLOSED, never AUTO."""
    updates = _record_node(state, node_name, success=False, error=error_msg, start_time=start_time)
    # FAIL-CLOSED: error → UNRESOLVED
    updates["decision"] = "UNRESOLVED"
    return updates

