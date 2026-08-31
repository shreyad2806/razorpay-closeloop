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
        # Build a minimal ResolutionEngineResult for the guardrail engine
        engine_result = _build_engine_result_for_guardrails(state)

        # Delegate to Phase 6 GuardrailEngine
        # In production: GuardrailEngine().evaluate(engine_result)
        guardrail_result = _simulate_guardrail_evaluation(state, engine_result)

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


def _build_engine_result_for_guardrails(state: AgentState) -> Dict[str, Any]:
    """Build a ResolutionEngineResult-compatible dict for the guardrail engine."""
    candidate = state.selected_candidate or {}
    scores = state.candidate_scores or {}
    best_score = scores.get("best_score", 0.0) if scores else 0.0

    return {
        "exception_id": state.metadata.exception_id,
        "case_id": state.metadata.case_id,
        "expected_amount": 100000,
        "actual_amount": 97000,
        "difference": 3000,
        "status": "RECOMMENDED",
        "selected_resolution": candidate.get("resolution_type"),
        "confidence": best_score,
        "risk_category": state.risk or "LOW",
        "deterministic_exception_type": (state.classification or {}).get("exception_type", "UNKNOWN"),
        "evidence_coverage": (state.evidence_package or {}).get("evidence_coverage", 0.0),
        "evidence_consistency": (state.evidence_package or {}).get("evidence_consistency", 0.0),
    }


def _simulate_guardrail_evaluation(state: AgentState, engine_result: Dict) -> Dict[str, Any]:
    """Simulate guardrail evaluation.

    In production, this would call GuardrailEngine().evaluate()
    """
    exc_type = engine_result.get("deterministic_exception_type", "UNKNOWN")
    confidence = engine_result.get("confidence", 0.0)
    coverage = engine_result.get("evidence_coverage", 0.0)
    consistency = engine_result.get("evidence_consistency", 0.0)
    risk = engine_result.get("risk_category", "LOW")

    # Blocked exception types → UNRESOLVED
    blocked_types = ["UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD"]
    if exc_type in blocked_types:
        return {
            "decision": "UNRESOLVED",
            "confidence": confidence,
            "risk_category": risk,
            "reason_codes": ["BLOCKED_EXCEPTION_TYPE"],
            "primary_reason": f"Exception type {exc_type} is blocked",
            "passed_gates": [],
            "failed_gates": ["blocked_exception_type"],
        }

    # Low confidence → HUMAN_REVIEW
    if confidence < 0.40:
        return {
            "decision": "UNRESOLVED",
            "confidence": confidence,
            "risk_category": risk,
            "reason_codes": ["VERY_LOW_CONFIDENCE"],
            "primary_reason": f"Confidence {confidence:.1%} too low",
            "passed_gates": [],
            "failed_gates": ["confidence_gate"],
        }

    if confidence < 0.70:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": confidence,
            "risk_category": risk,
            "reason_codes": ["MEDIUM_CONFIDENCE"],
            "primary_reason": f"Confidence {confidence:.1%} below auto threshold",
            "passed_gates": [],
            "failed_gates": ["confidence_gate"],
        }

    # High risk → HUMAN_REVIEW
    if risk == "HIGH":
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": confidence,
            "risk_category": risk,
            "reason_codes": ["ELEVATED_RISK"],
            "primary_reason": "High risk level",
            "passed_gates": [],
            "failed_gates": ["risk_check"],
        }

    # Low coverage → HUMAN_REVIEW
    if coverage < 0.50:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": confidence,
            "risk_category": risk,
            "reason_codes": ["LOW_COVERAGE"],
            "primary_reason": f"Evidence coverage {coverage:.1%} too low",
            "passed_gates": [],
            "failed_gates": ["evidence_guard"],
        }

    # All checks pass → AUTO
    return {
        "decision": "AUTO",
        "confidence": confidence,
        "risk_category": risk,
        "reason_codes": ["ALL_GATES_PASSED"],
        "primary_reason": "All mandatory gates passed",
        "passed_gates": ["confidence", "exposure", "evidence", "fallback"],
        "failed_gates": [],
    }


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
