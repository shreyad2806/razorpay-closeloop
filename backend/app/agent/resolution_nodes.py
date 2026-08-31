"""
Resolution nodes for Razorpay CloseLoop Phase 7D.

Nodes that orchestrate the resolution pipeline:
- Generate Candidates (Phase 5)
- Score Resolution (Phase 5)
- Select Best Candidate (Phase 5)

Each node delegates to existing services.
Each node has structured failure handling.
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


def _fail_node(
    state: AgentState,
    node_name: str,
    error_msg: str,
    start_time: float,
) -> Dict[str, Any]:
    """Create failure state update."""
    updates = _record_node(state, node_name, success=False, error=error_msg, start_time=start_time)
    updates["metadata"]["workflow_status"] = WorkflowStatus.FAILED.value
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Generate Candidates Node
# ─────────────────────────────────────────────────────────────────────────────


def generate_candidates(state: AgentState) -> Dict[str, Any]:
    """Generate resolution candidates from intelligence signals.

    Delegates to Phase 5 CandidateGenerator.

    Stores:
    - candidates
    """
    start_time = time.perf_counter()
    node_name = "generate_candidates"

    # Validate prerequisites
    if not state.classification:
        return _fail_node(state, node_name, "No classification available", start_time)
    if not state.evidence_package:
        return _fail_node(state, node_name, "No evidence package available", start_time)

    try:
        # Simulate candidate generation
        # In production: CandidateGenerator().generate(intelligence, package, explanation, quality)
        candidates = _simulate_candidate_generation(state)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["candidates"] = candidates
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Candidate generation failed: {str(e)}", start_time)


def _simulate_candidate_generation(state: AgentState) -> Dict[str, Any]:
    """Simulate candidate generation."""
    classification = state.classification or {}
    evidence = state.evidence_package or {}
    exc_type = classification.get("exception_type", "UNKNOWN")

    fees = evidence.get("fees", [])
    refunds = evidence.get("refunds", [])
    settlements = evidence.get("settlements", [])

    candidates = []

    # Deterministic candidate from exception type
    if exc_type == "FEE_DIFFERENCE" and fees:
        total_fees = sum(f.get("amount", 0) for f in fees)
        candidates.append({
            "candidate_id": "CAND-FEE-001",
            "resolution_type": "FEE_ADJUSTMENT",
            "amount_paise": total_fees,
            "direction": "CREDIT",
            "evidence_record_ids": [f["fee_id"] for f in fees],
            "source": "deterministic_evidence",
        })
    elif exc_type == "REFUND_ADJUSTMENT" and refunds:
        total_refunds = sum(r.get("amount", 0) for r in refunds)
        candidates.append({
            "candidate_id": "CAND-REF-001",
            "resolution_type": "REFUND_ADJUSTMENT",
            "amount_paise": total_refunds,
            "direction": "DEBIT",
            "evidence_record_ids": [r["refund_id"] for r in refunds],
            "source": "deterministic_evidence",
        })
    elif exc_type == "EXACT_MATCH":
        candidates.append({
            "candidate_id": "CAND-NONE-001",
            "resolution_type": "NO_ACTION",
            "amount_paise": 0,
            "direction": "NONE",
            "evidence_record_ids": [],
            "source": "deterministic_evidence",
        })
    else:
        # Unknown or complex — no candidate
        pass

    return {
        "exception_id": state.metadata.exception_id,
        "status": "CANDIDATES_GENERATED" if candidates else "UNRESOLVED",
        "candidates": candidates,
        "candidate_count": len(candidates),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Score Resolution Node
# ─────────────────────────────────────────────────────────────────────────────


def score_resolution(state: AgentState) -> Dict[str, Any]:
    """Score resolution candidates.

    Delegates to Phase 5 CandidateScoringService.

    Stores:
    - candidate scores
    """
    start_time = time.perf_counter()
    node_name = "score_resolution"

    if not state.candidates:
        return _fail_node(state, node_name, "No candidates available", start_time)

    try:
        # Simulate scoring
        # In production: CandidateScoringService().score_and_rank(candidates, intelligence)
        scores = _simulate_scoring(state)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["candidate_scores"] = scores
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Scoring failed: {str(e)}", start_time)


def _simulate_scoring(state: AgentState) -> Dict[str, Any]:
    """Simulate candidate scoring."""
    candidates = state.candidates or {}
    candidate_list = candidates.get("candidates", [])

    scored = []
    for c in candidate_list:
        # Simple scoring based on source
        evidence_score = 0.9 if c.get("source") == "deterministic_evidence" else 0.5
        ml_score = 0.0  # No ML in simulation
        historical_score = 0.0  # No historical in simulation
        financial_score = 0.95 if c.get("amount_paise", 0) > 0 else 1.0

        final_score = (
            0.35 * evidence_score
            + 0.20 * ml_score
            + 0.15 * historical_score
            + 0.30 * financial_score
        )

        scored.append({
            "candidate_id": c["candidate_id"],
            "evidence_score": evidence_score,
            "ml_score": ml_score,
            "historical_score": historical_score,
            "financial_consistency_score": financial_score,
            "novelty_penalty": 0.0,
            "conflict_penalty": 0.0,
            "final_score": round(final_score, 3),
        })

    # Sort by final score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "scored_candidates": scored,
        "best_score": scored[0]["final_score"] if scored else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Select Best Candidate Node
# ─────────────────────────────────────────────────────────────────────────────


def select_best_candidate(state: AgentState) -> Dict[str, Any]:
    """Select the best resolution candidate.

    Delegates to Phase 5 CandidateSelector.

    Stores:
    - selected candidate
    - confidence
    - risk
    - decision
    """
    start_time = time.perf_counter()
    node_name = "select_best_candidate"

    if not state.candidates or not state.candidate_scores:
        return _fail_node(state, node_name, "No candidates or scores available", start_time)

    try:
        # Simulate selection
        # In production: CandidateSelector().select(candidates, intelligence)
        selection = _simulate_selection(state)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["selected_candidate"] = selection["selected_candidate"]
        updates["confidence"] = selection["confidence"]
        updates["risk"] = selection["risk"]
        updates["decision"] = selection["decision"]
        updates["metadata"]["current_node"] = node_name
        updates["metadata"]["workflow_status"] = WorkflowStatus.COMPLETED.value
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Selection failed: {str(e)}", start_time)


def _simulate_selection(state: AgentState) -> Dict[str, Any]:
    """Simulate candidate selection."""
    candidates = state.candidates or {}
    scores = state.candidate_scores or {}
    candidate_list = candidates.get("candidates", [])
    scored_list = scores.get("scored_candidates", [])

    if not candidate_list or not scored_list:
        return {
            "selected_candidate": None,
            "confidence": 0.0,
            "risk": "HIGH",
            "decision": "UNRESOLVED",
        }

    # Select best scored candidate
    best_scored = scored_list[0]
    best_candidate = None
    for c in candidate_list:
        if c["candidate_id"] == best_scored["candidate_id"]:
            best_candidate = c
            break

    if not best_candidate:
        best_candidate = candidate_list[0]

    # Determine confidence and risk
    confidence = best_scored["final_score"]
    if confidence >= 0.7:
        risk = "LOW"
        decision = "AUTO"
    elif confidence >= 0.4:
        risk = "MEDIUM"
        decision = "HUMAN_REVIEW"
    else:
        risk = "HIGH"
        decision = "UNRESOLVED"

    return {
        "selected_candidate": best_candidate,
        "confidence": confidence,
        "risk": risk,
        "decision": decision,
    }
