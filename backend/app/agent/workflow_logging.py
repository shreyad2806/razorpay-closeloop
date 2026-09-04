"""
LangGraph Workflow Observability for Razorpay CloseLoop.

Wraps workflow execution with detailed node-level structured logging.

Logs produced:
  GRAPH_STARTED        — workflow begins
  NODE_STARTED         — each node begins execution
  NODE_COMPLETED       — each node completes
  NODE_FAILED          — each node fails
  NODE_TIMING          — per-node timing summary
  ROUTING_DECISION     — conditional routing outcome
  GRAPH_COMPLETED      — workflow ends (with decision, timing, path)
  GRAPH_FAILED         — workflow crashes

Every entry includes: run_id, exception_id, batch_id, workflow_id, node_name,
duration_ms, decision, risk, confidence, guardrail result, verification result.
"""

import time
from typing import Any, Dict, List, Optional

from app.core.structured_logging import (
    StructuredLogger,
    WorkflowEvent,
    generate_run_id,
    set_correlation_ids,
)
from app.schemas.agent_state import AgentState


# Dedicated LangGraph observability logger
_graph_logger = StructuredLogger("closeloop.langgraph", component="langgraph")

# Ordered log of all events for this workflow run (in-memory, per run)
GraphNodeRecord = Dict[str, Any]


class WorkflowExecutionLog:
    """Captures an ordered trace of every node execution in a workflow run.

    Produced automatically by `log_workflow_execution`. Readable after
    the workflow completes to reconstruct the exact execution path.
    """

    def __init__(self) -> None:
        self.events: List[GraphNodeRecord] = []
        self.node_timings: Dict[str, float] = {}  # node_name -> elapsed_ms
        self.total_ms: float = 0.0
        self.final_decision: str = ""
        self.final_risk: str = ""
        self.final_confidence: float = 0.0
        self.nodes_in_order: List[str] = []
        self.failed_nodes: List[str] = []
        self.guardrail_decision: str = ""
        self.verification_result: str = ""

    def record(self, event: GraphNodeRecord) -> None:
        self.events.append(event)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self.events),
            "nodes_in_order": self.nodes_in_order,
            "node_timings_ms": self.node_timings,
            "total_ms": round(self.total_ms, 2),
            "failed_nodes": self.failed_nodes,
            "final_decision": self.final_decision,
            "final_risk": self.final_risk,
            "final_confidence": self.final_confidence,
            "guardrail_decision": self.guardrail_decision,
            "verification_result": self.verification_result,
        }


# Module-level store so workflow_logging.log can be read after execution
_last_execution_log: Optional[WorkflowExecutionLog] = None


def get_last_execution_log() -> Optional[WorkflowExecutionLog]:
    """Return the execution log from the most recent run_workflow call."""
    return _last_execution_log


def set_last_execution_log(log: WorkflowExecutionLog) -> None:
    """Store the execution log from the most recent run."""
    global _last_execution_log
    _last_execution_log = log


# ─────────────────────────────────────────────────────────────────────────────
# Node name → human-readable label mapping
# ─────────────────────────────────────────────────────────────────────────────

_NODE_LABELS: Dict[str, str] = {
    "load_exception":        "Load Exception",
    "gather_evidence":       "Gather Evidence",
    "build_evidence_graph":  "Build Evidence Graph",
    "classify_exception":    "Classify Exception",
    "retrieve_similar_cases":"Retrieve Similar Cases",
    "generate_candidates":   "Generate Candidates",
    "score_resolution":      "Score Resolution",
    "select_best_candidate": "Select Best Candidate",
    "apply_guardrails":      "Apply Guardrails",
    "verify_resolution":     "Verify Resolution",
    "human_review":          "Human Review",
    "escalation":            "Escalation",
    "resolve_action_boundary":"Resolve Action Boundary",
    "execute_resolution":    "Execute Resolution",
    "verify_execution":      "Verify Execution",
    "rollback_resolution":   "Rollback Resolution",
    "record_outcome":        "Record Outcome",
}


# ─────────────────────────────────────────────────────────────────────────────
# State snapshot helpers — extract key safety fields without exposing data
# ─────────────────────────────────────────────────────────────────────────────

def _safe_state_snapshot(state: AgentState) -> Dict[str, Any]:
    """Extract only observability-safe fields from workflow state.

    Does NOT include full evidence, candidates, or financial records.
    Includes only: IDs, decision fields, timing, node count.
    """
    candidate = state.selected_candidate or {}
    ver = state.verification
    guardrail = state.guardrail_result or {}
    return {
        "exception_id": state.metadata.exception_id,
        "workflow_id": state.metadata.workflow_id,
        "nodes_executed_count": len(state.metadata.nodes_executed),
        "decision": state.decision,
        "confidence": state.confidence,
        "risk": state.risk,
        "candidate_type": candidate.get("resolution_type"),
        "candidate_amount_paise": candidate.get("amount_paise"),
        "guardrail_decision": guardrail.get("decision"),
        "guardrail_exposure_paise": guardrail.get("financial_exposure_paise"),
        "verification_status": ver.verification_status.value if ver else "NOT_SET",
        "workflow_status": state.metadata.workflow_status.value,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core observability functions — called by workflow.py
# ─────────────────────────────────────────────────────────────────────────────

def log_graph_started(
    exception_id: str,
    workflow_id: str,
    batch_id: str = "",
    case_id: Optional[str] = None,
) -> WorkflowExecutionLog:
    """Log GRAPH_STARTED and create the execution log for this run."""
    run_id = generate_run_id()
    set_correlation_ids(
        exception_id=exception_id,
        workflow_id=workflow_id,
        batch_id=batch_id,
    )

    execution_log = WorkflowExecutionLog()
    execution_log.record({
        "event": WorkflowEvent.GRAPH_STARTED.value,
        "timestamp_ms": time.perf_counter(),
    })

    _graph_logger.info(
        WorkflowEvent.GRAPH_STARTED.value,
        f"LangGraph workflow started for exception {exception_id}",
        run_id=run_id,
        exception_id=exception_id,
        workflow_id=workflow_id,
        batch_id=batch_id,
        case_id=case_id,
    )

    return execution_log


def log_node_started(
    node_name: str,
    state: AgentState,
    execution_log: WorkflowExecutionLog,
) -> float:
    """Log NODE_STARTED and return start_time for the node."""
    start_time = time.perf_counter()
    snapshot = _safe_state_snapshot(state)

    execution_log.record({
        "event": WorkflowEvent.NODE_STARTED.value,
        "node": node_name,
        "label": _NODE_LABELS.get(node_name, node_name),
    })

    _graph_logger.info(
        WorkflowEvent.NODE_STARTED.value,
        f"Node started: {_NODE_LABELS.get(node_name, node_name)}",
        node_name=node_name,
        node_label=_NODE_LABELS.get(node_name, node_name),
        nodes_so_far=len(state.metadata.nodes_executed),
        current_decision=snapshot["decision"],
        current_confidence=snapshot["confidence"],
        current_risk=snapshot["risk"],
        verification=snapshot["verification_status"],
    )

    return start_time


def log_node_completed(
    node_name: str,
    state: AgentState,
    start_time: float,
    execution_log: WorkflowExecutionLog,
    *,
    node_output_summary: Optional[Dict[str, Any]] = None,
) -> float:
    """Log NODE_COMPLETED with timing and state changes.

    Returns elapsed_ms for convenience.
    """
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    snapshot = _safe_state_snapshot(state)

    execution_log.record({
        "event": WorkflowEvent.NODE_COMPLETED.value,
        "node": node_name,
        "label": _NODE_LABELS.get(node_name, node_name),
        "elapsed_ms": round(elapsed_ms, 2),
        "decision": snapshot["decision"],
        "confidence": snapshot["confidence"],
        "risk": snapshot["risk"],
        "verification": snapshot["verification_status"],
    })

    execution_log.node_timings[node_name] = round(elapsed_ms, 2)
    execution_log.nodes_in_order.append(node_name)

    _graph_logger.info(
        WorkflowEvent.NODE_COMPLETED.value,
        f"Node completed: {_NODE_LABELS.get(node_name, node_name)} ({round(elapsed_ms, 1)}ms)",
        node_name=node_name,
        duration_ms=round(elapsed_ms, 2),
        decision=snapshot["decision"],
        confidence=snapshot["confidence"],
        risk=snapshot["risk"],
        verification=snapshot["verification_status"],
        workflow_status=snapshot["workflow_status"],
    )

    return elapsed_ms


def log_node_failed(
    node_name: str,
    error: str,
    start_time: float,
    execution_log: WorkflowExecutionLog,
) -> float:
    """Log NODE_FAILED with timing and error details."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    execution_log.record({
        "event": WorkflowEvent.NODE_FAILED.value,
        "node": node_name,
        "label": _NODE_LABELS.get(node_name, node_name),
        "elapsed_ms": round(elapsed_ms, 2),
        "error": error,
    })
    execution_log.failed_nodes.append(node_name)

    _graph_logger.error(
        WorkflowEvent.NODE_FAILED.value,
        f"Node failed: {_NODE_LABELS.get(node_name, node_name)} — {error[:120]}",
        node_name=node_name,
        duration_ms=round(elapsed_ms, 2),
        error_type="node_failure",
        error_message=error[:300],
    )

    return elapsed_ms


def log_routing_decision(
    from_node: str,
    to_node: str,
    decision: str,
    state: AgentState,
    execution_log: WorkflowExecutionLog,
) -> None:
    """Log ROUTING_DECISION — conditional edge chosen."""
    execution_log.record({
        "event": WorkflowEvent.ROUTING_DECISION.value,
        "from_node": from_node,
        "to_node": to_node,
        "decision": decision,
    })

    _graph_logger.info(
        WorkflowEvent.ROUTING_DECISION.value,
        f"Routing: {_NODE_LABELS.get(from_node, from_node)} → {_NODE_LABELS.get(to_node, to_node)} (decision={decision})",
        from_node=from_node,
        to_node=to_node,
        decision=decision,
        risk=state.risk,
        confidence=state.confidence,
    )


def log_graph_completed(
    state: AgentState,
    total_elapsed_ms: float,
    execution_log: WorkflowExecutionLog,
) -> None:
    """Log GRAPH_COMPLETED with final decision and timing summary."""
    snapshot = _safe_state_snapshot(state)
    guardrail = state.guardrail_result or {}

    execution_log.total_ms = round(total_elapsed_ms, 2)
    execution_log.final_decision = snapshot["decision"]
    execution_log.final_risk = snapshot["risk"]
    execution_log.final_confidence = snapshot["confidence"]
    execution_log.guardrail_decision = guardrail.get("decision", "")
    execution_log.verification_result = snapshot["verification_status"]

    execution_log.record({
        "event": WorkflowEvent.GRAPH_COMPLETED.value,
        "total_ms": execution_log.total_ms,
        "nodes_in_order": execution_log.nodes_in_order,
        "failed_nodes": execution_log.failed_nodes,
    })

    _graph_logger.success(
        WorkflowEvent.GRAPH_COMPLETED.value,
        f"Workflow complete: decision={snapshot['decision']} "
        f"risk={snapshot['risk']} confidence={snapshot['confidence']} "
        f"nodes={len(execution_log.nodes_in_order)} "
        f"total={round(total_elapsed_ms, 1)}ms",
        duration_ms=round(total_elapsed_ms, 2),
        decision=snapshot["decision"],
        confidence=snapshot["confidence"],
        risk=snapshot["risk"],
        nodes_in_order=execution_log.nodes_in_order,
        failed_nodes=execution_log.failed_nodes,
        guardrail_decision=guardrail.get("decision"),
        guardrail_exposure_paise=guardrail.get("financial_exposure_paise"),
        verification_status=snapshot["verification_status"],
        total_node_timings_ms=execution_log.node_timings,
    )


def log_graph_failed(
    error: str,
    total_elapsed_ms: float,
    execution_log: WorkflowExecutionLog,
    exception_id: str = "",
    workflow_id: str = "",
) -> None:
    """Log GRAPH_FAILED when the workflow crashes."""
    execution_log.total_ms = round(total_elapsed_ms, 2)

    execution_log.record({
        "event": WorkflowEvent.GRAPH_FAILED.value,
        "total_ms": execution_log.total_ms,
        "error": error,
        "failed_nodes": execution_log.failed_nodes,
    })

    _graph_logger.failure(
        WorkflowEvent.GRAPH_FAILED.value,
        f"Workflow failed after {round(total_elapsed_ms, 1)}ms: {error[:150]}",
        duration_ms=round(total_elapsed_ms, 2),
        exception_id=exception_id,
        workflow_id=workflow_id,
        error_type="graph_crash",
        error_message=error[:500],
        failed_nodes=execution_log.failed_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node Wrappers — wrap LangGraph nodes with observability logging
# ─────────────────────────────────────────────────────────────────────────────

def make_observed_node(
    node_fn,
    node_name: str,
    execution_log: WorkflowExecutionLog,
):
    """Wrap a LangGraph node function with structured lifecycle logging.

    Logs NODE_STARTED → (original function runs) → NODE_COMPLETED / NODE_FAILED.
    Does not alter the node's return value or state mutations.
    Only observes — never interferes.

    Args:
        node_fn: The original node function (state -> dict).
        node_name: Canonical node name (e.g., "gather_evidence").
        execution_log: The WorkflowExecutionLog accumulating events.

    Returns:
        A new function with the same signature as node_fn.
    """
    def _observed_node(state: AgentState):
        # Log NODE_STARTED with current state snapshot
        start_time = log_node_started(node_name, state, execution_log)
        try:
            result = node_fn(state)
        except Exception as exc:
            # Log failure, re-raise so LangGraph handles it as before
            log_node_failed(node_name, str(exc), start_time, execution_log)
            raise

        # Reconstruct state from the result to log NODE_COMPLETED
        # (the result dict is applied to state by LangGraph after the node)
        log_node_completed(node_name, state, start_time, execution_log)
        return result

    # Preserve the original function's name and docstring for debugging
    _observed_node.__name__ = f"observed_{node_fn.__name__}"
    _observed_node.__doc__ = node_fn.__doc__
    return _observed_node
