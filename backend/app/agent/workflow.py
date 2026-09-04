"""
LangGraph Workflow for Razorpay CloseLoop Phase 8G.

Complete stateful graph with execution integration.

Structure:
  START → Load Exception → Gather Evidence → Build Graph → Classify
  → Retrieve Similar → Generate Candidates → Score → Select
  → Apply Guardrails → [Conditional Routing]
      → AUTO: Verify → Resolve → Execute → Verify Execution → Outcome → END
      → HUMAN: Human Review → Verify → Resolve → Execute → Verify Execution → Outcome → END
      → UNRESOLVED: Escalation → END
      Verification Fail → Rollback → Outcome/Escalation → END
"""

import time as _time
import uuid
from datetime import datetime
from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.core.structured_logging import (
    WorkflowEvent, agent_logger, set_correlation_ids,
)
from app.agent.execution_nodes import (
    execute_resolution,
    rollback_resolution,
    verify_execution,
)
from app.agent.guardrail_node import apply_guardrails
from app.agent.investigation_nodes import (
    build_evidence_graph,
    classify_exception,
    gather_evidence,
    retrieve_similar_cases,
)
from app.agent.nodes import load_exception
from app.agent.outcome_node import record_outcome
from app.agent.resolve_node import resolve_action_boundary
from app.agent.resolution_nodes import (
    generate_candidates,
    score_resolution,
    select_best_candidate,
)
from app.agent.routing import (
    route_after_execution,
    route_after_execution_verification,
    route_after_guardrails,
    route_after_human_review,
    route_after_resolve,
    route_after_rollback,
    route_after_verification,
)
from app.agent.terminal_nodes import (
    escalation,
    human_review,
    verify_resolution,
)
from app.agent.workflow_logging import (
    WorkflowExecutionLog,
    get_last_execution_log,
    log_graph_completed,
    log_graph_failed,
    log_graph_started,
    log_node_completed,
    log_node_started,
    log_routing_decision,
    make_observed_node,
    set_last_execution_log,
)
from app.schemas.agent_state import (
    AgentState,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Builder
# ─────────────────────────────────────────────────────────────────────────────


def create_workflow():
    """Create the LangGraph workflow graph with execution integration.

    Complete flow:
        START → load_exception → ... → apply_guardrails
            → AUTO: verify → resolve → execute → verify_execution → outcome → END
            → HUMAN: human_review → verify → resolve → execute → verify_execution → outcome → END
            → UNRESOLVED: escalation → END
            Verification Fail → rollback → outcome/escalation → END

    Returns:
        Compiled LangGraph workflow
    """
    graph = StateGraph(AgentState)

    # ── Investigation Nodes ──
    graph.add_node("load_exception", load_exception)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("build_evidence_graph", build_evidence_graph)
    graph.add_node("classify_exception", classify_exception)
    graph.add_node("retrieve_similar_cases", retrieve_similar_cases)

    # ── Resolution Nodes ──
    graph.add_node("generate_candidates", generate_candidates)
    graph.add_node("score_resolution", score_resolution)
    graph.add_node("select_best_candidate", select_best_candidate)

    # ── Guardrail Node ──
    graph.add_node("apply_guardrails", apply_guardrails)

    # ── Terminal / Flow Nodes ──
    graph.add_node("verify_resolution", verify_resolution)
    graph.add_node("human_review", human_review)
    graph.add_node("escalation", escalation)

    # ── Resolve / Action Boundary ──
    graph.add_node("resolve_action_boundary", resolve_action_boundary)

    # ── Phase 8: Execution / Verification / Rollback ──
    graph.add_node("execute_resolution", execute_resolution)
    graph.add_node("verify_execution", verify_execution)
    graph.add_node("rollback_resolution", rollback_resolution)

    # ── Outcome / Reward ──
    graph.add_node("record_outcome", record_outcome)

    # ── Linear Edges (Investigation → Resolution → Guardrails) ──
    graph.add_edge(START, "load_exception")
    graph.add_edge("load_exception", "gather_evidence")
    graph.add_edge("gather_evidence", "build_evidence_graph")
    graph.add_edge("build_evidence_graph", "classify_exception")
    graph.add_edge("classify_exception", "retrieve_similar_cases")
    graph.add_edge("retrieve_similar_cases", "generate_candidates")
    graph.add_edge("generate_candidates", "score_resolution")
    graph.add_edge("score_resolution", "select_best_candidate")
    graph.add_edge("select_best_candidate", "apply_guardrails")

    # ── Conditional Routing after Guardrails ──
    graph.add_conditional_edges(
        "apply_guardrails",
        route_after_guardrails,
        {
            "verify_resolution": "verify_resolution",
            "human_review": "human_review",
            "escalation": "escalation",
        },
    )

    # ── Conditional Routing after Verification ──
    graph.add_conditional_edges(
        "verify_resolution",
        route_after_verification,
        {
            "resolve_action_boundary": "resolve_action_boundary",
            "escalation": "escalation",
        },
    )

    # ── Conditional Routing after Human Review ──
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "verify_resolution": "verify_resolution",
            "escalation": "escalation",
        },
    )

    # ── Conditional Routing after Resolve ──
    graph.add_conditional_edges(
        "resolve_action_boundary",
        route_after_resolve,
        {
            "execute_resolution": "execute_resolution",
            "escalation": "escalation",
        },
    )

    # ── Conditional Routing after Execution ──
    graph.add_conditional_edges(
        "execute_resolution",
        route_after_execution,
        {
            "verify_execution": "verify_execution",
            "escalation": "escalation",
        },
    )

    # ── Conditional Routing after Execution Verification ──
    graph.add_conditional_edges(
        "verify_execution",
        route_after_execution_verification,
        {
            "record_outcome": "record_outcome",
            "rollback_resolution": "rollback_resolution",
        },
    )

    # ── Conditional Routing after Rollback ──
    graph.add_conditional_edges(
        "rollback_resolution",
        route_after_rollback,
        {
            "record_outcome": "record_outcome",
            "escalation": "escalation",
        },
    )

    # ── Terminal Edges ──
    graph.add_edge("record_outcome", END)
    graph.add_edge("escalation", END)

    return graph.compile()


def create_workflow_with_observability(exec_log: 'WorkflowExecutionLog'):
    """Create a workflow with every node wrapped by observability logging.

    This is functionally identical to create_workflow() but each node function
    is intercepted by make_observed_node to log:
      NODE_STARTED → node runs → NODE_COMPLETED / NODE_FAILED

    The execution_log accumulates an ordered trace of all node events.
    """
    graph = StateGraph(AgentState)

    # Wrap every node with observability — order must match create_workflow()
    graph.add_node("load_exception",          make_observed_node(load_exception,          "load_exception",          exec_log))
    graph.add_node("gather_evidence",         make_observed_node(gather_evidence,         "gather_evidence",         exec_log))
    graph.add_node("build_evidence_graph",    make_observed_node(build_evidence_graph,    "build_evidence_graph",    exec_log))
    graph.add_node("classify_exception",      make_observed_node(classify_exception,      "classify_exception",      exec_log))
    graph.add_node("retrieve_similar_cases",  make_observed_node(retrieve_similar_cases,  "retrieve_similar_cases",  exec_log))
    graph.add_node("generate_candidates",     make_observed_node(generate_candidates,     "generate_candidates",     exec_log))
    graph.add_node("score_resolution",        make_observed_node(score_resolution,        "score_resolution",        exec_log))
    graph.add_node("select_best_candidate",   make_observed_node(select_best_candidate,   "select_best_candidate",   exec_log))
    graph.add_node("apply_guardrails",        make_observed_node(apply_guardrails,        "apply_guardrails",        exec_log))
    graph.add_node("verify_resolution",       make_observed_node(verify_resolution,       "verify_resolution",       exec_log))
    graph.add_node("human_review",            make_observed_node(human_review,            "human_review",            exec_log))
    graph.add_node("escalation",              make_observed_node(escalation,              "escalation",              exec_log))
    graph.add_node("resolve_action_boundary", make_observed_node(resolve_action_boundary, "resolve_action_boundary", exec_log))
    graph.add_node("execute_resolution",      make_observed_node(execute_resolution,      "execute_resolution",      exec_log))
    graph.add_node("verify_execution",        make_observed_node(verify_execution,        "verify_execution",        exec_log))
    graph.add_node("rollback_resolution",     make_observed_node(rollback_resolution,     "rollback_resolution",     exec_log))
    graph.add_node("record_outcome",          make_observed_node(record_outcome,          "record_outcome",          exec_log))

    # ── All edges are identical to create_workflow() ──
    graph.add_edge(START, "load_exception")
    graph.add_edge("load_exception", "gather_evidence")
    graph.add_edge("gather_evidence", "build_evidence_graph")
    graph.add_edge("build_evidence_graph", "classify_exception")
    graph.add_edge("classify_exception", "retrieve_similar_cases")
    graph.add_edge("retrieve_similar_cases", "generate_candidates")
    graph.add_edge("generate_candidates", "score_resolution")
    graph.add_edge("score_resolution", "select_best_candidate")
    graph.add_edge("select_best_candidate", "apply_guardrails")

    graph.add_conditional_edges(
        "apply_guardrails",
        route_after_guardrails,
        {
            "verify_resolution": "verify_resolution",
            "human_review": "human_review",
            "escalation": "escalation",
        },
    )

    graph.add_conditional_edges(
        "verify_resolution",
        route_after_verification,
        {
            "resolve_action_boundary": "resolve_action_boundary",
            "escalation": "escalation",
        },
    )

    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "verify_resolution": "verify_resolution",
            "escalation": "escalation",
        },
    )

    graph.add_conditional_edges(
        "resolve_action_boundary",
        route_after_resolve,
        {
            "execute_resolution": "execute_resolution",
            "escalation": "escalation",
        },
    )

    graph.add_conditional_edges(
        "execute_resolution",
        route_after_execution,
        {
            "verify_execution": "verify_execution",
            "escalation": "escalation",
        },
    )

    graph.add_conditional_edges(
        "verify_execution",
        route_after_execution_verification,
        {
            "record_outcome": "record_outcome",
            "rollback_resolution": "rollback_resolution",
        },
    )

    graph.add_conditional_edges(
        "rollback_resolution",
        route_after_rollback,
        {
            "record_outcome": "record_outcome",
            "escalation": "escalation",
        },
    )

    graph.add_edge("record_outcome", END)
    graph.add_edge("escalation", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Runner
# ─────────────────────────────────────────────────────────────────────────────


def create_initial_state(
    exception_id: str,
    case_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> AgentState:
    """Create initial agent state for a workflow run."""
    if not workflow_id:
        workflow_id = f"WF-{uuid.uuid4().hex[:8].upper()}"

    metadata = WorkflowMetadata(
        workflow_id=workflow_id,
        exception_id=exception_id,
        case_id=case_id,
        workflow_status=WorkflowStatus.PENDING,
    )

    return AgentState(metadata=metadata)


def run_workflow(
    exception_id: str,
    case_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    batch_id: str = "",
) -> AgentState:
    """Run the workflow for a single exception.

    Produces detailed structured logs for every node execution:
      GRAPH_STARTED → NODE_STARTED → NODE_COMPLETED (×N) → ROUTING_DECISION → … → GRAPH_COMPLETED

    Stores the execution log so it can be inspected after the run via
    ``get_last_execution_log()``.
    """
    initial_state = create_initial_state(
        exception_id=exception_id,
        case_id=case_id,
        workflow_id=workflow_id,
    )

    wf_id = initial_state.metadata.workflow_id

    # Start the execution trace and set correlation IDs
    exec_log = log_graph_started(
        exception_id=exception_id,
        workflow_id=wf_id,
        batch_id=batch_id,
        case_id=case_id,
    )
    set_last_execution_log(exec_log)
    graph_start = exec_log.events[0]["timestamp_ms"]

    try:
        workflow = create_workflow_with_observability(exec_log)
        result = workflow.invoke(initial_state)

        if isinstance(result, dict):
            result = AgentState(**result)

        total_ms = (_time.perf_counter() - graph_start) * 1000
        log_graph_completed(result, total_ms, exec_log)
        return result

    except Exception as e:
        total_ms = (_time.perf_counter() - graph_start) * 1000
        log_graph_failed(str(e), total_ms, exec_log,
                         exception_id=exception_id, workflow_id=wf_id)
        raise
