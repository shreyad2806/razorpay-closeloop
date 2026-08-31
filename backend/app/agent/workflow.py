"""
LangGraph Workflow for Razorpay CloseLoop Phase 7K.

Complete stateful graph with conditional routing.

Structure:
  START → Load Exception → Gather Evidence → Build Graph → Classify
  → Retrieve Similar → Generate Candidates → Score → Select
  → Apply Guardrails → [Conditional Routing]
      → AUTO: Verify → Resolve → Outcome → END
      → HUMAN_REVIEW: Human Review → Verify → Resolve → Outcome → END
      → UNRESOLVED: Escalation → END
"""

import uuid
from datetime import datetime
from typing import Optional

from langgraph.graph import END, START, StateGraph

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
    route_after_guardrails,
    route_after_human_review,
    route_after_resolve,
    route_after_verification,
)
from app.agent.terminal_nodes import (
    escalation,
    human_review,
    verify_resolution,
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
    """Create the LangGraph workflow graph with complete conditional routing.

    Complete flow:
        START → load_exception → ... → apply_guardrails
            → AUTO: verify_resolution → resolve_action_boundary → record_outcome → END
            → HUMAN_REVIEW: human_review → verify_resolution → resolve_action_boundary → record_outcome → END
            → UNRESOLVED: escalation → END

    Safety boundaries:
        - Guardrails cannot be bypassed
        - Verification must pass before resolve
        - Resolve produces action request only (no execution)
        - Outcome records everything

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
            "record_outcome": "record_outcome",
            "escalation": "escalation",
        },
    )

    # ── Terminal Edges ──
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
) -> AgentState:
    """Run the workflow for a single exception."""
    initial_state = create_initial_state(
        exception_id=exception_id,
        case_id=case_id,
        workflow_id=workflow_id,
    )

    workflow = create_workflow()
    result = workflow.invoke(initial_state)

    if isinstance(result, dict):
        return AgentState(**result)
    return result
