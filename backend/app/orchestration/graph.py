"""
Controlled LangGraph orchestration for Razorpay CloseLoop Phase 11.

The graph is an orchestrator only. It never computes financial truth, never
mutates financial records, never bypasses policy/authorization and never closes
an exception:

    START
      → load_exception
      → collect_evidence
      → get_intelligence          (advisory)
      → get_historical_context    (advisory)
      → generate_resolution       (structured proposal)
      → policy_evaluation         (Phase 8 policy/guardrails)
          ├─ AUTO          → execute_if_authorized   (Phase 8 → Phase 9)
          ├─ HUMAN_REVIEW  → human_review  → END (WAITING_FOR_HUMAN_APPROVAL)
          └─ otherwise      → escalation
      → post_execution_verification  (Phase 10 — sole closure arbiter)
      → finalize → END

Execution is bounded: every node increments a hard step counter and the runner
sets an explicit LangGraph ``recursion_limit``. There are no graph cycles.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from mcp.client import MCPClient
from app.schemas.orchestration import (
    DEFAULT_MAX_STEPS,
    OrchestrationStage as Stage,
    OrchestrationState,
    OrchestrationStatus as Status,
)

#: LangGraph recursion limit for a single orchestration run.
GRAPH_RECURSION_LIMIT = DEFAULT_MAX_STEPS + 5


class OrchestrationStepLimitError(RuntimeError):
    """Raised when the graph exceeds its hard node-execution bound."""


# ─────────────────────────────────────────────────────────────────────────────
# Node plumbing
# ─────────────────────────────────────────────────────────────────────────────


def _tick(state: OrchestrationState) -> int:
    """Increment the step counter and fail closed past the bound."""
    steps = state.steps + 1
    if steps > state.max_steps:
        raise OrchestrationStepLimitError(
            f"orchestration exceeded max_steps={state.max_steps}"
        )
    return steps


def _call(
    client: MCPClient,
    state: OrchestrationState,
    tool_name: str,
) -> Dict[str, Any]:
    return client.call_tool(
        tool_name=tool_name,
        parameters={"exception_id": state.exception_id},
        workflow_id=state.workflow_id,
        agent_id="phase11-orchestrator",
        exception_id=state.exception_id,
    )


def _invocations(state: OrchestrationState, tool_name: str) -> list:
    return list(state.tool_invocations) + [tool_name]


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


def make_nodes(client: MCPClient) -> Dict[str, Callable[[OrchestrationState], Dict[str, Any]]]:
    """Build the node functions bound to a specific MCP client."""

    def load_exception(state: OrchestrationState) -> Dict[str, Any]:
        result = _call(client, state, "get_exception_context")
        data = result.get("data") or {}
        if not result.get("success") or not data.get("found"):
            error = data.get("error") or result.get("error") or "exception not found"
            return {
                "steps": _tick(state),
                "stage": Stage.ESCALATION,
                "status": Status.FAILED,
                "errors": list(state.errors) + [error],
                "tool_invocations": _invocations(state, "get_exception_context"),
            }
        ctx = data["exception"]
        return {
            "steps": _tick(state),
            "stage": Stage.LOAD_EXCEPTION,
            "status": Status.RUNNING,
            "exception_context": ctx,
            "case_id": ctx.get("case_id"),
            "payment_id": ctx.get("payment_id"),
            "merchant_id": ctx.get("merchant_id"),
            "tool_invocations": _invocations(state, "get_exception_context"),
        }

    def collect_evidence(state: OrchestrationState) -> Dict[str, Any]:
        result = _call(client, state, "get_evidence")
        data = result.get("data") or {}
        return {
            "steps": _tick(state),
            "stage": Stage.COLLECT_EVIDENCE,
            "evidence": data if result.get("success") else {"found": False},
            "tool_invocations": _invocations(state, "get_evidence"),
        }

    def get_intelligence(state: OrchestrationState) -> Dict[str, Any]:
        result = _call(client, state, "classify_exception")
        data = result.get("data") or {}
        return {
            "steps": _tick(state),
            "stage": Stage.GET_INTELLIGENCE,
            "intelligence": data,
            "tool_invocations": _invocations(state, "classify_exception"),
        }

    def get_historical_context(state: OrchestrationState) -> Dict[str, Any]:
        result = _call(client, state, "retrieve_historical_cases")
        data = result.get("data") or {}
        return {
            "steps": _tick(state),
            "stage": Stage.GET_HISTORICAL_CONTEXT,
            "historical_context": data,
            "tool_invocations": _invocations(state, "retrieve_historical_cases"),
        }

    def generate_resolution(state: OrchestrationState) -> Dict[str, Any]:
        result = _call(client, state, "generate_resolution_proposals")
        data = result.get("data") or {}
        proposals = data.get("proposals") or []
        return {
            "steps": _tick(state),
            "stage": Stage.GENERATE_RESOLUTION,
            "proposals": proposals,
            "selected_proposal": proposals[0] if proposals else None,
            "tool_invocations": _invocations(state, "generate_resolution_proposals"),
        }

    def policy_evaluation(state: OrchestrationState) -> Dict[str, Any]:
        if not state.selected_proposal:
            return {
                "steps": _tick(state),
                "stage": Stage.POLICY_EVALUATION,
                "authorization": {"decision": "UNRESOLVED", "primary_reason": "no proposal"},
                "tool_invocations": _invocations(state, "evaluate_resolution_policy"),
            }
        result = client.call_tool(
            tool_name="evaluate_resolution_policy",
            parameters={"proposal": state.selected_proposal},
            workflow_id=state.workflow_id,
            agent_id="phase11-orchestrator",
            exception_id=state.exception_id,
        )
        data = result.get("data") or {}
        authorization = {
            "decision": data.get("decision", "UNRESOLVED"),
            "risk_category": data.get("risk_category"),
            "reason_codes": data.get("reason_codes", []),
            "primary_reason": data.get("primary_reason"),
            "confidence": data.get("confidence"),
            "required_approval": data.get("required_approval", False),
        }
        return {
            "steps": _tick(state),
            "stage": Stage.POLICY_EVALUATION,
            "authorization": authorization,
            "tool_invocations": _invocations(state, "evaluate_resolution_policy"),
        }

    def execute_if_authorized(state: OrchestrationState) -> Dict[str, Any]:
        exec_params: Dict[str, Any] = {
            "exception_id": state.exception_id,
            "proposal": state.selected_proposal,
            "workflow_id": state.workflow_id,
            "idempotency_key": f"key-{state.workflow_id}-{state.exception_id}",
        }
        if state.approval_record:
            exec_params["approval_record"] = state.approval_record
        result = client.call_tool(
            tool_name="request_authorized_execution",
            parameters=exec_params,
            workflow_id=state.workflow_id,
            agent_id="phase11-orchestrator",
            exception_id=state.exception_id,
        )
        data = result.get("data") or {}
        return {
            "steps": _tick(state),
            "stage": Stage.EXECUTE_IF_AUTHORIZED,
            "execution": data.get("execution"),
            "authorization": data.get("authorization", state.authorization),
            "warnings": list(state.warnings)
            + ([data["error"]] if data.get("error") else []),
            "tool_invocations": _invocations(state, "request_authorized_execution"),
        }

    def post_execution_verification(state: OrchestrationState) -> Dict[str, Any]:
        result = client.call_tool(
            tool_name="verify_post_execution_reconciliation",
            parameters={
                "exception_id": state.exception_id,
                "execution": state.execution,
            },
            workflow_id=state.workflow_id,
            agent_id="phase11-orchestrator",
            exception_id=state.exception_id,
        )
        data = result.get("data") or {}
        return {
            "steps": _tick(state),
            "stage": Stage.POST_EXECUTION_VERIFICATION,
            "verification": data,
            "audit_refs": list(state.audit_refs) + list(data.get("audit_event_ids", [])),
            "tool_invocations": _invocations(
                state, "verify_post_execution_reconciliation"
            ),
        }

    def human_review(state: OrchestrationState) -> Dict[str, Any]:
        return {
            "steps": _tick(state),
            "stage": Stage.HUMAN_REVIEW,
            "status": Status.WAITING_FOR_HUMAN_APPROVAL,
            "final_status": Status.WAITING_FOR_HUMAN_APPROVAL.value,
        }

    def escalation(state: OrchestrationState) -> Dict[str, Any]:
        status = (
            Status.FAILED
            if state.stage == Stage.ESCALATION and state.status == Status.FAILED
            else Status.ESCALATED
        )
        return {
            "steps": _tick(state),
            "stage": Stage.ESCALATION,
            "status": status,
            "final_status": status.value,
        }

    def finalize(state: OrchestrationState) -> Dict[str, Any]:
        verification = state.verification or {}
        if verification.get("financially_verified"):
            status = Status.VERIFIED
        else:
            status = Status.NOT_VERIFIED
        return {
            "steps": _tick(state),
            "stage": Stage.FINALIZE,
            "status": status,
            "final_status": status.value,
        }

    return {
        "load_exception": load_exception,
        "collect_evidence": collect_evidence,
        "get_intelligence": get_intelligence,
        "get_historical_context": get_historical_context,
        "generate_resolution": generate_resolution,
        "policy_evaluation": policy_evaluation,
        "execute_if_authorized": execute_if_authorized,
        "post_execution_verification": post_execution_verification,
        "human_review": human_review,
        "escalation": escalation,
        "finalize": finalize,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing (fail-closed)
# ─────────────────────────────────────────────────────────────────────────────


def route_after_load(state: OrchestrationState) -> str:
    if state.status == Status.FAILED or not state.exception_context:
        return "escalation"
    return "collect_evidence"


def route_after_policy(state: OrchestrationState) -> str:
    decision = (state.authorization or {}).get("decision")
    if decision == "AUTO":
        return "execute_if_authorized"
    if decision == "HUMAN_REVIEW":
        # A valid human approval lets a later invocation continue; without one
        # the graph stops before any financial execution.
        if state.approval_record:
            return "execute_if_authorized"
        return "human_review"
    return "escalation"


def route_after_execute(state: OrchestrationState) -> str:
    execution = state.execution or {}
    if execution.get("status") == "EXECUTED":
        return "post_execution_verification"
    warning = " ".join(state.warnings)
    if "HUMAN_REVIEW" in warning:
        return "human_review"
    return "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_orchestration_graph(client: MCPClient):
    """Compile the bounded controlled orchestration graph."""
    nodes = make_nodes(client)
    graph = StateGraph(OrchestrationState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "load_exception")
    graph.add_conditional_edges(
        "load_exception",
        route_after_load,
        {"collect_evidence": "collect_evidence", "escalation": "escalation"},
    )
    graph.add_edge("collect_evidence", "get_intelligence")
    graph.add_edge("get_intelligence", "get_historical_context")
    graph.add_edge("get_historical_context", "generate_resolution")
    graph.add_edge("generate_resolution", "policy_evaluation")
    graph.add_conditional_edges(
        "policy_evaluation",
        route_after_policy,
        {
            "execute_if_authorized": "execute_if_authorized",
            "human_review": "human_review",
            "escalation": "escalation",
        },
    )
    graph.add_conditional_edges(
        "execute_if_authorized",
        route_after_execute,
        {
            "post_execution_verification": "post_execution_verification",
            "human_review": "human_review",
            "escalation": "escalation",
        },
    )
    graph.add_edge("post_execution_verification", "finalize")

    graph.add_edge("finalize", END)
    graph.add_edge("human_review", END)
    graph.add_edge("escalation", END)

    return graph.compile()


def create_initial_state(
    exception_id: str,
    workflow_id: Optional[str] = None,
    case_id: Optional[str] = None,
    approval_record: Optional[Dict[str, Any]] = None,
) -> OrchestrationState:
    return OrchestrationState(
        workflow_id=workflow_id or f"WF-{uuid.uuid4().hex[:8].upper()}",
        exception_id=exception_id,
        case_id=case_id,
        approval_record=approval_record,
        status=Status.PENDING,
    )


def run_orchestration(
    exception_id: str,
    client: MCPClient,
    *,
    workflow_id: Optional[str] = None,
    case_id: Optional[str] = None,
    approval_record: Optional[Dict[str, Any]] = None,
    recursion_limit: int = GRAPH_RECURSION_LIMIT,
) -> OrchestrationState:
    """Run the controlled orchestration once and return the final state."""
    initial = create_initial_state(
        exception_id, workflow_id=workflow_id, case_id=case_id,
        approval_record=approval_record,
    )
    graph = build_orchestration_graph(client)
    result = graph.invoke(initial, config={"recursion_limit": recursion_limit})
    if isinstance(result, dict):
        result = OrchestrationState(**result)
    return result


__all__ = [
    "build_orchestration_graph",
    "create_initial_state",
    "run_orchestration",
    "make_nodes",
    "route_after_load",
    "route_after_policy",
    "route_after_execute",
    "OrchestrationStepLimitError",
    "GRAPH_RECURSION_LIMIT",
]
