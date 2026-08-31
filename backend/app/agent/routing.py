"""
Centralized Routing Policy for Razorpay CloseLoop Phase 7K.

Defines deterministic routing rules for the complete workflow.

All routing logic is centralized here.
Do not scatter decision strings throughout the graph.

Complete flow:
  Guardrails → [AUTO / HUMAN_REVIEW / UNRESOLVED]
  AUTO → verify_resolution → [verified → resolve → outcome / failed → escalation]
  HUMAN_REVIEW → human_review → [approved → verification → resolve → outcome / rejected → escalation]
  UNRESOLVED → escalation → END
"""

from app.schemas.agent_state import AgentState, HumanApprovalStatus


# ─────────────────────────────────────────────────────────────────────────────
# Route Targets
# ─────────────────────────────────────────────────────────────────────────────

ROUTE_VERIFICATION = "verify_resolution"
ROUTE_HUMAN_REVIEW = "human_review"
ROUTE_ESCALATION = "escalation"
ROUTE_RESOLVE = "resolve_action_boundary"
ROUTE_OUTCOME = "record_outcome"
ROUTE_END = "__end__"


# ─────────────────────────────────────────────────────────────────────────────
# Route After Guardrails
# ─────────────────────────────────────────────────────────────────────────────


def route_after_guardrails(state: AgentState) -> str:
    """Route based on guardrail decision.

    Rules:
    - AUTO → verification
    - HUMAN_REVIEW → human approval
    - UNRESOLVED → escalation
    - HIGH RISK → human review (never auto)
    - UNKNOWN / blocked types → escalation (defense in depth)
    - Invalid → fail closed (escalation)

    This is the SINGLE source of routing truth for guardrail output.
    """
    decision = state.decision

    # No decision → fail closed
    if not decision:
        return ROUTE_ESCALATION

    # UNKNOWN / blocked types → escalation (defense in depth)
    classification = state.classification or {}
    exc_type = classification.get("exception_type")
    if exc_type in ("UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD"):
        if decision != "UNRESOLVED":
            return ROUTE_ESCALATION

    # HIGH RISK → never auto
    risk = state.risk or "LOW"
    if risk == "HIGH" and decision == "AUTO":
        return ROUTE_HUMAN_REVIEW

    # Standard routing
    if decision == "AUTO":
        return ROUTE_VERIFICATION
    elif decision == "HUMAN_REVIEW":
        return ROUTE_HUMAN_REVIEW
    elif decision == "UNRESOLVED":
        return ROUTE_ESCALATION
    else:
        # Invalid decision → fail closed
        return ROUTE_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Route After Verification
# ─────────────────────────────────────────────────────────────────────────────


def route_after_verification(state: AgentState) -> str:
    """Route after verification node.

    Rules:
    - VERIFIED → resolve_action_boundary
    - FAILED → escalation
    - NOT_REQUIRED → resolve_action_boundary (shouldn't happen, but fail open to resolve)
    """
    verification = state.verification
    status = verification.verification_status.value if verification else "NOT_REQUIRED"

    if status == "VERIFIED":
        return ROUTE_RESOLVE
    elif status == "FAILED":
        return ROUTE_ESCALATION
    else:
        # NOT_REQUIRED or PENDING → resolve (defensive)
        return ROUTE_RESOLVE


# ─────────────────────────────────────────────────────────────────────────────
# Route After Human Review
# ─────────────────────────────────────────────────────────────────────────────


def route_after_human_review(state: AgentState) -> str:
    """Route after human_review node.

    In the synchronous flow, human_review just sets WAITING_FOR_HUMAN.
    For the integrated workflow, the human_review node records the review.
    After recording:
    - If approval was already processed (APPROVED) → verification
    - If PENDING → escalation (async flow would pause here)
    - If REJECTED → escalation
    """
    approval = state.human_review.approval_status

    if approval == HumanApprovalStatus.APPROVED:
        return ROUTE_VERIFICATION
    elif approval == HumanApprovalStatus.REJECTED:
        return ROUTE_ESCALATION
    else:
        # PENDING → escalation (would pause in async flow)
        return ROUTE_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Route After Resolve
# ─────────────────────────────────────────────────────────────────────────────


def route_after_resolve(state: AgentState) -> str:
    """Route after resolve_action_boundary node.

    Rules:
    - Success → execute_resolution
    - Rejection (ACTION_REJECTED in warnings) → escalation
    """
    # Check for rejection warnings
    for w in state.warnings:
        if "ACTION_REJECTED" in w:
            return ROUTE_ESCALATION

    # Check if resolve node succeeded
    if state.metadata.current_node == "resolve_action_boundary":
        return "execute_resolution"

    # Default → escalation (fail closed)
    return ROUTE_ESCALATION


def route_after_execution(state: AgentState) -> str:
    """Route after execute_resolution node.

    Rules:
    - Executed → verify_execution
    - Failed → escalation
    """
    exec_status = state.execution_status
    if exec_status == "EXECUTED":
        return "verify_execution"
    else:
        return ROUTE_ESCALATION


def route_after_execution_verification(state: AgentState) -> str:
    """Route after verify_execution node.

    Rules:
    - VERIFIED → record_outcome (SUCCESS)
    - FAILED → rollback_resolution
    """
    ver_state = state.verification
    status = ver_state.verification_status.value if ver_state else "NOT_REQUIRED"

    if status == "VERIFIED":
        return ROUTE_OUTCOME
    elif status == "FAILED":
        return "rollback_resolution"
    else:
        return ROUTE_ESCALATION


def route_after_rollback(state: AgentState) -> str:
    """Route after rollback_resolution node.

    Rules:
    - ROLLED_BACK → record_outcome
    - ROLLBACK_FAILED → escalation
    """
    rollback = state.rollback_result
    if rollback:
        rb_status = rollback.get("status")
        if rb_status == "ROLLED_BACK":
            return ROUTE_OUTCOME

    return ROUTE_ESCALATION
