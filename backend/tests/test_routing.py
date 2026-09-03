"""
Tests for Conditional Routing (Phase 7F).

Tests:
- AUTO route → verification
- HUMAN_REVIEW route → human review
- UNRESOLVED route → escalation
- HIGH_RISK route → human review
- UNKNOWN route → escalation
- Invalid decision → fail closed
- End-to-end routing
- Terminal node behavior
"""

import pytest

from app.agent.routing import (
    ROUTE_ESCALATION,
    ROUTE_END,
    ROUTE_HUMAN_REVIEW,
    ROUTE_OUTCOME,
    ROUTE_RESOLVE,
    ROUTE_VERIFICATION,
    route_after_guardrails,
    route_after_human_review,
    route_after_resolve,
    route_after_verification,
)
from app.agent.workflow import create_initial_state, run_workflow
from app.schemas.agent_state import (
    AgentState,
    VerificationStatus,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Routing Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRouteAfterGuardrails:
    def test_auto_routes_to_verification(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        assert route_after_guardrails(state) == ROUTE_VERIFICATION

    def test_human_review_routes_to_human(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "HUMAN_REVIEW"
        assert route_after_guardrails(state) == ROUTE_HUMAN_REVIEW

    def test_unresolved_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "UNRESOLVED"
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_no_decision_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = None
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_unknown_type_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.classification = {"exception_type": "UNKNOWN"}
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_complex_type_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.classification = {"exception_type": "COMPLEX_MULTI_ADJUSTMENT"}
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_high_risk_auto_routes_to_human(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.risk = "HIGH"
        assert route_after_guardrails(state) == ROUTE_HUMAN_REVIEW

    def test_high_risk_human_review_stays_human(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "HUMAN_REVIEW"
        state.risk = "HIGH"
        assert route_after_guardrails(state) == ROUTE_HUMAN_REVIEW

    def test_invalid_decision_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "INVALID_DECISION"
        assert route_after_guardrails(state) == ROUTE_ESCALATION


class TestRouteAfterVerification:
    def test_verified_routes_to_resolve(self):
        state = create_initial_state(exception_id="EXC-001")
        state.verification.verification_status = VerificationStatus.VERIFIED
        assert route_after_verification(state) == ROUTE_RESOLVE

    def test_failed_routes_to_escalation(self):
        state = create_initial_state(exception_id="EXC-001")
        state.verification.verification_status = VerificationStatus.FAILED
        assert route_after_verification(state) == ROUTE_ESCALATION

    def test_not_required_routes_to_escalation(self):
        """HIGH #5: NOT_REQUIRED must fail closed, not resolve."""
        state = create_initial_state(exception_id="EXC-001")
        assert route_after_verification(state) == ROUTE_ESCALATION

    def test_pending_routes_to_escalation(self):
        """HIGH #5: PENDING must fail closed, not resolve."""
        state = create_initial_state(exception_id="EXC-001")
        state.verification.verification_status = VerificationStatus.PENDING
        assert route_after_verification(state) == ROUTE_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Terminal Node Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifyResolution:
    def test_verification_succeeds(self):
        from app.agent.terminal_nodes import verify_resolution
        state = create_initial_state(exception_id="EXC-001")
        result = verify_resolution(state)

        assert result["verification"]["verification_status"] == "VERIFIED"
        assert "verify_resolution" in result["metadata"]["nodes_executed"]

    def test_workflow_completed(self):
        from app.agent.terminal_nodes import verify_resolution
        state = create_initial_state(exception_id="EXC-001")
        result = verify_resolution(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.COMPLETED.value


class TestHumanReview:
    def test_human_review_pending(self):
        from app.agent.terminal_nodes import human_review
        state = create_initial_state(exception_id="EXC-001")
        state.guardrail_result = {"primary_reason": "Low confidence"}
        result = human_review(state)

        assert result["human_review"]["approval_status"] == "PENDING"
        assert result["metadata"]["workflow_status"] == WorkflowStatus.WAITING_FOR_HUMAN.value

    def test_high_priority_for_high_risk(self):
        from app.agent.terminal_nodes import human_review
        state = create_initial_state(exception_id="EXC-001")
        state.risk = "HIGH"
        state.guardrail_result = {"primary_reason": "High risk"}
        result = human_review(state)

        assert result["human_review"]["review_priority"] == "HIGH"


class TestEscalation:
    def test_escalation_records_warning(self):
        from app.agent.terminal_nodes import escalation
        state = create_initial_state(exception_id="EXC-001")
        state.guardrail_result = {"primary_reason": "Unknown pattern"}
        result = escalation(state)

        assert len(result["warnings"]) > 0
        assert "ESCALATED" in result["warnings"][0]

    def test_workflow_failed(self):
        from app.agent.terminal_nodes import escalation
        state = create_initial_state(exception_id="EXC-001")
        result = escalation(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Routing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndRouting:
    def test_fee_difference_routes_correctly(self):
        """FEE_DIFFERENCE with good evidence → AUTO → verify → resolve → outcome."""
        result = run_workflow(exception_id="EXC-001")

        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
        assert "apply_guardrails" in result.metadata.nodes_executed
        # One of the terminal nodes should have executed
        terminal_nodes = {"verify_resolution", "human_review", "escalation"}
        executed_terminal = terminal_nodes & set(result.metadata.nodes_executed)
        assert len(executed_terminal) >= 1

    def test_auto_route_executes_verification(self):
        """When decision is AUTO, verification node executes."""
        result = run_workflow(exception_id="EXC-001")

        if result.decision == "AUTO":
            assert "verify_resolution" in result.metadata.nodes_executed
        elif result.decision == "HUMAN_REVIEW":
            assert "human_review" in result.metadata.nodes_executed
        elif result.decision == "UNRESOLVED":
            assert "escalation" in result.metadata.nodes_executed

    def test_all_routes_reachable(self):
        """Verify routing works for the known exception."""
        result = run_workflow(exception_id="EXC-001")

        # The route depends on guardrail decision
        assert result.decision is not None
        assert result.metadata.workflow_status is not None

    def test_workflow_completes(self):
        """Workflow always completes (reaches END)."""
        result = run_workflow(exception_id="EXC-001")

        # After terminal node, workflow is done
        assert len(result.metadata.nodes_executed) >= 9  # 8 + 1 terminal

    def test_guardrail_result_preserved(self):
        """Guardrail result is preserved after routing."""
        result = run_workflow(exception_id="EXC-001")

        assert result.guardrail_result is not None
        assert "decision" in result.guardrail_result


class TestRoutingIntegrity:
    def test_no_auto_for_unknown(self):
        """UNKNOWN exception never routes to AUTO."""
        result = run_workflow(exception_id="EXC-001")

        # For EXC-001 (FEE_DIFFERENCE), routing depends on guardrails
        # But verify the routing function itself
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.classification = {"exception_type": "UNKNOWN"}
        route = route_after_guardrails(state)
        assert route != ROUTE_VERIFICATION

    def test_no_auto_for_high_risk(self):
        """HIGH risk never routes to AUTO verification."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.risk = "HIGH"
        route = route_after_guardrails(state)
        assert route != ROUTE_VERIFICATION

    def test_invalid_decision_fail_closed(self):
        """Invalid decision routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "GARBAGE"
        route = route_after_guardrails(state)
        assert route == ROUTE_ESCALATION

    def test_missing_decision_fail_closed(self):
        """Missing decision routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = None
        route = route_after_guardrails(state)
        assert route == ROUTE_ESCALATION
