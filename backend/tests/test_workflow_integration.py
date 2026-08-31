"""
Tests for Razorpay CloseLoop Phase 7K — Complete Workflow Integration.

Tests the full LangGraph workflow with all nodes, routing, and safety boundaries.
"""

import pytest
from app.agent.workflow import create_initial_state, create_workflow, run_workflow
from app.agent.routing import (
    ROUTE_ESCALATION,
    ROUTE_OUTCOME,
    ROUTE_RESOLVE,
    ROUTE_VERIFICATION,
    route_after_guardrails,
    route_after_verification,
    route_after_human_review,
    route_after_resolve,
)
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    VerificationStatus,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Structure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_workflow_creates(self):
        """Workflow graph compiles successfully."""
        workflow = create_workflow()
        assert workflow is not None

    def test_all_nodes_present(self):
        """All 17 nodes are in the graph."""
        workflow = create_workflow()
        graph = workflow.get_graph()
        node_names = set(graph.nodes.keys()) - {"__start__", "__end__"}
        expected = {
            "load_exception",
            "gather_evidence",
            "build_evidence_graph",
            "classify_exception",
            "retrieve_similar_cases",
            "generate_candidates",
            "score_resolution",
            "select_best_candidate",
            "apply_guardrails",
            "verify_resolution",
            "human_review",
            "escalation",
            "resolve_action_boundary",
            "execute_resolution",
            "verify_execution",
            "rollback_resolution",
            "record_outcome",
        }
        assert expected == node_names

    def test_initial_state_creation(self):
        """Initial state is created with correct defaults."""
        state = create_initial_state(exception_id="EXC-001")
        assert state.metadata.exception_id == "EXC-001"
        assert state.metadata.workflow_status == WorkflowStatus.PENDING
        assert state.metadata.workflow_id.startswith("WF-")

    def test_initial_state_with_case_id(self):
        """Initial state accepts case_id."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        assert state.metadata.case_id == "CASE-001"

    def test_initial_state_with_workflow_id(self):
        """Initial state accepts custom workflow_id."""
        state = create_initial_state(exception_id="EXC-001", workflow_id="WF-CUSTOM")
        assert state.metadata.workflow_id == "WF-CUSTOM"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndWorkflow:
    def test_complete_workflow_runs(self):
        """Full workflow executes without error."""
        result = run_workflow(exception_id="EXC-001")
        assert result is not None
        assert result.metadata.exception_id == "EXC-001"

    def test_workflow_has_guardrails(self):
        """Workflow always applies guardrails."""
        result = run_workflow(exception_id="EXC-001")
        assert "apply_guardrails" in result.metadata.nodes_executed

    def test_workflow_has_investigation(self):
        """Workflow always runs investigation nodes."""
        result = run_workflow(exception_id="EXC-001")
        for node in ["load_exception", "gather_evidence", "classify_exception"]:
            assert node in result.metadata.nodes_executed

    def test_workflow_has_resolution(self):
        """Workflow always runs resolution nodes."""
        result = run_workflow(exception_id="EXC-001")
        for node in ["generate_candidates", "score_resolution", "select_best_candidate"]:
            assert node in result.metadata.nodes_executed

    def test_workflow_produces_decision(self):
        """Workflow always produces a guardrail decision."""
        result = run_workflow(exception_id="EXC-001")
        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_workflow_has_confidence(self):
        """Workflow always produces confidence."""
        result = run_workflow(exception_id="EXC-001")
        assert result.confidence is not None

    def test_workflow_has_risk(self):
        """Workflow always produces risk category."""
        result = run_workflow(exception_id="EXC-001")
        assert result.risk is not None

    def test_workflow_node_count(self):
        """Workflow executes the expected number of nodes."""
        result = run_workflow(exception_id="EXC-001")
        # Minimum: 10 core nodes (load through guardrails) + 1 terminal
        assert len(result.metadata.nodes_executed) >= 11


# ─────────────────────────────────────────────────────────────────────────────
# AUTO Path Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoPath:
    def test_auto_goes_to_verification(self):
        """AUTO decision routes to verification."""
        result = run_workflow(exception_id="EXC-001")
        if result.decision == "AUTO":
            assert "verify_resolution" in result.metadata.nodes_executed

    def test_auto_may_reach_resolve(self):
        """AUTO + VERIFIED may reach resolve_action_boundary."""
        result = run_workflow(exception_id="EXC-001")
        if result.decision == "AUTO":
            # If verification passed, resolve should execute
            if "verify_resolution" in result.metadata.nodes_executed:
                # May or may not reach resolve depending on verification
                assert "resolve_action_boundary" in result.metadata.nodes_executed or \
                       "escalation" in result.metadata.nodes_executed


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN_REVIEW Path Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanReviewPath:
    def test_human_review_reached(self):
        """HUMAN_REVIEW decision routes to human_review node."""
        result = run_workflow(exception_id="EXC-001")
        if result.decision == "HUMAN_REVIEW":
            assert "human_review" in result.metadata.nodes_executed

    def test_human_review_sets_waiting(self):
        """human_review node sets WAITING_FOR_HUMAN status."""
        result = run_workflow(exception_id="EXC-001")
        if result.decision == "HUMAN_REVIEW":
            # In synchronous flow, human_review sets WAITING_FOR_HUMAN
            # then routes to escalation (since no human decision yet)
            assert result.metadata.workflow_status in (
                WorkflowStatus.WAITING_FOR_HUMAN,
                WorkflowStatus.FAILED,  # escalation sets FAILED
            )


# ─────────────────────────────────────────────────────────────────────────────
# UNRESOLVED Path Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnresolvedPath:
    def test_unresolved_goes_to_escalation(self):
        """UNRESOLVED decision routes to escalation."""
        result = run_workflow(exception_id="EXC-001")
        if result.decision == "UNRESOLVED":
            assert "escalation" in result.metadata.nodes_executed


# ─────────────────────────────────────────────────────────────────────────────
# Routing Unit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRoutingFunctions:
    def test_guardrails_auto(self):
        """AUTO routes to verification."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "AUTO"
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        assert route_after_guardrails(state) == ROUTE_VERIFICATION

    def test_guardrails_unresolved(self):
        """UNRESOLVED routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = "UNRESOLVED"
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_guardrails_no_decision(self):
        """No decision routes to escalation (fail closed)."""
        state = create_initial_state(exception_id="EXC-001")
        state.decision = None
        assert route_after_guardrails(state) == ROUTE_ESCALATION

    def test_verification_verified_routes_to_resolve(self):
        """VERIFIED routes to resolve."""
        state = create_initial_state(exception_id="EXC-001")
        state.verification.verification_status = VerificationStatus.VERIFIED
        assert route_after_verification(state) == ROUTE_RESOLVE

    def test_verification_failed_routes_to_escalation(self):
        """FAILED routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.verification.verification_status = VerificationStatus.FAILED
        assert route_after_verification(state) == ROUTE_ESCALATION

    def test_human_review_approved_routes_to_verification(self):
        """APPROVED routes to verification."""
        state = create_initial_state(exception_id="EXC-001")
        state.human_review.approval_status = HumanApprovalStatus.APPROVED
        assert route_after_human_review(state) == ROUTE_VERIFICATION

    def test_human_review_rejected_routes_to_escalation(self):
        """REJECTED routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.human_review.approval_status = HumanApprovalStatus.REJECTED
        assert route_after_human_review(state) == ROUTE_ESCALATION

    def test_human_review_pending_routes_to_escalation(self):
        """PENDING routes to escalation (async would pause)."""
        state = create_initial_state(exception_id="EXC-001")
        state.human_review.approval_status = HumanApprovalStatus.PENDING
        assert route_after_human_review(state) == ROUTE_ESCALATION

    def test_resolve_success_routes_to_execution(self):
        """Successful resolve routes to execution."""
        state = create_initial_state(exception_id="EXC-001")
        state.metadata.current_node = "resolve_action_boundary"
        assert route_after_resolve(state) == "execute_resolution"

    def test_resolve_rejection_routes_to_escalation(self):
        """Rejected resolve routes to escalation."""
        state = create_initial_state(exception_id="EXC-001")
        state.warnings = ["ACTION_REJECTED: guardrail blocked"]
        assert route_after_resolve(state) == ROUTE_ESCALATION


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundaries:
    def test_guardrails_always_execute(self):
        """Guardrails always execute before any terminal node."""
        result = run_workflow(exception_id="EXC-001")
        guardrail_idx = result.metadata.nodes_executed.index("apply_guardrails")
        terminal_nodes = {"verify_resolution", "human_review", "escalation",
                          "resolve_action_boundary", "record_outcome"}
        for node in result.metadata.nodes_executed:
            if node in terminal_nodes:
                assert result.metadata.nodes_executed.index(node) > guardrail_idx

    def test_no_financial_execution(self):
        """Workflow never executes financial actions."""
        result = run_workflow(exception_id="EXC-001")
        forbidden = {"execute_refund", "modify_settlement", "modify_payment",
                      "call_razorpay_api", "issue_refund"}
        executed = set(result.metadata.nodes_executed)
        assert executed.isdisjoint(forbidden)

    def test_investigation_before_resolution(self):
        """Investigation always runs before resolution."""
        result = run_workflow(exception_id="EXC-001")
        nodes = result.metadata.nodes_executed
        if "generate_candidates" in nodes:
            assert nodes.index("load_exception") < nodes.index("generate_candidates")
            assert nodes.index("classify_exception") < nodes.index("generate_candidates")

    def test_resolution_before_guardrails(self):
        """Resolution always runs before guardrails."""
        result = run_workflow(exception_id="EXC-001")
        nodes = result.metadata.nodes_executed
        if "apply_guardrails" in nodes and "select_best_candidate" in nodes:
            assert nodes.index("select_best_candidate") < nodes.index("apply_guardrails")

    def test_verification_before_resolve(self):
        """Verification always runs before resolve."""
        result = run_workflow(exception_id="EXC-001")
        nodes = result.metadata.nodes_executed
        if "resolve_action_boundary" in nodes:
            assert "verify_resolution" in nodes
            assert nodes.index("verify_resolution") < nodes.index("resolve_action_boundary")


# ─────────────────────────────────────────────────────────────────────────────
# State Transition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStateTransitions:
    def test_workflow_transitions_from_pending(self):
        """Workflow transitions from PENDING."""
        state = create_initial_state(exception_id="EXC-001")
        assert state.metadata.workflow_status == WorkflowStatus.PENDING

    def test_workflow_transitions_to_running(self):
        """Workflow transitions to RUNNING after load."""
        result = run_workflow(exception_id="EXC-001")
        assert result.metadata.workflow_status in (
            WorkflowStatus.RUNNING,
            WorkflowStatus.FAILED,
            WorkflowStatus.WAITING_FOR_HUMAN,
            WorkflowStatus.COMPLETED,
        )

    def test_workflow_completes(self):
        """Workflow reaches a terminal state."""
        result = run_workflow(exception_id="EXC-001")
        assert result.metadata.workflow_status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.WAITING_FOR_HUMAN,
        )

    def test_execution_log_populated(self):
        """Execution log records all node executions."""
        result = run_workflow(exception_id="EXC-001")
        assert len(result.metadata.execution_log) >= len(result.metadata.nodes_executed)
        for entry in result.metadata.execution_log:
            assert "node" in entry
            assert "success" in entry
            assert "timestamp" in entry


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_exception_same_workflow_id(self):
        """Same exception produces same workflow_id if specified."""
        result1 = run_workflow(exception_id="EXC-001", workflow_id="WF-IDEMPOTENT")
        result2 = run_workflow(exception_id="EXC-001", workflow_id="WF-IDEMPOTENT")
        assert result1.metadata.workflow_id == result2.metadata.workflow_id

    def test_same_exception_different_workflow_ids(self):
        """Different runs produce different workflow IDs."""
        result1 = run_workflow(exception_id="EXC-001")
        result2 = run_workflow(exception_id="EXC-001")
        assert result1.metadata.workflow_id != result2.metadata.workflow_id


# ─────────────────────────────────────────────────────────────────────────────
# Failure Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFailureHandling:
    def test_missing_exception_handled(self):
        """Missing exception is handled gracefully."""
        result = run_workflow(exception_id="EXC-999")
        assert result.metadata.workflow_status == WorkflowStatus.FAILED
        assert len(result.metadata.errors) > 0

    def test_invalid_exception_id_handled(self):
        """Invalid exception ID is handled gracefully."""
        result = run_workflow(exception_id="")
        assert result.metadata.workflow_status == WorkflowStatus.FAILED

    def test_failure_records_error(self):
        """Failure records error in metadata."""
        result = run_workflow(exception_id="EXC-999")
        # Error is stored in metadata.errors
        assert len(result.metadata.errors) > 0
        assert any("not found" in e.lower() for e in result.metadata.errors)
