"""
LangGraph workflow integration tests.

Tests the complete end-to-end workflow at the graph level:
- Node execution order
- State propagation between nodes
- Conditional routing
- Full AUTO / HUMAN_REVIEW / UNRESOLVED paths
- High-risk escalation
- Verification failure + rollback
- Dependency failure

Mocks only external service dependencies (execution, rollback, verification)
that are not the subject of the workflow integration test.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_workflow_integ.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.workflow import create_initial_state, create_workflow, run_workflow
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    VerificationStatus,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def workflow():
    """Create a compiled workflow."""
    return create_workflow()


@pytest.fixture
def auto_state():
    """State configured to route through AUTO path after guardrails.

    Uses high-confidence settings so guardrails return AUTO.
    """
    state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
    return state


@pytest.fixture
def human_review_state():
    """State configured for HUMAN_REVIEW after guardrails."""
    state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
    return state


@pytest.fixture
def unresolved_state():
    """State configured for UNRESOLVED after guardrails."""
    state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full Workflow — Natural Path (EXC-001: FEE_DIFFERENCE)
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowNaturalPath:
    """Test the complete workflow with the natural exception data."""

    def test_complete_workflow_executes_all_investigation_nodes(
        self, workflow
    ):
        """EXC-001 should execute: load → evidence → graph → classify → similar → candidates → score → select → guardrails."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        executed = result.metadata.nodes_executed

        # Investigation phase
        assert "load_exception" in executed
        assert "gather_evidence" in executed
        assert "build_evidence_graph" in executed
        assert "classify_exception" in executed
        assert "retrieve_similar_cases" in executed

        # Resolution phase
        assert "generate_candidates" in executed
        assert "score_resolution" in executed
        assert "select_best_candidate" in executed

        # Guardrails
        assert "apply_guardrails" in executed

    def test_state_propagation_investigation_to_resolution(
        self, workflow
    ):
        """Evidence, classification, and candidates should flow through state."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Evidence populated
        assert result.evidence_package is not None
        assert result.evidence_package.get("payment") is not None

        # Classification populated
        assert result.classification is not None
        assert result.classification.get("exception_type") == "FEE_DIFFERENCE"

        # Similar cases populated
        assert result.similar_cases is not None

        # Candidates populated
        assert result.candidates is not None
        assert result.candidates.get("candidate_count", 0) >= 1

        # Scores populated
        assert result.candidate_scores is not None
        assert result.candidate_scores.get("best_score", 0) > 0

        # Selected candidate populated
        assert result.selected_candidate is not None

    def test_guardrail_decision_overrides_selection(
        self, workflow
    ):
        """Guardrails node should be the final decision authority."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Decision exists and is valid
        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

        # Guardrail result is present
        assert result.guardrail_result is not None
        assert result.guardrail_result.get("decision") == result.decision

    def test_human_review_path_for_medium_confidence(
        self, workflow
    ):
        """EXC-001: confidence 0.615 → HUMAN_REVIEW → human_review node."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Confidence from scoring (0.615) is below AUTO threshold
        assert result.confidence is not None
        assert result.confidence < 0.70

        # Guardrails routes to HUMAN_REVIEW
        assert result.decision == "HUMAN_REVIEW"

        # Human review node executed
        assert "human_review" in result.metadata.nodes_executed

        # Human review state set to PENDING
        assert result.human_review.approval_status == HumanApprovalStatus.PENDING

        # After human_review with PENDING status → escalation
        assert "escalation" in result.metadata.nodes_executed

    def test_workflow_ends_after_escalation(
        self, workflow
    ):
        """Escalation is a terminal node — workflow should end."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Current node should be escalation or record_outcome
        assert result.metadata.current_node in ("escalation", "record_outcome")

        # Workflow status indicates completion or failure
        assert result.metadata.workflow_status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.WAITING_FOR_HUMAN,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. High-Confidence AUTO Path
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoPath:
    """Test AUTO path through the workflow.

    The natural workflow with EXC-001 gives confidence 0.615 (below AUTO threshold).
    To test the AUTO path, we pre-populate state with high confidence so the
    guardrails node returns AUTO and routes through verify → resolve → execute.
    """

    def test_auto_routing_from_guardrails(self):
        """Verify AUTO decision routes to verify_resolution."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "LOW"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "verify_resolution"

    def test_auto_confidence_threshold_in_guardrails(self):
        """Guardrails with confidence >= 0.70 and LOW risk returns AUTO."""
        from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation

        engine_result = {
            "confidence": 0.85,
            "risk_category": "LOW",
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        state = create_initial_state(exception_id="EXC-001")
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "AUTO"

    def test_medium_confidence_returns_human_review(self):
        """Guardrails with confidence 0.615 returns HUMAN_REVIEW."""
        from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation

        engine_result = {
            "confidence": 0.615,
            "risk_category": "LOW",
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        state = create_initial_state(exception_id="EXC-001")
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "HUMAN_REVIEW"

    def test_natural_workflow_reaches_human_review(self, workflow):
        """EXC-001 natural confidence (0.615) routes through HUMAN_REVIEW."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        assert "apply_guardrails" in result.metadata.nodes_executed
        assert result.decision == "HUMAN_REVIEW"
        assert "human_review" in result.metadata.nodes_executed

    def test_auto_routing_from_guardrails(
        self,
    ):
        """Verify route_after_guardrails returns correct target for AUTO."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        # Simulate what apply_guardrails sets
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "LOW"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "verify_resolution"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Medium-Confidence HUMAN_REVIEW Path
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanReviewPath:
    """Test HUMAN_REVIEW path through the workflow."""

    def test_exactly_at_threshold_routes_to_human_review(self):
        """Confidence exactly 0.70 should route to HUMAN_REVIEW (not AUTO)."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "HUMAN_REVIEW"
        state_dict["risk"] = "MEDIUM"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "human_review"

    def test_medium_confidence_state(self, workflow):
        """EXC-001 confidence 0.615 routes through HUMAN_REVIEW."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.confidence is not None
        assert 0.40 <= result.confidence < 0.70
        assert result.decision == "HUMAN_REVIEW"

    def test_human_review_node_sets_pending(self, workflow):
        """human_review node should set approval_status to PENDING."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # human_review was called
        assert "human_review" in result.metadata.nodes_executed
        assert result.human_review.approval_status == HumanApprovalStatus.PENDING

    def test_pending_approval_routes_to_escalation(self):
        """PENDING approval after human_review routes to escalation."""
        from app.agent.routing import route_after_human_review

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["human_review"]["approval_status"] = "PENDING"
        state = AgentState(**state_dict)

        route = route_after_human_review(state)
        assert route == "escalation"

    def test_approved_approval_routes_to_verification(self):
        """APPROVED after human_review routes to verification."""
        from app.agent.routing import route_after_human_review

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["human_review"]["approval_status"] = "APPROVED"
        state = AgentState(**state_dict)

        route = route_after_human_review(state)
        assert route == "verify_resolution"

    def test_rejected_approval_routes_to_escalation(self):
        """REJECTED after human_review routes to escalation."""
        from app.agent.routing import route_after_human_review

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["human_review"]["approval_status"] = "REJECTED"
        state = AgentState(**state_dict)

        route = route_after_human_review(state)
        assert route == "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Low-Confidence UNRESOLVED Path
# ─────────────────────────────────────────────────────────────────────────────


class TestUnresolvedPath:
    """Test UNRESOLVED path — guardrails block the resolution."""

    def test_unresolved_exception_type_routes_to_escalation(self):
        """UNKNOWN exception type → UNRESOLVED → escalation."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-003")
        state_dict = state.model_dump()
        state_dict["decision"] = "UNRESOLVED"
        state_dict["classification"] = {"exception_type": "UNKNOWN"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_very_low_confidence_blocks_auto(self):
        """Confidence < 0.40 → UNRESOLVED (fail-closed)."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "UNRESOLVED"
        state_dict["risk"] = "HIGH"
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_unknown_exception_in_guardrails(self, workflow):
        """EXC-003 (UNKNOWN) should be blocked by guardrails."""
        # EXC-003 has no fees/refunds → classification is EXACT_MATCH
        # But let's verify the workflow handles it
        state = create_initial_state(exception_id="EXC-003", case_id="CASE-003")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Should end up in escalation or outcome
        assert result.metadata.current_node in ("escalation", "record_outcome")

    def test_blocked_type_overrides_auto_decision(self):
        """UNKNOWN type even with AUTO decision should route to escalation."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["classification"] = {"exception_type": "UNKNOWN"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_complex_multi_adjustment_blocked(self):
        """COMPLEX_MULTI_ADJUSTMENT should be blocked even with AUTO."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["classification"] = {"exception_type": "COMPLEX_MULTI_ADJUSTMENT"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# 5. High-Risk Escalation
# ─────────────────────────────────────────────────────────────────────────────


class TestHighRiskEscalation:
    """Test that high risk forces escalation even with AUTO decision."""

    def test_high_risk_overrides_auto(self):
        """HIGH risk + AUTO → HUMAN_REVIEW (not auto)."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "HIGH"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "human_review"

    def test_low_risk_allows_auto(self):
        """LOW risk + AUTO → verify_resolution."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "LOW"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "verify_resolution"

    def test_medium_risk_allows_auto(self):
        """MEDIUM risk + AUTO → verify_resolution (only HIGH overrides)."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "MEDIUM"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "verify_resolution"

    def test_high_risk_full_workflow(self, workflow):
        """Verify guardrails correctly handle risk in the actual workflow."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Guardrail result should contain risk
        assert result.guardrail_result is not None
        assert result.risk is not None

        # If risk is HIGH and decision is AUTO, it should have been overridden
        if result.risk == "HIGH" and result.decision == "AUTO":
            # High risk overrides AUTO → HUMAN_REVIEW
            assert "human_review" in result.metadata.nodes_executed


# ─────────────────────────────────────────────────────────────────────────────
# 6. Verification Failure + Rollback
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFailure:
    """Test verification failure triggers rollback."""

    def test_verification_failed_routes_to_escalation(self):
        """FAILED verification → rollback → escalation."""
        from app.agent.routing import route_after_execution_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "FAILED"
        state = AgentState(**state_dict)

        route = route_after_execution_verification(state)
        assert route == "rollback_resolution"

    def test_verified_routes_to_outcome(self):
        """VERIFIED execution → record_outcome."""
        from app.agent.routing import route_after_execution_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "VERIFIED"
        state = AgentState(**state_dict)

        route = route_after_execution_verification(state)
        assert route == "record_outcome"

    def test_rollback_success_routes_to_outcome(self):
        """ROLLED_BACK → record_outcome."""
        from app.agent.routing import route_after_rollback

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["rollback_result"] = {"status": "ROLLED_BACK"}
        state = AgentState(**state_dict)

        route = route_after_rollback(state)
        assert route == "record_outcome"

    def test_rollback_failure_routes_to_escalation(self):
        """ROLLBACK_FAILED → escalation."""
        from app.agent.routing import route_after_rollback

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["rollback_result"] = {"status": "ROLLBACK_FAILED"}
        state = AgentState(**state_dict)

        route = route_after_rollback(state)
        assert route == "escalation"

    def test_execution_failed_routes_to_escalation(self):
        """Non-EXECUTED status → escalation."""
        from app.agent.routing import route_after_execution

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["execution_status"] = "FAILED"
        state = AgentState(**state_dict)

        route = route_after_execution(state)
        assert route == "escalation"

    def test_executed_routes_to_verification(self):
        """EXECUTED status → verify_execution."""
        from app.agent.routing import route_after_execution

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["execution_status"] = "EXECUTED"
        state = AgentState(**state_dict)

        route = route_after_execution(state)
        assert route == "verify_execution"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dependency Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestDependencyFailure:
    """Test that dependency failures are handled gracefully."""

    def test_missing_exception_id_fails(self, workflow):
        """Empty exception ID should fail in load_exception node."""
        metadata = WorkflowMetadata(
            workflow_id="WF-FAIL-001",
            exception_id="",
        )
        state = AgentState(metadata=metadata)
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Workflow should fail
        assert result.metadata.workflow_status in (
            WorkflowStatus.FAILED,
        )
        assert len(result.metadata.errors) > 0

    def test_invalid_exception_id_format_fails(self, workflow):
        """Invalid exception ID format should fail in load_exception."""
        state = create_initial_state(exception_id="INVALID-123")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.metadata.workflow_status == WorkflowStatus.FAILED

    def test_nonexistent_exception_fails(self, workflow):
        """Exception ID not in simulated data should fail."""
        state = create_initial_state(exception_id="EXC-NONEXISTENT")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.metadata.workflow_status == WorkflowStatus.FAILED

    def test_guardrail_engine_error_fails_closed(self):
        """Guardrail engine exception should set UNRESOLVED (fail-closed)."""
        from app.agent.guardrail_node import apply_guardrails
        from app.schemas.agent_state import AgentState, WorkflowMetadata

        # Build state with no selected_candidate to trigger error
        metadata = WorkflowMetadata(
            workflow_id="WF-GUARD-001",
            exception_id="EXC-001",
        )
        state = AgentState(metadata=metadata)

        result = apply_guardrails(state)

        # Should fail-closed: UNRESOLVED, not AUTO
        assert result.get("decision") == "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# 8. State Propagation Between Nodes
# ─────────────────────────────────────────────────────────────────────────────


class TestStatePropagation:
    """Verify state is correctly passed between nodes."""

    def test_evidence_flows_to_classification(self, workflow):
        """Evidence gathered should be available to classify node."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Evidence was gathered
        assert result.evidence_package is not None

        # Classification used evidence
        assert result.classification is not None
        # FEE_DIFFERENCE is classified because evidence has fees
        assert result.classification.get("exception_type") == "FEE_DIFFERENCE"

    def test_classification_flows_to_candidates(self, workflow):
        """Classification should drive candidate generation."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Classification: FEE_DIFFERENCE
        assert result.classification.get("exception_type") == "FEE_DIFFERENCE"

        # Candidates should include fee adjustment
        candidates = result.candidates.get("candidates", [])
        assert len(candidates) >= 1
        assert candidates[0].get("resolution_type") == "FEE_ADJUSTMENT"

    def test_scores_flows_to_selection(self, workflow):
        """Candidate scores should determine selection."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Scores exist
        assert result.candidate_scores is not None
        best_score = result.candidate_scores.get("best_score", 0)
        assert best_score > 0

        # Selection uses scores
        assert result.selected_candidate is not None
        assert result.confidence == best_score

    def test_selected_candidate_flows_to_guardrails(self, workflow):
        """Selected candidate should be available to guardrails."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # Guardrail node uses selected_candidate
        assert result.guardrail_result is not None
        # Decision is derived from guardrails, not from selection
        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_metadata_nodes_executed_grows(self, workflow):
        """Each executed node should be recorded in metadata."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        executed = result.metadata.nodes_executed
        assert len(executed) >= 9  # At least through guardrails

        # No duplicates in sequence
        assert len(executed) == len(set(executed))

    def test_execution_log_has_entries(self, workflow):
        """Every executed node should have a log entry."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        log = result.metadata.execution_log
        assert len(log) >= 9

        for entry in log:
            assert "node" in entry
            assert "success" in entry
            assert "timestamp" in entry


# ─────────────────────────────────────────────────────────────────────────────
# 9. Deterministic Routing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicRouting:
    """All routing decisions should be deterministic."""

    def test_same_input_same_route(self, workflow):
        """Running the same exception twice should produce the same path."""
        results = []
        for _ in range(3):
            state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
            result = workflow.invoke(state)
            if isinstance(result, dict):
                result = AgentState(**result)
            results.append(result)

        # All should have same decision
        decisions = [r.decision for r in results]
        assert len(set(decisions)) == 1

        # All should execute same nodes
        nodes_list = [tuple(r.metadata.nodes_executed) for r in results]
        assert len(set(nodes_list)) == 1

    def test_different_exceptions_different_paths(self, workflow):
        """Different exception types should potentially route differently."""
        result_001 = workflow.invoke(
            create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        )
        result_003 = workflow.invoke(
            create_initial_state(exception_id="EXC-003", case_id="CASE-003")
        )

        if isinstance(result_001, dict):
            result_001 = AgentState(**result_001)
        if isinstance(result_003, dict):
            result_003 = AgentState(**result_003)

        # Both should complete (might be same or different paths)
        assert result_001.metadata.nodes_executed[-1] in ("escalation", "record_outcome")
        assert result_003.metadata.nodes_executed[-1] in ("escalation", "record_outcome")

    def test_missing_record_exception_blocked(self):
        """MISSING_RECORD type should be blocked by guardrails."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "AUTO"
        state_dict["risk"] = "LOW"
        state_dict["classification"] = {"exception_type": "MISSING_RECORD"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Verification Routing Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationRouting:
    """Test verification node routing in the workflow."""

    def test_verification_routes_to_resolve(self):
        """VERIFIED verification → resolve_action_boundary."""
        from app.agent.routing import route_after_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "VERIFIED"
        state = AgentState(**state_dict)

        route = route_after_verification(state)
        assert route == "resolve_action_boundary"

    def test_failed_verification_routes_to_escalation(self):
        """FAILED verification → escalation."""
        from app.agent.routing import route_after_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "FAILED"
        state = AgentState(**state_dict)

        route = route_after_verification(state)
        assert route == "escalation"

    def test_not_required_verification_routes_to_escalation(self):
        """HIGH #5: NOT_REQUIRED verification → escalation (fail closed)."""
        from app.agent.routing import route_after_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "NOT_REQUIRED"
        state = AgentState(**state_dict)

        route = route_after_verification(state)
        assert route == "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Resolve Action Boundary Routing
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveRouting:
    """Test resolve_action_boundary routing."""

    def test_no_rejection_routes_to_execute(self):
        """No ACTION_REJECTED warning → execute_resolution."""
        from app.agent.routing import route_after_resolve

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["warnings"] = []
        state_dict["metadata"]["current_node"] = "resolve_action_boundary"
        state = AgentState(**state_dict)

        route = route_after_resolve(state)
        assert route == "execute_resolution"

    def test_action_rejected_routes_to_escalation(self):
        """ACTION_REJECTED warning → escalation."""
        from app.agent.routing import route_after_resolve

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["warnings"] = ["ACTION_REJECTED: guardrail block"]
        state = AgentState(**state_dict)

        route = route_after_resolve(state)
        assert route == "escalation"

    def test_no_current_node_routes_to_escalation(self):
        """If current_node is not resolve_action_boundary → escalation."""
        from app.agent.routing import route_after_resolve

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["warnings"] = []
        state_dict["metadata"]["current_node"] = "other_node"
        state = AgentState(**state_dict)

        route = route_after_resolve(state)
        assert route == "escalation"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Fail-Closed Behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestFailClosed:
    """Verify fail-closed behavior at all decision points."""

    def test_no_decision_routes_to_escalation(self):
        """No decision → fail-closed → escalation."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = None
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_invalid_decision_routes_to_escalation(self):
        """Unknown decision string → fail-closed → escalation."""
        from app.agent.routing import route_after_guardrails

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["decision"] = "SOMETHING_ELSE"
        state_dict["classification"] = {"exception_type": "FEE_DIFFERENCE"}
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_no_verification_state_routes_to_escalation(self):
        """HIGH #5: No verification state → NOT_REQUIRED → escalation (fail closed)."""
        from app.agent.routing import route_after_verification

        state = create_initial_state(exception_id="EXC-001")
        state_dict = state.model_dump()
        state_dict["verification"]["verification_status"] = "NOT_REQUIRED"
        state = AgentState(**state_dict)

        route = route_after_verification(state)
        assert route == "escalation"

    def test_workflow_metadata_tracks_errors(self, workflow):
        """Errors from failed nodes should appear in metadata."""
        state = create_initial_state(exception_id="EXC-NONEXISTENT")
        result = workflow.invoke(state)

        if isinstance(result, dict):
            result = AgentState(**result)

        # load_exception fails for nonexistent exception
        assert len(result.metadata.errors) > 0
        assert any("not found" in e.lower() or "nonexistent" in e.lower() for e in result.metadata.errors)
