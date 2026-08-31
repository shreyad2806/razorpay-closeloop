"""
Tests for Guardrail Node (Phase 7E).

Tests:
- AUTO decision
- HUMAN_REVIEW decision
- UNRESOLVED decision
- high exposure
- low confidence
- conflicting evidence
- unknown pattern
- guardrail failure
- bypass protection verification
- end-to-end workflow
"""

import pytest

from app.agent.guardrail_node import apply_guardrails
from app.agent.workflow import create_initial_state, run_workflow
from app.schemas.agent_state import AgentState, WorkflowStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_state_with_selection(
    exc_type="FEE_DIFFERENCE",
    confidence=0.75,
    risk="LOW",
    coverage=0.90,
    consistency=0.85,
):
    """Create state with selected candidate ready for guardrails."""
    state = create_initial_state(exception_id="EXC-001")
    state.classification = {
        "exception_id": "EXC-001",
        "exception_type": exc_type,
        "confidence": confidence,
    }
    state.evidence_package = {
        "exception_id": "EXC-001",
        "evidence_coverage": coverage,
        "evidence_consistency": consistency,
    }
    state.candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-FEE-001",
                "resolution_type": "FEE_ADJUSTMENT",
                "amount_paise": 3000,
            }
        ],
    }
    state.candidate_scores = {
        "scored_candidates": [
            {"candidate_id": "CAND-FEE-001", "final_score": confidence}
        ],
        "best_score": confidence,
    }
    state.selected_candidate = {
        "candidate_id": "CAND-FEE-001",
        "resolution_type": "FEE_ADJUSTMENT",
        "amount_paise": 3000,
    }
    state.confidence = confidence
    state.risk = risk
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail Node Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailNode:
    def test_auto_decision(self):
        """High confidence + known type + good evidence → AUTO."""
        state = _make_state_with_selection(
            confidence=0.80, risk="LOW", coverage=0.90, consistency=0.85
        )
        result = apply_guardrails(state)

        assert result["decision"] == "AUTO"
        assert result["guardrail_result"] is not None
        assert "ALL_GATES_PASSED" in result["guardrail_result"]["reason_codes"]

    def test_human_review_low_confidence(self):
        """Medium confidence → HUMAN_REVIEW."""
        state = _make_state_with_selection(
            confidence=0.55, risk="LOW", coverage=0.90, consistency=0.85
        )
        result = apply_guardrails(state)

        assert result["decision"] == "HUMAN_REVIEW"
        assert "MEDIUM_CONFIDENCE" in result["guardrail_result"]["reason_codes"]

    def test_unresolved_very_low_confidence(self):
        """Very low confidence → UNRESOLVED."""
        state = _make_state_with_selection(
            confidence=0.30, risk="LOW", coverage=0.90, consistency=0.85
        )
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"
        assert "VERY_LOW_CONFIDENCE" in result["guardrail_result"]["reason_codes"]

    def test_unresolved_unknown_type(self):
        """Unknown exception type → UNRESOLVED."""
        state = _make_state_with_selection(
            exc_type="UNKNOWN", confidence=0.95, risk="LOW"
        )
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"
        assert "BLOCKED_EXCEPTION_TYPE" in result["guardrail_result"]["reason_codes"]

    def test_unresolved_complex_type(self):
        """Complex multi-adjustment → UNRESOLVED."""
        state = _make_state_with_selection(
            exc_type="COMPLEX_MULTI_ADJUSTMENT", confidence=0.90, risk="LOW"
        )
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"

    def test_human_review_high_risk(self):
        """High risk → HUMAN_REVIEW."""
        state = _make_state_with_selection(
            confidence=0.80, risk="HIGH", coverage=0.90, consistency=0.85
        )
        result = apply_guardrails(state)

        assert result["decision"] == "HUMAN_REVIEW"
        assert "ELEVATED_RISK" in result["guardrail_result"]["reason_codes"]

    def test_human_review_low_coverage(self):
        """Low evidence coverage → HUMAN_REVIEW."""
        state = _make_state_with_selection(
            confidence=0.80, risk="LOW", coverage=0.30, consistency=0.85
        )
        result = apply_guardrails(state)

        assert result["decision"] == "HUMAN_REVIEW"
        assert "LOW_COVERAGE" in result["guardrail_result"]["reason_codes"]

    def test_missing_candidate_fails(self):
        """No selected candidate → failure."""
        state = create_initial_state(exception_id="EXC-001")
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"
        assert any("No selected candidate" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = _make_state_with_selection(confidence=0.80)
        result = apply_guardrails(state)

        assert "apply_guardrails" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Bypass Protection Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBypassProtection:
    def test_guardrail_result_stored(self):
        """Guardrail result is stored in state."""
        state = _make_state_with_selection(confidence=0.80)
        result = apply_guardrails(state)

        assert result["guardrail_result"] is not None
        assert "decision" in result["guardrail_result"]

    def test_decision_from_guardrails(self):
        """Decision comes FROM guardrails, not from LangGraph."""
        state = _make_state_with_selection(
            exc_type="UNKNOWN", confidence=0.99
        )
        result = apply_guardrails(state)

        # Even with 99% confidence, UNKNOWN → UNRESOLVED
        assert result["decision"] == "UNRESOLVED"

    def test_cannot_override_blocked_type(self):
        """Cannot override blocked exception type."""
        for blocked in ["UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD"]:
            state = _make_state_with_selection(
                exc_type=blocked, confidence=0.99, risk="LOW"
            )
            result = apply_guardrails(state)
            assert result["decision"] == "UNRESOLVED", f"{blocked} should be UNRESOLVED"

    def test_cannot_override_low_confidence(self):
        """Cannot override low confidence."""
        state = _make_state_with_selection(confidence=0.20, risk="LOW")
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"

    def test_cannot_override_high_risk(self):
        """Cannot override high risk."""
        state = _make_state_with_selection(confidence=0.80, risk="HIGH")
        result = apply_guardrails(state)

        assert result["decision"] == "HUMAN_REVIEW"

    def test_fail_closed_on_error(self):
        """Unexpected error → UNRESOLVED, never AUTO."""
        state = create_initial_state(exception_id="EXC-001")
        # No selected candidate → triggers error path
        result = apply_guardrails(state)

        assert result["decision"] == "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndWorkflow:
    def test_full_workflow_with_guardrails(self):
        """Test complete workflow including guardrails."""
        result = run_workflow(exception_id="EXC-001")

        assert isinstance(result, AgentState)
        assert result.decision is not None
        assert result.guardrail_result is not None
        assert "apply_guardrails" in result.metadata.nodes_executed

    def test_all_nodes_executed(self):
        """Test all nodes including guardrails executed."""
        result = run_workflow(exception_id="EXC-001")

        expected_nodes = [
            "load_exception",
            "gather_evidence",
            "build_evidence_graph",
            "classify_exception",
            "retrieve_similar_cases",
            "generate_candidates",
            "score_resolution",
            "select_best_candidate",
            "apply_guardrails",
        ]
        for node in expected_nodes:
            assert node in result.metadata.nodes_executed

    def test_workflow_completes(self):
        """Test workflow completes with guardrails."""
        result = run_workflow(exception_id="EXC-001")

        # Status depends on routing: COMPLETED, WAITING_FOR_HUMAN, or FAILED
        assert result.metadata.workflow_status is not None

    def test_decision_is_valid(self):
        """Test decision is one of the allowed values."""
        result = run_workflow(exception_id="EXC-001")

        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_guardrail_result_has_fields(self):
        """Test guardrail result has required fields."""
        result = run_workflow(exception_id="EXC-001")

        gr = result.guardrail_result
        assert "decision" in gr
        assert "confidence" in gr
        assert "risk_category" in gr
        assert "reason_codes" in gr

    def test_exception_id_preserved(self):
        """Test exception ID preserved through guardrails."""
        result = run_workflow(exception_id="EXC-001")

        assert result.metadata.exception_id == "EXC-001"
        assert result.guardrail_result is not None
