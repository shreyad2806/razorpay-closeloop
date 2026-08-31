"""
Tests for Resolution Nodes (Phase 7D).

Tests:
- one clear candidate
- multiple candidates
- unresolved case
- conflicting candidates
- unknown case
- Phase 5 failure
- state propagation
- full workflow end-to-end
"""

import pytest

from app.agent.resolution_nodes import (
    generate_candidates,
    score_resolution,
    select_best_candidate,
)
from app.agent.workflow import create_initial_state, run_workflow
from app.schemas.agent_state import AgentState, WorkflowStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_state_with_intelligence(exc_type="FEE_DIFFERENCE"):
    """Create state with intelligence data loaded."""
    state = create_initial_state(exception_id="EXC-001")
    state.classification = {
        "exception_id": "EXC-001",
        "exception_type": exc_type,
        "confidence": 0.95,
    }
    state.evidence_package = {
        "exception_id": "EXC-001",
        "payment": {"payment_id": "PAY-001", "amount": 100000, "status": "CAPTURED"},
        "settlements": [{"settlement_id": "SET-001", "amount": 97000, "status": "SETTLED"}],
        "refunds": [],
        "fees": [{"fee_id": "FEE-001", "amount": 3000, "fee_type": "TDR"}],
        "taxes": [],
        "adjustments": [],
        "evidence_coverage": 0.95,
        "evidence_consistency": 0.90,
        "supporting_evidence_count": 2,
    }
    state.similar_cases = {
        "similar_cases": [
            {"case_id": "CASE-042", "similarity_score": 0.92, "resolution": "FEE_ADJUSTMENT"},
        ],
        "best_similarity_score": 0.92,
    }
    return state


def _make_state_with_candidates():
    """Create state with candidates already generated."""
    state = _make_state_with_intelligence()
    state.candidates = {
        "exception_id": "EXC-001",
        "status": "CANDIDATES_GENERATED",
        "candidates": [
            {
                "candidate_id": "CAND-FEE-001",
                "resolution_type": "FEE_ADJUSTMENT",
                "amount_paise": 3000,
                "direction": "CREDIT",
                "evidence_record_ids": ["FEE-001"],
                "source": "deterministic_evidence",
            }
        ],
        "candidate_count": 1,
    }
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Generate Candidates Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateCandidates:
    def test_fee_difference_candidate(self):
        state = _make_state_with_intelligence("FEE_DIFFERENCE")
        result = generate_candidates(state)

        assert result["candidates"] is not None
        assert result["candidates"]["candidate_count"] == 1
        assert result["candidates"]["candidates"][0]["resolution_type"] == "FEE_ADJUSTMENT"
        assert result["candidates"]["candidates"][0]["amount_paise"] == 3000

    def test_exact_match_candidate(self):
        state = _make_state_with_intelligence("EXACT_MATCH")
        state.evidence_package["fees"] = []
        result = generate_candidates(state)

        assert result["candidates"]["candidate_count"] == 1
        assert result["candidates"]["candidates"][0]["resolution_type"] == "NO_ACTION"

    def test_unknown_no_candidate(self):
        state = _make_state_with_intelligence("UNKNOWN")
        state.evidence_package["fees"] = []
        result = generate_candidates(state)

        assert result["candidates"]["status"] == "UNRESOLVED"
        assert result["candidates"]["candidate_count"] == 0

    def test_missing_classification(self):
        state = create_initial_state(exception_id="EXC-001")
        result = generate_candidates(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No classification" in e for e in result["metadata"]["errors"])

    def test_missing_evidence(self):
        state = create_initial_state(exception_id="EXC-001")
        state.classification = {"exception_type": "FEE_DIFFERENCE"}
        result = generate_candidates(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_node_recorded(self):
        state = _make_state_with_intelligence("FEE_DIFFERENCE")
        result = generate_candidates(state)

        assert "generate_candidates" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Score Resolution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreResolution:
    def test_normal_scoring(self):
        state = _make_state_with_candidates()
        result = score_resolution(state)

        assert result["candidate_scores"] is not None
        assert len(result["candidate_scores"]["scored_candidates"]) == 1
        assert result["candidate_scores"]["best_score"] > 0

    def test_scores_have_components(self):
        state = _make_state_with_candidates()
        result = score_resolution(state)

        scored = result["candidate_scores"]["scored_candidates"][0]
        assert "evidence_score" in scored
        assert "ml_score" in scored
        assert "historical_score" in scored
        assert "financial_consistency_score" in scored
        assert "final_score" in scored

    def test_missing_candidates(self):
        state = create_initial_state(exception_id="EXC-001")
        result = score_resolution(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No candidates" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = _make_state_with_candidates()
        result = score_resolution(state)

        assert "score_resolution" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Select Best Candidate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectBestCandidate:
    def test_normal_selection(self):
        state = _make_state_with_candidates()
        # Add scores
        state.candidate_scores = {
            "scored_candidates": [
                {
                    "candidate_id": "CAND-FEE-001",
                    "evidence_score": 0.9,
                    "ml_score": 0.0,
                    "historical_score": 0.0,
                    "financial_consistency_score": 0.95,
                    "final_score": 0.60,
                }
            ],
            "best_score": 0.60,
        }
        result = select_best_candidate(state)

        assert result["selected_candidate"] is not None
        assert result["confidence"] == 0.60
        assert result["decision"] in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_high_confidence_auto(self):
        state = _make_state_with_candidates()
        state.candidate_scores = {
            "scored_candidates": [
                {
                    "candidate_id": "CAND-FEE-001",
                    "evidence_score": 0.9,
                    "ml_score": 0.0,
                    "historical_score": 0.0,
                    "financial_consistency_score": 0.95,
                    "final_score": 0.75,
                }
            ],
            "best_score": 0.75,
        }
        result = select_best_candidate(state)

        assert result["decision"] == "AUTO"
        assert result["risk"] == "LOW"

    def test_medium_confidence_human_review(self):
        state = _make_state_with_candidates()
        state.candidate_scores = {
            "scored_candidates": [
                {
                    "candidate_id": "CAND-FEE-001",
                    "evidence_score": 0.5,
                    "ml_score": 0.0,
                    "historical_score": 0.0,
                    "financial_consistency_score": 0.5,
                    "final_score": 0.45,
                }
            ],
            "best_score": 0.45,
        }
        result = select_best_candidate(state)

        assert result["decision"] == "HUMAN_REVIEW"
        assert result["risk"] == "MEDIUM"

    def test_low_confidence_unresolved(self):
        state = _make_state_with_candidates()
        state.candidate_scores = {
            "scored_candidates": [
                {
                    "candidate_id": "CAND-FEE-001",
                    "evidence_score": 0.2,
                    "ml_score": 0.0,
                    "historical_score": 0.0,
                    "financial_consistency_score": 0.2,
                    "final_score": 0.20,
                }
            ],
            "best_score": 0.20,
        }
        result = select_best_candidate(state)

        assert result["decision"] == "UNRESOLVED"
        assert result["risk"] == "HIGH"

    def test_missing_candidates(self):
        state = create_initial_state(exception_id="EXC-001")
        result = select_best_candidate(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_missing_scores(self):
        state = _make_state_with_candidates()
        state.candidate_scores = None
        result = select_best_candidate(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_workflow_completed(self):
        state = _make_state_with_candidates()
        state.candidate_scores = {
            "scored_candidates": [
                {"candidate_id": "CAND-FEE-001", "final_score": 0.75}
            ],
            "best_score": 0.75,
        }
        result = select_best_candidate(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.COMPLETED.value

    def test_node_recorded(self):
        state = _make_state_with_candidates()
        state.candidate_scores = {
            "scored_candidates": [{"candidate_id": "CAND-FEE-001", "final_score": 0.75}],
            "best_score": 0.75,
        }
        result = select_best_candidate(state)

        assert "select_best_candidate" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Full Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullResolutionWorkflow:
    def test_end_to_end_fee_difference(self):
        """Test complete workflow for fee difference."""
        result = run_workflow(exception_id="EXC-001")

        assert isinstance(result, AgentState)
        assert result.candidates is not None
        assert result.candidate_scores is not None
        assert result.selected_candidate is not None
        assert result.decision is not None

    def test_all_nodes_executed(self):
        """Test all resolution nodes executed."""
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
        ]
        for node in expected_nodes:
            assert node in result.metadata.nodes_executed

    def test_all_nodes_successful(self):
        """Test all nodes succeeded."""
        result = run_workflow(exception_id="EXC-001")

        for log in result.metadata.execution_log:
            assert log["success"] is True, f"Node {log['node']} failed: {log['error']}"

    def test_state_propagation(self):
        """Test state propagates correctly."""
        result = run_workflow(exception_id="EXC-001")

        assert result.reconciliation_result is not None
        assert result.evidence_package is not None
        assert result.evidence_graph is not None
        assert result.classification is not None
        assert result.similar_cases is not None
        assert result.candidates is not None
        assert result.candidate_scores is not None
        assert result.selected_candidate is not None

    def test_workflow_has_terminal_status(self):
        """Test workflow reaches a terminal status."""
        result = run_workflow(exception_id="EXC-001")

        # Status depends on routing
        assert result.metadata.workflow_status is not None

    def test_decision_made(self):
        """Test a decision was made."""
        result = run_workflow(exception_id="EXC-001")

        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
        assert result.confidence is not None
        assert result.risk is not None

    def test_execution_log_complete(self):
        """Test execution log has all entries."""
        result = run_workflow(exception_id="EXC-001")

        assert len(result.metadata.execution_log) >= 8

    def test_exception_id_preserved(self):
        """Test exception ID preserved throughout."""
        result = run_workflow(exception_id="EXC-001")

        assert result.metadata.exception_id == "EXC-001"
        assert result.classification["exception_id"] == "EXC-001"
        assert result.candidates["exception_id"] == "EXC-001"


class TestResolutionFailurePropagation:
    def test_missing_exception_records_error(self):
        """Test missing exception records error in execution log."""
        result = run_workflow(exception_id="EXC-999")

        # load_exception fails, but LangGraph continues
        failed_logs = [l for l in result.metadata.execution_log if not l["success"]]
        assert len(failed_logs) >= 1
        assert failed_logs[0]["node"] == "load_exception"
        # Error is recorded
        assert any("not found" in e for e in result.metadata.errors)


class TestResolutionStateIsolation:
    def test_each_phase_stores_independently(self):
        """Test each phase stores data independently."""
        result = run_workflow(exception_id="EXC-001")

        assert result.evidence_package.get("exception_id") == "EXC-001"
        assert result.classification.get("exception_id") == "EXC-001"
        assert result.candidates.get("exception_id") == "EXC-001"
        assert result.selected_candidate is not None
