"""
Tests for Investigation Nodes (Phase 7C).

Tests:
- normal investigation flow
- missing evidence
- graph failure
- ML unavailable
- retrieval unavailable
- partial state
- node failure
- correct state propagation
- full workflow end-to-end
"""

import pytest

from app.agent.investigation_nodes import (
    build_evidence_graph,
    classify_exception,
    gather_evidence,
    retrieve_similar_cases,
)
from app.agent.workflow import create_initial_state, run_workflow
from app.schemas.agent_state import AgentState, WorkflowStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_state_with_evidence():
    """Create state with evidence already loaded."""
    state = create_initial_state(exception_id="EXC-001")
    state.reconciliation_result = {"exception_id": "EXC-001", "status": "LOADED"}
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
    return state


def _make_state_with_classification():
    """Create state with classification already loaded."""
    state = _make_state_with_evidence()
    state.evidence_graph = {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}
    state.classification = {
        "exception_id": "EXC-001",
        "exception_type": "FEE_DIFFERENCE",
        "confidence": 0.95,
    }
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Gather Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGatherEvidence:
    def test_normal_evidence_gathering(self):
        state = create_initial_state(exception_id="EXC-001")
        state.reconciliation_result = {"exception_id": "EXC-001", "status": "LOADED"}
        result = gather_evidence(state)

        assert result["evidence_package"] is not None
        assert result["evidence_package"]["exception_id"] == "EXC-001"
        assert "payment" in result["evidence_package"]
        assert "settlements" in result["evidence_package"]

    def test_evidence_stores_coverage(self):
        state = create_initial_state(exception_id="EXC-001")
        state.reconciliation_result = {"exception_id": "EXC-001", "status": "LOADED"}
        result = gather_evidence(state)

        assert result["evidence_package"]["evidence_coverage"] == 0.95
        assert result["evidence_package"]["evidence_consistency"] == 0.90

    def test_missing_exception_id(self):
        state = create_initial_state(exception_id="")
        result = gather_evidence(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No exception ID" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = create_initial_state(exception_id="EXC-001")
        state.reconciliation_result = {"exception_id": "EXC-001", "status": "LOADED"}
        result = gather_evidence(state)

        assert "gather_evidence" in result["metadata"]["nodes_executed"]
        assert result["metadata"]["execution_log"][-1]["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Build Evidence Graph Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildEvidenceGraph:
    def test_normal_graph_building(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        assert result["evidence_graph"] is not None
        assert result["evidence_graph"]["node_count"] > 0
        assert result["evidence_graph"]["edge_count"] > 0

    def test_graph_has_payment_node(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        node_ids = [n["id"] for n in result["evidence_graph"]["nodes"]]
        assert "PAY-001" in node_ids

    def test_graph_has_settlement_node(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        node_ids = [n["id"] for n in result["evidence_graph"]["nodes"]]
        assert "SET-001" in node_ids

    def test_graph_has_fee_node(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        node_ids = [n["id"] for n in result["evidence_graph"]["nodes"]]
        assert "FEE-001" in node_ids

    def test_graph_has_edges(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        assert len(result["evidence_graph"]["edges"]) >= 2

    def test_missing_evidence_package(self):
        state = create_initial_state(exception_id="EXC-001")
        result = build_evidence_graph(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No evidence package" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = _make_state_with_evidence()
        result = build_evidence_graph(state)

        assert "build_evidence_graph" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Classify Exception Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyException:
    def test_normal_classification(self):
        state = _make_state_with_evidence()
        result = classify_exception(state)

        assert result["classification"] is not None
        assert result["classification"]["exception_type"] == "FEE_DIFFERENCE"
        assert result["classification"]["confidence"] == 0.95

    def test_classification_with_refunds(self):
        state = _make_state_with_evidence()
        state.evidence_package["fees"] = []
        state.evidence_package["refunds"] = [{"refund_id": "REF-001", "amount": 1500}]
        result = classify_exception(state)

        assert result["classification"]["exception_type"] == "REFUND_ADJUSTMENT"

    def test_classification_exact_match(self):
        state = _make_state_with_evidence()
        state.evidence_package["fees"] = []
        state.evidence_package["refunds"] = []
        result = classify_exception(state)

        assert result["classification"]["exception_type"] == "EXACT_MATCH"

    def test_missing_evidence_package(self):
        state = create_initial_state(exception_id="EXC-001")
        result = classify_exception(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No evidence package" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = _make_state_with_evidence()
        result = classify_exception(state)

        assert "classify_exception" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve Similar Cases Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveSimilarCases:
    def test_normal_similarity_search(self):
        state = _make_state_with_classification()
        result = retrieve_similar_cases(state)

        assert result["similar_cases"] is not None
        assert result["similar_cases"]["total_indexed"] == 150
        assert len(result["similar_cases"]["similar_cases"]) == 3
        assert result["similar_cases"]["best_similarity_score"] == 0.92

    def test_similar_cases_have_resolution(self):
        state = _make_state_with_classification()
        result = retrieve_similar_cases(state)

        for case in result["similar_cases"]["similar_cases"]:
            assert "resolution" in case
            assert "outcome" in case

    def test_missing_classification(self):
        state = create_initial_state(exception_id="EXC-001")
        result = retrieve_similar_cases(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("No classification" in e for e in result["metadata"]["errors"])

    def test_node_recorded(self):
        state = _make_state_with_classification()
        result = retrieve_similar_cases(state)

        assert "retrieve_similar_cases" in result["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# Full Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullInvestigationWorkflow:
    def test_end_to_end_investigation(self):
        """Test complete investigation workflow."""
        result = run_workflow(exception_id="EXC-001")

        assert isinstance(result, AgentState)
        assert result.evidence_package is not None
        assert result.evidence_graph is not None
        assert result.classification is not None
        assert result.similar_cases is not None

    def test_all_nodes_executed(self):
        """Test all investigation nodes executed."""
        result = run_workflow(exception_id="EXC-001")

        expected_nodes = [
            "load_exception",
            "gather_evidence",
            "build_evidence_graph",
            "classify_exception",
            "retrieve_similar_cases",
        ]
        for node in expected_nodes:
            assert node in result.metadata.nodes_executed

    def test_all_nodes_successful(self):
        """Test all nodes succeeded."""
        result = run_workflow(exception_id="EXC-001")

        for log in result.metadata.execution_log:
            assert log["success"] is True, f"Node {log['node']} failed: {log['error']}"

    def test_state_propagation(self):
        """Test state propagates correctly through nodes."""
        result = run_workflow(exception_id="EXC-001")

        # Each phase produced data
        assert result.reconciliation_result is not None
        assert result.evidence_package is not None
        assert result.evidence_graph is not None
        assert result.classification is not None
        assert result.similar_cases is not None

    def test_workflow_has_status(self):
        """Test workflow has a status after running."""
        result = run_workflow(exception_id="EXC-001")

        assert result.metadata.workflow_status is not None

    def test_current_node_set(self):
        """Test current node is set to last executed."""
        result = run_workflow(exception_id="EXC-001")

        assert result.metadata.current_node is not None

    def test_execution_log_complete(self):
        """Test execution log has entries for all nodes."""
        result = run_workflow(exception_id="EXC-001")

        assert len(result.metadata.execution_log) >= 5

    def test_timing_recorded(self):
        """Test timing is recorded for all nodes."""
        result = run_workflow(exception_id="EXC-001")

        for log in result.metadata.execution_log:
            assert log["elapsed_ms"] is not None
            assert log["elapsed_ms"] >= 0

    def test_exception_id_preserved(self):
        """Test exception ID preserved throughout."""
        result = run_workflow(exception_id="EXC-001")

        assert result.metadata.exception_id == "EXC-001"
        assert result.reconciliation_result["exception_id"] == "EXC-001"
        assert result.evidence_package["exception_id"] == "EXC-001"
        assert result.classification["exception_id"] == "EXC-001"


class TestInvestigationFailurePropagation:
    def test_missing_exception_records_error(self):
        """Test missing exception records error."""
        result = run_workflow(exception_id="EXC-999")

        assert len(result.metadata.errors) > 0
        assert "Exception not found" in result.metadata.errors[0]
        # load_exception failed, but LangGraph continues
        failed_logs = [l for l in result.metadata.execution_log if not l["success"]]
        assert len(failed_logs) >= 1
        assert failed_logs[0]["node"] == "load_exception"

    def test_invalid_id_records_error(self):
        """Test invalid ID records error."""
        result = run_workflow(exception_id="INVALID")

        assert any("Invalid" in e for e in result.metadata.errors)
        failed_logs = [l for l in result.metadata.execution_log if not l["success"]]
        assert len(failed_logs) >= 1


class TestInvestigationStateIsolation:
    def test_node_data_isolation(self):
        """Test each node's data is independent."""
        result = run_workflow(exception_id="EXC-001")

        # Each phase stored data independently
        assert result.evidence_package.get("exception_id") == "EXC-001"
        assert result.classification.get("exception_id") == "EXC-001"
        assert result.similar_cases.get("query_exception_type") == "FEE_DIFFERENCE"

    def test_graph_independent_of_classification(self):
        """Test graph is built before classification."""
        result = run_workflow(exception_id="EXC-001")

        # Graph has nodes and edges
        assert result.evidence_graph["node_count"] > 0
        # Classification has type
        assert result.classification["exception_type"] is not None
