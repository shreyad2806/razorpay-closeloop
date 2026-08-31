"""
Tests for LangGraph Workflow (Phase 7B).

Tests:
1. valid exception
2. missing exception
3. invalid exception ID
4. repeated execution
5. node failure
6. workflow creation
7. initial state creation
8. node execution recording
9. observability metadata
"""

import pytest

from app.agent.nodes import load_exception, _record_node_execution
from app.agent.workflow import create_workflow, create_initial_state, run_workflow
from app.schemas.agent_state import (
    AgentState,
    NodeStatus,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Creation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowCreation:
    def test_create_workflow(self):
        workflow = create_workflow()
        assert workflow is not None

    def test_workflow_is_compiled(self):
        workflow = create_workflow()
        # Compiled workflows have an invoke method
        assert hasattr(workflow, "invoke")


# ─────────────────────────────────────────────────────────────────────────────
# Initial State Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInitialState:
    def test_create_initial_state(self):
        state = create_initial_state(exception_id="EXC-001")
        assert state.metadata.exception_id == "EXC-001"
        assert state.metadata.workflow_status == WorkflowStatus.PENDING
        assert state.metadata.workflow_id.startswith("WF-")

    def test_create_with_case_id(self):
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        assert state.metadata.case_id == "CASE-001"

    def test_create_with_workflow_id(self):
        state = create_initial_state(
            exception_id="EXC-001", workflow_id="WF-CUSTOM"
        )
        assert state.metadata.workflow_id == "WF-CUSTOM"


# ─────────────────────────────────────────────────────────────────────────────
# Load Exception Node Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadExceptionNode:
    def test_valid_exception(self):
        """Test loading a valid exception."""
        state = create_initial_state(exception_id="EXC-001")
        result = load_exception(state)

        assert result["reconciliation_result"]["exception_id"] == "EXC-001"
        assert result["reconciliation_result"]["case_id"] == "CASE-001"
        assert result["metadata"]["workflow_status"] == WorkflowStatus.RUNNING.value
        assert "load_exception" in result["metadata"]["nodes_executed"]

    def test_missing_exception(self):
        """Test loading a non-existent exception."""
        state = create_initial_state(exception_id="EXC-999")
        result = load_exception(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("not found" in e for e in result["metadata"]["errors"])
        assert result["metadata"]["execution_log"][-1]["success"] is False

    def test_invalid_exception_id_format(self):
        """Test loading with invalid ID format."""
        state = create_initial_state(exception_id="INVALID-001")
        result = load_exception(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("Invalid" in e for e in result["metadata"]["errors"])

    def test_empty_exception_id(self):
        """Test loading with empty exception ID."""
        state = create_initial_state(exception_id="")
        result = load_exception(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value
        assert any("Missing" in e for e in result["metadata"]["errors"])

    def test_whitespace_exception_id(self):
        """Test loading with whitespace-only exception ID."""
        state = create_initial_state(exception_id="   ")
        result = load_exception(state)

        assert result["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_second_valid_exception(self):
        """Test loading a different valid exception."""
        state = create_initial_state(exception_id="EXC-002")
        result = load_exception(state)

        assert result["reconciliation_result"]["exception_id"] == "EXC-002"
        assert result["reconciliation_result"]["case_id"] == "CASE-002"

    def test_unknown_exception_type(self):
        """Test loading exception with UNKNOWN type."""
        state = create_initial_state(exception_id="EXC-003")
        result = load_exception(state)

        assert result["reconciliation_result"]["exception_id"] == "EXC-003"
        assert result["metadata"]["workflow_status"] == WorkflowStatus.RUNNING.value


# ─────────────────────────────────────────────────────────────────────────────
# Repeated Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRepeatedExecution:
    def test_same_exception_twice(self):
        """Test running the same exception twice produces same result."""
        state1 = create_initial_state(exception_id="EXC-001")
        result1 = load_exception(state1)

        state2 = create_initial_state(exception_id="EXC-001")
        result2 = load_exception(state2)

        assert result1["reconciliation_result"] == result2["reconciliation_result"]

    def test_different_exceptions(self):
        """Test running different exceptions produces different results."""
        state1 = create_initial_state(exception_id="EXC-001")
        result1 = load_exception(state1)

        state2 = create_initial_state(exception_id="EXC-002")
        result2 = load_exception(state2)

        assert result1["reconciliation_result"]["case_id"] != result2["reconciliation_result"]["case_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Node Execution Recording Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNodeExecutionRecording:
    def test_success_recorded(self):
        state = create_initial_state(exception_id="EXC-001")
        result = load_exception(state)

        log = result["metadata"]["execution_log"][-1]
        assert log["node"] == "load_exception"
        assert log["success"] is True
        assert log["error"] is None
        assert log["elapsed_ms"] is not None

    def test_failure_recorded(self):
        state = create_initial_state(exception_id="EXC-999")
        result = load_exception(state)

        log = result["metadata"]["execution_log"][-1]
        assert log["node"] == "load_exception"
        assert log["success"] is False
        assert log["error"] is not None

    def test_nodes_executed_accumulated(self):
        state = create_initial_state(exception_id="EXC-001")
        result = load_exception(state)

        assert "load_exception" in result["metadata"]["nodes_executed"]
        assert len(result["metadata"]["nodes_executed"]) == 1

    def test_timestamp_recorded(self):
        state = create_initial_state(exception_id="EXC-001")
        result = load_exception(state)

        log = result["metadata"]["execution_log"][-1]
        assert "timestamp" in log


# ─────────────────────────────────────────────────────────────────────────────
# Full Workflow Run Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullWorkflowRun:
    def test_run_workflow_valid(self):
        """Test full workflow execution with valid exception."""
        result = run_workflow(exception_id="EXC-001")

        assert isinstance(result, AgentState)
        assert result.reconciliation_result is not None
        assert result.reconciliation_result["exception_id"] == "EXC-001"

    def test_run_workflow_missing(self):
        """Test full workflow execution with missing exception records error."""
        result = run_workflow(exception_id="EXC-999")

        assert isinstance(result, AgentState)
        assert len(result.metadata.errors) > 0
        # load_exception fails but LangGraph continues
        failed_logs = [l for l in result.metadata.execution_log if not l["success"]]
        assert len(failed_logs) >= 1
        assert failed_logs[0]["node"] == "load_exception"

    def test_run_workflow_invalid_id(self):
        """Test full workflow execution with invalid ID records error."""
        result = run_workflow(exception_id="INVALID")

        assert isinstance(result, AgentState)
        assert any("Invalid" in e for e in result.metadata.errors)

    def test_run_workflow_with_workflow_id(self):
        """Test workflow with custom workflow ID."""
        result = run_workflow(
            exception_id="EXC-001", workflow_id="WF-TEST-001"
        )

        assert result.metadata.workflow_id == "WF-TEST-001"

    def test_run_workflow_preserves_exception_id(self):
        """Test workflow preserves exception ID throughout."""
        result = run_workflow(exception_id="EXC-002")

        assert result.metadata.exception_id == "EXC-002"
        assert result.reconciliation_result["exception_id"] == "EXC-002"


# ─────────────────────────────────────────────────────────────────────────────
# Observability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestObservability:
    def test_workflow_id_preserved(self):
        result = run_workflow(exception_id="EXC-001", workflow_id="WF-OBS-001")
        assert result.metadata.workflow_id == "WF-OBS-001"

    def test_exception_id_preserved(self):
        result = run_workflow(exception_id="EXC-001")
        assert result.metadata.exception_id == "EXC-001"

    def test_execution_log_populated(self):
        result = run_workflow(exception_id="EXC-001")
        assert len(result.metadata.execution_log) > 0

    def test_nodes_executed_populated(self):
        result = run_workflow(exception_id="EXC-001")
        assert "load_exception" in result.metadata.nodes_executed

    def test_timing_recorded(self):
        result = run_workflow(exception_id="EXC-001")
        log = result.metadata.execution_log[0]
        assert log["elapsed_ms"] is not None
        assert log["elapsed_ms"] >= 0

    def test_error_recorded_on_failure(self):
        result = run_workflow(exception_id="EXC-999")
        assert len(result.metadata.errors) > 0
        assert "not found" in result.metadata.errors[0]


# ─────────────────────────────────────────────────────────────────────────────
# State Transition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStateTransitions:
    def test_valid_workflow_completes(self):
        """Valid workflow runs through all nodes."""
        result = run_workflow(exception_id="EXC-001")
        assert len(result.metadata.nodes_executed) >= 1

    def test_failed_load_records_error(self):
        """Failed load records error in log."""
        result = run_workflow(exception_id="EXC-999")
        failed_logs = [l for l in result.metadata.execution_log if not l["success"]]
        assert len(failed_logs) >= 1

    def test_current_node_set(self):
        """Current node is set to last executed node."""
        result = run_workflow(exception_id="EXC-001")
        assert result.metadata.current_node is not None


# ─────────────────────────────────────────────────────────────────────────────
# Record Node Execution Helper Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordNodeExecution:
    def test_success_record(self):
        state = create_initial_state(exception_id="EXC-001")
        result = _record_node_execution(state, "test_node", success=True)

        assert result["metadata"]["nodes_executed"] == ["test_node"]
        assert result["metadata"]["execution_log"][0]["success"] is True

    def test_failure_record(self):
        state = create_initial_state(exception_id="EXC-001")
        result = _record_node_execution(
            state, "test_node", success=False, error="test error"
        )

        assert result["metadata"]["errors"] == ["test error"]
        assert result["metadata"]["execution_log"][0]["success"] is False

    def test_elapsed_time(self):
        import time

        state = create_initial_state(exception_id="EXC-001")
        start = time.perf_counter()
        time.sleep(0.01)
        result = _record_node_execution(
            state, "test_node", success=True, start_time=start
        )

        assert result["metadata"]["execution_log"][0]["elapsed_ms"] > 0
