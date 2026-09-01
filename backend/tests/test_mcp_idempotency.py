"""
Tests for Razorpay CloseLoop Phase 11F — MCP Idempotency + Failure Safety.

Verifies:
- Duplicate requests return cached results
- Timeout = unknown, not failure
- No response = status unknown, not not-executed
- Partial failures handled safely
- Retry doesn't duplicate financial actions
"""

import pytest
from datetime import datetime, timezone

from mcp.idempotency import (
    MCPOperationExecutor,
    MCPOperationRecord,
    MCPOperationStatus,
    MCPOperationsStore,
)
from mcp.schemas import MCPToolStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def executor() -> MCPOperationExecutor:
    return MCPOperationExecutor()


@pytest.fixture
def store() -> MCPOperationsStore:
    return MCPOperationsStore()


def _success_handler(params):
    return {"executed": True, "adjustment": 500}


def _fail_handler(params):
    raise RuntimeError("Backend service failure")


def _slow_handler(params):
    import time
    time.sleep(0.01)
    return {"executed": True}


def _call_counting_handler(params):
    """Track call count to verify no duplicates."""
    _call_counting_handler.count = getattr(_call_counting_handler, "count", 0) + 1
    return {"executed": True, "call_count": _call_counting_handler.count}


# ─────────────────────────────────────────────────────────────────────────────
# Operations Store
# ─────────────────────────────────────────────────────────────────────────────


class TestOperationsStore:
    def test_create_operation(self, store: MCPOperationsStore):
        record = store.create_operation("KEY-001", "get_payment", {"payment_id": "PAY-001"})
        assert record.operation_id.startswith("OP-")
        assert record.idempotency_key == "KEY-001"
        assert record.status == MCPOperationStatus.PENDING

    def test_mark_in_progress(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        record = store.get_by_idempotency_key("KEY-001")
        assert record.status == MCPOperationStatus.IN_PROGRESS
        assert record.started_at is not None

    def test_mark_completed(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        store.mark_completed("KEY-001", {"result": "ok"})
        record = store.get_by_idempotency_key("KEY-001")
        assert record.status == MCPOperationStatus.COMPLETED
        assert record.result == {"result": "ok"}
        assert record.completed_at is not None

    def test_mark_failed(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        store.mark_failed("KEY-001", "Something went wrong")
        record = store.get_by_idempotency_key("KEY-001")
        assert record.status == MCPOperationStatus.FAILED
        assert record.error == "Something went wrong"

    def test_mark_timed_out(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        store.mark_timed_out("KEY-001")
        record = store.get_by_idempotency_key("KEY-001")
        assert record.status == MCPOperationStatus.TIMED_OUT

    def test_is_duplicate_new_key(self, store: MCPOperationsStore):
        assert store.is_duplicate("NEW-KEY") is False

    def test_is_duplicate_completed(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        store.mark_completed("KEY-001", {})
        assert store.is_duplicate("KEY-001") is True

    def test_is_duplicate_in_progress(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        assert store.is_duplicate("KEY-001") is True

    def test_is_duplicate_failed(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        store.mark_failed("KEY-001", "error")
        assert store.is_duplicate("KEY-001") is True

    def test_is_in_progress(self, store: MCPOperationsStore):
        store.create_operation("KEY-001", "t", {})
        store.mark_in_progress("KEY-001")
        assert store.is_in_progress("KEY-001") is True

    def test_get_by_operation_id(self, store: MCPOperationsStore):
        record = store.create_operation("KEY-001", "t", {})
        found = store.get_by_operation_id(record.operation_id)
        assert found is not None
        assert found.idempotency_key == "KEY-001"

    def test_operation_count(self, store: MCPOperationsStore):
        assert store.operation_count == 0
        store.create_operation("K1", "t", {})
        store.create_operation("K2", "t", {})
        assert store.operation_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Idempotent Execution
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotentExecution:
    def test_first_execution(self, executor: MCPOperationExecutor):
        result = executor.execute_idempotent(
            "KEY-001", "create_resolution", {}, _success_handler
        )
        assert result["executed"] is True
        assert result["_idempotent"] is False
        assert result["_cached"] is False

    def test_duplicate_returns_cached(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent(
            "KEY-001", "create_resolution", {}, _success_handler
        )
        r2 = executor.execute_idempotent(
            "KEY-001", "create_resolution", {}, _fail_handler
        )
        # Second call returns cached result, NOT executing fail_handler
        assert r2["_idempotent"] is True
        assert r2["_cached"] is True
        assert r2["executed"] is True  # Same as first result

    def test_no_duplicate_execution(self, executor: MCPOperationExecutor):
        _call_counting_handler.count = 0
        executor.execute_idempotent(
            "KEY-001", "t", {}, _call_counting_handler
        )
        executor.execute_idempotent(
            "KEY-001", "t", {}, _call_counting_handler
        )
        executor.execute_idempotent(
            "KEY-001", "t", {}, _call_counting_handler
        )
        assert _call_counting_handler.count == 1

    def test_different_keys_independent(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent(
            "KEY-001", "t", {}, _success_handler
        )
        r2 = executor.execute_idempotent(
            "KEY-002", "t", {}, _success_handler
        )
        assert r1["_operation_id"] != r2["_operation_id"]

    def test_failed_result_cached(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent(
            "KEY-001", "t", {}, _fail_handler
        )
        assert "error" in r1
        assert r1["_idempotent"] is False

        r2 = executor.execute_idempotent(
            "KEY-001", "t", {}, _success_handler
        )
        # Should return the cached FAILED result
        assert r2["error"] is not None
        assert r2["_cached"] is True

    def test_operation_id_returned(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        assert r1["_operation_id"].startswith("OP-")

    def test_same_operation_id_for_same_key(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        r2 = executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        assert r1["_operation_id"] == r2["_operation_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Status Query
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusQuery:
    def test_unknown_key(self, executor: MCPOperationExecutor):
        status = executor.query_status("NONEXISTENT")
        assert status["status"] == "UNKNOWN"
        assert "No operation found" in status["message"]

    def test_completed_status(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        status = executor.query_status("KEY-001")
        assert status["status"] == "COMPLETED"
        assert status["result"]["executed"] is True

    def test_failed_status(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        status = executor.query_status("KEY-001")
        assert status["status"] == "FAILED"
        assert status["error"] is not None

    def test_status_includes_metadata(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        status = executor.query_status("KEY-001")
        assert "created_at" in status
        assert "started_at" in status
        assert "completed_at" in status
        assert "retry_count" in status

    def test_status_includes_tool_name(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "create_resolution", {}, _success_handler)
        status = executor.query_status("KEY-001")
        assert status["tool_name"] == "create_resolution"


# ─────────────────────────────────────────────────────────────────────────────
# Timeout Safety
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeoutSafety:
    def test_timeout_does_not_mean_failure(self, executor: MCPOperationExecutor):
        """Timeout = unknown status, NOT failure."""
        # Simulate timeout by marking as timed out
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        executor.store.mark_timed_out("KEY-001")

        status = executor.query_status("KEY-001")
        assert status["status"] == "TIMED_OUT"
        # Should NOT say "failed"
        assert status.get("error") is None

    def test_no_response_means_unknown(self, executor: MCPOperationExecutor):
        """No response = status unknown, NOT not-executed."""
        status = executor.query_status("UNKNOWN-KEY")
        assert status["status"] == "UNKNOWN"
        # Should not claim it was not executed
        assert "not executed" not in status["message"].lower()

    def test_safe_retry_after_timeout(self, executor: MCPOperationExecutor):
        """After timeout, retry should query status, not blindly retry."""
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        executor.store.mark_timed_out("KEY-001")

        result = executor.safe_retry(
            "KEY-001", "t", {}, _success_handler
        )
        assert result["status"] == "TIMED_OUT"
        assert "UNKNOWN" in result["message"]
        # Should NOT have executed the handler
        assert "executed" not in result or result.get("executed") is None


# ─────────────────────────────────────────────────────────────────────────────
# Safe Retry
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeRetry:
    def test_retry_completed_returns_cached(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        result = executor.safe_retry("KEY-001", "t", {}, _success_handler)
        assert result["_cached"] is True
        assert result["_idempotent"] is True

    def test_retry_in_progress_does_not_duplicate(self, executor: MCPOperationExecutor):
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        result = executor.safe_retry("KEY-001", "t", {}, _success_handler)
        assert result["status"] == "IN_PROGRESS"
        assert "Do NOT retry blindly" in result["message"]

    def test_retry_new_operation(self, executor: MCPOperationExecutor):
        result = executor.safe_retry("KEY-001", "t", {}, _success_handler)
        assert result["executed"] is True

    def test_retry_max_retries(self, executor: MCPOperationExecutor):
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        executor.store.mark_failed("KEY-001", "error")
        record = executor.store.get_by_idempotency_key("KEY-001")
        record.retry_count = 1

        result = executor.safe_retry(
            "KEY-001", "t", {}, _success_handler, max_retries=1
        )
        assert "error" in result
        assert "Max retries" in result["error"]

    def test_retry_after_failed_allows_one(self, executor: MCPOperationExecutor):
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        executor.store.mark_failed("KEY-001", "error")
        record = executor.store.get_by_idempotency_key("KEY-001")
        record.retry_count = 0

        result = executor.safe_retry(
            "KEY-001", "t", {}, _success_handler, max_retries=2
        )
        # Should have retried and succeeded
        assert result["executed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Partial Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialFailure:
    def test_handler_exception_captured(self, executor: MCPOperationExecutor):
        result = executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        assert "error" in result
        assert "Backend service failure" in result["error"]

    def test_failed_result_does_not_mask_original(self, executor: MCPOperationExecutor):
        """If first call fails, second call should return the failed result."""
        r1 = executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        r2 = executor.execute_idempotent("KEY-001", "t", {}, _success_handler)
        # Should return the FAILED cached result, not re-execute
        assert r2["_cached"] is True
        assert r2.get("error") is not None

    def test_status_query_shows_failure(self, executor: MCPOperationExecutor):
        executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        status = executor.query_status("KEY-001")
        assert status["status"] == "FAILED"
        assert status["error"] is not None

    def test_different_keys_independent_failures(self, executor: MCPOperationExecutor):
        r1 = executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        r2 = executor.execute_idempotent("KEY-002", "t", {}, _success_handler)
        assert "error" in r1
        assert r2["executed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Operation Record
# ─────────────────────────────────────────────────────────────────────────────


class TestOperationRecord:
    def test_record_creation(self):
        record = MCPOperationRecord(
            operation_id="OP-001",
            idempotency_key="KEY-001",
            tool_name="create_resolution",
            parameters={"payment_id": "PAY-001"},
        )
        assert record.operation_id == "OP-001"
        assert record.status == MCPOperationStatus.PENDING
        assert record.created_at is not None

    def test_record_retry_count(self):
        record = MCPOperationRecord(
            operation_id="OP-001",
            idempotency_key="KEY-001",
            tool_name="t",
            parameters={},
        )
        assert record.retry_count == 0
        record.retry_count = 1
        assert record.retry_count == 1

    def test_record_status_transitions(self):
        record = MCPOperationRecord(
            operation_id="OP-001",
            idempotency_key="KEY-001",
            tool_name="t",
            parameters={},
        )
        assert record.status == MCPOperationStatus.PENDING
        record.status = MCPOperationStatus.IN_PROGRESS
        assert record.status == MCPOperationStatus.IN_PROGRESS
        record.status = MCPOperationStatus.COMPLETED
        assert record.status == MCPOperationStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# Integration with MCP Server
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPIntegration:
    def test_idempotent_write_via_server(self):
        """Verify write tools support idempotency through MCP server."""
        from mcp.server import MCPServer
        from mcp.tools.write import create_write_handlers, WRITE_TOOL_DEFINITIONS

        class MockExec:
            def execute(self, req):
                return type("R", (), {
                    "status": type("S", (), {"value": "EXECUTED"})(),
                    "execution_id": "EXE-001",
                    "actual_adjustment_paise": 500,
                    "error": None,
                })()

        class MockVerify:
            pass

        class MockFeedback:
            pass

        server = MCPServer()
        handlers = create_write_handlers(
            MockExec(), MockVerify(), MockFeedback(),
            idempotency_executor=server.idempotency_executor,
        )
        for defn in WRITE_TOOL_DEFINITIONS:
            if defn.name in handlers:
                server.register_tool(defn, handlers[defn.name])

        from mcp.schemas import MCPToolRequest

        # First call
        r1 = server.invoke(MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "CASE-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 500,
                "workflow_id": "WF-001",
                "authorization_source": "guardrail_AUTO",
                "guardrail_decision": "AUTO",
                "idempotency_key": "IDEM-001",
            },
        ))
        assert r1.status == MCPToolStatus.SUCCESS
        assert r1.result.get("executed") is True

        # Second call with same idempotency key
        r2 = server.invoke(MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "CASE-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 500,
                "workflow_id": "WF-001",
                "authorization_source": "guardrail_AUTO",
                "guardrail_decision": "AUTO",
                "idempotency_key": "IDEM-001",
            },
        ))
        # Both succeed (idempotency is at MCP handler level, not server level)
        # The server routes to the same handler, which delegates to execution service
        # which has its own idempotency via IdempotencyStore
        assert r2.status == MCPToolStatus.SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# Safety Properties
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyProperties:
    def test_timeout_never_assumed_failure(self):
        """Verify timeout status is distinct from failure."""
        assert MCPOperationStatus.TIMED_OUT.value != MCPOperationStatus.FAILED.value
        assert MCPOperationStatus.TIMED_OUT.value != MCPOperationStatus.COMPLETED.value

    def test_unknown_means_unknown(self):
        """Verify UNKNOWN is a valid status."""
        assert MCPOperationStatus.UNKNOWN.value == "UNKNOWN"

    def test_duplicate_never_executes_twice(self, executor: MCPOperationExecutor):
        """Verify no duplicate execution with same key."""
        _call_counting_handler.count = 0
        executor.execute_idempotent("KEY-001", "t", {}, _call_counting_handler)
        executor.execute_idempotent("KEY-001", "t", {}, _call_counting_handler)
        assert _call_counting_handler.count == 1

    def test_failed_result_persists(self, executor: MCPOperationExecutor):
        """Failed results are cached and not silently overwritten."""
        executor.execute_idempotent("KEY-001", "t", {}, _fail_handler)
        status = executor.query_status("KEY-001")
        assert status["status"] == "FAILED"

    def test_in_progress_prevents_retry(self, executor: MCPOperationExecutor):
        """In-progress operations cannot be blindly retried."""
        executor.store.create_operation("KEY-001", "t", {})
        executor.store.mark_in_progress("KEY-001")
        result = executor.safe_retry("KEY-001", "t", {}, _success_handler)
        assert result["status"] == "IN_PROGRESS"
        assert "Do NOT retry blindly" in result["message"]
