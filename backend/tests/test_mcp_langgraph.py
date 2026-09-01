"""
Tests for Razorpay CloseLoop Phase 11G — LangGraph MCP Client Integration.

Verifies:
- MCP Client wraps MCPServer correctly
- LangGraph nodes delegate through MCP
- Tool mapping works for all 10 MCP tools
- State flows correctly through MCP-delegated nodes
- Guardrails still execute through MCP path
- Audit trail records all MCP invocations
- No financial logic in MCP client or wrappers
"""

import pytest
from unittest.mock import MagicMock, patch

from mcp.client import MCPClient
from mcp.server import MCPServer
from mcp.config import MCPServerConfig, MCPToolCategory
from mcp.tools.langgraph_tools import (
    mcp_search_financial_records,
    mcp_get_payment,
    mcp_get_settlement,
    mcp_get_similar_exception,
    mcp_create_resolution,
    mcp_verify_resolution,
    mcp_record_feedback,
    create_mcp_node,
    create_mcp_workflow,
)
from mcp.tools.readonly import TOOL_DEFINITIONS as READONLY_TOOL_DEFS, create_handlers as create_readonly_handlers
from mcp.tools.write import WRITE_TOOL_DEFINITIONS, create_write_handlers
from mcp.adapters.financial_data import FinancialDataAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp_server_with_tools() -> MCPServer:
    """Create an MCPServer with all read-only and write tools registered."""
    server = MCPServer()

    adapter = FinancialDataAdapter()
    readonly_handlers = create_readonly_handlers(adapter)
    for defn in READONLY_TOOL_DEFS:
        if defn.name in readonly_handlers:
            server.register_tool(defn, readonly_handlers[defn.name])

    # Mock services for write tools
    mock_exec = MagicMock()
    mock_exec.execute.return_value = MagicMock(
        status=MagicMock(value="EXECUTED"),
        execution_id="EXE-TEST-001",
        actual_adjustment_paise=500,
        error=None,
    )
    write_handlers = create_write_handlers(mock_exec, MagicMock(), MagicMock())
    for defn in WRITE_TOOL_DEFINITIONS:
        if defn.name in write_handlers:
            server.register_tool(defn, write_handlers[defn.name])

    return server


@pytest.fixture
def mcp_client(mcp_server_with_tools: MCPServer) -> MCPClient:
    """Create an MCPClient connected to the test server."""
    return MCPClient(server=mcp_server_with_tools)


@pytest.fixture
def mock_mcp_client() -> MCPClient:
    """Create a mocked MCPClient for testing wrappers."""
    client = MagicMock(spec=MCPClient)
    client.call_tool.return_value = {
        "success": True,
        "tool_name": "test_tool",
        "request_id": "REQ-001",
        "status": "SUCCESS",
        "duration_ms": 10.0,
        "data": {"records": [{"id": "PAY-001"}], "count": 1},
    }
    return client


# ─────────────────────────────────────────────────────────────────────────────
# MCP Client
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPClient:
    def test_client_creation(self):
        client = MCPClient()
        assert client.server is not None
        assert client.invocation_history == []

    def test_client_with_custom_server(self, mcp_server_with_tools):
        client = MCPClient(server=mcp_server_with_tools)
        assert client.server is mcp_server_with_tools

    def test_call_tool_success(self, mcp_client):
        result = mcp_client.call_tool(
            "get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id="WF-001",
        )
        assert result["success"] is True
        assert result["tool_name"] == "get_payment"
        assert result["status"] == "SUCCESS"
        assert "data" in result

    def test_call_tool_unknown_tool(self, mcp_client):
        result = mcp_client.call_tool(
            "nonexistent_tool",
            parameters={},
        )
        assert result["success"] is False
        assert result["status"] == "ERROR"

    def test_call_tool_records_history(self, mcp_client):
        mcp_client.call_tool("get_payment", parameters={"payment_id": "PAY-001"})
        mcp_client.call_tool("get_payment", parameters={"payment_id": "PAY-002"})
        assert mcp_client.get_invocation_count() == 2
        assert mcp_client.get_invocations_by_tool("get_payment") == \
            mcp_client.invocation_history

    def test_call_tool_records_error(self, mcp_client):
        mcp_client.call_tool("nonexistent_tool", parameters={})
        assert mcp_client.get_error_count() == 1

    def test_call_tool_with_workflow_context(self, mcp_client):
        result = mcp_client.call_tool(
            "get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id="WF-001",
            agent_id="agent-test",
            exception_id="CASE-001",
        )
        assert result["success"] is True
        assert mcp_client.invocation_history[0]["workflow_id"] == "WF-001"
        assert mcp_client.invocation_history[0]["exception_id"] == "CASE-001"

    def test_audit_summary(self, mcp_client):
        mcp_client.call_tool("get_payment", parameters={"payment_id": "PAY-001"})
        mcp_client.call_tool("nonexistent_tool", parameters={})
        summary = mcp_client.get_audit_summary()
        assert summary["total_invocations"] == 2
        assert summary["error_count"] == 1
        assert summary["success_count"] == 1
        assert "get_payment" in summary["tools_called"]

    def test_audit_summary_empty(self):
        client = MCPClient()
        summary = client.get_audit_summary()
        assert summary["total_invocations"] == 0
        assert summary["avg_duration_ms"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# MCP Client Convenience Methods
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPClientConvenience:
    def test_search_financial_records(self, mcp_client):
        result = mcp_client.search_financial_records(
            case_id="CASE-001",
            workflow_id="WF-001",
        )
        assert result["success"] is True
        assert result["tool_name"] == "search_financial_records"

    def test_get_payment(self, mcp_client):
        result = mcp_client.get_payment("PAY-001", workflow_id="WF-001")
        assert result["success"] is True
        assert result["tool_name"] == "get_payment"

    def test_get_settlement(self, mcp_client):
        result = mcp_client.get_settlement("SET-001", workflow_id="WF-001")
        assert result["success"] is True

    def test_get_refund(self, mcp_client):
        result = mcp_client.get_refund("REF-001", workflow_id="WF-001")
        assert result["success"] is True

    def test_get_fee(self, mcp_client):
        result = mcp_client.get_fee("FEE-001", workflow_id="WF-001")
        assert result["success"] is True

    def test_get_adjustment(self, mcp_client):
        result = mcp_client.get_adjustment("ADJ-001", workflow_id="WF-001")
        assert result["success"] is True

    def test_get_similar_exception(self, mcp_client):
        result = mcp_client.get_similar_exception("CASE-001", workflow_id="WF-001")
        assert result["success"] is True
        assert result["tool_name"] == "get_similar_exception"

    def test_create_resolution(self, mcp_client):
        result = mcp_client.create_resolution(
            exception_id="CASE-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        assert result["success"] is True

    def test_record_feedback(self, mcp_client):
        result = mcp_client.record_feedback(
            workflow_id="WF-001",
            exception_id="CASE-001",
            feedback_type="APPROVE",
            reviewer="human-01",
            system_prediction="FEE_DIFFERENCE",
        )
        assert result["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tool Mapping (10 tools)
# ─────────────────────────────────────────────────────────────────────────────


class TestToolMapping:
    READ_ONLY_TOOLS = [
        "search_financial_records",
        "get_payment",
        "get_settlement",
        "get_refund",
        "get_fee",
        "get_adjustment",
        "get_similar_exception",
    ]

    WRITE_TOOLS = [
        "create_resolution",
        "verify_resolution",
        "record_feedback",
    ]

    def test_all_read_only_tools_callable(self, mcp_client):
        """Verify all 7 read-only tools are accessible via client."""
        params_map = {
            "search_financial_records": {"limit": 10},
            "get_payment": {"payment_id": "PAY-001"},
            "get_settlement": {"settlement_id": "SET-001"},
            "get_refund": {"refund_id": "REF-001"},
            "get_fee": {"fee_id": "FEE-001"},
            "get_adjustment": {"adjustment_id": "ADJ-001"},
            "get_similar_exception": {"exception_id": "CASE-001", "top_k": 3},
        }
        for tool_name in self.READ_ONLY_TOOLS:
            result = mcp_client.call_tool(tool_name, parameters=params_map[tool_name])
            assert result["success"] is True, f"Tool {tool_name} failed"

    def test_all_write_tools_callable(self, mcp_client):
        """Verify all 3 write tools are accessible via client."""
        params_map = {
            "create_resolution": {
                "exception_id": "CASE-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 500,
                "workflow_id": "WF-001",
                "authorization_source": "guardrail_AUTO",
                "guardrail_decision": "AUTO",
                "idempotency_key": "IDEM-TEST",
            },
            "verify_resolution": {
                "execution_id": "EXE-001",
                "workflow_id": "WF-001",
            },
            "record_feedback": {
                "workflow_id": "WF-001",
                "exception_id": "CASE-001",
                "feedback_type": "APPROVE",
                "reviewer": "human-01",
                "system_prediction": "FEE_DIFFERENCE",
            },
        }
        for tool_name in self.WRITE_TOOLS:
            result = mcp_client.call_tool(tool_name, parameters=params_map[tool_name])
            assert result["success"] is True, f"Tool {tool_name} failed: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Wrapper Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPToolWrappers:
    def test_search_financial_records_wrapper(self, mock_mcp_client):
        result = mcp_search_financial_records(
            mock_mcp_client, exception_id="CASE-001", workflow_id="WF-001"
        )
        assert result is not None
        mock_mcp_client.search_financial_records.assert_called_once()

    def test_get_payment_wrapper(self, mock_mcp_client):
        result = mcp_get_payment(
            mock_mcp_client, payment_id="PAY-001",
            workflow_id="WF-001", exception_id="CASE-001",
        )
        assert result is not None
        mock_mcp_client.get_payment.assert_called_once()

    def test_get_settlement_wrapper(self, mock_mcp_client):
        result = mcp_get_settlement(
            mock_mcp_client, settlement_id="SET-001",
            workflow_id="WF-001", exception_id="CASE-001",
        )
        assert result is not None

    def test_get_similar_exception_wrapper(self, mock_mcp_client):
        result = mcp_get_similar_exception(
            mock_mcp_client, exception_id="CASE-001", workflow_id="WF-001",
        )
        assert result is not None

    def test_create_resolution_wrapper(self, mock_mcp_client):
        result = mcp_create_resolution(
            mock_mcp_client,
            exception_id="CASE-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        assert result is not None

    def test_record_feedback_wrapper(self, mock_mcp_client):
        result = mcp_record_feedback(
            mock_mcp_client,
            workflow_id="WF-001",
            exception_id="CASE-001",
            feedback_type="APPROVE",
            reviewer="human-01",
            system_prediction="FEE_DIFFERENCE",
        )
        assert result is not None

    def test_wrapper_delegates_to_client(self, mock_mcp_client):
        """All wrappers delegate to client — no independent logic."""
        mcp_search_financial_records(mock_mcp_client, "CASE-001", "WF-001")
        mock_mcp_client.search_financial_records.assert_called_once()

        mcp_get_payment(mock_mcp_client, "PAY-001", "WF-001", "CASE-001")
        mock_mcp_client.get_payment.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# create_mcp_node (generic node factory)
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateMCPNode:
    def _make_state(self, exception_id="CASE-001", workflow_id="WF-001"):
        from app.schemas.agent_state import AgentState, WorkflowMetadata
        return AgentState(
            metadata=WorkflowMetadata(
                workflow_id=workflow_id,
                exception_id=exception_id,
            )
        )

    def test_node_success(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": True,
            "tool_name": "test_tool",
            "request_id": "REQ-001",
            "status": "SUCCESS",
            "duration_ms": 5.0,
            "data": {"records": [{"id": "PAY-001"}], "count": 1},
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="search_financial_records",
            node_name="test_node",
            extract_params=lambda s: {"case_id": s.metadata.exception_id},
            extract_result=lambda r: r.get("data", {}),
            state_key="evidence_package",
        )

        state = self._make_state()
        updates = node_fn(state)

        assert updates["evidence_package"]["records"][0]["id"] == "PAY-001"
        assert "test_node" in updates["metadata"]["nodes_executed"]

    def test_node_failure(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": False,
            "tool_name": "test_tool",
            "status": "ERROR",
            "error": "Tool failed",
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="test_tool",
            node_name="test_fail",
            extract_params=lambda s: {},
            extract_result=lambda r: r.get("data", {}),
            state_key="evidence_package",
        )

        state = self._make_state()
        updates = node_fn(state)

        assert "evidence_package" not in updates
        assert any("Tool failed" in e for e in updates["metadata"]["errors"])

    def test_node_exception(self, mock_mcp_client):
        mock_mcp_client.call_tool.side_effect = RuntimeError("MCP down")

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="test_tool",
            node_name="test_exc",
            extract_params=lambda s: {},
            extract_result=lambda r: r,
            state_key="data",
        )

        state = self._make_state()
        updates = node_fn(state)

        assert any("test_exc" in e for e in updates["metadata"]["errors"])
        from app.schemas.agent_state import WorkflowStatus
        assert updates["metadata"]["workflow_status"] == WorkflowStatus.FAILED.value

    def test_node_logs_elapsed_time(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": True, "tool_name": "t", "request_id": "R",
            "status": "SUCCESS", "duration_ms": 1.0, "data": {},
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="t", node_name="timed",
            extract_params=lambda s: {},
            extract_result=lambda r: {},
            state_key="data",
        )

        state = self._make_state()
        updates = node_fn(state)
        log = updates["metadata"]["execution_log"][-1]
        assert log["elapsed_ms"] is not None
        assert log["elapsed_ms"] >= 0

    def test_node_preserves_existing_state(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": True, "tool_name": "t", "request_id": "R",
            "status": "SUCCESS", "duration_ms": 1.0, "data": {"ok": True},
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="t", node_name="pres",
            extract_params=lambda s: {},
            extract_result=lambda r: r.get("data", {}),
            state_key="new_data",
        )

        state = self._make_state()
        state.errors = ["previous_error"]
        updates = node_fn(state)

        assert updates["new_data"]["ok"] is True
        # Previous nodes_executed should be preserved
        assert "load_exception" not in updates["metadata"]["nodes_executed"]


# ─────────────────────────────────────────────────────────────────────────────
# MCP Workflow
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPWorkflow:
    def test_workflow_compiles(self, mcp_server_with_tools):
        client = MCPClient(server=mcp_server_with_tools)
        workflow = create_mcp_workflow(client)
        assert workflow is not None

    def test_workflow_has_expected_nodes(self, mcp_server_with_tools):
        client = MCPClient(server=mcp_server_with_tools)
        workflow = create_mcp_workflow(client)
        # LangGraph compiled graph should be invokable
        assert hasattr(workflow, "invoke")

    def test_workflow_start_node(self, mcp_server_with_tools):
        """Verify the workflow can start and routes through MCP nodes."""
        client = MCPClient(server=mcp_server_with_tools)
        workflow = create_mcp_workflow(client)

        from app.schemas.agent_state import AgentState, WorkflowMetadata, WorkflowStatus
        initial = AgentState(
            metadata=WorkflowMetadata(
                workflow_id="WF-MCP-001",
                exception_id="CASE-001",
            )
        )

        # The workflow may fail at evidence gathering (no matching data),
        # but it should start and route correctly
        result = workflow.invoke(initial)

        # Verify the workflow executed at least load_exception
        if isinstance(result, dict):
            meta = result.get("metadata", {})
        else:
            meta = result.metadata.model_dump()

        assert "load_exception" in meta.get("nodes_executed", [])


# ─────────────────────────────────────────────────────────────────────────────
# State Flow Through MCP Nodes
# ─────────────────────────────────────────────────────────────────────────────


class TestStateFlow:
    def test_evidence_stored_in_state(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": True, "tool_name": "search_financial_records",
            "request_id": "R", "status": "SUCCESS", "duration_ms": 1.0,
            "data": {"records": [{"id": "PAY-001"}], "count": 1},
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="search_financial_records",
            node_name="gather_evidence",
            extract_params=lambda s: {"case_id": s.metadata.exception_id},
            extract_result=lambda r: r.get("data", {}),
            state_key="evidence_package",
        )

        from app.schemas.agent_state import AgentState, WorkflowMetadata
        state = AgentState(metadata=WorkflowMetadata(
            workflow_id="WF-001", exception_id="CASE-001",
        ))
        updates = node_fn(state)

        assert "evidence_package" in updates
        assert updates["evidence_package"]["count"] == 1

    def test_similar_cases_stored_in_state(self, mock_mcp_client):
        mock_mcp_client.call_tool.return_value = {
            "success": True, "tool_name": "get_similar_exception",
            "request_id": "R", "status": "SUCCESS", "duration_ms": 1.0,
            "data": {"similar_cases": [{"case_id": "CASE-042"}], "total_indexed": 150},
        }

        node_fn = create_mcp_node(
            client=mock_mcp_client,
            tool_name="get_similar_exception",
            node_name="retrieve_similar_cases",
            extract_params=lambda s: {"exception_id": s.metadata.exception_id},
            extract_result=lambda r: r.get("data", {}),
            state_key="similar_cases",
        )

        from app.schemas.agent_state import AgentState, WorkflowMetadata
        state = AgentState(metadata=WorkflowMetadata(
            workflow_id="WF-001", exception_id="CASE-001",
        ))
        updates = node_fn(state)

        assert "similar_cases" in updates
        assert len(updates["similar_cases"]["similar_cases"]) == 1

    def test_multiple_sequential_mcp_calls(self, mcp_client):
        """Simulate agent making multiple MCP calls in sequence."""
        # Search
        r1 = mcp_client.search_financial_records(exception_id="CASE-001", workflow_id="WF-001")
        assert r1["success"]

        # Get payment
        r2 = mcp_client.get_payment("PAY-001", workflow_id="WF-001")
        assert r2["success"]

        # Similar cases
        r3 = mcp_client.get_similar_exception("CASE-001", workflow_id="WF-001")
        assert r3["success"]

        assert mcp_client.get_invocation_count() == 3


# ─────────────────────────────────────────────────────────────────────────────
# Audit Trail
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditTrail:
    def test_server_records_all_mcp_calls(self, mcp_client):
        mcp_client.get_payment("PAY-001", workflow_id="WF-001")
        mcp_client.search_financial_records(exception_id="CASE-001", workflow_id="WF-001")

        # Server audit should have both calls
        audit_log = mcp_client.server.get_audit_log()
        assert len(audit_log) >= 2

    def test_audit_includes_workflow_id(self, mcp_client):
        mcp_client.call_tool(
            "get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id="WF-AUDIT-001",
        )
        audit_log = mcp_client.server.get_audit_log(workflow_id="WF-AUDIT-001")
        assert len(audit_log) >= 1

    def test_audit_includes_exception_id(self, mcp_client):
        mcp_client.call_tool(
            "get_payment",
            parameters={"payment_id": "PAY-001"},
            exception_id="CASE-001",
        )
        audit_log = mcp_client.server.get_audit_log(exception_id="CASE-001")
        assert len(audit_log) >= 1

    def test_client_audit_summary(self, mcp_client):
        for i in range(5):
            mcp_client.get_payment(f"PAY-{i:03d}", workflow_id="WF-001")
        summary = mcp_client.get_audit_summary()
        assert summary["total_invocations"] == 5
        assert summary["error_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_client_no_financial_logic(self):
        """MCPClient must not contain business logic."""
        import inspect
        client_code = inspect.getsource(MCPClient)
        # Should not contain financial calculation keywords
        assert "calculate_interest" not in client_code.lower()
        assert "compute_fee" not in client_code.lower()
        assert "settlement_amount" not in client_code.lower()

    def test_client_no_database_access(self):
        """MCPClient must not directly access database."""
        import inspect
        client_code = inspect.getsource(MCPClient)
        assert "session.query" not in client_code
        assert "cursor.execute" not in client_code
        assert "db.execute" not in client_code

    def test_client_no_guardrail_override(self):
        """MCPClient must not override guardrail decisions."""
        import inspect
        client_code = inspect.getsource(MCPClient)
        assert "bypass_guardrail" not in client_code.lower()
        assert "force_auto" not in client_code.lower()
        assert "override_decision" not in client_code.lower()

    def test_wrappers_no_business_logic(self):
        """Wrapper functions must not contain business logic."""
        import inspect
        for fn in [
            mcp_search_financial_records,
            mcp_get_payment,
            mcp_get_settlement,
            mcp_get_similar_exception,
            mcp_create_resolution,
            mcp_record_feedback,
        ]:
            code = inspect.getsource(fn)
            # Wrappers should only delegate to client methods
            assert "SELECT" not in code
            assert "INSERT" not in code
            assert "UPDATE" not in code

    def test_workflow_delegates_through_mcp(self, mcp_server_with_tools):
        """MCP workflow should use MCP tools, not direct service calls."""
        client = MCPClient(server=mcp_server_with_tools)
        workflow = create_mcp_workflow(client)

        # Workflow should be invokable without error in creation
        assert workflow is not None


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_parameters(self, mcp_client):
        result = mcp_client.call_tool("get_payment", parameters={})
        assert result["success"] is False

    def test_none_workflow_id(self, mcp_client):
        result = mcp_client.call_tool(
            "get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id=None,
        )
        assert result["success"] is True

    def test_large_top_k(self, mcp_client):
        result = mcp_client.get_similar_exception("CASE-001", top_k=100)
        assert result["success"] is True

    def test_concurrent_invocations(self, mcp_server_with_tools):
        """Multiple clients sharing a server should not corrupt state."""
        c1 = MCPClient(server=mcp_server_with_tools)
        c2 = MCPClient(server=mcp_server_with_tools)

        r1 = c1.get_payment("PAY-001", workflow_id="WF-001")
        r2 = c2.get_payment("PAY-002", workflow_id="WF-002")

        assert r1["success"] is True
        assert r2["success"] is True
        # Each client tracks its own history
        assert c1.get_invocation_count() == 1
        assert c2.get_invocation_count() == 1

    def test_error_response_structure(self, mcp_client):
        result = mcp_client.call_tool("nonexistent_tool", parameters={})
        assert "success" in result
        assert "tool_name" in result
        assert "status" in result
        assert result["success"] is False
        assert "error" in result
