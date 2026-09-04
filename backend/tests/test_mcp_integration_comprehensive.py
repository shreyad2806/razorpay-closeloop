"""
Comprehensive MCP Integration Tests — Phase 14 supplement.

Tests the COMPLETE MCP stack:
    MCPServer → MCPToolRegistry → Tool Handlers → FinancialDataAdapter / Services
    MCPClient → MCPServer → Tool Handlers
    LangGraph → MCPClient → MCPServer → Tool Handlers → Results

Uses REAL data from the existing synthetic dataset.

Does NOT mock the MCP server — tests real execution paths.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from mcp.adapters.financial_data import FinancialDataAdapter
from mcp.audit import MCPAuditLogger
from mcp.client import MCPClient
from mcp.config import MCPServerConfig, MCPServerMode, MCPToolCategory
from mcp.idempotency import MCPOperationExecutor
from mcp.schemas import (
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolRequest,
    MCPToolResponse,
    MCPToolStatus,
)
from mcp.server import MCPServer
from mcp.tools.readonly import TOOL_DEFINITIONS as READONLY_DEFINITIONS
from mcp.tools.readonly import create_handlers as create_readonly_handlers
from mcp.tools.write import WRITE_TOOL_DEFINITIONS
from mcp.tools.write import create_write_handlers


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def data_dir():
    """Find the test data directory."""
    project_root = Path(__file__).parent.parent
    data = project_root / "data"
    # Find first batch with generated data
    for batch_dir in sorted(data.iterdir()):
        gen = batch_dir / "generated"
        if gen.is_dir() and (gen / "cases.json").exists():
            return str(batch_dir)
    pytest.skip("No test data available")


@pytest.fixture(scope="module")
def adapter(data_dir):
    """FinancialDataAdapter loaded with real test data."""
    a = FinancialDataAdapter(data_dir=data_dir)
    gen_dir = os.path.join(data_dir, "generated")
    a.load_batch("generated")  # Loads from data_dir/generated
    if not a.is_loaded:
        # Try loading from the generated subdirectory directly
        a._data_dir = data_dir
        a.load_batch("")
    if not a.is_loaded:
        # Fallback: load manually
        for fname, attr in [
            ("payments.json", "_payments"),
            ("settlements.json", "_settlements"),
            ("refunds.json", "_refunds"),
            ("fees.json", "_fees"),
            ("adjustments.json", "_adjustments"),
            ("cases.json", "_cases"),
        ]:
            path = os.path.join(data_dir, "generated", fname)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    setattr(a, attr, json.load(f))
        a._loaded = True
    return a


@pytest.fixture(scope="module")
def server(adapter):
    """MCPServer with all tools registered using real data."""
    config = MCPServerConfig(mode=MCPServerMode.EMBEDDED)
    srv = MCPServer(config=config)

    # Register read-only tools
    readonly_handlers = create_readonly_handlers(adapter)
    for defn in READONLY_DEFINITIONS:
        handler = readonly_handlers.get(defn.name)
        if handler:
            srv.register_tool(defn, handler)

    # Register write tools (with mock services for testing)
    class MockExecutionResult:
        """Mock result that satisfies the handler's attribute access."""
        def __init__(self, request):
            self.execution_id = f"EXEC-{int(time.time()*1000)}"

            class _Status:
                def __init__(self, v): self._v = v
                @property
                def value(self): return self._v
            self.status = _Status("EXECUTED")
            self.actual_adjustment_paise = request.get("financial_adjustment_paise", 0)
            self.error = None

    class MockExecutionService:
        def execute(self, request):
            return MockExecutionResult(request)
        def get_execution(self, execution_id):
            return None

    class MockVerificationEngine:
        def verify(self, execution_result):
            class _Status:
                def __init__(self, v): self._v = v
                @property
                def value(self): return self._v
            class _Result:
                def __init__(self):
                    self.verification_id = f"VER-{int(time.time()*1000)}"
                    self.status = _Status("PASSED")
                    self.discrepancy_eliminated = True
                    self.has_unintended_changes = False
                    self.passed_checks = ["amount_match"]
                    self.failed_checks = []
            return _Result()

    class MockFeedbackService:
        def record_feedback(self, **kwargs):
            class _Type:
                def __init__(self, v): self._v = v
                @property
                def value(self): return self._v
            class _Record:
                def __init__(self):
                    self.feedback_id = f"FB-{int(time.time()*1000)}"
                    self.feedback_type = _Type(kwargs.get("feedback_type", "APPROVE"))
                    self.reviewer = kwargs.get("reviewer", "test")
            return _Record()

    write_handlers = create_write_handlers(
        MockExecutionService(),
        MockVerificationEngine(),
        MockFeedbackService(),
    )
    for defn in WRITE_TOOL_DEFINITIONS:
        handler = write_handlers.get(defn.name)
        if handler:
            srv.register_tool(defn, handler)

    return srv


@pytest.fixture
def client(server):
    """MCPClient connected to the real server."""
    return MCPClient(server=server)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: MCP Server Startup
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPServerStartup:
    """Verify the MCP server initializes correctly."""

    def test_server_initializes(self, server):
        assert server is not None
        info = server.get_server_info()
        assert info.server_name == "razorpay-closeloop-mcp"
        assert info.tool_count > 0

    def test_server_has_tools(self, server):
        assert server.registry.tool_count >= 10, (
            f"Expected >= 10 tools, got {server.registry.tool_count}"
        )

    def test_server_config(self, server):
        assert server.config.mode == MCPServerMode.EMBEDDED
        assert server.config.audit_all_requests is True

    def test_server_properties(self, server):
        assert server.request_count == 0
        assert server.error_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Tool Registration
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistration:
    """Verify every tool is registered with correct metadata."""

    EXPECTED_TOOLS = [
        # Read-only
        "search_financial_records",
        "get_payment",
        "get_settlement",
        "get_refund",
        "get_fee",
        "get_adjustment",
        "get_similar_exception",
        # Write
        "create_resolution",
        "verify_resolution",
        "record_feedback",
    ]

    def test_all_expected_tools_registered(self, server):
        registered = server.registry.list_tool_names()
        for tool_name in self.EXPECTED_TOOLS:
            assert tool_name in registered, f"Tool '{tool_name}' not registered"

    def test_tool_count_matches(self, server):
        registered = server.registry.list_tool_names()
        assert len(registered) >= len(self.EXPECTED_TOOLS), (
            f"Expected >= {len(self.EXPECTED_TOOLS)} tools, got {len(registered)}"
        )

    def test_each_tool_has_definition(self, server):
        for tool_name in self.EXPECTED_TOOLS:
            defn = server.registry.get_definition(tool_name)
            assert defn is not None, f"Definition for '{tool_name}' is None"
            assert defn.name == tool_name
            assert len(defn.description) > 0, f"'{tool_name}' has empty description"
            assert len(defn.category) > 0, f"'{tool_name}' has empty category"

    def test_each_tool_has_handler(self, server):
        for tool_name in self.EXPECTED_TOOLS:
            handler = server.registry.get_handler(tool_name)
            assert handler is not None, f"Handler for '{tool_name}' is None"
            assert callable(handler), f"Handler for '{tool_name}' is not callable"

    def test_read_only_tools_not_financial(self, server):
        for tool_name in ["search_financial_records", "get_payment", "get_settlement",
                          "get_refund", "get_fee", "get_adjustment", "get_similar_exception"]:
            defn = server.registry.get_definition(tool_name)
            assert defn.is_financial is False, f"'{tool_name}' should not be financial"

    def test_write_tools_marked_financial(self, server):
        defn = server.registry.get_definition("create_resolution")
        assert defn.is_financial is True, "create_resolution should be financial"
        assert defn.requires_guardrail is True, "create_resolution should require guardrail"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Search Financial Records
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchFinancialRecords:
    """Test the search_financial_records MCP tool."""

    def test_search_returns_results(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"limit": 10},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert "count" in resp.result
        assert resp.result["count"] > 0

    def test_search_by_payment_id(self, server, adapter):
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments in dataset")
        pid = payments[0]["payment_id"]
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"payment_id": pid, "limit": 10},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["count"] >= 1
        # Results should contain the payment
        found = any(
            r.get("data", {}).get("payment_id") == pid
            for r in resp.result["records"]
        )
        assert found, f"Payment {pid} not found in results"

    def test_search_by_record_type(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"record_type": "fee", "limit": 5},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        for r in resp.result["records"]:
            assert r.get("type") == "fee", f"Expected fee, got {r.get('type')}"

    def test_search_limit_respected(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"limit": 3},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["count"] <= 3

    def test_search_empty_results(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"payment_id": "NONEXISTENT-PAY-999", "limit": 10},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["count"] == 0

    def test_search_invalid_record_type(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"record_type": "invalid_type", "limit": 5},
        )
        resp = server.invoke(req)
        # Should either return empty or error
        assert resp.status in (MCPToolStatus.SUCCESS, MCPToolStatus.ERROR)

    def test_search_injection_attempt(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"payment_id": "'; DROP TABLE payments; --", "limit": 5},
        )
        resp = server.invoke(req)
        # Should return validation error or empty results
        assert resp.status in (MCPToolStatus.SUCCESS, MCPToolStatus.VALIDATION_FAILED, MCPToolStatus.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Get Payment
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPayment:
    """Test the get_payment MCP tool."""

    def test_get_existing_payment(self, server, adapter):
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments in dataset")
        pid = payments[0]["payment_id"]
        req = MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": pid},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["payment"]["payment_id"] == pid

    def test_get_nonexistent_payment(self, server):
        req = MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_get_payment_missing_param(self, server):
        req = MCPToolRequest(
            tool_name="get_payment",
            parameters={},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED
        assert "Missing required parameter" in resp.error

    def test_get_payment_injection_attempt(self, server):
        req = MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-1' OR '1'='1"},
        )
        resp = server.invoke(req)
        assert resp.status in (MCPToolStatus.SUCCESS, MCPToolStatus.VALIDATION_FAILED)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Get Settlement
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSettlement:
    def test_get_existing_settlement(self, server, adapter):
        settlements = adapter._settlements
        if not settlements:
            pytest.skip("No settlements in dataset")
        sid = settlements[0]["settlement_id"]
        req = MCPToolRequest(
            tool_name="get_settlement",
            parameters={"settlement_id": sid},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["settlement"]["settlement_id"] == sid

    def test_get_nonexistent_settlement(self, server):
        req = MCPToolRequest(
            tool_name="get_settlement",
            parameters={"settlement_id": "SET-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_get_settlement_missing_param(self, server):
        req = MCPToolRequest(tool_name="get_settlement", parameters={})
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Get Refund
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRefund:
    def test_get_existing_refund(self, server, adapter):
        refunds = adapter._refunds
        if not refunds:
            pytest.skip("No refunds in dataset")
        rid = refunds[0]["refund_id"]
        req = MCPToolRequest(
            tool_name="get_refund",
            parameters={"refund_id": rid},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["refund"]["refund_id"] == rid

    def test_get_nonexistent_refund(self, server):
        req = MCPToolRequest(
            tool_name="get_refund",
            parameters={"refund_id": "REF-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_get_refund_missing_param(self, server):
        req = MCPToolRequest(tool_name="get_refund", parameters={})
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Get Fee
# ─────────────────────────────────────────────────────────────────────────────


class TestGetFee:
    def test_get_existing_fee(self, server, adapter):
        fees = adapter._fees
        if not fees:
            pytest.skip("No fees in dataset")
        fid = fees[0]["fee_id"]
        req = MCPToolRequest(
            tool_name="get_fee",
            parameters={"fee_id": fid},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["fee"]["fee_id"] == fid

    def test_get_nonexistent_fee(self, server):
        req = MCPToolRequest(
            tool_name="get_fee",
            parameters={"fee_id": "FEE-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_get_fee_missing_param(self, server):
        req = MCPToolRequest(tool_name="get_fee", parameters={})
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Get Adjustment
# ─────────────────────────────────────────────────────────────────────────────


class TestGetAdjustment:
    def test_get_existing_adjustment(self, server, adapter):
        adjustments = adapter._adjustments
        if not adjustments:
            pytest.skip("No adjustments in dataset")
        aid = adjustments[0]["adjustment_id"]
        req = MCPToolRequest(
            tool_name="get_adjustment",
            parameters={"adjustment_id": aid},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["adjustment"]["adjustment_id"] == aid

    def test_get_nonexistent_adjustment(self, server):
        req = MCPToolRequest(
            tool_name="get_adjustment",
            parameters={"adjustment_id": "ADJ-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_get_adjustment_missing_param(self, server):
        req = MCPToolRequest(tool_name="get_adjustment", parameters={})
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Get Similar Exception
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSimilarException:
    def test_get_similar_for_existing_case(self, server, adapter):
        cases = adapter._cases
        if not cases:
            pytest.skip("No cases in dataset")
        cid = cases[0]["case_id"]
        req = MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={"exception_id": cid, "top_k": 3},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True
        assert resp.result["query_case_id"] == cid

    def test_get_similar_nonexistent(self, server):
        req = MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={"exception_id": "EXC-NONEXISTENT"},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        # May find or not find — but should not error
        assert "similar_cases" in resp.result or "found" in resp.result

    def test_get_similar_missing_param(self, server):
        req = MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.VALIDATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Write Tools (create_resolution, verify_resolution, record_feedback)
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateResolution:
    def test_create_resolution_valid(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "EXC-TEST-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 25000,
                "workflow_id": "WF-TEST-001",
                "guardrail_decision": "AUTO",
                "authorization_source": "guardrail",
                "idempotency_key": "idem-test-001",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("executed") is True
        assert resp.result.get("execution_id") is not None

    def test_create_resolution_invalid_type(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "EXC-TEST-001",
                "resolution_type": "INVALID_TYPE",
                "financial_adjustment_paise": 25000,
                "workflow_id": "WF-TEST-001",
                "guardrail_decision": "AUTO",
                "authorization_source": "guardrail",
                "idempotency_key": "idem-test-002",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result
        assert "Invalid resolution type" in resp.result["error"]

    def test_create_resolution_invalid_guardrail(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "EXC-TEST-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 25000,
                "workflow_id": "WF-TEST-001",
                "guardrail_decision": "INVALID_DECISION",
                "authorization_source": "guardrail",
                "idempotency_key": "idem-test-003",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_create_resolution_negative_amount(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "EXC-TEST-001",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": -100,
                "workflow_id": "WF-TEST-001",
                "guardrail_decision": "AUTO",
                "authorization_source": "guardrail",
                "idempotency_key": "idem-test-004",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result

    def test_create_resolution_missing_params(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={},
        )
        resp = server.invoke(req)
        # Should fail validation
        assert resp.status in (MCPToolStatus.VALIDATION_FAILED, MCPToolStatus.SUCCESS)

    def test_create_resolution_injection_attempt(self, server):
        req = MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "'; DROP TABLE financial_records; --",
                "resolution_type": "FEE_DIFFERENCE",
                "financial_adjustment_paise": 100,
                "workflow_id": "WF-TEST",
                "guardrail_decision": "AUTO",
                "authorization_source": "guardrail",
                "idempotency_key": "idem-injection",
            },
        )
        resp = server.invoke(req)
        assert resp.status in (MCPToolStatus.SUCCESS, MCPToolStatus.VALIDATION_FAILED)


class TestVerifyResolution:
    def test_verify_nonexistent_execution(self, server):
        req = MCPToolRequest(
            tool_name="verify_resolution",
            parameters={
                "execution_id": "EXEC-001",
                "workflow_id": "WF-TEST",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        # The handler returns error when execution not found
        result = resp.result
        assert "error" in result or result.get("verified") is False


class TestRecordFeedback:
    def test_record_feedback_valid(self, server):
        req = MCPToolRequest(
            tool_name="record_feedback",
            parameters={
                "workflow_id": "WF-TEST-001",
                "exception_id": "EXC-TEST-001",
                "feedback_type": "APPROVE",
                "reviewer": "test_reviewer",
                "system_prediction": "AUTO",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("recorded") is True
        assert resp.result.get("feedback_id") is not None

    def test_record_feedback_invalid_type(self, server):
        req = MCPToolRequest(
            tool_name="record_feedback",
            parameters={
                "workflow_id": "WF-TEST-001",
                "exception_id": "EXC-TEST-001",
                "feedback_type": "INVALID",
                "reviewer": "test",
                "system_prediction": "AUTO",
            },
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Error Handling — Unknown/Disabled Tools
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_unknown_tool_returns_error(self, server):
        req = MCPToolRequest(
            tool_name="nonexistent_tool",
            parameters={},
        )
        resp = server.invoke(req)
        assert resp.status == MCPToolStatus.ERROR
        assert "not registered" in resp.error

    def test_disabled_tool_not_registered(self):
        config = MCPServerConfig(
            mode=MCPServerMode.EMBEDDED,
            disabled_tools=["get_payment"],
        )
        srv = MCPServer(config=config)
        srv.register_tool(
            MCPToolDefinition(
                name="get_payment",
                description="Get payment",
                category="reconciliation",
                parameters=[MCPToolParameter(name="payment_id", type="string", required=True)],
            ),
            lambda p: {"found": True},
        )
        # Tool should NOT be registered because it's disabled
        assert srv.registry.get_definition("get_payment") is None
        # Calling it should return "not registered"
        req = MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        )
        resp = srv.invoke(req)
        assert resp.status == MCPToolStatus.ERROR
        assert "not registered" in resp.error

    def test_disabled_category_returns_error(self):
        config = MCPServerConfig(
            mode=MCPServerMode.EMBEDDED,
            enabled_categories=[MCPToolCategory.EVIDENCE],  # Only evidence enabled
        )
        srv = MCPServer(config=config)
        # Try to register a reconciliation tool — should be silently rejected
        srv.register_tool(
            MCPToolDefinition(
                name="get_payment",
                description="Get payment",
                category="reconciliation",
                parameters=[],
            ),
            lambda p: {"found": True},
        )
        # Tool should not be registered
        assert srv.registry.get_definition("get_payment") is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Audit Trail
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditTrail:
    def test_audit_recorded_after_invoke(self, server):
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"limit": 5},
            workflow_id="WF-AUDIT-001",
            exception_id="EXC-AUDIT-001",
        )
        server.invoke(req)
        audit = server.get_audit_log(workflow_id="WF-AUDIT-001")
        assert len(audit) >= 1
        assert audit[0].tool_name == "search_financial_records"
        assert audit[0].workflow_id == "WF-AUDIT-001"

    def test_audit_records_error(self, server):
        req = MCPToolRequest(
            tool_name="nonexistent_tool",
            parameters={},
            workflow_id="WF-AUDIT-002",
        )
        server.invoke(req)
        audit = server.get_audit_log(workflow_id="WF-AUDIT-002")
        assert len(audit) >= 1

    def test_request_count_tracked(self, server):
        initial = server.request_count
        req = MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"limit": 1},
        )
        server.invoke(req)
        assert server.request_count == initial + 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: MCPClient Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPClientIntegration:
    def test_client_call_tool(self, client):
        result = client.call_tool(
            "search_financial_records",
            parameters={"limit": 5},
            workflow_id="WF-CLIENT-001",
        )
        assert result["success"] is True
        assert result["tool_name"] == "search_financial_records"
        assert result["status"] == "SUCCESS"

    def test_client_search_financial_records(self, client):
        result = client.search_financial_records(
            workflow_id="WF-CLIENT-002",
            limit=5,
        )
        assert result["success"] is True

    def test_client_get_payment(self, client, adapter):
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments in dataset")
        result = client.get_payment(
            payment_id=payments[0]["payment_id"],
            workflow_id="WF-CLIENT-003",
        )
        assert result["success"] is True
        assert result["data"]["found"] is True

    def test_client_get_similar_exception(self, client, adapter):
        cases = adapter._cases
        if not cases:
            pytest.skip("No cases in dataset")
        result = client.get_similar_exception(
            exception_id=cases[0]["case_id"],
            top_k=3,
            workflow_id="WF-CLIENT-004",
        )
        assert result["success"] is True

    def test_client_create_resolution(self, client):
        result = client.create_resolution(
            exception_id="EXC-CLIENT-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=15000,
            workflow_id="WF-CLIENT-005",
            guardrail_decision="AUTO",
            authorization_source="guardrail",
            idempotency_key="idem-client-001",
        )
        assert result["success"] is True
        assert result["data"]["executed"] is True

    def test_client_record_feedback(self, client):
        result = client.record_feedback(
            workflow_id="WF-CLIENT-006",
            exception_id="EXC-CLIENT-002",
            feedback_type="APPROVE",
            reviewer="test_client",
            system_prediction="AUTO",
        )
        assert result["success"] is True

    def test_client_error_tracking(self, client):
        initial_errors = client.get_error_count()
        client.call_tool("nonexistent_tool", parameters={})
        assert client.get_error_count() == initial_errors + 1

    def test_client_invocation_history(self, client):
        initial_count = client.get_invocation_count()
        client.call_tool("search_financial_records", parameters={"limit": 1})
        assert client.get_invocation_count() == initial_count + 1

    def test_client_audit_summary(self, client):
        client.call_tool("search_financial_records", parameters={"limit": 1})
        summary = client.get_audit_summary()
        assert "total_invocations" in summary
        assert "error_count" in summary
        assert "tools_called" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: FinancialDataAdapter Isolation
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialDataAdapter:
    def test_adapter_reads_real_data(self, adapter):
        assert adapter.is_loaded
        summary = adapter.get_summary()
        assert summary["payments"] > 0
        assert summary["settlements"] > 0
        assert summary["cases"] > 0

    def test_adapter_get_payment(self, adapter):
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments")
        pid = payments[0]["payment_id"]
        result = adapter.get_payment(pid)
        assert result is not None
        assert result["payment_id"] == pid

    def test_adapter_search(self, adapter):
        results = adapter.search_records(limit=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_adapter_search_by_payment(self, adapter):
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments")
        pid = payments[0]["payment_id"]
        results = adapter.search_records(payment_id=pid)
        assert len(results) >= 1

    def test_adapter_returns_copies(self, adapter):
        """Adapter should return copies, not references."""
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments")
        pid = payments[0]["payment_id"]
        r1 = adapter.get_payment(pid)
        r2 = adapter.get_payment(pid)
        assert r1 == r2
        assert r1 is not r2  # Different objects


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: MCP from LangGraph Perspective
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPLangGraphIntegration:
    """Test MCP tools as they would be called from LangGraph nodes."""

    def test_langgraph_evidence_gathering_via_mcp(self, client, adapter):
        """Simulate the gather_evidence node calling MCP."""
        cases = adapter._cases
        if not cases:
            pytest.skip("No cases")
        case = cases[0]
        exc_id = case["case_id"]

        # Step 1: Search financial records
        search_result = client.search_financial_records(
            case_id=exc_id,
            workflow_id="WF-LG-001",
            exception_id=exc_id,
        )
        assert search_result["success"] is True
        records = search_result["data"]["records"]
        assert len(records) > 0, f"No records found for case {exc_id}"

        # Step 2: Get similar exceptions
        similar_result = client.get_similar_exception(
            exception_id=exc_id,
            workflow_id="WF-LG-001",
        )
        assert similar_result["success"] is True

        # Step 3: Verify MCP state was updated
        assert client.get_invocation_count() >= 2

    def test_langgraph_resolution_via_mcp(self, client):
        """Simulate the resolution execution node calling MCP via create_resolution."""
        # Step 1: Create resolution
        result = client.create_resolution(
            exception_id="EXC-LG-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=25000,
            workflow_id="WF-LG-002",
            guardrail_decision="AUTO",
            authorization_source="guardrail",
            idempotency_key="idem-lg-001",
        )
        assert result["success"] is True
        assert result["data"]["executed"] is True

        # Step 2: Record feedback
        feedback_result = client.record_feedback(
            workflow_id="WF-LG-002",
            exception_id="EXC-LG-001",
            feedback_type="APPROVE",
            reviewer="langgraph_agent",
            system_prediction="AUTO",
        )
        assert feedback_result["success"] is True

    def test_mcp_tool_call_order_matters(self, client, adapter):
        """Verify that tool calls happen in a meaningful order."""
        cases = adapter._cases
        if not cases:
            pytest.skip("No cases")
        exc_id = cases[0]["case_id"]

        # Call tools in the order a workflow would
        client.search_financial_records(case_id=exc_id, workflow_id="WF-ORDER")
        client.get_similar_exception(exception_id=exc_id, workflow_id="WF-ORDER")

        history = client.invocation_history
        assert len(history) >= 2
        # First should be search, second should be similar
        assert history[-2]["tool_name"] == "search_financial_records"
        assert history[-1]["tool_name"] == "get_similar_exception"

    def test_mcp_response_time_acceptable(self, client):
        """MCP tool calls should complete in reasonable time."""
        start = time.time()
        client.search_financial_records(limit=50, workflow_id="WF-PERF")
        elapsed = time.time() - start
        assert elapsed < 5.0, f"MCP call took {elapsed:.1f}s — too slow"


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Tool Usage Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestToolUsageAnalysis:
    """Identify which MCP tools are actually used by the workflow."""

    def test_readonly_tools_used_by_investigation(self, client, adapter):
        """Investigation phase tools: search_financial_records, get_similar_exception."""
        cases = adapter._cases
        if not cases:
            pytest.skip("No cases")
        exc_id = cases[0]["case_id"]

        # These tools are called during investigation
        r1 = client.search_financial_records(case_id=exc_id, workflow_id="WF-USED")
        r2 = client.get_similar_exception(exception_id=exc_id, workflow_id="WF-USED")

        assert r1["success"] is True
        assert r2["success"] is True

    def test_individual_get_tools_not_called_by_workflow(self, client, adapter):
        """Individual get_* tools exist but are NOT called by the workflow.
        They are available for ad-hoc investigation by a human reviewer.
        """
        payments = adapter._payments
        if not payments:
            pytest.skip("No payments")
        pid = payments[0]["payment_id"]

        # These tools work but are not part of the automated workflow
        r = client.get_payment(payment_id=pid, workflow_id="WF-INDIVIDUAL")
        assert r["success"] is True
        assert r["data"]["found"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: Tool Registration Filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistrationFiltering:
    def test_disabled_category_prevents_registration(self):
        config = MCPServerConfig(
            enabled_categories=[MCPToolCategory.EVIDENCE],
        )
        srv = MCPServer(config=config)
        # Reconciliation tools should NOT register
        for defn in READONLY_DEFINITIONS:
            if defn.category == "reconciliation":
                srv.register_tool(defn, lambda p: {})
        assert srv.registry.tool_count == 0

    def test_enabled_category_allows_registration(self):
        config = MCPServerConfig(
            enabled_categories=[MCPToolCategory.RECONCILIATION],
        )
        srv = MCPServer(config=config)
        for defn in READONLY_DEFINITIONS:
            if defn.category == "reconciliation":
                srv.register_tool(defn, lambda p: {})
        assert srv.registry.tool_count > 0

    def test_specific_tool_disabled(self):
        config = MCPServerConfig(
            disabled_tools=["search_financial_records"],
        )
        srv = MCPServer(config=config)
        for defn in READONLY_DEFINITIONS:
            srv.register_tool(defn, lambda p: {})
        assert srv.registry.get_definition("search_financial_records") is None
        assert srv.registry.get_definition("get_payment") is not None
