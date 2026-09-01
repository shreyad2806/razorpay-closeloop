"""
Tests for Razorpay CloseLoop Phase 11A — MCP Server Foundation.

Verifies MCP server, tool registry, schemas, configuration, validation, audit.
"""

import pytest
from datetime import datetime, timezone

from mcp.config import MCPServerConfig, MCPServerMode, MCPToolCategory
from mcp.schemas import (
    MCPAuditRecord,
    MCPServerInfo,
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolRequest,
    MCPToolResponse,
    MCPToolStatus,
)
from mcp.tools.registry import MCPToolRegistry
from mcp.server import MCPServer
from mcp.validation import validate_request, validate_parameters
from mcp.audit import MCPAuditLogger


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> MCPServerConfig:
    return MCPServerConfig()


@pytest.fixture
def registry() -> MCPToolRegistry:
    return MCPToolRegistry()


@pytest.fixture
def server(config: MCPServerConfig) -> MCPServer:
    return MCPServer(config)


@pytest.fixture
def audit_logger() -> MCPAuditLogger:
    return MCPAuditLogger()


def _make_tool_def(
    name: str = "test_tool",
    category: str = "evidence",
    required_params: list = None,
    is_financial: bool = False,
    requires_guardrail: bool = False,
) -> MCPToolDefinition:
    params = []
    for p in (required_params or []):
        params.append(MCPToolParameter(
            name=p, type="string", required=True,
        ))
    return MCPToolDefinition(
        name=name,
        description=f"Test tool: {name}",
        category=category,
        parameters=params,
        is_financial=is_financial,
        requires_guardrail=requires_guardrail,
    )


def _make_handler(return_value=None):
    def handler(params):
        return return_value or {"status": "ok", "params": params}
    return handler


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = MCPServerConfig()
        assert cfg.server_name == "razorpay-closeloop-mcp"
        assert cfg.mode == MCPServerMode.EMBEDDED
        assert cfg.require_guardrail_approval is True
        assert cfg.require_verification is True
        assert cfg.audit_all_requests is True
        assert cfg.max_concurrent_requests == 10

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_SERVER_NAME", "test-server")
        monkeypatch.setenv("MCP_MODE", "http")
        monkeypatch.setenv("MCP_PORT", "9090")
        monkeypatch.setenv("MCP_REQUIRE_GUARDRAIL", "false")
        cfg = MCPServerConfig.from_env()
        assert cfg.server_name == "test-server"
        assert cfg.mode == MCPServerMode.HTTP
        assert cfg.port == 9090
        assert cfg.require_guardrail_approval is False

    def test_all_categories_enabled_by_default(self):
        cfg = MCPServerConfig()
        assert len(cfg.enabled_categories) == len(MCPToolCategory)

    def test_config_frozen(self):
        cfg = MCPServerConfig()
        with pytest.raises(Exception):
            cfg.server_name = "changed"


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────


class TestToolDefinition:
    def test_basic_definition(self):
        tool = MCPToolDefinition(
            name="get_evidence",
            description="Get evidence for an exception",
            category="evidence",
        )
        assert tool.name == "get_evidence"
        assert tool.requires_guardrail is False
        assert tool.is_financial is False

    def test_financial_tool(self):
        tool = MCPToolDefinition(
            name="execute_resolution",
            description="Execute a financial resolution",
            category="execution",
            is_financial=True,
            requires_guardrail=True,
            requires_verification=True,
        )
        assert tool.is_financial is True
        assert tool.requires_guardrail is True

    def test_tool_with_parameters(self):
        tool = MCPToolDefinition(
            name="reconcile",
            description="Reconcile exception",
            category="reconciliation",
            parameters=[
                MCPToolParameter(name="exception_id", type="string", required=True),
                MCPToolParameter(name="amount", type="number", required=False, default=0),
            ],
        )
        assert len(tool.parameters) == 2
        assert tool.parameters[0].required is True
        assert tool.parameters[1].default == 0


class TestToolRequest:
    def test_basic_request(self):
        req = MCPToolRequest(
            tool_name="get_evidence",
            parameters={"exception_id": "exc-001"},
        )
        assert req.tool_name == "get_evidence"
        assert req.idempotency_key is None

    def test_request_with_idempotency(self):
        req = MCPToolRequest(
            tool_name="execute_resolution",
            parameters={},
            idempotency_key="idem-001",
            workflow_id="wf-001",
        )
        assert req.idempotency_key == "idem-001"


class TestToolResponse:
    def test_success_response(self):
        resp = MCPToolResponse(
            request_id="REQ-001",
            tool_name="test",
            status=MCPToolStatus.SUCCESS,
            result={"key": "value"},
        )
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result == {"key": "value"}

    def test_error_response(self):
        resp = MCPToolResponse(
            request_id="REQ-002",
            tool_name="test",
            status=MCPToolStatus.ERROR,
            error="Something failed",
        )
        assert resp.status == MCPToolStatus.ERROR
        assert resp.error == "Something failed"

    def test_guardrail_blocked(self):
        resp = MCPToolResponse(
            request_id="REQ-003",
            tool_name="test",
            status=MCPToolStatus.GUARDRAIL_BLOCKED,
            error="Blocked by guardrails",
        )
        assert resp.status == MCPToolStatus.GUARDRAIL_BLOCKED


class TestToolStatus:
    def test_all_statuses(self):
        assert MCPToolStatus.SUCCESS.value == "SUCCESS"
        assert MCPToolStatus.ERROR.value == "ERROR"
        assert MCPToolStatus.GUARDRAIL_BLOCKED.value == "GUARDRAIL_BLOCKED"
        assert MCPToolStatus.VALIDATION_FAILED.value == "VALIDATION_FAILED"
        assert MCPToolStatus.TIMEOUT.value == "TIMEOUT"
        assert MCPToolStatus.UNAVAILABLE.value == "UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_register_tool(self, registry: MCPToolRegistry):
        defn = _make_tool_def("my_tool")
        handler = _make_handler()
        registry.register_tool(defn, handler)
        assert registry.has_tool("my_tool")
        assert registry.tool_count == 1

    def test_get_definition(self, registry: MCPToolRegistry):
        defn = _make_tool_def("my_tool")
        registry.register_tool(defn, _make_handler())
        found = registry.get_definition("my_tool")
        assert found is not None
        assert found.name == "my_tool"

    def test_get_handler(self, registry: MCPToolRegistry):
        defn = _make_tool_def("my_tool")
        handler = _make_handler({"result": 42})
        registry.register_tool(defn, handler)
        h = registry.get_handler("my_tool")
        assert h is not None
        assert h({}) == {"result": 42}

    def test_list_tools(self, registry: MCPToolRegistry):
        registry.register_tool(_make_tool_def("t1", "evidence"), _make_handler())
        registry.register_tool(_make_tool_def("t2", "evidence"), _make_handler())
        registry.register_tool(_make_tool_def("t3", "reconciliation"), _make_handler())
        assert len(registry.list_tools()) == 3
        assert len(registry.list_tools("evidence")) == 2
        assert len(registry.list_tools("reconciliation")) == 1

    def test_list_tool_names(self, registry: MCPToolRegistry):
        registry.register_tool(_make_tool_def("a"), _make_handler())
        registry.register_tool(_make_tool_def("b"), _make_handler())
        names = registry.list_tool_names()
        assert "a" in names
        assert "b" in names

    def test_get_tools_by_category(self, registry: MCPToolRegistry):
        registry.register_tool(_make_tool_def("t1", "evidence"), _make_handler())
        registry.register_tool(_make_tool_def("t2", "evidence"), _make_handler())
        grouped = registry.get_tools_by_category()
        assert "evidence" in grouped
        assert len(grouped["evidence"]) == 2

    def test_unregister_tool(self, registry: MCPToolRegistry):
        registry.register_tool(_make_tool_def("my_tool"), _make_handler())
        assert registry.unregister_tool("my_tool") is True
        assert registry.has_tool("my_tool") is False
        assert registry.unregister_tool("nonexistent") is False

    def test_invoke_tool_success(self, registry: MCPToolRegistry):
        defn = _make_tool_def("my_tool")
        handler = _make_handler({"key": "value"})
        registry.register_tool(defn, handler)
        resp = registry.invoke_tool(MCPToolRequest(tool_name="my_tool"))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result == {"key": "value"}

    def test_invoke_tool_not_found(self, registry: MCPToolRegistry):
        resp = registry.invoke_tool(MCPToolRequest(tool_name="nonexistent"))
        assert resp.status == MCPToolStatus.ERROR
        assert "not found" in resp.error

    def test_invoke_tool_no_handler(self, registry: MCPToolRegistry):
        defn = _make_tool_def("no_handler")
        # Register definition only, no handler
        registry._definitions["no_handler"] = defn
        resp = registry.invoke_tool(MCPToolRequest(tool_name="no_handler"))
        assert resp.status == MCPToolStatus.UNAVAILABLE

    def test_invoke_tool_missing_required_param(self, registry: MCPToolRegistry):
        defn = _make_tool_def("param_tool", required_params=["exception_id"])
        registry.register_tool(defn, _make_handler())
        resp = registry.invoke_tool(
            MCPToolRequest(tool_name="param_tool", parameters={})
        )
        assert resp.status == MCPToolStatus.VALIDATION_FAILED
        assert "exception_id" in resp.error

    def test_invoke_tool_with_params(self, registry: MCPToolRegistry):
        defn = _make_tool_def("param_tool", required_params=["exception_id"])
        registry.register_tool(defn, _make_handler())
        resp = registry.invoke_tool(
            MCPToolRequest(
                tool_name="param_tool",
                parameters={"exception_id": "exc-001"},
            )
        )
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["params"]["exception_id"] == "exc-001"

    def test_invoke_tool_exception(self, registry: MCPToolRegistry):
        defn = _make_tool_def("crash_tool")
        def handler(params):
            raise RuntimeError("Tool crashed")
        registry.register_tool(defn, handler)
        resp = registry.invoke_tool(MCPToolRequest(tool_name="crash_tool"))
        assert resp.status == MCPToolStatus.ERROR
        assert "Tool crashed" in resp.error


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPServer:
    def test_server_creation(self, server: MCPServer):
        info = server.get_server_info()
        assert info.server_name == "razorpay-closeloop-mcp"
        assert info.tool_count == 0

    def test_register_tool(self, server: MCPServer):
        defn = _make_tool_def("my_tool", "evidence")
        server.register_tool(defn, _make_handler())
        assert server.registry.has_tool("my_tool")

    def test_register_disabled_category(self):
        config = MCPServerConfig(enabled_categories=[MCPToolCategory.EVIDENCE])
        server = MCPServer(config)
        defn = _make_tool_def("recon_tool", "reconciliation")
        server.register_tool(defn, _make_handler())
        # Should NOT be registered (category not enabled)
        assert server.registry.has_tool("recon_tool") is False

    def test_register_disabled_tool(self):
        config = MCPServerConfig(disabled_tools=["bad_tool"])
        server = MCPServer(config)
        defn = _make_tool_def("bad_tool")
        server.register_tool(defn, _make_handler())
        assert server.registry.has_tool("bad_tool") is False

    def test_invoke_tool(self, server: MCPServer):
        defn = _make_tool_def("my_tool")
        server.register_tool(defn, _make_handler({"result": "ok"}))
        resp = server.invoke(MCPToolRequest(tool_name="my_tool"))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result == {"result": "ok"}

    def test_invoke_unknown_tool(self, server: MCPServer):
        resp = server.invoke(MCPToolRequest(tool_name="nonexistent"))
        assert resp.status == MCPToolStatus.ERROR
        assert "not registered" in resp.error

    def test_invoke_disabled_tool(self):
        config = MCPServerConfig(disabled_tools=["disabled_tool"])
        server = MCPServer(config)
        defn = _make_tool_def("disabled_tool")
        server.register_tool(defn, _make_handler())
        resp = server.invoke(MCPToolRequest(tool_name="disabled_tool"))
        assert resp.status == MCPToolStatus.ERROR
        assert "disabled" in resp.error

    def test_invoke_disabled_category(self):
        config = MCPServerConfig(enabled_categories=[MCPToolCategory.EVIDENCE])
        server = MCPServer(config)
        defn = _make_tool_def("recon_tool", "reconciliation")
        # Won't be registered due to category filter
        server.register_tool(defn, _make_handler())
        resp = server.invoke(MCPToolRequest(tool_name="recon_tool"))
        assert resp.status == MCPToolStatus.ERROR

    def test_audit_logging(self, server: MCPServer):
        defn = _make_tool_def("my_tool")
        server.register_tool(defn, _make_handler())
        server.invoke(MCPToolRequest(tool_name="my_tool", request_id="REQ-001"))
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].tool_name == "my_tool"
        assert audit[0].status == MCPToolStatus.SUCCESS

    def test_audit_filtered(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        server.register_tool(_make_tool_def("t2"), _make_handler())
        server.invoke(MCPToolRequest(tool_name="t1"))
        server.invoke(MCPToolRequest(tool_name="t2"))
        audit = server.get_audit_log(tool_name="t1")
        assert len(audit) == 1
        assert audit[0].tool_name == "t1"

    def test_request_count(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        server.invoke(MCPToolRequest(tool_name="t1"))
        server.invoke(MCPToolRequest(tool_name="t1"))
        assert server.request_count == 2

    def test_error_count(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        server.invoke(MCPToolRequest(tool_name="t1"))
        server.invoke(MCPToolRequest(tool_name="nonexistent"))
        assert server.error_count == 1

    def test_server_info(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        info = server.get_server_info()
        assert info.tool_count == 1
        assert info.version == "1.0.0"
        assert info.uptime_seconds is not None
        assert info.uptime_seconds >= 0

    def test_request_id_generated(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        resp = server.invoke(MCPToolRequest(tool_name="t1"))
        assert resp.request_id.startswith("REQ-")

    def test_financial_tool_audited(self, server: MCPServer):
        defn = _make_tool_def("exec_tool", "execution", is_financial=True)
        server.register_tool(defn, _make_handler())
        server.invoke(MCPToolRequest(tool_name="exec_tool"))
        audit = server.get_audit_log()
        assert audit[0].is_financial is True

    def test_duration_recorded(self, server: MCPServer):
        server.register_tool(_make_tool_def("t1"), _make_handler())
        resp = server.invoke(MCPToolRequest(tool_name="t1"))
        assert resp.duration_ms is not None
        assert resp.duration_ms >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_request(self):
        defn = _make_tool_def("t1", required_params=["exc_id"])
        req = MCPToolRequest(tool_name="t1", parameters={"exc_id": "x"})
        ok, err = validate_request(req, defn)
        assert ok is True
        assert err is None

    def test_missing_required(self):
        defn = _make_tool_def("t1", required_params=["exc_id"])
        req = MCPToolRequest(tool_name="t1", parameters={})
        ok, err = validate_request(req, defn)
        assert ok is False
        assert "exc_id" in err

    def test_none_definition(self):
        req = MCPToolRequest(tool_name="t1")
        ok, err = validate_request(req, None)
        assert ok is False

    def test_validate_parameters_ok(self):
        defn = _make_tool_def("t1", required_params=["exc_id"])
        ok, err = validate_parameters({"exc_id": "x"}, defn)
        assert ok is True

    def test_validate_parameters_missing(self):
        defn = _make_tool_def("t1", required_params=["exc_id"])
        ok, err = validate_parameters({}, defn)
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_record(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="t1",
            category="evidence",
            parameters={},
            status=MCPToolStatus.SUCCESS,
        )
        assert rec.record_id.startswith("MCPAUD-")
        assert audit_logger.record_count == 1

    def test_query(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "e", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("R2", "t2", "e", {}, MCPToolStatus.ERROR)
        all_records = audit_logger.query()
        assert len(all_records) == 2

    def test_query_by_tool(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "e", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("R2", "t2", "e", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("R3", "t1", "e", {}, MCPToolStatus.SUCCESS)
        records = audit_logger.query(tool_name="t1")
        assert len(records) == 2

    def test_query_by_status(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "e", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("R2", "t1", "e", {}, MCPToolStatus.ERROR)
        failed = audit_logger.query(status=MCPToolStatus.ERROR)
        assert len(failed) == 1

    def test_financial_actions(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "e", {}, MCPToolStatus.SUCCESS, is_financial=False)
        audit_logger.record("R2", "t2", "x", {}, MCPToolStatus.SUCCESS, is_financial=True)
        fin = audit_logger.get_financial_actions()
        assert len(fin) == 1
        assert fin[0].is_financial is True

    def test_failed_requests(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "e", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("R2", "t1", "e", {}, MCPToolStatus.ERROR)
        audit_logger.record("R3", "t1", "e", {}, MCPToolStatus.GUARDRAIL_BLOCKED)
        failed = audit_logger.get_failed_requests()
        assert len(failed) == 2

    def test_max_records_trimmed(self):
        logger = MCPAuditLogger(max_records=3)
        for i in range(5):
            logger.record(f"R{i}", "t", "c", {}, MCPToolStatus.SUCCESS)
        assert logger.record_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Category Enums
# ─────────────────────────────────────────────────────────────────────────────


class TestCategories:
    def test_all_categories(self):
        cats = list(MCPToolCategory)
        assert len(cats) == 10
        assert MCPToolCategory.RECONCILIATION in cats
        assert MCPToolCategory.EVIDENCE in cats
        assert MCPToolCategory.GUARDRAILS in cats
        assert MCPToolCategory.EXECUTION in cats
        assert MCPToolCategory.LINEAGE in cats


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_mcp_server_no_financial_logic(self, server: MCPServer):
        """MCP server delegates to handlers, no financial logic."""
        assert not hasattr(server, 'execute_refund')
        assert not hasattr(server, 'modify_settlement')
        assert not hasattr(server, 'authorize_payment')

    def test_tool_registry_no_financial_logic(self, registry: MCPToolRegistry):
        """Tool registry only routes, no financial logic."""
        assert not hasattr(registry, 'execute_refund')
        assert not hasattr(registry, 'modify_settlement')

    def test_handler_delegates(self, server: MCPServer):
        """Tool handlers are registered externally, MCP just routes."""
        defn = _make_tool_def("t1")
        called_with = {}

        def handler(params):
            called_with.update(params)
            return {"delegated": True}

        server.register_tool(defn, handler)
        server.invoke(MCPToolRequest(
            tool_name="t1",
            parameters={"exc_id": "exc-001"},
        ))
        assert called_with == {"exc_id": "exc-001"}

    def test_audit_record_immutability(self, audit_logger: MCPAuditLogger):
        """Audit records are append-only."""
        rec = audit_logger.record(
            "R1", "t1", "c", {}, MCPToolStatus.SUCCESS
        )
        # Record should not have mutation methods
        assert not hasattr(rec, 'delete')
        assert not hasattr(rec, 'modify')


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Flow
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_register_invoke_audit(self):
        """Register tool → invoke → audit trail."""
        server = MCPServer()

        defn = MCPToolDefinition(
            name="get_evidence",
            description="Get evidence for exception",
            category="evidence",
            parameters=[
                MCPToolParameter(name="exception_id", type="string", required=True),
            ],
        )

        def get_evidence(params):
            return {
                "exception_id": params["exception_id"],
                "evidence": {"fees": [], "settlements": []},
            }

        server.register_tool(defn, get_evidence)

        # Invoke
        resp = server.invoke(MCPToolRequest(
            tool_name="get_evidence",
            parameters={"exception_id": "exc-001"},
            request_id="REQ-001",
            workflow_id="wf-001",
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["exception_id"] == "exc-001"

        # Audit
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].request_id == "REQ-001"
        assert audit[0].workflow_id == "wf-001"
        assert audit[0].guardrail_checked is False

    def test_multiple_tools_workflow(self):
        """Simulate a workflow with multiple tool invocations."""
        server = MCPServer()

        # Register tools
        tools = [
            (MCPToolDefinition(
                name="load_exception",
                description="Load exception data",
                category="reconciliation",
                parameters=[MCPToolParameter(name="exception_id", type="string", required=True)],
            ), lambda p: {"exception": {"id": p["exception_id"]}}),

            (MCPToolDefinition(
                name="gather_evidence",
                description="Gather financial evidence",
                category="evidence",
            ), lambda p: {"evidence": {"fees": [100]}}),

            (MCPToolDefinition(
                name="classify",
                description="Classify exception",
                category="classification",
            ), lambda p: {"classification": "FEE_DIFFERENCE", "confidence": 0.92}),
        ]

        for defn, handler in tools:
            server.register_tool(defn, handler)

        # Simulate workflow
        resp1 = server.invoke(MCPToolRequest(
            tool_name="load_exception",
            parameters={"exception_id": "exc-001"},
        ))
        assert resp1.status == MCPToolStatus.SUCCESS

        resp2 = server.invoke(MCPToolRequest(tool_name="gather_evidence"))
        assert resp2.status == MCPToolStatus.SUCCESS

        resp3 = server.invoke(MCPToolRequest(tool_name="classify"))
        assert resp3.status == MCPToolStatus.SUCCESS
        assert resp3.result["classification"] == "FEE_DIFFERENCE"

        # All audited
        assert server.request_count == 3
        audit = server.get_audit_log()
        assert len(audit) == 3
