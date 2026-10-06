"""
Phase 11 tests — controlled MCP/tool boundary.

Every tool is a narrow domain capability with structured input/output. The
boundary exposes no raw SQL, no arbitrary code execution, no raw ORM session,
no provider mutation API and no secrets.
"""

from __future__ import annotations

import pytest

from app.orchestration.tools import ClosedLoopToolkit, TOOL_DEFINITIONS
from mcp.schemas import MCPToolRequest, MCPToolStatus
from mcp.server import MCPServer

from test_phase11_langgraph_orchestration import (
    EXCEPTION_ID,
    _detect_fee_mismatch,
    _make_client,
    _stage_fee_mismatch_provider,
)

EXPECTED_TOOLS = {
    "get_exception_context",
    "get_evidence",
    "get_reconciliation_result",
    "classify_exception",
    "retrieve_historical_cases",
    "generate_resolution_proposals",
    "evaluate_resolution_policy",
    "get_authorization_decision",
    "request_authorized_execution",
    "verify_post_execution_reconciliation",
    "get_verification_status",
}


def test_tool_allowlist_is_narrow():
    names = {d.name for d in TOOL_DEFINITIONS}
    assert names == EXPECTED_TOOLS


def test_read_tools_are_not_financial():
    read_tools = {
        "get_exception_context",
        "get_evidence",
        "get_reconciliation_result",
        "classify_exception",
        "retrieve_historical_cases",
        "get_reconciliation_result",
    }
    for definition in TOOL_DEFINITIONS:
        if definition.name in read_tools:
            assert definition.is_financial is False
            assert definition.requires_guardrail is False


def test_execution_tool_is_financial_and_guardrailed():
    tool = next(d for d in TOOL_DEFINITIONS if d.name == "request_authorized_execution")
    assert tool.is_financial is True
    assert tool.requires_guardrail is True
    assert tool.requires_verification is True


def test_tools_expose_no_secret_parameters():
    banned = ("secret", "password", "api_key", "token", "credential", "connection")
    for definition in TOOL_DEFINITIONS:
        lowered = definition.name.lower()
        for bad in banned:
            assert bad not in lowered
        for param in definition.parameters:
            pl = param.name.lower()
            for bad in banned:
                assert bad not in pl


def test_tools_do_not_expose_session_or_provider():
    for definition in TOOL_DEFINITIONS:
        for param in definition.parameters:
            assert param.name not in {"session", "provider", "db", "sessionmaker"}


def test_unknown_tool_is_rejected():
    server = MCPServer()
    response = server.invoke(MCPToolRequest(tool_name="does_not_exist"))
    assert response.status == MCPToolStatus.ERROR


def test_read_tool_returns_detached_copy(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    result = toolkit.get_exception_context({"exception_id": EXCEPTION_ID})
    assert result["found"] is True
    result["exception"]["status"] = "TAMPERED"
    fresh = toolkit.get_exception_context({"exception_id": EXCEPTION_ID})
    assert fresh["exception"]["status"] != "TAMPERED"


def test_tools_are_registered_and_audited(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    _toolkit, server, _client = _make_client(db_session, provider)
    assert server.registry.tool_count == len(TOOL_DEFINITIONS)
    server.invoke(
        MCPToolRequest(
            tool_name="get_exception_context",
            parameters={"exception_id": EXCEPTION_ID},
            workflow_id="WF-P11-MCP",
        )
    )
    audit = server.get_audit_log()
    assert len(audit) == 1
    assert audit[0].tool_name == "get_exception_context"
    assert audit[0].is_financial is False


def test_execution_tool_requires_identity(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    result = toolkit.request_authorized_execution({"exception_id": EXCEPTION_ID})
    assert result["executed"] is False
    assert "workflow_id" in result["error"]


def test_verification_tool_requires_execution(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    result = toolkit.verify_post_execution_reconciliation(
        {"exception_id": EXCEPTION_ID}
    )
    assert result["verified"] is False


def test_toolkit_has_no_sql_or_code_execution_surface():
    for attribute in ("execute_sql", "execute", "query", "raw", "eval", "exec", "compile"):
        assert not hasattr(ClosedLoopToolkit, attribute)
