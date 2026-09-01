"""
Tests for Razorpay CloseLoop Phase 11B — MCP Read-Only Financial Tools.

Verifies controlled read-only financial data access through MCP tools.
"""

import json
import os
import tempfile
import pytest
from typing import Any, Dict

from mcp.adapters.financial_data import FinancialDataAdapter
from mcp.tools.readonly import TOOL_DEFINITIONS, create_handlers
from mcp.tools.registry import MCPToolRegistry
from mcp.server import MCPServer
from mcp.schemas import MCPToolRequest, MCPToolStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary data directory with sample financial records."""
    batch_dir = tmp_path / "batch_001" / "generated"
    batch_dir.mkdir(parents=True)

    payments = [
        {"payment_id": "PAY-001", "merchant_id": "MER-001", "amount": 10000, "currency": "INR", "status": "captured"},
        {"payment_id": "PAY-002", "merchant_id": "MER-001", "amount": 20000, "currency": "INR", "status": "captured"},
        {"payment_id": "PAY-003", "merchant_id": "MER-002", "amount": 5000, "currency": "INR", "status": "captured"},
    ]
    settlements = [
        {"settlement_id": "SET-001", "payment_id": "PAY-001", "merchant_id": "MER-001", "amount": 9800, "status": "settled"},
        {"settlement_id": "SET-002", "payment_id": "PAY-002", "merchant_id": "MER-001", "amount": 19500, "status": "settled"},
    ]
    refunds = [
        {"refund_id": "REF-001", "payment_id": "PAY-001", "amount": 200, "status": "processed"},
    ]
    fees = [
        {"fee_id": "FEE-001", "payment_id": "PAY-001", "amount": 100, "fee_type": "platform_fee"},
        {"fee_id": "FEE-002", "payment_id": "PAY-002", "amount": 200, "fee_type": "platform_fee"},
    ]
    adjustments = [
        {"adjustment_id": "ADJ-001", "payment_id": "PAY-001", "amount": 50, "adjustment_type": "credit"},
    ]
    cases = [
        {"case_id": "CASE-001", "payment_id": "PAY-001", "merchant_id": "MER-001", "scenario": "FEE_DIFFERENCE", "difference": -200, "risk_category": "low"},
        {"case_id": "CASE-002", "payment_id": "PAY-002", "merchant_id": "MER-001", "scenario": "REFUND_ADJUSTMENT", "difference": -500, "risk_category": "medium"},
        {"case_id": "CASE-003", "payment_id": "PAY-003", "merchant_id": "MER-002", "scenario": "FEE_DIFFERENCE", "difference": -100, "risk_category": "low"},
    ]
    merchants = [
        {"merchant_id": "MER-001", "name": "Test Merchant 1"},
        {"merchant_id": "MER-002", "name": "Test Merchant 2"},
    ]

    for name, data in [
        ("payments.json", payments), ("settlements.json", settlements),
        ("refunds.json", refunds), ("fees.json", fees),
        ("adjustments.json", adjustments), ("cases.json", cases),
        ("merchants.json", merchants),
    ]:
        (batch_dir / name).write_text(json.dumps(data), encoding="utf-8")

    return str(tmp_path)


@pytest.fixture
def adapter(sample_data_dir) -> FinancialDataAdapter:
    a = FinancialDataAdapter(data_dir=sample_data_dir)
    a.load_batch("batch_001")
    return a


@pytest.fixture
def handlers(adapter: FinancialDataAdapter):
    return create_handlers(adapter)


@pytest.fixture
def server_with_tools(adapter: FinancialDataAdapter) -> MCPServer:
    server = MCPServer()
    handlers = create_handlers(adapter)
    for defn in TOOL_DEFINITIONS:
        if defn.name in handlers:
            server.register_tool(defn, handlers[defn.name])
    return server


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialDataAdapter:
    def test_load_batch(self, adapter: FinancialDataAdapter):
        assert adapter.is_loaded is True

    def test_load_nonexistent_batch(self, tmp_path):
        a = FinancialDataAdapter(data_dir=str(tmp_path))
        assert a.load_batch("nonexistent") is False
        assert a.is_loaded is False

    def test_get_payment(self, adapter: FinancialDataAdapter):
        p = adapter.get_payment("PAY-001")
        assert p is not None
        assert p["payment_id"] == "PAY-001"
        assert p["amount"] == 10000

    def test_get_payment_not_found(self, adapter: FinancialDataAdapter):
        assert adapter.get_payment("PAY-999") is None

    def test_get_settlement(self, adapter: FinancialDataAdapter):
        s = adapter.get_settlement("SET-001")
        assert s is not None
        assert s["settlement_id"] == "SET-001"

    def test_get_settlement_not_found(self, adapter: FinancialDataAdapter):
        assert adapter.get_settlement("SET-999") is None

    def test_get_refund(self, adapter: FinancialDataAdapter):
        r = adapter.get_refund("REF-001")
        assert r is not None
        assert r["refund_id"] == "REF-001"

    def test_get_refund_not_found(self, adapter: FinancialDataAdapter):
        assert adapter.get_refund("REF-999") is None

    def test_get_fee(self, adapter: FinancialDataAdapter):
        f = adapter.get_fee("FEE-001")
        assert f is not None
        assert f["fee_id"] == "FEE-001"

    def test_get_fee_not_found(self, adapter: FinancialDataAdapter):
        assert adapter.get_fee("FEE-999") is None

    def test_get_adjustment(self, adapter: FinancialDataAdapter):
        a = adapter.get_adjustment("ADJ-001")
        assert a is not None
        assert a["adjustment_id"] == "ADJ-001"

    def test_get_adjustment_not_found(self, adapter: FinancialDataAdapter):
        assert adapter.get_adjustment("ADJ-999") is None

    def test_get_case(self, adapter: FinancialDataAdapter):
        c = adapter.get_case("CASE-001")
        assert c is not None
        assert c["case_id"] == "CASE-001"

    def test_get_settlements_for_payment(self, adapter: FinancialDataAdapter):
        results = adapter.get_settlements_for_payment("PAY-001")
        assert len(results) == 1

    def test_get_refunds_for_payment(self, adapter: FinancialDataAdapter):
        results = adapter.get_refunds_for_payment("PAY-001")
        assert len(results) == 1

    def test_get_fees_for_payment(self, adapter: FinancialDataAdapter):
        results = adapter.get_fees_for_payment("PAY-001")
        assert len(results) == 1

    def test_get_adjustments_for_payment(self, adapter: FinancialDataAdapter):
        results = adapter.get_adjustments_for_payment("PAY-001")
        assert len(results) == 1

    def test_search_all(self, adapter: FinancialDataAdapter):
        results = adapter.search_records()
        assert len(results) > 0

    def test_search_by_merchant(self, adapter: FinancialDataAdapter):
        results = adapter.search_records(merchant_id="MER-001")
        assert len(results) >= 3  # payments + settlements + cases

    def test_search_by_payment(self, adapter: FinancialDataAdapter):
        results = adapter.search_records(payment_id="PAY-001")
        assert len(results) >= 3  # payment + settlement + refund + fee + adjustment

    def test_search_by_record_type(self, adapter: FinancialDataAdapter):
        results = adapter.search_records(record_type="payment")
        assert len(results) == 3
        assert all(r["type"] == "payment" for r in results)

    def test_search_limit(self, adapter: FinancialDataAdapter):
        results = adapter.search_records(limit=2)
        assert len(results) == 2

    def test_search_not_found(self, adapter: FinancialDataAdapter):
        results = adapter.search_records(merchant_id="MER-999")
        assert len(results) == 0

    def test_summary(self, adapter: FinancialDataAdapter):
        s = adapter.get_summary()
        assert s["payments"] == 3
        assert s["settlements"] == 2
        assert s["refunds"] == 1
        assert s["fees"] == 2
        assert s["adjustments"] == 1
        assert s["cases"] == 3

    def test_returns_copy_not_reference(self, adapter: FinancialDataAdapter):
        p1 = adapter.get_payment("PAY-001")
        p2 = adapter.get_payment("PAY-001")
        assert p1 == p2
        p1["amount"] = 999999
        p2_orig = adapter.get_payment("PAY-001")
        assert p2_orig["amount"] == 10000  # Not mutated


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────


class TestToolDefinitions:
    def test_all_7_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 7

    def test_tool_names(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        expected = {
            "search_financial_records",
            "get_payment",
            "get_settlement",
            "get_refund",
            "get_fee",
            "get_adjustment",
            "get_similar_exception",
        }
        assert names == expected

    def test_all_readonly(self):
        for tool in TOOL_DEFINITIONS:
            assert tool.is_financial is False, f"{tool.name} should not be financial"
            assert tool.requires_guardrail is False, f"{tool.name} should not require guardrail"

    def test_all_idempotent(self):
        for tool in TOOL_DEFINITIONS:
            assert tool.idempotent is True, f"{tool.name} should be idempotent"

    def test_get_payment_requires_id(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "get_payment")
        required = [p.name for p in tool.parameters if p.required]
        assert "payment_id" in required

    def test_get_settlement_requires_id(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "get_settlement")
        required = [p.name for p in tool.parameters if p.required]
        assert "settlement_id" in required

    def test_get_refund_requires_id(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "get_refund")
        required = [p.name for p in tool.parameters if p.required]
        assert "refund_id" in required

    def test_get_fee_requires_id(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "get_fee")
        required = [p.name for p in tool.parameters if p.required]
        assert "fee_id" in required

    def test_get_adjustment_requires_id(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "get_adjustment")
        required = [p.name for p in tool.parameters if p.required]
        assert "adjustment_id" in required

    def test_search_optional_filters(self):
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "search_financial_records")
        required = [p.name for p in tool.parameters if p.required]
        assert len(required) == 0  # All filters optional


# ─────────────────────────────────────────────────────────────────────────────
# Tool Handlers
# ─────────────────────────────────────────────────────────────────────────────


class TestToolHandlers:
    def test_search_records(self, handlers):
        result = handlers["search_financial_records"]({"merchant_id": "MER-001"})
        assert result["count"] >= 3

    def test_get_payment_found(self, handlers):
        result = handlers["get_payment"]({"payment_id": "PAY-001"})
        assert result["found"] is True
        assert result["payment"]["amount"] == 10000

    def test_get_payment_not_found(self, handlers):
        result = handlers["get_payment"]({"payment_id": "PAY-999"})
        assert result["found"] is False
        assert "not found" in result["error"]

    def test_get_settlement_found(self, handlers):
        result = handlers["get_settlement"]({"settlement_id": "SET-001"})
        assert result["found"] is True

    def test_get_settlement_not_found(self, handlers):
        result = handlers["get_settlement"]({"settlement_id": "SET-999"})
        assert result["found"] is False

    def test_get_refund_found(self, handlers):
        result = handlers["get_refund"]({"refund_id": "REF-001"})
        assert result["found"] is True

    def test_get_refund_not_found(self, handlers):
        result = handlers["get_refund"]({"refund_id": "REF-999"})
        assert result["found"] is False

    def test_get_fee_found(self, handlers):
        result = handlers["get_fee"]({"fee_id": "FEE-001"})
        assert result["found"] is True

    def test_get_fee_not_found(self, handlers):
        result = handlers["get_fee"]({"fee_id": "FEE-999"})
        assert result["found"] is False

    def test_get_adjustment_found(self, handlers):
        result = handlers["get_adjustment"]({"adjustment_id": "ADJ-001"})
        assert result["found"] is True

    def test_get_adjustment_not_found(self, handlers):
        result = handlers["get_adjustment"]({"adjustment_id": "ADJ-999"})
        assert result["found"] is False

    def test_get_similar_exception_found(self, handlers):
        result = handlers["get_similar_exception"]({"exception_id": "CASE-001"})
        assert result["found"] is True
        assert result["query_scenario"] == "FEE_DIFFERENCE"
        assert result["count"] >= 1  # CASE-003 has same scenario

    def test_get_similar_exception_not_found(self, handlers):
        result = handlers["get_similar_exception"]({"exception_id": "CASE-999"})
        assert result["found"] is False

    def test_get_similar_exception_top_k(self, handlers):
        result = handlers["get_similar_exception"]({"exception_id": "CASE-001", "top_k": "1"})
        assert result["count"] <= 1

    def test_search_record_type_filter(self, handlers):
        result = handlers["search_financial_records"]({"record_type": "refund"})
        assert all(r["type"] == "refund" for r in result["records"])

    def test_search_limit(self, handlers):
        result = handlers["search_financial_records"]({"limit": "1"})
        assert result["count"] <= 1


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestServerIntegration:
    def test_tools_registered(self, server_with_tools: MCPServer):
        assert server_with_tools.registry.tool_count == 7

    def test_invoke_get_payment(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True

    def test_invoke_get_settlement(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_settlement",
            parameters={"settlement_id": "SET-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True

    def test_invoke_get_refund(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_refund",
            parameters={"refund_id": "REF-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS

    def test_invoke_get_fee(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_fee",
            parameters={"fee_id": "FEE-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS

    def test_invoke_get_adjustment(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_adjustment",
            parameters={"adjustment_id": "ADJ-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS

    def test_invoke_search(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="search_financial_records",
            parameters={"merchant_id": "MER-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["count"] >= 1

    def test_invoke_similar_exception(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_similar_exception",
            parameters={"exception_id": "CASE-001"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is True

    def test_invoke_missing_required_param(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={},
        ))
        assert resp.status == MCPToolStatus.VALIDATION_FAILED

    def test_invoke_not_found(self, server_with_tools: MCPServer):
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-999"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result["found"] is False

    def test_audit_trail(self, server_with_tools: MCPServer):
        server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        audit = server_with_tools.get_audit_log()
        assert len(audit) == 1
        assert audit[0].tool_name == "get_payment"
        assert audit[0].is_financial is False

    def test_all_tools_audited(self, server_with_tools: MCPServer):
        tools = [
            ("get_payment", {"payment_id": "PAY-001"}),
            ("get_settlement", {"settlement_id": "SET-001"}),
            ("get_refund", {"refund_id": "REF-001"}),
            ("get_fee", {"fee_id": "FEE-001"}),
            ("get_adjustment", {"adjustment_id": "ADJ-001"}),
            ("search_financial_records", {"merchant_id": "MER-001"}),
            ("get_similar_exception", {"exception_id": "CASE-001"}),
        ]
        for tool_name, params in tools:
            server_with_tools.invoke(MCPToolRequest(tool_name=tool_name, parameters=params))
        audit = server_with_tools.get_audit_log()
        assert len(audit) == 7


# ─────────────────────────────────────────────────────────────────────────────
# Security Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSecurity:
    def test_no_sql_execution(self, adapter: FinancialDataAdapter):
        """Verify adapter does NOT execute SQL."""
        assert not hasattr(adapter, 'execute')
        assert not hasattr(adapter, 'query')
        assert not hasattr(adapter, 'raw_sql')

    def test_no_data_modification(self, adapter: FinancialDataAdapter):
        """Verify adapter does NOT modify data."""
        assert not hasattr(adapter, 'insert')
        assert not hasattr(adapter, 'update')
        assert not hasattr(adapter, 'delete')
        assert not hasattr(adapter, 'create')
        assert not hasattr(adapter, 'modify')

    def test_readonly_returns_copies(self, adapter: FinancialDataAdapter):
        """Verify data returned is a copy, not a reference to internal state."""
        p = adapter.get_payment("PAY-001")
        p["amount"] = 999999999
        p2 = adapter.get_payment("PAY-001")
        assert p2["amount"] == 10000  # Original unchanged

    def test_search_has_limit(self, adapter: FinancialDataAdapter):
        """Verify search respects the limit parameter."""
        results = adapter.search_records(limit=1)
        assert len(results) <= 1

    def test_no_arbitrary_query(self, server_with_tools: MCPServer):
        """Verify arbitrary tool names are rejected."""
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="DROP TABLE payments",
            parameters={},
        ))
        assert resp.status == MCPToolStatus.ERROR

    def test_no_arbitrary_parameters_bypass(self, server_with_tools: MCPServer):
        """Verify blocked parameters are rejected by validation."""
        resp = server_with_tools.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001", "sql": "SELECT * FROM payments"},
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        # SQL parameter should be rejected by validation
        assert "error" in resp.result
        assert "not allowed" in resp.result["error"]

    def test_all_readonly_tools_safe(self):
        """Verify all tool definitions are marked as non-financial and no guardrail."""
        for tool in TOOL_DEFINITIONS:
            assert tool.is_financial is False
            assert tool.requires_guardrail is False
            assert tool.requires_verification is False


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_dataset(self, tmp_path):
        batch_dir = tmp_path / "batch_001" / "generated"
        batch_dir.mkdir(parents=True)
        # Create empty JSON files
        for name in ["payments.json", "settlements.json", "refunds.json",
                      "fees.json", "adjustments.json", "cases.json", "merchants.json"]:
            (batch_dir / name).write_text("[]")

        adapter = FinancialDataAdapter(data_dir=str(tmp_path))
        assert adapter.load_batch("batch_001") is True
        assert adapter.get_summary()["payments"] == 0
        results = adapter.search_records()
        assert len(results) == 0

    def test_no_batch_directory(self, tmp_path):
        adapter = FinancialDataAdapter(data_dir=str(tmp_path))
        assert adapter.load_batch("nonexistent") is False
        assert adapter.is_loaded is False

    def test_missing_json_files(self, tmp_path):
        batch_dir = tmp_path / "batch_001" / "generated"
        batch_dir.mkdir(parents=True)
        # Only payments.json exists
        (batch_dir / "payments.json").write_text('[{"payment_id": "PAY-001", "amount": 100}]')

        adapter = FinancialDataAdapter(data_dir=str(tmp_path))
        # Should fail or partially load
        result = adapter.load_batch("batch_001")
        # Depends on implementation - may succeed with partial data or fail

    def test_malformed_json(self, tmp_path):
        batch_dir = tmp_path / "batch_001" / "generated"
        batch_dir.mkdir(parents=True)
        for name in ["payments.json", "settlements.json", "refunds.json",
                      "fees.json", "adjustments.json", "cases.json", "merchants.json"]:
            (batch_dir / name).write_text("not valid json {{{")

        adapter = FinancialDataAdapter(data_dir=str(tmp_path))
        result = adapter.load_batch("batch_001")
        # Should handle gracefully (not crash)
        assert isinstance(result, bool)
