"""
Tests for Razorpay CloseLoop Phase 11H — MCP Fallback.

Verifies:
- MCP available → uses MCP path
- MCP unavailable → falls back to internal adapter for reads
- MCP unavailable → escalates for writes (never direct-write)
- Both paths use the same FinancialDataAdapter
- Execution path tracked in audit
- Fallback events recorded
- MCP failure → fallback succeeds
- MCP failure + fallback failure → escalation
- Write operations never fall back to database
"""

import pytest
from unittest.mock import MagicMock, patch

from mcp.fallback import (
    ExecutionPath,
    FallbackResult,
    InternalServiceAdapter,
    MCPFallbackRouter,
)
from mcp.client import MCPClient
from mcp.server import MCPServer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp_server() -> MCPServer:
    """Create a test MCPServer with read-only tools."""
    from mcp.tools.readonly import TOOL_DEFINITIONS, create_handlers
    from mcp.adapters.financial_data import FinancialDataAdapter

    server = MCPServer()
    adapter = FinancialDataAdapter()
    handlers = create_handlers(adapter)
    for defn in TOOL_DEFINITIONS:
        if defn.name in handlers:
            server.register_tool(defn, handlers[defn.name])
    return server


@pytest.fixture
def mcp_client(mcp_server: MCPServer) -> MCPClient:
    return MCPClient(server=mcp_server)


@pytest.fixture
def internal_adapter() -> InternalServiceAdapter:
    return InternalServiceAdapter()


@pytest.fixture
def router_mcp_available(mcp_client: MCPClient, internal_adapter: InternalServiceAdapter) -> MCPFallbackRouter:
    return MCPFallbackRouter(
        mcp_client=mcp_client,
        internal_adapter=internal_adapter,
        mcp_available=True,
    )


@pytest.fixture
def router_mcp_unavailable(internal_adapter: InternalServiceAdapter) -> MCPFallbackRouter:
    # Create a client that will fail
    bad_server = MCPServer()
    bad_client = MCPClient(server=bad_server)
    return MCPFallbackRouter(
        mcp_client=bad_client,
        internal_adapter=internal_adapter,
        mcp_available=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execution Path
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionPath:
    def test_mcp_path_value(self):
        assert ExecutionPath.MCP.value == "MCP"

    def test_internal_path_value(self):
        assert ExecutionPath.INTERNAL.value == "INTERNAL"

    def test_escalated_path_value(self):
        assert ExecutionPath.ESCALATED.value == "ESCALATED"


# ─────────────────────────────────────────────────────────────────────────────
# FallbackResult
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackResult:
    def test_success_result(self):
        r = FallbackResult(
            success=True, data={"key": "val"},
            execution_path=ExecutionPath.MCP,
            tool_name="test", duration_ms=10.0,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["execution_path"] == "MCP"
        assert d["fallback_used"] is False
        assert d["data"]["key"] == "val"

    def test_error_result(self):
        r = FallbackResult(
            success=False, error="failed",
            execution_path=ExecutionPath.ESCALATED,
            tool_name="test", fallback_used=True,
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "failed"
        assert d["fallback_used"] is True

    def test_no_data_no_error(self):
        r = FallbackResult(success=True, execution_path=ExecutionPath.INTERNAL)
        d = r.to_dict()
        assert "data" not in d
        assert "error" not in d


# ─────────────────────────────────────────────────────────────────────────────
# Internal Service Adapter
# ─────────────────────────────────────────────────────────────────────────────


class TestInternalServiceAdapter:
    def test_adapter_creation(self, internal_adapter):
        assert internal_adapter.adapter is not None

    def test_get_payment(self, internal_adapter):
        result = internal_adapter.get_payment("PAY-001")
        # May or may not exist depending on data
        assert result is None or isinstance(result, dict)

    def test_search_records(self, internal_adapter):
        results = internal_adapter.search_records(limit=10)
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_uses_same_adapter_as_mcp(self, internal_adapter, mcp_server):
        """Both paths must use the same FinancialDataAdapter type."""
        from mcp.adapters.financial_data import FinancialDataAdapter
        assert isinstance(internal_adapter.adapter, FinancialDataAdapter)


# ─────────────────────────────────────────────────────────────────────────────
# Read Operations — MCP Available
# ─────────────────────────────────────────────────────────────────────────────


class TestReadWithMCP:
    def test_search_records_via_mcp(self, router_mcp_available):
        result = router_mcp_available.search_financial_records(
            workflow_id="WF-001", limit=10,
        )
        assert isinstance(result, FallbackResult)
        assert result.execution_path == ExecutionPath.MCP
        assert result.fallback_used is False

    def test_get_payment_via_mcp(self, router_mcp_available):
        result = router_mcp_available.get_payment("PAY-001", workflow_id="WF-001")
        assert isinstance(result, FallbackResult)
        assert result.execution_path == ExecutionPath.MCP

    def test_get_settlement_via_mcp(self, router_mcp_available):
        result = router_mcp_available.get_settlement("SET-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.MCP

    def test_get_refund_via_mcp(self, router_mcp_available):
        result = router_mcp_available.get_refund("REF-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.MCP

    def test_get_fee_via_mcp(self, router_mcp_available):
        result = router_mcp_available.get_fee("FEE-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.MCP

    def test_get_adjustment_via_mcp(self, router_mcp_available):
        result = router_mcp_available.get_adjustment("ADJ-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.MCP

    def test_no_fallback_when_mcp_works(self, router_mcp_available):
        result = router_mcp_available.search_financial_records(limit=5)
        assert result.fallback_used is False
        assert len(router_mcp_available.fallback_log) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Read Operations — MCP Unavailable (Fallback)
# ─────────────────────────────────────────────────────────────────────────────


class TestReadWithFallback:
    def test_search_records_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.search_financial_records(
            workflow_id="WF-001", limit=10,
        )
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True
        assert result.success is True
        assert result.data is not None

    def test_get_payment_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_payment("PAY-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_get_settlement_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_settlement("SET-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_get_refund_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_refund("REF-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_get_fee_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_fee("FEE-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_get_adjustment_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_adjustment("ADJ-001", workflow_id="WF-001")
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_fallback_recorded(self, router_mcp_unavailable):
        router_mcp_unavailable.search_financial_records(limit=5)
        assert len(router_mcp_unavailable.fallback_log) == 1
        log = router_mcp_unavailable.fallback_log[0]
        assert log["tool_name"] == "search_financial_records"
        assert log["escalated"] is False

    def test_fallback_uses_same_adapter(self, router_mcp_unavailable):
        """Fallback path returns data from FinancialDataAdapter, same as MCP."""
        result = router_mcp_unavailable.search_financial_records(limit=10)
        # Internal adapter returns records with 'type' and 'data' keys
        assert "records" in result.data
        assert isinstance(result.data["records"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Write Operations — MCP Unavailable (Escalation)
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteEscalation:
    def test_create_resolution_escalates(self, router_mcp_unavailable):
        result = router_mcp_unavailable.create_resolution(
            exception_id="CASE-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.success is False
        assert "MCP unavailable" in result.error

    def test_verify_resolution_escalates(self, router_mcp_unavailable):
        result = router_mcp_unavailable.verify_resolution(
            execution_id="EXE-001",
            workflow_id="WF-001",
        )
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.success is False

    def test_record_feedback_escalates(self, router_mcp_unavailable):
        result = router_mcp_unavailable.record_feedback(
            workflow_id="WF-001",
            exception_id="CASE-001",
            feedback_type="APPROVE",
            reviewer="human-01",
            system_prediction="FEE_DIFFERENCE",
        )
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.success is False

    def test_write_never_direct_write(self, router_mcp_unavailable):
        """Write operations must NEVER fall back to direct database writes."""
        result = router_mcp_unavailable.create_resolution(
            exception_id="CASE-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        # Must be ESCALATED, never INTERNAL for writes
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_write_escalation_recorded(self, router_mcp_unavailable):
        router_mcp_unavailable.create_resolution(
            exception_id="CASE-001",
            resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        assert len(router_mcp_unavailable.fallback_log) == 1
        log = router_mcp_unavailable.fallback_log[0]
        assert log["escalated"] is True

    def test_similar_exception_escalates(self, router_mcp_unavailable):
        """Similar exception retrieval escalates (no internal ML fallback)."""
        result = router_mcp_unavailable.get_similar_exception(
            exception_id="CASE-001",
            workflow_id="WF-001",
        )
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.success is False


# ─────────────────────────────────────────────────────────────────────────────
# MCP Failure with Fallback Recovery
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFailureFallbackRecovery:
    def test_mcp_error_triggers_fallback(self):
        """MCP server returns error → fallback to internal adapter."""
        # Create server with no tools registered (will return "not registered" error)
        empty_server = MCPServer()
        client = MCPClient(server=empty_server)
        internal = InternalServiceAdapter()

        router = MCPFallbackRouter(
            mcp_client=client,
            internal_adapter=internal,
            mcp_available=True,  # MCP is "available" but tool not registered
        )

        result = router.search_financial_records(limit=5)
        assert result.fallback_used is True
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.success is True

    def test_mcp_exception_triggers_fallback(self):
        """MCP server throws exception → fallback to internal adapter."""
        mock_client = MagicMock(spec=MCPClient)
        mock_client.call_tool.side_effect = RuntimeError("MCP connection lost")

        internal = InternalServiceAdapter()
        router = MCPFallbackRouter(
            mcp_client=mock_client,
            internal_adapter=internal,
            mcp_available=True,
        )

        result = router.get_payment("PAY-001")
        assert result.fallback_used is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_both_fail_escalates(self):
        """MCP fails AND internal fallback fails → escalation."""
        mock_client = MagicMock(spec=MCPClient)
        mock_client.call_tool.side_effect = RuntimeError("MCP down")

        mock_internal = MagicMock(spec=InternalServiceAdapter)
        mock_internal.get_payment.side_effect = RuntimeError("DB down")

        router = MCPFallbackRouter(
            mcp_client=mock_client,
            internal_adapter=mock_internal,
            mcp_available=True,
        )

        result = router.get_payment("PAY-001")
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.fallback_used is True
        assert "MCP failed" in result.error
        assert "internal fallback failed" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# MCP Toggle (Dynamic Availability)
# ─────────────────────────────────────────────────────────────────────────────


class TestDynamicAvailability:
    def test_toggle_mcp_off(self, router_mcp_available):
        """Switch MCP from available to unavailable mid-session."""
        # MCP available → MCP path
        result1 = router_mcp_available.search_financial_records(limit=5)
        assert result1.execution_path == ExecutionPath.MCP

        # Toggle MCP off
        router_mcp_available.mcp_available = False

        # Now fallback
        result2 = router_mcp_available.search_financial_records(limit=5)
        assert result2.execution_path == ExecutionPath.INTERNAL
        assert result2.fallback_used is True

    def test_toggle_mcp_on(self, mcp_server, internal_adapter):
        """Switch MCP from unavailable to available mid-session."""
        # Start with bad client
        bad_server = MCPServer()
        bad_client = MCPClient(server=bad_server)
        router = MCPFallbackRouter(
            mcp_client=bad_client,
            internal_adapter=internal_adapter,
            mcp_available=False,
        )

        # MCP unavailable → fallback
        result1 = router.search_financial_records(limit=5)
        assert result1.execution_path == ExecutionPath.INTERNAL

        # Toggle MCP on with proper client
        router._mcp_client = MCPClient(server=mcp_server)
        router.mcp_available = True

        # Now MCP path
        result2 = router.search_financial_records(limit=5)
        assert result2.execution_path == ExecutionPath.MCP
        assert result2.fallback_used is False


# ─────────────────────────────────────────────────────────────────────────────
# Audit Summary
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditSummary:
    def test_fallback_summary(self, router_mcp_unavailable):
        router_mcp_unavailable.search_financial_records(limit=5)
        router_mcp_unavailable.get_payment("PAY-001")
        router_mcp_unavailable.create_resolution(
            exception_id="CASE-001", resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500, workflow_id="WF-001",
            guardrail_decision="AUTO", authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )

        summary = router_mcp_unavailable.get_fallback_summary()
        assert summary["total_fallbacks"] == 3
        assert summary["escalations"] == 1
        assert summary["mcp_available"] is False

    def test_no_fallback_when_mcp_works(self, router_mcp_available):
        router_mcp_available.search_financial_records(limit=5)
        summary = router_mcp_available.get_fallback_summary()
        assert summary["total_fallbacks"] == 0

    def test_fallback_log_immutability(self, router_mcp_unavailable):
        router_mcp_unavailable.search_financial_records(limit=5)
        log1 = router_mcp_unavailable.fallback_log
        log2 = router_mcp_unavailable.fallback_log
        assert log1 == log2
        # Mutating one should not affect the other
        log1.append({"fake": True})
        assert len(router_mcp_unavailable.fallback_log) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_no_direct_db_write(self):
        """FallbackRouter must not have database write methods."""
        import inspect
        source = inspect.getsource(MCPFallbackRouter)
        assert "INSERT" not in source
        assert "UPDATE" not in source
        assert "DELETE" not in source

    def test_no_financial_logic(self):
        """FallbackRouter must not contain financial business logic."""
        import inspect
        source = inspect.getsource(MCPFallbackRouter)
        assert "calculate_interest" not in source.lower()
        assert "compute_fee" not in source.lower()

    def test_write_never_uses_internal_for_execution(self, router_mcp_unavailable):
        """Internal fallback is READ-ONLY. Writes always escalate."""
        result = router_mcp_unavailable.create_resolution(
            exception_id="CASE-001", resolution_type="FEE_DIFFERENCE",
            financial_adjustment_paise=500, workflow_id="WF-001",
            guardrail_decision="AUTO", authorization_source="guardrail_AUTO",
            idempotency_key="IDEM-001",
        )
        assert result.execution_path != ExecutionPath.INTERNAL

    def test_read_paths_use_same_adapter(self, mcp_server, internal_adapter):
        """Both MCP and internal fallback use FinancialDataAdapter."""
        from mcp.adapters.financial_data import FinancialDataAdapter

        client = MCPClient(server=mcp_server)
        router = MCPFallbackRouter(
            mcp_client=client,
            internal_adapter=internal_adapter,
            mcp_available=True,
        )

        # MCP path
        r1 = router.search_financial_records(limit=5)
        assert r1.execution_path == ExecutionPath.MCP

        # Fallback path
        router.mcp_available = False
        r2 = router.search_financial_records(limit=5)
        assert r2.execution_path == ExecutionPath.INTERNAL

        # Both return data (even if different structures, same adapter backs them)
        assert r1.success is True
        assert r2.success is True


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_search_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.search_financial_records(limit=0)
        assert result.success is True
        assert result.data["count"] == 0

    def test_nonexistent_payment_fallback(self, router_mcp_unavailable):
        result = router_mcp_unavailable.get_payment("NONEXISTENT")
        assert result.success is True
        assert result.data["found"] is False

    def test_concurrent_routers(self, mcp_server, internal_adapter):
        """Multiple routers sharing a client should not corrupt state."""
        client = MCPClient(server=mcp_server)
        r1 = MCPFallbackRouter(mcp_client=client, internal_adapter=internal_adapter)
        r2 = MCPFallbackRouter(mcp_client=client, internal_adapter=internal_adapter, mcp_available=False)

        res1 = r1.search_financial_records(limit=5)
        res2 = r2.search_financial_records(limit=5)

        assert res1.execution_path == ExecutionPath.MCP
        assert res2.execution_path == ExecutionPath.INTERNAL

    def test_fallback_after_mcp_recovery(self, mcp_server, internal_adapter):
        """MCP fails → fallback → MCP recovers → MCP path again."""
        client = MCPClient(server=mcp_server)
        router = MCPFallbackRouter(
            mcp_client=client,
            internal_adapter=internal_adapter,
            mcp_available=False,
        )

        # Fallback
        r1 = router.search_financial_records(limit=5)
        assert r1.execution_path == ExecutionPath.INTERNAL

        # Recover
        router.mcp_available = True
        r2 = router.search_financial_records(limit=5)
        assert r2.execution_path == ExecutionPath.MCP

    def test_all_read_tools_have_fallback(self, mcp_server, internal_adapter):
        """Every read operation should have a fallback path."""
        client = MCPClient(server=mcp_server)
        router = MCPFallbackRouter(
            mcp_client=client,
            internal_adapter=internal_adapter,
            mcp_available=False,
        )

        methods = [
            ("search_financial_records", {"limit": 5}),
            ("get_payment", {"payment_id": "PAY-001"}),
            ("get_settlement", {"settlement_id": "SET-001"}),
            ("get_refund", {"refund_id": "REF-001"}),
            ("get_fee", {"fee_id": "FEE-001"}),
            ("get_adjustment", {"adjustment_id": "ADJ-001"}),
        ]

        for method_name, kwargs in methods:
            method = getattr(router, method_name)
            result = method(**kwargs)
            assert result.execution_path == ExecutionPath.INTERNAL, \
                f"{method_name} did not fall back to INTERNAL"
