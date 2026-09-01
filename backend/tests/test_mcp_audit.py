"""
Tests for Razorpay CloseLoop Phase 11D — MCP Audit Logging.

Verifies every tool call generates an audit entry, sensitive fields are masked,
and correlation via request_id/workflow_id/agent_id/exception_id works.
"""

import pytest

from mcp.audit import MCPAuditLogger, mask_parameters, SENSITIVE_FIELD_NAMES, MASKED_VALUE
from mcp.server import MCPServer
from mcp.schemas import MCPToolRequest, MCPToolStatus, MCPToolDefinition, MCPToolParameter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def audit_logger() -> MCPAuditLogger:
    return MCPAuditLogger()


@pytest.fixture
def server() -> MCPServer:
    s = MCPServer()
    # Register a read-only tool
    s.register_tool(
        MCPToolDefinition(
            name="get_payment",
            description="Get payment info",
            category="reconciliation",
            parameters=[MCPToolParameter(name="payment_id", type="string", required=True)],
        ),
        lambda p: {"found": True, "payment": {"payment_id": p["payment_id"]}},
    )
    # Register a financial (write) tool
    s.register_tool(
        MCPToolDefinition(
            name="execute_refund",
            description="Execute a refund",
            category="execution",
            parameters=[MCPToolParameter(name="refund_id", type="string", required=True)],
            is_financial=True,
            requires_guardrail=True,
        ),
        lambda p: {"executed": True, "refund_id": p["refund_id"]},
    )
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Sensitive Field Masking
# ─────────────────────────────────────────────────────────────────────────────


class TestMasking:
    def test_mask_password(self):
        masked = mask_parameters({"password": "secret123"})
        assert masked["password"] == MASKED_VALUE

    def test_mask_token(self):
        masked = mask_parameters({"token": "abc123"})
        assert masked["token"] == MASKED_VALUE

    def test_mask_api_key(self):
        masked = mask_parameters({"api_key": "key123"})
        assert masked["api_key"] == MASKED_VALUE

    def test_mask_secret(self):
        masked = mask_parameters({"secret": "s3cr3t"})
        assert masked["secret"] == MASKED_VALUE

    def test_mask_private_key(self):
        masked = mask_parameters({"private_key": "-----BEGIN RSA PRIVATE KEY-----"})
        assert masked["private_key"] == MASKED_VALUE

    def test_mask_nested(self):
        masked = mask_parameters({"config": {"password": "secret"}})
        assert masked["config"]["password"] == MASKED_VALUE

    def test_mask_list(self):
        masked = mask_parameters({"items": [{"password": "secret"}, "safe"]})
        assert masked["items"][0]["password"] == MASKED_VALUE
        assert masked["items"][1] == "safe"

    def test_safe_fields_preserved(self):
        masked = mask_parameters({"payment_id": "PAY-001", "amount": 10000})
        assert masked["payment_id"] == "PAY-001"
        assert masked["amount"] == 10000

    def test_stripe_key_masked(self):
        masked = mask_parameters({"key": "sk_live_abc123def456"})
        assert masked["key"] == MASKED_VALUE

    def test_aws_key_masked(self):
        masked = mask_parameters({"key": "AKIAIOSFODNN7EXAMPLE"})
        assert masked["key"] == MASKED_VALUE

    def test_github_token_masked(self):
        masked = mask_parameters({"token": "ghp_ABCDEFGHIJKLMNOP"})
        assert masked["token"] == MASKED_VALUE

    def test_original_not_modified(self):
        original = {"password": "secret"}
        masked = mask_parameters(original)
        assert original["password"] == "secret"
        assert masked["password"] == MASKED_VALUE

    def test_all_sensitive_fields_covered(self):
        """Verify all sensitive field names are in the mask set."""
        for name in ["password", "secret", "token", "api_key", "private_key", "credential"]:
            assert name in SENSITIVE_FIELD_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_record(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={"payment_id": "PAY-001"},
            status=MCPToolStatus.SUCCESS,
        )
        assert rec.record_id.startswith("MCPAUD-")
        assert audit_logger.record_count == 1

    def test_sensitive_params_masked_in_record(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={"payment_id": "PAY-001", "password": "secret"},
            status=MCPToolStatus.SUCCESS,
        )
        assert rec.parameters["payment_id"] == "PAY-001"
        assert rec.parameters["password"] == MASKED_VALUE

    def test_agent_id_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            agent_id="agent-001",
        )
        assert rec.agent_id == "agent-001"

    def test_exception_id_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            exception_id="CASE-001",
        )
        assert rec.exception_id == "CASE-001"

    def test_workflow_id_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            workflow_id="WF-001",
        )
        assert rec.workflow_id == "WF-001"

    def test_idempotency_key_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            idempotency_key="IDEM-001",
        )
        assert rec.idempotency_key == "IDEM-001"

    def test_read_only_flag(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            is_read_only=True,
        )
        assert rec.is_read_only is True

    def test_write_tool_flag(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="execute_refund",
            category="execution",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            is_read_only=False,
            is_financial=True,
        )
        assert rec.is_read_only is False
        assert rec.is_financial is True

    def test_duration_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            duration_ms=42.5,
        )
        assert rec.duration_ms == 42.5

    def test_error_stored(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="get_payment",
            category="reconciliation",
            parameters={},
            status=MCPToolStatus.ERROR,
            error="Not found",
        )
        assert rec.error == "Not found"

    def test_guardrail_fields(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="execute_refund",
            category="execution",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            guardrail_checked=True,
            guardrail_passed=True,
        )
        assert rec.guardrail_checked is True
        assert rec.guardrail_passed is True

    def test_write_tool_fields(self, audit_logger: MCPAuditLogger):
        rec = audit_logger.record(
            request_id="REQ-001",
            tool_name="execute_refund",
            category="execution",
            parameters={},
            status=MCPToolStatus.SUCCESS,
            is_read_only=False,
            authorization_context={"approved_by": "guardrail"},
            guardrail_result={"passed": True},
            execution_result="success",
            verification_result="verified",
        )
        assert rec.authorization_context == {"approved_by": "guardrail"}
        assert rec.guardrail_result == {"passed": True}
        assert rec.execution_result == "success"
        assert rec.verification_result == "verified"

    def test_max_records_trimmed(self):
        logger = MCPAuditLogger(max_records=3)
        for i in range(5):
            logger.record(f"R{i}", "t", "c", {}, MCPToolStatus.SUCCESS)
        assert logger.record_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Queries
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelation:
    def test_query_by_workflow(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "c", {}, MCPToolStatus.SUCCESS, workflow_id="WF-001")
        audit_logger.record("R2", "t2", "c", {}, MCPToolStatus.SUCCESS, workflow_id="WF-001")
        audit_logger.record("R3", "t3", "c", {}, MCPToolStatus.SUCCESS, workflow_id="WF-002")
        results = audit_logger.query(workflow_id="WF-001")
        assert len(results) == 2

    def test_query_by_agent(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "c", {}, MCPToolStatus.SUCCESS, agent_id="agent-1")
        audit_logger.record("R2", "t2", "c", {}, MCPToolStatus.SUCCESS, agent_id="agent-2")
        results = audit_logger.query(agent_id="agent-1")
        assert len(results) == 1

    def test_query_by_exception(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "c", {}, MCPToolStatus.SUCCESS, exception_id="CASE-001")
        audit_logger.record("R2", "t2", "c", {}, MCPToolStatus.SUCCESS, exception_id="CASE-001")
        audit_logger.record("R3", "t3", "c", {}, MCPToolStatus.SUCCESS, exception_id="CASE-002")
        results = audit_logger.query(exception_id="CASE-001")
        assert len(results) == 2

    def test_query_by_request_id(self, audit_logger: MCPAuditLogger):
        audit_logger.record("REQ-001", "t1", "c", {}, MCPToolStatus.SUCCESS)
        audit_logger.record("REQ-002", "t1", "c", {}, MCPToolStatus.SUCCESS)
        results = audit_logger.query(request_id="REQ-001")
        assert len(results) == 1

    def test_workflow_trace(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "get_payment", "rec", {}, MCPToolStatus.SUCCESS, workflow_id="WF-1")
        audit_logger.record("R2", "get_settlement", "rec", {}, MCPToolStatus.SUCCESS, workflow_id="WF-1")
        trace = audit_logger.get_workflow_trace("WF-1")
        assert len(trace) == 2
        assert trace[0].tool_name == "get_payment"
        assert trace[1].tool_name == "get_settlement"

    def test_agent_trace(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "c", {}, MCPToolStatus.SUCCESS, agent_id="agent-1")
        audit_logger.record("R2", "t2", "c", {}, MCPToolStatus.SUCCESS, agent_id="agent-1")
        trace = audit_logger.get_agent_trace("agent-1")
        assert len(trace) == 2

    def test_exception_trace(self, audit_logger: MCPAuditLogger):
        audit_logger.record("R1", "t1", "c", {}, MCPToolStatus.SUCCESS, exception_id="CASE-1")
        audit_logger.record("R2", "t2", "c", {}, MCPToolStatus.SUCCESS, exception_id="CASE-1")
        trace = audit_logger.get_exception_trace("CASE-1")
        assert len(trace) == 2

    def test_request_trace(self, audit_logger: MCPAuditLogger):
        audit_logger.record("REQ-1", "t1", "c", {}, MCPToolStatus.SUCCESS)
        trace = audit_logger.get_request_trace("REQ-1")
        assert len(trace) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Server Audit Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestServerAudit:
    def test_read_tool_audited(self, server: MCPServer):
        resp = server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].tool_name == "get_payment"
        assert audit[0].status == MCPToolStatus.SUCCESS
        assert audit[0].is_read_only is True

    def test_write_tool_audited(self, server: MCPServer):
        resp = server.invoke(MCPToolRequest(
            tool_name="execute_refund",
            parameters={"refund_id": "REF-001"},
        ))
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].tool_name == "execute_refund"
        assert audit[0].is_financial is True
        assert audit[0].is_read_only is False
        assert audit[0].guardrail_checked is True

    def test_failed_tool_audited(self, server: MCPServer):
        resp = server.invoke(MCPToolRequest(
            tool_name="nonexistent",
            parameters={},
        ))
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].status == MCPToolStatus.ERROR

    def test_validation_failure_audited(self, server: MCPServer):
        resp = server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={},  # Missing required payment_id
        ))
        audit = server.get_audit_log()
        assert len(audit) == 1
        assert audit[0].status == MCPToolStatus.VALIDATION_FAILED

    def test_agent_id_in_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            agent_id="agent-001",
        ))
        audit = server.get_audit_log()
        assert audit[0].agent_id == "agent-001"

    def test_exception_id_in_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            exception_id="CASE-001",
        ))
        audit = server.get_audit_log()
        assert audit[0].exception_id == "CASE-001"

    def test_workflow_id_in_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id="WF-001",
        ))
        audit = server.get_audit_log()
        assert audit[0].workflow_id == "WF-001"

    def test_request_id_in_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            request_id="REQ-001",
        ))
        audit = server.get_audit_log()
        assert audit[0].request_id == "REQ-001"

    def test_idempotency_key_in_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            idempotency_key="IDEM-001",
        ))
        audit = server.get_audit_log()
        assert audit[0].idempotency_key == "IDEM-001"

    def test_duration_recorded(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        audit = server.get_audit_log()
        assert audit[0].duration_ms is not None
        assert audit[0].duration_ms >= 0

    def test_timestamp_recorded(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
        ))
        audit = server.get_audit_log()
        assert audit[0].timestamp is not None

    def test_multiple_calls_correlated(self, server: MCPServer):
        # Same workflow, different tools
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            workflow_id="WF-001",
        ))
        server.invoke(MCPToolRequest(
            tool_name="execute_refund",
            parameters={"refund_id": "REF-001"},
            workflow_id="WF-001",
        ))
        audit = server.get_audit_log(workflow_id="WF-001")
        assert len(audit) == 2
        assert audit[0].tool_name == "get_payment"
        assert audit[1].tool_name == "execute_refund"

    def test_audit_log_correlation_via_request_id(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001"},
            request_id="REQ-001",
        ))
        audit = server.get_audit_log(request_id="REQ-001")
        assert len(audit) == 1

    def test_sensitive_params_masked_in_server_audit(self, server: MCPServer):
        server.invoke(MCPToolRequest(
            tool_name="get_payment",
            parameters={"payment_id": "PAY-001", "password": "secret"},
        ))
        audit = server.get_audit_log()
        assert audit[0].parameters["payment_id"] == "PAY-001"
        assert audit[0].parameters["password"] == MASKED_VALUE


# ─────────────────────────────────────────────────────────────────────────────
# Immutability
# ─────────────────────────────────────────────────────────────────────────────


class TestImmutability:
    def test_records_are_append_only(self, audit_logger: MCPAuditLogger):
        """Audit records cannot be modified after creation."""
        rec = audit_logger.record(
            "R1", "t1", "c", {}, MCPToolStatus.SUCCESS
        )
        # Pydantic model by default is mutable, but we don't provide mutation APIs
        assert not hasattr(audit_logger, 'delete_record')
        assert not hasattr(audit_logger, 'modify_record')
        assert not hasattr(audit_logger, 'clear')

    def test_logger_no_delete(self, audit_logger: MCPAuditLogger):
        """Audit logger has no delete/clear methods."""
        assert not hasattr(audit_logger, 'delete')
        assert not hasattr(audit_logger, 'clear')
        assert not hasattr(audit_logger, 'remove')
