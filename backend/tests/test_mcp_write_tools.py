"""
Tests for Razorpay CloseLoop Phase 11E — Controlled MCP Write Tools.

Verifies create_resolution, verify_resolution, record_feedback
delegate to existing services and maintain safety.
"""

import pytest
from datetime import datetime

from mcp.tools.write import (
    ALLOWED_FEEDBACK_TYPES,
    ALLOWED_GUARDRAIL_DECISIONS,
    ALLOWED_RESOLUTION_TYPES,
    WRITE_TOOL_DEFINITIONS,
    create_write_handlers,
)
from mcp.tools.readonly import TOOL_DEFINITIONS as READONLY_DEFINITIONS
from mcp.server import MCPServer
from mcp.schemas import MCPToolRequest, MCPToolStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


class MockExecutionResult:
    def __init__(self, status="EXECUTED", error=None):
        self.execution_id = "EXE-001"
        self.status = type("Status", (), {"value": status})()
        self.actual_adjustment_paise = 500
        self.error = error


class MockVerificationResult:
    def __init__(self, status="PASSED"):
        self.verification_id = "VER-001"
        self.status = type("Status", (), {"value": status})()
        self.discrepancy_eliminated = True
        self.has_unintended_changes = False
        self.passed_checks = 5
        self.failed_checks = 0


class MockFeedbackRecord:
    def __init__(self):
        self.feedback_id = "FB-001"
        self.feedback_type = type("FBType", (), {"value": "approve"})()
        self.reviewer = "test_reviewer"


class MockExecutionService:
    def __init__(self):
        self._executions = {}

    def execute(self, action_request):
        result = MockExecutionResult()
        self._executions[action_request.get("idempotency_key", "")] = result
        return result

    def get_execution(self, execution_id):
        for k, v in self._executions.items():
            if v.execution_id == execution_id:
                return v
        return None


class MockVerificationEngine:
    def __init__(self):
        self.verify_count = 0

    def verify(self, execution_result):
        self.verify_count += 1
        return MockVerificationResult()


class MockFeedbackService:
    def __init__(self):
        self._feedback = []

    def record_feedback(self, **kwargs):
        record = MockFeedbackRecord()
        self._feedback.append((kwargs, record))
        return record


@pytest.fixture
def exec_service():
    return MockExecutionService()


@pytest.fixture
def verification_engine():
    return MockVerificationEngine()


@pytest.fixture
def feedback_service():
    return MockFeedbackService()


@pytest.fixture
def handlers(exec_service, verification_engine, feedback_service):
    return create_write_handlers(exec_service, verification_engine, feedback_service)


@pytest.fixture
def server_with_write_tools(exec_service, verification_engine, feedback_service):
    server = MCPServer()
    handlers = create_write_handlers(exec_service, verification_engine, feedback_service)
    for defn in WRITE_TOOL_DEFINITIONS:
        if defn.name in handlers:
            server.register_tool(defn, handlers[defn.name])
    return server


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteToolDefinitions:
    def test_all_3_tools_defined(self):
        assert len(WRITE_TOOL_DEFINITIONS) == 3

    def test_tool_names(self):
        names = {t.name for t in WRITE_TOOL_DEFINITIONS}
        assert names == {"create_resolution", "verify_resolution", "record_feedback"}

    def test_create_resolution_is_financial(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "create_resolution")
        assert tool.is_financial is True
        assert tool.requires_guardrail is True
        assert tool.requires_verification is True
        assert tool.idempotent is False

    def test_verify_resolution_not_financial(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "verify_resolution")
        assert tool.is_financial is False
        assert tool.idempotent is True

    def test_record_feedback_not_financial(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "record_feedback")
        assert tool.is_financial is False
        assert tool.idempotent is True

    def test_create_resolution_required_params(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "create_resolution")
        required = {p.name for p in tool.parameters if p.required}
        assert "exception_id" in required
        assert "resolution_type" in required
        assert "financial_adjustment_paise" in required
        assert "workflow_id" in required
        assert "authorization_source" in required
        assert "guardrail_decision" in required
        assert "idempotency_key" in required

    def test_write_tools_no_overlap_with_readonly(self):
        write_names = {t.name for t in WRITE_TOOL_DEFINITIONS}
        readonly_names = {t.name for t in READONLY_DEFINITIONS}
        assert write_names.isdisjoint(readonly_names)


# ─────────────────────────────────────────────────────────────────────────────
# Create Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateResolution:
    def _valid_params(self, **overrides):
        params = {
            "exception_id": "CASE-001",
            "resolution_type": "FEE_DIFFERENCE",
            "financial_adjustment_paise": 500,
            "workflow_id": "WF-001",
            "authorization_source": "guardrail_AUTO",
            "guardrail_decision": "AUTO",
            "idempotency_key": "IDEM-001",
        }
        params.update(overrides)
        return params

    def test_valid_create_resolution(self, handlers):
        result = handlers["create_resolution"](self._valid_params())
        assert result.get("executed") is True
        assert result.get("execution_id") is not None

    def test_invalid_exception_id(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(exception_id="CASE; DROP TABLE")
        )
        assert "error" in result

    def test_invalid_resolution_type(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(resolution_type="UNKNOWN_TYPE")
        )
        assert "error" in result
        assert "resolution type" in result["error"]

    def test_invalid_guardrail_decision(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(guardrail_decision="FORCE_AUTO")
        )
        assert "error" in result

    def test_missing_authorization(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(authorization_source="NONE")
        )
        assert "error" in result

    def test_missing_workflow_id(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(workflow_id="")
        )
        assert "error" in result

    def test_injection_in_exception_id(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(exception_id="CASE-001; DROP TABLE payments")
        )
        assert "error" in result

    def test_negative_adjustment_rejected(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(financial_adjustment_paise=-100)
        )
        assert "error" in result

    def test_zero_adjustment_allowed(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(financial_adjustment_paise=0)
        )
        assert result.get("executed") is True

    def test_high_value_resolution(self, handlers):
        result = handlers["create_resolution"](
            self._valid_params(financial_adjustment_paise=100000)
        )
        assert result.get("executed") is True

    def test_all_resolution_types_allowed(self, handlers):
        for rt in ALLOWED_RESOLUTION_TYPES:
            result = handlers["create_resolution"](
                self._valid_params(resolution_type=rt)
            )
            assert not result.get("error"), f"Resolution type {rt} should be allowed"

    def test_delegates_to_execution_service(self, handlers, exec_service):
        handlers["create_resolution"](self._valid_params())
        assert len(exec_service._executions) == 1

    def test_exception_in_service(self, handlers):
        class FailingService:
            def execute(self, req):
                raise RuntimeError("Service failure")

        fail_handlers = create_write_handlers(
            FailingService(), MockVerificationEngine(), MockFeedbackService()
        )
        result = fail_handlers["create_resolution"]({
            "exception_id": "CASE-001",
            "resolution_type": "FEE_DIFFERENCE",
            "financial_adjustment_paise": 500,
            "workflow_id": "WF-001",
            "authorization_source": "guardrail_AUTO",
            "guardrail_decision": "AUTO",
            "idempotency_key": "IDEM-001",
        })
        assert "error" in result
        assert "Service failure" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Verify Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifyResolution:
    def test_valid_verify(self, handlers, exec_service):
        # First create an execution
        exec_result = MockExecutionResult()
        exec_service._executions["EXE-001"] = exec_result

        result = handlers["verify_resolution"]({
            "execution_id": "EXE-001",
            "workflow_id": "WF-001",
        })
        assert result.get("verified") is True
        assert result.get("verification_id") is not None

    def test_execution_not_found(self, handlers):
        result = handlers["verify_resolution"]({
            "execution_id": "NONEXISTENT",
            "workflow_id": "WF-001",
        })
        assert result.get("verified") is False
        assert "error" in result

    def test_invalid_execution_id(self, handlers):
        result = handlers["verify_resolution"]({
            "execution_id": "EXE; DROP TABLE",
            "workflow_id": "WF-001",
        })
        assert "error" in result

    def test_injection_in_execution_id(self, handlers):
        result = handlers["verify_resolution"]({
            "execution_id": "EXE-001'; SELECT * FROM users; --",
            "workflow_id": "WF-001",
        })
        assert "error" in result

    def test_delegates_to_verification_engine(
        self, handlers, exec_service, verification_engine
    ):
        exec_result = MockExecutionResult()
        exec_service._executions["EXE-001"] = exec_result
        handlers["verify_resolution"]({
            "execution_id": "EXE-001",
            "workflow_id": "WF-001",
        })
        assert verification_engine.verify_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Record Feedback
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordFeedback:
    def _valid_params(self, **overrides):
        params = {
            "workflow_id": "WF-001",
            "exception_id": "CASE-001",
            "feedback_type": "APPROVE",
            "reviewer": "reviewer-001",
            "system_prediction": "FEE_DIFFERENCE",
        }
        params.update(overrides)
        return params

    def test_valid_approve(self, handlers):
        result = handlers["record_feedback"](self._valid_params())
        assert result.get("recorded") is True
        assert result.get("feedback_id") is not None

    def test_valid_reject(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(feedback_type="REJECT", reason="Incorrect resolution")
        )
        assert result.get("recorded") is True

    def test_valid_correct(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(feedback_type="CORRECT", reason="Wrong amount")
        )
        assert result.get("recorded") is True

    def test_valid_escalate(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(feedback_type="ESCALATE", reason="Complex case")
        )
        assert result.get("recorded") is True

    def test_invalid_feedback_type(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(feedback_type="INVALID")
        )
        assert "error" in result
        assert "feedback type" in result["error"]

    def test_invalid_workflow_id(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(workflow_id="WF; DROP")
        )
        assert "error" in result

    def test_injection_in_reviewer(self, handlers):
        result = handlers["record_feedback"](
            self._valid_params(reviewer="'; DROP TABLE users; --")
        )
        assert "error" in result

    def test_all_feedback_types_allowed(self, handlers):
        for ft in ALLOWED_FEEDBACK_TYPES:
            result = handlers["record_feedback"](
                self._valid_params(feedback_type=ft)
            )
            assert "error" not in result, f"Feedback type {ft} should be allowed"

    def test_delegates_to_feedback_service(self, handlers, feedback_service):
        handlers["record_feedback"](self._valid_params())
        assert len(feedback_service._feedback) == 1

    def test_exception_in_feedback_service(self, handlers):
        class FailingFeedback:
            def record_feedback(self, **kwargs):
                raise RuntimeError("Feedback failure")

        fail_handlers = create_write_handlers(
            MockExecutionService(), MockVerificationEngine(), FailingFeedback()
        )
        result = fail_handlers["record_feedback"]({
            "workflow_id": "WF-001",
            "exception_id": "CASE-001",
            "feedback_type": "APPROVE",
            "reviewer": "reviewer-001",
            "system_prediction": "FEE_DIFFERENCE",
        })
        assert "error" in result
        assert "Feedback failure" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Server Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestServerIntegration:
    def test_tools_registered(self, server_with_write_tools):
        # 3 write tools + possibly 0 read tools (no adapter registered)
        assert server_with_write_tools.registry.tool_count == 3

    def test_invoke_create_resolution(self, server_with_write_tools):
        resp = server_with_write_tools.invoke(MCPToolRequest(
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
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("executed") is True

    def test_invoke_record_feedback(self, server_with_write_tools):
        resp = server_with_write_tools.invoke(MCPToolRequest(
            tool_name="record_feedback",
            parameters={
                "workflow_id": "WF-001",
                "exception_id": "CASE-001",
                "feedback_type": "APPROVE",
                "reviewer": "reviewer-001",
                "system_prediction": "FEE_DIFFERENCE",
            },
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert resp.result.get("recorded") is True

    def test_write_tool_audited(self, server_with_write_tools):
        server_with_write_tools.invoke(MCPToolRequest(
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
        audit = server_with_write_tools.get_audit_log()
        assert len(audit) == 1
        assert audit[0].is_financial is True
        assert audit[0].is_read_only is False
        assert audit[0].guardrail_checked is True

    def test_feedback_tool_audited_as_readonly(self, server_with_write_tools):
        server_with_write_tools.invoke(MCPToolRequest(
            tool_name="record_feedback",
            parameters={
                "workflow_id": "WF-001",
                "exception_id": "CASE-001",
                "feedback_type": "APPROVE",
                "reviewer": "reviewer-001",
                "system_prediction": "FEE_DIFFERENCE",
            },
        ))
        audit = server_with_write_tools.get_audit_log()
        assert audit[0].is_read_only is True
        assert audit[0].is_financial is False

    def test_invalid_input_returns_error(self, server_with_write_tools):
        resp = server_with_write_tools.invoke(MCPToolRequest(
            tool_name="create_resolution",
            parameters={
                "exception_id": "CASE-001",
                "resolution_type": "INVALID",
                "financial_adjustment_paise": 500,
                "workflow_id": "WF-001",
                "authorization_source": "guardrail_AUTO",
                "guardrail_decision": "AUTO",
                "idempotency_key": "IDEM-001",
            },
        ))
        assert resp.status == MCPToolStatus.SUCCESS
        assert "error" in resp.result


# ─────────────────────────────────────────────────────────────────────────────
# Safety
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteSafety:
    def test_no_direct_database_access(self, handlers):
        """Write tools delegate to services, not to database."""
        assert not hasattr(handlers["create_resolution"], 'database')
        assert not hasattr(handlers["create_resolution"], 'db')

    def test_create_resolution_requires_guardrail(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "create_resolution")
        assert tool.requires_guardrail is True

    def test_create_resolution_requires_verification(self):
        tool = next(t for t in WRITE_TOOL_DEFINITIONS if t.name == "create_resolution")
        assert tool.requires_verification is True

    def test_write_tools_validate_inputs(self, handlers):
        """All write tools validate inputs before delegation."""
        result = handlers["create_resolution"]({})
        assert "error" in result

    def test_feedback_types_documented(self):
        """Feedback types match Phase 9 FeedbackType enum."""
        assert "APPROVE" in ALLOWED_FEEDBACK_TYPES
        assert "REJECT" in ALLOWED_FEEDBACK_TYPES
        assert "CORRECT" in ALLOWED_FEEDBACK_TYPES
        assert "ESCALATE" in ALLOWED_FEEDBACK_TYPES

    def test_resolution_types_documented(self):
        """Resolution types are documented."""
        assert "FEE_DIFFERENCE" in ALLOWED_RESOLUTION_TYPES
        assert "REFUND_ADJUSTMENT" in ALLOWED_RESOLUTION_TYPES
        assert "DUPLICATE" in ALLOWED_RESOLUTION_TYPES
