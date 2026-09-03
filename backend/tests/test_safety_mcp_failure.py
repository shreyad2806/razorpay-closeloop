"""
Adversarial safety tests for MCP failure scenarios.

Verifies that MCP failures are handled safely, with proper fallback
and escalation behavior.

MCP failure chain:
  1. MCP server unavailable / timeout / error
  2. MCPFallbackRouter handles fallback
  3. Read operations → InternalServiceAdapter (same FinancialDataAdapter)
  4. Write operations → ESCALATE (never direct-write)
  5. FallbackGuard marks mcp as CRITICAL (required)
  6. DecisionMatrix blocks AUTO when mcp is critical failure

Key design principles:
  - MCP is a required dependency for financial safety
  - Read fallback uses same underlying adapter
  - Write fallback escalates (never uncontrolled direct-write)
  - All fallback events are audited
  - No partial unsafe financial action occurs

No production logic is modified.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_mcp.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.confidence_gate import GateAction
from app.schemas.decision_matrix import AutomationDecision
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.failure_fallback import (
    ErrorCategory,
    FailureFallbackResult,
    FallbackAction,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.confidence_gate import ConfidenceGate
from app.services.exposure_guard import ExposureGuard
from app.services.fallback_guard import FallbackGuard, DEFAULT_POLICIES
from app.services.guardrail_engine import GuardrailEngine
from app.services.decision_matrix import AutomationDecisionMatrix
from mcp.fallback import (
    ExecutionPath,
    FallbackResult,
    InternalServiceAdapter,
    MCPFallbackRouter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engine(**kwargs):
    """Build a valid ResolutionEngineResult for testing."""
    from tests.test_safety_high_value import _make_candidate, _make_score
    defaults = dict(
        exception_id="EXC-MCP-001",
        case_id="CASE-MCP-001",
        payment_id="PAY-MCP-001",
        merchant_id="MER-MCP-01",
        expected_amount=100_000,
        actual_amount=90_000,
        difference=10_000,
        status=SelectionStatus.RECOMMENDED,
        confidence=0.85,
        risk_category="LOW",
        deterministic_exception_type="FEE_DIFFERENCE",
        evidence_coverage=0.90,
        evidence_consistency=0.85,
    )
    defaults.update(kwargs)
    if "selected_candidate" not in kwargs:
        defaults["selected_candidate"] = _make_candidate(defaults["difference"])
        defaults["selected_resolution"] = "FEE_ADJUSTMENT"
    if "selected_score" not in kwargs:
        defaults["selected_score"] = _make_score()
    if "ranked_candidates" not in kwargs:
        defaults["ranked_candidates"] = [defaults["selected_candidate"]]
    if "candidate_scores" not in kwargs:
        defaults["candidate_scores"] = [defaults["selected_score"]]
    return ResolutionEngineResult(**defaults)


def _make_evidence_result(passed=True, coverage=0.90, consistency=0.85):
    return EvidenceGuardResult(
        passed=passed,
        action=EvidenceAction.PASS if passed else EvidenceAction.BLOCK,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        has_conflict=False,
        is_novel=False,
        reason="test",
    )


def _make_exposure_result(passed=True, amount=10_000):
    return ExposureGuardResult(
        passed=passed,
        action=ExposureAction.PASS if passed else ExposureAction.BLOCK,
        adjustment_amount_paise=amount,
        max_auto_resolution_paise=50_000,
        reason="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Fallback Guard — MCP Dependency Policy
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackGuardMCPPolicy:
    """Test fallback guard dependency policy for MCP."""

    def test_mcp_is_required(self):
        """MCP dependency is marked as required."""
        policy = DEFAULT_POLICIES["mcp"]
        assert policy.is_required is True

    def test_mcp_fallback_action(self):
        """MCP failure → FAIL_CLOSED."""
        policy = DEFAULT_POLICIES["mcp"]
        assert policy.fallback_action == FallbackAction.FAIL_CLOSED

    def test_mcp_fallback_status(self):
        """MCP failure → HUMAN_REVIEW status."""
        policy = DEFAULT_POLICIES["mcp"]
        assert policy.fallback_status == "HUMAN_REVIEW"

    def test_mcp_error_category(self):
        """MCP failure category is MCP_UNAVAILABLE."""
        policy = DEFAULT_POLICIES["mcp"]
        assert policy.error_category == ErrorCategory.MCP_UNAVAILABLE

    def test_mcp_unavailable_fails_closed(self):
        """MCP unavailable → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is False
        assert result.has_critical_failure is True

    def test_mcp_unavailable_creates_critical_failure(self):
        """MCP failure creates a CRITICAL severity failure."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = guard.evaluate(status)
        mcp_failures = [f for f in result.failures if f.dependency_name == "mcp"]
        assert len(mcp_failures) == 1
        assert mcp_failures[0].severity.value == "CRITICAL"

    def test_mcp_in_critical_failures_list(self):
        """MCP failure is in critical_failures list."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = guard.evaluate(status)
        critical_names = [f.dependency_name for f in result.critical_failures]
        assert "mcp" in critical_names


# ─────────────────────────────────────────────────────────────────────────────
# Test: FallbackRouter — Read Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFallbackReadOperations:
    """Test MCP fallback for read operations."""

    def test_read_fallback_uses_internal_adapter(self):
        """Read fallback uses InternalServiceAdapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_payment("PAY-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_read_fallback_search_records(self):
        """Search records falls back to internal adapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.search_financial_records(case_id="CASE-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_read_fallback_get_settlement(self):
        """Get settlement falls back to internal adapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_settlement("SET-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_read_fallback_get_refund(self):
        """Get refund falls back to internal adapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_refund("REF-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_read_fallback_get_fee(self):
        """Get fee falls back to internal adapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_fee("FEE-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_read_fallback_get_adjustment(self):
        """Get adjustment falls back to internal adapter."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_adjustment("ADJ-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL

    def test_read_fallback_records_audit(self):
        """Read fallback records audit event."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001")
        assert len(router.fallback_log) == 1
        assert router.fallback_log[0]["tool_name"] == "get_payment"
        assert router.fallback_log[0]["escalated"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: FallbackRouter — Write Operations
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFallbackWriteOperations:
    """Test MCP fallback for write operations — must escalate, never direct-write."""

    def test_write_create_resolution_escalates(self):
        """create_resolution escalates when MCP unavailable."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_write_verify_resolution_escalates(self):
        """verify_resolution escalates when MCP unavailable."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.verify_resolution(
            execution_id="EXEC-001",
            workflow_id="WF-001",
        )
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_write_record_feedback_escalates(self):
        """record_feedback escalates when MCP unavailable."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.record_feedback(
            workflow_id="WF-001",
            exception_id="EXC-001",
            feedback_type="APPROVE",
            reviewer="human",
            system_prediction="AUTO",
        )
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_write_never_direct_database(self):
        """Write operations never fall back to direct database writes."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        # Must be ESCALATED, not INTERNAL
        assert result.execution_path != ExecutionPath.INTERNAL
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_write_records_audit(self):
        """Write escalation records audit event."""
        router = MCPFallbackRouter(mcp_available=False)
        router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        assert len(router.fallback_log) == 1
        assert router.fallback_log[0]["escalated"] is True

    def test_write_error_message_retained(self):
        """Write escalation retains error message."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        assert result.error is not None
        assert "MCP unavailable" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Test: FallbackRouter — Similar Exception
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFallbackSimilarException:
    """Test MCP fallback for similar exception retrieval."""

    def test_similar_exception_escalates_when_mcp_unavailable(self):
        """Similar exception retrieval escalates when MCP unavailable."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_similar_exception("EXC-001")
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED

    def test_similar_exception_error_message(self):
        """Similar exception failure has clear error message."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_similar_exception("EXC-001")
        assert "MCP unavailable" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Test: FallbackRouter — Audit Trail
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFallbackAuditTrail:
    """Test MCP fallback audit trail."""

    def test_fallback_log_records_timestamp(self):
        """Fallback log records timestamp."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001")
        assert "timestamp" in router.fallback_log[0]

    def test_fallback_log_records_workflow_id(self):
        """Fallback log records workflow_id."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001", workflow_id="WF-001")
        assert router.fallback_log[0]["workflow_id"] == "WF-001"

    def test_fallback_log_records_exception_id(self):
        """Fallback log records exception_id."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001", exception_id="EXC-001")
        assert router.fallback_log[0]["exception_id"] == "EXC-001"

    def test_fallback_summary_counts(self):
        """Fallback summary counts correctly."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001")
        router.get_settlement("SET-001")
        router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        summary = router.get_fallback_summary()
        assert summary["total_fallbacks"] == 3
        assert summary["escalations"] == 1
        assert summary["successful_fallbacks"] == 2

    def test_multiple_operations_recorded(self):
        """Multiple operations are all recorded."""
        router = MCPFallbackRouter(mcp_available=False)
        router.get_payment("PAY-001")
        router.get_settlement("SET-001")
        router.get_refund("REF-001")
        assert len(router.fallback_log) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test: Decision Matrix — MCP Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixMCPFailure:
    """Test decision matrix with MCP failure scenarios."""

    def _make_gate_result(self, passed=True, confidence=0.85):
        from app.schemas.confidence_gate import ConfidenceGateResult
        return ConfidenceGateResult(
            passed=passed,
            action=GateAction.CONTINUE if passed else GateAction.HUMAN_REVIEW,
            confidence=confidence,
            threshold=0.70,
            reason="test",
        )

    def test_mcp_unavailable_blocks_auto(self):
        """MCP unavailable → FAIL_CLOSED → not AUTO."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = self._make_gate_result(True, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result()
        fallback_r = FailureFallbackResult(
            can_proceed=False,
            action=FallbackAction.FAIL_CLOSED,
            fallback_status="HUMAN_REVIEW",
            failures=[],
            critical_failures=[],
            failed_categories=[ErrorCategory.MCP_UNAVAILABLE],
            has_critical_failure=True,
            reason="MCP unavailable",
            exception_id="EXC-MCP-001",
            case_id="CASE-MCP-001",
        )
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_mcp_healthy_allows_auto(self):
        """MCP healthy → can proceed → AUTO possible."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = self._make_gate_result(True, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result()
        fallback_r = FailureFallbackResult(
            can_proceed=True,
            action=FallbackAction.CONTINUE_WITHOUT,
            fallback_status="",
            failures=[],
            critical_failures=[],
            failed_categories=[],
            has_critical_failure=False,
            reason="All healthy",
            exception_id="EXC-MCP-001",
            case_id="CASE-MCP-001",
        )
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Guardrail Engine — MCP Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngineMCPFailure:
    """Test the complete guardrail engine with MCP failure."""

    def test_mcp_unavailable_blocks_auto(self):
        """MCP unavailable → FAIL_CLOSED → not AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.85, risk="LOW")
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.decision != AutomationDecision.AUTO

    def test_mcp_unavailable_fallback_result_records_critical(self):
        """Fallback result records MCP as critical failure."""
        engine = GuardrailEngine()
        engine_r = _engine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.fallback_result is not None
        assert result.fallback_result.has_critical_failure is True
        mcp_failures = [
            f for f in result.fallback_result.failures
            if f.dependency_name == "mcp"
        ]
        assert len(mcp_failures) == 1

    def test_mcp_unavailable_system_unhealthy(self):
        """MCP unavailable → system_healthy is False."""
        engine = GuardrailEngine()
        engine_r = _engine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.system_healthy is False

    def test_mcp_healthy_allows_auto_path(self):
        """MCP healthy → AUTO path possible."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.85, risk="LOW")
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Test: FallbackRouter — MCP Client Exception Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPFallbackClientExceptions:
    """Test MCP fallback when client throws exceptions."""

    def test_mcp_client_exception_falls_back(self):
        """MCP client exception → read falls back to internal."""
        router = MCPFallbackRouter(mcp_available=True)
        # Mock the MCP client to throw
        router._mcp_client.call_tool = MagicMock(side_effect=Exception("Connection refused"))
        result = router.get_payment("PAY-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL
        assert result.fallback_used is True

    def test_mcp_client_exception_records_error(self):
        """MCP client exception records the error in fallback log."""
        router = MCPFallbackRouter(mcp_available=True)
        router._mcp_client.call_tool = MagicMock(side_effect=Exception("Timeout"))
        router.get_payment("PAY-001")
        assert len(router.fallback_log) == 1
        assert "Timeout" in router.fallback_log[0]["mcp_error"]

    def test_mcp_client_error_response_falls_back(self):
        """MCP client returns error response → read falls back."""
        router = MCPFallbackRouter(mcp_available=True)
        router._mcp_client.call_tool = MagicMock(return_value={
            "success": False,
            "error": "Tool execution failed",
        })
        result = router.get_payment("PAY-001")
        assert result.success is True
        assert result.execution_path == ExecutionPath.INTERNAL


# ─────────────────────────────────────────────────────────────────────────────
# Test: No Partial Unsafe Financial Action
# ─────────────────────────────────────────────────────────────────────────────


class TestNoPartialUnsafeAction:
    """Verify no partial unsafe financial action occurs during MCP failure."""

    def test_write_not_partially_executed(self):
        """Write operation either fully executes via MCP or escalates."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        # Must be fully escalated, not partially executed
        assert result.success is False
        assert result.execution_path == ExecutionPath.ESCALATED
        assert result.data is None

    def test_read_does_not_modify_data(self):
        """Read fallback only reads, never modifies."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.get_payment("PAY-001")
        assert result.success is True
        # Read-only — no modification fields
        assert not hasattr(result, "execute")
        assert not hasattr(result, "apply")

    def test_escalation_retains_request_id(self):
        """Escalation retains request_id for audit."""
        router = MCPFallbackRouter(mcp_available=False)
        result = router.create_resolution(
            exception_id="EXC-001",
            resolution_type="FEE_ADJUSTMENT",
            financial_adjustment_paise=5000,
            workflow_id="WF-001",
            guardrail_decision="AUTO",
            authorization_source="guardrail_engine",
            idempotency_key="IDEM-001",
        )
        assert result.request_id is not None
        assert result.request_id.startswith("ESCALATE-")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Combined MCP + Other Failures
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinedMCPAndOtherFailures:
    """Test MCP failure combined with other dependency failures."""

    def test_mcp_plus_database_both_fail(self):
        """MCP + database both fail → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": False,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is False
        assert result.has_critical_failure is True

    def test_mcp_plus_llm_both_fail(self):
        """MCP + LLM fail → FAIL_CLOSED (MCP is required)."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": False,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is False
        assert result.has_critical_failure is True

    def test_only_llm_fails_mcp_healthy(self):
        """Only LLM fails → can proceed (MCP is required, LLM is not)."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: ExecutionPath Enum
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionPathEnum:
    """Test execution path enum values."""

    def test_mcp_path(self):
        """MCP execution path exists."""
        assert ExecutionPath.MCP.value == "MCP"

    def test_internal_path(self):
        """INTERNAL execution path exists."""
        assert ExecutionPath.INTERNAL.value == "INTERNAL"

    def test_escalated_path(self):
        """ESCALATED execution path exists."""
        assert ExecutionPath.ESCALATED.value == "ESCALATED"

    def test_only_three_paths(self):
        """Only three execution paths exist."""
        assert len(ExecutionPath) == 3
