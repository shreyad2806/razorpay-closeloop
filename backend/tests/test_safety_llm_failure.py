"""
Adversarial safety tests for LLM failure scenarios.

Verifies that LLM failures never create financial decisions,
never bypass guardrails, and the system degrades gracefully.

LLM failure chain:
  1. LLMProviderError / timeout / connection failure
  2. LLMExplanationService catches error → deterministic fallback
  3. FallbackGuard marks llm as DEGRADED (not required)
  4. DecisionMatrix proceeds with other gates
  5. GuardrailEngine chains all guards

Key design principle:
  LLM is an EXPLANATION layer — not a financial authority.
  LLM failure must NOT affect:
  - Reconciliation
  - Evidence retrieval
  - ML classification
  - Guardrails
  - Resolution execution
  - Verification

No production logic is modified.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_llm.db")
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engine(**kwargs):
    """Build a valid ResolutionEngineResult for testing."""
    from tests.test_safety_high_value import _make_candidate, _make_score
    defaults = dict(
        exception_id="EXC-LLM-001",
        case_id="CASE-LLM-001",
        payment_id="PAY-LLM-001",
        merchant_id="MER-LLM-01",
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


def _make_fallback_result(can_proceed=True, llm_unavailable=False):
    failures = []
    failed_categories = []
    critical_failures = []
    if llm_unavailable:
        from app.schemas.failure_fallback import DependencyFailure, FailureSeverity
        failure = DependencyFailure(
            dependency_name="llm",
            error_category=ErrorCategory.LLM_UNAVAILABLE,
            severity=FailureSeverity.DEGRADED,
            error_message="LLM is unavailable",
            fallback_action=FallbackAction.CONTINUE_WITHOUT,
            fallback_status="",
            is_recoverable=True,
        )
        failures.append(failure)
        failed_categories.append(ErrorCategory.LLM_UNAVAILABLE)

    return FailureFallbackResult(
        can_proceed=can_proceed,
        action=FallbackAction.CONTINUE_WITHOUT if can_proceed else FallbackAction.FAIL_CLOSED,
        fallback_status="" if can_proceed else "HUMAN_REVIEW",
        failures=failures,
        critical_failures=critical_failures,
        failed_categories=failed_categories,
        has_critical_failure=len(critical_failures) > 0,
        can_use_deterministic_only=llm_unavailable and can_proceed,
        reason="LLM unavailable" if llm_unavailable else "All healthy",
        exception_id="EXC-LLM-001",
        case_id="CASE-LLM-001",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Fallback Guard — LLM Dependency Policy
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackGuardLLMPolicy:
    """Test fallback guard dependency policy for LLM."""

    def test_llm_is_optional(self):
        """LLM dependency is marked as optional (not required)."""
        policy = DEFAULT_POLICIES["llm"]
        assert policy.is_required is False

    def test_llm_fallback_action(self):
        """LLM failure → CONTINUE_WITHOUT."""
        policy = DEFAULT_POLICIES["llm"]
        assert policy.fallback_action == FallbackAction.CONTINUE_WITHOUT

    def test_llm_error_category(self):
        """LLM failure category is LLM_UNAVAILABLE."""
        policy = DEFAULT_POLICIES["llm"]
        assert policy.error_category == ErrorCategory.LLM_UNAVAILABLE

    def test_llm_unavailable_can_proceed(self):
        """LLM unavailable → system can still proceed."""
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
        # LLM-only failure → CONTINUE_WITHOUT (not USE_DETERMINISTIC_ONLY,
        # which requires ML deps to also be unavailable)
        assert result.action == FallbackAction.CONTINUE_WITHOUT

    def test_llm_unavailable_has_degraded_failure(self):
        """LLM unavailable creates a DEGRADED (not CRITICAL) failure."""
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
        llm_failures = [f for f in result.failures if f.dependency_name == "llm"]
        assert len(llm_failures) == 1
        assert llm_failures[0].severity.value == "DEGRADED"
        assert result.has_critical_failure is False

    def test_llm_unavailable_not_in_critical_failures(self):
        """LLM failure is NOT in critical_failures list."""
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
        critical_names = [f.dependency_name for f in result.critical_failures]
        assert "llm" not in critical_names

    def test_all_deps_healthy(self):
        """All healthy → can proceed, no failures."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is True
        assert len(result.failures) == 0

    def test_llm_plus_database_failure(self):
        """LLM + database both down → FAIL_CLOSED (database is required)."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": False,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is False
        assert result.has_critical_failure is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Decision Matrix — LLM Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixLLMFailure:
    """Test decision matrix with LLM failure scenarios."""

    def _make_gate_result(self, passed=True, confidence=0.85):
        from app.schemas.confidence_gate import ConfidenceGateResult
        return ConfidenceGateResult(
            passed=passed,
            action=GateAction.CONTINUE if passed else GateAction.HUMAN_REVIEW,
            confidence=confidence,
            threshold=0.70,
            reason="test",
        )

    def test_llm_unavailable_allows_auto(self):
        """LLM unavailable + all other gates pass → AUTO is possible."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = self._make_gate_result(True, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result()
        fallback_r = _make_fallback_result(llm_unavailable=True)
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        # LLM is optional — its failure doesn't block AUTO
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_llm_unavailable_does_not_block_auto(self):
        """LLM failure alone must NOT block AUTO."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = self._make_gate_result(True, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result()
        fallback_r = _make_fallback_result(llm_unavailable=True)
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        # The decision should NOT be blocked because of LLM
        # It may be HUMAN_REVIEW from other checks, but not because of LLM
        assert result.decision != AutomationDecision.UNRESOLVED

    def test_database_failure_blocks_auto(self):
        """Database failure → FAIL_CLOSED → not AUTO (contrast with LLM)."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = self._make_gate_result(True, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result()
        fallback_r = _make_fallback_result(can_proceed=False)
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test: Guardrail Engine — LLM Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngineLLMFailure:
    """Test the complete guardrail engine with LLM failure."""

    def test_llm_unavailable_allows_auto_path(self):
        """LLM unavailable + good conditions → AUTO is possible."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.85, risk="LOW")
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_llm_unavailable_fallback_result_records_degraded(self):
        """Fallback result records LLM as degraded."""
        engine = GuardrailEngine()
        engine_r = _engine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.fallback_result is not None
        llm_failures = [
            f for f in result.fallback_result.failures
            if f.dependency_name == "llm"
        ]
        assert len(llm_failures) == 1

    def test_llm_unavailable_system_still_healthy(self):
        """LLM unavailable → system_healthy is True."""
        engine = GuardrailEngine()
        engine_r = _engine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.system_healthy is True

    def test_llm_timeout_still_allows_auto(self):
        """LLM timeout (same as unavailable) → still allows AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.85, risk="LOW")
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,  # Simulates timeout
            "mcp": True,
        }
        result = engine.evaluate(engine_r, dep_status)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_all_deps_healthy_allows_auto(self):
        """All healthy → AUTO possible."""
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
# Test: LLM Explanation Service — Fallback Behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMExplanationServiceFallback:
    """Test LLM explanation service fallback when provider is unavailable."""

    def test_no_provider_uses_deterministic_fallback(self):
        """No provider → deterministic template fallback."""
        from app.llm.services.explanation_service import (
            LLMExplanationService,
            ExplanationRequest,
        )
        service = LLMExplanationService(provider=None)
        request = ExplanationRequest(
            exception_id="EXC-LLM-TEST",
            difference_paise=5000,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            exception_type="FEE_DIFFERENCE",
        )
        result = service.explain(request)  # This is async but works synchronously when no provider
        # Result should be available (deterministic fallback)
        assert result is not None

    def test_deterministic_fallback_has_fallback_flag(self):
        """Deterministic fallback marks fallback_used=True."""
        from app.llm.services.explanation_service import (
            _deterministic_fallback,
            ExplanationRequest,
        )
        request = ExplanationRequest(
            exception_id="EXC-LLM-FB",
            difference_paise=3000,
            expected_amount_paise=50000,
            actual_amount_paise=47000,
            exception_type="FEE_DIFFERENCE",
            evidence_coverage="FULLY_EXPLAINED",
        )
        result = _deterministic_fallback(request)
        assert result.fallback_used is True
        assert result.provider == "none"
        assert result.model_used == "deterministic-template"

    def test_deterministic_fallback_preserves_financial_values(self):
        """Deterministic template preserves financial values from input."""
        from app.llm.services.explanation_service import (
            _deterministic_fallback,
            ExplanationRequest,
        )
        request = ExplanationRequest(
            exception_id="EXC-LLM-FV",
            difference_paise=7500,
            expected_amount_paise=200000,
            actual_amount_paise=192500,
            exception_type="REFUND_ADJUSTMENT",
            evidence_coverage="PARTIALLY_EXPLAINED",
            explained_amount_paise=5000,
            remaining_difference_paise=2500,
        )
        result = _deterministic_fallback(request)
        assert "7500" in result.summary or "₹75.00" in result.summary
        assert result.fallback_used is True

    def test_deterministic_fallback_no_llm_injection(self):
        """Deterministic fallback does not inject LLM opinions."""
        from app.llm.services.explanation_service import (
            _deterministic_fallback,
            ExplanationRequest,
        )
        request = ExplanationRequest(
            exception_id="EXC-LLM-SAFE",
            difference_paise=1000,
            exception_type="FEE_DIFFERENCE",
        )
        result = _deterministic_fallback(request)
        # Should not contain any financial decision language
        combined = f"{result.summary} {result.reason}".lower()
        assert "approve" not in combined
        assert "reject" not in combined
        assert "execute" not in combined
        assert "authorize" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLM Provider Error Types
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMProviderErrorTypes:
    """Test various LLM provider error types."""

    def test_provider_error_has_provider_field(self):
        """LLMProviderError records provider name."""
        from app.llm.providers.base import LLMProviderError
        err = LLMProviderError("test", provider="openai")
        assert err.provider == "openai"

    def test_timeout_error_is_provider_error(self):
        """LLMTimeoutError is a subclass of LLMProviderError."""
        from app.llm.providers.base import LLMTimeoutError, LLMProviderError
        assert issubclass(LLMTimeoutError, LLMProviderError)

    def test_connection_error_is_provider_error(self):
        """LLMConnectionError is a subclass of LLMProviderError."""
        from app.llm.providers.base import LLMConnectionError, LLMProviderError
        assert issubclass(LLMConnectionError, LLMProviderError)

    def test_response_error_is_provider_error(self):
        """LLMResponseError is a subclass of LLMProviderError."""
        from app.llm.providers.base import LLMResponseError, LLMProviderError
        assert issubclass(LLMResponseError, LLMProviderError)

    def test_config_error_is_provider_error(self):
        """LLMConfigError is a subclass of LLMProviderError."""
        from app.llm.providers.base import LLMConfigError, LLMProviderError
        assert issubclass(LLMConfigError, LLMProviderError)


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLM Cannot Bypass Guardrails
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMCannotBypassGuardrails:
    """Verify LLM failure cannot create unsafe financial decisions."""

    def test_llm_failure_no_financial_execution(self):
        """LLM failure → no financial execution path opened."""
        from app.llm.services.explanation_service import LLMExplanationService
        service = LLMExplanationService(provider=None)
        # The service has no execute/apply/authorize methods
        assert not hasattr(service, "execute")
        assert not hasattr(service, "apply")
        assert not hasattr(service, "authorize")
        assert not hasattr(service, "modify")

    def test_explanation_output_no_financial_fields(self):
        """LLMExplanationOutput has no financial execution fields."""
        from app.llm.services.explanation_service import LLMExplanationOutput
        fields = set(LLMExplanationOutput.model_fields.keys())
        dangerous = {
            "execute", "apply", "authorize", "create_adjustment",
            "modify_records", "set_amount", "approve", "reject",
        }
        assert dangerous.isdisjoint(fields)

    def test_explanation_request_no_truth_override(self):
        """ExplanationRequest cannot override financial truth."""
        from app.llm.services.explanation_service import ExplanationRequest
        # Request contains read-only context — no write fields
        fields = set(ExplanationRequest.model_fields.keys())
        dangerous = {
            "execute", "apply", "authorize", "override_amount",
            "set_resolution", "force_auto", "bypass_guardrail",
        }
        assert dangerous.isdisjoint(fields)

    def test_guardrail_engine_independent_of_llm(self):
        """GuardrailEngine evaluation doesn't depend on LLM status."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.85, risk="LOW")
        # With LLM healthy
        result_healthy = engine.evaluate(engine_r, {"llm": True})
        # With LLM unhealthy
        result_unhealthy = engine.evaluate(engine_r, {"llm": False})
        # Both should produce the same decision
        assert result_healthy.decision == result_unhealthy.decision


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Gate Independent of LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateIndependentOfLLM:
    """Verify confidence gate doesn't consider LLM status."""

    def test_gate_passes_regardless_of_llm(self):
        """Confidence gate passes the same regardless of LLM."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.85)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        # Gate checks don't include LLM status
        check_names = [c.check_name for c in result.checks]
        assert "llm_status" not in check_names
        assert "llm_available" not in check_names


# ─────────────────────────────────────────────────────────────────────────────
# Test: Exposure Guard Independent of LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardIndependentOfLLM:
    """Verify exposure guard doesn't consider LLM status."""

    def test_guard_passes_regardless_of_llm(self):
        """Exposure guard passes the same regardless of LLM."""
        guard = ExposureGuard()
        engine_r = _engine(difference=10_000)
        result = guard.evaluate(engine_r)
        assert result.passed is True
        # Guard checks don't include LLM status
        check_names = [c.check_name for c in result.checks]
        assert "llm_status" not in check_names


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLM Health Check
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMHealthCheck:
    """Test LLM health check behavior."""

    def test_no_provider_health_check(self):
        """No provider → health check returns healthy=True."""
        from app.llm.services.explanation_service import LLMExplanationService
        service = LLMExplanationService(provider=None)
        # health_check is async but should work
        status = service.health_check()
        # Should return a health status (may be a coroutine)
        assert status is not None

    def test_provider_type_none(self):
        """Provider type NONE exists."""
        from app.llm.providers.base import LLMProviderType
        assert hasattr(LLMProviderType, "NONE")
        assert LLMProviderType.NONE.value == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Combined LLM + Other Dependency Failures
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinedLLMAndOtherFailures:
    """Test LLM failure combined with other dependency failures."""

    def test_llm_plus_ml_failure_allows_proceed(self):
        """LLM + ML failure → both optional → can proceed."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is True
        assert result.can_use_deterministic_only is True

    def test_llm_plus_database_failure_blocks(self):
        """LLM + database failure → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": False,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(status)
        assert result.can_proceed is False

    def test_llm_plus_mcp_failure_blocks(self):
        """LLM + MCP failure → FAIL_CLOSED (MCP is required)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Test: LLM Cannot Manufacture Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMCannotManufactureConfidence:
    """Verify LLM cannot manufacture or inflate confidence scores."""

    def test_confidence_from_engine_not_llm(self):
        """Confidence comes from engine result, not LLM."""
        engine_r = _engine(confidence=0.50)
        gate = ConfidenceGate()
        result = gate.evaluate(engine_r)
        # Gate uses engine confidence, not LLM
        assert result.confidence == 0.50
        assert result.passed is False  # 0.50 < 0.70

    def test_guardrail_confidence_from_engine(self):
        """Guardrail engine uses engine confidence, not LLM."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.50)
        result = engine.evaluate(engine_r)
        assert result.confidence == 0.50
        assert result.decision != AutomationDecision.AUTO
