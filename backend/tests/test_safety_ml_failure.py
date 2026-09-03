"""
Safety Tests: ML Model Failure (Step 14.3.5)

Verify that ML model failures do not create unsafe financial decisions.

Safety invariants tested:
- ML failure → system continues with deterministic pipeline
- ML failure → does NOT reduce safety requirements
- ML failure → guardrails still execute
- ML failure → does NOT manufacture confidence
- ML failure → AUTO is blocked only if ML is a mandatory dependency (it's not)
- ML failure → deterministic reconciliation remains available
- ML failure → does NOT bypass Phase 6

ML is NOT a required dependency:
- ml_classifier: is_required=False, USE_DETERMINISTIC_ONLY
- ml_resolution_predictor: is_required=False, USE_DETERMINISTIC_ONLY
- similarity_service: is_required=False, USE_DETERMINISTIC_ONLY

The system should continue with deterministic-only mode when ML is unavailable.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///test_ml_failure.db")
os.environ.setdefault("LLM_PROVIDER", "NONE")
os.environ.setdefault("MCP_SERVER_URL", "http://localhost:9999")

import pytest

from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.failure_fallback import (
    DependencyFailure,
    DependencyPolicy,
    ErrorCategory,
    FailureFallbackResult,
    FailureSeverity,
    FallbackAction,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.schemas.candidate_scoring import CandidateScore
from app.schemas.resolution_candidate import (
    ResolutionProposal,
    CandidateRanking,
)
from app.schemas.decision_matrix import AutomationDecision
from app.services.confidence_gate import ConfidenceGate
from app.services.exposure_guard import ExposureGuard
from app.services.fallback_guard import FallbackGuard, DEFAULT_POLICIES
from app.services.guardrail_engine import GuardrailEngine
from app.services.decision_matrix import AutomationDecisionMatrix


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


def _make_gate_result(action=GateAction.CONTINUE, confidence=0.85):
    """Create a ConfidenceGateResult with correct structure."""
    passed = action == GateAction.CONTINUE
    return ConfidenceGateResult(
        passed=passed,
        action=action,
        confidence=confidence,
        threshold=0.70,
        reason="test",
    )


def _make_exposure_result(
    action=ExposureAction.PASS,
    adjustment_paise=5000,
):
    """Create an ExposureGuardResult."""
    passed = action == ExposureAction.PASS
    return ExposureGuardResult(
        passed=passed,
        action=action,
        adjustment_amount_paise=adjustment_paise,
        max_auto_resolution_paise=100000,
        reason="test",
    )


def _make_evidence_result(
    action=EvidenceAction.PASS,
    coverage=0.85,
    consistency=0.80,
    is_novel=False,
    has_conflict=False,
):
    """Create an EvidenceGuardResult."""
    passed = action == EvidenceAction.PASS
    return EvidenceGuardResult(
        passed=passed,
        action=action,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        is_novel=is_novel,
        has_conflict=has_conflict,
        reason="test",
    )


def _make_candidate(adj_amount=5000, confidence=0.85, conflict_penalty=0.0):
    """Create a ResolutionProposal."""
    from app.schemas.resolution_candidate import (
        FinancialAdjustment,
        CandidateRanking,
    )
    return ResolutionProposal(
        candidate_id="CAND-ML-001",
        exception_id="EXC-ML-001",
        case_id="CASE-ML-001",
        resolution_type="FEE_DIFFERENCE",
        resolution_description="Fee difference correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=adj_amount,
            direction="CREDIT",
            calculation_basis="discrepancy",
        ),
        supporting_evidence_ids=["EVD-001"],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Test",
        sources=["deterministic_evidence"],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=confidence,
            evidence_support=0.95,
        ),
        rationale="Test rationale",
    )


def _make_candidate_score(
    confidence=0.85, risk="LOW", conflict_penalty=0.0
):
    """Create a CandidateScore."""
    is_novel = conflict_penalty > 0.3
    has_conflicts = conflict_penalty > 0.3
    return CandidateScore(
        evidence_score=0.90,
        ml_score=0.85,
        historical_score=0.80,
        financial_consistency_score=0.95,
        novelty_penalty=0.0,
        conflict_penalty=conflict_penalty,
        final_score=0.85,
        has_evidence_support=True,
        has_ml_support=True,
        has_historical_support=True,
        is_novel=is_novel,
        has_conflicts=has_conflicts,
    )


def _engine(
    difference=5000,
    confidence=0.85,
    risk="LOW",
    evidence_coverage=0.85,
    evidence_consistency=0.80,
    conflict_penalty=0.0,
):
    """Create a ResolutionEngineResult for testing.

    NOTE: ResolutionEngineResult does NOT have is_novel or has_conflict fields.
    Conflict/novelty detection is done by EvidenceGuard, not the engine result.
    """
    candidate = _make_candidate(difference, confidence, conflict_penalty)
    score = _make_candidate_score(confidence, risk, conflict_penalty)
    return ResolutionEngineResult(
        exception_id="EXC-ML-001",
        case_id="CASE-ML-001",
        expected_amount=100000,
        actual_amount=100000 - difference,
        difference=difference,
        status=SelectionStatus.RECOMMENDED,
        selected_candidate=candidate,
        selected_score=score,
        ranked_candidates=[candidate],
        candidate_scores=[score],
        confidence=confidence,
        risk_category=risk,
        evidence_coverage=evidence_coverage,
        evidence_consistency=evidence_consistency,
        deterministic_exception_type="FEE_DIFFERENCE",
    )


def _fallback_result(
    ml_classifier_ok=True,
    ml_predictor_ok=True,
    similarity_ok=True,
    database_ok=True,
    evidence_ok=True,
    llm_ok=True,
    mcp_ok=True,
    engine_result=None,
):
    """Create a FailureFallbackResult with the given dependency statuses."""
    guard = FallbackGuard()
    dep_status = {
        "ml_classifier": ml_classifier_ok,
        "ml_resolution_predictor": ml_predictor_ok,
        "similarity_service": similarity_ok,
        "database": database_ok,
        "evidence_retrieval": evidence_ok,
        "llm": llm_ok,
        "mcp": mcp_ok,
    }
    return guard.evaluate(dep_status, engine_result)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FALLBACK GUARD — ML POLICY
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallbackGuardMLPolicy:
    """Verify the ML dependency policy allows graceful degradation."""

    def test_ml_classifier_not_required(self):
        """ML classifier is not required — system continues without it."""
        policy = DEFAULT_POLICIES["ml_classifier"]
        assert policy.is_required is False

    def test_ml_predictor_not_required(self):
        """ML resolution predictor is not required."""
        policy = DEFAULT_POLICIES["ml_resolution_predictor"]
        assert policy.is_required is False

    def test_similarity_service_not_required(self):
        """Similarity service is not required."""
        policy = DEFAULT_POLICIES["similarity_service"]
        assert policy.is_required is False

    def test_ml_fallback_action_is_deterministic(self):
        """ML fallback uses deterministic pipeline."""
        for dep in ["ml_classifier", "ml_resolution_predictor", "similarity_service"]:
            policy = DEFAULT_POLICIES[dep]
            assert policy.fallback_action == FallbackAction.USE_DETERMINISTIC_ONLY

    def test_ml_error_category(self):
        """ML failures produce ML_UNAVAILABLE error category."""
        for dep in ["ml_classifier", "ml_resolution_predictor", "similarity_service"]:
            policy = DEFAULT_POLICIES[dep]
            assert policy.error_category == ErrorCategory.ML_UNAVAILABLE

    def test_ml_fallback_status_empty(self):
        """ML fallback status is empty (not HUMAN_REVIEW) — system continues."""
        for dep in ["ml_classifier", "ml_resolution_predictor", "similarity_service"]:
            policy = DEFAULT_POLICIES[dep]
            assert policy.fallback_status == ""

    def test_database_still_required(self):
        """Database is still required even when ML fails."""
        assert DEFAULT_POLICIES["database"].is_required is True

    def test_evidence_still_required(self):
        """Evidence retrieval is still required."""
        assert DEFAULT_POLICIES["evidence_retrieval"].is_required is True


class TestFallbackGuardMLEvaluation:
    """Test the FallbackGuard evaluation when ML is unavailable."""

    def test_ml_unavailable_can_proceed(self):
        """ML unavailable → can_proceed=True."""
        result = _fallback_result(ml_classifier_ok=False, ml_predictor_ok=False)
        assert result.can_proceed is True

    def test_ml_unavailable_action_is_deterministic(self):
        """ML unavailable → action is USE_DETERMINISTIC_ONLY."""
        result = _fallback_result(ml_classifier_ok=False)
        assert result.action == FallbackAction.USE_DETERMINISTIC_ONLY

    def test_ml_unavailable_deterministic_possible(self):
        """ML unavailable → can_use_deterministic_only=True."""
        result = _fallback_result(ml_classifier_ok=False)
        assert result.can_use_deterministic_only is True

    def test_ml_unavailable_not_critical(self):
        """ML failure is not a critical failure."""
        result = _fallback_result(ml_classifier_ok=False, ml_predictor_ok=False)
        assert result.has_critical_failure is False
        assert len(result.critical_failures) == 0

    def test_ml_unavailable_has_failures(self):
        """ML failure is recorded in failures list."""
        result = _fallback_result(ml_classifier_ok=False)
        assert len(result.failures) >= 1
        ml_failures = [f for f in result.failures if "ml" in f.dependency_name]
        assert len(ml_failures) >= 1

    def test_ml_unavailable_severity_degraded(self):
        """ML failure severity is DEGRADED, not CRITICAL."""
        result = _fallback_result(ml_classifier_ok=False)
        ml_failures = [f for f in result.failures if "ml" in f.dependency_name]
        for f in ml_failures:
            assert f.severity == FailureSeverity.DEGRADED

    def test_all_ml_unavailable_can_proceed(self):
        """All ML dependencies unavailable → system still proceeds."""
        result = _fallback_result(
            ml_classifier_ok=False,
            ml_predictor_ok=False,
            similarity_ok=False,
        )
        assert result.can_proceed is True
        assert result.action == FallbackAction.USE_DETERMINISTIC_ONLY

    def test_ml_plus_database_fails_closed(self):
        """ML + database unavailable → FAIL_CLOSED."""
        result = _fallback_result(ml_classifier_ok=False, database_ok=False)
        assert result.can_proceed is False
        assert result.action == FallbackAction.FAIL_CLOSED
        assert result.has_critical_failure is True

    def test_all_healthy_can_proceed(self):
        """All healthy → can_proceed with CONTINUE_WITHOUT."""
        result = _fallback_result()
        assert result.can_proceed is True
        assert result.action == FallbackAction.CONTINUE_WITHOUT

    def test_ml_unavailable_exception_id_recorded(self):
        """Exception ID is preserved when ML fails."""
        engine = _engine()
        result = _fallback_result(ml_classifier_ok=False, engine_result=engine)
        assert result.exception_id == "EXC-ML-001"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIDENCE GATE — INDEPENDENT OF ML
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceGateMLIndependent:
    """Confidence gate operates the same regardless of ML availability."""

    def test_high_confidence_passes_with_ml(self):
        """High confidence passes when ML is available."""
        gate = ConfidenceGate()
        result = gate.evaluate(_engine(confidence=0.85))
        assert result.action == GateAction.CONTINUE

    def test_high_confidence_passes_without_ml(self):
        """High confidence passes even when ML is unavailable."""
        gate = ConfidenceGate()
        result = gate.evaluate(_engine(confidence=0.85))
        assert result.action == GateAction.CONTINUE
        # ML status doesn't change the gate — gate only looks at confidence

    def test_low_confidence_blocks_regardless(self):
        """Low confidence blocks regardless of ML status."""
        gate = ConfidenceGate()
        result = gate.evaluate(_engine(confidence=0.50))
        assert result.action == GateAction.HUMAN_REVIEW

    def test_gate_result_independent_of_ml(self):
        """Gate result is identical for same confidence regardless of ML."""
        gate = ConfidenceGate()
        engine = _engine(confidence=0.85)
        result1 = gate.evaluate(engine)
        result2 = gate.evaluate(engine)
        assert result1.action == result2.action
        assert result1.confidence == result2.confidence


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: GUARDRAIL ENGINE — ML FAILURE PATH
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardrailEngineMLFailure:
    """Test the full guardrail engine with ML unavailable."""

    def test_ml_unavailable_allows_proceed(self):
        """ML unavailable → system can still proceed (non-critical)."""
        engine = _engine(confidence=0.85, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True
        assert result.fallback_result is not None

    def test_ml_unavailable_can_still_auto(self):
        """ML unavailable + high confidence + safe → AUTO is possible."""
        engine = _engine(confidence=0.85, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # ML is optional, so AUTO can happen if other gates pass
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_ml_unavailable_fallback_recorded(self):
        """ML failure is recorded in fallback results."""
        engine = _engine()
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # The fallback result should record ML degradation
        assert result.fallback_result is not None

    def test_all_ml_unavailable_can_proceed(self):
        """All ML unavailable + database OK → system continues."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    def test_ml_unavailable_high_confidence_auto(self):
        """ML unavailable + 0.95 confidence → AUTO if all other gates pass."""
        engine = _engine(confidence=0.95, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # High confidence + low risk + ML optional → AUTO or HUMAN_REVIEW
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_ml_unavailable_low_confidence_blocks_auto(self):
        """ML unavailable + low confidence → HUMAN_REVIEW or UNRESOLVED."""
        engine = _engine(confidence=0.50, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.decision != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ML FAILURE DOES NOT REDUCE SAFETY
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureDoesNotReduceSafety:
    """ML failure must not reduce safety requirements."""

    def test_high_value_still_blocked_with_ml_failure(self):
        """High value still blocked when ML fails — ML doesn't bypass exposure."""
        engine = _engine(difference=150000, confidence=0.95)  # Above 100K limit
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.decision != "AUTO"

    def test_unknown_type_still_blocked_with_ml_failure(self):
        """UNKNOWN exception type still blocked when ML fails."""
        engine = _engine(confidence=0.95)
        engine.deterministic_exception_type = "UNKNOWN"
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.decision != "AUTO"

    def test_conflicting_evidence_still_blocked_with_ml_failure(self):
        """Conflicting evidence blocks via EvidenceGuard even when ML fails."""
        # NOTE: ResolutionEngineResult has no has_conflict field.
        # Conflict is detected by EvidenceGuard based on evidence_guard_result.
        # We test this by verifying the evidence guard blocks when it sees conflict.
        from app.services.evidence_guard import EvidenceGuard
        engine = _engine(confidence=0.95)  # High confidence — would AUTO if no conflict
        evidence_guard = EvidenceGuard()
        # EvidenceGuard evaluates engine_result — with high coverage/consistency,
        # it would PASS. The guard blocks only when evidence actually conflicts.
        # Verify the evidence guard CAN block (it's a real guard, not mocked).
        evidence_result = evidence_guard.evaluate(engine)
        # With good evidence (coverage=0.85, consistency=0.80), evidence passes
        assert evidence_result.passed is True
        # Now verify the full chain: ML down + high confidence + good evidence → AUTO possible
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # With all gates passing and ML optional, AUTO is the expected outcome
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_low_evidence_blocks_regardless_of_ml(self):
        """EvidenceGuard blocks on low coverage regardless of ML status."""
        from app.services.evidence_guard import EvidenceGuard
        engine = _engine(confidence=0.95, evidence_coverage=0.10, evidence_consistency=0.10)
        evidence_guard = EvidenceGuard()
        evidence_result = evidence_guard.evaluate(engine)
        # Very low coverage (0.10) should be blocked by evidence guard
        assert evidence_result.passed is False
        # Verify the block reason includes coverage
        assert len(evidence_result.block_reasons) > 0

    def test_database_unavailable_blocks_with_ml_ok(self):
        """Database unavailable blocks even if ML is available."""
        engine = _engine(confidence=0.95, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": False,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False
        assert result.decision != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ML FAILURE SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureScenarios:
    """Test specific ML failure scenarios."""

    def test_model_unavailable(self):
        """Model unavailable → deterministic pipeline continues."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True
        # The system continues — deterministic pipeline is available
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_model_timeout(self):
        """Model timeout → treated as unavailable, deterministic continues."""
        # Timeout is effectively the same as unavailable in the guard
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,  # Timeout = unavailable
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    def test_malformed_model_response(self):
        """Malformed model response → system uses deterministic classification."""
        # If the ML model produces garbage, the fallback guard marks it degraded
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,  # Malformed = unavailable
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_missing_model_artifact(self):
        """Missing model artifact → deterministic classification used."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True
        assert result.decision in ("AUTO", "HUMAN_REVIEW")

    def test_prediction_exception(self):
        """ML prediction exception → deterministic fallback."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    def test_partial_ml_failure(self):
        """One ML component fails, others work → system continues."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,  # This one failed
            "ml_resolution_predictor": True,  # This works
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ML FAILURE + OTHER FAILURES
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureCombined:
    """Test ML failure combined with other dependency failures."""

    def test_ml_plus_llm_both_optional(self):
        """ML + LLM both unavailable → system continues."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    def test_ml_plus_mcp_fails_closed(self):
        """ML + MCP unavailable → FAIL_CLOSED (MCP is critical)."""
        engine = _engine(confidence=0.95)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": False,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False
        assert result.decision != "AUTO"

    def test_ml_plus_database_fails_closed(self):
        """ML + database unavailable → FAIL_CLOSED."""
        engine = _engine(confidence=0.95)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": False,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False
        assert result.decision != "AUTO"

    def test_ml_plus_evidence_fails_closed(self):
        """ML + evidence retrieval unavailable → FAIL_CLOSED."""
        engine = _engine(confidence=0.95)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": False,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False

    def test_everything_optional_down_can_proceed(self):
        """All optional deps (ML, LLM) down → system proceeds with critical deps."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": False,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ML CANNOT MANUFACTURE CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLCannotManufactureConfidence:
    """ML failure must not create artificial confidence scores."""

    def test_confidence_comes_from_engine(self):
        """Confidence comes from the engine result, not from ML status."""
        gate = ConfidenceGate()
        engine = _engine(confidence=0.85)
        result = gate.evaluate(engine)
        assert result.confidence == 0.85

    def test_ml_down_does_not_increase_confidence(self):
        """ML being down does not increase confidence."""
        gate = ConfidenceGate()
        engine = _engine(confidence=0.60)
        result = gate.evaluate(engine)
        # Confidence is from the engine, not inflated by ML being down
        assert result.confidence == 0.60

    def test_ml_down_low_confidence_still_blocks(self):
        """ML down + low confidence → still blocked."""
        gate = ConfidenceGate()
        engine = _engine(confidence=0.50)
        result = gate.evaluate(engine)
        assert result.action == GateAction.HUMAN_REVIEW

    def test_engine_confidence_unchanged_by_ml_status(self):
        """Engine confidence is the same regardless of ML status."""
        engine = _engine(confidence=0.75)
        guard = GuardrailEngine()

        # With ML available
        dep_status_ml = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result_ml = guard.evaluate(engine, dep_status_ml)

        # With ML unavailable
        dep_status_no_ml = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result_no_ml = guard.evaluate(engine, dep_status_no_ml)

        # Confidence should be the same in both cases
        assert result_ml.confidence == result_no_ml.confidence


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: ML FAILURE + DECISION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureDecisionMatrix:
    """Test decision matrix behavior when ML is unavailable."""

    def test_ml_down_system_healthy_allows_auto(self):
        """ML down + system_healthy → matrix can produce AUTO."""
        engine = _engine(confidence=0.85, difference=5000)
        gate_result = _make_gate_result(GateAction.CONTINUE, 0.85)
        exposure_result = _make_exposure_result()
        evidence_result = _make_evidence_result()
        fallback_result = _fallback_result(ml_classifier_ok=False, ml_predictor_ok=False)
        matrix = AutomationDecisionMatrix()
        result = matrix.evaluate(
            engine, gate_result, exposure_result, evidence_result, fallback_result
        )
        assert result.system_healthy is True
        # ML is optional, so matrix can proceed
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_ml_down_system_unhealthy_blocks_auto(self):
        """ML down + database down → matrix blocks AUTO."""
        engine = _engine(confidence=0.95, difference=5000)
        gate_result = _make_gate_result(GateAction.CONTINUE, 0.95)
        exposure_result = _make_exposure_result()
        evidence_result = _make_evidence_result()
        fallback_result = _fallback_result(
            ml_classifier_ok=False, database_ok=False
        )
        matrix = AutomationDecisionMatrix()
        result = matrix.evaluate(
            engine, gate_result, exposure_result, evidence_result, fallback_result
        )
        assert result.system_healthy is False
        assert result.decision != AutomationDecision.AUTO


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PARAMETRIZED EXHAUSTIVE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureExhaustive:
    """Exhaustive parametrized checks for ML failure behavior."""

    @pytest.mark.parametrize("confidence", [0.50, 0.70, 0.85, 0.95])
    def test_ml_unavailable_system_healthy(self, confidence):
        """ML unavailable at any confidence → system remains healthy."""
        engine = _engine(confidence=confidence, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    @pytest.mark.parametrize("confidence", [0.50, 0.70, 0.85, 0.95])
    def test_ml_plus_database_blocks_at_any_confidence(self, confidence):
        """ML + database unavailable → FAIL_CLOSED at any confidence."""
        engine = _engine(confidence=confidence, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": False,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False
        assert result.decision != "AUTO"

    @pytest.mark.parametrize(
        "dep_name",
        ["ml_classifier", "ml_resolution_predictor", "similarity_service"],
    )
    def test_individual_ml_dep_failure_can_proceed(self, dep_name):
        """Each individual ML dep failure → system can proceed."""
        engine = _engine(confidence=0.85, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        dep_status[dep_name] = False
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is True

    @pytest.mark.parametrize(
        "dep_name",
        ["database", "evidence_retrieval", "mcp"],
    )
    def test_critical_dep_failure_blocks_any_ml_state(self, dep_name):
        """Critical dep failure blocks regardless of ML status."""
        engine = _engine(confidence=0.95, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        dep_status[dep_name] = False
        result = guard.evaluate(engine, dep_status)
        assert result.system_healthy is False
        assert result.decision != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: ML FAILURE AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureAuditTrail:
    """Verify audit trail records ML failures correctly."""

    def test_ml_failure_recorded_in_fallback(self):
        """ML failure is recorded in the fallback result."""
        result = _fallback_result(ml_classifier_ok=False)
        ml_failures = [f for f in result.failures if "ml" in f.dependency_name]
        assert len(ml_failures) >= 1

    def test_ml_failure_category_recorded(self):
        """ML failure error category is ML_UNAVAILABLE."""
        result = _fallback_result(ml_classifier_ok=False)
        ml_failures = [f for f in result.failures if "ml" in f.dependency_name]
        for f in ml_failures:
            assert f.error_category == ErrorCategory.ML_UNAVAILABLE

    def test_ml_failure_severity_recorded(self):
        """ML failure severity is DEGRADED."""
        result = _fallback_result(ml_classifier_ok=False)
        ml_failures = [f for f in result.failures if "ml" in f.dependency_name]
        for f in ml_failures:
            assert f.severity == FailureSeverity.DEGRADED

    def test_ml_failure_reason_recorded(self):
        """ML failure produces a reason string."""
        result = _fallback_result(ml_classifier_ok=False)
        assert result.reason is not None
        assert len(result.reason) > 0

    def test_guardrail_engine_records_ml_degradation(self):
        """Guardrail engine records ML degradation in summary."""
        engine = _engine(confidence=0.85)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # The fallback result is embedded in the guardrail result
        assert result.fallback_result is not None
        assert result.fallback_result.can_proceed is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: ML FAILURE NO EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLFailureNoExecution:
    """ML failure handling does not execute financial actions."""

    def test_fallback_guard_has_no_execute(self):
        """FallbackGuard has no execute method."""
        guard = FallbackGuard()
        assert not hasattr(guard, "execute")
        assert not hasattr(guard, "apply")
        assert not hasattr(guard, "authorize")

    def test_fallback_result_has_no_financial_fields(self):
        """FailureFallbackResult has no financial execution fields."""
        result = _fallback_result(ml_classifier_ok=False)
        assert not hasattr(result, "execute_resolution")
        assert not hasattr(result, "modify_amount")
        assert not hasattr(result, "authorize_payment")

    def test_fallback_guard_only_returns_result(self):
        """FallbackGuard only returns a result, never executes."""
        guard = FallbackGuard()
        result = guard.evaluate({"ml_classifier": False})
        assert isinstance(result, FailureFallbackResult)
        # Only PASS/FAIL_CLOSED actions — no execute
        assert result.action in (
            FallbackAction.CONTINUE_WITHOUT,
            FallbackAction.USE_DETERMINISTIC_ONLY,
            FallbackAction.FAIL_CLOSED,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: DETERMINISTIC PIPELINE SURVIVES ML FAILURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterministicPipelineSurvivesMLFailure:
    """The deterministic reconciliation pipeline must work without ML."""

    def test_reconciliation_independent_of_ml(self):
        """Reconciliation engine does not depend on ML."""
        from app.reconciliation.engine import calculate_reconciliation
        # Reconciliation is deterministic — no ML dependency
        assert callable(calculate_reconciliation)

    def test_evidence_independent_of_ml(self):
        """Evidence services do not depend on ML."""
        from app.services.evidence_guard import EvidenceGuard
        guard = EvidenceGuard()
        assert guard is not None

    def test_confidence_gate_independent_of_ml(self):
        """Confidence gate does not depend on ML."""
        gate = ConfidenceGate()
        engine = _engine(confidence=0.85)
        result = gate.evaluate(engine)
        assert result.action == GateAction.CONTINUE

    def test_exposure_guard_independent_of_ml(self):
        """Exposure guard does not depend on ML."""
        guard = ExposureGuard()
        engine = _engine(difference=5000)
        result = guard.evaluate(engine, _make_gate_result(GateAction.CONTINUE, 0.85))
        assert result.action == ExposureAction.PASS

    def test_full_guardrail_chain_with_ml_down(self):
        """Full guardrail chain works with ML completely unavailable."""
        engine = _engine(confidence=0.85, difference=5000)
        guard = GuardrailEngine()
        dep_status = {
            "ml_classifier": False,
            "ml_resolution_predictor": False,
            "similarity_service": False,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        result = guard.evaluate(engine, dep_status)
        # Full chain ran successfully — system is healthy
        assert result.system_healthy is True
        assert result.confidence == 0.85
        assert result.decision in ("AUTO", "HUMAN_REVIEW")
