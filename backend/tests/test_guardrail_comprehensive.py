"""
Comprehensive guardrail and confidence gate integration tests.

Tests the complete GuardrailEngine pipeline (Phase 6) end-to-end:
- Confidence Gate (6A) + Exposure Guard (6B) + Evidence Guard (6C) +
  Fallback Guard (6D) + Decision Matrix (6E) = GuardrailEngineResult

Focus areas:
- AUTO cannot happen when ANY mandatory safety condition fails
- Fail-closed behavior
- Configured thresholds tested against actual values (not hardcoded assumptions)
- Integer paise financial exposure
- Every safety condition tested independently
"""

import pytest

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.confidence_gate import ConfidenceGateConfig
from app.schemas.decision_matrix import AutomationDecision, DecisionConfig
from app.schemas.evidence_guard import EvidenceGuardConfig
from app.schemas.exposure_guard import ExposureGuardConfig
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.confidence_gate import ConfidenceGate
from app.services.decision_matrix import AutomationDecisionMatrix
from app.services.evidence_guard import EvidenceGuard
from app.services.exposure_guard import ExposureGuard
from app.services.fallback_guard import FallbackGuard
from app.services.guardrail_engine import GuardrailEngine


# ============================================================================
# HELPERS
# ============================================================================

def _engine_result(
    exception_id="EXC-001",
    status=SelectionStatus.RECOMMENDED,
    confidence=0.85,
    risk_category="LOW",
    evidence_coverage=0.85,
    evidence_consistency=0.90,
    difference=5000,
    adjustment_paise=5000,
    resolution_type="FEE_ADJUSTMENT",
    exception_type="FEE_DIFFERENCE",
    has_evidence_ids=True,
    is_novel=False,
    has_conflict=False,
    explanation_status="FULLY_EXPLAINED",
    selected_score=None,
):
    """Build a ResolutionEngineResult fixture."""
    candidate = None
    score = selected_score

    if status == SelectionStatus.RECOMMENDED:
        evidence_ids = ["EVD-001", "EVD-002"] if has_evidence_ids else []
        candidate = ResolutionProposal(
            candidate_id=f"CAND-{exception_id}-DET",
            exception_id=exception_id,
            case_id="CASE-001",
            resolution_type=resolution_type,
            resolution_description=f"Apply {resolution_type}",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION" if "FEE" in resolution_type else "SETTLEMENT_ADJUSTMENT",
                amount_paise=adjustment_paise,
                direction="CREDIT" if difference > 0 else "DEBIT",
                calculation_basis="evidence_trace",
            ),
            supporting_evidence_ids=evidence_ids,
            evidence_compatible=True,
            evidence_coverage=evidence_coverage,
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(rank=1, confidence_score=confidence, evidence_support=evidence_coverage),
            rationale=f"Resolution: {resolution_type}",
        )

        if score is None:
            score = CandidateScore(
                evidence_score=evidence_coverage,
                ml_score=0.0,
                historical_score=0.8 if not is_novel else 0.0,
                financial_consistency_score=1.0 if adjustment_paise == abs(difference) else 0.5,
                novelty_penalty=0.0 if not is_novel else 0.12,
                conflict_penalty=0.0 if not has_conflict else 0.075,
                final_score=confidence,
                weighted_evidence=evidence_coverage * 0.35,
                weighted_ml=0.0,
                weighted_historical=0.8 * 0.15,
                weighted_financial=1.0 * 0.30,
                has_evidence_support=len(evidence_ids) > 0,
                has_ml_support=False,
                has_historical_support=not is_novel,
                is_novel=is_novel,
                has_conflicts=has_conflict,
            )

    return ResolutionEngineResult(
        exception_id=exception_id,
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=100000 - difference,
        difference=difference,
        status=status,
        selected_resolution=resolution_type if status == SelectionStatus.RECOMMENDED else None,
        selected_candidate=candidate,
        selected_score=score,
        ranked_candidates=[candidate] if candidate else [],
        candidate_scores=[score] if score else [],
        confidence=confidence,
        risk_category=risk_category,
        deterministic_exception_type=exception_type,
        evidence_explanation_status=explanation_status,
        evidence_coverage=evidence_coverage,
        evidence_consistency=evidence_consistency,
        # HIGH #8: Explicitly set verified-safe values (not None/unknown)
        has_conflict=has_conflict,
        is_novel=is_novel,
    )


def _dep_status(**overrides):
    """Build dependency status with optional overrides."""
    status = {
        "ml_classifier": True,
        "ml_resolution_predictor": True,
        "similarity_service": True,
        "database": True,
        "evidence_retrieval": True,
        "llm": True,
        "mcp": True,
    }
    status.update(overrides)
    return status


# ============================================================================
# 1. CONFIDENCE THRESHOLD TESTS
# ============================================================================

class TestConfidenceThresholds:
    """Test the confidence gate with various confidence levels."""

    def test_confidence_above_threshold(self):
        """Confidence above min_confidence → gate passes."""
        gate = ConfidenceGate()
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.95)
        gate_result = gate.evaluate(result)
        assert gate_result.passed is True
        assert gate_result.action.value == "CONTINUE"

    def test_confidence_exactly_at_threshold(self):
        """Confidence exactly at min_confidence → gate passes."""
        config = ConfidenceGateConfig(min_confidence=0.70)
        gate = ConfidenceGate(config=config)
        result = _engine_result(confidence=0.70, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.95)
        gate_result = gate.evaluate(result)
        assert gate_result.passed is True

    def test_confidence_below_threshold(self):
        """Confidence below min_confidence → gate blocks."""
        gate = ConfidenceGate()
        result = _engine_result(confidence=0.50, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.95)
        gate_result = gate.evaluate(result)
        assert gate_result.passed is False
        assert gate_result.action.value == "HUMAN_REVIEW"

    def test_very_low_confidence_triggers_unresolved(self):
        """Very low confidence (< min_confidence_for_human) → UNRESOLVED via decision matrix."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.10, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.95)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED

    def test_configured_threshold_used(self):
        """Verify the gate uses the configured threshold, not a hardcoded value."""
        config = ConfidenceGateConfig(min_confidence=0.60)
        gate = ConfidenceGate(config=config)
        # At 0.59 should fail, at 0.60 should pass
        result_below = _engine_result(confidence=0.59, risk_category="LOW",
                                       evidence_coverage=0.9, evidence_consistency=0.95)
        result_above = _engine_result(confidence=0.60, risk_category="LOW",
                                       evidence_coverage=0.9, evidence_consistency=0.95)
        assert gate.evaluate(result_below).passed is False
        assert gate.evaluate(result_above).passed is True


# ============================================================================
# 2. RISK LEVEL TESTS
# ============================================================================

class TestRiskLevels:
    """Test risk-level blocking in the decision matrix."""

    def test_low_risk_can_auto(self):
        """LOW risk → can potentially AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.AUTO

    def test_high_risk_blocks_auto(self):
        """HIGH risk → cannot AUTO (decision matrix blocks)."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="HIGH",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        # HIGH risk should not be AUTO
        assert gr.decision != AutomationDecision.AUTO

    def test_medium_risk_blocks_auto(self):
        """MEDIUM risk → cannot AUTO (only LOW allowed for AUTO)."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="MEDIUM",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_risk_is_evaluated_independent_of_confidence(self):
        """High confidence + HIGH risk → still not AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.99, risk_category="HIGH",
                                evidence_coverage=0.99, evidence_consistency=0.99,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO


# ============================================================================
# 3. FINANCIAL EXPOSURE TESTS (INTEGER PAISE)
# ============================================================================

class TestFinancialExposure:
    """Test financial exposure guard using integer paise."""

    def test_low_exposure_allows_auto(self):
        """Low exposure (within max_auto_resolution) → exposure guard passes."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.financial_exposure_paise == 10000
        assert isinstance(gr.financial_exposure_paise, int)

    def test_high_exposure_blocks_auto(self):
        """Exposure above max_auto_resolution_paise → blocks."""
        engine = GuardrailEngine()
        # max_auto_resolution_paise default is 50000
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=60000, difference=60000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_very_high_exposure_triggers_unresolved(self):
        """Exposure above max_exposure_for_human → UNRESOLVED."""
        engine = GuardrailEngine()
        # max_exposure_for_human default is 100000
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=150000, difference=150000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED

    def test_exposure_boundary_at_limit(self):
        """Exposure exactly at max_auto_resolution_paise → passes exposure guard."""
        config = ExposureGuardConfig(max_auto_resolution_paise=50000)
        guard = ExposureGuard(config=config)
        result = _engine_result(adjustment_paise=50000, difference=50000,
                                confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95)
        exp_result = guard.evaluate(result)
        assert exp_result.passed is True

    def test_exposure_just_over_limit(self):
        """Exposure one paise over limit → blocks."""
        config = ExposureGuardConfig(max_auto_resolution_paise=50000)
        guard = ExposureGuard(config=config)
        result = _engine_result(adjustment_paise=50001, difference=50001,
                                confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95)
        exp_result = guard.evaluate(result)
        assert exp_result.passed is False

    def test_exposure_is_integer_paise(self):
        """Financial exposure is always stored as integer paise."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=25000, difference=25000)
        gr = engine.evaluate(result, _dep_status())
        assert isinstance(gr.financial_exposure_paise, int)
        assert gr.financial_exposure_paise == 25000

    def test_zero_exposure_no_action(self):
        """Zero adjustment → exposure guard passes."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=0, difference=0,
                                resolution_type="NO_ACTION", exception_type="EXACT_MATCH")
        gr = engine.evaluate(result, _dep_status())
        assert gr.financial_exposure_paise == 0


# ============================================================================
# 4. EVIDENCE SAFETY TESTS
# ============================================================================

class TestEvidenceSafety:
    """Test evidence guard with various evidence conditions."""

    def test_missing_evidence_blocks_auto(self):
        """No supporting evidence → evidence guard blocks."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.9,
                                has_evidence_ids=False, adjustment_paise=5000,
                                difference=5000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_low_evidence_coverage_blocks_auto(self):
        """Evidence coverage below min_evidence_coverage → blocks."""
        engine = GuardrailEngine()
        # min_evidence_coverage default is 0.50
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.30, evidence_consistency=0.90,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_low_evidence_consistency_blocks_auto(self):
        """Evidence consistency below min → blocks."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.30,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_good_evidence_passes(self):
        """Good coverage + consistency → evidence guard passes."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        # With good evidence + low risk + high confidence, should be AUTO
        assert gr.decision == AutomationDecision.AUTO


# ============================================================================
# 5. CONFLICT AND NOVELTY TESTS
# ============================================================================

class TestConflictAndNovelty:
    """Test conflict and novelty blocking.

    NOTE: The evidence guard extracts has_conflict and is_novel from its own
    checks (not from engine_result). The engine_result flags are not yet
    propagated through the guardrail pipeline. This means the decision matrix
    sees conflict/novelty = False unless a low-confidence penalty triggers
    indirectly. This is a known design gap documented in the audit.
    """

    def test_conflict_penalty_in_candidate_blocks_auto(self):
        """High conflict penalty in selected_score → confidence gate blocks."""
        engine = GuardrailEngine()
        score = CandidateScore(
            evidence_score=0.9, ml_score=0.0, historical_score=0.8,
            financial_consistency_score=1.0, novelty_penalty=0.0,
            conflict_penalty=0.15,  # Exceeds max_conflict_penalty of 0.10
            final_score=0.85, weighted_evidence=0.315, weighted_ml=0.0,
            weighted_historical=0.12, weighted_financial=0.30,
            has_evidence_support=True, has_ml_support=False,
            has_historical_support=True, is_novel=False, has_conflicts=True,
        )
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.90,
                                adjustment_paise=5000, difference=5000,
                                selected_score=score)
        gr = engine.evaluate(result, _dep_status())
        # High conflict penalty triggers confidence gate → not AUTO
        assert gr.decision != AutomationDecision.AUTO

    def test_novelty_penalty_in_candidate_blocks_auto(self):
        """High novelty penalty in selected_score → confidence gate blocks."""
        engine = GuardrailEngine()
        score = CandidateScore(
            evidence_score=0.9, ml_score=0.0, historical_score=0.0,
            financial_consistency_score=1.0,
            novelty_penalty=0.15,  # Exceeds max_novelty_penalty of 0.10
            conflict_penalty=0.0, final_score=0.75,
            weighted_evidence=0.315, weighted_ml=0.0,
            weighted_historical=0.0, weighted_financial=0.30,
            has_evidence_support=True, has_ml_support=False,
            has_historical_support=False, is_novel=True, has_conflicts=False,
        )
        result = _engine_result(confidence=0.75, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.90,
                                adjustment_paise=5000, difference=5000,
                                selected_score=score)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_no_conflict_no_novelty_allows_auto(self):
        """No conflict + no novelty → can AUTO (with other conditions met)."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000,
                                has_conflict=False, is_novel=False)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.AUTO


# ============================================================================
# 6. DEPENDENCY FAILURE / FAIL-CLOSED TESTS
# ============================================================================

class TestDependencyFailures:
    """Test fail-closed behavior for dependency failures."""

    def test_database_failure_fails_closed(self):
        """Database down → cannot AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status(database=False))
        assert gr.decision != AutomationDecision.AUTO
        assert gr.system_healthy is False
        assert "database" in gr.critical_failures

    def test_mcp_failure_fails_closed(self):
        """MCP down → cannot AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status(mcp=False))
        assert gr.decision != AutomationDecision.AUTO
        assert "mcp" in gr.critical_failures

    def test_evidence_retrieval_failure_fails_closed(self):
        """Evidence retrieval down → cannot AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status(evidence_retrieval=False))
        assert gr.decision != AutomationDecision.AUTO

    def test_ml_unavailable_does_not_block_auto(self):
        """ML unavailable → optional dep, deterministic pipeline continues."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status(ml_classifier=False, ml_resolution_predictor=False))
        # ML is optional — should still allow AUTO with good signals
        assert gr.decision == AutomationDecision.AUTO

    def test_llm_unavailable_does_not_block_auto(self):
        """LLM unavailable → optional dep, pipeline continues."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status(llm=False))
        assert gr.decision == AutomationDecision.AUTO

    def test_all_deps_healthy_allows_auto(self):
        """All dependencies healthy → system_healthy=True."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.system_healthy is True
        assert len(gr.critical_failures) == 0


# ============================================================================
# 7. UNRESOLVED / HUMAN_REVIEW STATUS TESTS
# ============================================================================

class TestEngineStatus:
    """Test how engine status affects guardrail decisions."""

    def test_engine_unresolved_always_unresolved(self):
        """Engine status UNRESOLVED → decision UNRESOLVED regardless of confidence."""
        engine = GuardrailEngine()
        result = _engine_result(status=SelectionStatus.UNRESOLVED, confidence=0.95,
                                risk_category="LOW", evidence_coverage=0.95,
                                evidence_consistency=0.95)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED

    def test_engine_human_review_not_auto(self):
        """Engine status HUMAN_REVIEW → decision not AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(status=SelectionStatus.HUMAN_REVIEW, confidence=0.95,
                                risk_category="LOW", evidence_coverage=0.95,
                                evidence_consistency=0.95)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO


# ============================================================================
# 8. UNKNOWN EXCEPTION TYPE TESTS
# ============================================================================

class TestUnknownException:
    """Test UNKNOWN exception type handling."""

    def test_unknown_exception_unresolved(self):
        """UNKNOWN exception type → UNRESOLVED."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                exception_type="UNKNOWN")
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED

    def test_complex_multi_adjustment_unresolved(self):
        """COMPLEX_MULTI_ADJUSTMENT → UNRESOLVED."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                exception_type="COMPLEX_MULTI_ADJUSTMENT")
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED

    def test_missing_record_unresolved(self):
        """MISSING_RECORD → UNRESOLVED."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                exception_type="MISSING_RECORD")
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED


# ============================================================================
# 9. BLOCKED RESOLUTION TYPE TESTS
# ============================================================================

class TestBlockedResolutionTypes:
    """Test that blocked resolution types cannot AUTO."""

    def test_unknown_unresolved_resolution_blocked(self):
        """UNKNOWN_UNRESOLVED resolution → not AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                resolution_type="UNKNOWN_UNRESOLVED",
                                exception_type="UNKNOWN")
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_missing_record_escalation_blocked(self):
        """MISSING_RECORD_ESCALATION resolution → not AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                resolution_type="MISSING_RECORD_ESCALATION",
                                exception_type="MISSING_RECORD")
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO


# ============================================================================
# 10. AUTO BLOCKING — CORE SAFETY GUARANTEE
# ============================================================================

class TestAutoBlocking:
    """
    CORE SAFETY GUARANTEE: AUTO cannot happen when ANY mandatory
    safety condition fails.

    NOTE: has_conflict/is_novel flags on engine_result are not yet propagated
    through the evidence guard pipeline (documented design gap). Conflict and
    novelty blocking is tested via CandidateScore penalties instead.
    """

    def _assert_not_auto(self, desc, **kwargs):
        """Helper to assert a configuration does NOT produce AUTO."""
        engine = GuardrailEngine()
        result = _engine_result(**kwargs)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO, (
            f"Expected NOT AUTO for: {desc}, got {gr.decision.value}"
        )

    def test_auto_blocked_by_low_confidence(self):
        self._assert_not_auto("low confidence", confidence=0.50, risk_category="LOW",
                               evidence_coverage=0.9, evidence_consistency=0.95)

    def test_auto_blocked_by_high_risk(self):
        self._assert_not_auto("high risk", confidence=0.95, risk_category="HIGH",
                               evidence_coverage=0.95, evidence_consistency=0.95,
                               adjustment_paise=5000, difference=5000)

    def test_auto_blocked_by_high_exposure(self):
        self._assert_not_auto("high exposure", confidence=0.95, risk_category="LOW",
                               evidence_coverage=0.95, evidence_consistency=0.95,
                               adjustment_paise=60000, difference=60000)

    def test_auto_blocked_by_missing_evidence(self):
        self._assert_not_auto("missing evidence", confidence=0.95, risk_category="LOW",
                               evidence_coverage=0.9, evidence_consistency=0.9,
                               has_evidence_ids=False, adjustment_paise=5000,
                               difference=5000)

    def test_auto_blocked_by_low_coverage(self):
        self._assert_not_auto("low coverage", confidence=0.95, risk_category="LOW",
                               evidence_coverage=0.20, evidence_consistency=0.95,
                               adjustment_paise=5000, difference=5000)

    def test_auto_blocked_by_low_consistency(self):
        self._assert_not_auto("low consistency", confidence=0.95, risk_category="LOW",
                               evidence_coverage=0.95, evidence_consistency=0.20,
                               adjustment_paise=5000, difference=5000)

    def test_auto_blocked_by_high_conflict_penalty(self):
        """High conflict penalty in CandidateScore → confidence gate blocks."""
        score = CandidateScore(
            evidence_score=0.95, ml_score=0.0, historical_score=0.8,
            financial_consistency_score=1.0, novelty_penalty=0.0,
            conflict_penalty=0.15,  # > max_conflict_penalty (0.10)
            final_score=0.95, weighted_evidence=0.33, weighted_ml=0.0,
            weighted_historical=0.12, weighted_financial=0.30,
            has_evidence_support=True, has_ml_support=False,
            has_historical_support=True, is_novel=False, has_conflicts=True,
        )
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.95, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                selected_score=score)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_auto_blocked_by_high_novelty_penalty(self):
        """High novelty penalty in CandidateScore → confidence gate blocks."""
        score = CandidateScore(
            evidence_score=0.95, ml_score=0.0, historical_score=0.0,
            financial_consistency_score=1.0, novelty_penalty=0.15,  # > max_novelty_penalty (0.10)
            conflict_penalty=0.0, final_score=0.85,
            weighted_evidence=0.33, weighted_ml=0.0,
            weighted_historical=0.0, weighted_financial=0.30,
            has_evidence_support=True, has_ml_support=False,
            has_historical_support=False, is_novel=True, has_conflicts=False,
        )
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000,
                                selected_score=score)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision != AutomationDecision.AUTO

    def test_auto_blocked_by_unknown_exception(self):
        self._assert_not_auto("unknown exception", confidence=0.95, risk_category="LOW",
                               evidence_coverage=0.95, evidence_consistency=0.95,
                               adjustment_paise=5000, difference=5000,
                               exception_type="UNKNOWN")

    def test_auto_blocked_by_db_down(self):
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.95, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        gr = engine.evaluate(result, _dep_status(database=False))
        assert gr.decision != AutomationDecision.AUTO

    def test_auto_blocked_by_engine_unresolved(self):
        self._assert_not_auto("engine unresolved",
                               status=SelectionStatus.UNRESOLVED, confidence=0.99,
                               risk_category="LOW", evidence_coverage=0.99,
                               evidence_consistency=0.99)

    def test_auto_blocked_by_medium_risk(self):
        self._assert_not_auto("medium risk", confidence=0.95, risk_category="MEDIUM",
                               evidence_coverage=0.95, evidence_consistency=0.95,
                               adjustment_paise=10000, difference=10000)


# ============================================================================
# 11. FAIL-CLOSED ON INTERNAL ERROR
# ============================================================================

class TestFailClosedOnInternalError:
    """Test that the guardrail engine fails closed on internal errors."""

    def test_fail_closed_on_error(self):
        """Internal error → UNRESOLVED, never AUTO."""
        engine = GuardrailEngine()
        # Create a broken engine result that causes an error
        result = _engine_result(confidence=0.95, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        # Corrupt the engine to force an error
        original_gate = engine.confidence_gate
        engine.confidence_gate = None  # This will cause an AttributeError
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.UNRESOLVED
        assert "guardrail_engine" in gr.critical_failures
        # Restore
        engine.confidence_gate = original_gate


# ============================================================================
# 12. GATE AUDIT TRAIL
# ============================================================================

class TestGateAuditTrail:
    """Verify the guardrail engine produces a full audit trail."""

    def test_audit_trail_has_all_guard_results(self):
        """GuardrailEngineResult contains results from all 5 guards."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.90,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.confidence_gate_result is not None
        assert gr.exposure_guard_result is not None
        assert gr.evidence_guard_result is not None
        assert gr.fallback_result is not None
        assert gr.decision_result is not None

    def test_audit_trail_has_gates(self):
        """Decision result contains passed/failed gates."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.90, evidence_consistency=0.90,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert len(gr.passed_gates) > 0
        assert isinstance(gr.passed_gates[0].gate_name, str)

    def test_auto_result_has_all_gates_passed(self):
        """When AUTO, reason includes ALL_GATES_PASSED."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.AUTO
        assert any("ALL_GATES_PASSED" in str(rc) for rc in gr.reason_codes)

    def test_is_recommendation_only_always_true(self):
        """GuardrailEngineResult is always a recommendation, never execution."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.90, risk_category="LOW",
                                evidence_coverage=0.95, evidence_consistency=0.95,
                                adjustment_paise=10000, difference=10000)
        gr = engine.evaluate(result, _dep_status())
        assert gr.is_recommendation_only is True


# ============================================================================
# 13. INTEGRATION: AUTO ONLY WITH ALL CONDITIONS MET
# ============================================================================

class TestAutoOnlyWithAllConditions:
    """
    Verify that AUTO is returned ONLY when all safety conditions are met.
    """

    def test_auto_requires_all_conditions(self):
        """AUTO only when: high confidence + LOW risk + low exposure + good evidence + no conflict + no novelty + healthy deps."""
        engine = GuardrailEngine()
        result = _engine_result(
            confidence=0.90,
            risk_category="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            adjustment_paise=10000,
            difference=10000,
            has_conflict=False,
            is_novel=False,
            exception_type="FEE_DIFFERENCE",
            resolution_type="FEE_ADJUSTMENT",
        )
        gr = engine.evaluate(result, _dep_status())
        assert gr.decision == AutomationDecision.AUTO

    def test_removing_any_condition_stops_auto(self):
        """Removing any single good condition should stop AUTO."""
        engine = GuardrailEngine()
        base = dict(
            confidence=0.90,
            risk_category="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            adjustment_paise=10000,
            difference=10000,
            has_conflict=False,
            is_novel=False,
            exception_type="FEE_DIFFERENCE",
            resolution_type="FEE_ADJUSTMENT",
        )

        # Verify base produces AUTO
        gr = engine.evaluate(_engine_result(**base), _dep_status())
        assert gr.decision == AutomationDecision.AUTO

        # Each modification should stop AUTO
        # Note: has_conflict/is_novel flags on engine_result are not propagated
        # through the evidence guard pipeline (documented gap). Test via penalties instead.
        modifications = {
            "confidence": 0.50,
            "risk_category": "HIGH",
            "evidence_coverage": 0.20,
            "evidence_consistency": 0.20,
        }

        for field, value in modifications.items():
            modified = {**base, field: value}
            gr = engine.evaluate(_engine_result(**modified), _dep_status())
            assert gr.decision != AutomationDecision.AUTO, (
                f"Modifying {field}={value} should block AUTO, got {gr.decision.value}"
            )


# ============================================================================
# 14. DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Guardrail evaluation must be deterministic."""

    def test_same_input_same_output(self):
        """Same engine result + same deps → same decision."""
        engine = GuardrailEngine()
        result = _engine_result(confidence=0.85, risk_category="LOW",
                                evidence_coverage=0.9, evidence_consistency=0.95,
                                adjustment_paise=5000, difference=5000)
        deps = _dep_status()
        gr1 = engine.evaluate(result, deps)
        gr2 = engine.evaluate(result, deps)
        assert gr1.decision == gr2.decision
        assert gr1.confidence == gr2.confidence
        assert gr1.financial_exposure_paise == gr2.financial_exposure_paise
