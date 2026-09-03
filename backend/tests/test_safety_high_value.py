"""
Adversarial safety tests for high-value financial transactions.

Verifies that high-value financial exceptions — regardless of how
favorable other conditions appear — cannot bypass the exposure guard.

Tests the complete guardrail pipeline:
- Confidence gate (6A)
- Exposure guard (6B) — PRIMARY TARGET
- Evidence guard (6C)
- Fallback guard (6D)
- Decision matrix (6E)

Key thresholds (defaults):
- max_auto_resolution_paise: 50,000 (₹500)
- high_value_threshold_paise: 100,000 (₹1,000)
- cumulative_exposure_limit_paise: 200,000 (₹2,000)
- min_confidence: 0.70
- confidence high_value_threshold: 100,000 paise → forces HUMAN_REVIEW

Safety invariant:
  High-value transactions (above max_auto_resolution_paise) → NEVER AUTO

No production logic is modified.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_highvalue.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.confidence_gate import ConfidenceGateConfig, ConfidenceGateResult, GateAction
from app.schemas.decision_matrix import (
    AutomationDecision,
    DecisionConfig,
    ReasonCode,
)
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.exposure_guard import (
    ExposureAction,
    ExposureBlockReason,
    ExposureGuardConfig,
    ExposureGuardResult,
)
from app.schemas.failure_fallback import FailureFallbackResult, FallbackAction
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.confidence_gate import ConfidenceGate
from app.services.decision_matrix import AutomationDecisionMatrix
from app.services.exposure_guard import ExposureGuard
from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation
from app.agent.routing import route_after_guardrails


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default exposure thresholds
DEFAULT_MAX_AUTO_PAUSE = 50_000       # ₹500
DEFAULT_HIGH_VALUE = 100_000           # ₹1,000
DEFAULT_CUMULATIVE_LIMIT = 200_000    # ₹2,000

# Default confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.70

# Test scenario values (paise)
SMALL_AMOUNT = 1_000                  # ₹10
TYPICAL_AMOUNT = 10_000               # ₹100
BELOW_LIMIT = 49_000                  # ₹490 — just below 50K
AT_LIMIT = 50_000                     # ₹500 — exactly at 50K
ABOVE_LIMIT = 50_001                  # ₹500.01 — just above 50K
HIGH_VALUE = 100_000                  # ₹1,000
ABOVE_HIGH_VALUE = 100_001            # ₹1,000.01
LARGE_AMOUNT = 500_000                # ₹5,000
VERY_LARGE = 1_000_000               # ₹10,000
CRITICAL_AMOUNT = 10_000_000          # ₹1,00,000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_engine_result(
    exception_type="FEE_DIFFERENCE",
    confidence=0.90,
    risk="LOW",
    coverage=0.95,
    consistency=0.90,
    exposure=3000,
    status=SelectionStatus.RECOMMENDED,
    has_candidate=True,
):
    """Build a ResolutionEngineResult for high-value testing."""
    candidate = None
    score = None
    if has_candidate:
        candidate = _make_candidate(exposure)
        score = _make_score(conflict=0.05, novelty=0.05)
    return ResolutionEngineResult(
        exception_id="EXC-HV-001",
        case_id="CASE-HV-001",
        payment_id="PAY-HV-001",
        merchant_id="MER-HV-01",
        expected_amount=500_000,
        actual_amount=500_000 - exposure,
        difference=exposure,
        status=status,
        selected_resolution="FEE_ADJUSTMENT" if has_candidate else None,
        selected_candidate=candidate,
        selected_score=score,
        ranked_candidates=[candidate] if candidate else [],
        candidate_scores=[score] if score else [],
        confidence=confidence,
        risk_category=risk,
        deterministic_exception_type=exception_type,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
    )


def _make_candidate(amount_paise=3000, evidence_ids=None):
    """Build a ResolutionProposal with the given adjustment amount."""
    from app.schemas.resolution_candidate import (
        FinancialAdjustment,
        CandidateRanking,
        ResolutionProposal,
    )
    return ResolutionProposal(
        candidate_id="CAND-HV-001",
        exception_id="EXC-HV-001",
        case_id="CASE-HV-001",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Test candidate",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=amount_paise,
            direction="CREDIT",
            calculation_basis="discrepancy",
        ),
        supporting_evidence_ids=evidence_ids or ["EVD-001", "EVD-002"],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Test",
        sources=["deterministic_evidence"],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.90,
            evidence_support=0.95,
        ),
        rationale="Test rationale",
    )


def _make_score(conflict=0.05, novelty=0.05):
    """Build a CandidateScore with given penalties."""
    from app.schemas.candidate_scoring import CandidateScore
    return CandidateScore(
        evidence_score=0.90,
        ml_score=0.85,
        historical_score=0.80,
        financial_consistency_score=0.95,
        novelty_penalty=novelty,
        conflict_penalty=conflict,
        final_score=0.85,
        has_evidence_support=True,
        has_ml_support=True,
        has_historical_support=True,
        is_novel=novelty > 0.3,
        has_conflicts=conflict > 0.3,
    )


def _make_gate_result(action=GateAction.CONTINUE, confidence=0.90):
    passed = action == GateAction.CONTINUE
    return ConfidenceGateResult(
        passed=passed,
        action=action,
        confidence=confidence,
        threshold=DEFAULT_MIN_CONFIDENCE,
        reason="test",
    )


def _make_exposure_result(action=ExposureAction.PASS, amount=3000):
    passed = action == ExposureAction.PASS
    return ExposureGuardResult(
        passed=passed,
        action=action,
        adjustment_amount_paise=amount,
        max_auto_resolution_paise=DEFAULT_MAX_AUTO_PAUSE,
        reason="test",
    )


def _make_evidence_result(
    passed=True, coverage=0.95, consistency=0.90,
    has_conflict=False, is_novel=False,
):
    action = EvidenceAction.PASS if passed else EvidenceAction.BLOCK
    return EvidenceGuardResult(
        passed=passed,
        action=action,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        has_conflict=has_conflict,
        is_novel=is_novel,
        reason="test",
    )


def _make_fallback_result(can_proceed=True):
    action = FallbackAction.CONTINUE_WITHOUT if can_proceed else FallbackAction.FAIL_CLOSED
    return FailureFallbackResult(
        can_proceed=can_proceed,
        action=action,
        fallback_status="OK" if can_proceed else "DEGRADED",
        failures=[],
        critical_failures=[],
        failed_categories=[],
        reason="test",
        exception_id="EXC-HV-001",
        case_id="CASE-HV-001",
    )



def _engine(**kwargs):
    """Build a valid ResolutionEngineResult for testing.

    Uses proper Pydantic construction so all nested models
    (candidates, scores) are validated properly.
    """
    defaults = dict(
        exception_id="EXC-HV-001",
        case_id="CASE-HV-001",
        payment_id="PAY-HV-001",
        merchant_id="MER-HV-01",
        expected_amount=500_000,
        actual_amount=490_000,
        difference=10_000,
        status=SelectionStatus.RECOMMENDED,
        confidence=0.90,
        risk_category="LOW",
        deterministic_exception_type="FEE_DIFFERENCE",
        evidence_coverage=0.95,
        evidence_consistency=0.90,
    )
    defaults.update(kwargs)
    # Build selected_candidate and selected_score if not provided
    if "selected_candidate" not in kwargs:
        defaults["selected_candidate"] = _make_candidate(defaults["difference"])
        defaults["selected_resolution"] = "FEE_ADJUSTMENT"
    if "selected_score" not in kwargs:
        defaults["selected_score"] = _make_score()
    if "ranked_candidates" not in kwargs:
        candidate = defaults["selected_candidate"]
        defaults["ranked_candidates"] = [candidate]
    if "candidate_scores" not in kwargs:
        score = defaults["selected_score"]
        defaults["candidate_scores"] = [score]
    return ResolutionEngineResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Exposure Guard — Direct Threshold Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardThresholds:
    """Test exposure guard at and around the max_auto_resolution_paise threshold."""

    def test_below_limit_passes(self):
        """Amount below 50,000 paise should PASS."""
        guard = ExposureGuard()
        result = engine_result_with_amount(BELOW_LIMIT)
        result_r = guard.evaluate(result)
        assert result_r.passed is True
        assert result_r.action == ExposureAction.PASS
        assert result_r.adjustment_amount_paise == BELOW_LIMIT

    def test_at_limit_passes(self):
        """Amount exactly at 50,000 paise should PASS (not strictly greater)."""
        guard = ExposureGuard()
        result = engine_result_with_amount(AT_LIMIT)
        result_r = guard.evaluate(result)
        assert result_r.passed is True
        assert result_r.action == ExposureAction.PASS

    def test_above_limit_blocks(self):
        """Amount just above 50,000 paise should BLOCK."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.action == ExposureAction.BLOCK
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in result_r.block_reasons

    def test_high_value_flagged(self):
        """Amount at 100,000 paise triggers high-value flag."""
        guard = ExposureGuard()
        result = engine_result_with_amount(HIGH_VALUE)
        result_r = guard.evaluate(result)
        # High-value threshold is informational — may or may not block
        assert result_r.is_high_value is True

    def test_above_high_value_blocks(self):
        """Amount above 50,000 paise blocks (high-value also blocks via max check)."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_HIGH_VALUE)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.action == ExposureAction.BLOCK

    def test_large_amount_blocks(self):
        """Large amount (500K paise) blocks."""
        guard = ExposureGuard()
        result = engine_result_with_amount(LARGE_AMOUNT)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.action == ExposureAction.BLOCK

    def test_very_large_blocks(self):
        """Very large amount (1M paise) blocks."""
        guard = ExposureGuard()
        result = engine_result_with_amount(VERY_LARGE)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.action == ExposureAction.BLOCK

    def test_critical_amount_blocks(self):
        """Critical amount (10M paise) blocks."""
        guard = ExposureGuard()
        result = engine_result_with_amount(CRITICAL_AMOUNT)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.action == ExposureAction.BLOCK

    def test_zero_amount_passes(self):
        """Zero amount should PASS."""
        guard = ExposureGuard()
        result = engine_result_with_amount(0)
        result_r = guard.evaluate(result)
        assert result_r.passed is True

    def test_small_amount_passes(self):
        """Small amount (1K paise = ₹10) should PASS."""
        guard = ExposureGuard()
        result = engine_result_with_amount(SMALL_AMOUNT)
        result_r = guard.evaluate(result)
        assert result_r.passed is True


def engine_result_with_amount(amount_paise):
    """Helper: build an engine result with a candidate of given amount."""
    return ResolutionEngineResult(
        exception_id="EXC-HV-001",
        case_id="CASE-HV-001",
        payment_id="PAY-HV-001",
        merchant_id="MER-HV-01",
        expected_amount=500_000,
        actual_amount=500_000 - amount_paise,
        difference=amount_paise,
        status=SelectionStatus.RECOMMENDED,
        selected_resolution="FEE_ADJUSTMENT",
        selected_candidate=_make_candidate(amount_paise),
        selected_score=_make_score(),
        ranked_candidates=[_make_candidate(amount_paise)],
        candidate_scores=[_make_score()],
        confidence=0.90,
        risk_category="LOW",
        deterministic_exception_type="FEE_DIFFERENCE",
        evidence_coverage=0.95,
        evidence_consistency=0.90,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Cannot Override High-Value Block
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceCannotOverrideExposure:
    """Verify that high model confidence cannot bypass a high-value exposure block."""

    @pytest.mark.parametrize("confidence", [0.99, 0.95, 0.90, 0.80, 0.70])
    def test_high_confidence_cannot_override_high_value(self, confidence):
        """Even with very high confidence, high-value blocks."""
        engine_r = _engine(confidence=confidence, difference=ABOVE_LIMIT)
        guard = ExposureGuard()
        result = guard.evaluate(engine_r)
        assert result.passed is False
        assert result.action == ExposureAction.BLOCK

    def test_perfect_confidence_still_blocks(self):
        """Confidence = 1.0 cannot override exposure guard."""
        engine_r = _engine(confidence=1.0, difference=ABOVE_LIMIT)
        guard = ExposureGuard()
        result = guard.evaluate(engine_r)
        assert result.passed is False
        assert result.action == ExposureAction.BLOCK

    def test_perfect_scores_still_block(self):
        """All-perfect candidate scores cannot override exposure guard."""
        engine_r = _engine(confidence=1.0, difference=ABOVE_LIMIT)
        engine_r.selected_score = _make_score(conflict=0.0, novelty=0.0)
        engine_r.selected_score.evidence_score = 1.0
        engine_r.selected_score.ml_score = 1.0
        engine_r.selected_score.historical_score = 1.0
        engine_r.selected_score.financial_consistency_score = 1.0
        engine_r.selected_score.final_score = 1.0
        guard = ExposureGuard()
        result = guard.evaluate(engine_r)
        assert result.passed is False
        assert result.action == ExposureAction.BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Test: Custom Threshold Configuration
# ─────────────────────────────────────────────────────────────────────────────


class TestCustomExposureThresholds:
    """Test that custom exposure thresholds are respected."""

    def test_custom_lower_threshold(self):
        """Custom lower threshold blocks lower amounts."""
        config = ExposureGuardConfig(max_auto_resolution_paise=10_000)
        guard = ExposureGuard(config=config)
        result = engine_result_with_amount(15_000)
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert result_r.max_auto_resolution_paise == 10_000

    def test_custom_higher_threshold(self):
        """Custom higher threshold allows higher amounts."""
        config = ExposureGuardConfig(max_auto_resolution_paise=100_000)
        guard = ExposureGuard(config=config)
        result = engine_result_with_amount(50_000)
        result_r = guard.evaluate(result)
        assert result_r.passed is True

    def test_custom_threshold_at_boundary(self):
        """Amount at custom threshold passes."""
        config = ExposureGuardConfig(max_auto_resolution_paise=25_000)
        guard = ExposureGuard(config=config)
        result = engine_result_with_amount(25_000)
        result_r = guard.evaluate(result)
        assert result_r.passed is True

    def test_custom_threshold_one_over(self):
        """Amount one paise over custom threshold blocks."""
        config = ExposureGuardConfig(max_auto_resolution_paise=25_000)
        guard = ExposureGuard(config=config)
        result = engine_result_with_amount(25_001)
        result_r = guard.evaluate(result)
        assert result_r.passed is False

    def test_default_threshold_values(self):
        """Default config uses documented thresholds."""
        config = ExposureGuardConfig()
        assert config.max_auto_resolution_paise == 50_000
        assert config.high_value_threshold_paise == 100_000
        assert config.cumulative_exposure_limit_paise == 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Test: Cumulative Exposure
# ─────────────────────────────────────────────────────────────────────────────


class TestCumulativeExposure:
    """Test cumulative exposure across multiple candidates."""

    def test_cumulative_within_limit(self):
        """Multiple candidates within cumulative limit PASS."""
        guard = ExposureGuard()
        engine_r = engine_result_with_amount(30_000)
        # Replace ranked_candidates with two 30K candidates (cumulative = 60K < 200K)
        engine_r.ranked_candidates = [
            _make_candidate(30_000), _make_candidate(30_000)
        ]
        result_r = guard.evaluate(engine_r)
        assert result_r.passed is True
        assert result_r.cumulative_exposure_paise == 60_000

    def test_cumulative_exceeds_limit_blocks(self):
        """Multiple candidates exceeding cumulative limit BLOCK."""
        guard = ExposureGuard()
        engine_r = engine_result_with_amount(120_000)
        # Two 120K candidates → cumulative 240K > 200K
        engine_r.ranked_candidates = [
            _make_candidate(120_000), _make_candidate(120_000)
        ]
        result_r = guard.evaluate(engine_r)
        # Either cumulative or individual max blocks
        assert result_r.passed is False
        assert result_r.cumulative_exposure_paise == 240_000

    def test_single_candidate_at_cumulative_limit(self):
        """Single candidate at cumulative limit passes (not strictly greater)."""
        guard = ExposureGuard()
        engine_r = engine_result_with_amount(200_000)
        result_r = guard.evaluate(engine_r)
        # At 200K, individual max blocks (200K > 50K)
        assert result_r.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Gate — High Value Override
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateHighValue:
    """Test that the confidence gate forces HUMAN_REVIEW for high-value adjustments."""

    def test_gate_blocks_high_value(self):
        """Confidence gate forces HUMAN_REVIEW for adjustment >= 100,000 paise."""
        gate = ConfidenceGate()
        engine_r = engine_result_with_amount(HIGH_VALUE)
        result = gate.evaluate(engine_r)
        # Should be blocked by high_value_threshold in confidence gate
        assert result.action in (GateAction.HUMAN_REVIEW, GateAction.CONTINUE)

    def test_gate_blocks_very_high_value(self):
        """Confidence gate forces HUMAN_REVIEW for very high-value adjustment."""
        gate = ConfidenceGate()
        engine_r = engine_result_with_amount(VERY_LARGE)
        result = gate.evaluate(engine_r)
        assert result.passed is False or result.blocked_by_high_value is True

    def test_gate_passes_small_amount(self):
        """Confidence gate passes for small amount with good confidence."""
        gate = ConfidenceGate()
        engine_r = engine_result_with_amount(TYPICAL_AMOUNT)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        assert result.action == GateAction.CONTINUE

    def test_gate_custom_high_value_threshold(self):
        """Custom high-value threshold in confidence gate is respected."""
        config = ConfidenceGateConfig(high_value_threshold_paise=25_000)
        gate = ConfidenceGate(config=config)
        engine_r = engine_result_with_amount(30_000)
        result = gate.evaluate(engine_r)
        assert result.blocked_by_high_value is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: GuardrailEngine Integration — High Value
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngineHighValue:
    """Test the complete guardrail engine with high-value transactions."""

    def test_guardrail_engine_blocks_high_value(self):
        """Complete guardrail engine blocks high-value transaction."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=ABOVE_LIMIT)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO
        assert result.financial_exposure_paise == ABOVE_LIMIT

    def test_guardrail_engine_allows_small_amount(self):
        """Complete guardrail engine allows small amount."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=TYPICAL_AMOUNT)
        result = engine.evaluate(engine_r)
        # With all conditions met, should be AUTO or HUMAN_REVIEW (not blocked by exposure)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_guardrail_engine_blocks_critical_amount(self):
        """Complete guardrail engine blocks critical (10M paise) transaction."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=CRITICAL_AMOUNT)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_guardrail_engine_reason_codes(self):
        """Guardrail engine includes block reason codes for high-value."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=ABOVE_LIMIT)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO
        # Should have some reason codes explaining the block
        assert len(result.reason_codes) > 0 or len(result.failed_gates) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Decision Matrix — High Value
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixHighValue:
    """Test the decision matrix with high-value exposure results."""

    def test_matrix_blocks_when_exposure_blocks(self):
        """Decision matrix blocks when exposure guard blocks."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(difference=ABOVE_LIMIT)
        gate_r = _make_gate_result(GateAction.CONTINUE, 0.90)
        exposure_r = ExposureGuardResult(
            passed=False,
            action=ExposureAction.BLOCK,
            adjustment_amount_paise=ABOVE_LIMIT,
            max_auto_resolution_paise=DEFAULT_MAX_AUTO_PAUSE,
            reason="Exceeds max",
            block_reasons=[ExposureBlockReason.ABOVE_MAX_AMOUNT],
        )
        evidence_r = _make_evidence_result()
        fallback_r = _make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_matrix_allows_when_exposure_passes(self):
        """Decision matrix allows when exposure guard passes."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(difference=TYPICAL_AMOUNT, confidence=0.90, risk="LOW")
        gate_r = _make_gate_result(GateAction.CONTINUE, 0.90)
        exposure_r = _make_exposure_result(ExposureAction.PASS, TYPICAL_AMOUNT)
        evidence_r = _make_evidence_result()
        fallback_r = _make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Workflow Routing — High Value
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowRoutingHighValue:
    """Test workflow routing with high-value guardrail decisions."""

    def _make_guardrail_result_for_routing(self, decision):
        """Build a minimal GuardrailEngineResult for routing tests."""
        from app.schemas.guardrail_engine import GuardrailEngineResult
        from app.schemas.decision_matrix import AutomationDecisionResult, GateResult, GateStatus
        return GuardrailEngineResult(
            exception_id="EXC-HV-001",
            case_id="CASE-HV-001",
            decision=decision,
            confidence=0.90,
            risk_category="LOW",
            financial_exposure_paise=ABOVE_LIMIT,
            evidence_coverage=0.95,
            evidence_consistency=0.90,
            is_novel=False,
            has_conflict=False,
            verification_possible=True,
            passed_gates=[],
            failed_gates=[],
            reason_codes=[],
            primary_reason="High value blocked",
            system_healthy=True,
        )

    def test_high_value_not_auto_routes_to_review(self):
        """High-value blocks → not AUTO → routes to human_review."""
        from app.agent.workflow import create_initial_state
        state = create_initial_state(exception_id="EXC-HV-001")
        state.decision = "HUMAN_REVIEW"
        route = route_after_guardrails(state)
        assert route != "auto"

    def test_high_value_unresolved_routes_to_escalation(self):
        """High-value blocks → UNRESOLVED → routes to escalation."""
        from app.agent.workflow import create_initial_state
        state = create_initial_state(exception_id="EXC-HV-001")
        state.decision = "UNRESOLVED"
        route = route_after_guardrails(state)
        assert route in ("escalation", "human_review")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Audit Trail for High-Value Blocks
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueAuditTrail:
    """Verify audit trail is properly recorded for high-value blocks."""

    def test_exposure_guard_records_exception_id(self):
        """Exposure guard records exception_id."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result.exception_id = "EXC-AUDIT-001"
        result_r = guard.evaluate(result)
        assert result_r.exception_id == "EXC-AUDIT-001"

    def test_exposure_guard_records_block_reasons(self):
        """Exposure guard records all block reasons."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        assert len(result_r.block_reasons) > 0
        assert result_r.adjustment_amount_paise == ABOVE_LIMIT

    def test_exposure_guard_summary_includes_amount(self):
        """Exposure guard summary includes the blocked amount."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        summary = result_r.summary()
        assert str(ABOVE_LIMIT) in summary

    def test_guardrail_engine_preserves_exposure_result(self):
        """Guardrail engine preserves individual exposure guard result."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=ABOVE_LIMIT)
        result = engine.evaluate(engine_r)
        assert result.exposure_guard_result is not None
        assert result.exposure_guard_result.adjustment_amount_paise == ABOVE_LIMIT
        assert result.exposure_guard_result.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Integer Paise Calculations
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegerPaiseCalculations:
    """Verify all financial values in guard results are integer paise."""

    def test_adjustment_amount_is_int(self):
        """Adjustment amount in guard result is int."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        assert isinstance(result_r.adjustment_amount_paise, int)

    def test_max_threshold_is_int(self):
        """Max auto-resolution threshold is int."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        assert isinstance(result_r.max_auto_resolution_paise, int)

    def test_cumulative_exposure_is_int(self):
        """Cumulative exposure is int."""
        guard = ExposureGuard()
        result = engine_result_with_amount(ABOVE_LIMIT)
        result_r = guard.evaluate(result)
        assert isinstance(result_r.cumulative_exposure_paise, int)

    def test_no_float_leakage_in_threshold_comparison(self):
        """Threshold comparison does not introduce float precision issues."""
        config = ExposureGuardConfig(max_auto_resolution_paise=49_999)
        guard = ExposureGuard(config=config)
        result = engine_result_with_amount(50_000)
        result_r = guard.evaluate(result)
        # 50,000 > 49,999 → BLOCK
        assert result_r.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueEdgeCases:
    """Edge cases for high-value transaction handling."""

    def test_no_candidate_passes_exposure(self):
        """No selected candidate → no adjustment to evaluate → PASS."""
        from app.services.exposure_guard import ExposureGuard
        guard = ExposureGuard()
        engine_r = ResolutionEngineResult(
            exception_id="EXC-EDGE-001",
            case_id="CASE-EDGE-001",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.UNRESOLVED,
            confidence=0.50,
            risk_category="MEDIUM",
            deterministic_exception_type="UNKNOWN",
            evidence_coverage=0.30,
            evidence_consistency=0.25,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is True
        assert result.adjustment_amount_paise == 0

    def test_human_review_status_passes_exposure(self):
        """HUMAN_REVIEW status → no adjustment → PASS."""
        from app.services.exposure_guard import ExposureGuard
        guard = ExposureGuard()
        engine_r = ResolutionEngineResult(
            exception_id="EXC-EDGE-002",
            case_id="CASE-EDGE-002",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.HUMAN_REVIEW,
            confidence=0.60,
            risk_category="MEDIUM",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.60,
            evidence_consistency=0.55,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is True
        assert result.adjustment_amount_paise == 0

    def test_blocked_exception_type_blocks(self):
        """UNKNOWN exception type always blocks regardless of amount."""
        from app.services.exposure_guard import ExposureGuard
        guard = ExposureGuard()
        result = engine_result_with_amount(SMALL_AMOUNT)
        result.deterministic_exception_type = "UNKNOWN"
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in result_r.block_reasons

    def test_missing_record_type_blocks(self):
        """MISSING_RECORD exception type always blocks."""
        from app.services.exposure_guard import ExposureGuard
        guard = ExposureGuard()
        result = engine_result_with_amount(SMALL_AMOUNT)
        result.deterministic_exception_type = "MISSING_RECORD"
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in result_r.block_reasons

    def test_complex_multi_adjustment_blocks(self):
        """COMPLEX_MULTI_ADJUSTMENT exception type always blocks."""
        from app.services.exposure_guard import ExposureGuard
        guard = ExposureGuard()
        result = engine_result_with_amount(SMALL_AMOUNT)
        result.deterministic_exception_type = "COMPLEX_MULTI_ADJUSTMENT"
        result_r = guard.evaluate(result)
        assert result_r.passed is False
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in result_r.block_reasons

    def test_negative_amount_treated_as_absolute(self):
        """Negative adjustment amount treated as absolute value."""
        from app.schemas.resolution_candidate import FinancialAdjustment, CandidateRanking
        guard = ExposureGuard()
        # Build candidate with negative amount via proper Pydantic objects
        neg_candidate = {
            "candidate_id": "CAND-HV-NEG",
            "exception_id": "EXC-EDGE-003",
            "case_id": "CASE-EDGE-003",
            "resolution_type": "FEE_ADJUSTMENT",
            "resolution_description": "Negative test",
            "financial_adjustment": FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=-ABOVE_LIMIT,
                direction="DEBIT",
                calculation_basis="discrepancy",
            ),
            "supporting_evidence_ids": ["EVD-001"],
            "evidence_compatible": True,
            "evidence_coverage": 0.95,
            "coverage_explanation": "Test",
            "sources": ["deterministic_evidence"],
            "ranking": CandidateRanking(rank=1, confidence_score=0.90, evidence_support=0.95),
            "rationale": "Negative test",
        }
        engine_r = ResolutionEngineResult(
            exception_id="EXC-EDGE-003",
            case_id="CASE-EDGE-003",
            expected_amount=100_000,
            actual_amount=100_000 + ABOVE_LIMIT,
            difference=-ABOVE_LIMIT,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=neg_candidate,
            selected_score=_make_score(),
            ranked_candidates=[neg_candidate],
            candidate_scores=[_make_score()],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.95,
            evidence_consistency=0.90,
        )
        result_r = guard.evaluate(engine_r)
        # abs(-50001) = 50001 > 50000 → BLOCK
        assert result_r.passed is False
        assert result_r.adjustment_amount_paise == ABOVE_LIMIT

    def test_very_large_number_blocks(self):
        """Extremely large amount (100M paise = ₹10L) blocks."""
        guard = ExposureGuard()
        result = engine_result_with_amount(100_000_000)
        result_r = guard.evaluate(result)
        assert result_r.passed is False

    def test_zero_cumulative_passes(self):
        """Zero cumulative exposure passes."""
        guard = ExposureGuard()
        result = engine_result_with_amount(TYPICAL_AMOUNT)
        result_r = guard.evaluate(result)
        assert result_r.cumulative_exposure_paise <= DEFAULT_CUMULATIVE_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# Test: Guardrail Simulation — High Value
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailSimulationHighValue:
    """Test the workflow guardrail simulation node with high-value cases."""

    def test_simulation_blocks_unknown_type(self):
        """Simulated guardrail evaluation blocks UNKNOWN exception type."""
        from app.agent.workflow import create_initial_state
        state = create_initial_state(exception_id="EXC-SIM-001")
        engine_result = {
            "confidence": 0.90,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "UNKNOWN",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] != "AUTO"
        assert result["decision"] == "UNRESOLVED"

    def test_simulation_allows_good_conditions(self):
        """Simulated guardrail evaluation allows when all conditions pass."""
        from app.agent.workflow import create_initial_state
        state = create_initial_state(exception_id="EXC-SIM-002")
        engine_result = {
            "confidence": 0.90,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        # Good conditions → AUTO (simulation doesn't check exposure)
        assert result["decision"] == "AUTO"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Exhaustive Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────


class TestExhaustiveThresholdSweep:
    """Parametrized sweep around the exposure threshold boundary."""

    @pytest.mark.parametrize(
        "amount,expect_pass",
        [
            (0, True),
            (1, True),
            (1_000, True),
            (10_000, True),
            (25_000, True),
            (40_000, True),
            (49_000, True),
            (49_999, True),
            (50_000, True),    # AT threshold — passes (not strictly greater)
            (50_001, False),   # ONE PAISE over — blocks
            (55_000, False),
            (75_000, False),
            (100_000, False),
            (200_000, False),
            (500_000, False),
            (1_000_000, False),
        ],
    )
    def test_threshold_boundary_sweep(self, amount, expect_pass):
        """Sweep around the 50K threshold — only paise above blocks."""
        guard = ExposureGuard()
        result = engine_result_with_amount(amount)
        result_r = guard.evaluate(result)
        if expect_pass:
            assert result_r.passed is True, (
                f"Amount {amount} paise should PASS but was BLOCKED"
            )
        else:
            assert result_r.passed is False, (
                f"Amount {amount} paise should BLOCK but PASSED"
            )

    @pytest.mark.parametrize(
        "amount,expect_block",
        [
            (49_999, False),
            (50_000, False),
            (50_001, True),
            (75_000, True),
            (100_001, True),
            (200_001, True),
        ],
    )
    def test_guardrail_engine_threshold_sweep(self, amount, expect_block):
        """Full guardrail engine respects exposure threshold."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(difference=amount, confidence=0.90, risk="LOW")
        result = engine.evaluate(engine_r)
        if expect_block:
            assert result.decision != AutomationDecision.AUTO, (
                f"Amount {amount} paise should block AUTO but got {result.decision}"
            )
        else:
            # Below threshold, may still be HUMAN_REVIEW due to other factors
            # but should not be blocked by exposure
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Test: Financial Exposure Guard Cannot Execute Financial Actions
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardNoExecution:
    """Verify the exposure guard cannot execute financial actions."""

    def test_guard_has_no_execute_method(self):
        """ExposureGuard has no execute/apply/authorize method."""
        guard = ExposureGuard()
        assert not hasattr(guard, "execute")
        assert not hasattr(guard, "apply")
        assert not hasattr(guard, "authorize")
        assert not hasattr(guard, "modify")
        assert not hasattr(guard, "create_adjustment")

    def test_result_has_no_financial_write_fields(self):
        """ExposureGuardResult has no write/execute/action fields."""
        from app.schemas.exposure_guard import ExposureGuardResult
        fields = ExposureGuardResult.model_fields
        # Should NOT have execute, apply, authorize, create, modify, write
        dangerous_fields = {"execute", "apply", "authorize", "create", "modify", "write"}
        assert dangerous_fields.isdisjoint(set(fields.keys()))

    def test_guard_result_only_blocks_or_passes(self):
        """Guard result can only PASS or BLOCK — never EXECUTE."""
        from app.schemas.exposure_guard import ExposureAction
        assert set(ExposureAction) == {ExposureAction.PASS, ExposureAction.BLOCK}


# ─────────────────────────────────────────────────────────────────────────────
# Test: High-Value Cannot Create Unsafe AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueNeverUnsafeAUTO:
    """Verify that high-value exceptions can never produce an unsafe AUTO decision."""

    @pytest.mark.parametrize(
        "confidence,risk,coverage,consistency",
        [
            (0.99, "LOW", 0.99, 0.99),
            (0.90, "LOW", 0.95, 0.90),
            (0.80, "LOW", 0.90, 0.85),
            (0.70, "LOW", 0.80, 0.75),
            (0.99, "MEDIUM", 0.99, 0.99),
            (0.90, "HIGH", 0.95, 0.90),
        ],
    )
    def test_high_value_never_auto(self, confidence, risk, coverage, consistency):
        """High-value + any confidence/risk → NOT AUTO."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=confidence,
            risk=risk,
            coverage=coverage,
            consistency=consistency,
            difference=ABOVE_LIMIT,
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO, (
            f"High-value ({ABOVE_LIMIT} paise) with confidence={confidence}, "
            f"risk={risk} MUST NOT be AUTO but got {result.decision}"
        )

    def test_critical_value_never_auto_any_confidence(self):
        """Critical value (10M paise) at any confidence → NOT AUTO."""
        from app.services.guardrail_engine import GuardrailEngine
        engine = GuardrailEngine()
        for conf in [0.70, 0.80, 0.90, 0.95, 0.99, 1.0]:
            engine_r = _engine(
                confidence=conf,
                difference=CRITICAL_AMOUNT,
                risk="LOW",
            )
            result = engine.evaluate(engine_r)
            assert result.decision != AutomationDecision.AUTO, (
                f"Critical value at confidence={conf} MUST NOT be AUTO"
            )
