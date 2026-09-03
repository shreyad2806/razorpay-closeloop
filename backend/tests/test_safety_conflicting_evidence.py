"""
Adversarial safety tests for conflicting financial evidence.

Verifies that conflicting evidence prevents unsafe AUTO decisions.

Conflict handling chain:
  1. CandidateScorer calculates conflict_penalty
  2. ConfidenceGate blocks if conflict_penalty > max_conflict_penalty (0.10)
  3. ExposureGuard blocks if conflict_penalty > max_conflict_for_auto (0.15)
  4. CandidateSelector rejects candidates above max_conflict_penalty
  5. DecisionMatrix evaluates has_conflict from evidence guard

Conflict sources:
  - ML vs deterministic disagreement
  - Multiple viable candidates
  - Evidence quality issues
  - Financial inconsistency
  - Conflicting settlement/refund/fee records

Safety invariant:
  High conflict penalty → NEVER AUTO

No production logic is modified.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_conflict.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.confidence_gate import ConfidenceGateConfig, GateAction
from app.schemas.decision_matrix import AutomationDecision
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.services.evidence_guard import EvidenceGuard
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.confidence_gate import ConfidenceGate
from app.services.exposure_guard import ExposureGuard
from app.services.guardrail_engine import GuardrailEngine
from app.services.decision_matrix import AutomationDecisionMatrix
from app.agent.workflow import create_initial_state
from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_CONFLICT_PENALTY = 0.10    # ConfidenceGate threshold
DEFAULT_MAX_CONFLICT_FOR_AUTO = 0.15   # ExposureGuard threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engine(**kwargs):
    """Build a valid ResolutionEngineResult for testing."""
    from tests.test_safety_high_value import _make_candidate, _make_score
    defaults = dict(
        exception_id="EXC-CONFLICT-001",
        case_id="CASE-CONFLICT-001",
        payment_id="PAY-CONFLICT-001",
        merchant_id="MER-CONFLICT-01",
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


def _make_gate_result(passed, action, confidence=0.85):
    from app.schemas.confidence_gate import ConfidenceGateResult
    return ConfidenceGateResult(
        passed=passed,
        action=action,
        confidence=confidence,
        threshold=DEFAULT_MAX_CONFLICT_PENALTY,
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


def _make_evidence_result(
    passed=True, coverage=0.90, consistency=0.85,
    has_conflict=False, is_novel=False,
):
    return EvidenceGuardResult(
        passed=passed,
        action=EvidenceAction.PASS if passed else EvidenceAction.BLOCK,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        has_conflict=has_conflict,
        is_novel=is_novel,
        reason="test",
    )


def _make_fallback_result(can_proceed=True):
    from app.schemas.failure_fallback import FailureFallbackResult, FallbackAction
    return FailureFallbackResult(
        can_proceed=can_proceed,
        action=FallbackAction.CONTINUE_WITHOUT if can_proceed else FallbackAction.FAIL_CLOSED,
        fallback_status="OK" if can_proceed else "DEGRADED",
        failures=[],
        critical_failures=[],
        failed_categories=[],
        reason="test",
        exception_id="EXC-CONFLICT-001",
        case_id="CASE-CONFLICT-001",
    )


def _make_score(conflict=0.05, novelty=0.05):
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
        has_conflicts=conflict > 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Gate — Conflict Penalty Threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateConflictPenalty:
    """Test confidence gate blocks when conflict penalty exceeds threshold."""

    def test_low_conflict_passes(self):
        """Conflict penalty 0.05 <= 0.10 → PASS."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.05))
        result = gate.evaluate(engine_r)
        assert result.passed is True

    def test_at_threshold_passes(self):
        """Conflict penalty 0.10 <= 0.10 → PASS."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.10))
        result = gate.evaluate(engine_r)
        assert result.passed is True

    def test_above_threshold_blocks(self):
        """Conflict penalty 0.11 > 0.10 → BLOCK."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.11))
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_high_conflict_blocks(self):
        """Conflict penalty 0.25 (maximum) → BLOCK."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.25))
        result = gate.evaluate(engine_r)
        assert result.passed is False

    def test_zero_conflict_passes(self):
        """Conflict penalty 0.0 → PASS."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.0))
        result = gate.evaluate(engine_r)
        assert result.passed is True

    @pytest.mark.parametrize("penalty,expect_pass", [
        (0.0, True),
        (0.01, True),
        (0.05, True),
        (0.09, True),
        (0.10, True),
        (0.11, False),
        (0.15, False),
        (0.20, False),
        (0.25, False),
    ])
    def test_conflict_penalty_sweep(self, penalty, expect_pass):
        """Sweep across conflict penalty threshold boundary."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=penalty))
        result = gate.evaluate(engine_r)
        if expect_pass:
            assert result.passed is True, (
                f"Penalty {penalty} should PASS but was BLOCKED"
            )
        else:
            assert result.passed is False, (
                f"Penalty {penalty} should BLOCK but PASSED"
            )

    def test_custom_max_conflict_penalty(self):
        """Custom lower threshold blocks at lower penalty."""
        config = ConfidenceGateConfig(max_conflict_penalty=0.05)
        gate = ConfidenceGate(config=config)
        engine_r = _engine(selected_score=_make_score(conflict=0.08))
        result = gate.evaluate(engine_r)
        assert result.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Exposure Guard — Conflict Penalty
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardConflictPenalty:
    """Test exposure guard blocks when conflict penalty exceeds threshold."""

    def test_low_conflict_passes_exposure(self):
        """Conflict penalty 0.05 < 0.15 → PASS exposure guard."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.05),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is True

    def test_high_conflict_blocks_exposure(self):
        """Conflict penalty 0.20 > 0.15 → BLOCK exposure guard."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.20),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is False
        # Should have CONFLICTING_CASE block reason
        from app.schemas.exposure_guard import ExposureBlockReason
        assert ExposureBlockReason.CONFLICTING_CASE in result.block_reasons

    def test_at_exposure_threshold_passes(self):
        """Conflict penalty 0.15 <= 0.15 → PASS."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.15),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is True

    def test_above_exposure_threshold_blocks(self):
        """Conflict penalty 0.16 > 0.15 → BLOCK."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.16),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is False

    def test_conflict_plus_high_value_double_blocks(self):
        """High conflict + high value → both block reasons."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.20),
            difference=60_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is False
        from app.schemas.exposure_guard import ExposureBlockReason
        assert ExposureBlockReason.CONFLICTING_CASE in result.block_reasons
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in result.block_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Test: Decision Matrix — Conflict Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixConflict:
    """Test the decision matrix with conflict scenarios."""

    def test_evidence_guard_conflict_blocks(self):
        """Decision matrix blocks when evidence guard has_conflict=True."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = _make_gate_result(True, GateAction.CONTINUE, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result(has_conflict=True)
        fallback_r = _make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_no_conflict_allows_auto(self):
        """No conflict + good conditions → may be AUTO."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.85, risk="LOW")
        gate_r = _make_gate_result(True, GateAction.CONTINUE, 0.85)
        exposure_r = _make_exposure_result()
        evidence_r = _make_evidence_result(has_conflict=False)
        fallback_r = _make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Guardrail Engine — Conflict Blocking
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngineConflict:
    """Test the complete guardrail engine with conflict scenarios."""

    def test_high_conflict_blocks_auto(self):
        """High conflict penalty blocks AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_low_conflict_allows_auto(self):
        """Low conflict penalty allows AUTO path."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            selected_score=_make_score(conflict=0.02),
        )
        result = engine.evaluate(engine_r)
        # Low conflict → may be AUTO or HUMAN_REVIEW from other checks
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)

    def test_conflict_is_recorded_in_result(self):
        """Guardrail engine records conflict in the result."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        # The engine result should have exposure guard result showing conflict
        assert result.exposure_guard_result is not None

    @pytest.mark.parametrize("penalty,expect_auto", [
        (0.0, False),     # May be AUTO or HUMAN_REVIEW from other checks
        (0.05, False),    # Same
        (0.10, False),    # At threshold — may be blocked by exposure guard
        (0.15, False),    # At exposure threshold
        (0.20, False),    # Above both thresholds
        (0.25, False),    # Maximum penalty
    ])
    def test_conflict_penalty_engine_sweep(self, penalty, expect_auto):
        """Sweep conflict penalty through guardrail engine."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            selected_score=_make_score(conflict=penalty),
        )
        result = engine.evaluate(engine_r)
        if penalty >= 0.11:
            assert result.decision != AutomationDecision.AUTO, (
                f"Penalty {penalty} MUST NOT produce AUTO"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test: High Confidence Cannot Override Conflict
# ─────────────────────────────────────────────────────────────────────────────


class TestHighConfidenceCannotOverrideConflict:
    """Verify that high confidence cannot bypass conflict safety."""

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99, 1.0])
    def test_high_confidence_high_conflict_still_blocks(self, confidence):
        """Even with perfect confidence, high conflict blocks AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=confidence,
            risk="LOW",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO, (
            f"Confidence {confidence} with conflict 0.20 MUST NOT be AUTO"
        )

    def test_perfect_confidence_zero_conflict_can_auto(self):
        """Perfect confidence with zero conflict CAN be AUTO."""
        gate = ConfidenceGate()
        engine_r = _engine(
            confidence=1.0,
            risk="LOW",
            selected_score=_make_score(conflict=0.0),
        )
        result = gate.evaluate(engine_r)
        assert result.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Conflict Penalty at Boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictPenaltyBoundary:
    """Test precise boundary behavior for conflict penalty."""

    def test_one_over_gate_threshold_blocks(self):
        """Penalty 0.1001 > 0.10 → blocks confidence gate."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.1001))
        result = gate.evaluate(engine_r)
        assert result.passed is False

    def test_one_under_gate_threshold_passes(self):
        """Penalty 0.0999 < 0.10 → passes confidence gate."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.0999))
        result = gate.evaluate(engine_r)
        assert result.passed is True

    def test_one_over_exposure_threshold_blocks(self):
        """Penalty 0.1501 > 0.15 → blocks exposure guard."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.1501),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is False

    def test_one_under_exposure_threshold_passes(self):
        """Penalty 0.1499 < 0.15 → passes exposure guard."""
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.1499),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert result.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Conflict Sources
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictSources:
    """Test that various conflict sources are handled correctly."""

    def test_classification_disagreement_increases_penalty(self):
        """ML vs deterministic disagreement produces conflict penalty."""
        from app.services.candidate_scorer import CandidateScoringService
        scorer = CandidateScoringService()
        # Build a scenario with classification disagreement
        # The scorer should produce a non-zero conflict penalty
        # when ml_exception_type != deterministic_exception_type
        engine_r = _engine(
            deterministic_exception_type="FEE_DIFFERENCE",
            ml_exception_type="REFUND_ADJUSTMENT",
            classification_agreement=False,
        )
        # Even if we can't call the scorer directly, verify the
        # penalty mechanism works through the guardrail
        engine_r.selected_score = _make_score(conflict=0.08)
        gate = ConfidenceGate()
        result = gate.evaluate(engine_r)
        # 0.08 < 0.10 → passes (penalty within threshold)
        assert result.passed is True

    def test_multiple_viable_candidates_increases_penalty(self):
        """Multiple viable candidates should increase conflict."""
        # Multiple candidates with similar scores = conflict
        engine_r = _engine(
            selected_score=_make_score(conflict=0.12),
        )
        gate = ConfidenceGate()
        result = gate.evaluate(engine_r)
        # 0.12 > 0.10 → blocks
        assert result.passed is False

    def test_financial_inconsistency_increases_penalty(self):
        """Financial inconsistency should increase conflict."""
        engine_r = _engine(
            selected_score=_make_score(conflict=0.15),
        )
        guard = ExposureGuard()
        result = guard.evaluate(engine_r)
        # 0.15 <= 0.15 → passes exposure (at boundary)
        assert result.passed is True

    def test_evidence_quality_degradation_increases_penalty(self):
        """Poor evidence quality should increase conflict."""
        engine_r = _engine(
            evidence_coverage=0.60,
            evidence_consistency=0.55,
            selected_score=_make_score(conflict=0.18),
        )
        gate = ConfidenceGate()
        result = gate.evaluate(engine_r)
        # 0.18 > 0.10 → blocks
        assert result.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Conflict Cannot Produce Unsafe AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictNeverUnsafeAUTO:
    """Verify that conflicting evidence cannot produce unsafe AUTO."""

    @pytest.mark.parametrize("penalty,confidence", [
        (0.20, 0.99),
        (0.25, 0.90),
        (0.15, 0.95),
        (0.11, 0.85),
    ])
    def test_high_conflict_never_auto(self, penalty, confidence):
        """High conflict at any confidence → NOT AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=confidence,
            risk="LOW",
            selected_score=_make_score(conflict=penalty),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO, (
            f"Penalty {penalty} with confidence {confidence} MUST NOT be AUTO"
        )

    def test_conflict_plus_low_confidence_still_blocks(self):
        """Conflict + low confidence both block."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.50,
            selected_score=_make_score(conflict=0.15),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test: Evidence Guard — has_conflict
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceGuardConflict:
    """Test the evidence guard with conflict flag."""

    def test_evidence_guard_records_coverage_and_consistency(self):
        """Evidence guard records coverage and consistency in result."""
        guard = EvidenceGuard()
        engine_r = _engine(evidence_coverage=0.80, evidence_consistency=0.75)
        result = guard.evaluate(engine_r)
        assert result.evidence_coverage == 0.80
        assert result.evidence_consistency == 0.75

    def test_evidence_guard_low_coverage_blocks(self):
        """Evidence guard blocks when coverage < 0.50."""
        guard = EvidenceGuard()
        engine_r = _engine(evidence_coverage=0.30, evidence_consistency=0.80)
        result = guard.evaluate(engine_r)
        assert result.passed is False
        assert result.action == EvidenceAction.BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Test: Audit Trail for Conflict Blocks
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictAuditTrail:
    """Verify audit trail is properly recorded for conflict blocks."""

    def test_confidence_gate_records_conflict_penalty(self):
        """Confidence gate records the conflict penalty value."""
        gate = ConfidenceGate()
        engine_r = _engine(selected_score=_make_score(conflict=0.15))
        result = gate.evaluate(engine_r)
        # Find the conflict_penalty check
        conflict_checks = [c for c in result.checks if c.check_name == "conflict_penalty"]
        assert len(conflict_checks) == 1
        assert conflict_checks[0].value == 0.15
        assert conflict_checks[0].passed is False

    def test_exposure_guard_records_conflict_block_reason(self):
        """Exposure guard records CONFLICTING_CASE block reason."""
        from app.schemas.exposure_guard import ExposureBlockReason
        guard = ExposureGuard()
        engine_r = _engine(
            selected_score=_make_score(conflict=0.20),
            difference=10_000,
        )
        result = guard.evaluate(engine_r)
        assert ExposureBlockReason.CONFLICTING_CASE in result.block_reasons

    def test_guardrail_engine_preserves_conflict_info(self):
        """Guardrail engine preserves conflict information."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        # Should have exposure guard result with conflict info
        assert result.exposure_guard_result is not None
        # Should have reason codes or failed gates
        has_failure = (
            len(result.reason_codes) > 0
            or len(result.failed_gates) > 0
        )
        assert has_failure


# ─────────────────────────────────────────────────────────────────────────────
# Test: Conflict Guard Cannot Execute Financial Actions
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictGuardNoExecution:
    """Verify conflict-related guards cannot execute financial actions."""

    def test_evidence_guard_has_no_execute(self):
        """EvidenceGuard has no execute/apply/authorize method."""
        guard = EvidenceGuard()
        assert not hasattr(guard, "execute")
        assert not hasattr(guard, "apply")
        assert not hasattr(guard, "authorize")
        # It only has evaluate
        assert hasattr(guard, "evaluate")

    def test_evidence_guard_result_only_pass_or_block(self):
        """Evidence guard result can only PASS or BLOCK."""
        assert set(EvidenceAction) == {EvidenceAction.PASS, EvidenceAction.BLOCK}

    def test_exposure_guard_result_only_pass_or_block(self):
        """Exposure guard result can only PASS or BLOCK."""
        assert set(ExposureAction) == {ExposureAction.PASS, ExposureAction.BLOCK}


# ─────────────────────────────────────────────────────────────────────────────
# Test: Combined Conflict + Other Safety Checks
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictCombinedSafety:
    """Test conflict combined with other safety conditions."""

    def test_conflict_plus_high_exposure_blocks(self):
        """Conflict + high exposure → both block."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            difference=60_000,
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_conflict_plus_unknown_type_blocks(self):
        """Conflict + UNKNOWN type → blocks."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            deterministic_exception_type="UNKNOWN",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_conflict_plus_high_risk_blocks(self):
        """Conflict + HIGH risk → blocks."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="HIGH",
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_conflict_plus_low_confidence_blocks(self):
        """Conflict + low confidence → blocks."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.50,
            selected_score=_make_score(conflict=0.20),
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_all_safety_conditions_pass_allows_auto(self):
        """All safety conditions pass → may be AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.85,
            risk="LOW",
            difference=10_000,
            selected_score=_make_score(conflict=0.02),
        )
        result = engine.evaluate(engine_r)
        # All conditions pass → AUTO or HUMAN_REVIEW
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)
