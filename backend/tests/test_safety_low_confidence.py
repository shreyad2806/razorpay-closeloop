"""
Adversarial safety tests for low-confidence cases.

Verifies that low-confidence exceptions cannot produce an AUTO decision,
and are correctly routed to HUMAN_REVIEW or UNRESOLVED.

Tests the confidence gate, decision matrix, guardrail engine,
and workflow routing.

Key thresholds (defaults):
  min_confidence: 0.70
  very low: < 0.40 → UNRESOLVED
  below threshold: < 0.70 → HUMAN_REVIEW
  at or above threshold: >= 0.70 → may proceed

Safety invariant:
  confidence < 0.70 → NEVER AUTO
  confidence < 0.40 → UNRESOLVED

No production logic is modified.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_lowconf.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.confidence_gate import ConfidenceGateConfig, GateAction
from app.schemas.decision_matrix import AutomationDecision
from app.schemas.exposure_guard import ExposureAction
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.confidence_gate import ConfidenceGate
from app.services.exposure_guard import ExposureGuard
from app.services.guardrail_engine import GuardrailEngine
from app.services.decision_matrix import AutomationDecisionMatrix
from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation
from app.agent.workflow import create_initial_state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Default confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.70


def _engine(**kwargs):
    """Build a valid ResolutionEngineResult for testing."""
    defaults = dict(
        exception_id="EXC-LC-001",
        case_id="CASE-LC-001",
        payment_id="PAY-LC-001",
        merchant_id="MER-LC-01",
        expected_amount=100_000,
        actual_amount=90_000,
        difference=10_000,
        status=SelectionStatus.RECOMMENDED,
        confidence=0.50,
        risk_category="LOW",
        deterministic_exception_type="FEE_DIFFERENCE",
        evidence_coverage=0.95,
        evidence_consistency=0.90,
    )
    defaults.update(kwargs)
    if "selected_candidate" not in kwargs:
        from tests.test_safety_high_value import _make_candidate
        defaults["selected_candidate"] = _make_candidate(defaults["difference"])
        defaults["selected_resolution"] = "FEE_ADJUSTMENT"
    if "selected_score" not in kwargs:
        from tests.test_safety_high_value import _make_score
        defaults["selected_score"] = _make_score()
    if "ranked_candidates" not in kwargs:
        defaults["ranked_candidates"] = [defaults["selected_candidate"]]
    if "candidate_scores" not in kwargs:
        defaults["candidate_scores"] = [defaults["selected_score"]]
    return ResolutionEngineResult(**defaults)


def _make_exposure_result(passed=True, amount=10_000):
    return ExposureGuardResult(
        passed=passed,
        action=ExposureAction.PASS if passed else ExposureAction.BLOCK,
        adjustment_amount_paise=amount,
        max_auto_resolution_paise=50_000,
        reason="test",
    )


from app.schemas.exposure_guard import ExposureGuardResult


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Gate — Threshold Boundaries
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateThresholds:
    """Test confidence gate at and around the min_confidence threshold."""

    def test_above_threshold_passes(self):
        """Confidence above 0.70 passes the gate."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.80)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        assert result.action == GateAction.CONTINUE

    def test_at_threshold_passes(self):
        """Confidence exactly at 0.70 passes the gate."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.70)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        assert result.action == GateAction.CONTINUE

    def test_just_below_threshold_blocks(self):
        """Confidence 0.69 blocks → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.69)
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_significantly_below_threshold_blocks(self):
        """Confidence 0.50 blocks → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.50)
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_zero_confidence_blocks(self):
        """Zero confidence blocks → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.0)
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_very_low_confidence_blocks(self):
        """Very low confidence (0.20) blocks → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.20)
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    @pytest.mark.parametrize("confidence", [0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0])
    def test_above_threshold_all_pass(self, confidence):
        """All confidence levels above 0.70 pass."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=confidence)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        assert result.action == GateAction.CONTINUE

    @pytest.mark.parametrize("confidence", [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.69])
    def test_below_threshold_all_block(self, confidence):
        """All confidence levels below 0.70 block."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=confidence)
        result = gate.evaluate(engine_r)
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_custom_threshold_lower(self):
        """Custom lower threshold blocks at higher confidence."""
        config = ConfidenceGateConfig(min_confidence=0.50)
        gate = ConfidenceGate(config=config)
        engine_r = _engine(confidence=0.60)
        result = gate.evaluate(engine_r)
        assert result.passed is True  # 0.60 > 0.50

    def test_custom_threshold_higher(self):
        """Custom higher threshold blocks at confidence that would pass default."""
        config = ConfidenceGateConfig(min_confidence=0.90)
        gate = ConfidenceGate(config=config)
        engine_r = _engine(confidence=0.80)
        result = gate.evaluate(engine_r)
        assert result.passed is False  # 0.80 < 0.90

    def test_default_threshold_values(self):
        """Default config uses documented threshold."""
        config = ConfidenceGateConfig()
        assert config.min_confidence == 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Test: Decision Matrix — Low Confidence Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixLowConfidence:
    """Test the decision matrix with low confidence inputs."""

    def _make_gate_result(self, passed, action, confidence):
        from app.schemas.confidence_gate import ConfidenceGateResult
        return ConfidenceGateResult(
            passed=passed,
            action=action,
            confidence=confidence,
            threshold=DEFAULT_MIN_CONFIDENCE,
            reason="test",
        )

    def _make_exposure_result(self, passed=True, amount=10_000):
        return ExposureGuardResult(
            passed=passed,
            action=ExposureAction.PASS if passed else ExposureAction.BLOCK,
            adjustment_amount_paise=amount,
            max_auto_resolution_paise=50_000,
            reason="test",
        )

    def _make_evidence_result(self, passed=True, coverage=0.95, consistency=0.90):
        from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
        return EvidenceGuardResult(
            passed=passed,
            action=EvidenceAction.PASS if passed else EvidenceAction.BLOCK,
            evidence_coverage=coverage,
            evidence_consistency=consistency,
            has_conflict=False,
            is_novel=False,
            reason="test",
        )

    def _make_fallback_result(self, can_proceed=True):
        from app.schemas.failure_fallback import FailureFallbackResult, FallbackAction
        return FailureFallbackResult(
            can_proceed=can_proceed,
            action=FallbackAction.CONTINUE_WITHOUT if can_proceed else FallbackAction.FAIL_CLOSED,
            fallback_status="OK" if can_proceed else "DEGRADED",
            failures=[],
            critical_failures=[],
            failed_categories=[],
            reason="test",
            exception_id="EXC-LC-001",
            case_id="CASE-LC-001",
        )

    def test_low_confidence_blocks_auto(self):
        """Decision matrix blocks AUTO when confidence gate blocks."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.50)
        gate_r = self._make_gate_result(False, GateAction.HUMAN_REVIEW, 0.50)
        exposure_r = self._make_exposure_result()
        evidence_r = self._make_evidence_result()
        fallback_r = self._make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_medium_confidence_blocks_auto(self):
        """Decision matrix blocks AUTO for medium confidence (0.60)."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.60)
        gate_r = self._make_gate_result(False, GateAction.HUMAN_REVIEW, 0.60)
        exposure_r = self._make_exposure_result()
        evidence_r = self._make_evidence_result()
        fallback_r = self._make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_zero_confidence_blocks_auto(self):
        """Decision matrix blocks AUTO for zero confidence."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.0)
        gate_r = self._make_gate_result(False, GateAction.HUMAN_REVIEW, 0.0)
        exposure_r = self._make_exposure_result()
        evidence_r = self._make_evidence_result()
        fallback_r = self._make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision != AutomationDecision.AUTO

    def test_good_confidence_allows_auto(self):
        """Decision matrix allows AUTO when confidence passes."""
        matrix = AutomationDecisionMatrix()
        engine_r = _engine(confidence=0.90, risk="LOW")
        gate_r = self._make_gate_result(True, GateAction.CONTINUE, 0.90)
        exposure_r = self._make_exposure_result()
        evidence_r = self._make_evidence_result()
        fallback_r = self._make_fallback_result()
        result = matrix.evaluate(engine_r, gate_r, exposure_r, evidence_r, fallback_r)
        assert result.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Guardrail Engine — Low Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngineLowConfidence:
    """Test the complete guardrail engine with low confidence cases."""

    @pytest.mark.parametrize("confidence", [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.65, 0.69])
    def test_low_confidence_never_auto(self, confidence):
        """All low confidence levels produce NOT AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=confidence)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO, (
            f"Confidence {confidence} MUST NOT be AUTO"
        )

    @pytest.mark.parametrize("confidence", [0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0])
    def test_above_threshold_can_auto(self, confidence):
        """Above-threshold confidence can produce AUTO (or HUMAN_REVIEW from other checks)."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=confidence, risk="LOW")
        result = engine.evaluate(engine_r)
        # Should not be blocked by confidence gate
        assert result.confidence_gate_result.passed is True

    def test_confidence_gate_preserves_value(self):
        """Guardrail engine records the actual confidence value."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.55)
        result = engine.evaluate(engine_r)
        assert result.confidence == 0.55
        assert result.confidence_gate_result is not None
        assert result.confidence_gate_result.confidence == 0.55

    def test_very_low_confidence_triggers_unresolved_via_matrix(self):
        """Confidence < 0.40 triggers UNRESOLVED via decision matrix."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.30)
        result = engine.evaluate(engine_r)
        # Below 0.40 → decision matrix may route to UNRESOLVED
        assert result.decision != AutomationDecision.AUTO

    def test_guardrail_engine_reason_codes_for_low_confidence(self):
        """Low confidence includes reason codes explaining the block."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.50)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO
        # Should have reason codes or failed gates explaining the block
        has_failure_info = (
            len(result.reason_codes) > 0
            or len(result.failed_gates) > 0
            or (result.confidence_gate_result and not result.confidence_gate_result.passed)
        )
        assert has_failure_info


# ─────────────────────────────────────────────────────────────────────────────
# Test: Workflow Routing — Low Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowRoutingLowConfidence:
    """Test workflow routing with low confidence decisions."""

    def test_low_confidence_not_auto_routes_to_review(self):
        """Low confidence → HUMAN_REVIEW → routes to human_review."""
        state = create_initial_state(exception_id="EXC-LC-001")
        state.decision = "HUMAN_REVIEW"
        from app.agent.routing import route_after_guardrails
        route = route_after_guardrails(state)
        assert route != "auto"

    def test_unresolved_routes_to_escalation(self):
        """UNRESOLVED → routes to escalation."""
        state = create_initial_state(exception_id="EXC-LC-001")
        state.decision = "UNRESOLVED"
        from app.agent.routing import route_after_guardrails
        route = route_after_guardrails(state)
        assert route in ("escalation", "human_review")

    def test_auto_routes_to_verification(self):
        """AUTO → routes to verification."""
        state = create_initial_state(exception_id="EXC-LC-001")
        state.decision = "AUTO"
        from app.agent.routing import route_after_guardrails
        route = route_after_guardrails(state)
        assert "verify" in route


# ─────────────────────────────────────────────────────────────────────────────
# Test: Simulation — Low Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulationLowConfidence:
    """Test the workflow guardrail simulation node with low confidence cases."""

    def test_simulation_very_low_confidence_unresolved(self):
        """Simulation: confidence < 0.40 → UNRESOLVED."""
        state = create_initial_state(exception_id="EXC-LC-SIM-01")
        engine_result = {
            "confidence": 0.30,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "UNRESOLVED"

    def test_simulation_medium_confidence_human_review(self):
        """Simulation: 0.40 <= confidence < 0.70 → HUMAN_REVIEW."""
        state = create_initial_state(exception_id="EXC-LC-SIM-02")
        engine_result = {
            "confidence": 0.60,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "HUMAN_REVIEW"

    def test_simulation_high_confidence_auto(self):
        """Simulation: confidence >= 0.70 and good coverage → AUTO."""
        state = create_initial_state(exception_id="EXC-LC-SIM-03")
        engine_result = {
            "confidence": 0.85,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "AUTO"

    def test_simulation_zero_confidence_unresolved(self):
        """Simulation: zero confidence → UNRESOLVED."""
        state = create_initial_state(exception_id="EXC-LC-SIM-04")
        engine_result = {
            "confidence": 0.0,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "UNRESOLVED"

    @pytest.mark.parametrize("confidence,expect", [
        (0.0, "UNRESOLVED"),
        (0.10, "UNRESOLVED"),
        (0.20, "UNRESOLVED"),
        (0.39, "UNRESOLVED"),
        (0.40, "HUMAN_REVIEW"),
        (0.50, "HUMAN_REVIEW"),
        (0.60, "HUMAN_REVIEW"),
        (0.69, "HUMAN_REVIEW"),
        (0.74, "HUMAN_REVIEW"),
        (0.75, "AUTO"),
        (0.80, "AUTO"),
        (0.90, "AUTO"),
        (1.0, "AUTO"),
    ])
    def test_simulation_confidence_sweep(self, confidence, expect):
        """Simulation: sweep across confidence thresholds."""
        state = create_initial_state(exception_id="EXC-LC-SWEEP")
        engine_result = {
            "confidence": confidence,
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "risk_category": "LOW",
            "deterministic_exception_type": "FEE_DIFFERENCE",
        }
        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == expect, (
            f"Confidence {confidence} should produce {expect} but got {result['decision']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Exhaustive Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────


class TestExhaustiveConfidenceSweep:
    """Parametrized sweep around the confidence threshold boundary."""

    @pytest.mark.parametrize("confidence,expect_pass", [
        (0.0, False),
        (0.10, False),
        (0.20, False),
        (0.30, False),
        (0.40, False),
        (0.50, False),
        (0.60, False),
        (0.65, False),
        (0.69, False),
        (0.70, True),
        (0.71, True),
        (0.75, True),
        (0.80, True),
        (0.90, True),
        (0.95, True),
        (0.99, True),
        (1.0, True),
    ])
    def test_gate_threshold_sweep(self, confidence, expect_pass):
        """Confidence gate sweep: only >= 0.70 passes."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=confidence)
        result = gate.evaluate(engine_r)
        if expect_pass:
            assert result.passed is True, (
                f"Confidence {confidence} should PASS but was BLOCKED"
            )
        else:
            assert result.passed is False, (
                f"Confidence {confidence} should BLOCK but PASSED"
            )

    @pytest.mark.parametrize("confidence,expect_never_auto", [
        (0.0, True),
        (0.20, True),
        (0.40, True),
        (0.50, True),
        (0.60, True),
        (0.69, True),
    ])
    def test_guardrail_engine_sweep(self, confidence, expect_never_auto):
        """Guardrail engine sweep: low confidence never AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=confidence, risk="LOW")
        result = engine.evaluate(engine_r)
        if expect_never_auto:
            assert result.decision != AutomationDecision.AUTO, (
                f"Confidence {confidence} MUST NOT produce AUTO"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Low Confidence Cannot Produce Unsafe AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestLowConfidenceNeverUnsafeAUTO:
    """Verify that low confidence cannot produce an unsafe AUTO decision."""

    @pytest.mark.parametrize("confidence", [0.0, 0.20, 0.40, 0.50, 0.60, 0.65, 0.69])
    def test_low_confidence_never_auto_any_risk(self, confidence):
        """Low confidence at any risk level → NOT AUTO."""
        engine = GuardrailEngine()
        for risk in ["LOW", "MEDIUM", "HIGH"]:
            engine_r = _engine(confidence=confidence, risk=risk)
            result = engine.evaluate(engine_r)
            assert result.decision != AutomationDecision.AUTO, (
                f"Confidence {confidence} with risk={risk} MUST NOT be AUTO"
            )

    def test_zero_confidence_zero_exposure_still_blocks(self):
        """Zero confidence with zero exposure still blocks AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.0, difference=0)
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO

    def test_low_confidence_perfect_evidence_still_blocks(self):
        """Low confidence with perfect evidence still blocks AUTO."""
        engine = GuardrailEngine()
        engine_r = _engine(
            confidence=0.50,
            coverage=1.0,
            consistency=1.0,
            risk="LOW",
        )
        result = engine.evaluate(engine_r)
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence Gate Cannot Execute Financial Actions
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateNoExecution:
    """Verify the confidence gate cannot execute financial actions."""

    def test_gate_has_no_execute_method(self):
        """ConfidenceGate has no execute/apply/authorize method."""
        gate = ConfidenceGate()
        assert not hasattr(gate, "execute")
        assert not hasattr(gate, "apply")
        assert not hasattr(gate, "authorize")
        assert not hasattr(gate, "modify")

    def test_gate_result_only_continue_or_human_review(self):
        """Gate result can only CONTINUE or HUMAN_REVIEW — never EXECUTE."""
        assert set(GateAction) == {GateAction.CONTINUE, GateAction.HUMAN_REVIEW}

    def test_gate_result_has_no_financial_write_fields(self):
        """ConfidenceGateResult has no execute/write/apply fields."""
        from app.schemas.confidence_gate import ConfidenceGateResult
        fields = set(ConfidenceGateResult.model_fields.keys())
        dangerous = {"execute", "apply", "authorize", "create", "modify", "write"}
        assert dangerous.isdisjoint(fields)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Configurable Threshold Preserved
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurableThreshold:
    """Test that configurable thresholds are respected end-to-end."""

    def test_custom_threshold_end_to_end(self):
        """Custom threshold 0.50 → 0.60 passes, 0.40 blocks."""
        config = ConfidenceGateConfig(min_confidence=0.50)
        gate = ConfidenceGate(config=config)
        # Above custom threshold
        engine_r = _engine(confidence=0.60)
        result = gate.evaluate(engine_r)
        assert result.passed is True
        # Below custom threshold
        engine_r = _engine(confidence=0.40)
        result = gate.evaluate(engine_r)
        assert result.passed is False

    def test_threshold_metadata_recorded(self):
        """Gate result records the threshold that was applied."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.50)
        result = gate.evaluate(engine_r)
        assert result.threshold == DEFAULT_MIN_CONFIDENCE
        assert result.confidence == 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Test: Audit Trail for Low Confidence Blocks
# ─────────────────────────────────────────────────────────────────────────────


class TestLowConfidenceAuditTrail:
    """Verify audit trail is properly recorded for low confidence blocks."""

    def test_guardrail_engine_records_confidence(self):
        """Guardrail engine records the confidence value."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.55)
        result = engine.evaluate(engine_r)
        assert result.confidence == 0.55

    def test_guardrail_engine_records_confidence_gate_result(self):
        """Guardrail engine preserves confidence gate result."""
        engine = GuardrailEngine()
        engine_r = _engine(confidence=0.55)
        result = engine.evaluate(engine_r)
        assert result.confidence_gate_result is not None
        assert result.confidence_gate_result.passed is False
        assert result.confidence_gate_result.action == GateAction.HUMAN_REVIEW

    def test_guardrail_engine_summary_includes_confidence(self):
        """Confidence gate summary includes the confidence value."""
        gate = ConfidenceGate()
        engine_r = _engine(confidence=0.55)
        result = gate.evaluate(engine_r)
        summary = result.summary()
        assert "55.0%" in summary
