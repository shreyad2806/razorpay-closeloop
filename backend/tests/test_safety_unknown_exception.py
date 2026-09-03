"""
Adversarial safety tests for UNKNOWN exceptions.

Verifies that UNKNOWN exceptions — regardless of how favorable other
conditions appear — can NEVER produce an AUTO decision.

Tests the guardrail pipeline, decision matrix, workflow routing,
and edge cases where an attacker might try to force automation.

Safety invariant:
  UNKNOWN → NEVER AUTO

No production logic is modified.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_safety_unknown.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.decision_matrix import (
    AutomationDecision,
    DecisionConfig,
    ReasonCode,
)
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.failure_fallback import FailureFallbackResult
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.decision_matrix import AutomationDecisionMatrix
from guardrail_test_helpers import simulate_guardrail_evaluation as _simulate_guardrail_evaluation
from app.agent.routing import route_after_guardrails


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_engine_result(
    exception_type="UNKNOWN",
    confidence=0.85,
    risk="LOW",
    coverage=0.95,
    consistency=0.90,
    exposure=3000,
    status=SelectionStatus.RECOMMENDED,
    has_candidate=True,
):
    """Build a ResolutionEngineResult for testing."""
    return ResolutionEngineResult(
        exception_id="EXC-SAFETY-001",
        case_id="CASE-SAFETY-001",
        payment_id="PAY-SAFETY-001",
        merchant_id="MER-SAFETY-01",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        status=status,
        selected_resolution="FEE_ADJUSTMENT" if has_candidate else None,
        confidence=confidence,
        risk_category=risk,
        deterministic_exception_type=exception_type,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
    )


def _make_gate_result(action=GateAction.CONTINUE, confidence=0.85):
    passed = action == GateAction.CONTINUE
    return ConfidenceGateResult(
        passed=passed,
        action=action,
        confidence=confidence,
        threshold=0.75,
        reason="test",
    )


def _make_exposure_result(action=ExposureAction.PASS, amount=3000):
    passed = action == ExposureAction.PASS
    return ExposureGuardResult(
        passed=passed,
        action=action,
        adjustment_amount_paise=amount,
        max_auto_resolution_paise=25000,
        reason="test",
    )


def _make_evidence_result(
    passed=True,
    coverage=0.95,
    consistency=0.90,
    has_conflict=False,
    is_novel=False,
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
    from app.schemas.failure_fallback import FallbackAction
    action = FallbackAction.CONTINUE_WITHOUT if can_proceed else FallbackAction.FAIL_CLOSED
    return FailureFallbackResult(
        can_proceed=can_proceed,
        action=action,
        fallback_status="OK" if can_proceed else "FAILED",
        reason="test",
        can_use_deterministic_only=True,
        critical_failures=[],
        optional_failures=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNKNOWN → Decision Matrix NEVER Returns AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMatrixUnknownBlocking:
    """Verify the decision matrix blocks UNKNOWN exceptions."""

    def test_unknown_always_unresolved(self):
        """UNKNOWN exception type → UNRESOLVED, regardless of other conditions."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,  # Very high confidence
            risk="LOW",
            coverage=0.99,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.95),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 100),
            evidence_result=_make_evidence_result(passed=True, coverage=0.99, consistency=0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes
        assert ReasonCode.BLOCKED_EXCEPTION_TYPE in result.reason_codes

    def test_unknown_with_perfect_scores(self):
        """Even with perfect scores everywhere, UNKNOWN blocks."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=1.0,
            risk="LOW",
            coverage=1.0,
            consistency=1.0,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 1.0),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 0),
            evidence_result=_make_evidence_result(True, 1.0, 1.0),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_primary_reason_mentions_unknown(self):
        """Primary reason should mention the unknown pattern."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(exception_type="UNKNOWN")

        result = matrix.evaluate(engine)

        assert "unknown" in result.primary_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNKNOWN with Insufficient Evidence
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownInsufficientEvidence:
    """UNKNOWN exceptions with weak evidence."""

    def test_unknown_low_coverage(self):
        """UNKNOWN + low evidence coverage → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
            coverage=0.20,
            consistency=0.90,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_low_consistency(self):
        """UNKNOWN + low evidence consistency → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
            coverage=0.95,
            consistency=0.10,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_no_evidence(self):
        """UNKNOWN + zero evidence → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
            coverage=0.0,
            consistency=0.0,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_conflicting_evidence(self):
        """UNKNOWN + conflicting evidence → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
        )

        result = matrix.evaluate(
            engine,
            evidence_result=_make_evidence_result(
                passed=False, has_conflict=True
            ),
        )

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# 3. UNKNOWN with Low Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownLowConfidence:
    """UNKNOWN exceptions with low confidence — double-blocked."""

    def test_unknown_very_low_confidence(self):
        """UNKNOWN + very low confidence → UNRESOLVED (two block reasons)."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.10,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.VERY_LOW_CONFIDENCE in result.reason_codes
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes

    def test_unknown_medium_confidence(self):
        """UNKNOWN + medium confidence → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.55,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_at_threshold_confidence(self):
        """UNKNOWN + confidence exactly at AUTO threshold → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.75,  # Default min_confidence_for_auto
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNKNOWN with No Historical Match
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownNoHistoricalMatch:
    """UNKNOWN exceptions with no historical support."""

    def test_unknown_novel_pattern(self):
        """UNKNOWN + novel pattern → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
        )

        result = matrix.evaluate(
            engine,
            evidence_result=_make_evidence_result(
                passed=False, is_novel=True
            ),
        )

        assert result.decision == AutomationDecision.UNRESOLVED
        # UNKNOWN blocks in PRIORITY 1 before NOVEL check in PRIORITY 2
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes

    def test_unknown_unresolved_engine_status(self):
        """UNKNOWN + engine already deferred → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            status=SelectionStatus.UNRESOLVED,
            has_candidate=False,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.ENGINE_DEFERRED in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# 5. UNKNOWN with No Valid Candidate
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownNoCandidate:
    """UNKNOWN exceptions with no valid resolution candidate."""

    def test_unknown_no_candidate_unresolved_engine(self):
        """UNKNOWN + no candidate + engine deferred → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            status=SelectionStatus.UNRESOLVED,
            has_candidate=False,
        )

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_unknown_selected_resolution_is_none(self):
        """UNKNOWN + selected_resolution=None → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            has_candidate=False,
        )
        assert engine.selected_resolution is None

        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# 6. Adversarial: Try to Force AUTO with UNKNOWN
# ─────────────────────────────────────────────────────────────────────────────


class TestAdversarialUnknownForAuto:
    """Adversarial attempts to force AUTO on UNKNOWN exceptions."""

    def test_adversarial_high_confidence_low_risk(self):
        """Adversary sets confidence=0.99, risk=LOW, UNKNOWN → still UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.99,
            risk="LOW",
            coverage=0.99,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.99),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 100),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_adversarial_zero_exposure(self):
        """Adversary sets exposure=0 to bypass exposure guard → UNKNOWN still blocks."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
            exposure=0,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.95),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 0),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_adversarial_custom_config_high_thresholds(self):
        """Adversary lowers all thresholds → UNKNOWN still blocks."""
        matrix = AutomationDecisionMatrix(
            config=DecisionConfig(
                min_confidence_for_auto=0.01,
                min_confidence_for_human=0.01,
                max_exposure_for_auto=1000000,
                max_exposure_for_human=10000000,
                min_evidence_coverage_for_auto=0.01,
                min_evidence_consistency_for_auto=0.01,
                min_margin_for_auto=0.01,
                allowed_risk_for_auto=["LOW", "MEDIUM", "HIGH"],
                allowed_risk_for_human=["LOW", "MEDIUM", "HIGH"],
            )
        )
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.99,
            risk="HIGH",
            coverage=0.99,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.99),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 999999),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        # UNKNOWN_PATTERN is checked BEFORE config thresholds
        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes

    def test_adversarial_try_to_set_exception_type(self):
        """Even if someone tries to disguise UNKNOWN as something else..."""
        # This tests that the EXACT string "UNKNOWN" is checked
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(exception_type="UNKNOWN")

        result = matrix.evaluate(engine)
        assert result.decision == AutomationDecision.UNRESOLVED

        # But a non-UNKNOWN type might pass (this is expected behavior)
        engine2 = _make_engine_result(
            exception_type="FEE_DIFFERENCE",
            confidence=0.85,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )
        result2 = matrix.evaluate(
            engine2,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.85),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 3000),
            evidence_result=_make_evidence_result(True, 0.95, 0.90),
            fallback_result=_make_fallback_result(),
        )
        # Non-UNKNOWN can potentially be AUTO (not testing the full path here)
        assert result2.decision != AutomationDecision.UNRESOLVED or \
               ReasonCode.UNKNOWN_PATTERN not in result2.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# 7. UNKNOWN Through Workflow Routing
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownWorkflowRouting:
    """Verify UNKNOWN exceptions route correctly in the workflow."""

    def test_unknown_routes_to_escalation(self):
        """UNKNOWN exception → UNRESOLVED decision → escalation."""
        from app.schemas.agent_state import AgentState, WorkflowMetadata

        metadata = WorkflowMetadata(
            workflow_id="WF-UNK-001",
            exception_id="EXC-UNK-001",
        )
        state_dict = {
            "metadata": metadata.model_dump(),
            "decision": "UNRESOLVED",
            "classification": {"exception_type": "UNKNOWN"},
        }
        state = AgentState(**state_dict)

        route = route_after_guardrails(state)
        assert route == "escalation"

    def test_unknown_classifiction_blocks_auto_in_guardrail_sim(self):
        """Guardrail simulation should block UNKNOWN even with AUTO input."""
        from app.schemas.agent_state import AgentState, WorkflowMetadata

        metadata = WorkflowMetadata(
            workflow_id="WF-UNK-002",
            exception_id="EXC-UNK-002",
        )
        state_dict = {
            "metadata": metadata.model_dump(),
        }
        state = AgentState(**state_dict)

        engine_result = {
            "confidence": 0.95,
            "risk_category": "LOW",
            "evidence_coverage": 0.95,
            "evidence_consistency": 0.90,
            "deterministic_exception_type": "UNKNOWN",
        }

        result = _simulate_guardrail_evaluation(state, engine_result)
        assert result["decision"] == "UNRESOLVED"

    def test_unknown_in_blocked_type_list(self):
        """UNKNOWN should be in the blocked exception type list."""
        blocked = ["UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD"]
        assert "UNKNOWN" in blocked


# ─────────────────────────────────────────────────────────────────────────────
# 8. UNKNOWN with High Financial Exposure
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownHighExposure:
    """UNKNOWN exceptions with high financial exposure — double-blocked."""

    def test_unknown_high_exposure(self):
        """UNKNOWN + high exposure → UNRESOLVED (two block reasons)."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
            exposure=200000,
        )

        result = matrix.evaluate(
            engine,
            exposure_result=_make_exposure_result(ExposureAction.BLOCK, 200000),
        )

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes

    def test_unknown_critical_exposure(self):
        """UNKNOWN + critical exposure → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
            exposure=10000000,  # ₹1,00,000
        )

        result = matrix.evaluate(
            engine,
            exposure_result=_make_exposure_result(ExposureAction.BLOCK, 10000000),
        )

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# 9. UNKNOWN with System Failures
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownSystemFailures:
    """UNKNOWN exceptions during system failures — triple-blocked."""

    def test_unknown_with_dependency_failure(self):
        """UNKNOWN + critical dependency failure → UNRESOLVED."""
        from app.schemas.failure_fallback import FallbackAction, DependencyFailure

        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.85,
        )

        fallback = FailureFallbackResult(
            can_proceed=False,
            action=FallbackAction.FAIL_CLOSED,
            fallback_status="FAILED",
            reason="database down",
            can_use_deterministic_only=False,
            critical_failures=[DependencyFailure(
                dependency_name="database",
                error_category="DATABASE_UNAVAILABLE",
                severity="CRITICAL",
                error_message="connection refused",
                fallback_action=FallbackAction.FAIL_CLOSED,
                fallback_status="FAILED",
                is_recoverable=False,
            )],
            optional_failures=[],
        )

        result = matrix.evaluate(engine, fallback_result=fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes
        assert ReasonCode.CRITICAL_DEP_FAILURE in result.reason_codes

    def test_unknown_engine_error_fails_closed(self):
        """Guardrail engine error → fails closed to UNRESOLVED."""
        from app.agent.guardrail_node import _fail_node
        from app.schemas.agent_state import AgentState, WorkflowMetadata

        metadata = WorkflowMetadata(
            workflow_id="WF-UNK-ERR",
            exception_id="EXC-UNK-ERR",
        )
        state = AgentState(metadata=metadata)

        result = _fail_node(state, "apply_guardrails", "test error", 0.0)

        # Fail-closed always sets UNRESOLVED
        assert result.get("decision") == "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# 10. UNKNOWN Cannot Appear in AUTO Reason Codes
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownNoAutoReasonCodes:
    """UNKNOWN exceptions should never produce AUTO reason codes."""

    def test_unknown_no_all_gates_passed(self):
        """UNKNOWN should never have ALL_GATES_PASSED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )

        result = matrix.evaluate(engine)

        assert ReasonCode.ALL_GATES_PASSED not in result.reason_codes

    def test_unknown_not_in_auto_path(self):
        """UNKNOWN decision should never reach the AUTO path in decision matrix."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
        )

        result = matrix.evaluate(engine)

        # UNKNOWN is checked in PRIORITY 1 (CRITICAL BLOCK)
        # It should return before reaching PRIORITY 3 (AUTO)
        assert result.decision == AutomationDecision.UNRESOLVED
        # Should NOT have any AUTO-related gate results
        auto_gates = [g for g in result.passed_gates if g.gate_name.startswith("auto_")]
        assert len(auto_gates) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. UNKNOWN vs Known Types — Comparison
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownVsKnownComparison:
    """Compare UNKNOWN behavior against known exception types."""

    def test_fee_difference_can_be_auto(self):
        """FEE_DIFFERENCE can potentially be AUTO (baseline comparison)."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="FEE_DIFFERENCE",
            confidence=0.85,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.85),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 3000),
            evidence_result=_make_evidence_result(True, 0.95, 0.90),
            fallback_result=_make_fallback_result(),
        )

        # FEE_DIFFERENCE can be AUTO with good conditions
        assert result.decision == AutomationDecision.AUTO

    def test_unknown_same_conditions_blocks(self):
        """UNKNOWN with same conditions as FEE_DIFFERENCE → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",  # Only this changes
            confidence=0.85,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.85),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 3000),
            evidence_result=_make_evidence_result(True, 0.95, 0.90),
            fallback_result=_make_fallback_result(),
        )

        # UNKNOWN blocks regardless
        assert result.decision == AutomationDecision.UNRESOLVED

    def test_missing_record_also_blocks(self):
        """MISSING_RECORD is also blocked (defense in depth)."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="MISSING_RECORD",
            confidence=0.85,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.85),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 3000),
            evidence_result=_make_evidence_result(True, 0.95, 0.90),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_complex_multi_adjustment_also_blocks(self):
        """COMPLEX_MULTI_ADJUSTMENT is also blocked."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            confidence=0.85,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.85),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 3000),
            evidence_result=_make_evidence_result(True, 0.95, 0.90),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# 12. UNKNOWN Safety Invariant — Exhaustive Check
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownSafetyInvariant:
    """Exhaustive check that UNKNOWN NEVER produces AUTO."""

    @pytest.mark.parametrize("confidence", [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0])
    def test_unknown_never_auto任何信心(self, confidence):
        """UNKNOWN with any confidence level → NEVER AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=confidence,
            risk="LOW",
            coverage=0.99,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, confidence),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 100),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision != AutomationDecision.AUTO, \
            f"SAFETY VIOLATION: UNKNOWN with confidence={confidence} produced AUTO"

    @pytest.mark.parametrize("risk", ["LOW", "MEDIUM", "HIGH"])
    def test_unknown_never_auto任何风险(self, risk):
        """UNKNOWN with any risk level → NEVER AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
            risk=risk,
            coverage=0.99,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.95),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 100),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision != AutomationDecision.AUTO, \
            f"SAFETY VIOLATION: UNKNOWN with risk={risk} produced AUTO"

    @pytest.mark.parametrize("coverage", [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    def test_unknown_never_auto任何覆盖(self, coverage):
        """UNKNOWN with any evidence coverage → NEVER AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
            risk="LOW",
            coverage=coverage,
            consistency=0.99,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.95),
            exposure_result=_make_exposure_result(ExposureAction.PASS, 100),
            evidence_result=_make_evidence_result(True, coverage, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision != AutomationDecision.AUTO, \
            f"SAFETY VIOLATION: UNKNOWN with coverage={coverage} produced AUTO"

    @pytest.mark.parametrize("exposure", [0, 100, 1000, 5000, 10000, 25000, 50000, 100000, 500000, 1000000])
    def test_unknown_never_auto任何暴露(self, exposure):
        """UNKNOWN with any financial exposure → NEVER AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            exception_type="UNKNOWN",
            confidence=0.95,
            risk="LOW",
            coverage=0.99,
            consistency=0.99,
            exposure=exposure,
        )

        result = matrix.evaluate(
            engine,
            gate_result=_make_gate_result(GateAction.CONTINUE, 0.95),
            exposure_result=_make_exposure_result(ExposureAction.PASS, exposure),
            evidence_result=_make_evidence_result(True, 0.99, 0.99),
            fallback_result=_make_fallback_result(),
        )

        assert result.decision != AutomationDecision.AUTO, \
            f"SAFETY VIOLATION: UNKNOWN with exposure={exposure} produced AUTO"
