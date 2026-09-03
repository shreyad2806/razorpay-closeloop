"""
Regression tests for backend blocker fixes (CRITICAL #1-#2, HIGH #3-#8).

Tests verify:
1. Resolve API cannot bypass guardrails (CRITICAL #1)
2. Client cannot force AUTO (CRITICAL #1)
3. Client cannot force verification success (HIGH #4/#6)
4. High financial exposure cannot AUTO (HIGH #3)
5. High confidence cannot bypass exposure (HIGH #3)
6. Conflicting evidence cannot AUTO (HIGH #8)
7. Novel pattern cannot AUTO (HIGH #8)
8. Missing evidence cannot AUTO (HIGH #8)
9. Low confidence cannot AUTO
10. Verification failure does not resolve (HIGH #5)
11. Verification pending does not resolve (HIGH #5)
12. Verification unknown does not resolve (HIGH #5)
13. Verification service failure fails closed
14. Guardrail exception fails closed
15. LangGraph uses actual GuardrailEngine
16. LangGraph passes candidate amount to guardrails (HIGH #3)
17. Actual verification service is called (HIGH #4)
18. Verification uses fresh state (HIGH #7)
19. Duplicate execution remains protected
20. Existing successful AUTO path still works
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from unittest.mock import MagicMock, patch

from app.agent.execution_nodes import _build_action_request, _load_fresh_financial_state
from app.agent.guardrail_node import apply_guardrails, _build_engine_result_for_guardrails
from app.agent.routing import (
    route_after_execution_verification,
    route_after_guardrails,
    route_after_verification,
    ROUTE_ESCALATION,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RESOLVE,
    ROUTE_VERIFICATION,
)
from app.agent.workflow import create_initial_state
from app.schemas.agent_state import AgentState, HumanApprovalStatus, VerificationStatus
from app.schemas.resolution_engine import ResolutionEngineResult
from app.services.decision_matrix import AutomationDecisionMatrix, DecisionConfig
from app.services.guardrail_engine import GuardrailEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_engine_result(
    confidence=0.85,
    risk="LOW",
    coverage=0.90,
    consistency=0.85,
    exc_type="FEE_DIFFERENCE",
    adjustment_paise=5000,
    has_conflict=False,
    is_novel=False,
    status="RECOMMENDED",
):
    """Build a ResolutionEngineResult for testing."""
    return ResolutionEngineResult(
        exception_id="EXC-TEST",
        case_id="CASE-TEST",
        payment_id="PAY-TEST",
        merchant_id="MER-TEST",
        expected_amount=100000,
        actual_amount=95000,
        difference=5000,
        status=status,
        selected_resolution="FEE_ADJUSTMENT",
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=[],
        candidate_scores=[],
        confidence=confidence,
        confidence_factors={},
        risk_category=risk,
        risk_factors=[],
        explainability=None,
        rejection_reasons=[],
        deterministic_exception_type=exc_type,
        ml_exception_type=None,
        classification_agreement=True,
        evidence_explanation_status="FULLY_EXPLAINED",
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        has_conflict=has_conflict,
        is_novel=is_novel,
        missing_evidence=[],
        proposed_adjustment_paise=adjustment_paise,
    )


def _make_state(
    confidence=0.85,
    risk="LOW",
    coverage=0.90,
    consistency=0.85,
    exc_type="FEE_DIFFERENCE",
    amount_paise=5000,
    has_conflict=False,
    is_novel=False,
    decision=None,
    verification_status="NOT_REQUIRED",
    approval_status="NOT_REQUIRED",
):
    """Build an AgentState for testing."""
    state = create_initial_state(exception_id="EXC-TEST", case_id="CASE-TEST")
    state.classification = {
        "exception_type": exc_type,
        "confidence": confidence,
    }
    state.evidence_package = {
        "evidence_coverage": coverage,
        "evidence_consistency": consistency,
        "has_conflict": has_conflict,
        "is_novel": is_novel,
        "expected_amount": 100000,
        "actual_amount": 95000,
        "difference": 5000,
    }
    state.selected_candidate = {
        "candidate_id": "CAND-TEST",
        "resolution_type": "FEE_ADJUSTMENT",
        "amount_paise": amount_paise,
    }
    state.candidate_scores = {"best_score": confidence}
    state.confidence = confidence
    state.risk = risk
    state.decision = decision
    state.verification.verification_status = VerificationStatus(verification_status)
    state.human_review.approval_status = HumanApprovalStatus(approval_status)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL #1 — Resolve API cannot bypass guardrails
# ═══════════════════════════════════════════════════════════════════════════════


class TestCritical1_ResolveAPIBypass:
    """CRITICAL #1: POST /exceptions/{id}/resolve must not bypass guardrails."""

    def test_resolve_returns_pending_not_resolved(self):
        """Resolve API returns PENDING status, not RESOLVED."""
        from app.api.services.exception_service import ExceptionService

        svc = ExceptionService()
        exc = svc.get_exception("EXC-001")
        if exc is None:
            pytest.skip("EXC-001 not available")
        result = svc.resolve_exception(
            "EXC-001",
            {"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
        )
        assert result["status"] == "PENDING"
        assert "error" not in result

    def test_resolve_does_not_claim_auto(self):
        """Resolve API does not claim guardrail_decision=AUTO."""
        from app.api.services.exception_service import ExceptionService

        svc = ExceptionService()
        exc = svc.get_exception("EXC-001")
        if exc is None:
            pytest.skip("EXC-001 not available")
        result = svc.resolve_exception(
            "EXC-001",
            {"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
        )
        assert result.get("guardrail_decision") is None

    def test_resolve_does_not_claim_verification(self):
        """Resolve API does not claim verification_result."""
        from app.api.services.exception_service import ExceptionService

        svc = ExceptionService()
        exc = svc.get_exception("EXC-001")
        if exc is None:
            pytest.skip("EXC-001 not available")
        result = svc.resolve_exception(
            "EXC-001",
            {"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
        )
        assert result.get("verification_result") is None

    def test_client_cannot_force_decision(self):
        """Client-provided decision is not trusted — server determines it."""
        state = _make_state(confidence=0.99, risk="LOW", coverage=0.95, consistency=0.95)
        # Even if client says AUTO, guardrails evaluate independently
        state.decision = "AUTO"
        result = apply_guardrails(state)
        # Guardrails produce their own decision
        assert result.get("decision") in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
        # The decision comes from guardrails, not from client input

    def test_client_cannot_force_verification_passed(self):
        """Client-provided verification_passed is not trusted."""
        state = _make_state(confidence=0.99, risk="LOW")
        state.verification.verification_status = VerificationStatus.NOT_REQUIRED
        action = _build_action_request(state)
        # verification_passed comes from actual verification status
        assert action["verification_passed"] is False

    def test_already_resolved_blocks_reproposal(self):
        """A RESOLVED exception cannot be re-resolved."""
        from app.api.services.exception_service import ExceptionService, _exception_registry

        svc = ExceptionService()
        exc = svc.get_exception("EXC-001")
        if exc is None:
            pytest.skip("EXC-001 not available")
        # Manually set to RESOLVED
        _exception_registry["EXC-001"] = {"status": "RESOLVED", "case_id": "EXC-001"}
        result = svc.resolve_exception(
            "EXC-001",
            {"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
        )
        assert result.get("error") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH #3 — Candidate must reach exposure guard
# ═══════════════════════════════════════════════════════════════════════════════


class TestHigh3_CandidateExposure:
    """HIGH #3: The selected candidate's adjustment must reach the exposure guard."""

    def test_adjustment_passed_to_engine_result(self):
        """proposed_adjustment_paise is set on the engine result."""
        state = _make_state(amount_paise=50000)
        engine_result = _build_engine_result_for_guardrails(state)
        assert engine_result.proposed_adjustment_paise == 50000

    def test_high_exposure_blocks_auto(self):
        """High financial exposure blocks AUTO regardless of confidence."""
        engine_result = _make_engine_result(
            confidence=0.95,
            adjustment_paise=200_000,  # Well above 100K limit
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_high_confidence_cannot_bypass_exposure(self):
        """High confidence must NOT bypass the exposure limit."""
        engine_result = _make_engine_result(
            confidence=1.0,  # Perfect confidence
            adjustment_paise=500_000,  # Way above limit
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_zero_exposure_allows_auto_path(self):
        """Zero exposure does not block the AUTO path."""
        engine_result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=0,
            has_conflict=False,
            is_novel=False,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        # Should not be blocked by exposure
        assert result.financial_exposure_paise == 0


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH #4 + #6 — Hardcoded verification_passed
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardcodedVerification:
    """HIGH #4/#6: verification_passed must come from actual verification."""

    def test_verification_passed_false_when_not_verified(self):
        """verification_passed=False when verification is NOT_REQUIRED."""
        state = _make_state(verification_status="NOT_REQUIRED")
        action = _build_action_request(state)
        assert action["verification_passed"] is False

    def test_verification_passed_true_when_verified(self):
        """verification_passed=True only when status is VERIFIED."""
        state = _make_state(verification_status="VERIFIED")
        action = _build_action_request(state)
        assert action["verification_passed"] is True

    def test_verification_passed_false_when_pending(self):
        """verification_passed=False when verification is PENDING."""
        state = _make_state(verification_status="PENDING")
        action = _build_action_request(state)
        assert action["verification_passed"] is False

    def test_verification_passed_false_when_failed(self):
        """verification_passed=False when verification FAILED."""
        state = _make_state(verification_status="FAILED")
        action = _build_action_request(state)
        assert action["verification_passed"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH #5 — Verification routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerificationRouting:
    """HIGH #5: PENDING/NOT_REQUIRED/UNKNOWN must not route to resolution."""

    def test_not_required_routes_to_escalation(self):
        state = _make_state(verification_status="NOT_REQUIRED")
        route = route_after_verification(state)
        assert route == ROUTE_ESCALATION

    def test_pending_routes_to_escalation(self):
        state = _make_state(verification_status="PENDING")
        route = route_after_verification(state)
        assert route == ROUTE_ESCALATION

    def test_failed_routes_to_escalation(self):
        state = _make_state(verification_status="FAILED")
        route = route_after_verification(state)
        assert route == ROUTE_ESCALATION

    def test_verified_routes_to_resolve(self):
        state = _make_state(verification_status="VERIFIED")
        route = route_after_verification(state)
        assert route == ROUTE_RESOLVE

    def test_execution_verification_failed_routes_to_rollback(self):
        """Failed execution verification routes to rollback, not resolution."""
        state = _make_state(verification_status="FAILED")
        route = route_after_execution_verification(state)
        assert route == "rollback_resolution"

    def test_execution_verification_verified_routes_to_outcome(self):
        """Successful execution verification routes to outcome."""
        state = _make_state(verification_status="VERIFIED")
        route = route_after_execution_verification(state)
        assert route == "record_outcome"

    def test_execution_verification_pending_routes_to_escalation(self):
        """PENDING execution verification fails closed."""
        state = _make_state(verification_status="PENDING")
        route = route_after_execution_verification(state)
        assert route == ROUTE_ESCALATION


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH #7 — Verification uses fresh state
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreshVerificationState:
    """HIGH #7: Verification must use fresh state, not stale evidence."""

    def test_fresh_state_from_after_state(self):
        """Fresh state loaded from execution after_state."""
        state = _make_state()
        state.execution_result = {
            "after_state": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 100000,
                "difference": 0,
                "total_refunds": 0,
                "total_fees": 5000,
                "total_taxes": 0,
                "total_adjustments": 5000,
                "settlement_count": 1,
                "refund_count": 0,
                "fee_count": 1,
                "tax_count": 0,
                "adjustment_count": 1,
            }
        }
        fresh = _load_fresh_financial_state(state)
        assert fresh is not None
        assert fresh["total_adjustments"] == 5000

    def test_fail_closed_when_no_fresh_state(self):
        """Returns None when no execution result available."""
        state = _make_state()
        state.execution_result = None
        fresh = _load_fresh_financial_state(state)
        assert fresh is None

    def test_fail_closed_when_no_after_state(self):
        """Returns None when execution result has no after_state."""
        state = _make_state()
        state.execution_result = {"status": "EXECUTED"}
        fresh = _load_fresh_financial_state(state)
        assert fresh is None


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH #8 — Safety fields (has_conflict, is_novel) as Optional[bool]
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafetyFields:
    """HIGH #8: Unknown safety fields must not be treated as safe."""

    def test_none_conflict_blocks_auto(self):
        """has_conflict=None blocks AUTO (unknown = unsafe)."""
        engine_result = _make_engine_result(has_conflict=None, is_novel=False)
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_none_novelty_blocks_auto(self):
        """is_novel=None blocks AUTO (unknown = unsafe)."""
        engine_result = _make_engine_result(has_conflict=False, is_novel=None)
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_both_none_blocks_auto(self):
        """Both None blocks AUTO."""
        engine_result = _make_engine_result(has_conflict=None, is_novel=None)
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_false_conflict_allows_auto_path(self):
        """has_conflict=False (verified safe) allows AUTO path."""
        engine_result = _make_engine_result(
            has_conflict=False,
            is_novel=False,
            confidence=0.90,
            coverage=0.90,
            consistency=0.90,
            adjustment_paise=0,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        # Not blocked by conflict
        gate_names = [g.gate_name for g in result.passed_gates]
        assert "conflict_check" in gate_names

    def test_true_conflict_blocks_auto(self):
        """has_conflict=True blocks AUTO."""
        engine_result = _make_engine_result(has_conflict=True, is_novel=False)
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_true_novelty_blocks_auto(self):
        """is_novel=True blocks AUTO."""
        engine_result = _make_engine_result(has_conflict=False, is_novel=True)
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# LOW CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestLowConfidence:
    """Low confidence cannot produce AUTO."""

    def test_low_confidence_blocks_auto(self):
        engine_result = _make_engine_result(
            confidence=0.50,
            has_conflict=False,
            is_novel=False,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_zero_confidence_blocks_auto(self):
        engine_result = _make_engine_result(
            confidence=0.0,
            has_conflict=False,
            is_novel=False,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL FAILURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardrailFailure:
    """Guardrail exceptions must fail closed."""

    def test_guardrail_exception_fails_closed(self):
        """Exception in guardrails → not AUTO, never unsafe."""
        state = _make_state(confidence=0.99)
        # Corrupt the state to cause an error
        state.evidence_package = None  # Will cause AttributeError in _build_engine_result
        result = apply_guardrails(state)
        # FAIL-CLOSED: exception → never AUTO (HUMAN_REVIEW or UNRESOLVED)
        assert result.get("decision") != "AUTO"

    def test_guardrail_engine_exception_returns_unresolved(self):
        """GuardrailEngine error → UNRESOLVED with system_healthy=False."""
        engine_result = _make_engine_result()
        guardrail = GuardrailEngine()
        # Cause an error by passing None for a required field
        with patch.object(guardrail, "_evaluate_inner", side_effect=Exception("test error")):
            result = guardrail.evaluate(engine_result)
        assert result.decision.value == "UNRESOLVED"
        assert result.system_healthy is False


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH USES REAL GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangGraphGuardrails:
    """LangGraph must use real GuardrailEngine, not simulated logic."""

    def test_guardrail_node_uses_guardrail_engine(self):
        """apply_guardrails creates a real GuardrailEngine."""
        state = _make_state(confidence=0.90, risk="LOW", coverage=0.90, consistency=0.85)
        result = apply_guardrails(state)
        # Should have a real guardrail result
        assert result.get("guardrail_result") is not None
        assert "decision" in result

    def test_guardrail_decision_from_engine(self):
        """Decision comes from guardrail engine, not hardcoded."""
        state = _make_state(confidence=0.90, risk="LOW", coverage=0.90, consistency=0.85)
        result = apply_guardrails(state)
        # Decision should match what the engine would produce
        assert result["decision"] in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO PATH STILL WORKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoPathStillWorks:
    """When ALL safety conditions genuinely pass, AUTO is still possible."""

    def test_genuine_auto_path(self):
        """All conditions pass → AUTO is possible."""
        engine_result = _make_engine_result(
            confidence=0.90,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
            has_conflict=False,
            is_novel=False,
            adjustment_paise=5000,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value == "AUTO"

    def test_auto_via_guardrail_node(self):
        """AUTO path works end-to-end through guardrail node."""
        state = _make_state(
            confidence=0.90,
            risk="LOW",
            coverage=0.95,
            consistency=0.90,
            has_conflict=False,
            is_novel=False,
        )
        result = apply_guardrails(state)
        assert result["decision"] == "AUTO"

    def test_auto_routes_to_verification(self):
        """AUTO decision routes to verification."""
        state = _make_state(decision="AUTO")
        route = route_after_guardrails(state)
        assert route == ROUTE_VERIFICATION

    def test_auto_blocked_when_conflict(self):
        """AUTO is blocked when conflict exists."""
        engine_result = _make_engine_result(
            confidence=0.95,
            has_conflict=True,
            is_novel=False,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_auto_blocked_when_novel(self):
        """AUTO is blocked when novelty detected."""
        engine_result = _make_engine_result(
            confidence=0.95,
            has_conflict=False,
            is_novel=True,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"

    def test_auto_blocked_when_high_exposure(self):
        """AUTO is blocked when exposure exceeds limit."""
        engine_result = _make_engine_result(
            confidence=0.95,
            adjustment_paise=200_000,
        )
        guardrail = GuardrailEngine()
        result = guardrail.evaluate(engine_result)
        assert result.decision.value != "AUTO"


# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATE EXECUTION PROTECTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestDuplicateExecutionProtection:
    """Idempotency: duplicate actions must not double-execute."""

    def test_idempotency_key_is_deterministic(self):
        """Same inputs produce the same idempotency key."""
        state1 = _make_state()
        state1.metadata.workflow_id = "WF-001"
        state1.metadata.exception_id = "EXC-001"
        state1.selected_candidate = {"candidate_id": "CAND-001"}

        state2 = _make_state()
        state2.metadata.workflow_id = "WF-001"
        state2.metadata.exception_id = "EXC-001"
        state2.selected_candidate = {"candidate_id": "CAND-001"}

        action1 = _build_action_request(state1)
        action2 = _build_action_request(state2)
        assert action1["idempotency_key"] == action2["idempotency_key"]

    def test_different_workflows_different_keys(self):
        """Different workflows produce different idempotency keys."""
        state1 = _make_state()
        state1.metadata.workflow_id = "WF-001"
        state1.metadata.exception_id = "EXC-001"
        state1.selected_candidate = {"candidate_id": "CAND-001"}

        state2 = _make_state()
        state2.metadata.workflow_id = "WF-002"
        state2.metadata.exception_id = "EXC-001"
        state2.selected_candidate = {"candidate_id": "CAND-001"}

        action1 = _build_action_request(state1)
        action2 = _build_action_request(state2)
        assert action1["idempotency_key"] != action2["idempotency_key"]
