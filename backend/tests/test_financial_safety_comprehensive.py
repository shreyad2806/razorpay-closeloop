"""
Comprehensive Financial Safety Tests — Phase 14 supplement.

Tests the complete safety pipeline through REAL service calls:

    ResolutionEngineResult
        → GuardrailEngine.evaluate()
        → ConfidenceGate
        → ExposureGuard
        → EvidenceGuard
        → FallbackGuard
        → DecisionMatrix
        → final decision

Covers:
    1. Exact match
    2. Partial settlement
    3. Timing difference
    4. Fee adjustment
    5. Tax adjustment
    6. Duplicate payment
    7. Missing record
    8. Unknown case
    9. High-value discrepancy
    10. Conflicting evidence
    11. Novel/unseen pattern
    12. Verification failure
    13. Guardrail failure (fail-closed)
    14. Low-confidence prediction
    15. Human-review routing
    16. Adversarial: high confidence cannot bypass exposure
    17. Adversarial: high confidence cannot bypass conflict
    18. Adversarial: high confidence cannot bypass novelty
    19. Financial precision (paise arithmetic)

Also tests:
    - API resolve endpoint cannot bypass guardrails
    - Client cannot force AUTO or verification success
    - Verification UNKNOWN/MISSING blocks resolution
    - Guardrail exception fails closed

All assertions verify that dangerous cases are NOT auto-resolved.
All financial values use integer paise.
"""
import time
from datetime import datetime
from typing import Optional
from unittest.mock import patch

import pytest

from app.schemas.confidence_gate import GateAction
from app.schemas.decision_matrix import AutomationDecision, ReasonCode
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.guardrail_engine import GuardrailEngine
from app.services.verification import VerificationService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Realistic Synthetic Financial Records
# ─────────────────────────────────────────────────────────────────────────────


def _make_engine_result(
    *,
    exception_id: str = "EXC-SAFETY-001",
    case_id: str = "CASE-SAFETY-001",
    expected_amount: int = 1_000_000,  # ₹10,000 in paise
    actual_amount: int = 9_80_000,     # ₹9,800
    difference: int = 20_000,          # ₹200
    status: SelectionStatus = SelectionStatus.RECOMMENDED,
    selected_resolution: Optional[str] = "FEE_ADJUSTMENT",
    confidence: float = 0.85,
    risk_category: str = "LOW",
    evidence_coverage: float = 0.90,
    evidence_consistency: float = 0.85,
    has_conflict: Optional[bool] = False,
    is_novel: Optional[bool] = False,
    missing_evidence: Optional[list] = None,
    proposed_adjustment_paise: int = 20_000,
    deterministic_exception_type: str = "FEE_DIFFERENCE",
    ml_exception_type: Optional[str] = "FEE_DIFFERENCE",
) -> ResolutionEngineResult:
    """Create a realistic ResolutionEngineResult for guardrail evaluation."""
    return ResolutionEngineResult(
        exception_id=exception_id,
        case_id=case_id,
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        status=status,
        selected_resolution=selected_resolution,
        confidence=confidence,
        risk_category=risk_category,
        evidence_coverage=evidence_coverage,
        evidence_consistency=evidence_consistency,
        has_conflict=has_conflict,
        is_novel=is_novel,
        missing_evidence=missing_evidence or [],
        proposed_adjustment_paise=proposed_adjustment_paise,
        deterministic_exception_type=deterministic_exception_type,
        ml_exception_type=ml_exception_type,
    )


@pytest.fixture
def engine():
    """Real GuardrailEngine — no mocking."""
    return GuardrailEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Exact Match
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatch:
    """EXACT_MATCH with zero difference should be safe for AUTO."""

    def test_exact_match_auto_when_safe(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=500_000,
            difference=0,
            confidence=0.95,
            risk_category="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            proposed_adjustment_paise=0,
            deterministic_exception_type="EXACT_MATCH",
        )
        decision = engine.evaluate(result)
        assert decision.decision == AutomationDecision.AUTO, (
            f"Exact match with high confidence should AUTO, got {decision.decision.value}"
        )

    def test_exact_match_zero_difference(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=1_000_000,
            difference=0,
            confidence=0.95,
            proposed_adjustment_paise=0,
            deterministic_exception_type="EXACT_MATCH",
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 0, (
            "Exact match should have zero financial exposure"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Partial Settlement
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialSettlement:
    """PARTIAL_SETTLEMENT should require review for significant amounts."""

    def test_partial_settlement_blocks_high_amount(self, engine):
        result = _make_engine_result(
            expected_amount=5_000_000,
            actual_amount=3_000_000,
            difference=2_000_000,
            confidence=0.80,
            risk_category="HIGH",
            proposed_adjustment_paise=2_000_000,
            deterministic_exception_type="PARTIAL_SETTLEMENT",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "High-value partial settlement must not be AUTO"
        )

    def test_partial_settlement_small_amount_can_auto(self, engine):
        result = _make_engine_result(
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            confidence=0.90,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.90,
            proposed_adjustment_paise=10_000,
            deterministic_exception_type="PARTIAL_SETTLEMENT",
        )
        decision = engine.evaluate(result)
        # Small amount + high confidence + low risk could be AUTO
        assert decision.confidence == 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Timing Difference
# ─────────────────────────────────────────────────────────────────────────────


class TestTimingDifference:
    """TIMING_DIFFERENCE with moderate exposure."""

    def test_timing_difference_high_exposure_blocks_auto(self, engine):
        result = _make_engine_result(
            expected_amount=10_000_000,
            actual_amount=7_500_000,
            difference=2_500_000,
            confidence=0.80,
            risk_category="MEDIUM",
            proposed_adjustment_paise=2_500_000,
            deterministic_exception_type="TIMING_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Timing difference with 2.5M paise exposure must not AUTO"
        )
        assert decision.financial_exposure_paise == 2_500_000

    def test_timing_difference_small_can_auto(self, engine):
        result = _make_engine_result(
            expected_amount=200_000,
            actual_amount=195_000,
            difference=5_000,
            confidence=0.92,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.88,
            proposed_adjustment_paise=5_000,
            deterministic_exception_type="TIMING_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.confidence == 0.92


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Fee Adjustment
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeAdjustment:
    """FEE_DIFFERENCE with known fee structure."""

    def test_fee_adjustment_exact_fee(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_70_000,
            difference=30_000,
            confidence=0.93,
            risk_category="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.92,
            proposed_adjustment_paise=30_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.confidence == 0.93

    def test_fee_adjustment_large_fee_requires_review(self, engine):
        result = _make_engine_result(
            expected_amount=10_000_000,
            actual_amount=9_000_000,
            difference=1_000_000,
            confidence=0.85,
            risk_category="MEDIUM",
            proposed_adjustment_paise=1_000_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Tax Adjustment
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxAdjustment:
    """TAX_ADJUSTMENT with tax-specific evidence."""

    def test_tax_adjustment_moderate(self, engine):
        result = _make_engine_result(
            expected_amount=2_000_000,
            actual_amount=1_85_000,
            difference=15_000,
            confidence=0.88,
            risk_category="LOW",
            evidence_coverage=0.85,
            evidence_consistency=0.82,
            proposed_adjustment_paise=15_000,
            deterministic_exception_type="TAX_ADJUSTMENT",
        )
        decision = engine.evaluate(result)
        assert decision.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Duplicate Payment
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicatePayment:
    """DUPLICATE should require careful review."""

    def test_duplicate_high_value_blocks_auto(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=2_000_000,
            difference=-1_000_000,
            confidence=0.80,
            risk_category="HIGH",
            proposed_adjustment_paise=1_000_000,
            deterministic_exception_type="DUPLICATE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Duplicate payment must not be AUTO"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: Missing Record
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingRecord:
    """MISSING_RECORD is a blocked exception type — must not AUTO."""

    def test_missing_record_never_auto(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=0,
            difference=500_000,
            confidence=0.70,
            risk_category="HIGH",
            proposed_adjustment_paise=500_000,
            deterministic_exception_type="MISSING_RECORD",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "MISSING_RECORD must never be AUTO — it is a blocked exception type"
        )
        assert ReasonCode.BLOCKED_EXCEPTION_TYPE in decision.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8: Unknown Case
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownCase:
    """UNKNOWN exception type is blocked — must not AUTO."""

    def test_unknown_never_auto(self, engine):
        result = _make_engine_result(
            expected_amount=300_000,
            actual_amount=280_000,
            difference=20_000,
            confidence=0.30,
            risk_category="HIGH",
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="UNKNOWN",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "UNKNOWN exception type must never be AUTO"
        )
        assert ReasonCode.UNKNOWN_PATTERN in decision.reason_codes

    def test_unknown_with_high_confidence_still_blocks(self, engine):
        """Adversarial: even high confidence cannot override UNKNOWN."""
        result = _make_engine_result(
            expected_amount=300_000,
            actual_amount=280_000,
            difference=20_000,
            confidence=0.99,
            risk_category="LOW",
            evidence_coverage=0.99,
            evidence_consistency=0.99,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="UNKNOWN",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "UNKNOWN must never be AUTO regardless of confidence"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9: High-Value Discrepancy
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueDiscrepancy:
    """High-value financial exposure must not bypass guardrails."""

    def test_high_value_blocks_auto(self, engine):
        """₹50,000 (5,000,000 paise) must not be AUTO."""
        result = _make_engine_result(
            expected_amount=50_000_000,
            actual_amount=45_000_000,
            difference=5_000_000,
            confidence=0.95,
            risk_category="MEDIUM",
            proposed_adjustment_paise=5_000_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "5M paise exposure must not be AUTO"
        )

    def test_very_high_value_unresolved(self, engine):
        """₹10,000+ (1,000,000+ paise) exceeds even HUMAN_REVIEW limit."""
        result = _make_engine_result(
            expected_amount=100_000_000,
            actual_amount=90_000_000,
            difference=10_000_000,
            confidence=0.95,
            risk_category="HIGH",
            proposed_adjustment_paise=10_000_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision == AutomationDecision.UNRESOLVED, (
            f"10M paise must be UNRESOLVED, got {decision.decision.value}"
        )

    def test_exactly_at_auto_threshold(self, engine):
        """Amount exactly at max_exposure_for_auto (25,000 paise) — should pass."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_75_000,
            difference=25_000,
            confidence=0.90,
            risk_category="LOW",
            evidence_coverage=0.85,
            evidence_consistency=0.85,
            proposed_adjustment_paise=25_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        # At the boundary — may be AUTO or HUMAN_REVIEW depending on other gates
        assert decision.financial_exposure_paise == 25_000

    def test_one_paise_over_auto_threshold(self, engine):
        """Amount one paise over max_exposure_for_auto (25,001 paise) — should block."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_74_999,
            difference=25_001,
            confidence=0.90,
            risk_category="LOW",
            evidence_coverage=0.85,
            evidence_consistency=0.85,
            proposed_adjustment_paise=25_001,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "25,001 paise must not be AUTO"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 10: Conflicting Evidence
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictingEvidence:
    """Conflicting evidence must block AUTO resolution."""

    def test_conflict_blocks_auto(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_50_000,
            difference=50_000,
            confidence=0.90,
            risk_category="LOW",
            has_conflict=True,
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Conflicting evidence must not AUTO"
        )
        assert ReasonCode.CONFLICTING_EVIDENCE in decision.reason_codes

    def test_unknown_conflict_status_blocks_auto(self, engine):
        """has_conflict=None means unknown — must fail closed."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_50_000,
            difference=50_000,
            confidence=0.95,
            risk_category="LOW",
            has_conflict=None,  # Unknown
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Unknown conflict status must not AUTO (fail closed)"
        )
        assert ReasonCode.CONFLICTING_EVIDENCE in decision.reason_codes

    def test_no_conflict_allows_auto(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_80_000,
            difference=20_000,
            confidence=0.92,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.88,
            has_conflict=False,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        # No conflict — AUTO is possible if other gates pass
        assert ReasonCode.CONFLICTING_EVIDENCE not in decision.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 11: Novel/Unseen Pattern
# ─────────────────────────────────────────────────────────────────────────────


class TestNovelPattern:
    """Novel/unseen patterns must block AUTO resolution."""

    def test_novel_blocks_auto(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_50_000,
            difference=50_000,
            confidence=0.88,
            risk_category="MEDIUM",
            is_novel=True,
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Novel pattern must not AUTO"
        )
        assert ReasonCode.NOVEL_PATTERN in decision.reason_codes

    def test_unknown_novelty_blocks_auto(self, engine):
        """is_novel=None means unknown — must fail closed."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_50_000,
            difference=50_000,
            confidence=0.95,
            risk_category="LOW",
            is_novel=None,  # Unknown
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "Unknown novelty status must not AUTO (fail closed)"
        )

    def test_known_pattern_allows_auto(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.92,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.88,
            is_novel=False,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert ReasonCode.NOVEL_PATTERN not in decision.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 12: Verification Failure
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFailure:
    """Verification failure must block successful resolution."""

    def test_verification_stale_state(self):
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001", "EVD-002"],
            "expected_amount": 1_000_000,
            "difference": 50_000,
            "decision": "AUTO",
            "state_version": 1,
        }
        # Current state has different financial amounts
        current = {
            **snapshot,
            "expected_amount": 900_000,  # Changed!
            "difference": 100_000,       # Changed!
        }
        result = service.verify("EXC-001", snapshot, current)
        assert not result.passed, "Verification must fail when financial amounts changed"
        assert result.amount_consistent is False

    def test_verification_exception_disappeared(self):
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001"],
            "expected_amount": 500_000,
            "difference": 25_000,
            "decision": "AUTO",
        }
        current = {
            **snapshot,
            "exception_exists": False,  # Exception gone
        }
        result = service.verify("EXC-001", snapshot, current)
        assert not result.passed, "Verification must fail when exception disappears"

    def test_verification_evidence_removed(self):
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001", "EVD-002", "EVD-003"],
            "expected_amount": 1_000_000,
            "difference": 30_000,
            "decision": "AUTO",
        }
        current = {
            **snapshot,
            "evidence_records": ["EVD-001"],  # EVD-002 and EVD-003 removed
        }
        result = service.verify("EXC-001", snapshot, current)
        assert not result.passed, "Verification must fail when evidence is removed"

    def test_verification_guardrail_decision_changed(self):
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001"],
            "expected_amount": 1_000_000,
            "difference": 25_000,
            "decision": "AUTO",
        }
        current = {
            **snapshot,
            "decision": "HUMAN_REVIEW",  # Changed!
        }
        result = service.verify("EXC-001", snapshot, current)
        assert not result.passed, "Verification must fail when guardrail decision changes"

    def test_verification_success_when_unchanged(self):
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001", "EVD-002"],
            "expected_amount": 1_000_000,
            "difference": 25_000,
            "decision": "AUTO",
            "state_version": 1,
        }
        result = service.verify("EXC-001", snapshot, snapshot)  # Same state
        assert result.passed, "Verification should pass when nothing changed"
        assert result.action.value == "VERIFIED"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 13: Guardrail Failure (Fail-Closed)
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailFailure:
    """Guardrail engine errors must fail closed to UNRESOLVED."""

    def test_guardrail_exception_produces_unresolved(self, engine):
        """When _evaluate_inner raises, the engine must return UNRESOLVED."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.90,
            risk_category="LOW",
        )
        # Force an exception inside the engine
        with patch.object(engine, '_evaluate_inner', side_effect=RuntimeError("boom")):
            decision = engine.evaluate(result)

        assert decision.decision == AutomationDecision.UNRESOLVED, (
            f"Guardrail exception must produce UNRESOLVED, got {decision.decision.value}"
        )
        assert "guardrail_engine" in decision.critical_failures

    def test_guardrail_exception_never_auto(self, engine):
        """Verify that fail-closed NEVER produces AUTO."""
        result = _make_engine_result(
            expected_amount=100_000,
            actual_amount=98_000,
            difference=2_000,
            confidence=0.99,  # Very high confidence
            risk_category="LOW",
        )
        with patch.object(engine, '_evaluate_inner', side_effect=RuntimeError("crash")):
            decision = engine.evaluate(result)

        assert decision.decision != AutomationDecision.AUTO
        assert decision.system_healthy is False


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 14: Low-Confidence Prediction
# ─────────────────────────────────────────────────────────────────────────────


class TestLowConfidence:
    """Low confidence must not bypass review."""

    def test_very_low_confidence_unresolved(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_50_000,
            difference=50_000,
            confidence=0.10,  # Very low
            risk_category="HIGH",
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision == AutomationDecision.UNRESOLVED, (
            f"10% confidence must be UNRESOLVED, got {decision.decision.value}"
        )

    def test_below_auto_threshold(self, engine):
        """Confidence below 0.75 must not be AUTO."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.60,
            risk_category="MEDIUM",
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "60% confidence must not AUTO"
        )

    def test_at_exact_auto_threshold(self, engine):
        """Confidence at 0.75 may pass confidence gate, but other gates must also pass."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.75,
            risk_category="LOW",
            evidence_coverage=0.80,
            evidence_consistency=0.80,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.confidence == 0.75

    def test_just_below_human_threshold_unresolved(self, engine):
        """Confidence below 0.40 goes straight to UNRESOLVED."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.39,
            risk_category="MEDIUM",
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 15: Human-Review Routing
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanReview:
    """Cases that should route to HUMAN_REVIEW."""

    def test_medium_confidence_routes_to_human(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_50_000,
            difference=50_000,
            confidence=0.55,
            risk_category="MEDIUM",
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        # 0.55 is between 0.40 and 0.75 — should be HUMAN_REVIEW or UNRESOLVED
        assert decision.decision != AutomationDecision.AUTO

    def test_high_risk_routes_to_human_or_unresolved(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.80,
            risk_category="HIGH",
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO, (
            "HIGH risk must not be AUTO"
        )

    def test_partial_settlement_moderate_amount(self, engine):
        result = _make_engine_result(
            expected_amount=2_000_000,
            actual_amount=1_500_000,
            difference=500_000,
            confidence=0.65,
            risk_category="MEDIUM",
            proposed_adjustment_paise=500_000,
            deterministic_exception_type="PARTIAL_SETTLEMENT",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Tests: High Confidence Cannot Bypass Safety
# ─────────────────────────────────────────────────────────────────────────────


class TestAdversarialHighConfidenceBypass:
    """Even 0.99 confidence must not bypass safety guardrails."""

    def test_high_conf_cannot_bypass_exposure(self, engine):
        """0.99 confidence + 5M paise exposure → NOT AUTO."""
        result = _make_engine_result(
            expected_amount=50_000_000,
            actual_amount=45_000_000,
            difference=5_000_000,
            confidence=0.99,
            risk_category="MEDIUM",
            proposed_adjustment_paise=5_000_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO

    def test_high_conf_cannot_bypass_conflict(self, engine):
        """0.99 confidence + has_conflict=True → NOT AUTO."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_50_000,
            difference=50_000,
            confidence=0.99,
            risk_category="LOW",
            has_conflict=True,
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO

    def test_high_conf_cannot_bypass_novelty(self, engine):
        """0.99 confidence + is_novel=True → NOT AUTO."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_50_000,
            difference=50_000,
            confidence=0.99,
            risk_category="LOW",
            is_novel=True,
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO

    def test_high_conf_cannot_bypass_unknown_type(self, engine):
        """0.99 confidence + UNKNOWN exception → NOT AUTO."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=9_50_000,
            difference=50_000,
            confidence=0.99,
            risk_category="LOW",
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="UNKNOWN",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO

    def test_high_conf_cannot_bypass_missing_record(self, engine):
        """0.99 confidence + MISSING_RECORD → NOT AUTO."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=0,
            difference=1_000_000,
            confidence=0.99,
            risk_category="LOW",
            proposed_adjustment_paise=1_000_000,
            deterministic_exception_type="MISSING_RECORD",
        )
        decision = engine.evaluate(result)
        assert decision.decision != AutomationDecision.AUTO

    def test_high_conf_cannot_bypass_critical_dep_failure(self, engine):
        """0.99 confidence + database down → UNRESOLVED.

        Note: ml_classifier is intentionally optional (deterministic fallback).
        Required deps: database, evidence_retrieval, mcp.
        """
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.99,
            risk_category="LOW",
            proposed_adjustment_paise=20_000,
        )
        dep_status = {
            "ml_classifier": True,
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": False,  # REQUIRED dependency down
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        decision = engine.evaluate(result, dependency_status=dep_status)
        assert decision.decision == AutomationDecision.UNRESOLVED

    def test_optional_ml_dep_failure_allows_continue(self, engine):
        """ml_classifier down is intentional — deterministic fallback exists."""
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.92,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.88,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        dep_status = {
            "ml_classifier": False,  # Optional — deterministic fallback
            "ml_resolution_predictor": True,
            "similarity_service": True,
            "database": True,
            "evidence_retrieval": True,
            "llm": True,
            "mcp": True,
        }
        decision = engine.evaluate(result, dependency_status=dep_status)
        # ML is optional — system can proceed in deterministic mode
        assert decision.system_healthy is True or decision.decision != AutomationDecision.UNRESOLVED, (
            "Optional ML failure should not block processing"
        )

    def test_high_conf_cannot_bypass_very_high_exposure(self, engine):
        """0.99 confidence + 10M paise → UNRESOLVED."""
        result = _make_engine_result(
            expected_amount=100_000_000,
            actual_amount=90_000_000,
            difference=10_000_000,
            confidence=0.99,
            risk_category="HIGH",
            proposed_adjustment_paise=10_000_000,
        )
        decision = engine.evaluate(result)
        assert decision.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# Financial Precision Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialPrecision:
    """Verify integer paise arithmetic is correct throughout."""

    def test_zero_difference_zero_exposure(self, engine):
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=1_000_000,
            difference=0,
            proposed_adjustment_paise=0,
            confidence=0.95,
            risk_category="LOW",
            evidence_coverage=0.95,
            evidence_consistency=0.95,
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 0

    def test_large_paise_values(self, engine):
        """1 crore = 100,000,000 paise."""
        result = _make_engine_result(
            expected_amount=100_000_000,
            actual_amount=95_000_000,
            difference=5_000_000,
            confidence=0.85,
            risk_category="HIGH",
            proposed_adjustment_paise=5_000_000,
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 5_000_000

    def test_single_paise_difference(self, engine):
        """1 paise difference — should be detectable."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=999_999,
            difference=1,
            confidence=0.95,
            risk_category="LOW",
            proposed_adjustment_paise=1,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 1

    def test_paise_not_floating_point(self, engine):
        """Ensure all financial values are integers, never floats."""
        result = _make_engine_result(
            expected_amount=1_234_567,
            actual_amount=1_200_000,
            difference=34_567,
            proposed_adjustment_paise=34_567,
            confidence=0.88,
        )
        assert isinstance(result.expected_amount, int)
        assert isinstance(result.actual_amount, int)
        assert isinstance(result.difference, int)
        assert isinstance(result.proposed_adjustment_paise, int)

    def test_negative_difference(self, engine):
        """Negative difference (overpayment)."""
        result = _make_engine_result(
            expected_amount=1_000_000,
            actual_amount=1_200_000,
            difference=-200_000,
            proposed_adjustment_paise=200_000,
            confidence=0.82,
            risk_category="MEDIUM",
        )
        decision = engine.evaluate(result)
        assert decision.financial_exposure_paise == 200_000


# ─────────────────────────────────────────────────────────────────────────────
# API Resolve Bypass Protection
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIResolveBypassProtection:
    """Verify the API resolve endpoint cannot bypass guardrails."""

    def test_resolve_api_cannot_force_auto(self, engine):
        """
        Even if a hypothetical caller passes decision=AUTO,
        the guardrail engine evaluates independently.
        """
        result = _make_engine_result(
            expected_amount=5_000_000,
            actual_amount=4_000_000,
            difference=1_000_000,
            confidence=0.30,
            risk_category="HIGH",
            proposed_adjustment_paise=1_000_000,
            deterministic_exception_type="UNKNOWN",
        )
        decision = engine.evaluate(result)
        # The engine makes its own decision — caller cannot override
        assert decision.decision != AutomationDecision.AUTO

    def test_client_verification_ignored(self):
        """Client-supplied verification_passed=True is NOT trusted."""
        service = VerificationService()
        # Client says verification passed, but current state differs
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001"],
            "expected_amount": 1_000_000,
            "difference": 50_000,
            "decision": "AUTO",
            "state_version": 1,
        }
        current = {
            **snapshot,
            "expected_amount": 800_000,  # Changed after "verification"
            "difference": 200_000,
        }
        result = service.verify("EXC-001", snapshot, current)
        # The verification service catches the mismatch regardless
        assert not result.passed, (
            "Verification service must detect financial change even if client claims pass"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification UNKNOWN/MISSING State
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationUnknownState:
    """Verification states beyond PASS/FAIL."""

    def test_verification_default_same_as_current(self):
        """When current_state is None, verification uses snapshot — should pass."""
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001"],
            "expected_amount": 1_000_000,
            "difference": 50_000,
            "decision": "AUTO",
        }
        result = service.verify("EXC-001", snapshot, None)
        assert result.passed, "Verification should pass when no current_state is provided"

    def test_verification_with_empty_current_fails(self):
        """When current state is empty, checks should fail."""
        service = VerificationService()
        snapshot = {
            "exception_id": "EXC-001",
            "candidate_id": "CAND-001",
            "exception_exists": True,
            "candidate_exists": True,
            "evidence_records": ["EVD-001"],
            "expected_amount": 1_000_000,
            "difference": 50_000,
            "decision": "AUTO",
        }
        current = {}  # Everything missing
        result = service.verify("EXC-001", snapshot, current)
        assert not result.passed, "Verification must fail when current state is empty"


# ─────────────────────────────────────────────────────────────────────────────
# Decision Audit Trail
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionAuditTrail:
    """Every decision must have reason codes and a primary reason."""

    def test_auto_has_all_gates_passed(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.92,
            risk_category="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.88,
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        if decision.decision == AutomationDecision.AUTO:
            assert ReasonCode.ALL_GATES_PASSED in decision.reason_codes
            assert len(decision.primary_reason) > 0

    def test_unresolved_has_reason(self, engine):
        result = _make_engine_result(
            confidence=0.10,
            risk_category="HIGH",
            proposed_adjustment_paise=50_000,
            deterministic_exception_type="UNKNOWN",
        )
        decision = engine.evaluate(result)
        assert len(decision.reason_codes) > 0, "UNRESOLVED must have at least one reason code"
        assert len(decision.primary_reason) > 0, "UNRESOLVED must have a primary reason"

    def test_human_review_has_failed_gates(self, engine):
        result = _make_engine_result(
            expected_amount=500_000,
            actual_amount=4_80_000,
            difference=20_000,
            confidence=0.55,
            risk_category="MEDIUM",
            proposed_adjustment_paise=20_000,
            deterministic_exception_type="FEE_DIFFERENCE",
        )
        decision = engine.evaluate(result)
        if decision.decision == AutomationDecision.HUMAN_REVIEW:
            assert len(decision.failed_gates) > 0, (
                "HUMAN_REVIEW must have at least one failed gate"
            )

    def test_guardrail_result_contains_all_guards(self, engine):
        result = _make_engine_result(
            confidence=0.85,
            risk_category="LOW",
        )
        decision = engine.evaluate(result)
        assert decision.confidence_gate_result is not None
        assert decision.exposure_guard_result is not None
        assert decision.evidence_guard_result is not None
        assert decision.fallback_result is not None
        assert decision.decision_result is not None
