"""
Tests for ExposureGuard (Phase 6B).

Tests:
- Guard configuration
- Maximum auto-resolution amount
- High-value threshold
- Cumulative exposure
- Blocked exception types
- Blocked resolution types
- Conflict penalty
- Supporting evidence count
- Zero adjustment
- Small adjustment
- Exactly at maximum
- Above maximum
- High confidence + high exposure (hard block)
- Low confidence + low exposure
- UNRESOLVED/HUMAN_REVIEW engine result
- Hard-block verification
"""

import pytest

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.exposure_guard import (
    ExposureAction,
    ExposureBlockReason,
    ExposureCheck,
    ExposureGuardConfig,
    ExposureGuardResult,
)
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import (
    ExplainabilityDetail,
    ExplainabilityLevel,
    SelectionStatus,
)
from app.services.exposure_guard import ExposureGuard


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_score(
    evidence=0.8,
    ml=0.9,
    historical=0.7,
    financial=0.85,
    final=0.82,
    novelty=0.0,
    conflict=0.0,
):
    return CandidateScore(
        evidence_score=evidence,
        ml_score=ml,
        historical_score=historical,
        financial_consistency_score=financial,
        final_score=final,
        novelty_penalty=novelty,
        conflict_penalty=conflict,
        weighted_evidence=evidence * 0.35,
        weighted_ml=ml * 0.20,
        weighted_historical=historical * 0.15,
        weighted_financial=financial * 0.30,
        has_evidence_support=evidence > 0,
        has_ml_support=ml > 0,
        has_historical_support=historical > 0,
        is_novel=novelty > 0,
        has_conflicts=conflict > 0,
    )


def _make_candidate(
    resolution_type="FEE_ADJUSTMENT",
    amount_paise=3000,
    evidence_ids=None,
):
    if evidence_ids is None:
        evidence_ids = ["FEE-001"]
    return ResolutionProposal(
        candidate_id="CAND-EXC-001",
        exception_id="EXC-001",
        case_id="CASE-001",
        resolution_type=resolution_type,
        resolution_description="Fee adjustment",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=amount_paise,
            direction="CREDIT",
            evidence_record_id="FEE-001",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=evidence_ids,
        evidence_compatible=True,
        evidence_coverage=0.95,
        sources=["deterministic_evidence"],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.85,
            evidence_support=0.9,
            ml_support=0.8,
            historical_support=0.7,
        ),
        rationale="Fee explains the discrepancy.",
        is_recommendation_only=True,
    )


def _make_engine_result(
    status=SelectionStatus.RECOMMENDED,
    confidence=0.85,
    risk="LOW",
    selected_resolution="FEE_ADJUSTMENT",
    exception_type="FEE_DIFFERENCE",
    adjustment_paise=3000,
    score=None,
    candidates=None,
):
    if score is None:
        score = _make_score(final=confidence, financial=0.9)
    candidate = None
    if status == SelectionStatus.RECOMMENDED and selected_resolution:
        candidate = _make_candidate(
            resolution_type=selected_resolution,
            amount_paise=adjustment_paise,
        )
    ranked = candidates if candidates is not None else ([candidate] if candidate else [])
    return ResolutionEngineResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        status=status,
        selected_resolution=selected_resolution,
        selected_candidate=candidate,
        selected_score=score if status == SelectionStatus.RECOMMENDED else None,
        ranked_candidates=ranked,
        candidate_scores=[score] if score and status == SelectionStatus.RECOMMENDED else [],
        confidence=confidence,
        risk_category=risk,
        explainability=ExplainabilityDetail(
            level=ExplainabilityLevel.FULLY_EXPLAINABLE,
            has_evidence_trace=True,
            has_financial_trace=True,
            source_count=2,
        ),
        deterministic_exception_type=exception_type,
        classification_agreement=True,
        evidence_explanation_status="FULLY_EXPLAINED",
        evidence_coverage=0.9,
        evidence_consistency=0.9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardConfig:
    """Tests for guard configuration."""

    def test_default_config(self):
        config = ExposureGuardConfig()
        assert config.max_auto_resolution_paise == 50000
        assert config.high_value_threshold_paise == 100000
        assert config.cumulative_exposure_limit_paise == 200000
        assert "UNKNOWN" in config.blocked_exception_types
        assert "COMPLEX_MULTI_ADJUSTMENT" in config.blocked_exception_types
        assert "MISSING_RECORD" in config.blocked_exception_types
        assert "UNKNOWN_UNRESOLVED" in config.blocked_resolution_types
        assert "MISSING_RECORD_ESCALATION" in config.blocked_resolution_types
        assert "MULTI_ADJUSTMENT" in config.blocked_resolution_types

    def test_custom_config(self):
        config = ExposureGuardConfig(
            max_auto_resolution_paise=10000,
            high_value_threshold_paise=50000,
            cumulative_exposure_limit_paise=100000,
        )
        assert config.max_auto_resolution_paise == 10000
        assert config.high_value_threshold_paise == 50000
        assert config.cumulative_exposure_limit_paise == 100000


class TestExposureAction:
    """Tests for ExposureAction enum."""

    def test_values(self):
        assert ExposureAction.PASS.value == "PASS"
        assert ExposureAction.BLOCK.value == "BLOCK"


class TestExposureBlockReason:
    """Tests for ExposureBlockReason enum."""

    def test_values(self):
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT.value == "ABOVE_MAX_AMOUNT"
        assert ExposureBlockReason.HIGH_RISK_CATEGORY.value == "HIGH_RISK_CATEGORY"
        assert ExposureBlockReason.CONFLICTING_CASE.value == "CONFLICTING_CASE"
        assert ExposureBlockReason.NO_EXPOSURE_DATA.value == "NO_EXPOSURE_DATA"


class TestExposureCheck:
    """Tests for ExposureCheck schema."""

    def test_create(self):
        check = ExposureCheck(
            check_name="test",
            passed=True,
            value=100.0,
            threshold=200.0,
            reason="Passed",
        )
        assert check.passed is True
        assert check.block_reason is None


class TestExposureGuardResult:
    """Tests for ExposureGuardResult schema."""

    def test_passed_result(self):
        result = ExposureGuardResult(
            passed=True,
            action=ExposureAction.PASS,
            adjustment_amount_paise=3000,
            max_auto_resolution_paise=50000,
            reason="All checks passed",
        )
        assert result.passed is True
        assert result.action == ExposureAction.PASS
        assert result.guard_version == "1.0.0"

    def test_summary(self):
        result = ExposureGuardResult(
            passed=True,
            action=ExposureAction.PASS,
            adjustment_amount_paise=3000,
            max_auto_resolution_paise=50000,
            reason="OK",
        )
        s = result.summary()
        assert "PASS" in s
        assert "3000" in s


# ─────────────────────────────────────────────────────────────────────────────
# Core Guard Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureGuardEvaluation:
    """Tests for the ExposureGuard evaluation logic."""

    def test_small_adjustment_passes(self):
        """Small adjustment within limits → PASS."""
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=3000)
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.action == ExposureAction.PASS
        assert output.adjustment_amount_paise == 3000

    def test_zero_adjustment_passes(self):
        """Zero adjustment → PASS (nothing to block)."""
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=0)
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.adjustment_amount_paise == 0

    def test_exactly_at_maximum_passes(self):
        """Adjustment exactly at max → PASS (not exceeding)."""
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=50000)
        output = guard.evaluate(result)

        assert output.passed is True

    def test_above_maximum_blocks(self):
        """Adjustment above max → BLOCK."""
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.95, adjustment_paise=50001)
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == ExposureAction.BLOCK
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in output.block_reasons

    def test_far_above_maximum_blocks(self):
        """Very large adjustment → BLOCK."""
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.99, adjustment_paise=500000)
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == ExposureAction.BLOCK

    def test_unresolved_engine_result_passes(self):
        """Engine returned UNRESOLVED → guard passes (no adjustment to evaluate)."""
        guard = ExposureGuard()
        result = _make_engine_result(status=SelectionStatus.UNRESOLVED)
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.adjustment_amount_paise == 0

    def test_human_review_engine_result_passes(self):
        """Engine returned HUMAN_REVIEW → guard passes (no adjustment to evaluate)."""
        guard = ExposureGuard()
        result = _make_engine_result(status=SelectionStatus.HUMAN_REVIEW)
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.adjustment_amount_paise == 0


# ─────────────────────────────────────────────────────────────────────────────
# Hard Block Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHardBlock:
    """Tests for hard-block behavior — exposure cannot be overridden."""

    def test_high_confidence_high_exposure_blocks(self):
        """CRITICAL: High confidence + high exposure → BLOCK.

        This is the core hard-block test. Even with 99% confidence,
        exposure above the maximum MUST block auto-resolution.
        """
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.99,
            adjustment_paise=100000,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == ExposureAction.BLOCK
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in output.block_reasons

    def test_maximum_confidence_exceeds_max_blocks(self):
        """Even 100% confidence cannot override exposure limit."""
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=1.0,
            adjustment_paise=60000,
        )
        output = guard.evaluate(result)

        assert output.passed is False

    def test_low_confidence_low_exposure_passes(self):
        """Low confidence + low exposure → PASS.

        The exposure guard only cares about financial exposure,
        not confidence. (Confidence is handled by the confidence gate.)
        """
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.30,
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.adjustment_amount_paise == 1000


# ─────────────────────────────────────────────────────────────────────────────
# High-Value Threshold Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueThreshold:
    """Tests for high-value threshold (informational flag)."""

    def test_below_high_value(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=50000)
        output = guard.evaluate(result)

        assert output.is_high_value is False

    def test_at_high_value(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=100000)
        output = guard.evaluate(result)

        assert output.is_high_value is True

    def test_above_high_value(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=200000)
        output = guard.evaluate(result)

        assert output.is_high_value is True
        # But it's also above max (50000), so blocked
        assert output.passed is False

    def test_custom_high_value_threshold(self):
        guard = ExposureGuard(ExposureGuardConfig(high_value_threshold_paise=5000))
        result = _make_engine_result(confidence=0.90, adjustment_paise=6000)
        output = guard.evaluate(result)

        assert output.is_high_value is True


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative Exposure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCumulativeExposure:
    """Tests for cumulative exposure across candidates."""

    def test_cumulative_within_limit(self):
        """Multiple candidates within cumulative limit → PASS."""
        guard = ExposureGuard()
        c1 = _make_candidate(amount_paise=10000)
        c2 = _make_candidate(amount_paise=20000)
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=10000,
            candidates=[c1, c2],
        )
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.cumulative_exposure_paise == 30000

    def test_cumulative_exceeds_limit(self):
        """Multiple candidates exceeding cumulative limit → BLOCK."""
        guard = ExposureGuard()
        c1 = _make_candidate(amount_paise=100000)
        c2 = _make_candidate(amount_paise=100000)
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=100000,
            candidates=[c1, c2],
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.cumulative_exposure_paise == 200000
        # Individual check also blocks (100000 > 50000)
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in output.block_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Blocked Exception Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockedExceptionType:
    """Tests for blocked exception types."""

    def test_fee_difference_not_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            exception_type="FEE_DIFFERENCE",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        exc_check = next(c for c in output.checks if c.check_name == "blocked_exception_type")
        assert exc_check.passed is True

    def test_unknown_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.95,
            exception_type="UNKNOWN",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        exc_check = next(c for c in output.checks if c.check_name == "blocked_exception_type")
        assert exc_check.passed is False
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in output.block_reasons
        assert output.passed is False

    def test_complex_multi_adjustment_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in output.block_reasons

    def test_missing_record_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            exception_type="MISSING_RECORD",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is False

    def test_custom_blocked_list(self):
        guard = ExposureGuard(ExposureGuardConfig(blocked_exception_types=["FEE_DIFFERENCE"]))
        result = _make_engine_result(
            confidence=0.90,
            exception_type="FEE_DIFFERENCE",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Blocked Resolution Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockedResolutionType:
    """Tests for blocked resolution types."""

    def test_fee_adjustment_not_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            selected_resolution="FEE_ADJUSTMENT",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        res_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert res_check.passed is True

    def test_unknown_unresolved_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.95,
            selected_resolution="UNKNOWN_UNRESOLVED",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        res_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert res_check.passed is False
        assert output.passed is False

    def test_missing_record_escalation_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            selected_resolution="MISSING_RECORD_ESCALATION",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is False

    def test_multi_adjustment_blocked(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            selected_resolution="MULTI_ADJUSTMENT",
            adjustment_paise=1000,
        )
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Penalty Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictPenalty:
    """Tests for conflict penalty check."""

    def test_no_conflict_passes(self):
        guard = ExposureGuard()
        score = _make_score(conflict=0.0, final=0.90)
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=1000,
            score=score,
        )
        output = guard.evaluate(result)

        conflict_check = next(c for c in output.checks if c.check_name == "conflict_penalty")
        assert conflict_check.passed is True

    def test_high_conflict_blocks(self):
        guard = ExposureGuard()
        score = _make_score(conflict=0.20, final=0.75)
        result = _make_engine_result(
            confidence=0.75,
            adjustment_paise=1000,
            score=score,
        )
        output = guard.evaluate(result)

        conflict_check = next(c for c in output.checks if c.check_name == "conflict_penalty")
        assert conflict_check.passed is False
        assert ExposureBlockReason.CONFLICTING_CASE in output.block_reasons
        assert output.passed is False

    def test_custom_conflict_threshold(self):
        guard = ExposureGuard(ExposureGuardConfig(max_conflict_for_auto=0.05))
        score = _make_score(conflict=0.10, final=0.80)
        result = _make_engine_result(
            confidence=0.80,
            adjustment_paise=1000,
            score=score,
        )
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Supporting Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSupportingEvidence:
    """Tests for supporting evidence count check."""

    def test_evidence_present_passes(self):
        guard = ExposureGuard()
        candidate = _make_candidate(evidence_ids=["FEE-001", "FEE-002"])
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        result.selected_candidate = candidate
        output = guard.evaluate(result)

        ev_check = next(c for c in output.checks if c.check_name == "insufficient_evidence")
        assert ev_check.passed is True

    def test_no_evidence_blocks(self):
        guard = ExposureGuard()
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        result.selected_candidate = candidate
        output = guard.evaluate(result)

        ev_check = next(c for c in output.checks if c.check_name == "insufficient_evidence")
        assert ev_check.passed is False
        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiFailure:
    """Tests for multiple simultaneous failures."""

    def test_amount_and_exception_type_fail(self):
        """Both amount exceeds max AND exception type is blocked."""
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=100000,
            exception_type="UNKNOWN",
        )
        output = guard.evaluate(result)

        assert output.passed is False
        failed_checks = [c for c in output.checks if not c.passed]
        assert len(failed_checks) >= 2
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in output.block_reasons
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in output.block_reasons

    def test_all_default_blocks_active(self):
        """All default blocked types + high amount → multiple blocks."""
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.99,
            adjustment_paise=200000,
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            selected_resolution="MULTI_ADJUSTMENT",
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert len(output.block_reasons) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Override Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurationOverrides:
    """Tests for configurable thresholds."""

    def test_strict_threshold_blocks_more(self):
        """Lower max blocks more cases."""
        strict = ExposureGuard(ExposureGuardConfig(max_auto_resolution_paise=5000))
        relaxed = ExposureGuard(ExposureGuardConfig(max_auto_resolution_paise=100000))
        result = _make_engine_result(confidence=0.90, adjustment_paise=10000)

        strict_output = strict.evaluate(result)
        relaxed_output = relaxed.evaluate(result)

        assert strict_output.passed is False
        assert relaxed_output.passed is True

    def test_empty_blocked_lists(self):
        """Empty blocked lists → type checks pass for any resolution."""
        guard = ExposureGuard(ExposureGuardConfig(
            blocked_exception_types=[],
            blocked_resolution_types=[],
        ))
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=1000,
            exception_type="UNKNOWN",
            selected_resolution="UNKNOWN_UNRESOLVED",
        )
        output = guard.evaluate(result)

        exc_check = next(c for c in output.checks if c.check_name == "blocked_exception_type")
        assert exc_check.passed is True
        res_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert res_check.passed is True

    def test_very_strict_cumulative_limit(self):
        guard = ExposureGuard(ExposureGuardConfig(cumulative_exposure_limit_paise=1000))
        c1 = _make_candidate(amount_paise=600)
        c2 = _make_candidate(amount_paise=600)
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=600,
            candidates=[c1, c2],
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.cumulative_exposure_paise == 1200


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardMetadata:
    """Tests for guard metadata and traceability."""

    def test_exception_id_preserved(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        output = guard.evaluate(result)

        assert output.exception_id == "EXC-001"
        assert output.case_id == "CASE-001"

    def test_adjustment_amount_recorded(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=7500)
        output = guard.evaluate(result)

        assert output.adjustment_amount_paise == 7500

    def test_max_threshold_recorded(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        output = guard.evaluate(result)

        assert output.max_auto_resolution_paise == 50000

    def test_guard_version(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        output = guard.evaluate(result)

        assert output.guard_version == "1.0.0"

    def test_all_checks_present(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=1000)
        output = guard.evaluate(result)

        check_names = {c.check_name for c in output.checks}
        expected = {
            "max_auto_resolution",
            "high_value_threshold",
            "cumulative_exposure",
            "blocked_exception_type",
            "blocked_resolution_type",
            "conflict_penalty",
            "insufficient_evidence",
        }
        assert expected.issubset(check_names)

    def test_block_reasons_recorded(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=100000,
            exception_type="UNKNOWN",
        )
        output = guard.evaluate(result)

        assert len(output.block_reasons) >= 2
        assert ExposureBlockReason.ABOVE_MAX_AMOUNT in output.block_reasons
        assert ExposureBlockReason.HIGH_RISK_CATEGORY in output.block_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardSummary:
    """Tests for guard result summary."""

    def test_pass_summary(self):
        guard = ExposureGuard()
        result = _make_engine_result(confidence=0.90, adjustment_paise=3000)
        output = guard.evaluate(result)

        s = output.summary()
        assert "PASS" in s
        assert "3000" in s

    def test_block_summary(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.95,
            adjustment_paise=100000,
        )
        output = guard.evaluate(result)

        s = output.summary()
        assert "BLOCKED" in s
        assert "100000" in s

    def test_high_value_in_summary(self):
        guard = ExposureGuard()
        result = _make_engine_result(
            confidence=0.90,
            adjustment_paise=100000,
        )
        output = guard.evaluate(result)

        s = output.summary()
        assert "HIGH VALUE" in s
