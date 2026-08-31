"""
Tests for ConfidenceGate (Phase 6A).

Tests:
- Gate configuration
- Confidence threshold checks
- Financial consistency checks
- Evidence coverage checks
- Risk level checks
- High-value adjustment checks
- Conflict penalty checks
- Novelty penalty checks
- Blocked resolution type checks
- Supporting evidence count checks
- UNRESOLVED engine result
- HUMAN_REVIEW engine result
- All checks pass
- No candidate
- Configurable thresholds
- Missing score data
"""

import pytest
from datetime import datetime

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.confidence_gate import (
    ConfidenceGateConfig,
    ConfidenceGateResult,
    GateAction,
    GateCheck,
    RiskOverrideLevel,
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
from app.services.confidence_gate import ConfidenceGate


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
    """Create a CandidateScore for testing."""
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
    """Create a ResolutionProposal for testing."""
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
    evidence_coverage=0.9,
    adjustment_paise=3000,
    score=None,
):
    """Create a ResolutionEngineResult for testing."""
    if score is None:
        score = _make_score(final=confidence, financial=0.9)
    candidate = None
    if status == SelectionStatus.RECOMMENDED and selected_resolution:
        candidate = _make_candidate(
            resolution_type=selected_resolution,
            amount_paise=adjustment_paise,
        )
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
        ranked_candidates=[candidate] if candidate else [],
        candidate_scores=[score] if score and status == SelectionStatus.RECOMMENDED else [],
        confidence=confidence,
        risk_category=risk,
        explainability=ExplainabilityDetail(
            level=ExplainabilityLevel.FULLY_EXPLAINABLE,
            has_evidence_trace=True,
            has_financial_trace=True,
            source_count=2,
        ),
        deterministic_exception_type="FEE_DIFFERENCE",
        classification_agreement=True,
        evidence_explanation_status="FULLY_EXPLAINED",
        evidence_coverage=evidence_coverage,
        evidence_consistency=0.9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateConfig:
    """Tests for gate configuration."""

    def test_default_config(self):
        config = ConfidenceGateConfig()
        assert config.min_confidence == 0.70
        assert config.min_financial_consistency == 0.60
        assert config.min_evidence_coverage == 0.40
        assert "HIGH" not in config.allowed_risk_levels
        assert "LOW" in config.allowed_risk_levels
        assert "MEDIUM" in config.allowed_risk_levels

    def test_custom_config(self):
        config = ConfidenceGateConfig(
            min_confidence=0.80,
            min_financial_consistency=0.75,
            high_value_threshold_paise=200000,
        )
        assert config.min_confidence == 0.80
        assert config.min_financial_consistency == 0.75
        assert config.high_value_threshold_paise == 200000

    def test_blocked_resolution_types(self):
        config = ConfidenceGateConfig()
        assert "UNKNOWN_UNRESOLVED" in config.blocked_resolution_types
        assert "MISSING_RECORD_ESCALATION" in config.blocked_resolution_types


class TestGateAction:
    """Tests for GateAction enum."""

    def test_values(self):
        assert GateAction.CONTINUE.value == "CONTINUE"
        assert GateAction.HUMAN_REVIEW.value == "HUMAN_REVIEW"


class TestGateCheck:
    """Tests for GateCheck schema."""

    def test_create(self):
        check = GateCheck(
            check_name="confidence",
            passed=True,
            value=0.85,
            threshold=0.70,
            reason="Passed",
        )
        assert check.passed is True
        assert check.value == 0.85
        assert check.threshold == 0.70


class TestConfidenceGateResult:
    """Tests for ConfidenceGateResult schema."""

    def test_passed_result(self):
        result = ConfidenceGateResult(
            passed=True,
            action=GateAction.CONTINUE,
            confidence=0.85,
            threshold=0.70,
            reason="All checks passed",
        )
        assert result.passed is True
        assert result.action == GateAction.CONTINUE
        assert result.gate_version == "1.0.0"

    def test_blocked_result(self):
        result = ConfidenceGateResult(
            passed=False,
            action=GateAction.HUMAN_REVIEW,
            confidence=0.50,
            threshold=0.70,
            reason="Below threshold",
        )
        assert result.passed is False
        assert result.action == GateAction.HUMAN_REVIEW

    def test_summary(self):
        result = ConfidenceGateResult(
            passed=True,
            action=GateAction.CONTINUE,
            confidence=0.85,
            threshold=0.70,
            reason="All checks passed",
        )
        s = result.summary()
        assert "PASSED" in s
        assert "85.0%" in s


# ─────────────────────────────────────────────────────────────────────────────
# Core Gate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceGateEvaluation:
    """Tests for the ConfidenceGate evaluation logic."""

    def test_high_confidence_passes(self):
        """Confidence clearly above threshold → CONTINUE."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is True
        assert output.action == GateAction.CONTINUE
        assert output.confidence == 0.90
        assert output.threshold == 0.70
        assert len(output.checks) > 0

    def test_exact_threshold_passes(self):
        """Confidence exactly at threshold → CONTINUE."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.70, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is True
        assert output.action == GateAction.CONTINUE

    def test_below_threshold_blocked(self):
        """Confidence below threshold → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.50, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.action == GateAction.HUMAN_REVIEW
        assert "50.0%" in output.reason

    def test_missing_confidence_blocked(self):
        """Zero confidence → HUMAN_REVIEW."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.0, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.action == GateAction.HUMAN_REVIEW

    def test_max_confidence_passes(self):
        """Confidence at maximum → CONTINUE."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=1.0, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is True
        assert output.action == GateAction.CONTINUE

    def test_unresolved_engine_result(self):
        """Engine returned UNRESOLVED → gate blocks."""
        gate = ConfidenceGate()
        result = _make_engine_result(status=SelectionStatus.UNRESOLVED)
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.action == GateAction.HUMAN_REVIEW
        assert "UNRESOLVED" in output.reason

    def test_human_review_engine_result(self):
        """Engine returned HUMAN_REVIEW → gate blocks."""
        gate = ConfidenceGate()
        result = _make_engine_result(status=SelectionStatus.HUMAN_REVIEW)
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.action == GateAction.HUMAN_REVIEW
        assert "HUMAN_REVIEW" in output.reason


# ─────────────────────────────────────────────────────────────────────────────
# Financial Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialConsistencyCheck:
    """Tests for financial consistency gate check."""

    def test_high_financial_consistency_passes(self):
        gate = ConfidenceGate()
        score = _make_score(financial=0.95, final=0.90)
        result = _make_engine_result(confidence=0.90, score=score, risk="LOW")
        output = gate.evaluate(result)

        fin_check = next(c for c in output.checks if c.check_name == "financial_consistency")
        assert fin_check.passed is True

    def test_low_financial_consistency_blocks(self):
        gate = ConfidenceGate()
        score = _make_score(financial=0.30, final=0.80)
        result = _make_engine_result(confidence=0.85, score=score, risk="LOW")
        output = gate.evaluate(result)

        fin_check = next(c for c in output.checks if c.check_name == "financial_consistency")
        assert fin_check.passed is False
        assert output.passed is False

    def test_custom_financial_threshold(self):
        gate = ConfidenceGate(ConfidenceGateConfig(min_financial_consistency=0.90))
        score = _make_score(financial=0.85, final=0.85)
        result = _make_engine_result(confidence=0.85, score=score, risk="LOW")
        output = gate.evaluate(result)

        fin_check = next(c for c in output.checks if c.check_name == "financial_consistency")
        assert fin_check.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Coverage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceCoverageCheck:
    """Tests for evidence coverage gate check."""

    def test_high_coverage_passes(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, evidence_coverage=0.95, risk="LOW")
        output = gate.evaluate(result)

        ev_check = next(c for c in output.checks if c.check_name == "evidence_coverage")
        assert ev_check.passed is True

    def test_low_coverage_blocks(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, evidence_coverage=0.10, risk="LOW")
        output = gate.evaluate(result)

        ev_check = next(c for c in output.checks if c.check_name == "evidence_coverage")
        assert ev_check.passed is False
        assert output.passed is False

    def test_zero_coverage_blocks(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, evidence_coverage=0.0, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Risk Level Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskLevelCheck:
    """Tests for risk level gate check."""

    def test_low_risk_passes(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        risk_check = next(c for c in output.checks if c.check_name == "risk_level")
        assert risk_check.passed is True

    def test_medium_risk_passes(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="MEDIUM")
        output = gate.evaluate(result)

        risk_check = next(c for c in output.checks if c.check_name == "risk_level")
        assert risk_check.passed is True

    def test_high_risk_blocks(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.95, risk="HIGH")
        output = gate.evaluate(result)

        risk_check = next(c for c in output.checks if c.check_name == "risk_level")
        assert risk_check.passed is False
        assert output.blocked_by_risk is True
        assert output.passed is False

    def test_custom_risk_config(self):
        """Only LOW is allowed."""
        gate = ConfidenceGate(ConfidenceGateConfig(allowed_risk_levels=["LOW"]))
        result = _make_engine_result(confidence=0.90, risk="MEDIUM")
        output = gate.evaluate(result)

        risk_check = next(c for c in output.checks if c.check_name == "risk_level")
        assert risk_check.passed is False
        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# High-Value Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHighValueAdjustment:
    """Tests for high-value adjustment gate check."""

    def test_small_adjustment_passes(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.90, risk="LOW", adjustment_paise=5000
        )
        output = gate.evaluate(result)

        hv_check = next(c for c in output.checks if c.check_name == "high_value_adjustment")
        assert hv_check.passed is True
        assert output.blocked_by_high_value is False

    def test_large_adjustment_blocks(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.95, risk="LOW", adjustment_paise=100000
        )
        output = gate.evaluate(result)

        hv_check = next(c for c in output.checks if c.check_name == "high_value_adjustment")
        assert hv_check.passed is False
        assert output.blocked_by_high_value is True
        assert output.passed is False

    def test_custom_threshold(self):
        gate = ConfidenceGate(ConfidenceGateConfig(high_value_threshold_paise=5000))
        result = _make_engine_result(
            confidence=0.90, risk="LOW", adjustment_paise=6000
        )
        output = gate.evaluate(result)

        hv_check = next(c for c in output.checks if c.check_name == "high_value_adjustment")
        assert hv_check.passed is False

    def test_adjustment_at_threshold_blocks(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.90, risk="LOW", adjustment_paise=100000
        )
        output = gate.evaluate(result)

        hv_check = next(c for c in output.checks if c.check_name == "high_value_adjustment")
        assert hv_check.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Penalty Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictPenaltyCheck:
    """Tests for conflict penalty gate check."""

    def test_no_conflict_passes(self):
        gate = ConfidenceGate()
        score = _make_score(conflict=0.0, final=0.90)
        result = _make_engine_result(confidence=0.90, score=score, risk="LOW")
        output = gate.evaluate(result)

        conflict_check = next(c for c in output.checks if c.check_name == "conflict_penalty")
        assert conflict_check.passed is True

    def test_high_conflict_blocks(self):
        gate = ConfidenceGate()
        score = _make_score(conflict=0.20, final=0.75)
        result = _make_engine_result(confidence=0.75, score=score, risk="LOW")
        output = gate.evaluate(result)

        conflict_check = next(c for c in output.checks if c.check_name == "conflict_penalty")
        assert conflict_check.passed is False
        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Novelty Penalty Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoveltyPenaltyCheck:
    """Tests for novelty penalty gate check."""

    def test_no_novelty_passes(self):
        gate = ConfidenceGate()
        score = _make_score(novelty=0.0, final=0.90)
        result = _make_engine_result(confidence=0.90, score=score, risk="LOW")
        output = gate.evaluate(result)

        novelty_check = next(c for c in output.checks if c.check_name == "novelty_penalty")
        assert novelty_check.passed is True

    def test_high_novelty_blocks(self):
        gate = ConfidenceGate()
        score = _make_score(novelty=0.20, final=0.75)
        result = _make_engine_result(confidence=0.75, score=score, risk="LOW")
        output = gate.evaluate(result)

        novelty_check = next(c for c in output.checks if c.check_name == "novelty_penalty")
        assert novelty_check.passed is False
        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Blocked Resolution Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockedResolutionType:
    """Tests for blocked resolution type gate check."""

    def test_fee_adjustment_not_blocked(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.90, risk="LOW", selected_resolution="FEE_ADJUSTMENT"
        )
        output = gate.evaluate(result)

        type_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert type_check.passed is True

    def test_unknown_unresolved_blocked(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.95, risk="LOW", selected_resolution="UNKNOWN_UNRESOLVED"
        )
        output = gate.evaluate(result)

        type_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert type_check.passed is False
        assert output.blocked_by_blocked_type is True
        assert output.passed is False

    def test_missing_record_escalation_blocked(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.95, risk="LOW", selected_resolution="MISSING_RECORD_ESCALATION"
        )
        output = gate.evaluate(result)

        type_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert type_check.passed is False
        assert output.blocked_by_blocked_type is True

    def test_custom_blocked_list(self):
        gate = ConfidenceGate(
            ConfidenceGateConfig(blocked_resolution_types=["FEE_ADJUSTMENT"])
        )
        result = _make_engine_result(
            confidence=0.90, risk="LOW", selected_resolution="FEE_ADJUSTMENT"
        )
        output = gate.evaluate(result)

        type_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert type_check.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Supporting Evidence Count Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSupportingEvidenceCount:
    """Tests for supporting evidence count gate check."""

    def test_evidence_present_passes(self):
        gate = ConfidenceGate()
        candidate = _make_candidate(evidence_ids=["FEE-001", "FEE-002"])
        score = _make_score(final=0.90)
        result = _make_engine_result(confidence=0.90, risk="LOW", score=score)
        result.selected_candidate = candidate
        output = gate.evaluate(result)

        ev_count_check = next(c for c in output.checks if c.check_name == "supporting_evidence_count")
        assert ev_count_check.passed is True

    def test_no_evidence_blocks(self):
        gate = ConfidenceGate()
        candidate = _make_candidate(evidence_ids=[])
        score = _make_score(final=0.90)
        result = _make_engine_result(confidence=0.90, risk="LOW", score=score)
        result.selected_candidate = candidate
        output = gate.evaluate(result)

        ev_count_check = next(c for c in output.checks if c.check_name == "supporting_evidence_count")
        assert ev_count_check.passed is False
        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiFailure:
    """Tests for multiple simultaneous failures."""

    def test_multiple_checks_fail(self):
        """Low confidence + high risk → both fail, first failure is primary reason."""
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.30, risk="HIGH")
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.action == GateAction.HUMAN_REVIEW
        failed_checks = [c for c in output.checks if not c.passed]
        assert len(failed_checks) >= 2

    def test_high_value_plus_high_risk(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.90, risk="HIGH", adjustment_paise=200000
        )
        output = gate.evaluate(result)

        assert output.passed is False
        assert output.blocked_by_risk is True
        # High value check also fails but risk was first
        hv_check = next(c for c in output.checks if c.check_name == "high_value_adjustment")
        assert hv_check.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Override Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurationOverrides:
    """Tests for configurable thresholds."""

    def test_strict_threshold_blocks_more(self):
        """Higher min_confidence blocks more candidates."""
        strict = ConfidenceGate(ConfidenceGateConfig(min_confidence=0.90))
        relaxed = ConfidenceGate(ConfidenceGateConfig(min_confidence=0.50))
        result = _make_engine_result(confidence=0.70, risk="LOW")

        strict_output = strict.evaluate(result)
        relaxed_output = relaxed.evaluate(result)

        assert strict_output.passed is False
        assert relaxed_output.passed is True

    def test_no_blocked_types(self):
        """Empty blocked list → type check passes for any resolution."""
        gate = ConfidenceGate(ConfidenceGateConfig(blocked_resolution_types=[]))
        result = _make_engine_result(
            confidence=0.90, risk="LOW", selected_resolution="UNKNOWN_UNRESOLVED"
        )
        output = gate.evaluate(result)

        type_check = next(c for c in output.checks if c.check_name == "blocked_resolution_type")
        assert type_check.passed is True

    def test_strict_evidence_threshold(self):
        gate = ConfidenceGate(ConfidenceGateConfig(min_evidence_coverage=0.95))
        result = _make_engine_result(confidence=0.90, evidence_coverage=0.80, risk="LOW")
        output = gate.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGateMetadata:
    """Tests for gate metadata and traceability."""

    def test_exception_id_preserved(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        assert output.exception_id == "EXC-001"
        assert output.case_id == "CASE-001"

    def test_adjustment_amount_recorded(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.90, risk="LOW", adjustment_paise=5000
        )
        output = gate.evaluate(result)

        assert output.adjustment_amount_paise == 5000

    def test_gate_version(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        assert output.gate_version == "1.0.0"

    def test_all_checks_present(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        check_names = {c.check_name for c in output.checks}
        expected_checks = {
            "confidence_threshold",
            "financial_consistency",
            "evidence_coverage",
            "risk_level",
            "high_value_adjustment",
            "conflict_penalty",
            "novelty_penalty",
            "blocked_resolution_type",
            "supporting_evidence_count",
        }
        assert expected_checks.issubset(check_names)

    def test_check_values_recorded(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.85, risk="LOW")
        output = gate.evaluate(result)

        conf_check = next(c for c in output.checks if c.check_name == "confidence_threshold")
        assert conf_check.value == 0.85
        assert conf_check.threshold == 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGateSummary:
    """Tests for gate result summary."""

    def test_passed_summary(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.90, risk="LOW")
        output = gate.evaluate(result)

        s = output.summary()
        assert "PASSED" in s
        assert "90.0%" in s

    def test_blocked_by_risk_summary(self):
        gate = ConfidenceGate()
        result = _make_engine_result(confidence=0.95, risk="HIGH")
        output = gate.evaluate(result)

        s = output.summary()
        assert "BLOCKED" in s
        assert "Risk level" in s

    def test_blocked_by_high_value_summary(self):
        gate = ConfidenceGate()
        result = _make_engine_result(
            confidence=0.95, risk="LOW", adjustment_paise=100000
        )
        output = gate.evaluate(result)

        s = output.summary()
        assert "BLOCKED" in s
        assert "100000" in s
