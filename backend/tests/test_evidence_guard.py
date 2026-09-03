"""
Tests for EvidenceGuard (Phase 6C).

Tests:
- Guard configuration
- Conflicting evidence (hard block)
- Missing evidence (hard block)
- Evidence coverage (threshold)
- Evidence consistency (threshold)
- Novel pattern (hard block)
- Explanation status (hard block)
- Supporting evidence count
- Evidence trace requirement
- Complete evidence
- Partial evidence
- High ML confidence + conflicting evidence
- UNRESOLVED/HUMAN_REVIEW engine result
- Hard-block verification
- Configuration overrides
"""

import pytest

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.evidence_guard import (
    EvidenceAction,
    EvidenceBlockReason,
    EvidenceGuardCheck,
    EvidenceGuardConfig,
    EvidenceGuardResult,
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
from app.services.evidence_guard import EvidenceGuard


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_score(final=0.85, financial=0.9, conflict=0.0, novelty=0.0):
    return CandidateScore(
        evidence_score=0.8,
        ml_score=0.9,
        historical_score=0.7,
        financial_consistency_score=financial,
        final_score=final,
        novelty_penalty=novelty,
        conflict_penalty=conflict,
        weighted_evidence=0.28,
        weighted_ml=0.18,
        weighted_historical=0.105,
        weighted_financial=financial * 0.30,
        has_evidence_support=True,
        has_ml_support=True,
        has_historical_support=True,
        is_novel=novelty > 0,
        has_conflicts=conflict > 0,
    )


def _make_candidate(evidence_ids=None, coverage=0.95):
    if evidence_ids is None:
        evidence_ids = ["FEE-001"]
    return ResolutionProposal(
        candidate_id="CAND-EXC-001",
        exception_id="EXC-001",
        case_id="CASE-001",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Fee adjustment",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=3000,
            direction="CREDIT",
            evidence_record_id="FEE-001",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=evidence_ids,
        evidence_compatible=True,
        evidence_coverage=coverage,
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
    evidence_coverage=0.9,
    evidence_consistency=0.85,
    explanation_status="FULLY_EXPLAINED",
    candidate=None,
):
    if candidate is None and status == SelectionStatus.RECOMMENDED:
        candidate = _make_candidate()
    score = _make_score(final=confidence)
    return ResolutionEngineResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        status=status,
        selected_resolution="FEE_ADJUSTMENT" if status == SelectionStatus.RECOMMENDED else None,
        selected_candidate=candidate,
        selected_score=score if status == SelectionStatus.RECOMMENDED else None,
        ranked_candidates=[candidate] if candidate else [],
        candidate_scores=[score] if status == SelectionStatus.RECOMMENDED else [],
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
        evidence_explanation_status=explanation_status,
        evidence_coverage=evidence_coverage,
        evidence_consistency=evidence_consistency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceGuardConfig:
    """Tests for guard configuration."""

    def test_default_config(self):
        config = EvidenceGuardConfig()
        assert config.min_evidence_coverage == 0.50
        assert config.min_evidence_consistency == 0.50
        assert config.max_missing_evidence == 0
        assert config.block_on_conflict is True
        assert config.block_on_novelty is True
        assert config.allowed_explanation_statuses == ["FULLY_EXPLAINED"]
        assert config.require_evidence_trace is True
        assert config.min_supporting_evidence == 1

    def test_custom_config(self):
        config = EvidenceGuardConfig(
            min_evidence_coverage=0.80,
            min_evidence_consistency=0.75,
            block_on_conflict=False,
        )
        assert config.min_evidence_coverage == 0.80
        assert config.min_evidence_consistency == 0.75
        assert config.block_on_conflict is False


class TestEvidenceAction:
    def test_values(self):
        assert EvidenceAction.PASS.value == "PASS"
        assert EvidenceAction.BLOCK.value == "BLOCK"


class TestEvidenceBlockReason:
    def test_values(self):
        assert EvidenceBlockReason.CONFLICTING_EVIDENCE.value == "CONFLICTING_EVIDENCE"
        assert EvidenceBlockReason.MISSING_EVIDENCE.value == "MISSING_EVIDENCE"
        assert EvidenceBlockReason.LOW_COVERAGE.value == "LOW_COVERAGE"
        assert EvidenceBlockReason.NOVEL_PATTERN.value == "NOVEL_PATTERN"


class TestEvidenceGuardResult:
    def test_passed_result(self):
        result = EvidenceGuardResult(
            passed=True,
            action=EvidenceAction.PASS,
            evidence_coverage=0.9,
            evidence_consistency=0.85,
            reason="OK",
        )
        assert result.passed is True
        assert result.guard_version == "1.0.0"

    def test_summary(self):
        result = EvidenceGuardResult(
            passed=True,
            action=EvidenceAction.PASS,
            evidence_coverage=0.9,
            evidence_consistency=0.85,
            reason="OK",
        )
        s = result.summary()
        assert "PASS" in s
        assert "90.0%" in s


# ─────────────────────────────────────────────────────────────────────────────
# Core Guard Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceGuardEvaluation:
    """Tests for the EvidenceGuard evaluation logic."""

    def test_complete_evidence_passes(self):
        """Complete evidence with high coverage and consistency → PASS."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.95,
            evidence_consistency=0.90,
        )
        output = guard.evaluate(result)

        assert output.passed is True
        assert output.action == EvidenceAction.PASS
        assert output.evidence_coverage == 0.95
        assert output.evidence_consistency == 0.90

    def test_exactly_at_threshold_passes(self):
        """Coverage exactly at threshold → PASS."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.50,
            evidence_consistency=0.50,
        )
        output = guard.evaluate(result)

        assert output.passed is True

    def test_low_coverage_blocks(self):
        """Coverage below threshold → BLOCK."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.20,
            evidence_consistency=0.90,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == EvidenceAction.BLOCK
        assert EvidenceBlockReason.LOW_COVERAGE in output.block_reasons

    def test_low_consistency_blocks(self):
        """Consistency below threshold → BLOCK."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.90,
            evidence_consistency=0.20,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert EvidenceBlockReason.LOW_CONSISTENCY in output.block_reasons

    def test_zero_coverage_blocks(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.0)
        output = guard.evaluate(result)

        assert output.passed is False

    def test_unresolved_engine_result_passes(self):
        """Engine returned UNRESOLVED → guard passes (engine already deferred)."""
        guard = EvidenceGuard()
        result = _make_engine_result(status=SelectionStatus.UNRESOLVED)
        output = guard.evaluate(result)

        assert output.passed is True

    def test_human_review_engine_result_passes(self):
        """Engine returned HUMAN_REVIEW → guard passes."""
        guard = EvidenceGuard()
        result = _make_engine_result(status=SelectionStatus.HUMAN_REVIEW)
        output = guard.evaluate(result)

        assert output.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictBlock:
    """Tests for conflicting evidence hard block."""

    def test_no_conflict_passes(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.9)
        output = guard.evaluate(result)

        assert output.passed is True
        # HIGH #8: has_conflict=None (unknown) passes through from engine result.
        assert output.has_conflict is not True

    def test_conflict_blocks(self):
        """Conflicting evidence → BLOCK regardless of coverage."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.99,
            evidence_consistency=0.99,
        )
        # Manually set conflict (would come from intelligence layer)
        result.evidence_explanation_status = "CONFLICTING"
        output = guard.evaluate(result)

        # The CONFLICTING status blocks via explanation_status check
        assert output.passed is False

    def test_high_confidence_conflict_blocks(self):
        """CRITICAL: High ML confidence + conflicting evidence → BLOCK."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            confidence=0.99,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            explanation_status="CONFLICTING",
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == EvidenceAction.BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Missing Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingEvidence:
    """Tests for missing evidence handling."""

    def test_no_evidence_records_blocks(self):
        """No supporting evidence → BLOCK."""
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        assert output.passed is False
        assert EvidenceBlockReason.MISSING_EVIDENCE in output.block_reasons

    def test_evidence_present_passes(self):
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=["FEE-001"])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        assert output.passed is True

    def test_custom_max_missing(self):
        """Allow 1 missing evidence record."""
        config = EvidenceGuardConfig(max_missing_evidence=1)
        guard = EvidenceGuard(config)
        # The missing_evidence count comes from the intelligence layer
        # which we don't set in the engine result directly
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.9)
        output = guard.evaluate(result)

        assert output.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Coverage Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCoverageThreshold:
    """Tests for evidence coverage threshold."""

    def test_high_coverage_passes(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.95)
        output = guard.evaluate(result)

        assert output.passed is True

    def test_marginal_coverage_blocks(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.49)
        output = guard.evaluate(result)

        assert output.passed is False

    def test_custom_threshold(self):
        guard = EvidenceGuard(EvidenceGuardConfig(min_evidence_coverage=0.90))
        result = _make_engine_result(evidence_coverage=0.85)
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConsistencyThreshold:
    """Tests for evidence consistency threshold."""

    def test_high_consistency_passes(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_consistency=0.95)
        output = guard.evaluate(result)

        assert output.passed is True

    def test_low_consistency_blocks(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_consistency=0.30)
        output = guard.evaluate(result)

        assert output.passed is False
        assert EvidenceBlockReason.LOW_CONSISTENCY in output.block_reasons

    def test_custom_threshold(self):
        guard = EvidenceGuard(EvidenceGuardConfig(min_evidence_consistency=0.80))
        result = _make_engine_result(evidence_consistency=0.75)
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Novelty Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoveltyBlock:
    """Tests for novel pattern blocking."""

    def test_known_pattern_passes(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.9)
        output = guard.evaluate(result)

        assert output.passed is True
        # HIGH #8: is_novel=None (unknown) passes through from engine result.
        assert output.is_novel is not True

    def test_novel_case_not_directly_settable(self):
        """Novelty is tracked but requires external detection.

        The evidence guard checks is_novel, which must be set
        by the intelligence layer before evaluation.
        """
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.9)
        output = guard.evaluate(result)

        # HIGH #8: is_novel=None (unknown) passes through from engine result.
        assert output.is_novel is not True

    def test_block_on_novelty_disabled(self):
        """When block_on_novelty is False, novelty doesn't block."""
        config = EvidenceGuardConfig(block_on_novelty=False)
        guard = EvidenceGuard(config)
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.9)
        output = guard.evaluate(result)

        # Even if novelty were true, it wouldn't block
        assert output.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Status Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExplanationStatus:
    """Tests for explanation status blocking."""

    def test_fully_explained_passes(self):
        guard = EvidenceGuard()
        result = _make_engine_result(explanation_status="FULLY_EXPLAINED")
        output = guard.evaluate(result)

        assert output.passed is True

    def test_partially_explained_blocks(self):
        guard = EvidenceGuard()
        result = _make_engine_result(
            explanation_status="PARTIALLY_EXPLAINED",
            evidence_coverage=0.9,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert EvidenceBlockReason.UNEXPLAINED in output.block_reasons

    def test_unexplained_blocks(self):
        guard = EvidenceGuard()
        result = _make_engine_result(
            explanation_status="UNEXPLAINED",
            evidence_coverage=0.9,
        )
        output = guard.evaluate(result)

        assert output.passed is False

    def test_custom_allowed_statuses(self):
        config = EvidenceGuardConfig(
            allowed_explanation_statuses=["FULLY_EXPLAINED", "PARTIALLY_EXPLAINED"]
        )
        guard = EvidenceGuard(config)
        result = _make_engine_result(
            explanation_status="PARTIALLY_EXPLAINED",
            evidence_coverage=0.9,
        )
        output = guard.evaluate(result)

        assert output.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Trace Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceTrace:
    """Tests for evidence trace requirement."""

    def test_trace_present_passes(self):
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=["FEE-001"])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        trace_check = next(c for c in output.checks if c.check_name == "evidence_trace")
        assert trace_check.passed is True

    def test_no_trace_blocks(self):
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        trace_check = next(c for c in output.checks if c.check_name == "evidence_trace")
        assert trace_check.passed is False

    def test_trace_not_required(self):
        config = EvidenceGuardConfig(require_evidence_trace=False)
        guard = EvidenceGuard(config)
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        trace_checks = [c for c in output.checks if c.check_name == "evidence_trace"]
        assert len(trace_checks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Hard Block Verification Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHardBlock:
    """Tests for hard-block behavior."""

    def test_high_ml_confidence_cannot_override_missing_evidence(self):
        """CRITICAL: 99% ML confidence + no evidence → BLOCK."""
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(
            confidence=0.99,
            candidate=candidate,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert output.action == EvidenceAction.BLOCK

    def test_high_confidence_low_coverage_blocks(self):
        """High confidence + low coverage → BLOCK."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            confidence=0.95,
            evidence_coverage=0.20,
            evidence_consistency=0.90,
        )
        output = guard.evaluate(result)

        assert output.passed is False

    def test_high_confidence_explaining_blocks(self):
        """High confidence + UNEXPLAINED → BLOCK."""
        guard = EvidenceGuard()
        result = _make_engine_result(
            confidence=0.99,
            explanation_status="UNEXPLAINED",
            evidence_coverage=0.90,
        )
        output = guard.evaluate(result)

        assert output.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiFailure:
    """Tests for multiple simultaneous failures."""

    def test_low_coverage_and_low_consistency(self):
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.20,
            evidence_consistency=0.20,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        failed_checks = [c for c in output.checks if not c.passed]
        assert len(failed_checks) >= 2

    def test_all_blocks_active(self):
        """Coverage + consistency + explanation + no evidence → multiple blocks."""
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(
            evidence_coverage=0.10,
            evidence_consistency=0.10,
            explanation_status="UNEXPLAINED",
            candidate=candidate,
        )
        output = guard.evaluate(result)

        assert output.passed is False
        assert len(output.block_reasons) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Override Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurationOverrides:
    """Tests for configurable thresholds."""

    def test_strict_coverage_blocks_more(self):
        strict = EvidenceGuard(EvidenceGuardConfig(min_evidence_coverage=0.90))
        relaxed = EvidenceGuard(EvidenceGuardConfig(min_evidence_coverage=0.30))
        result = _make_engine_result(evidence_coverage=0.50)

        strict_output = strict.evaluate(result)
        relaxed_output = relaxed.evaluate(result)

        assert strict_output.passed is False
        assert relaxed_output.passed is True

    def test_disable_all_blocks(self):
        """Disable conflict and novelty blocking."""
        config = EvidenceGuardConfig(
            block_on_conflict=False,
            block_on_novelty=False,
            min_evidence_coverage=0.0,
            min_evidence_consistency=0.0,
            max_missing_evidence=10,
            require_evidence_trace=False,
            min_supporting_evidence=0,
            allowed_explanation_statuses=[
                "FULLY_EXPLAINED",
                "PARTIALLY_EXPLAINED",
                "UNEXPLAINED",
                "CONFLICTING",
            ],
        )
        guard = EvidenceGuard(config)
        candidate = _make_candidate(evidence_ids=[])
        result = _make_engine_result(
            evidence_coverage=0.0,
            evidence_consistency=0.0,
            explanation_status="UNEXPLAINED",
            candidate=candidate,
        )
        output = guard.evaluate(result)

        assert output.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardMetadata:
    """Tests for guard metadata and traceability."""

    def test_exception_id_preserved(self):
        guard = EvidenceGuard()
        result = _make_engine_result()
        output = guard.evaluate(result)

        assert output.exception_id == "EXC-001"
        assert output.case_id == "CASE-001"

    def test_evidence_metrics_recorded(self):
        guard = EvidenceGuard()
        result = _make_engine_result(
            evidence_coverage=0.85,
            evidence_consistency=0.78,
        )
        output = guard.evaluate(result)

        assert output.evidence_coverage == 0.85
        assert output.evidence_consistency == 0.78

    def test_guard_version(self):
        guard = EvidenceGuard()
        result = _make_engine_result()
        output = guard.evaluate(result)

        assert output.guard_version == "1.0.0"

    def test_all_checks_present(self):
        guard = EvidenceGuard()
        candidate = _make_candidate(evidence_ids=["FEE-001"])
        result = _make_engine_result(candidate=candidate)
        output = guard.evaluate(result)

        check_names = {c.check_name for c in output.checks}
        expected = {
            "missing_evidence",
            "evidence_coverage",
            "evidence_consistency",
            "explanation_status",
            "supporting_evidence_count",
            "evidence_trace",
        }
        assert expected.issubset(check_names)

    def test_explanation_status_recorded(self):
        guard = EvidenceGuard()
        result = _make_engine_result(explanation_status="FULLY_EXPLAINED")
        output = guard.evaluate(result)

        assert output.explanation_status == "FULLY_EXPLAINED"


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardSummary:
    """Tests for guard result summary."""

    def test_pass_summary(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.9, evidence_consistency=0.85)
        output = guard.evaluate(result)

        s = output.summary()
        assert "PASS" in s
        assert "90.0%" in s

    def test_block_summary(self):
        guard = EvidenceGuard()
        result = _make_engine_result(evidence_coverage=0.10)
        output = guard.evaluate(result)

        s = output.summary()
        assert "BLOCKED" in s
        assert "10.0%" in s

    def test_conflict_in_summary(self):
        guard = EvidenceGuard()
        result = _make_engine_result(
            explanation_status="CONFLICTING",
            evidence_coverage=0.9,
        )
        output = guard.evaluate(result)

        s = output.summary()
        # Explanation status blocks, not conflict flag directly
        assert "BLOCKED" in s
