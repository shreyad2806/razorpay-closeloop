"""
Tests for evidence quality scoring.

Covers:
- Perfect explanation
- Partial explanation
- No explanation
- Conflicting evidence
- Missing evidence
- Unknown case
- Zero discrepancy
- Consistency score formula
- Coverage score formula
- Novelty logic
- Score ranges
- Ground truth separation
"""

import os
import sys
from pathlib import Path

import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_evidence_quality.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
    StructuralConflict,
)
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.evidence_quality import NoveltyLevel
from app.services.explanation_engine import DeterministicExplanationEngine
from app.services.evidence_quality import (
    EvidenceQualityScorer,
    MISSING_EVIDENCE_PENALTY,
    CONFLICT_PENALTY,
    NO_EVIDENCE_PENALTY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def scorer():
    """Provide an evidence quality scorer."""
    return EvidenceQualityScorer()


@pytest.fixture
def explainer():
    """Provide an explanation engine."""
    return DeterministicExplanationEngine()


def _make_package(**kwargs):
    """Helper to create an EvidencePackage with defaults."""
    defaults = {
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "payment_id": "PAY-001",
        "expected_amount": 100000,
        "actual_amount": 100000,
        "difference": 0,
        "exception_type": "EXACT_MATCH",
    }
    defaults.update(kwargs)
    return EvidencePackage(**defaults)


def _make_record(record_id, entity_type, amount, **kwargs):
    """Helper to create an EvidenceRecord."""
    defaults = {
        "record_id": record_id,
        "entity_type": entity_type,
        "relationship": "CALCULATION_COMPONENT",
        "amount": amount,
    }
    defaults.update(kwargs)
    return EvidenceRecord(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Perfect Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPerfectExplanation:
    """Tests for perfect explanation scoring."""

    def test_perfect_explanation_high_quality(self, scorer, explainer):
        """Test that perfect explanation has high quality scores."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score >= 0.8
        assert result.coverage_score == 1.0
        assert result.is_high_quality()
        assert not result.needs_review()

    def test_perfect_explanation_consistency_1(self, scorer, explainer):
        """Test that perfect explanation has consistency 1.0."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score == 1.0

    def test_perfect_explanation_coverage_1(self, scorer, explainer):
        """Test that perfect explanation has coverage 1.0."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.coverage_score == 1.0

    def test_perfect_explanation_no_conflict(self, scorer, explainer):
        """Test that perfect explanation has no conflict."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.conflict is False

    def test_perfect_explanation_fully_explained(self, scorer, explainer):
        """Test that perfect explanation has fully_explained=True."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.fully_explained is True
        assert result.partially_explained is False


# ─────────────────────────────────────────────────────────────────────────────
# Partial Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialExplanation:
    """Tests for partial explanation scoring."""

    def test_partial_explanation_coverage(self, scorer, explainer):
        """Test that partial explanation has proportional coverage."""
        # difference = -1000, explained = -300, coverage = 300/1000 = 0.3
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.coverage_score == pytest.approx(0.3, abs=0.01)
        assert result.partially_explained is True
        assert result.fully_explained is False

    def test_partial_explanation_needs_review(self, scorer, explainer):
        """Test that partial explanation needs review."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.needs_review() is True

    def test_partial_explanation_consistency_reduced(self, scorer, explainer):
        """Test that partial explanation has reduced consistency."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# No Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoExplanation:
    """Tests for no explanation scoring."""

    def test_no_explanation_coverage_zero(self, scorer, explainer):
        """Test that no explanation has coverage 0."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.coverage_score == 0.0
        assert result.fully_explained is False
        assert result.partially_explained is False

    def test_no_explanation_low_consistency(self, scorer, explainer):
        """Test that no explanation has low consistency."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score < 0.8

    def test_no_explanation_needs_review(self, scorer, explainer):
        """Test that no explanation needs review."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.needs_review() is True


# ─────────────────────────────────────────────────────────────────────────────
# Conflicting Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictingEvidence:
    """Tests for conflicting evidence scoring."""

    def test_conflicting_evidence_detected(self, scorer, explainer):
        """Test that conflicting evidence is detected."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.conflict is True
        assert result.needs_review() is True

    def test_conflicting_evidence_reduces_consistency(self, scorer, explainer):
        """Test that conflicting evidence reduces consistency."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        # Consistency reduced by explanation conflict penalty
        assert result.consistency_score < 1.0

    def test_structural_conflict_reduces_consistency(self, scorer, explainer):
        """Test that structural conflicts in evidence reduce consistency."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
            conflicts=[
                StructuralConflict(
                    conflict_type="MULTIPLE_SETTLEMENTS",
                    description="Multiple settlements found",
                    affected_records=["SET-001", "SET-002"],
                ),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Missing Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingEvidence:
    """Tests for missing evidence scoring."""

    def test_missing_evidence_reduces_consistency(self, scorer, explainer):
        """Test that missing evidence reduces consistency."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score < 1.0
        assert "SETTLEMENT" in result.missing_evidence

    def test_missing_evidence_penalty_amount(self, scorer, explainer):
        """Test that missing evidence penalty is correct."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        # Consistency = 1.0 - 0.15 (missing) - 0.30 (no evidence penalty) - remainder penalty
        # But the remainder penalty depends on the explanation
        assert result.consistency_breakdown["missing_evidence_penalty"] == -MISSING_EVIDENCE_PENALTY

    def test_multiple_missing_evidence(self, scorer, explainer):
        """Test that multiple missing records compound penalty."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
                MissingEvidence(
                    entity_type="REFUND",
                    expected=True,
                    reason="No refund found",
                ),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score < 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Unknown Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownCase:
    """Tests for unknown case scoring."""

    def test_unknown_case_novelty(self, scorer, explainer):
        """Test that unknown case has novel novelty."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            exception_type="UNKNOWN",
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.novelty == NoveltyLevel.NOVEL_NO_HISTORICAL

    def test_unknown_case_unexplained(self, scorer, explainer):
        """Test that unknown case with no evidence is unexplained."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            exception_type="UNKNOWN",
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.fully_explained is False
        assert result.partially_explained is False

    def test_known_type_with_evidence_not_novel(self, scorer, explainer):
        """Test that known type with explained evidence is not novel."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            exception_type="FEE_DIFFERENCE",
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.novelty == NoveltyLevel.KNOWN_PATTERN


# ─────────────────────────────────────────────────────────────────────────────
# Zero Discrepancy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroDiscrepancy:
    """Tests for zero discrepancy scoring."""

    def test_zero_discrepancy_coverage_1(self, scorer, explainer):
        """Test that zero discrepancy has coverage 1.0."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.coverage_score == 1.0

    def test_zero_discrepancy_consistency_1(self, scorer, explainer):
        """Test that zero discrepancy has consistency 1.0."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_score == 1.0

    def test_zero_discrepancy_high_quality(self, scorer, explainer):
        """Test that zero discrepancy is high quality."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.is_high_quality()
        assert not result.needs_review()


# ─────────────────────────────────────────────────────────────────────────────
# Score Range Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreRanges:
    """Tests for score range validation."""

    def test_consistency_in_range(self, scorer, explainer):
        """Test that consistency score is in [0.0, 1.0]."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement",
                ),
                MissingEvidence(
                    entity_type="REFUND",
                    expected=True,
                    reason="No refund",
                ),
            ],
            conflicts=[
                StructuralConflict(
                    conflict_type="MULTIPLE_SETTLEMENTS",
                    description="Multiple settlements",
                    affected_records=["SET-001"],
                ),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert 0.0 <= result.consistency_score <= 1.0

    def test_coverage_in_range(self, scorer, explainer):
        """Test that coverage score is in [0.0, 1.0]."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert 0.0 <= result.coverage_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Breakdown Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScoringBreakdown:
    """Tests for scoring breakdown traceability."""

    def test_breakdown_has_all_fields(self, scorer, explainer):
        """Test that breakdown contains all expected fields."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_breakdown is not None
        assert "base" in result.consistency_breakdown
        assert "missing_evidence_penalty" in result.consistency_breakdown
        assert "conflict_penalty" in result.consistency_breakdown
        assert "no_evidence_penalty" in result.consistency_breakdown
        assert "remainder_penalty" in result.consistency_breakdown

    def test_breakdown_base_is_1(self, scorer, explainer):
        """Test that breakdown base is always 1.0."""
        pkg = _make_package(difference=0)
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.consistency_breakdown["base"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Verify that scoring does not reference ground truth."""

    GROUND_TRUTH_TERMS = [
        "ground_truth",
        "true_exception_type",
        "true_resolution",
    ]

    def test_scorer_code_no_ground_truth(self):
        """Test that scorer source has no ground truth references."""
        import inspect
        from app.services.evidence_quality import EvidenceQualityScorer

        source = inspect.getsource(EvidenceQualityScorer)
        for term in self.GROUND_TRUTH_TERMS:
            assert term not in source, f"Ground truth reference found: {term}"

    def test_quality_result_no_ground_truth(self, scorer, explainer):
        """Test that EvidenceQualityResult has no ground truth fields."""
        pkg = _make_package(difference=0)
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert not hasattr(result, "true_exception_type")
        assert not hasattr(result, "true_resolution")
        assert not hasattr(result, "resolvable")
        assert not hasattr(result, "risk_category")


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Count Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceCount:
    """Tests for supporting evidence count."""

    def test_evidence_count_matches(self, scorer, explainer):
        """Test that evidence count matches explanation supporting IDs."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 300),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 200),
            ],
        )
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.supporting_evidence_count == len(explanation.supporting_evidence_ids)

    def test_zero_evidence_count(self, scorer, explainer):
        """Test that zero discrepancy has zero evidence count."""
        pkg = _make_package(difference=0)
        explanation = explainer.explain(pkg)
        result = scorer.score(pkg, explanation)

        assert result.supporting_evidence_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScoringConstants:
    """Tests for scoring configuration constants."""

    def test_missing_evidence_penalty(self):
        """Test missing evidence penalty is reasonable."""
        assert 0 < MISSING_EVIDENCE_PENALTY < 0.5

    def test_conflict_penalty(self):
        """Test conflict penalty is reasonable."""
        assert 0 < CONFLICT_PENALTY < 0.5

    def test_no_evidence_penalty(self):
        """Test no evidence penalty is reasonable."""
        assert 0 < NO_EVIDENCE_PENALTY < 0.5
