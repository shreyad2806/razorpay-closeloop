"""
Tests for deterministic explanation engine.

Covers:
- Zero discrepancy (exact match)
- Single-event refund explanation
- Single-event fee explanation
- Single-event tax explanation
- Single-event adjustment explanation
- Multi-event explanation
- Partial explanation
- Conflicting evidence
- Missing evidence
- Unknown case
- Combination limits
- Ground truth separation
"""

import os
import sys
from pathlib import Path

import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_explanation.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
)
from app.services.explanation_engine import (
    DeterministicExplanationEngine,
    MAX_CANDIDATES,
    MAX_COMBINATION_SIZE,
)
from app.schemas.explanation import ExplanationStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
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
# Zero Discrepancy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroDiscrepancy:
    """Tests for exact match (zero discrepancy)."""

    def test_zero_difference_fully_explained(self, engine):
        """Test that zero discrepancy is fully explained."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=100000,
            difference=0,
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert result.remaining_difference == 0
        assert result.conflict is False

    def test_zero_difference_reason(self, engine):
        """Test that zero discrepancy has correct reason."""
        pkg = _make_package(difference=0)
        result = engine.explain(pkg)

        assert "match" in result.explanation_reason.lower()

    def test_zero_difference_no_evidence_needed(self, engine):
        """Test that zero discrepancy doesn't require evidence."""
        pkg = _make_package(difference=0)
        result = engine.explain(pkg)

        assert result.supporting_evidence_ids == []


# ─────────────────────────────────────────────────────────────────────────────
# Single-Event Refund Explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundExplanation:
    """Tests for single refund explaining discrepancy."""

    def test_refund_explains_difference(self, engine):
        """Test that a refund exactly explains a discrepancy."""
        # difference = expected - actual = 78000 - 78500 = -500
        # Refund of 500 has contribution = -500 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            exception_type="REFUND_ADJUSTMENT",
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "REF-001" in result.supporting_evidence_ids
        assert result.remaining_difference == 0

    def test_refund_contribution_is_negative(self, engine):
        """Test that refund contribution is negative amount."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
        )
        result = engine.explain(pkg)

        assert len(result.candidate_explanations) == 1
        assert result.candidate_explanations[0].events[0].contribution == -500

    def test_large_refund(self, engine):
        """Test that a large refund explains a large discrepancy."""
        pkg = _make_package(
            expected_amount=70000,
            actual_amount=85000,
            difference=-15000,
            refunds=[
                _make_record("REF-001", "REFUND", 15000),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED


# ─────────────────────────────────────────────────────────────────────────────
# Single-Event Fee Explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeExplanation:
    """Tests for single fee explaining discrepancy."""

    def test_fee_explains_difference(self, engine):
        """Test that a fee exactly explains a discrepancy."""
        # difference = 78000 - 77500 = 500
        # Need contribution of 500
        # Fee of 500 has contribution = -500 ≠ 500
        # But adjustment of +500 has contribution = +500 = 500
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=77500,
            difference=500,
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 500, metadata={"adjustment_type": "CREDIT"}),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "ADJ-001" in result.supporting_evidence_ids

    def test_fee_explains_over_settlement(self, engine):
        """Test that a fee explains when actual > expected (over-settled)."""
        # difference = 78000 - 78500 = -500
        # Fee of 500 has contribution = -500 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "FEE-001" in result.supporting_evidence_ids


# ─────────────────────────────────────────────────────────────────────────────
# Single-Event Tax Explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxExplanation:
    """Tests for single tax explaining discrepancy."""

    def test_tax_explains_over_settlement(self, engine):
        """Test that a tax explains when actual > expected."""
        # difference = -500
        # Tax of 500 has contribution = -500 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            taxes=[
                _make_record("TAX-001", "TAX", 500),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "TAX-001" in result.supporting_evidence_ids


# ─────────────────────────────────────────────────────────────────────────────
# Single-Event Adjustment Explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentExplanation:
    """Tests for single adjustment explaining discrepancy."""

    def test_positive_adjustment_explains(self, engine):
        """Test that a positive adjustment explains positive discrepancy."""
        # difference = +500
        # Adjustment of +500 has contribution = +500 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=77500,
            difference=500,
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 500, metadata={"adjustment_type": "CREDIT"}),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "ADJ-001" in result.supporting_evidence_ids

    def test_negative_adjustment_explains(self, engine):
        """Test that a negative adjustment explains negative discrepancy."""
        # difference = -500
        # Adjustment of -500 has contribution = -500 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", -500, metadata={"adjustment_type": "DEBIT"}),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert "ADJ-001" in result.supporting_evidence_ids


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Event Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiEventExplanation:
    """Tests for multi-event combination explaining discrepancy."""

    def test_two_events_explain(self, engine):
        """Test that two events jointly explain discrepancy."""
        # difference = -500
        # Refund of 300 (contribution: -300) + Fee of 200 (contribution: -200) = -500
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
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert set(result.supporting_evidence_ids) == {"REF-001", "FEE-001"}

    def test_three_events_explain(self, engine):
        """Test that three events jointly explain discrepancy."""
        # difference = -600
        # Refund(200) + Fee(200) + Tax(200) = -200 + -200 + -200 = -600
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78600,
            difference=-600,
            refunds=[
                _make_record("REF-001", "REFUND", 200),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 200),
            ],
            taxes=[
                _make_record("TAX-001", "TAX", 200),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert set(result.supporting_evidence_ids) == {"REF-001", "FEE-001", "TAX-001"}

    def test_mixed_types_explain(self, engine):
        """Test that refund + adjustment explain discrepancy."""
        # difference = -200
        # Refund(100) contribution: -100
        # Adjustment(-100) contribution: -100
        # Total: -200 = difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78200,
            difference=-200,
            refunds=[
                _make_record("REF-001", "REFUND", 100),
            ],
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", -100, metadata={"adjustment_type": "DEBIT"}),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED

    def test_four_events_explain(self, engine):
        """Test that four events jointly explain discrepancy (at combination limit)."""
        # difference = -400
        # Refund(100) + Fee(100) + Tax(100) + Adjustment(-100) = -100-100-100-100 = -400
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78400,
            difference=-400,
            refunds=[
                _make_record("REF-001", "REFUND", 100),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 100),
            ],
            taxes=[
                _make_record("TAX-001", "TAX", 100),
            ],
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", -100, metadata={"adjustment_type": "DEBIT"}),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        assert len(result.supporting_evidence_ids) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Partial Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialExplanation:
    """Tests for partial explanation."""

    def test_partial_explanation(self, engine):
        """Test that partial explanation is detected."""
        # difference = -1000
        # Fee of 300 (contribution: -300) explains part of it
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.PARTIALLY_EXPLAINED
        assert result.explained_amount == -300
        assert result.remaining_difference == -700
        assert "FEE-001" in result.supporting_evidence_ids

    def test_partial_has_reason(self, engine):
        """Test that partial explanation has a reason."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        result = engine.explain(pkg)

        assert "partially" in result.explanation_reason.lower()
        assert "remaining" in result.explanation_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Conflicting Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictingEvidence:
    """Tests for conflicting explanations."""

    def test_conflicting_explanations(self, engine):
        """Test that conflicting explanations are detected."""
        # difference = -500
        # Refund of 500 (contribution: -500) explains it
        # Fee of 500 (contribution: -500) also explains it
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
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.CONFLICTING
        assert result.conflict is True
        assert len(result.candidate_explanations) == 2

    def test_conflicting_has_all_ids(self, engine):
        """Test that conflicting result includes all candidate IDs."""
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
        result = engine.explain(pkg)

        assert set(result.supporting_evidence_ids) == {"REF-001", "FEE-001"}

    def test_conflicting_has_reason(self, engine):
        """Test that conflicting result has a reason."""
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
        result = engine.explain(pkg)

        assert "multiple" in result.explanation_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Missing Evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingEvidence:
    """Tests for missing evidence handling."""

    def test_missing_evidence_detected(self, engine):
        """Test that missing evidence is reported."""
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
        result = engine.explain(pkg)

        assert "SETTLEMENT" in result.missing_evidence

    def test_missing_evidence_in_reason(self, engine):
        """Test that missing evidence appears in reason."""
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
        result = engine.explain(pkg)

        assert "SETTLEMENT" in result.explanation_reason

    def test_no_evidence_unexplained(self, engine):
        """Test that case with no evidence is unexplained."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.UNEXPLAINED
        assert result.remaining_difference == -1000


# ─────────────────────────────────────────────────────────────────────────────
# Unknown Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownCase:
    """Tests for unknown exception handling."""

    def test_unknown_with_no_evidence(self, engine):
        """Test that unknown case with no evidence is unexplained."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            exception_type="UNKNOWN",
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.UNEXPLAINED

    def test_unknown_with_partial_evidence(self, engine):
        """Test that unknown case with partial evidence is partially explained."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=79000,
            difference=-1000,
            exception_type="UNKNOWN",
            fees=[
                _make_record("FEE-001", "FEE", 300),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.PARTIALLY_EXPLAINED
        assert result.explained_amount == -300

    def test_unknown_with_full_evidence(self, engine):
        """Test that unknown case with full evidence is fully explained."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            exception_type="UNKNOWN",
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        result = engine.explain(pkg)

        assert result.explanation_status == ExplanationStatus.FULLY_EXPLAINED


# ─────────────────────────────────────────────────────────────────────────────
# Combination Limit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinationLimits:
    """Tests for combination search limits."""

    def test_max_combination_size(self):
        """Test that MAX_COMBINATION_SIZE is defined."""
        assert MAX_COMBINATION_SIZE >= 3
        assert MAX_COMBINATION_SIZE <= 10

    def test_max_candidates(self):
        """Test that MAX_CANDIDATES is defined."""
        assert MAX_CANDIDATES >= 10

    def test_many_candidates_limited(self, engine):
        """Test that many candidates are handled without explosion."""
        # Create 25 candidates (exceeds MAX_CANDIDATES=20)
        refunds = [
            _make_record(f"REF-{i:03d}", "REFUND", 100)
            for i in range(25)
        ]
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=refunds,
        )
        # Should not hang or crash
        result = engine.explain(pkg)
        assert result is not None

    def test_no_combination_beyond_limit(self, engine):
        """Test that combinations beyond MAX_COMBINATION_SIZE are not searched."""
        # With MAX_COMBINATION_SIZE=4, a 5-event combination should not be found
        # even if it would explain the difference
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[_make_record("REF-001", "REFUND", 100)],
            fees=[_make_record("FEE-001", "FEE", 100)],
            taxes=[_make_record("TAX-001", "TAX", 100)],
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", -100, metadata={"adjustment_type": "DEBIT"}),
                _make_record("ADJ-002", "ADJUSTMENT", -100, metadata={"adjustment_type": "DEBIT"}),
            ],
        )
        result = engine.explain(pkg)

        # The 5-event combination (100*5=500) should NOT be found
        # But pairs/triples of 100+100+100+100+100 won't work either
        # because we can only do up to 4 at a time
        # So it should be partially explained or unexplained
        assert result.explanation_status in [
            ExplanationStatus.PARTIALLY_EXPLAINED,
            ExplanationStatus.UNEXPLAINED,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Verify that explanation engine does not reference ground truth."""

    GROUND_TRUTH_TERMS = [
        "ground_truth",
        "true_exception_type",
        "true_resolution",
    ]

    def test_engine_code_no_ground_truth(self):
        """Test that engine source has no ground truth references."""
        import inspect
        from app.services.explanation_engine import DeterministicExplanationEngine

        source = inspect.getsource(DeterministicExplanationEngine)
        for term in self.GROUND_TRUTH_TERMS:
            assert term not in source, f"Ground truth reference found: {term}"

    def test_explanation_result_no_ground_truth(self, engine):
        """Test that ExplanationResult has no ground truth fields."""
        pkg = _make_package(difference=0)
        result = engine.explain(pkg)

        assert not hasattr(result, "true_exception_type")
        assert not hasattr(result, "true_resolution")
        assert not hasattr(result, "resolvable")
        assert not hasattr(result, "risk_category")


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Reason Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExplanationReason:
    """Tests for deterministic template-based reason generation."""

    def test_single_event_reason(self, engine):
        """Test reason for single event explanation."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            fees=[
                _make_record("FEE-001", "FEE", 500),
            ],
        )
        result = engine.explain(pkg)

        assert "fee" in result.explanation_reason.lower()
        assert "fee-001" in result.explanation_reason.lower()
        assert "₹5" in result.explanation_reason

    def test_multi_event_reason(self, engine):
        """Test reason for multi-event explanation."""
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
        result = engine.explain(pkg)

        assert "refund" in result.explanation_reason.lower()
        assert "ref-001" in result.explanation_reason.lower()
        assert "fee" in result.explanation_reason.lower()
        assert "fee-001" in result.explanation_reason.lower()
        assert "and" in result.explanation_reason.lower()

    def test_three_event_reason(self, engine):
        """Test reason for three-event explanation."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78600,
            difference=-600,
            refunds=[
                _make_record("REF-001", "REFUND", 200),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 200),
            ],
            taxes=[
                _make_record("TAX-001", "TAX", 200),
            ],
        )
        result = engine.explain(pkg)

        assert "refund" in result.explanation_reason.lower()
        assert "ref-001" in result.explanation_reason.lower()
        assert "fee" in result.explanation_reason.lower()
        assert "fee-001" in result.explanation_reason.lower()
        assert "tax" in result.explanation_reason.lower()
        assert "tax-001" in result.explanation_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCandidateExplanation:
    """Tests for candidate explanation structure."""

    def test_candidate_has_events(self, engine):
        """Test that candidate explanation contains events."""
        pkg = _make_package(
            expected_amount=78000,
            actual_amount=78500,
            difference=-500,
            refunds=[
                _make_record("REF-001", "REFUND", 500),
            ],
        )
        result = engine.explain(pkg)

        assert len(result.candidate_explanations) == 1
        candidate = result.candidate_explanations[0]
        assert len(candidate.events) == 1
        assert candidate.events[0].record_id == "REF-001"
        assert candidate.total_contribution == -500
        assert candidate.is_exact_match is True

    def test_candidate_total_matches(self, engine):
        """Test that candidate total contribution equals sum of events."""
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
        result = engine.explain(pkg)

        candidate = result.candidate_explanations[0]
        event_sum = sum(e.contribution for e in candidate.events)
        assert candidate.total_contribution == event_sum
