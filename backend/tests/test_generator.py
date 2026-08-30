"""
Unit tests for the synthetic financial data generator.

Tests cover:
- Deterministic seed behavior
- Unique IDs
- Relationship integrity
- Amount generation
- Timestamp generation
- Every exception scenario
- Ground truth consistency
- No data leakage
- Deterministic seed behavior
"""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generator.adjustments import generate_adjustments
from app.generator.cases import generate_cases
from app.generator.fees import generate_fees
from app.generator.merchants import generate_merchants
from app.generator.payments import generate_payments
from app.generator.refunds import generate_refunds
from app.generator.rng import DeterministicRNG
from app.generator.settlements import generate_settlements
from app.generator.taxes import generate_taxes
from app.generator.validation import validate_dataset
from app.schemas.case import GroundTruth
from app.schemas.config import GeneratorConfig
from app.schemas.enums import ExceptionType, ResolutionType, RiskCategory


@pytest.fixture
def small_config():
    """Configuration for small test datasets."""
    return GeneratorConfig(
        random_seed=42,
        num_merchants=5,
        num_cases=20,
    )


@pytest.fixture
def rng():
    """Deterministic RNG with fixed seed."""
    return DeterministicRNG(42)


def _generate_all(small_config, rng=None):
    """Helper to generate all entities for testing."""
    if rng is None:
        rng = DeterministicRNG(small_config.random_seed)
    merchants = generate_merchants(small_config, rng)
    payments = generate_payments(small_config, merchants, rng)
    settlements = generate_settlements(small_config, payments, merchants, rng)
    refunds = generate_refunds(small_config, payments, rng)
    fees = generate_fees(small_config, payments, rng)
    taxes = generate_taxes(small_config, payments, rng)
    adjustments = generate_adjustments(small_config, payments, rng)
    cases, ground_truth = generate_cases(
        small_config, payments, refunds, fees, taxes, adjustments, settlements, rng
    )
    return {
        "merchants": merchants,
        "payments": payments,
        "settlements": settlements,
        "refunds": refunds,
        "fees": fees,
        "taxes": taxes,
        "adjustments": adjustments,
        "cases": cases,
        "ground_truth": ground_truth,
    }


class TestDeterministicRNG:
    """Tests for deterministic random number generation."""

    def test_same_seed_produces_same_output(self):
        """Same seed should produce identical sequences."""
        rng1 = DeterministicRNG(42)
        rng2 = DeterministicRNG(42)

        seq1 = [rng1.randint(1, 100) for _ in range(100)]
        seq2 = [rng2.randint(1, 100) for _ in range(100)]

        assert seq1 == seq2

    def test_different_seeds_produce_different_output(self):
        """Different seeds should produce different sequences."""
        rng1 = DeterministicRNG(42)
        rng2 = DeterministicRNG(123)

        seq1 = [rng1.randint(1, 100) for _ in range(100)]
        seq2 = [rng2.randint(1, 100) for _ in range(100)]

        assert seq1 != seq2

    def test_random_timestamp_within_range(self, rng):
        """Timestamps should fall within the specified range."""
        from datetime import datetime

        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 30)

        for _ in range(50):
            ts = rng.random_timestamp(start, end)
            assert start <= ts <= end

    def test_random_amount_within_range(self, rng):
        """Amounts should fall within the specified range."""
        for _ in range(50):
            amount = rng.random_amount(10000, 1000000)
            assert 10000 <= amount <= 1000000

    def test_should_trigger_respects_probability(self, rng):
        """should_trigger should respect the given probability."""
        for _ in range(10):
            assert rng.should_trigger(1.0)
        for _ in range(10):
            assert not rng.should_trigger(0.0)


class TestMerchantGeneration:
    """Tests for merchant generation."""

    def test_generates_correct_count(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        assert len(merchants) == small_config.num_merchants

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        ids = [m.merchant_id for m in merchants]
        assert len(ids) == len(set(ids))

    def test_valid_id_format(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        for m in merchants:
            assert m.merchant_id.startswith("MER-")
            assert len(m.merchant_id) == 8

    def test_deterministic_generation(self, small_config):
        rng1 = DeterministicRNG(small_config.random_seed)
        rng2 = DeterministicRNG(small_config.random_seed)
        merchants1 = generate_merchants(small_config, rng1)
        merchants2 = generate_merchants(small_config, rng2)
        for m1, m2 in zip(merchants1, merchants2):
            assert m1.merchant_id == m2.merchant_id
            assert m1.name == m2.name


class TestPaymentGeneration:
    """Tests for payment generation."""

    def test_generates_correct_count(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        assert len(payments) == small_config.num_cases

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        ids = [p.payment_id for p in payments]
        assert len(ids) == len(set(ids))

    def test_valid_merchant_references(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        merchant_ids = {m.merchant_id for m in merchants}
        for p in payments:
            assert p.merchant_id in merchant_ids

    def test_positive_amounts(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        for p in payments:
            assert p.amount > 0


class TestSettlementGeneration:
    """Tests for settlement generation."""

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        settlements = generate_settlements(small_config, payments, merchants, rng)
        ids = [s.settlement_id for s in settlements]
        assert len(ids) == len(set(ids))

    def test_valid_payment_references(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        settlements = generate_settlements(small_config, payments, merchants, rng)
        payment_ids = {p.payment_id for p in payments}
        for s in settlements:
            assert s.payment_id in payment_ids


class TestRefundGeneration:
    """Tests for refund generation."""

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        refunds = generate_refunds(small_config, payments, rng)
        ids = [r.refund_id for r in refunds]
        assert len(ids) == len(set(ids))

    def test_positive_amounts(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        refunds = generate_refunds(small_config, payments, rng)
        for r in refunds:
            assert r.amount > 0


class TestFeeGeneration:
    """Tests for fee generation."""

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        fees = generate_fees(small_config, payments, rng)
        ids = [f.fee_id for f in fees]
        assert len(ids) == len(set(ids))

    def test_positive_amounts(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        fees = generate_fees(small_config, payments, rng)
        for f in fees:
            assert f.amount > 0


class TestTaxGeneration:
    """Tests for tax generation."""

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        taxes = generate_taxes(small_config, payments, rng)
        ids = [t.tax_id for t in taxes]
        assert len(ids) == len(set(ids))

    def test_positive_amounts(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        taxes = generate_taxes(small_config, payments, rng)
        for t in taxes:
            assert t.amount > 0


class TestAdjustmentGeneration:
    """Tests for adjustment generation."""

    def test_unique_ids(self, small_config, rng):
        merchants = generate_merchants(small_config, rng)
        payments = generate_payments(small_config, merchants, rng)
        adjustments = generate_adjustments(small_config, payments, rng)
        ids = [a.adjustment_id for a in adjustments]
        assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExactMatchScenario:
    """Tests for EXACT_MATCH scenario."""

    def test_exact_match_has_zero_difference(self):
        """EXACT_MATCH cases should have difference = 0."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        exact_match_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.EXACT_MATCH
        ]
        assert len(exact_match_cases) > 0

        for case in exact_match_cases:
            assert case.difference == 0
            assert case.expected_amount == case.actual_amount
            assert case.resolvable is True
            assert case.risk_category == RiskCategory.LOW

    def test_exact_match_ground_truth(self):
        """EXACT_MATCH ground truth should have matching expected and actual."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        exact_match_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.EXACT_MATCH
        ]
        assert len(exact_match_gt) > 0

        for gt in exact_match_gt:
            assert gt.expected_amount == gt.actual_amount
            assert gt.difference == 0
            assert gt.true_resolution == ResolutionType.NO_ACTION
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestFeeDifferenceScenario:
    """Tests for FEE_DIFFERENCE scenario."""

    def test_fee_difference_creates_discrepancy(self):
        """FEE_DIFFERENCE cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        fee_diff_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.FEE_DIFFERENCE
        ]
        assert len(fee_diff_cases) > 0

        for case in fee_diff_cases:
            assert case.difference != 0
            assert case.resolvable is True
            assert case.risk_category in (RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH)

    def test_fee_difference_ground_truth(self):
        """FEE_DIFFERENCE ground truth should identify fee adjustment."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        fee_diff_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.FEE_DIFFERENCE
        ]
        assert len(fee_diff_gt) > 0

        for gt in fee_diff_gt:
            assert gt.true_resolution == ResolutionType.FEE_ADJUSTMENT
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestRefundAdjustmentScenario:
    """Tests for REFUND_ADJUSTMENT scenario."""

    def test_refund_adjustment_creates_discrepancy(self):
        """REFUND_ADJUSTMENT cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=500,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        refund_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.REFUND_ADJUSTMENT
        ]
        # May be 0 if no payments have refunds assigned to this scenario
        for case in refund_cases:
            assert case.difference != 0
            assert case.resolvable is True

    def test_refund_adjustment_ground_truth(self):
        """REFUND_ADJUSTMENT ground truth should identify refund adjustment."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=500,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        refund_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.REFUND_ADJUSTMENT
        ]
        for gt in refund_gt:
            assert gt.true_resolution == ResolutionType.REFUND_ADJUSTMENT
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestTaxAdjustmentScenario:
    """Tests for TAX_ADJUSTMENT scenario."""

    def test_tax_adjustment_creates_discrepancy(self):
        """TAX_ADJUSTMENT cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        tax_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.TAX_ADJUSTMENT
        ]
        assert len(tax_cases) > 0

        for case in tax_cases:
            assert case.difference != 0
            assert case.resolvable is True

    def test_tax_adjustment_ground_truth(self):
        """TAX_ADJUSTMENT ground truth should identify tax adjustment."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        tax_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.TAX_ADJUSTMENT
        ]
        assert len(tax_gt) > 0

        for gt in tax_gt:
            assert gt.true_resolution == ResolutionType.TAX_ADJUSTMENT
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestTimingDifferenceScenario:
    """Tests for TIMING_DIFFERENCE scenario."""

    def test_timing_difference_creates_discrepancy(self):
        """TIMING_DIFFERENCE cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        timing_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.TIMING_DIFFERENCE
        ]
        assert len(timing_cases) > 0

        for case in timing_cases:
            assert case.difference != 0
            assert case.resolvable is True

    def test_timing_difference_ground_truth(self):
        """TIMING_DIFFERENCE ground truth should identify timing reconciliation."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        timing_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.TIMING_DIFFERENCE
        ]
        assert len(timing_gt) > 0

        for gt in timing_gt:
            assert gt.true_resolution == ResolutionType.TIMING_RECONCILIATION
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestPartialSettlementScenario:
    """Tests for PARTIAL_SETTLEMENT scenario."""

    def test_partial_settlement_creates_discrepancy(self):
        """PARTIAL_SETTLEMENT cases should have actual < expected."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        partial_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.PARTIAL_SETTLEMENT
        ]
        assert len(partial_cases) > 0

        for case in partial_cases:
            assert case.actual_amount < case.expected_amount
            assert case.difference < 0
            assert case.resolvable is True

    def test_partial_settlement_ground_truth(self):
        """PARTIAL_SETTLEMENT ground truth should identify partial reconciliation."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        partial_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.PARTIAL_SETTLEMENT
        ]
        assert len(partial_gt) > 0

        for gt in partial_gt:
            assert gt.true_resolution == ResolutionType.PARTIAL_SETTLEMENT_RECONCILIATION
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestDuplicateScenario:
    """Tests for DUPLICATE scenario."""

    def test_duplicate_creates_discrepancy(self):
        """DUPLICATE cases should have actual > expected."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        dup_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.DUPLICATE
        ]
        assert len(dup_cases) > 0

        for case in dup_cases:
            assert case.actual_amount > case.expected_amount
            assert case.difference > 0
            assert case.resolvable is True
            assert case.risk_category == RiskCategory.HIGH

    def test_duplicate_ground_truth(self):
        """DUPLICATE ground truth should identify duplicate settlement."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        dup_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.DUPLICATE
        ]
        assert len(dup_gt) > 0

        for gt in dup_gt:
            assert gt.true_resolution == ResolutionType.DUPLICATE_SETTLEMENT
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestMissingRecordScenario:
    """Tests for MISSING_RECORD scenario."""

    def test_missing_record_creates_discrepancy(self):
        """MISSING_RECORD cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        missing_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.MISSING_RECORD
        ]
        assert len(missing_cases) > 0

        for case in missing_cases:
            assert case.difference != 0
            assert case.resolvable is False
            assert case.risk_category == RiskCategory.HIGH

    def test_missing_record_ground_truth(self):
        """MISSING_RECORD ground truth should identify missing record escalation."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        missing_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.MISSING_RECORD
        ]
        assert len(missing_gt) > 0

        for gt in missing_gt:
            assert gt.true_resolution == ResolutionType.MISSING_RECORD_ESCALATION
            assert gt.resolvable is False
            assert gt.verify_expected_amount()


class TestComplexMultiAdjustmentScenario:
    """Tests for COMPLEX_MULTI_ADJUSTMENT scenario."""

    def test_complex_multi_creates_discrepancy(self):
        """COMPLEX_MULTI_ADJUSTMENT cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=500,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        complex_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.COMPLEX_MULTI_ADJUSTMENT
        ]
        # May be 0 depending on distribution
        for case in complex_cases:
            assert case.difference != 0
            assert case.resolvable is True

    def test_complex_multi_ground_truth(self):
        """COMPLEX_MULTI_ADJUSTMENT ground truth should identify multi-adjustment."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=500,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        complex_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.COMPLEX_MULTI_ADJUSTMENT
        ]
        for gt in complex_gt:
            assert gt.true_resolution == ResolutionType.MULTI_ADJUSTMENT
            assert gt.resolvable is True
            assert gt.verify_expected_amount()


class TestUnknownScenario:
    """Tests for UNKNOWN scenario."""

    def test_unknown_creates_discrepancy(self):
        """UNKNOWN cases should have non-zero difference."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        unknown_cases = [
            c for c in data["cases"]
            if c.scenario == ExceptionType.UNKNOWN
        ]
        assert len(unknown_cases) > 0

        for case in unknown_cases:
            assert case.difference != 0
            assert case.resolvable is False
            assert case.risk_category == RiskCategory.HIGH

    def test_unknown_ground_truth(self):
        """UNKNOWN ground truth should have UNKNOWN_UNRESOLVED resolution."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=200,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        unknown_gt = [
            gt for gt in data["ground_truth"]
            if gt.true_exception_type == ExceptionType.UNKNOWN
        ]
        assert len(unknown_gt) > 0

        for gt in unknown_gt:
            assert gt.true_resolution == ResolutionType.UNKNOWN_UNRESOLVED
            assert gt.resolvable is False
            assert gt.verify_expected_amount()


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundTruthConsistency:
    """Tests for ground truth consistency across all scenarios."""

    def test_all_ground_truth_records_are_valid(self):
        """All ground truth records should pass verification."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=100,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        for gt in data["ground_truth"]:
            assert gt.verify_expected_amount(), (
                f"Ground truth {gt.case_id} failed verification"
            )

    def test_difference_equals_actual_minus_expected(self):
        """All cases should have difference = actual - expected."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=100,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        for case in data["cases"]:
            assert case.difference == case.actual_amount - case.expected_amount

    def test_ground_truth_matches_case(self):
        """Ground truth should match case records."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=100,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        case_map = {c.case_id: c for c in data["cases"]}
        for gt in data["ground_truth"]:
            case = case_map[gt.case_id]
            assert case.expected_amount == gt.expected_amount
            assert case.actual_amount == gt.actual_amount
            assert case.scenario == gt.true_exception_type


class TestDataLeakage:
    """Tests for no data leakage between ground truth and input features."""

    def test_ground_truth_not_in_payment_records(self):
        """Ground truth labels should not appear in payment records."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=50,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        # Payments should not have exception_type, resolution, etc.
        for payment in data["payments"]:
            assert not hasattr(payment, "true_exception_type")
            assert not hasattr(payment, "true_resolution")
            assert not hasattr(payment, "resolvable")

    def test_ground_truth_not_in_settlement_records(self):
        """Ground truth labels should not appear in settlement records."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=50,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        for settlement in data["settlements"]:
            assert not hasattr(settlement, "true_exception_type")
            assert not hasattr(settlement, "true_resolution")

    def test_ground_truth_separate_from_financial_records(self):
        """Ground truth should be in separate list from financial records."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=50,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        # Ground truth is a separate list
        assert "ground_truth" in data
        assert isinstance(data["ground_truth"], list)
        assert len(data["ground_truth"]) == len(data["cases"])

        # Verify ground truth items are GroundTruth instances
        for gt in data["ground_truth"]:
            assert isinstance(gt, GroundTruth)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Seed Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicGeneration:
    """Tests for deterministic dataset generation."""

    def test_same_seed_produces_same_dataset(self):
        """Same seed should produce identical datasets."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=50,
        )

        data1 = _generate_all(config)
        data2 = _generate_all(config)

        assert len(data1["cases"]) == len(data2["cases"])
        assert len(data1["ground_truth"]) == len(data2["ground_truth"])

        for c1, c2 in zip(data1["cases"], data2["cases"]):
            assert c1.case_id == c2.case_id
            assert c1.expected_amount == c2.expected_amount
            assert c1.actual_amount == c2.actual_amount
            assert c1.scenario == c2.scenario

        for gt1, gt2 in zip(data1["ground_truth"], data2["ground_truth"]):
            assert gt1.case_id == gt2.case_id
            assert gt1.expected_amount == gt2.expected_amount
            assert gt1.true_exception_type == gt2.true_exception_type

    def test_different_seeds_produce_different_datasets(self):
        """Different seeds should produce different datasets."""
        config1 = GeneratorConfig(random_seed=42, num_merchants=5, num_cases=50)
        config2 = GeneratorConfig(random_seed=99, num_merchants=5, num_cases=50)

        data1 = _generate_all(config1)
        data2 = _generate_all(config2)

        # At least some cases should differ
        different = False
        for c1, c2 in zip(data1["cases"], data2["cases"]):
            if c1.scenario != c2.scenario or c1.expected_amount != c2.expected_amount:
                different = True
                break
        assert different


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Distribution Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioDistribution:
    """Tests for configurable scenario distribution."""

    def test_all_scenarios_represented_in_large_dataset(self):
        """With enough cases, all exception types should appear."""
        config = GeneratorConfig(
            random_seed=42,
            num_merchants=10,
            num_cases=1000,
        )
        rng = DeterministicRNG(config.random_seed)
        data = _generate_all(config, rng)

        scenarios = {c.scenario for c in data["cases"]}
        for exc_type in ExceptionType:
            assert exc_type in scenarios, f"Missing exception type: {exc_type}"


# ─────────────────────────────────────────────────────────────────────────────
# Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    """Tests for dataset validation."""

    def test_valid_dataset_passes_validation(self, small_config, rng):
        """A properly generated dataset should pass validation."""
        data = _generate_all(small_config, rng)

        results = validate_dataset(
            merchants=data["merchants"],
            payments=data["payments"],
            settlements=data["settlements"],
            refunds=data["refunds"],
            fees=data["fees"],
            taxes=data["taxes"],
            adjustments=data["adjustments"],
            cases=data["cases"],
            ground_truth_records=data["ground_truth"],
        )

        assert results["valid"], f"Validation failed: {results['errors']}"
        assert results["error_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestSmallDatasetGeneration:
    """Integration test for complete small dataset generation."""

    def test_generate_small_dataset(self):
        """Generate a complete small dataset and verify integrity."""
        from app.generator.orchestrator import DatasetGenerator

        config = GeneratorConfig(
            random_seed=42,
            num_merchants=5,
            num_cases=50,
        )

        generator = DatasetGenerator(config)
        dataset = generator.generate()

        # Verify all entities are present
        assert "merchants" in dataset
        assert "payments" in dataset
        assert "settlements" in dataset
        assert "refunds" in dataset
        assert "fees" in dataset
        assert "taxes" in dataset
        assert "adjustments" in dataset
        assert "cases" in dataset
        assert "ground_truth" in dataset
        assert "manifest" in dataset

        # Verify counts
        manifest = dataset["manifest"]
        assert manifest["counts"]["merchants"] == 5
        assert manifest["counts"]["payments"] == 50
        assert manifest["counts"]["cases"] == 50

        # Verify validation passed
        assert manifest["validation"]["valid"]

        # Verify ground truth verification works
        for gt in dataset["ground_truth"]:
            assert gt.verify_expected_amount()

        # Verify scenario distribution is in manifest
        assert "scenario_distribution" in manifest
        assert "risk_distribution" in manifest
        assert "resolvable_count" in manifest
        assert "unresolved_count" in manifest
