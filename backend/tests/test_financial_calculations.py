"""
Comprehensive unit tests for all deterministic financial calculation functions.

Tests verify exact financial values using integer paise arithmetic.
No floating-point comparisons for financial values.

Financial rules tested:
- Expected settlement: payment - refunds - fees - taxes + adjustments
- Difference: expected - actual
- Match status determination
- Exception type classification
- Financial risk classification
- Reward category determination
- Aggregation functions (refunds, fees, taxes, adjustments, settlements)
- Calculation breakdown
"""

import pytest
from datetime import datetime, timezone

from app.reconciliation.contract import (
    aggregate_financial_records,
    calculate_difference,
    calculate_expected_amount,
    classify_exception_type,
    create_calculation_breakdown,
    determine_match_status,
)
from app.reconciliation.engine import (
    aggregate_adjustments,
    aggregate_fees,
    aggregate_refunds,
    aggregate_settlements,
    aggregate_taxes,
)
from app.schemas.enums import (
    AdjustmentType,
    ExceptionType,
    FeeType,
    MatchStatus,
    TaxType,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
    Tax,
)
from app.schemas.reconciliation import CalculationBreakdown
from app.services.reward_engine import (
    RewardEngine,
    classify_financial_risk,
    determine_reward_category,
)
from app.schemas.reward_engine import (
    FinancialRiskLevel,
    RewardCategory,
    RewardConfig,
)
from app.schemas.feedback import (
    ActualOutcomeRecord,
    DataLineage,
    FeedbackRecord,
    FeedbackType,
    FinancialImpact,
    OutcomeRecord,
    PredictionRecord,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def _make_payment(payment_id="PAY-001", amount=100000, merchant_id="MER-001"):
    """Create a test payment."""
    return Payment(
        payment_id=payment_id,
        merchant_id=merchant_id,
        amount=amount,
        payment_timestamp=datetime.now(timezone.utc),
    )


def _make_settlement(settlement_id="SET-001", payment_id="PAY-001", amount=100000):
    """Create a test settlement."""
    return Settlement(
        settlement_id=settlement_id,
        payment_id=payment_id,
        merchant_id="MER-001",
        amount=amount,
        settlement_timestamp=datetime.now(timezone.utc),
    )


def _make_refund(refund_id="REF-001", payment_id="PAY-001", amount=5000):
    """Create a test refund."""
    return Refund(
        refund_id=refund_id,
        payment_id=payment_id,
        amount=amount,
        refund_timestamp=datetime.now(timezone.utc),
    )


def _make_fee(fee_id="FEE-001", payment_id="PAY-001", amount=2000):
    """Create a test fee."""
    return Fee(
        fee_id=fee_id,
        payment_id=payment_id,
        amount=amount,
        fee_type=FeeType.TRANSACTION,
    )


def _make_tax(tax_id="TAX-001", payment_id="PAY-001", amount=1500):
    """Create a test tax."""
    return Tax(
        tax_id=tax_id,
        payment_id=payment_id,
        amount=amount,
        tax_type=TaxType.GST,
    )


def _make_adjustment(adj_id="ADJ-001", payment_id="PAY-001", amount=3000,
                     adj_type=AdjustmentType.CREDIT):
    """Create a test adjustment."""
    return Adjustment(
        adjustment_id=adj_id,
        payment_id=payment_id,
        amount=amount,
        adjustment_type=adj_type,
    )


def _make_outcome(
    workflow_id="WF-001",
    exception_id="EXC-001",
    decision="AUTO",
    was_executed=True,
    verification_passed=True,
    was_rolled_back=False,
    resolution_correct=True,
    adjustment_paise=50000,
    discrepancy_eliminated=True,
    unintended_changes=0,
    confidence=0.9,
):
    """Create a test OutcomeRecord."""
    return OutcomeRecord(
        outcome_id="OUT-001",
        workflow_id=workflow_id,
        exception_id=exception_id,
        case_id="CASE-001",
        decision=decision,
        prediction=PredictionRecord(
            resolution_type="FEE_ADJUSTMENT",
            resolution_confidence=confidence,
        ),
        actual_outcome=ActualOutcomeRecord(
            resolution_correct=resolution_correct,
            was_executed=was_executed,
            was_rolled_back=was_rolled_back,
        ),
        verification_passed=verification_passed,
        financial_impact=FinancialImpact(
            actual_adjustment_paise=adjustment_paise,
            requested_adjustment_paise=adjustment_paise,
            discrepancy_eliminated=discrepancy_eliminated,
            difference_after_paise=0 if discrepancy_eliminated else adjustment_paise,
            unintended_changes=unintended_changes,
        ),
        lineage=DataLineage(exception_id=exception_id),
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXPECTED AMOUNT CALCULATION
# Formula: payment - refunds - fees - taxes + adjustments
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpectedAmountCalculation:
    """Tests for calculate_expected_amount: payment - refunds - fees - taxes + adjustments."""

    def test_basic_calculation(self):
        """100000 - 5000 - 2000 - 1500 + 3000 = 94500 paise."""
        result = calculate_expected_amount(
            payment_amount=100000,
            total_refunds=5000,
            total_fees=2000,
            total_taxes=1500,
            total_adjustments=3000,
        )
        assert result == 94500

    def test_exact_match_no_deductions(self):
        """100000 - 0 - 0 - 0 + 0 = 100000 paise (exact match)."""
        result = calculate_expected_amount(
            payment_amount=100000,
            total_refunds=0,
            total_fees=0,
            total_taxes=0,
            total_adjustments=0,
        )
        assert result == 100000

    def test_zero_payment(self):
        """0 - 0 - 0 - 0 + 0 = 0 paise."""
        result = calculate_expected_amount(0, 0, 0, 0, 0)
        assert result == 0

    def test_large_payment(self):
        """10000000 - 500000 - 100000 - 50000 + 200000 = 9550000 paise (₹100,000)."""
        result = calculate_expected_amount(
            payment_amount=10_000_000,
            total_refunds=500_000,
            total_fees=100_000,
            total_taxes=50_000,
            total_adjustments=200_000,
        )
        assert result == 9_550_000

    def test_only_refunds(self):
        """100000 - 30000 - 0 - 0 + 0 = 70000 paise."""
        result = calculate_expected_amount(100000, 30000, 0, 0, 0)
        assert result == 70000

    def test_only_fees(self):
        """100000 - 0 - 5000 - 0 + 0 = 95000 paise."""
        result = calculate_expected_amount(100000, 0, 5000, 0, 0)
        assert result == 95000

    def test_only_taxes(self):
        """100000 - 0 - 0 - 8000 + 0 = 92000 paise."""
        result = calculate_expected_amount(100000, 0, 0, 8000, 0)
        assert result == 92000

    def test_only_adjustments(self):
        """100000 - 0 - 0 - 0 + 15000 = 115000 paise."""
        result = calculate_expected_amount(100000, 0, 0, 0, 15000)
        assert result == 115000

    def test_all_deductions_equal_payment(self):
        """100000 - 25000 - 25000 - 25000 + 0 = 25000 paise."""
        result = calculate_expected_amount(100000, 25000, 25000, 25000, 0)
        assert result == 25000

    def test_refunds_exceed_payment(self):
        """5000 - 10000 - 0 - 0 + 0 = -5000 paise (over-refunded)."""
        result = calculate_expected_amount(5000, 10000, 0, 0, 0)
        assert result == -5000

    def test_all_deductions_exceed_payment(self):
        """10000 - 5000 - 3000 - 4000 + 0 = -2000 paise."""
        result = calculate_expected_amount(10000, 5000, 3000, 4000, 0)
        assert result == -2000

    def test_negative_adjustment_debit(self):
        """100000 - 0 - 0 - 0 + (-5000) = 95000 paise (debit adjustment)."""
        result = calculate_expected_amount(100000, 0, 0, 0, -5000)
        assert result == 95000

    def test_multiple_refunds(self):
        """100000 - (3000+2000+1000) - 0 - 0 + 0 = 94000 paise."""
        result = calculate_expected_amount(100000, 6000, 0, 0, 0)
        assert result == 94000

    def test_multiple_fees(self):
        """100000 - 0 - (1000+500+200) - 0 + 0 = 98300 paise."""
        result = calculate_expected_amount(100000, 0, 1700, 0, 0)
        assert result == 98300

    def test_multiple_taxes(self):
        """100000 - 0 - 0 - (2000+1000) + 0 = 97000 paise."""
        result = calculate_expected_amount(100000, 0, 0, 3000, 0)
        assert result == 97000

    def test_multiple_adjustments(self):
        """100000 - 0 - 0 - 0 + (5000 + (-2000) + 1000) = 104000 paise."""
        result = calculate_expected_amount(100000, 0, 0, 0, 4000)
        assert result == 104000

    def test_partial_settlement_scenario(self):
        """Payment 100000, settled 60000: expected = 100000, actual = 60000, diff = 40000."""
        expected = calculate_expected_amount(100000, 0, 0, 0, 0)
        assert expected == 100000
        actual = 60000
        diff = calculate_difference(expected, actual)
        assert diff == 40000  # Under-settled by ₹400

    def test_zero_refunds_zero_fees_zero_taxes(self):
        """100000 - 0 - 0 - 0 + 5000 = 105000 paise."""
        result = calculate_expected_amount(100000, 0, 0, 0, 5000)
        assert result == 105000

    def test_minimal_amounts(self):
        """1 paise payment, 1 paise refund: 1 - 1 - 0 - 0 + 0 = 0."""
        result = calculate_expected_amount(1, 1, 0, 0, 0)
        assert result == 0

    def test_all_fields_nonzero(self):
        """200000 - 10000 - 5000 - 3000 + 8000 = 190000 paise."""
        result = calculate_expected_amount(200000, 10000, 5000, 3000, 8000)
        assert result == 190000


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DIFFERENCE CALCULATION
# Formula: expected - actual
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferenceCalculation:
    """Tests for calculate_difference: expected_amount - actual_amount."""

    def test_exact_match(self):
        """Same expected and actual = 0 difference."""
        assert calculate_difference(100000, 100000) == 0

    def test_under_settled(self):
        """Expected > actual = positive difference (under-settled)."""
        assert calculate_difference(100000, 80000) == 20000

    def test_over_settled(self):
        """Expected < actual = negative difference (over-settled)."""
        assert calculate_difference(80000, 100000) == -20000

    def test_zero_expected_zero_actual(self):
        """Both zero = 0."""
        assert calculate_difference(0, 0) == 0

    def test_zero_expected_positive_actual(self):
        """Over-settled when nothing expected."""
        assert calculate_difference(0, 50000) == -50000

    def test_positive_expected_zero_actual(self):
        """Under-settled when nothing paid."""
        assert calculate_difference(100000, 0) == 100000

    def test_one_paise_difference(self):
        """Smallest possible difference."""
        assert calculate_difference(100000, 99999) == 1

    def test_large_difference(self):
        """Large under-settlement."""
        assert calculate_difference(10_000_000, 1_000_000) == 9_000_000

    def test_symmetry(self):
        """difference(a,b) == -difference(b,a)."""
        assert calculate_difference(100, 200) == -calculate_difference(200, 100)

    def test_integer_arithmetic_no_floating_point(self):
        """All values are integers — no floating point."""
        result = calculate_difference(100000, 33333)
        assert isinstance(result, int)
        assert result == 66667


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AGGREGATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregateRefunds:
    """Tests for aggregate_refunds: sum of refund amounts for a payment."""

    def test_no_refunds(self):
        """Empty refund list = 0."""
        assert aggregate_refunds([], "PAY-001") == 0

    def test_single_refund(self):
        """One refund of 5000 = 5000."""
        refunds = [_make_refund(amount=5000)]
        assert aggregate_refunds(refunds, "PAY-001") == 5000

    def test_multiple_refunds(self):
        """Three refunds: 3000 + 2000 + 1000 = 6000."""
        refunds = [
            _make_refund(refund_id="REF-001", amount=3000),
            _make_refund(refund_id="REF-002", amount=2000),
            _make_refund(refund_id="REF-003", amount=1000),
        ]
        assert aggregate_refunds(refunds, "PAY-001") == 6000

    def test_filters_by_payment_id(self):
        """Only refunds for PAY-001 are counted."""
        refunds = [
            _make_refund(refund_id="REF-001", payment_id="PAY-001", amount=5000),
            _make_refund(refund_id="REF-002", payment_id="PAY-002", amount=8000),
        ]
        assert aggregate_refunds(refunds, "PAY-001") == 5000

    def test_zero_amount_refund(self):
        """Zero-amount refund still counts (data integrity)."""
        refunds = [_make_refund(amount=0)]
        assert aggregate_refunds(refunds, "PAY-001") == 0

    def test_large_refund(self):
        """Large refund: 500000 paise (₹5,000)."""
        refunds = [_make_refund(amount=500000)]
        assert aggregate_refunds(refunds, "PAY-001") == 500000

    def test_no_matching_payment(self):
        """No refunds for the given payment_id."""
        refunds = [_make_refund(payment_id="PAY-OTHER", amount=5000)]
        assert aggregate_refunds(refunds, "PAY-001") == 0


class TestAggregateFees:
    """Tests for aggregate_fees: sum of fee amounts for a payment."""

    def test_no_fees(self):
        """Empty fee list = 0."""
        assert aggregate_fees([], "PAY-001") == 0

    def test_single_fee(self):
        """One fee of 2000 = 2000."""
        fees = [_make_fee(amount=2000)]
        assert aggregate_fees(fees, "PAY-001") == 2000

    def test_multiple_fees(self):
        """Three fees: 1000 + 500 + 200 = 1700."""
        fees = [
            _make_fee(fee_id="FEE-001", amount=1000),
            _make_fee(fee_id="FEE-002", amount=500),
            _make_fee(fee_id="FEE-003", amount=200),
        ]
        assert aggregate_fees(fees, "PAY-001") == 1700

    def test_filters_by_payment_id(self):
        """Only fees for PAY-001 are counted."""
        fees = [
            _make_fee(payment_id="PAY-001", amount=2000),
            _make_fee(payment_id="PAY-002", amount=3000),
        ]
        assert aggregate_fees(fees, "PAY-001") == 2000


class TestAggregateTaxes:
    """Tests for aggregate_taxes: sum of tax amounts for a payment."""

    def test_no_taxes(self):
        """Empty tax list = 0."""
        assert aggregate_taxes([], "PAY-001") == 0

    def test_single_tax(self):
        """One tax of 1500 = 1500."""
        taxes = [_make_tax(amount=1500)]
        assert aggregate_taxes(taxes, "PAY-001") == 1500

    def test_multiple_taxes(self):
        """Two taxes: 2000 + 1000 = 3000."""
        taxes = [
            _make_tax(tax_id="TAX-001", amount=2000),
            _make_tax(tax_id="TAX-002", amount=1000),
        ]
        assert aggregate_taxes(taxes, "PAY-001") == 3000

    def test_filters_by_payment_id(self):
        """Only taxes for PAY-001 are counted."""
        taxes = [
            _make_tax(payment_id="PAY-001", amount=1500),
            _make_tax(payment_id="PAY-002", amount=2500),
        ]
        assert aggregate_taxes(taxes, "PAY-001") == 1500


class TestAggregateAdjustments:
    """Tests for aggregate_adjustments: sum of adjustment amounts for a payment."""

    def test_no_adjustments(self):
        """Empty adjustment list = 0."""
        assert aggregate_adjustments([], "PAY-001") == 0

    def test_positive_credit(self):
        """One credit adjustment: +5000."""
        adjs = [_make_adjustment(amount=5000)]
        assert aggregate_adjustments(adjs, "PAY-001") == 5000

    def test_negative_debit(self):
        """One debit adjustment: -3000."""
        adjs = [_make_adjustment(amount=-3000, adj_type=AdjustmentType.DEBIT)]
        assert aggregate_adjustments(adjs, "PAY-001") == -3000

    def test_mixed_adjustments(self):
        """Credit + debit: 5000 + (-3000) = 2000."""
        adjs = [
            _make_adjustment(adj_id="ADJ-001", amount=5000),
            _make_adjustment(adj_id="ADJ-002", amount=-3000, adj_type=AdjustmentType.DEBIT),
        ]
        assert aggregate_adjustments(adjs, "PAY-001") == 2000

    def test_filters_by_payment_id(self):
        """Only adjustments for PAY-001 are counted."""
        adjs = [
            _make_adjustment(payment_id="PAY-001", amount=5000),
            _make_adjustment(payment_id="PAY-002", amount=8000),
        ]
        assert aggregate_adjustments(adjs, "PAY-001") == 5000

    def test_fee_reversal_positive(self):
        """Fee reversal is positive (reverses a fee deduction)."""
        adjs = [_make_adjustment(amount=2000, adj_type=AdjustmentType.FEE_REVERSAL)]
        assert aggregate_adjustments(adjs, "PAY-001") == 2000

    def test_penalty_negative(self):
        """Penalty is negative (decreases settlement)."""
        adjs = [_make_adjustment(amount=-1000, adj_type=AdjustmentType.PENALTY)]
        assert aggregate_adjustments(adjs, "PAY-001") == -1000


class TestAggregateSettlements:
    """Tests for aggregate_settlements: total and count of settlements."""

    def test_no_settlements(self):
        """No settlements = (0, 0)."""
        total, count = aggregate_settlements([], "PAY-001")
        assert total == 0
        assert count == 0

    def test_single_settlement(self):
        """One settlement of 100000 = (100000, 1)."""
        settlements = [_make_settlement(amount=100000)]
        total, count = aggregate_settlements(settlements, "PAY-001")
        assert total == 100000
        assert count == 1

    def test_multiple_settlements(self):
        """Two settlements: 60000 + 40000 = (100000, 2)."""
        settlements = [
            _make_settlement(settlement_id="SET-001", amount=60000),
            _make_settlement(settlement_id="SET-002", amount=40000),
        ]
        total, count = aggregate_settlements(settlements, "PAY-001")
        assert total == 100000
        assert count == 2

    def test_duplicate_settlements(self):
        """Two identical settlements: 100000 + 100000 = (200000, 2)."""
        settlements = [
            _make_settlement(settlement_id="SET-001", amount=100000),
            _make_settlement(settlement_id="SET-002", amount=100000),
        ]
        total, count = aggregate_settlements(settlements, "PAY-001")
        assert total == 200000
        assert count == 2

    def test_filters_by_payment_id(self):
        """Only settlements for PAY-001 are counted."""
        settlements = [
            _make_settlement(payment_id="PAY-001", amount=100000),
            _make_settlement(payment_id="PAY-002", amount=200000),
        ]
        total, count = aggregate_settlements(settlements, "PAY-001")
        assert total == 100000
        assert count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE FINANCIAL RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregateFinancialRecords:
    """Tests for aggregate_financial_records: full aggregation for a payment."""

    def test_all_zeros(self):
        """No records = all zeros."""
        payment = _make_payment(amount=100000)
        result = aggregate_financial_records(payment, [], [], [], [])
        assert result == {
            "payment_amount": 100000,
            "total_refunds": 0,
            "total_fees": 0,
            "total_taxes": 0,
            "total_adjustments": 0,
        }

    def test_all_records(self):
        """Full set of records aggregated correctly."""
        payment = _make_payment(amount=100000)
        refunds = [_make_refund(amount=5000), _make_refund(refund_id="REF-002", amount=3000)]
        fees = [_make_fee(amount=2000)]
        taxes = [_make_tax(amount=1500)]
        adjs = [_make_adjustment(amount=4000)]

        result = aggregate_financial_records(payment, refunds, fees, taxes, adjs)
        assert result["payment_amount"] == 100000
        assert result["total_refunds"] == 8000
        assert result["total_fees"] == 2000
        assert result["total_taxes"] == 1500
        assert result["total_adjustments"] == 4000

    def test_filters_by_payment_id(self):
        """Only records for the payment's ID are aggregated."""
        payment = _make_payment(payment_id="PAY-001")
        refunds = [
            _make_refund(payment_id="PAY-001", amount=5000),
            _make_refund(payment_id="PAY-002", amount=8000),
        ]
        result = aggregate_financial_records(payment, refunds, [], [], [])
        assert result["total_refunds"] == 5000

    def test_mixed_adjustments_in_aggregation(self):
        """Credit + debit adjustments in aggregation."""
        payment = _make_payment(amount=100000)
        adjs = [
            _make_adjustment(amount=5000, adj_type=AdjustmentType.CREDIT),
            _make_adjustment(adj_id="ADJ-002", amount=-3000, adj_type=AdjustmentType.DEBIT),
        ]
        result = aggregate_financial_records(payment, [], [], [], adjs)
        assert result["total_adjustments"] == 2000


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CALCULATION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculationBreakdown:
    """Tests for CalculationBreakdown: detailed calculation audit trail."""

    def test_basic_breakdown(self):
        """Breakdown matches expected formula."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=100000,
            total_refunds=5000,
            total_fees=2000,
            total_taxes=1500,
            total_adjustments=3000,
        )
        assert breakdown.payment_amount == 100000
        assert breakdown.refund_deduction == 5000
        assert breakdown.fee_deduction == 2000
        assert breakdown.tax_deduction == 1500
        assert breakdown.adjustment_addition == 3000
        assert breakdown.expected_amount == 94500

    def test_breakdown_with_zeros(self):
        """All zeros except payment."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0,
        )
        assert breakdown.expected_amount == 100000
        assert breakdown.refund_deduction == 0
        assert breakdown.fee_deduction == 0
        assert breakdown.tax_deduction == 0
        assert breakdown.adjustment_addition == 0

    def test_breakdown_negative_adjustment(self):
        """Negative adjustment in breakdown."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=-5000,
        )
        assert breakdown.expected_amount == 95000
        assert breakdown.adjustment_addition == -5000

    def test_breakdown_large_values(self):
        """Large values in breakdown."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=10_000_000, total_refunds=500_000,
            total_fees=100_000, total_taxes=50_000,
            total_adjustments=200_000,
        )
        assert breakdown.expected_amount == 9_550_000

    def test_breakdown_integer_only(self):
        """All breakdown values are integers."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=100000, total_refunds=5000,
            total_fees=2000, total_taxes=1500,
            total_adjustments=3000,
        )
        for field in ['payment_amount', 'refund_deduction', 'fee_deduction',
                       'tax_deduction', 'adjustment_addition', 'expected_amount']:
            assert isinstance(getattr(breakdown, field), int)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MATCH STATUS DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetermineMatchStatus:
    """Tests for determine_match_status: MISSING, DUPLICATE, MATCHED, EXCEPTION."""

    def test_matched(self):
        """Settlement exists, single, zero difference → MATCHED."""
        assert determine_match_status(
            has_settlement=True, has_payment=True,
            difference=0, settlement_count=1,
        ) == MatchStatus.MATCHED

    def test_missing_no_settlement(self):
        """No settlement → MISSING."""
        assert determine_match_status(
            has_settlement=False, has_payment=True,
            difference=100000, settlement_count=0,
        ) == MatchStatus.MISSING

    def test_duplicate(self):
        """Multiple settlements with identical amounts → DUPLICATE."""
        assert determine_match_status(
            has_settlement=True, has_payment=True,
            difference=0, settlement_count=2,
        ) == MatchStatus.DUPLICATE

    def test_exception_positive_difference(self):
        """Settlement exists, non-zero difference → EXCEPTION."""
        assert determine_match_status(
            has_settlement=True, has_payment=True,
            difference=20000, settlement_count=1,
        ) == MatchStatus.EXCEPTION

    def test_exception_negative_difference(self):
        """Over-settled → EXCEPTION."""
        assert determine_match_status(
            has_settlement=True, has_payment=True,
            difference=-15000, settlement_count=1,
        ) == MatchStatus.EXCEPTION

    def test_missing_takes_priority_over_difference(self):
        """MISSING takes priority even with zero difference."""
        assert determine_match_status(
            has_settlement=False, has_payment=True,
            difference=0, settlement_count=0,
        ) == MatchStatus.MISSING

    def test_duplicate_takes_priority_over_exception(self):
        """DUPLICATE takes priority over non-zero difference."""
        assert determine_match_status(
            has_settlement=True, has_payment=True,
            difference=5000, settlement_count=3,
        ) == MatchStatus.DUPLICATE


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXCEPTION TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyExceptionType:
    """Tests for classify_exception_type: deterministic exception classification."""

    def test_exact_match(self):
        """MATCHED status → EXACT_MATCH."""
        result = classify_exception_type(
            match_status=MatchStatus.MATCHED,
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=0,
            settlement_count=1,
        )
        assert result == ExceptionType.EXACT_MATCH

    def test_missing_record(self):
        """MISSING status → MISSING_RECORD."""
        result = classify_exception_type(
            match_status=MatchStatus.MISSING,
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=100000,
            settlement_count=0,
        )
        assert result == ExceptionType.MISSING_RECORD

    def test_duplicate(self):
        """DUPLICATE status → DUPLICATE."""
        result = classify_exception_type(
            match_status=MatchStatus.DUPLICATE,
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=0,
            settlement_count=2,
        )
        assert result == ExceptionType.DUPLICATE

    def test_refund_adjustment(self):
        """Difference equals total refunds → REFUND_ADJUSTMENT."""
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=100000, total_refunds=5000, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=5000,
            settlement_count=1,
        )
        assert result == ExceptionType.REFUND_ADJUSTMENT

    def test_fee_difference(self):
        """Difference proportional to fees → FEE_DIFFERENCE."""
        # Fees = 10000, difference = 1500 (15% of fees, within 1-25%)
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=100000, total_refunds=0, total_fees=10000,
            total_taxes=0, total_adjustments=0, difference=1500,
            settlement_count=1,
        )
        assert result == ExceptionType.FEE_DIFFERENCE

    def test_tax_adjustment(self):
        """Difference proportional to taxes → TAX_ADJUSTMENT."""
        # Taxes = 10000, difference = 1000 (10% of taxes, within 1-20%)
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=10000, total_adjustments=0, difference=1000,
            settlement_count=1,
        )
        assert result == ExceptionType.TAX_ADJUSTMENT

    def test_timing_difference(self):
        """Moderate difference without clear cause → TIMING_DIFFERENCE."""
        # Difference = 5000 (within ₹1-₹500 range)
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=100000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=5000,
            settlement_count=1,
        )
        assert result == ExceptionType.TIMING_DIFFERENCE

    def test_complex_multi_adjustment(self):
        """Multiple components + large difference > 50000 → COMPLEX_MULTI_ADJUSTMENT."""
        # Refunds + fees + adjustments present, difference > 50000 (skips TIMING_DIFFERENCE)
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=500000, total_refunds=50000, total_fees=20000,
            total_taxes=0, total_adjustments=30000, difference=60000,
            settlement_count=1,
        )
        assert result == ExceptionType.COMPLEX_MULTI_ADJUSTMENT

    def test_partial_settlement(self):
        """Actual is 60% of expected → PARTIAL_SETTLEMENT."""
        # Expected = 500000 - 0 - 0 - 0 + 0 = 500000
        # Difference = 200000 (actual = 300000, which is 60% of expected)
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=500000, total_refunds=0, total_fees=0,
            total_taxes=0, total_adjustments=0, difference=200000,
            settlement_count=1,
        )
        assert result == ExceptionType.PARTIAL_SETTLEMENT

    def test_priority_refund_over_fee(self):
        """Refund adjustment takes priority over fee difference."""
        # Refunds present, difference = refunds → REFUND_ADJUSTMENT
        result = classify_exception_type(
            match_status=MatchStatus.EXCEPTION,
            payment_amount=100000, total_refunds=5000, total_fees=10000,
            total_taxes=0, total_adjustments=0, difference=5000,
            settlement_count=1,
        )
        assert result == ExceptionType.REFUND_ADJUSTMENT


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FINANCIAL RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyFinancialRisk:
    """Tests for classify_financial_risk: NEGLIGIBLE, LOW, MEDIUM, HIGH, CRITICAL."""

    def test_negligible(self):
        """< ₹100 (< 10000 paise) → NEGLIGIBLE."""
        assert classify_financial_risk(5000) == FinancialRiskLevel.NEGLIGIBLE

    def test_negligible_zero(self):
        """Zero → NEGLIGIBLE."""
        assert classify_financial_risk(0) == FinancialRiskLevel.NEGLIGIBLE

    def test_negligible_boundary(self):
        """9999 paise (< ₹100) → NEGLIGIBLE."""
        assert classify_financial_risk(9999) == FinancialRiskLevel.NEGLIGIBLE

    def test_low(self):
        """₹100 (10000 paise) → LOW."""
        assert classify_financial_risk(10000) == FinancialRiskLevel.LOW

    def test_low_boundary(self):
        """₹999 (99999 paise) → LOW."""
        assert classify_financial_risk(99999) == FinancialRiskLevel.LOW

    def test_medium(self):
        """₹1000 (100000 paise) → MEDIUM."""
        assert classify_financial_risk(100000) == FinancialRiskLevel.MEDIUM

    def test_medium_boundary(self):
        """₹9999 (999999 paise) → MEDIUM."""
        assert classify_financial_risk(999999) == FinancialRiskLevel.MEDIUM

    def test_high(self):
        """₹10000 (1000000 paise) → HIGH."""
        assert classify_financial_risk(1_000_000) == FinancialRiskLevel.HIGH

    def test_high_boundary(self):
        """₹100000 (10000000 paise) → HIGH."""
        assert classify_financial_risk(10_000_000) == FinancialRiskLevel.HIGH

    def test_critical(self):
        """₹100001 (10000100 paise) → CRITICAL."""
        assert classify_financial_risk(10_000_100) == FinancialRiskLevel.CRITICAL

    def test_critical_large(self):
        """₹10,00,000 (100000000 paise) → CRITICAL."""
        assert classify_financial_risk(100_000_000) == FinancialRiskLevel.CRITICAL

    def test_negative_amount_uses_absolute(self):
        """Negative amount uses absolute value for classification."""
        assert classify_financial_risk(-5000) == FinancialRiskLevel.NEGLIGIBLE
        assert classify_financial_risk(-1_000_000) == FinancialRiskLevel.HIGH

    def test_custom_threshold(self):
        """Custom high_value_threshold changes CRITICAL boundary."""
        # With threshold of 5000000 (₹50,000)
        assert classify_financial_risk(5_000_000, high_value_threshold=5_000_000) == FinancialRiskLevel.HIGH
        assert classify_financial_risk(5_000_100, high_value_threshold=5_000_000) == FinancialRiskLevel.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# 9. REWARD CATEGORY DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetermineRewardCategory:
    """Tests for determine_reward_category: decision tree for reward assignment."""

    def _outcome(self, **kwargs):
        """Helper to create OutcomeRecord with defaults."""
        defaults = dict(
            workflow_id="WF-001", exception_id="EXC-001", decision="AUTO",
            was_executed=True, verification_passed=True, was_rolled_back=False,
            resolution_correct=True, adjustment_paise=50000,
            discrepancy_eliminated=True, unintended_changes=0, confidence=0.9,
        )
        defaults.update(kwargs)
        return _make_outcome(**defaults)

    def test_correct_auto_resolution(self):
        """Executed + correct → CORRECT_AUTO_RESOLUTION."""
        outcome = self._outcome(was_executed=True, resolution_correct=True)
        assert determine_reward_category(outcome) == RewardCategory.CORRECT_AUTO_RESOLUTION

    def test_incorrect_auto_resolution(self):
        """Executed + incorrect → INCORRECT_AUTO_RESOLUTION."""
        outcome = self._outcome(was_executed=True, resolution_correct=False)
        assert determine_reward_category(outcome) == RewardCategory.INCORRECT_AUTO_RESOLUTION

    def test_high_value_error(self):
        """Executed + incorrect + high value → HIGH_VALUE_ERROR."""
        outcome = self._outcome(
            was_executed=True, resolution_correct=False,
            adjustment_paise=15_000_000,  # > ₹1,00,000
        )
        assert determine_reward_category(outcome) == RewardCategory.HIGH_VALUE_ERROR

    def test_verification_failure(self):
        """Executed + not verified + rolled back → VERIFICATION_FAILURE."""
        outcome = self._outcome(
            was_executed=True, verification_passed=False,
            was_rolled_back=True,
        )
        assert determine_reward_category(outcome) == RewardCategory.VERIFICATION_FAILURE

    def test_human_confirmed(self):
        """Human feedback APPROVE (not auto-executed) → HUMAN_CONFIRMED."""
        outcome = self._outcome(decision="HUMAN_REVIEW", was_executed=False)
        feedback = FeedbackRecord(
            feedback_id="FB-001", workflow_id="WF-001",
            exception_id="EXC-001", feedback_type=FeedbackType.APPROVE,
            reviewer="reviewer@test.com",
            system_prediction="FEE_ADJUSTMENT",
        )
        assert determine_reward_category(outcome, feedback) == RewardCategory.HUMAN_CONFIRMED

    def test_human_rejected(self):
        """Human feedback REJECT (not auto-executed) → INCORRECT_AUTO_RESOLUTION."""
        outcome = self._outcome(decision="HUMAN_REVIEW", was_executed=False)
        feedback = FeedbackRecord(
            feedback_id="FB-001", workflow_id="WF-001",
            exception_id="EXC-001", feedback_type=FeedbackType.REJECT,
            reviewer="reviewer@test.com",
            system_prediction="FEE_ADJUSTMENT",
        )
        assert determine_reward_category(outcome, feedback) == RewardCategory.INCORRECT_AUTO_RESOLUTION

    def test_escalated_correct(self):
        """Escalated + prediction correct → CORRECT_ESCALATION."""
        outcome = self._outcome(
            decision="UNRESOLVED", was_executed=False,
            resolution_correct=True,
        )
        assert determine_reward_category(outcome) == RewardCategory.CORRECT_ESCALATION

    def test_escalated_unnecessary(self):
        """Escalated + prediction correct but escalated anyway → depends on feedback."""
        outcome = self._outcome(
            decision="UNRESOLVED", was_executed=False,
            resolution_correct=True,
        )
        # Without feedback → CORRECT_ESCALATION (prediction was correct)
        assert determine_reward_category(outcome) == RewardCategory.CORRECT_ESCALATION

    def test_default_unnecessary_escalation(self):
        """No feedback, not executed → UNNECESSARY_ESCALATION."""
        outcome = self._outcome(
            decision="HUMAN_REVIEW", was_executed=False,
            resolution_correct=None,
        )
        assert determine_reward_category(outcome) == RewardCategory.UNNECESSARY_ESCALATION


# ═══════════════════════════════════════════════════════════════════════════════
# 10. REWARD ENGINE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewardEngine:
    """Tests for RewardEngine: transparent, deterministic reward calculation."""

    def test_correct_auto_resolution_reward(self):
        """Correct auto-resolution should have positive reward."""
        engine = RewardEngine()
        outcome = _make_outcome(
            was_executed=True, resolution_correct=True,
            verification_passed=True, confidence=0.9,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.category == RewardCategory.CORRECT_AUTO_RESOLUTION
        assert reward.reward_value > 0
        assert reward.resolution_correct is True

    def test_incorrect_auto_resolution_penalty(self):
        """Incorrect auto-resolution should have negative reward."""
        engine = RewardEngine()
        outcome = _make_outcome(
            was_executed=True, resolution_correct=False,
            confidence=0.8,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.category == RewardCategory.INCORRECT_AUTO_RESOLUTION
        assert reward.reward_value < 0

    def test_high_value_error_severe_penalty(self):
        """High-value error should have the most severe penalty."""
        engine = RewardEngine()
        outcome = _make_outcome(
            was_executed=True, resolution_correct=False,
            adjustment_paise=15_000_000, confidence=0.95,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.category == RewardCategory.HIGH_VALUE_ERROR
        assert reward.reward_value < -0.5

    def test_verification_failure_penalty(self):
        """Verification failure should penalize heavily."""
        engine = RewardEngine()
        outcome = _make_outcome(
            was_executed=True, verification_passed=False,
            was_rolled_back=True,
        )
        reward = engine.calculate_reward(outcome)
        assert reward.category == RewardCategory.VERIFICATION_FAILURE
        assert reward.reward_value < 0

    def test_reward_clamped_to_range(self):
        """Reward must be within [-1, 1]."""
        engine = RewardEngine()
        outcome = _make_outcome(
            was_executed=True, resolution_correct=True,
            verification_passed=True, confidence=1.0,
        )
        reward = engine.calculate_reward(outcome)
        assert -1.0 <= reward.reward_value <= 1.0

    def test_reward_has_breakdown(self):
        """Every reward has a transparent breakdown."""
        engine = RewardEngine()
        outcome = _make_outcome()
        reward = engine.calculate_reward(outcome)
        assert reward.breakdown is not None
        assert reward.breakdown.base_reward is not None
        assert reward.breakdown.verification_component is not None
        assert reward.breakdown.financial_risk_component is not None

    def test_deterministic_same_inputs(self):
        """Same inputs → same reward (deterministic)."""
        engine = RewardEngine()
        outcome = _make_outcome(confidence=0.85)
        r1 = engine.calculate_reward(outcome)
        r2 = engine.calculate_reward(outcome)
        assert r1.reward_value == r2.reward_value
        assert r1.category == r2.category

    def test_average_reward(self):
        """Average reward across multiple rewards."""
        engine = RewardEngine()
        engine.calculate_reward(_make_outcome(resolution_correct=True, was_executed=True))
        engine.calculate_reward(_make_outcome(resolution_correct=False, was_executed=True))
        avg = engine.average_reward()
        assert isinstance(avg, float)

    def test_category_counts(self):
        """Category counts are tracked correctly."""
        engine = RewardEngine()
        engine.calculate_reward(_make_outcome(resolution_correct=True, was_executed=True))
        engine.calculate_reward(_make_outcome(resolution_correct=True, was_executed=True))
        engine.calculate_reward(_make_outcome(resolution_correct=False, was_executed=True))
        counts = engine.category_counts()
        assert counts.get("CORRECT_AUTO_RESOLUTION", 0) == 2
        assert counts.get("INCORRECT_AUTO_RESOLUTION", 0) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. RECONCILIATION RESULT SELF-VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestReconciliationResultVerification:
    """Tests for ReconciliationResult.verify_calculation and compute methods."""

    def _make_result(self, payment=100000, refunds=0, fees=0, taxes=0,
                     adjustments=0, actual=None):
        """Create a ReconciliationResult for testing."""
        from app.schemas.reconciliation import ReconciliationResult
        expected = payment - refunds - fees - taxes + adjustments
        if actual is None:
            actual = expected
        return ReconciliationResult(
            reconciliation_id="REC-001",
            case_id="CASE-001",
            payment_id="PAY-001",
            merchant_id="MER-001",
            payment_amount=payment,
            total_refunds=refunds,
            total_fees=fees,
            total_taxes=taxes,
            total_adjustments=adjustments,
            expected_amount=expected,
            actual_amount=actual,
            difference=expected - actual,
            match_status=MatchStatus.MATCHED if expected == actual else MatchStatus.EXCEPTION,
            exception_type=ExceptionType.EXACT_MATCH if expected == actual else ExceptionType.UNKNOWN,
        )

    def test_verify_calculation_passes(self):
        """verify_calculation returns True when expected matches breakdown."""
        result = self._make_result(payment=100000, refunds=5000, fees=2000)
        assert result.verify_calculation() is True

    def test_verify_calculation_with_adjustments(self):
        """verify_calculation with adjustments."""
        result = self._make_result(payment=100000, adjustments=3000)
        assert result.verify_calculation() is True
        assert result.expected_amount == 103000

    def test_compute_difference(self):
        """compute_difference returns expected - actual."""
        result = self._make_result(payment=100000, actual=80000)
        assert result.compute_difference() == 20000

    def test_compute_expected_amount(self):
        """compute_expected_amount matches formula."""
        result = self._make_result(payment=100000, refunds=5000, fees=2000,
                                   taxes=1500, adjustments=3000)
        assert result.compute_expected_amount() == 94500

    def test_integer_only_in_result(self):
        """All financial fields in result are integers."""
        result = self._make_result()
        for field in ['payment_amount', 'total_refunds', 'total_fees',
                       'total_taxes', 'total_adjustments', 'expected_amount',
                       'actual_amount', 'difference']:
            assert isinstance(getattr(result, field), int)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EDGE CASES AND BOUNDARY VALUES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for financial calculations."""

    def test_minimum_paise(self):
        """1 paise payment, 1 paise refund = 0 expected."""
        result = calculate_expected_amount(1, 1, 0, 0, 0)
        assert result == 0

    def test_maximum_practical_amount(self):
        """₹10,00,000 (10 crore paise) — large but valid."""
        result = calculate_expected_amount(100_000_000, 0, 0, 0, 0)
        assert result == 100_000_000

    def test_all_amounts_equal(self):
        """Payment = refunds = fees = taxes: 100000 - 100000 - 0 - 0 + 0 = 0."""
        result = calculate_expected_amount(100000, 100000, 0, 0, 0)
        assert result == 0

    def test_refunds_plus_fees_exceed_payment(self):
        """Over-deducted: 50000 - 30000 - 25000 = -5000."""
        result = calculate_expected_amount(50000, 30000, 25000, 0, 0)
        assert result == -5000

    def test_adjustment_cancels_deductions(self):
        """Adjustment exactly cancels deductions: 100000 - 10000 + 10000 = 100000."""
        result = calculate_expected_amount(100000, 10000, 0, 0, 10000)
        assert result == 100000

    def test_all_negative_adjustments(self):
        """All adjustments are debits: 100000 - 0 - 0 - 0 + (-10000) = 90000."""
        result = calculate_expected_amount(100000, 0, 0, 0, -10000)
        assert result == 90000

    def test_very_small_difference(self):
        """1 paise difference is still an exception."""
        diff = calculate_difference(100000, 99999)
        assert diff == 1
        assert diff != 0

    def test_aggregation_performance(self):
        """Aggregating 1000 records should complete."""
        refunds = [_make_refund(refund_id=f"REF-{i}", amount=100) for i in range(1000)]
        total = aggregate_refunds(refunds, "PAY-001")
        assert total == 100000  # 1000 * 100

    def test_payment_id_isolation(self):
        """Different payment IDs produce isolated aggregations."""
        refunds = [
            _make_refund(refund_id="REF-001", payment_id="PAY-A", amount=5000),
            _make_refund(refund_id="REF-002", payment_id="PAY-B", amount=8000),
            _make_refund(refund_id="REF-003", payment_id="PAY-A", amount=3000),
        ]
        assert aggregate_refunds(refunds, "PAY-A") == 8000
        assert aggregate_refunds(refunds, "PAY-B") == 8000
        assert aggregate_refunds(refunds, "PAY-C") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. INTEGER-ONLY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegerOnlyVerification:
    """Verify all financial calculations use integer arithmetic."""

    def test_expected_amount_is_int(self):
        """calculate_expected_amount returns int."""
        result = calculate_expected_amount(100000, 5000, 2000, 1500, 3000)
        assert isinstance(result, int)

    def test_difference_is_int(self):
        """calculate_difference returns int."""
        result = calculate_difference(100000, 80000)
        assert isinstance(result, int)

    def test_aggregation_returns_int(self):
        """All aggregations return int."""
        assert isinstance(aggregate_refunds([], "PAY-001"), int)
        assert isinstance(aggregate_fees([], "PAY-001"), int)
        assert isinstance(aggregate_taxes([], "PAY-001"), int)
        assert isinstance(aggregate_adjustments([], "PAY-001"), int)
        total, count = aggregate_settlements([], "PAY-001")
        assert isinstance(total, int)
        assert isinstance(count, int)

    def test_breakdown_all_int(self):
        """CalculationBreakdown fields are all int."""
        breakdown = CalculationBreakdown.from_financial_records(
            payment_amount=100000, total_refunds=5000,
            total_fees=2000, total_taxes=1500, total_adjustments=3000,
        )
        for field in ['payment_amount', 'refund_deduction', 'fee_deduction',
                       'tax_deduction', 'adjustment_addition', 'expected_amount']:
            val = getattr(breakdown, field)
            assert isinstance(val, int), f"{field} is {type(val)}, expected int"
