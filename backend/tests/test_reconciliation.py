"""
Unit tests for the deterministic reconciliation engine.

Tests cover:
- Exact match scenarios
- Refund handling
- Fee handling
- Tax handling
- Adjustment handling
- Partial settlement
- Multiple settlements
- Zero adjustment
- Arithmetic consistency
- Edge cases
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.reconciliation.engine import (
    aggregate_adjustments,
    aggregate_fees,
    aggregate_refunds,
    aggregate_settlements,
    aggregate_taxes,
    calculate_reconciliation,
    reconcile_batch,
)
from app.schemas.enums import (
    AdjustmentType,
    ExceptionType,
    MatchStatus,
    FeeType,
    TaxType,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    PaymentStatus,
    Refund,
    RefundStatus,
    Settlement,
    SettlementStatus,
    Tax,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def base_payment():
    """Base payment fixture for tests."""
    return Payment(
        payment_id="PAY-000001",
        merchant_id="MER-0001",
        amount=100000,  # ₹1000
        currency="INR",
        status=PaymentStatus.CAPTURED,
        payment_timestamp=datetime(2025, 3, 15, 10, 0, 0),
    )


@pytest.fixture
def base_settlement():
    """Base settlement fixture for tests."""
    return Settlement(
        settlement_id="SET-000001",
        payment_id="PAY-000001",
        merchant_id="MER-0001",
        amount=100000,  # ₹1000
        currency="INR",
        status=SettlementStatus.SETTLED,
        settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exact Match Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatch:
    """Tests for exact match scenarios."""

    def test_exact_match_no_fees_taxes(self, base_payment, base_settlement):
        """Payment equals settlement with no fees or taxes."""
        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[base_settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.expected_amount == 100000
        assert result.actual_amount == 100000
        assert result.difference == 0
        assert result.match_status == MatchStatus.MATCHED
        assert result.exception_type == ExceptionType.EXACT_MATCH

    def test_exact_match_with_fees_taxes(self, base_payment):
        """Payment with fees and taxes equals settlement."""
        # Payment: 100000
        # Fees: 2000 (2%)
        # Taxes: 18000 (18%)
        # Expected: 100000 - 2000 - 18000 = 80000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=80000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        fees = [
            Fee(
                fee_id="FEE-000001",
                payment_id="PAY-000001",
                amount=2000,
                fee_type=FeeType.TRANSACTION,
            )
        ]
        taxes = [
            Tax(
                tax_id="TAX-000001",
                payment_id="PAY-000001",
                amount=18000,
                tax_type=TaxType.GST,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=fees,
            taxes=taxes,
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.expected_amount == 80000
        assert result.actual_amount == 80000
        assert result.difference == 0
        assert result.match_status == MatchStatus.MATCHED


# ─────────────────────────────────────────────────────────────────────────────
# Refund Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundHandling:
    """Tests for refund handling."""

    def test_refund_deducted_from_expected(self, base_payment):
        """Refund should be deducted from expected amount."""
        # Payment: 100000
        # Refund: 20000
        # Expected: 100000 - 20000 = 80000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=80000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        refunds = [
            Refund(
                refund_id="REF-000001",
                payment_id="PAY-000001",
                amount=20000,
                status=RefundStatus.PROCESSED,
                refund_timestamp=datetime(2025, 3, 15, 12, 0, 0),
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=refunds,
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_refunds == 20000
        assert result.expected_amount == 80000
        assert result.difference == 0

    def test_multiple_refunds_aggregated(self, base_payment):
        """Multiple refunds should be aggregated."""
        # Payment: 100000
        # Refunds: 10000 + 5000 = 15000
        # Expected: 100000 - 15000 = 85000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=85000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        refunds = [
            Refund(
                refund_id="REF-000001",
                payment_id="PAY-000001",
                amount=10000,
                status=RefundStatus.PROCESSED,
                refund_timestamp=datetime(2025, 3, 15, 12, 0, 0),
            ),
            Refund(
                refund_id="REF-000002",
                payment_id="PAY-000001",
                amount=5000,
                status=RefundStatus.PROCESSED,
                refund_timestamp=datetime(2025, 3, 15, 14, 0, 0),
            ),
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=refunds,
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_refunds == 15000
        assert result.expected_amount == 85000


# ─────────────────────────────────────────────────────────────────────────────
# Fee Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeHandling:
    """Tests for fee handling."""

    def test_fee_deducted_from_expected(self, base_payment):
        """Fee should be deducted from expected amount."""
        # Payment: 100000
        # Fee: 2000
        # Expected: 100000 - 2000 = 98000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=98000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        fees = [
            Fee(
                fee_id="FEE-000001",
                payment_id="PAY-000001",
                amount=2000,
                fee_type=FeeType.TRANSACTION,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=fees,
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_fees == 2000
        assert result.expected_amount == 98000
        assert result.difference == 0

    def test_multiple_fees_aggregated(self, base_payment):
        """Multiple fees should be aggregated."""
        # Payment: 100000
        # Fees: 2000 + 500 = 2500
        # Expected: 100000 - 2500 = 97500
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=97500,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        fees = [
            Fee(
                fee_id="FEE-000001",
                payment_id="PAY-000001",
                amount=2000,
                fee_type=FeeType.TRANSACTION,
            ),
            Fee(
                fee_id="FEE-000002",
                payment_id="PAY-000001",
                amount=500,
                fee_type=FeeType.PLATFORM,
            ),
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=fees,
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_fees == 2500
        assert result.expected_amount == 97500


# ─────────────────────────────────────────────────────────────────────────────
# Tax Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxHandling:
    """Tests for tax handling."""

    def test_tax_deducted_from_expected(self, base_payment):
        """Tax should be deducted from expected amount."""
        # Payment: 100000
        # Tax: 18000
        # Expected: 100000 - 18000 = 82000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=82000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        taxes = [
            Tax(
                tax_id="TAX-000001",
                payment_id="PAY-000001",
                amount=18000,
                tax_type=TaxType.GST,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=taxes,
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_taxes == 18000
        assert result.expected_amount == 82000
        assert result.difference == 0


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentHandling:
    """Tests for adjustment handling."""

    def test_credit_adjustment_increases_expected(self, base_payment):
        """Credit adjustment should increase expected amount."""
        # Payment: 100000
        # Credit: +5000
        # Expected: 100000 + 5000 = 105000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=105000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        adjustments = [
            Adjustment(
                adjustment_id="ADJ-000001",
                payment_id="PAY-000001",
                amount=5000,
                adjustment_type=AdjustmentType.CREDIT,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=adjustments,
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_adjustments == 5000
        assert result.expected_amount == 105000
        assert result.difference == 0

    def test_debit_adjustment_decreases_expected(self, base_payment):
        """Debit adjustment should decrease expected amount."""
        # Payment: 100000
        # Debit: -3000
        # Expected: 100000 - 3000 = 97000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=97000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        adjustments = [
            Adjustment(
                adjustment_id="ADJ-000001",
                payment_id="PAY-000001",
                amount=-3000,
                adjustment_type=AdjustmentType.DEBIT,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=adjustments,
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_adjustments == -3000
        assert result.expected_amount == 97000
        assert result.difference == 0


# ─────────────────────────────────────────────────────────────────────────────
# Partial Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialSettlement:
    """Tests for partial settlement scenarios."""

    def test_partial_settlement_detected(self, base_payment):
        """Partial settlement should be detected."""
        # Payment: 100000
        # Expected: 100000
        # Actual: 50000 (partial)
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=50000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.expected_amount == 100000
        assert result.actual_amount == 50000
        assert result.difference == 50000  # Under-settled
        assert result.match_status == MatchStatus.EXCEPTION


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Settlements Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleSettlements:
    """Tests for multiple settlement scenarios."""

    def test_duplicate_settlement_detected(self, base_payment):
        """Duplicate settlement should be detected."""
        # Payment: 100000
        # Settlements: 100000 + 100000 = 200000 (duplicate)
        settlements = [
            Settlement(
                settlement_id="SET-000001",
                payment_id="PAY-000001",
                merchant_id="MER-0001",
                amount=100000,
                currency="INR",
                status=SettlementStatus.SETTLED,
                settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
            ),
            Settlement(
                settlement_id="SET-000002",
                payment_id="PAY-000001",
                merchant_id="MER-0001",
                amount=100000,
                currency="INR",
                status=SettlementStatus.SETTLED,
                settlement_timestamp=datetime(2025, 3, 17, 10, 0, 0),
            ),
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=settlements,
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.actual_amount == 200000
        assert result.difference == -100000  # Over-settled
        assert result.match_status == MatchStatus.DUPLICATE
        assert result.exception_type == ExceptionType.DUPLICATE


# ─────────────────────────────────────────────────────────────────────────────
# Zero Adjustment Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroAdjustment:
    """Tests for zero adjustment scenarios."""

    def test_zero_adjustment_no_effect(self, base_payment, base_settlement):
        """Zero adjustment should have no effect on calculation."""
        adjustments = [
            Adjustment(
                adjustment_id="ADJ-000001",
                payment_id="PAY-000001",
                amount=0,
                adjustment_type=AdjustmentType.CORRECTION,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=adjustments,
            settlements=[base_settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.total_adjustments == 0
        assert result.expected_amount == 100000
        assert result.difference == 0


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArithmeticConsistency:
    """Tests for arithmetic consistency."""

    def test_complex_calculation(self, base_payment):
        """Complex calculation with all components."""
        # Payment: 100000
        # Refunds: 10000
        # Fees: 2000
        # Taxes: 18000
        # Adjustments: +5000
        # Expected: 100000 - 10000 - 2000 - 18000 + 5000 = 75000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=75000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )
        refunds = [
            Refund(
                refund_id="REF-000001",
                payment_id="PAY-000001",
                amount=10000,
                status=RefundStatus.PROCESSED,
                refund_timestamp=datetime(2025, 3, 15, 12, 0, 0),
            )
        ]
        fees = [
            Fee(
                fee_id="FEE-000001",
                payment_id="PAY-000001",
                amount=2000,
                fee_type=FeeType.TRANSACTION,
            )
        ]
        taxes = [
            Tax(
                tax_id="TAX-000001",
                payment_id="PAY-000001",
                amount=18000,
                tax_type=TaxType.GST,
            )
        ]
        adjustments = [
            Adjustment(
                adjustment_id="ADJ-000001",
                payment_id="PAY-000001",
                amount=5000,
                adjustment_type=AdjustmentType.CREDIT,
            )
        ]

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=refunds,
            fees=fees,
            taxes=taxes,
            adjustments=adjustments,
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        # Verify all components
        assert result.payment_amount == 100000
        assert result.total_refunds == 10000
        assert result.total_fees == 2000
        assert result.total_taxes == 18000
        assert result.total_adjustments == 5000

        # Verify expected amount
        expected = 100000 - 10000 - 2000 - 18000 + 5000
        assert result.expected_amount == expected
        assert result.expected_amount == 75000

        # Verify difference
        assert result.difference == 0
        assert result.match_status == MatchStatus.MATCHED

    def test_difference_formula(self, base_payment):
        """Verify difference = expected - actual."""
        # Payment: 100000
        # Expected: 100000
        # Actual: 95000
        # Difference: 100000 - 95000 = 5000
        settlement = Settlement(
            settlement_id="SET-000001",
            payment_id="PAY-000001",
            merchant_id="MER-0001",
            amount=95000,
            currency="INR",
            status=SettlementStatus.SETTLED,
            settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
        )

        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[settlement],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.expected_amount == 100000
        assert result.actual_amount == 95000
        assert result.difference == 5000  # expected - actual


# ─────────────────────────────────────────────────────────────────────────────
# Missing Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingSettlement:
    """Tests for missing settlement scenarios."""

    def test_missing_settlement_detected(self, base_payment):
        """Missing settlement should be detected."""
        # No settlement record
        result = calculate_reconciliation(
            payment=base_payment,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            settlements=[],
            case_id="CASE-000001",
            reconciliation_id="REC-000001",
        )

        assert result.expected_amount == 100000
        assert result.actual_amount == 0
        assert result.difference == 100000
        assert result.match_status == MatchStatus.MISSING
        assert result.exception_type == ExceptionType.MISSING_RECORD


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_reconciliation(self):
        """Test batch reconciliation of multiple payments."""
        payments = [
            Payment(
                payment_id=f"PAY-{i + 1:06d}",
                merchant_id="MER-0001",
                amount=100000,
                currency="INR",
                status=PaymentStatus.CAPTURED,
                payment_timestamp=datetime(2025, 3, 15, 10, 0, 0),
            )
            for i in range(5)
        ]
        settlements = [
            Settlement(
                settlement_id=f"SET-{i + 1:06d}",
                payment_id=f"PAY-{i + 1:06d}",
                merchant_id="MER-0001",
                amount=100000,
                currency="INR",
                status=SettlementStatus.SETTLED,
                settlement_timestamp=datetime(2025, 3, 16, 10, 0, 0),
            )
            for i in range(5)
        ]
        case_mapping = {f"PAY-{i + 1:06d}": f"CASE-{i + 1:06d}" for i in range(5)}

        results = reconcile_batch(
            payments=payments,
            settlements=settlements,
            refunds=[],
            fees=[],
            taxes=[],
            adjustments=[],
            case_mapping=case_mapping,
        )

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.payment_id == f"PAY-{i + 1:06d}"
            assert result.case_id == f"CASE-{i + 1:06d}"
            assert result.expected_amount == 100000
            assert result.difference == 0
