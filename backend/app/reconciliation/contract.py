"""
Reconciliation contract defining deterministic calculation and matching rules.

This module is the single source of truth for:
- How expected amounts are calculated
- How match status is determined
- How exception types are classified

All logic is deterministic. No randomness, no inference, no LLM.
"""

from typing import Dict, List, Optional, Tuple

from app.schemas.enums import (
    ExceptionType,
    MatchStatus,
    MissingRecordSubtype,
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


# ─────────────────────────────────────────────────────────────────────────────
# Calculation Contract
# ─────────────────────────────────────────────────────────────────────────────

# Sign convention (defined once, used everywhere):
# expected_amount = payment - refunds - fees - taxes + adjustments
# difference = expected_amount - actual_amount
#
# Positive difference: expected > actual (under-settled)
# Negative difference: expected < actual (over-settled)
# Zero difference: exact match

EXPECTED_AMOUNT_FORMULA = """
expected_amount = payment_amount
                - total_refunds
                - total_fees
                - total_taxes
                + total_adjustments
"""

DIFFERENCE_FORMULA = """
difference = expected_amount - actual_amount

Positive difference: expected > actual (under-settled)
Negative difference: expected < actual (over-settled)
Zero difference: exact match
"""


def calculate_expected_amount(
    payment_amount: int,
    total_refunds: int,
    total_fees: int,
    total_taxes: int,
    total_adjustments: int,
) -> int:
    """
    Calculate expected settlement amount using integer arithmetic.

    Formula: payment - refunds - fees - taxes + adjustments

    All values in paise (integer minor units).
    No floating-point arithmetic.

    Args:
        payment_amount: Original payment in paise
        total_refunds: Sum of refund amounts in paise
        total_fees: Sum of fee amounts in paise
        total_taxes: Sum of tax amounts in paise
        total_adjustments: Net adjustments in paise (positive=credit, negative=debit)

    Returns:
        Expected settlement amount in paise
    """
    return (
        payment_amount
        - total_refunds
        - total_fees
        - total_taxes
        + total_adjustments
    )


def calculate_difference(expected_amount: int, actual_amount: int) -> int:
    """
    Calculate difference between expected and actual settlement.

    Formula: difference = expected_amount - actual_amount

    Positive: under-settled (expected > actual)
    Negative: over-settled (expected < actual)
    Zero: exact match
    """
    return expected_amount - actual_amount


def create_calculation_breakdown(
    payment_amount: int,
    total_refunds: int,
    total_fees: int,
    total_taxes: int,
    total_adjustments: int,
) -> CalculationBreakdown:
    """
    Create a detailed calculation breakdown for evidence/audit.

    Returns a CalculationBreakdown showing each component.
    """
    return CalculationBreakdown.from_financial_records(
        payment_amount=payment_amount,
        total_refunds=total_refunds,
        total_fees=total_fees,
        total_taxes=total_taxes,
        total_adjustments=total_adjustments,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation Contract
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_financial_records(
    payment: Payment,
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
) -> Dict[str, int]:
    """
    Aggregate financial records for a single payment.

    Returns a dict with aggregated totals:
    - payment_amount
    - total_refunds
    - total_fees
    - total_taxes
    - total_adjustments
    """
    total_refunds = sum(r.amount for r in refunds if r.payment_id == payment.payment_id)
    total_fees = sum(f.amount for f in fees if f.payment_id == payment.payment_id)
    total_taxes = sum(t.amount for t in taxes if t.payment_id == payment.payment_id)
    total_adjustments = sum(
        a.amount for a in adjustments if a.payment_id == payment.payment_id
    )

    return {
        "payment_amount": payment.amount,
        "total_refunds": total_refunds,
        "total_fees": total_fees,
        "total_taxes": total_taxes,
        "total_adjustments": total_adjustments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Matching Contract
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for matching rules:
# 1. Exact ID match
# 2. Valid payment relationship
# 3. Case relationship
# 4. Missing record detection
# 5. Duplicate detection


def determine_match_status(
    has_settlement: bool,
    has_payment: bool,
    difference: int,
    settlement_count: int,
) -> MatchStatus:
    """
    Determine match status using deterministic rules.

    Priority:
    1. MISSING: No settlement record exists
    2. DUPLICATE: Multiple settlements for same payment
    3. MATCHED: Difference is zero
    4. EXCEPTION: Difference is non-zero

    Args:
        has_settlement: Whether a settlement record exists
        has_payment: Whether the payment exists
        difference: expected_amount - actual_amount
        settlement_count: Number of settlements for this payment

    Returns:
        Deterministic MatchStatus
    """
    if not has_settlement:
        return MatchStatus.MISSING

    if settlement_count > 1:
        return MatchStatus.DUPLICATE

    if difference == 0:
        return MatchStatus.MATCHED

    return MatchStatus.EXCEPTION


# ─────────────────────────────────────────────────────────────────────────────
# Exception Classification Contract
# ─────────────────────────────────────────────────────────────────────────────


def classify_exception_type(
    match_status: MatchStatus,
    payment_amount: int,
    total_refunds: int,
    total_fees: int,
    total_taxes: int,
    total_adjustments: int,
    difference: int,
    settlement_count: int,
) -> ExceptionType:
    """
    Classify exception type using deterministic rules.

    This is the core classification logic. It examines the financial
    components and determines which scenario explains the discrepancy.

    Args:
        match_status: Deterministic match status
        payment_amount: Original payment in paise
        total_refunds: Sum of refunds in paise
        total_fees: Sum of fees in paise
        total_taxes: Sum of taxes in paise
        total_adjustments: Net adjustments in paise
        difference: expected_amount - actual_amount
        settlement_count: Number of settlements for this payment

    Returns:
        Deterministic ExceptionType classification
    """
    # Rule 1: Exact match
    if match_status == MatchStatus.MATCHED:
        return ExceptionType.EXACT_MATCH

    # Rule 2: Missing record
    if match_status == MatchStatus.MISSING:
        return ExceptionType.MISSING_RECORD

    # Rule 3: Duplicate
    if match_status == MatchStatus.DUPLICATE:
        return ExceptionType.DUPLICATE

    # Rule 4: Exception classification based on financial components
    abs_diff = abs(difference)

    # Check for fee difference
    # If difference is small and proportional to fees, likely fee error
    if total_fees > 0:
        fee_ratio = abs_diff / total_fees if total_fees > 0 else 0
        if 0.01 <= fee_ratio <= 0.25:  # 1-25% of fees
            return ExceptionType.FEE_DIFFERENCE

    # Check for refund adjustment
    # If difference equals total refunds, refund not accounted for
    if total_refunds > 0 and abs_diff == total_refunds:
        return ExceptionType.REFUND_ADJUSTMENT

    # Check for tax adjustment
    # If difference is proportional to taxes, likely tax error
    if total_taxes > 0:
        tax_ratio = abs_diff / total_taxes if total_taxes > 0 else 0
        if 0.01 <= tax_ratio <= 0.20:  # 1-20% of taxes
            return ExceptionType.TAX_ADJUSTMENT

    # Check for partial settlement
    # If actual is significantly less than expected, likely partial
    if difference > 0 and abs_diff > 1000:  # Under-settled by >₹10
        # Check if it looks like a percentage
        expected = payment_amount - total_refunds - total_fees - total_taxes + total_adjustments
        if expected > 0:
            actual_ratio = (expected - difference) / expected
            if 0.20 <= actual_ratio <= 0.85:  # 20-85% of expected
                return ExceptionType.PARTIAL_SETTLEMENT

    # Check for timing difference
    # If difference is moderate and no clear financial explanation
    if 100 <= abs_diff <= 50000:  # ₹1 to ₹500
        # Timing differences are typically moderate
        return ExceptionType.TIMING_DIFFERENCE

    # Check for complex multi-adjustment
    # If multiple financial components are involved
    components_present = sum([
        total_refunds > 0,
        total_fees > 0,
        total_taxes > 0,
        total_adjustments != 0,
    ])
    if components_present >= 2 and abs_diff > 1000:
        return ExceptionType.COMPLEX_MULTI_ADJUSTMENT

    # Default: unknown
    return ExceptionType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Input Contract
# ─────────────────────────────────────────────────────────────────────────────


class ReconciliationInput:
    """
    Contract for reconciliation engine input.

    The engine reads ONLY financial input records.
    Ground truth labels are NOT included in this input.
    """

    def __init__(
        self,
        payments: List[Payment],
        settlements: List[Settlement],
        refunds: List[Refund],
        fees: List[Fee],
        taxes: List[Tax],
        adjustments: List[Adjustment],
    ):
        self.payments = payments
        self.settlements = settlements
        self.refunds = refunds
        self.fees = fees
        self.taxes = taxes
        self.adjustments = adjustments

        # Build lookup indices
        self._payment_map = {p.payment_id: p for p in payments}
        self._settlement_map: Dict[str, List[Settlement]] = {}
        for s in settlements:
            if s.payment_id not in self._settlement_map:
                self._settlement_map[s.payment_id] = []
            self._settlement_map[s.payment_id].append(s)

    def get_payment(self, payment_id: str) -> Optional[Payment]:
        """Get payment by ID."""
        return self._payment_map.get(payment_id)

    def get_settlements(self, payment_id: str) -> List[Settlement]:
        """Get all settlements for a payment."""
        return self._settlement_map.get(payment_id, [])

    def get_refunds(self, payment_id: str) -> List[Refund]:
        """Get all refunds for a payment."""
        return [r for r in self.refunds if r.payment_id == payment_id]

    def get_fees(self, payment_id: str) -> List[Fee]:
        """Get all fees for a payment."""
        return [f for f in self.fees if f.payment_id == payment_id]

    def get_taxes(self, payment_id: str) -> List[Tax]:
        """Get all taxes for a payment."""
        return [t for t in self.taxes if t.payment_id == payment_id]

    def get_adjustments(self, payment_id: str) -> List[Adjustment]:
        """Get all adjustments for a payment."""
        return [a for a in self.adjustments if a.payment_id == payment_id]
