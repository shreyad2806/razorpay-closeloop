"""
Deterministic matching and exception detection for Razorpay CloseLoop.

This module provides enhanced matching logic that:
1. Detects missing records
2. Detects duplicate settlements
3. Classifies exception types based on financial evidence

All logic is deterministic. No ground truth is used.
"""

from typing import Dict, List, Optional, Set, Tuple

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


# ─────────────────────────────────────────────────────────────────────────────
# Matching Evidence
# ─────────────────────────────────────────────────────────────────────────────


class MatchingEvidence:
    """
    Evidence collected during the matching process.

    This is used for deterministic classification and audit trail.
    """

    def __init__(
        self,
        payment_id: str,
        has_settlement: bool,
        settlement_count: int,
        settlement_amounts: List[int],
        has_refunds: bool,
        refund_count: int,
        total_refunds: int,
        has_fees: bool,
        fee_count: int,
        total_fees: int,
        has_taxes: bool,
        tax_count: int,
        total_taxes: int,
        has_adjustments: bool,
        adjustment_count: int,
        total_adjustments: int,
    ):
        self.payment_id = payment_id
        self.has_settlement = has_settlement
        self.settlement_count = settlement_count
        self.settlement_amounts = settlement_amounts
        self.has_refunds = has_refunds
        self.refund_count = refund_count
        self.total_refunds = total_refunds
        self.has_fees = has_fees
        self.fee_count = fee_count
        self.total_fees = total_fees
        self.has_taxes = has_taxes
        self.tax_count = tax_count
        self.total_taxes = total_taxes
        self.has_adjustments = has_adjustments
        self.adjustment_count = adjustment_count
        self.total_adjustments = total_adjustments


# ─────────────────────────────────────────────────────────────────────────────
# Missing Record Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_missing_records(
    payment: Payment,
    settlements: List[Settlement],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
) -> List[MissingRecordSubtype]:
    """
    Detect missing financial records for a payment.

    Returns a list of missing record subtypes.
    An empty list means no missing records detected.
    """
    missing = []

    # Check for missing settlement
    payment_settlements = [s for s in settlements if s.payment_id == payment.payment_id]
    if len(payment_settlements) == 0:
        missing.append(MissingRecordSubtype.MISSING_SETTLEMENT)

    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_duplicates(
    payment_id: str,
    settlements: List[Settlement],
) -> bool:
    """
    Detect duplicate settlement records for a payment.

    A duplicate is detected when:
    1. Multiple settlement records exist for the same payment_id
    2. The settlement amounts are identical (or very close)

    Returns True if duplicates are detected.
    """
    payment_settlements = [s for s in settlements if s.payment_id == payment_id]

    if len(payment_settlements) <= 1:
        return False

    # Check if amounts are identical (indicating duplicate)
    amounts = [s.amount for s in payment_settlements]
    if len(set(amounts)) == 1:
        # All amounts are identical - likely duplicate
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_matching_evidence(
    payment: Payment,
    settlements: List[Settlement],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
) -> MatchingEvidence:
    """
    Collect evidence from financial records for a payment.

    This evidence is used for deterministic classification.
    """
    # Filter records for this payment
    payment_settlements = [s for s in settlements if s.payment_id == payment.payment_id]
    payment_refunds = [r for r in refunds if r.payment_id == payment.payment_id]
    payment_fees = [f for f in fees if f.payment_id == payment.payment_id]
    payment_taxes = [t for t in taxes if t.payment_id == payment.payment_id]
    payment_adjustments = [a for a in adjustments if a.payment_id == payment.payment_id]

    return MatchingEvidence(
        payment_id=payment.payment_id,
        has_settlement=len(payment_settlements) > 0,
        settlement_count=len(payment_settlements),
        settlement_amounts=[s.amount for s in payment_settlements],
        has_refunds=len(payment_refunds) > 0,
        refund_count=len(payment_refunds),
        total_refunds=sum(r.amount for r in payment_refunds),
        has_fees=len(payment_fees) > 0,
        fee_count=len(payment_fees),
        total_fees=sum(f.amount for f in payment_fees),
        has_taxes=len(payment_taxes) > 0,
        tax_count=len(payment_taxes),
        total_taxes=sum(t.amount for t in payment_taxes),
        has_adjustments=len(payment_adjustments) > 0,
        adjustment_count=len(payment_adjustments),
        total_adjustments=sum(a.amount for a in payment_adjustments),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exception Classification
# ─────────────────────────────────────────────────────────────────────────────


def classify_exception_deterministic(
    match_status: MatchStatus,
    difference: int,
    evidence: MatchingEvidence,
    payment_amount: int,
) -> ExceptionType:
    """
    Classify exception type using deterministic rules based on evidence.

    This is the enhanced classification logic that uses collected evidence
    to make more accurate classifications.

    Args:
        match_status: Deterministic match status
        difference: expected_amount - actual_amount
        evidence: Collected matching evidence
        payment_amount: Original payment amount

    Returns:
        Deterministic ExceptionType classification
    """
    abs_diff = abs(difference)

    # Rule 1: Exact match
    if match_status == MatchStatus.MATCHED:
        return ExceptionType.EXACT_MATCH

    # Rule 2: Missing record
    if match_status == MatchStatus.MISSING:
        return ExceptionType.MISSING_RECORD

    # Rule 3: Duplicate
    if match_status == MatchStatus.DUPLICATE:
        return ExceptionType.DUPLICATE

    # Rule 4: Exception classification based on evidence

    # Calculate expected amount for reference
    expected = (
        payment_amount
        - evidence.total_refunds
        - evidence.total_fees
        - evidence.total_taxes
        + evidence.total_adjustments
    )

    # Check for refund adjustment
    # If difference equals total refunds, refund not accounted for
    if evidence.has_refunds and abs_diff == evidence.total_refunds:
        return ExceptionType.REFUND_ADJUSTMENT

    # Check for fee difference
    # If difference is proportional to fees, likely fee error
    if evidence.has_fees and evidence.total_fees > 0:
        fee_ratio = abs_diff / evidence.total_fees
        if 0.01 <= fee_ratio <= 0.25:  # 1-25% of fees
            return ExceptionType.FEE_DIFFERENCE

    # Check for tax adjustment
    # If difference is proportional to taxes, likely tax error
    if evidence.has_taxes and evidence.total_taxes > 0:
        tax_ratio = abs_diff / evidence.total_taxes
        if 0.01 <= tax_ratio <= 0.20:  # 1-20% of taxes
            return ExceptionType.TAX_ADJUSTMENT

    # Check for partial settlement
    # If actual is significantly less than expected, likely partial
    if difference > 0 and abs_diff > 1000:  # Under-settled by >₹10
        if expected > 0:
            actual_amount = expected - difference
            actual_ratio = actual_amount / expected
            if 0.20 <= actual_ratio <= 0.85:  # 20-85% of expected
                return ExceptionType.PARTIAL_SETTLEMENT

    # Check for timing difference
    # If difference is moderate and no clear financial explanation
    if 100 <= abs_diff <= 50000:  # ₹1 to ₹500
        return ExceptionType.TIMING_DIFFERENCE

    # Check for complex multi-adjustment
    # If multiple financial components are involved and contribute to discrepancy
    components_present = sum([
        evidence.has_refunds,
        evidence.has_fees,
        evidence.has_taxes,
        evidence.has_adjustments,
    ])
    if components_present >= 2 and abs_diff > 1000:
        return ExceptionType.COMPLEX_MULTI_ADJUSTMENT

    # Default: unknown
    return ExceptionType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Matching Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def match_and_classify(
    payment: Payment,
    settlements: List[Settlement],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
) -> Tuple[MatchStatus, ExceptionType, MatchingEvidence]:
    """
    Perform deterministic matching and classification for a payment.

    This is the enhanced matching pipeline that:
    1. Collects evidence from financial records
    2. Detects missing records
    3. Detects duplicates
    4. Classifies exception type

    Args:
        payment: The payment record
        settlements: All settlement records
        refunds: All refund records
        fees: All fee records
        taxes: All tax records
        adjustments: All adjustment records

    Returns:
        Tuple of (MatchStatus, ExceptionType, MatchingEvidence)
    """
    # Step 1: Collect evidence
    evidence = collect_matching_evidence(
        payment=payment,
        settlements=settlements,
        refunds=refunds,
        fees=fees,
        taxes=taxes,
        adjustments=adjustments,
    )

    # Step 2: Calculate expected amount
    expected = (
        payment.amount
        - evidence.total_refunds
        - evidence.total_fees
        - evidence.total_taxes
        + evidence.total_adjustments
    )

    # Step 3: Calculate actual amount
    actual = sum(evidence.settlement_amounts) if evidence.has_settlement else 0

    # Step 4: Calculate difference
    difference = expected - actual

    # Step 5: Determine match status
    if not evidence.has_settlement:
        match_status = MatchStatus.MISSING
    elif evidence.settlement_count > 1 and detect_duplicates(
        payment.payment_id, settlements
    ):
        match_status = MatchStatus.DUPLICATE
    elif difference == 0:
        match_status = MatchStatus.MATCHED
    else:
        match_status = MatchStatus.EXCEPTION

    # Step 6: Classify exception type
    exception_type = classify_exception_deterministic(
        match_status=match_status,
        difference=difference,
        evidence=evidence,
        payment_amount=payment.amount,
    )

    return match_status, exception_type, evidence
