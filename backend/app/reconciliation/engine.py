"""
Deterministic financial calculation engine for Razorpay CloseLoop.

This engine independently calculates expected and actual settlement amounts
from financial records. It does NOT read ground truth.

All calculations use integer arithmetic (paise).
No floating-point operations for financial comparison.
No randomness, no inference, no LLM.
"""

from typing import Dict, List, Optional, Tuple

from app.reconciliation.contract import (
    calculate_difference,
    calculate_expected_amount,
    create_calculation_breakdown,
)
from app.reconciliation.matching import (
    collect_matching_evidence,
    detect_duplicates,
    match_and_classify,
)
from app.schemas.enums import (
    AdjustmentType,
    ExceptionType,
    MatchStatus,
    ReconciliationStatus,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
    Tax,
)
from app.schemas.reconciliation import CalculationBreakdown, ReconciliationResult


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation Functions
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_refunds(refunds: List[Refund], payment_id: str) -> int:
    """
    Aggregate all valid refunds for a payment.

    Only includes refunds that belong to the specified payment.
    All amounts are integers in paise.
    """
    return sum(r.amount for r in refunds if r.payment_id == payment_id)


def aggregate_fees(fees: List[Fee], payment_id: str) -> int:
    """
    Aggregate all relevant fees for a payment.

    Inspects actual records — does not assume fees exist.
    """
    return sum(f.amount for f in fees if f.payment_id == payment_id)


def aggregate_taxes(taxes: List[Tax], payment_id: str) -> int:
    """
    Aggregate all relevant tax records for a payment.
    """
    return sum(t.amount for t in taxes if t.payment_id == payment_id)


def aggregate_adjustments(adjustments: List[Adjustment], payment_id: str) -> int:
    """
    Aggregate adjustments with proper sign handling.

    Adjustment types have different meanings:
    - CREDIT: positive (increases settlement)
    - DEBIT: negative (decreases settlement)
    - FEE_REVERSAL: positive (reverses a fee)
    - PENALTY: negative (decreases settlement)
    - BONUS: positive (increases settlement)
    - CORRECTION: signed (can be positive or negative)

    The sign is already embedded in the adjustment amount from the generator.
    """
    return sum(a.amount for a in adjustments if a.payment_id == payment_id)


def aggregate_settlements(settlements: List[Settlement], payment_id: str) -> Tuple[int, int]:
    """
    Aggregate settlement records for a payment.

    Returns:
        Tuple of (total_settlement_amount, settlement_count)

    For duplicate settlements, the total includes all settlements.
    The count is used for duplicate detection.
    """
    relevant = [s for s in settlements if s.payment_id == payment_id]
    total = sum(s.amount for s in relevant)
    return total, len(relevant)


# ─────────────────────────────────────────────────────────────────────────────
# Calculation Engine
# ─────────────────────────────────────────────────────────────────────────────


def calculate_reconciliation(
    payment: Payment,
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
    settlements: List[Settlement],
    case_id: str,
    reconciliation_id: str,
) -> ReconciliationResult:
    """
    Calculate deterministic reconciliation result for a single payment.

    This is the core calculation engine. It:
    1. Uses enhanced matching to collect evidence
    2. Calculates expected settlement amount
    3. Calculates actual settlement amount
    4. Determines difference
    5. Classifies match status and exception type

    Args:
        payment: The payment record
        refunds: All refund records (filtered by payment_id internally)
        fees: All fee records (filtered by payment_id internally)
        taxes: All tax records (filtered by payment_id internally)
        adjustments: All adjustment records (filtered by payment_id internally)
        settlements: All settlement records (filtered by payment_id internally)
        case_id: Reference to the case
        reconciliation_id: Unique reconciliation result ID

    Returns:
        Deterministic ReconciliationResult
    """
    # Step 1: Use enhanced matching to collect evidence and classify
    match_status, exception_type, evidence = match_and_classify(
        payment=payment,
        settlements=settlements,
        refunds=refunds,
        fees=fees,
        taxes=taxes,
        adjustments=adjustments,
    )

    # Step 2: Calculate expected settlement amount
    expected_amount = calculate_expected_amount(
        payment_amount=payment.amount,
        total_refunds=evidence.total_refunds,
        total_fees=evidence.total_fees,
        total_taxes=evidence.total_taxes,
        total_adjustments=evidence.total_adjustments,
    )

    # Step 3: Calculate actual settlement amount
    actual_settlement_total = sum(evidence.settlement_amounts) if evidence.has_settlement else 0

    # Step 4: Calculate difference
    difference = calculate_difference(expected_amount, actual_settlement_total)

    # Step 5: Create result
    result = ReconciliationResult(
        reconciliation_id=reconciliation_id,
        case_id=case_id,
        payment_id=payment.payment_id,
        merchant_id=payment.merchant_id,
        payment_amount=payment.amount,
        total_refunds=evidence.total_refunds,
        total_fees=evidence.total_fees,
        total_taxes=evidence.total_taxes,
        total_adjustments=evidence.total_adjustments,
        expected_amount=expected_amount,
        actual_amount=actual_settlement_total,
        difference=difference,
        match_status=match_status,
        exception_type=exception_type,
        reconciliation_status=ReconciliationStatus.PROCESSED,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────


def reconcile_batch(
    payments: List[Payment],
    settlements: List[Settlement],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
    case_mapping: Dict[str, str],  # payment_id -> case_id
) -> List[ReconciliationResult]:
    """
    Process a batch of payments and produce reconciliation results.

    Args:
        payments: List of payment records
        settlements: List of settlement records
        refunds: List of refund records
        fees: List of fee records
        taxes: List of tax records
        adjustments: List of adjustment records
        case_mapping: Mapping from payment_id to case_id

    Returns:
        List of deterministic ReconciliationResult objects
    """
    results = []

    for i, payment in enumerate(payments):
        reconciliation_id = f"REC-{i + 1:06d}"
        case_id = case_mapping.get(payment.payment_id, f"CASE-{i + 1:06d}")

        result = calculate_reconciliation(
            payment=payment,
            refunds=refunds,
            fees=fees,
            taxes=taxes,
            adjustments=adjustments,
            settlements=settlements,
            case_id=case_id,
            reconciliation_id=reconciliation_id,
        )
        results.append(result)

    return results
