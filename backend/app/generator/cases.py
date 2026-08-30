"""
Case generator for synthetic financial data.

Generates deterministic reconciliation cases with scenario injection.
Each scenario deliberately modifies financial records to create realistic
discrepancies that can be traced and verified.

Ground truth is generated using the Pydantic model for validation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.generator.rng import DeterministicRNG
from app.generator.scenarios import get_scenario_definition
from app.schemas.case import Case, GroundTruth
from app.schemas.config import GeneratorConfig
from app.schemas.enums import (
    ExceptionType,
    MissingRecordSubtype,
    ResolutionType,
    RiskCategory,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
    Tax,
)


def _aggregate_financial_data(
    payments: List[Payment],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
) -> Dict[str, Dict]:
    """
    Aggregate financial data by payment_id for efficient lookup.

    Returns a dict mapping payment_id to aggregated totals.
    """
    agg: Dict[str, Dict] = {}

    for p in payments:
        agg[p.payment_id] = {
            "payment_amount": p.amount,
            "merchant_id": p.merchant_id,
            "payment_timestamp": p.payment_timestamp,
            "total_refunds": 0,
            "total_fees": 0,
            "total_taxes": 0,
            "total_adjustments": 0,
            "refund_ids": [],
            "fee_ids": [],
            "tax_ids": [],
            "adjustment_ids": [],
        }

    for r in refunds:
        if r.payment_id in agg:
            agg[r.payment_id]["total_refunds"] += r.amount
            agg[r.payment_id]["refund_ids"].append(r.refund_id)

    for f in fees:
        if f.payment_id in agg:
            agg[f.payment_id]["total_fees"] += f.amount
            agg[f.payment_id]["fee_ids"].append(f.fee_id)

    for t in taxes:
        if t.payment_id in agg:
            agg[t.payment_id]["total_taxes"] += t.amount
            agg[t.payment_id]["tax_ids"].append(t.tax_id)

    for a in adjustments:
        if a.payment_id in agg:
            agg[a.payment_id]["total_adjustments"] += a.amount
            agg[a.payment_id]["adjustment_ids"].append(a.adjustment_id)

    return agg


def _compute_expected_settlement(agg: Dict) -> int:
    """
    Compute expected settlement amount using integer arithmetic.

    Formula: payment - refunds - fees - taxes + adjustments

    All values are in paise (integer minor units).
    """
    return (
        agg["payment_amount"]
        - agg["total_refunds"]
        - agg["total_fees"]
        - agg["total_taxes"]
        + agg["total_adjustments"]
    )


def _assign_scenario(
    rng: DeterministicRNG,
    config: GeneratorConfig,
) -> ExceptionType:
    """Assign an exception scenario based on configured distribution."""
    scenarios = list(config.scenario_distribution.weights.keys())
    weights = list(config.scenario_distribution.weights.values())
    return rng.choices(scenarios, weights=weights, k=1)[0]


def _determine_risk(
    scenario: ExceptionType,
    difference: int,
    rng: DeterministicRNG,
) -> RiskCategory:
    """Determine risk category based on scenario and discrepancy size."""
    if scenario == ExceptionType.EXACT_MATCH:
        return RiskCategory.LOW
    if scenario == ExceptionType.UNKNOWN:
        return RiskCategory.HIGH
    if scenario == ExceptionType.DUPLICATE:
        return RiskCategory.HIGH
    if scenario == ExceptionType.MISSING_RECORD:
        return RiskCategory.HIGH

    abs_diff = abs(difference)
    if abs_diff > 100000:  # > ₹1000
        return RiskCategory.HIGH
    elif abs_diff > 10000:  # > ₹100
        return RiskCategory.MEDIUM
    else:
        return RiskCategory.LOW


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Injectors
# ─────────────────────────────────────────────────────────────────────────────

def _inject_exact_match(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """EXACT_MATCH: expected == actual, no discrepancy."""
    expected = _compute_expected_settlement(payment_agg)
    return expected, expected, 0, None


def _inject_fee_difference(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    FEE_DIFFERENCE: Settlement applied wrong fee rate.

    The settlement amount reflects a fee that is different from the
    actual fee recorded. The difference equals the fee error.
    """
    expected = _compute_expected_settlement(payment_agg)
    # Fee error: 5-20% of the actual fees
    fee_error_pct = rng.random_percentage(0.05, 0.20)
    fee_error = int(payment_agg["total_fees"] * fee_error_pct)
    # Settlement used lower fees → higher actual settlement
    actual = expected + fee_error
    return expected, actual, actual - expected, None


def _inject_refund_adjustment(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    REFUND_ADJUSTMENT: Settlement doesn't account for a refund.

    The refund exists but settlement was computed without it.
    """
    expected = _compute_expected_settlement(payment_agg)
    if payment_agg["total_refunds"] > 0:
        # Settlement didn't deduct the refund
        actual = expected + payment_agg["total_refunds"]
        return expected, actual, actual - expected, None
    else:
        # No refunds → fall back to exact match
        return expected, expected, 0, None


def _inject_tax_adjustment(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    TAX_ADJUSTMENT: Settlement used incorrect tax calculation.

    The tax record shows correct amount, but settlement applied different tax.
    """
    expected = _compute_expected_settlement(payment_agg)
    tax_error_pct = rng.random_percentage(0.05, 0.15)
    tax_error = int(payment_agg["total_taxes"] * tax_error_pct)
    # Settlement used lower tax → higher actual settlement
    actual = expected + tax_error
    return expected, actual, actual - expected, None


def _inject_timing_difference(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    TIMING_DIFFERENCE: Settlement observed before all records arrived.

    The expected amount is correct based on all records, but the actual
    settlement was observed at an earlier point in time when fewer records
    were available. The difference represents records that arrived after
    the settlement observation.
    """
    expected = _compute_expected_settlement(payment_agg)
    # Timing lag: 10-50% of expected amount was not yet available
    timing_offset_pct = rng.random_percentage(0.10, 0.50)
    timing_offset = int(expected * timing_offset_pct)
    # Actual settlement is lower because records arrived later
    actual = expected - timing_offset
    return expected, actual, actual - expected, None


def _inject_partial_settlement(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    PARTIAL_SETTLEMENT: Only part of expected amount was settled.

    The remaining amount is pending or delayed.
    """
    expected = _compute_expected_settlement(payment_agg)
    # Settled 30-80% of expected
    partial_pct = rng.random_percentage(0.30, 0.80)
    actual = int(expected * partial_pct)
    return expected, actual, actual - expected, None


def _inject_duplicate(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    DUPLICATE: Settlement processed twice.

    The actual observed settlement amount is double the expected amount
    because the same settlement was processed twice.
    """
    expected = _compute_expected_settlement(payment_agg)
    actual = expected * 2
    return expected, actual, actual - expected, None


def _inject_missing_record(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    MISSING_RECORD: A required financial record is omitted.

    The actual settlement reflects the payment amount without the
    missing record's contribution.
    """
    expected = _compute_expected_settlement(payment_agg)

    # Choose which record is missing
    missing_subtypes = [
        MissingRecordSubtype.MISSING_SETTLEMENT,
        MissingRecordSubtype.MISSING_REFUND,
        MissingRecordSubtype.MISSING_FEE,
        MissingRecordSubtype.MISSING_TAX,
    ]
    # Filter to subtypes that make sense for this payment
    available = []
    if payment_agg["total_refunds"] > 0:
        available.append(MissingRecordSubtype.MISSING_REFUND)
    if payment_agg["total_fees"] > 0:
        available.append(MissingRecordSubtype.MISSING_FEE)
    if payment_agg["total_taxes"] > 0:
        available.append(MissingRecordSubtype.MISSING_TAX)
    available.append(MissingRecordSubtype.MISSING_SETTLEMENT)

    subtype = rng.choice(available)

    if subtype == MissingRecordSubtype.MISSING_SETTLEMENT:
        # Settlement record doesn't exist → actual = 0
        actual = 0
    elif subtype == MissingRecordSubtype.MISSING_REFUND:
        # Refund record is missing → settlement didn't deduct refund
        actual = expected + payment_agg["total_refunds"]
    elif subtype == MissingRecordSubtype.MISSING_FEE:
        # Fee record is missing → settlement didn't deduct fee
        actual = expected + payment_agg["total_fees"]
    elif subtype == MissingRecordSubtype.MISSING_TAX:
        # Tax record is missing → settlement didn't deduct tax
        actual = expected + payment_agg["total_taxes"]
    else:
        actual = expected

    return expected, actual, actual - expected, subtype


def _inject_complex_multi_adjustment(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    COMPLEX_MULTI_ADJUSTMENT: Multiple factors explain the discrepancy.

    Example: refund + fee error + tax error together explain the difference.
    The ground truth must prove that the combination explains it.
    """
    expected = _compute_expected_settlement(payment_agg)

    # Combine multiple factors
    factors = []

    # Factor 1: Refund not accounted for (if refunds exist)
    if payment_agg["total_refunds"] > 0:
        factors.append(payment_agg["total_refunds"])

    # Factor 2: Fee error
    if payment_agg["total_fees"] > 0:
        fee_error_pct = rng.random_percentage(0.05, 0.15)
        fee_error = int(payment_agg["total_fees"] * fee_error_pct)
        factors.append(fee_error)

    # Factor 3: Tax error
    if payment_agg["total_taxes"] > 0:
        tax_error_pct = rng.random_percentage(0.03, 0.10)
        tax_error = int(payment_agg["total_taxes"] * tax_error_pct)
        factors.append(-tax_error)  # Negative because lower tax = higher settlement

    # Ensure we have at least 2 factors
    if len(factors) < 2:
        # Add a random adjustment factor
        adj_factor = rng.randint(1000, 10000)
        factors.append(adj_factor)

    total_discrepancy = sum(factors)
    actual = expected + total_discrepancy

    return expected, actual, actual - expected, None


def _inject_unknown(
    payment_agg: Dict,
    rng: DeterministicRNG,
) -> Tuple[int, int, int, Optional[MissingRecordSubtype]]:
    """
    UNKNOWN: Genuinely unresolved pattern.

    The discrepancy cannot be explained by any known financial event.
    Uses a random offset that is not correlated with any existing record.
    """
    expected = _compute_expected_settlement(payment_agg)
    # Random offset: ±₹100-₹1000
    unknown_offset = rng.randint(-100000, 100000)
    actual = expected + unknown_offset
    return expected, actual, actual - expected, None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Injector Registry
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_INJECTORS = {
    ExceptionType.EXACT_MATCH: _inject_exact_match,
    ExceptionType.FEE_DIFFERENCE: _inject_fee_difference,
    ExceptionType.REFUND_ADJUSTMENT: _inject_refund_adjustment,
    ExceptionType.TAX_ADJUSTMENT: _inject_tax_adjustment,
    ExceptionType.TIMING_DIFFERENCE: _inject_timing_difference,
    ExceptionType.PARTIAL_SETTLEMENT: _inject_partial_settlement,
    ExceptionType.DUPLICATE: _inject_duplicate,
    ExceptionType.MISSING_RECORD: _inject_missing_record,
    ExceptionType.COMPLEX_MULTI_ADJUSTMENT: _inject_complex_multi_adjustment,
    ExceptionType.UNKNOWN: _inject_unknown,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_cases(
    config: GeneratorConfig,
    payments: List[Payment],
    refunds: List[Refund],
    fees: List[Fee],
    taxes: List[Tax],
    adjustments: List[Adjustment],
    settlements: List[Settlement],
    rng: DeterministicRNG,
) -> Tuple[List[Case], List[GroundTruth]]:
    """
    Generate reconciliation cases with scenario injection.

    This function:
    1. Aggregates financial data per payment
    2. Computes expected settlement amounts
    3. Assigns exception scenarios based on distribution
    4. Injects discrepancies to create actual settlement amounts
    5. Generates Case records and GroundTruth using Pydantic models

    Args:
        config: Generator configuration
        payments: List of generated payments
        refunds: List of generated refunds
        fees: List of generated fees
        taxes: List of generated taxes
        adjustments: List of generated adjustments
        settlements: List of generated settlements
        rng: Deterministic random number generator

    Returns:
        Tuple of (cases, ground_truth_records)
    """
    # Aggregate financial data
    agg = _aggregate_financial_data(payments, refunds, fees, taxes, adjustments)

    cases = []
    ground_truth_records = []

    for i, payment in enumerate(payments):
        case_id = f"CASE-{i + 1:06d}"
        payment_agg = agg[payment.payment_id]

        # Assign scenario
        scenario = _assign_scenario(rng, config)

        # Get scenario definition
        scenario_def = get_scenario_definition(scenario)

        # Inject scenario-specific discrepancy
        injector = SCENARIO_INJECTORS[scenario]
        expected, actual, difference, missing_subtype = injector(payment_agg, rng)

        # If REFUND_ADJUSTMENT had no refunds, fall back to EXACT_MATCH
        if scenario == ExceptionType.REFUND_ADJUSTMENT and expected == actual:
            scenario = ExceptionType.EXACT_MATCH
            scenario_def = get_scenario_definition(scenario)

        # Determine risk and resolution
        risk_category = _determine_risk(scenario, difference, rng)
        resolution = scenario_def.resolution
        resolvable = scenario_def.resolvable

        # Create case
        case = Case(
            case_id=case_id,
            payment_id=payment.payment_id,
            merchant_id=payment.merchant_id,
            expected_amount=expected,
            actual_amount=actual,
            difference=difference,
            scenario=scenario,
            observation_timestamp=payment.payment_timestamp,
            resolvable=resolvable,
            risk_category=risk_category,
        )
        cases.append(case)

        # Create ground truth record using Pydantic model
        ground_truth = GroundTruth(
            case_id=case_id,
            payment_id=payment.payment_id,
            payment_amount=payment_agg["payment_amount"],
            total_refunds=payment_agg["total_refunds"],
            total_fees=payment_agg["total_fees"],
            total_taxes=payment_agg["total_taxes"],
            total_adjustments=payment_agg["total_adjustments"],
            expected_amount=expected,
            actual_amount=actual,
            difference=difference,
            true_exception_type=scenario,
            true_resolution=resolution,
            resolvable=resolvable,
            risk_category=risk_category,
        )

        # Verify ground truth consistency
        assert ground_truth.verify_expected_amount(), (
            f"Ground truth verification failed for {case_id}: "
            f"payment={payment_agg['payment_amount']} - "
            f"refunds={payment_agg['total_refunds']} - "
            f"fees={payment_agg['total_fees']} - "
            f"taxes={payment_agg['total_taxes']} + "
            f"adjustments={payment_agg['total_adjustments']} != "
            f"expected={expected}"
        )

        ground_truth_records.append(ground_truth)

    return cases, ground_truth_records
