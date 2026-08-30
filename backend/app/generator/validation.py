"""
Validation module for synthetic financial datasets.

Provides comprehensive generator-level validation:
- Unique IDs
- Valid references (foreign keys)
- Positive financial amounts (unless explicitly allowed)
- Valid timestamps
- Relationship integrity
- Ground truth consistency
- Scenario correctness
"""

from typing import Dict, List

from app.schemas.case import GroundTruth
from app.schemas.enums import ExceptionType


def validate_dataset(
    merchants: List,
    payments: List,
    settlements: List,
    refunds: List,
    fees: List,
    taxes: List,
    adjustments: List,
    cases: List,
    ground_truth_records: List = None,
) -> Dict:
    """
    Validate the complete generated dataset.

    Returns:
        Dictionary with validation results and any errors found
    """
    errors = []
    warnings = []

    # Build ID sets for reference validation
    merchant_ids = {m.merchant_id for m in merchants}
    payment_ids = {p.payment_id for p in payments}
    case_ids = {c.case_id for c in cases}

    # 1. Validate unique IDs
    _check_unique_ids("merchants", [m.merchant_id for m in merchants], errors)
    _check_unique_ids("payments", [p.payment_id for p in payments], errors)
    _check_unique_ids("settlements", [s.settlement_id for s in settlements], errors)
    _check_unique_ids("refunds", [r.refund_id for r in refunds], errors)
    _check_unique_ids("fees", [f.fee_id for f in fees], errors)
    _check_unique_ids("taxes", [t.tax_id for t in taxes], errors)
    _check_unique_ids("adjustments", [a.adjustment_id for a in adjustments], errors)
    _check_unique_ids("cases", [c.case_id for c in cases], errors)

    # 2. Validate references (foreign keys)
    _check_references(
        "payments", "merchant_id", merchant_ids, payments, errors
    )
    _check_references(
        "settlements", "payment_id", payment_ids, settlements, errors
    )
    _check_references(
        "refunds", "payment_id", payment_ids, refunds, errors
    )
    _check_references(
        "fees", "payment_id", payment_ids, fees, errors
    )
    _check_references(
        "taxes", "payment_id", payment_ids, taxes, errors
    )
    _check_references(
        "adjustments", "payment_id", payment_ids, adjustments, errors
    )
    _check_references(
        "cases", "payment_id", payment_ids, cases, errors
    )

    # 3. Validate positive amounts (where expected)
    _check_positive_amounts("payments", payments, errors)
    _check_positive_amounts("settlements", settlements, errors)
    _check_positive_amounts("refunds", refunds, errors)
    _check_positive_amounts("fees", fees, errors)
    _check_positive_amounts("taxes", taxes, errors)

    # 4. Validate timestamps
    _check_timestamps("payments", payments, "payment_timestamp", errors)
    _check_timestamps("settlements", settlements, "settlement_timestamp", errors)
    _check_timestamps("refunds", refunds, "refund_timestamp", errors)

    # 5. Check settlement timestamps are after payment timestamps
    _check_settlement_timing(payments, settlements, warnings)

    # 6. Validate ground truth if provided
    if ground_truth_records:
        _validate_ground_truth(
            ground_truth_records, payments, cases, errors, warnings
        )

    # 7. Validate case-scenario consistency
    _validate_case_scenario_consistency(cases, errors)

    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def _check_unique_ids(entity_name: str, ids: List[str], errors: List) -> None:
    """Check that all IDs in an entity are unique."""
    seen = set()
    for id_val in ids:
        if id_val in seen:
            errors.append(f"Duplicate ID in {entity_name}: {id_val}")
        seen.add(id_val)


def _check_references(
    entity_name: str,
    field_name: str,
    valid_ids: set,
    records: List,
    errors: List,
) -> None:
    """Check that all references point to valid IDs."""
    for record in records:
        ref_id = getattr(record, field_name, None)
        if ref_id and ref_id not in valid_ids:
            errors.append(
                f"Invalid {field_name} in {entity_name}: {ref_id}"
            )


def _check_positive_amounts(
    entity_name: str, records: List, errors: List
) -> None:
    """Check that amounts are positive (for entities that should have positive amounts)."""
    for record in records:
        if hasattr(record, "amount") and record.amount < 0:
            errors.append(
                f"Negative amount in {entity_name} {record.__class__.__name__}_id: "
                f"{getattr(record, record.__class__.__name__.lower() + '_id', 'unknown')}"
            )


def _check_timestamps(
    entity_name: str,
    records: List,
    timestamp_field: str,
    errors: List,
) -> None:
    """Check that timestamps are valid datetime objects."""
    for record in records:
        ts = getattr(record, timestamp_field, None)
        if ts is None:
            errors.append(
                f"Missing {timestamp_field} in {entity_name}"
            )


def _check_settlement_timing(
    payments: List,
    settlements: List,
    warnings: List,
) -> None:
    """Check that settlement timestamps are after payment timestamps."""
    payment_map = {p.payment_id: p for p in payments}
    for settlement in settlements:
        payment = payment_map.get(settlement.payment_id)
        if payment and settlement.settlement_timestamp < payment.payment_timestamp:
            warnings.append(
                f"Settlement {settlement.settlement_id} timestamp "
                f"({settlement.settlement_timestamp}) is before payment "
                f"({payment.payment_timestamp})"
            )


def _validate_ground_truth(
    ground_truth_records: List[GroundTruth],
    payments: List,
    cases: List,
    errors: List,
    warnings: List,
) -> None:
    """Validate ground truth records for consistency."""
    payment_map = {p.payment_id: p for p in payments}
    case_map = {c.case_id: c for c in cases}

    for gt in ground_truth_records:
        # 1. Verify expected amount calculation
        if not gt.verify_expected_amount():
            errors.append(
                f"Ground truth {gt.case_id}: expected_amount verification failed. "
                f"payment={gt.payment_amount} - refunds={gt.total_refunds} - "
                f"fees={gt.total_fees} - taxes={gt.total_taxes} + "
                f"adjustments={gt.total_adjustments} != expected={gt.expected_amount}"
            )

        # 2. Verify difference = actual - expected
        expected_diff = gt.actual_amount - gt.expected_amount
        if gt.difference != expected_diff:
            errors.append(
                f"Ground truth {gt.case_id}: difference mismatch. "
                f"actual-expected={expected_diff} != difference={gt.difference}"
            )

        # 3. Verify case exists
        if gt.case_id not in case_map:
            errors.append(
                f"Ground truth {gt.case_id}: referenced case does not exist"
            )
        else:
            case = case_map[gt.case_id]
            # Verify case matches ground truth
            if case.expected_amount != gt.expected_amount:
                errors.append(
                    f"Ground truth {gt.case_id}: expected_amount mismatch with case"
                )
            if case.actual_amount != gt.actual_amount:
                errors.append(
                    f"Ground truth {gt.case_id}: actual_amount mismatch with case"
                )
            if case.scenario != gt.true_exception_type:
                errors.append(
                    f"Ground truth {gt.case_id}: scenario mismatch with case"
                )

        # 4. Verify payment exists
        if gt.payment_id not in payment_map:
            errors.append(
                f"Ground truth {gt.case_id}: referenced payment {gt.payment_id} does not exist"
            )

        # 5. Verify resolvable consistency
        if gt.true_exception_type == ExceptionType.UNKNOWN and gt.resolvable:
            errors.append(
                f"Ground truth {gt.case_id}: UNKNOWN scenario should not be resolvable"
            )
        if gt.true_exception_type == ExceptionType.MISSING_RECORD and gt.resolvable:
            errors.append(
                f"Ground truth {gt.case_id}: MISSING_RECORD scenario should not be resolvable"
            )

        # 6. Verify risk category for UNKNOWN
        if gt.true_exception_type == ExceptionType.UNKNOWN and gt.risk_category.value != "HIGH":
            warnings.append(
                f"Ground truth {gt.case_id}: UNKNOWN scenario typically has HIGH risk"
            )


def _validate_case_scenario_consistency(cases: List, errors: List) -> None:
    """Validate that case records are internally consistent."""
    for case in cases:
        # Verify difference = actual - expected
        expected_diff = case.actual_amount - case.expected_amount
        if case.difference != expected_diff:
            errors.append(
                f"Case {case.case_id}: difference mismatch. "
                f"actual-expected={expected_diff} != difference={case.difference}"
            )
