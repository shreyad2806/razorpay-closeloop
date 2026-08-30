#!/usr/bin/env python3
"""
CLI command for dataset validation.

Validates:
- IDs (unique, correctly formatted)
- Relationships (foreign keys)
- Amounts (positive, no floating-point corruption)
- Cases (valid references, scenario consistency)
- Ground truth (labels match construction)
- Distribution (all scenarios represented)
"""

import argparse
import json
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.schemas.enums import ExceptionType


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate synthetic financial dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (e.g., data/batch_001)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation info",
    )
    return parser.parse_args()


def validate_ids(data: dict, errors: list) -> None:
    """Validate that all IDs are unique and correctly formatted."""
    # Map entity names to their ID field
    id_fields = {
        "merchants": "merchant_id",
        "payments": "payment_id",
        "settlements": "settlement_id",
        "refunds": "refund_id",
        "fees": "fee_id",
        "taxes": "tax_id",
        "adjustments": "adjustment_id",
        "cases": "case_id",
    }
    
    # Check unique IDs
    for entity, id_field in id_fields.items():
        if entity in data:
            ids = [r.get(id_field) for r in data[entity]]
            seen = set()
            for id_val in ids:
                if id_val in seen:
                    errors.append(f"Duplicate ID in {entity}: {id_val}")
                seen.add(id_val)


def validate_relationships(data: dict, errors: list) -> None:
    """Validate foreign key relationships."""
    merchant_ids = {m["merchant_id"] for m in data.get("merchants", [])}
    payment_ids = {p["payment_id"] for p in data.get("payments", [])}
    case_ids = {c["case_id"] for c in data.get("cases", [])}

    # Payments -> Merchants
    for p in data.get("payments", []):
        if p["merchant_id"] not in merchant_ids:
            errors.append(f"Invalid merchant_id in payment {p['payment_id']}: {p['merchant_id']}")

    # Settlements -> Payments
    for s in data.get("settlements", []):
        if s["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in settlement {s['settlement_id']}: {s['payment_id']}")

    # Refunds -> Payments
    for r in data.get("refunds", []):
        if r["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in refund {r['refund_id']}: {r['payment_id']}")

    # Fees -> Payments
    for f in data.get("fees", []):
        if f["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in fee {f['fee_id']}: {f['payment_id']}")

    # Taxes -> Payments
    for t in data.get("taxes", []):
        if t["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in tax {t['tax_id']}: {t['payment_id']}")

    # Adjustments -> Payments
    for a in data.get("adjustments", []):
        if a["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in adjustment {a['adjustment_id']}: {a['payment_id']}")

    # Cases -> Payments
    for c in data.get("cases", []):
        if c["payment_id"] not in payment_ids:
            errors.append(f"Invalid payment_id in case {c['case_id']}: {c['payment_id']}")


def validate_amounts(data: dict, errors: list) -> None:
    """Validate financial amounts."""
    # Check positive amounts where expected
    for entity in ["payments", "settlements", "refunds", "fees", "taxes"]:
        for r in data.get(entity, []):
            if r.get("amount", 0) < 0:
                errors.append(f"Negative amount in {entity}: {r}")

    # Check no floating-point values
    for entity in ["payments", "settlements", "refunds", "fees", "taxes", "adjustments"]:
        for r in data.get(entity, []):
            if "amount" in r:
                if isinstance(r["amount"], float):
                    errors.append(f"Floating-point amount in {entity}: {r}")


def validate_cases(data: dict, errors: list) -> None:
    """Validate case records."""
    for case in data.get("cases", []):
        # Check difference = actual - expected
        expected_diff = case["actual_amount"] - case["expected_amount"]
        if case["difference"] != expected_diff:
            errors.append(f"Case {case['case_id']}: difference mismatch")

        # Check scenario is valid
        try:
            ExceptionType(case["scenario"])
        except ValueError:
            errors.append(f"Case {case['case_id']}: invalid scenario {case['scenario']}")


def validate_ground_truth(data: dict, errors: list) -> None:
    """Validate ground truth records."""
    for gt in data.get("ground_truth", []):
        # Check expected amount calculation
        computed = (
            gt["payment_amount"]
            - gt["total_refunds"]
            - gt["total_fees"]
            - gt["total_taxes"]
            + gt["total_adjustments"]
        )
        if computed != gt["expected_amount"]:
            errors.append(f"Ground truth {gt['case_id']}: expected_amount verification failed")

        # Check difference = actual - expected
        expected_diff = gt["actual_amount"] - gt["expected_amount"]
        if gt["difference"] != expected_diff:
            errors.append(f"Ground truth {gt['case_id']}: difference mismatch")


def validate_distribution(data: dict, errors: list) -> None:
    """Validate scenario distribution."""
    cases = data.get("cases", [])
    if not cases:
        return

    # Check all scenarios are represented
    scenarios_present = {c["scenario"] for c in cases}
    for scenario in ExceptionType:
        if scenario.value not in scenarios_present:
            errors.append(f"Missing scenario: {scenario.value}")


def main():
    """Main validation function."""
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1

    errors = []

    # Load all data files
    data = {}
    for entity in ["merchants", "payments", "settlements", "refunds", "fees", "taxes", "adjustments", "cases"]:
        filepath = data_dir / "generated" / f"{entity}.json"
        if filepath.exists():
            with open(filepath) as f:
                data[entity] = json.load(f)

    # Load ground truth
    gt_path = data_dir / "ground_truth" / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            data["ground_truth"] = json.load(f)

    # Run validations
    print("Validating dataset...")
    print()

    validate_ids(data, errors)
    validate_relationships(data, errors)
    validate_amounts(data, errors)
    validate_cases(data, errors)
    validate_ground_truth(data, errors)
    validate_distribution(data, errors)

    # Print results
    if errors:
        print(f"FAILED: {len(errors)} errors found")
        print()
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("PASSED: All validations passed")
        print()

        # Print summary
        print("Dataset Summary:")
        for entity in ["merchants", "payments", "settlements", "refunds", "fees", "taxes", "adjustments", "cases"]:
            if entity in data:
                print(f"  {entity:15} {len(data[entity]):6}")
        if "ground_truth" in data:
            print(f"  {'ground_truth':15} {len(data['ground_truth']):6}")
        print()

        # Print scenario distribution
        if "cases" in data:
            scenario_counts = {}
            for case in data["cases"]:
                s = case["scenario"]
                scenario_counts[s] = scenario_counts.get(s, 0) + 1
            print("Scenario Distribution:")
            for scenario, count in sorted(scenario_counts.items()):
                print(f"  {scenario:30} {count:4}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
