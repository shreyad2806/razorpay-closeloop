"""
Seed a curated demo batch for Razorpay CloseLoop.

Creates ~30 realistic financial reconciliation exceptions with:
- Diverse exception types (Exact Match, Partial Settlement, Timing, Tax, Fee, Duplicate, Missing, Unknown)
- Meaningful risk distribution (LOW, MEDIUM, HIGH)
- A mix of statuses (PENDING, APPROVED, REJECTED, ESCALATED)
- Internally consistent financial data
- Realistic merchant/payment/settlement relationships

This script is designed to be run once during startup to provide
a focused, realistic demo dataset.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

# Ensure the backend root is on the path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

DATA_DIR = BACKEND_ROOT / "data"
DEMO_BATCH_ID = "DEMO-001"

# ─── Curated Demo Cases ─────────────────────────────────────────────────────
# Each case represents a realistic financial reconciliation scenario.

DEMO_CASES = [
    # EXACT MATCH — perfectly reconciled
    {
        "case_id": "CASE-DEMO-001", "payment_id": "PAY-DEMO-001", "merchant_id": "MER-1001",
        "expected_amount": 4500000, "actual_amount": 4500000, "difference": 0,
        "scenario": "EXACT_MATCH", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-002", "payment_id": "PAY-DEMO-002", "merchant_id": "MER-1002",
        "expected_amount": 2350000, "actual_amount": 2350000, "difference": 0,
        "scenario": "EXACT_MATCH", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-003", "payment_id": "PAY-DEMO-003", "merchant_id": "MER-1003",
        "expected_amount": 8920000, "actual_amount": 8920000, "difference": 0,
        "scenario": "EXACT_MATCH", "risk_category": "LOW", "resolvable": True,
    },

    # PARTIAL SETTLEMENT — settlement received but less than payment
    {
        "case_id": "CASE-DEMO-004", "payment_id": "PAY-DEMO-004", "merchant_id": "MER-1001",
        "expected_amount": 6750000, "actual_amount": 5400000, "difference": -1350000,
        "scenario": "PARTIAL_SETTLEMENT", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-005", "payment_id": "PAY-DEMO-005", "merchant_id": "MER-1004",
        "expected_amount": 3200000, "actual_amount": 2880000, "difference": -320000,
        "scenario": "PARTIAL_SETTLEMENT", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-006", "payment_id": "PAY-DEMO-006", "merchant_id": "MER-1005",
        "expected_amount": 12500000, "actual_amount": 8750000, "difference": -3750000,
        "scenario": "PARTIAL_SETTLEMENT", "risk_category": "HIGH", "resolvable": True,
    },

    # TIMING DIFFERENCE — settlement arrived on a different date
    {
        "case_id": "CASE-DEMO-007", "payment_id": "PAY-DEMO-007", "merchant_id": "MER-1002",
        "expected_amount": 5600000, "actual_amount": 0, "difference": -5600000,
        "scenario": "TIMING_DIFFERENCE", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-008", "payment_id": "PAY-DEMO-008", "merchant_id": "MER-1006",
        "expected_amount": 1890000, "actual_amount": 0, "difference": -1890000,
        "scenario": "TIMING_DIFFERENCE", "risk_category": "MEDIUM", "resolvable": True,
    },

    # TAX ADJUSTMENT — GST/tax withholding applied
    {
        "case_id": "CASE-DEMO-009", "payment_id": "PAY-DEMO-009", "merchant_id": "MER-1003",
        "expected_amount": 7800000, "actual_amount": 6962000, "difference": -838000,
        "scenario": "TAX_ADJUSTMENT", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-010", "payment_id": "PAY-DEMO-010", "merchant_id": "MER-1007",
        "expected_amount": 4250000, "actual_amount": 3782500, "difference": -467500,
        "scenario": "TAX_ADJUSTMENT", "risk_category": "LOW", "resolvable": True,
    },

    # FEE DIFFERENCE — processing fee deducted
    {
        "case_id": "CASE-DEMO-011", "payment_id": "PAY-DEMO-011", "merchant_id": "MER-1001",
        "expected_amount": 9500000, "actual_amount": 9025000, "difference": -475000,
        "scenario": "FEE_DIFFERENCE", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-012", "payment_id": "PAY-DEMO-012", "merchant_id": "MER-1008",
        "expected_amount": 3100000, "actual_amount": 2945000, "difference": -155000,
        "scenario": "FEE_DIFFERENCE", "risk_category": "LOW", "resolvable": True,
    },

    # DUPLICATE — same payment processed twice
    {
        "case_id": "CASE-DEMO-013", "payment_id": "PAY-DEMO-013", "merchant_id": "MER-1004",
        "expected_amount": 4100000, "actual_amount": 8200000, "difference": 4100000,
        "scenario": "DUPLICATE", "risk_category": "HIGH", "resolvable": False,
    },
    {
        "case_id": "CASE-DEMO-014", "payment_id": "PAY-DEMO-014", "merchant_id": "MER-1009",
        "expected_amount": 2750000, "actual_amount": 5500000, "difference": 2750000,
        "scenario": "DUPLICATE", "risk_category": "HIGH", "resolvable": False,
    },

    # MISSING RECORD — payment exists but no settlement found
    {
        "case_id": "CASE-DEMO-015", "payment_id": "PAY-DEMO-015", "merchant_id": "MER-1005",
        "expected_amount": 5200000, "actual_amount": 0, "difference": -5200000,
        "scenario": "MISSING_RECORD", "risk_category": "HIGH", "resolvable": False,
    },
    {
        "case_id": "CASE-DEMO-016", "payment_id": "PAY-DEMO-016", "merchant_id": "MER-1010",
        "expected_amount": 1750000, "actual_amount": 0, "difference": -1750000,
        "scenario": "MISSING_RECORD", "risk_category": "MEDIUM", "resolvable": True,
    },

    # COMPLEX_MULTI_ADJUSTMENT — multiple adjustments applied
    {
        "case_id": "CASE-DEMO-017", "payment_id": "PAY-DEMO-017", "merchant_id": "MER-1002",
        "expected_amount": 11200000, "actual_amount": 9856000, "difference": -1344000,
        "scenario": "COMPLEX_MULTI_ADJUSTMENT", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-018", "payment_id": "PAY-DEMO-018", "merchant_id": "MER-1011",
        "expected_amount": 6800000, "actual_amount": 7480000, "difference": 680000,
        "scenario": "COMPLEX_MULTI_ADJUSTMENT", "risk_category": "MEDIUM", "resolvable": True,
    },

    # UNKNOWN — unclassifiable discrepancy
    {
        "case_id": "CASE-DEMO-019", "payment_id": "PAY-DEMO-019", "merchant_id": "MER-1006",
        "expected_amount": 8450000, "actual_amount": 7100000, "difference": -1350000,
        "scenario": "UNKNOWN", "risk_category": "HIGH", "resolvable": False,
    },
    {
        "case_id": "CASE-DEMO-020", "payment_id": "PAY-DEMO-020", "merchant_id": "MER-1012",
        "expected_amount": 2100000, "actual_amount": 3300000, "difference": 1200000,
        "scenario": "UNKNOWN", "risk_category": "HIGH", "resolvable": False,
    },

    # More varied cases for demo depth
    {
        "case_id": "CASE-DEMO-021", "payment_id": "PAY-DEMO-021", "merchant_id": "MER-1003",
        "expected_amount": 3650000, "actual_amount": 3650000, "difference": 0,
        "scenario": "EXACT_MATCH", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-022", "payment_id": "PAY-DEMO-022", "merchant_id": "MER-1007",
        "expected_amount": 5900000, "actual_amount": 4720000, "difference": -1180000,
        "scenario": "PARTIAL_SETTLEMENT", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-023", "payment_id": "PAY-DEMO-023", "merchant_id": "MER-1008",
        "expected_amount": 7200000, "actual_amount": 6408000, "difference": -792000,
        "scenario": "TAX_ADJUSTMENT", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-024", "payment_id": "PAY-DEMO-024", "merchant_id": "MER-1009",
        "expected_amount": 4800000, "actual_amount": 4560000, "difference": -240000,
        "scenario": "FEE_DIFFERENCE", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-025", "payment_id": "PAY-DEMO-025", "merchant_id": "MER-1010",
        "expected_amount": 13500000, "actual_amount": 13500000, "difference": 0,
        "scenario": "EXACT_MATCH", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-026", "payment_id": "PAY-DEMO-026", "merchant_id": "MER-1011",
        "expected_amount": 2900000, "actual_amount": 2610000, "difference": -290000,
        "scenario": "PARTIAL_SETTLEMENT", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-027", "payment_id": "PAY-DEMO-027", "merchant_id": "MER-1012",
        "expected_amount": 6100000, "actual_amount": 0, "difference": -6100000,
        "scenario": "TIMING_DIFFERENCE", "risk_category": "MEDIUM", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-028", "payment_id": "PAY-DEMO-028", "merchant_id": "MER-1001",
        "expected_amount": 4300000, "actual_amount": 8600000, "difference": 4300000,
        "scenario": "DUPLICATE", "risk_category": "HIGH", "resolvable": False,
    },
    {
        "case_id": "CASE-DEMO-029", "payment_id": "PAY-DEMO-029", "merchant_id": "MER-1004",
        "expected_amount": 1500000, "actual_amount": 1320000, "difference": -180000,
        "scenario": "FEE_DIFFERENCE", "risk_category": "LOW", "resolvable": True,
    },
    {
        "case_id": "CASE-DEMO-030", "payment_id": "PAY-DEMO-030", "merchant_id": "MER-1005",
        "expected_amount": 9800000, "actual_amount": 10290000, "difference": 490000,
        "scenario": "UNKNOWN", "risk_category": "HIGH", "resolvable": False,
    },
]

# ─── Status assignments ─────────────────────────────────────────────────────
# Distribute statuses to create a realistic workflow state.

STATUS_ASSIGNMENTS = {
    # APPROVED (5) — reviewer confirmed these are correct
    "CASE-DEMO-001": "APPROVED",
    "CASE-DEMO-002": "APPROVED",
    "CASE-DEMO-009": "APPROVED",
    "CASE-DEMO-011": "APPROVED",
    "CASE-DEMO-025": "APPROVED",

    # REJECTED (3) — reviewer marked as incorrect
    "CASE-DEMO-013": "REJECTED",
    "CASE-DEMO-019": "REJECTED",
    "CASE-DEMO-020": "REJECTED",

    # ESCALATED (4) — sent to senior reviewer
    "CASE-DEMO-006": "ESCALATED",
    "CASE-DEMO-015": "ESCALATED",
    "CASE-DEMO-028": "ESCALATED",
    "CASE-DEMO-030": "ESCALATED",

    # PENDING (18) — awaiting review (default for all others)
}


def make_timestamp(index: int) -> str:
    """Create realistic observation timestamps spread over the last 7 days."""
    base = datetime(2025, 9, 1, 10, 0, 0)
    offset = timedelta(hours=index * 3, minutes=index * 17 % 60)
    return (base + offset).isoformat()


def generate_payments(cases):
    """Generate payment records from cases."""
    payments = []
    for i, c in enumerate(cases):
        payments.append({
            "payment_id": c["payment_id"],
            "merchant_id": c["merchant_id"],
            "amount": c["expected_amount"],
            "currency": "INR",
            "status": "CAPTURED",
            "created_at": make_timestamp(i),
        })
    return payments


def generate_settlements(cases):
    """Generate settlement records matching expected amounts."""
    settlements = []
    for i, c in enumerate(cases):
        if c["scenario"] == "EXACT_MATCH":
            settlements.append({
                "settlement_id": f"SET-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": c["actual_amount"],
                "status": "SETTLED",
                "created_at": make_timestamp(i),
            })
        elif c["scenario"] == "PARTIAL_SETTLEMENT":
            settlements.append({
                "settlement_id": f"SET-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": c["actual_amount"],
                "status": "PARTIAL",
                "created_at": make_timestamp(i),
            })
        elif c["scenario"] == "DUPLICATE":
            # Two settlements for the same payment
            settlements.append({
                "settlement_id": f"SET-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": c["expected_amount"],
                "status": "SETTLED",
                "created_at": make_timestamp(i),
            })
            settlements.append({
                "settlement_id": f"SET-{c['payment_id'][4:]}-DUP",
                "payment_id": c["payment_id"],
                "amount": c["actual_amount"] - c["expected_amount"],
                "status": "SETTLED",
                "created_at": make_timestamp(i),
            })
    return settlements


def generate_refunds(cases):
    """Generate refund records where applicable."""
    refunds = []
    for i, c in enumerate(cases):
        if c["scenario"] == "REFUND_ADJUSTMENT" or (c["scenario"] == "UNKNOWN" and c["difference"] > 0):
            refunds.append({
                "refund_id": f"REF-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": abs(c["difference"]),
                "status": "REFUNDED",
                "created_at": make_timestamp(i),
            })
    return refunds


def generate_fees(cases):
    """Generate fee records for fee-difference scenarios."""
    fees = []
    for i, c in enumerate(cases):
        if c["scenario"] == "FEE_DIFFERENCE":
            fees.append({
                "fee_id": f"FEE-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": abs(c["difference"]),
                "type": "PROCESSING_FEE",
                "created_at": make_timestamp(i),
            })
    return fees


def generate_taxes(cases):
    """Generate tax records for tax-adjustment scenarios."""
    taxes = []
    for i, c in enumerate(cases):
        if c["scenario"] == "TAX_ADJUSTMENT":
            taxes.append({
                "tax_id": f"TAX-{c['payment_id'][4:]}",
                "payment_id": c["payment_id"],
                "amount": abs(c["difference"]),
                "type": "GST",
                "rate_percent": 18,
                "created_at": make_timestamp(i),
            })
    return taxes


def generate_merchants(cases):
    """Generate unique merchant records."""
    merchants = {}
    for c in cases:
        mid = c["merchant_id"]
        if mid not in merchants:
            merchants[mid] = {
                "merchant_id": mid,
                "name": f"Merchant {mid[-4:]}",
                "status": "ACTIVE",
            }
    return list(merchants.values())


def seed_demo_batch():
    """Create the demo batch directory with all required files."""
    batch_dir = DATA_DIR / DEMO_BATCH_ID
    generated_dir = batch_dir / "generated"
    ground_truth_dir = batch_dir / "ground_truth"

    # Don't re-seed if already exists
    if (generated_dir / "cases.json").exists():
        print(f"[SEED] Demo batch {DEMO_BATCH_ID} already exists, skipping.")
        return DEMO_BATCH_ID

    generated_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamps to cases
    cases = []
    for i, c in enumerate(DEMO_CASES):
        case = {**c, "observation_timestamp": make_timestamp(i)}
        cases.append(case)

    # Generate all financial records
    payments = generate_payments(cases)
    settlements = generate_settlements(cases)
    refunds = generate_refunds(cases)
    fees = generate_fees(cases)
    taxes = generate_taxes(cases)
    merchants = generate_merchants(cases)

    # Write all files
    files = {
        "cases.json": cases,
        "payments.json": payments,
        "settlements.json": settlements,
        "refunds.json": refunds,
        "fees.json": fees,
        "taxes.json": taxes,
        "adjustments.json": [],
        "merchants.json": merchants,
        "manifest.json": {
            "batch_id": DEMO_BATCH_ID,
            "generated_at": datetime.utcnow().isoformat(),
            "case_count": len(cases),
            "payment_count": len(payments),
            "settlement_count": len(settlements),
            "refund_count": len(refunds),
            "fee_count": len(fees),
            "tax_count": len(taxes),
        },
    }

    for filename, data in files.items():
        filepath = generated_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    # Write ground truth
    ground_truth = {
        "batch_id": DEMO_BATCH_ID,
        "cases": [
            {
                "case_id": c["case_id"],
                "payment_id": c["payment_id"],
                "expected_outcome": "RESOLVED" if c["resolvable"] else "ESCALATE",
                "ground_truth_status": STATUS_ASSIGNMENTS.get(c["case_id"], "PENDING"),
            }
            for c in cases
        ],
    }
    with open(ground_truth_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[SEED] Created demo batch {DEMO_BATCH_ID}:")
    print(f"  Cases: {len(cases)}")
    print(f"  Payments: {len(payments)}")
    print(f"  Settlements: {len(settlements)}")
    print(f"  Refunds: {len(refunds)}")
    print(f"  Fees: {len(fees)}")
    print(f"  Taxes: {len(taxes)}")
    print(f"  Merchants: {len(merchants)}")

    return DEMO_BATCH_ID


def seed_status_overrides():
    """
    Write status overrides that the ExceptionService can use
    to apply the curated statuses to demo exceptions.
    """
    overrides_path = DATA_DIR / DEMO_BATCH_ID / "status_overrides.json"
    with open(overrides_path, "w") as f:
        json.dump(STATUS_ASSIGNMENTS, f, indent=2)
    print(f"[SEED] Status overrides written: {len(STATUS_ASSIGNMENTS)} assignments")


if __name__ == "__main__":
    batch_id = seed_demo_batch()
    seed_status_overrides()
    print(f"\n[SEED] Demo batch ready: {batch_id}")
    print("[SEED] Run 'POST /batches' with source=synthetic to register it,")
    print("  or start the backend which will pick it up via /exceptions.")
