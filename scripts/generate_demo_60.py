#!/usr/bin/env python3
"""
Generate a realistic 60-record demo batch for hackathon demonstration.
Creates diverse financial reconciliation scenarios with realistic merchants.
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Determine base directory (script's parent directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "data", "DEMO-60", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Realistic merchant names (Indian e-commerce/fintech companies)
MERCHANTS = {
    "MER-3001": "Flipkart",
    "MER-3002": "Amazon India",
    "MER-3003": "Myntra",
    "MER-3004": "Swiggy",
    "MER-3005": "Zomato",
    "MER-3006": "OYO Rooms",
    "MER-3007": "MakeMyTrip",
    "MER-3008": "Paytm",
    "MER-3009": "PhonePe",
    "MER-3010": "Razorpay",
    "MER-3011": "Cred",
    "MER-3012": "PolicyBazaar",
    "MER-3013": "Zerodha",
    "MER-3014": "MagicBricks",
    "MER-3015": "Urban Ladder",
}

# Generate 60 payment records with various scenarios
payments = []
settlements = []
fees = []
refunds = []
taxes = []
cases = []

base_time = datetime(2025, 10, 1, 8, 0, 0)

# Track which payment IDs are used for each scenario type
scenario_assignments = {
    "EXACT_MATCH": [],
    "PARTIAL_SETTLEMENT": [],
    "TIMING_DIFFERENCE": [],
    "FEE_DIFFERENCE": [],
    "TAX_ADJUSTMENT": [],
    "REFUND_DIFFERENCE": [],
    "DUPLICATE": [],
    "MISSING_RECORD": [],
    "COMPLEX_MULTI_ADJUSTMENT": [],
}

# Generate 60 payments with realistic amounts and scenarios
for i in range(1, 61):
    merchant_id = f"MER-{3001 + (i % 15)}"
    payment_id = f"PAY-DEMO60-{i:03d}"
    case_id = f"CASE-DEMO60-{i:03d}"
    
    # Base amount varies by merchant (realistic transaction sizes)
    base_amount = 100000 + (i * 75000) + (hash(i) % 500000)
    
    # Determine scenario based on position (ensuring good distribution)
    scenario_idx = i % 10
    if scenario_idx == 0:
        scenario = "EXACT_MATCH"
        actual_amount = base_amount
        difference = 0
        risk = "LOW"
        resolvable = True
    elif scenario_idx == 1:
        scenario = "EXACT_MATCH"
        actual_amount = base_amount
        difference = 0
        risk = "LOW"
        resolvable = True
    elif scenario_idx == 2:
        scenario = "PARTIAL_SETTLEMENT"
        actual_amount = int(base_amount * 0.85)  # 85% settled
        difference = actual_amount - base_amount
        risk = "MEDIUM"
        resolvable = True
    elif scenario_idx == 3:
        scenario = "PARTIAL_SETTLEMENT"
        actual_amount = int(base_amount * 0.72)  # 72% settled
        difference = actual_amount - base_amount
        risk = "MEDIUM"
        resolvable = True
    elif scenario_idx == 4:
        scenario = "TIMING_DIFFERENCE"
        actual_amount = 0  # Not yet settled
        difference = -base_amount
        risk = "MEDIUM"
        resolvable = True
    elif scenario_idx == 5:
        scenario = "FEE_DIFFERENCE"
        fee_amount = int(base_amount * 0.025)  # 2.5% fee
        actual_amount = base_amount - fee_amount
        difference = -fee_amount
        risk = "LOW"
        resolvable = True
        fees.append({
            "fee_id": f"FEE-DEMO60-{i:03d}",
            "payment_id": payment_id,
            "amount": fee_amount,
            "fee_type": "TRANSACTION"
        })
    elif scenario_idx == 6:
        scenario = "TAX_ADJUSTMENT"
        tax_amount = int(base_amount * 0.18)  # 18% GST
        actual_amount = base_amount - tax_amount
        difference = -tax_amount
        risk = "LOW"
        resolvable = True
        taxes.append({
            "tax_id": f"TAX-DEMO60-{i:03d}",
            "payment_id": payment_id,
            "amount": tax_amount,
            "tax_type": "GST"
        })
    elif scenario_idx == 7:
        scenario = "REFUND_DIFFERENCE"
        refund_amount = int(base_amount * 0.35)  # 35% refunded
        actual_amount = base_amount - refund_amount
        difference = -refund_amount
        risk = "MEDIUM"
        resolvable = True
        refunds.append({
            "refund_id": f"REF-DEMO60-{i:03d}",
            "payment_id": payment_id,
            "amount": refund_amount,
            "status": "PROCESSED",
            "refund_timestamp": (base_time + timedelta(hours=i*3)).isoformat()
        })
    elif scenario_idx == 8:
        scenario = "DUPLICATE"
        actual_amount = base_amount * 2  # Duplicate payment recorded
        difference = base_amount
        risk = "HIGH"
        resolvable = False
    else:  # scenario_idx == 9
        scenario = "COMPLEX_MULTI_ADJUSTMENT"
        # Multiple adjustments: fee + tax + partial settlement
        fee_amt = int(base_amount * 0.02)
        tax_amt = int((base_amount - fee_amt) * 0.18)
        settlement_pct = 0.90
        actual_amount = int((base_amount - fee_amt - tax_amt) * settlement_pct)
        difference = actual_amount - base_amount
        risk = "HIGH"
        resolvable = True
    
    # Randomly escalate some to HIGH risk for variety
    if i in [5, 15, 25, 35, 45]:
        risk = "HIGH"
    elif i in [10, 20, 30, 40, 50]:
        risk = "MEDIUM"
    
    # Assign statuses: majority pending, some approved/rejected/escalated
    if i <= 10:
        status = "APPROVED"
    elif i <= 15:
        status = "REJECTED"
    elif i <= 20:
        status = "ESCALATED"
    else:
        status = "PENDING"
    
    payments.append({
        "payment_id": payment_id,
        "merchant_id": merchant_id,
        "amount": base_amount,
        "currency": "INR",
        "status": "CAPTURED",
        "payment_timestamp": (base_time + timedelta(hours=i*3)).isoformat()
    })
    
    # Settlements: most payments get settlements, some don't (timing differences)
    if scenario != "TIMING_DIFFERENCE" and scenario != "MISSING_RECORD":
        settlement_amount = actual_amount if actual_amount > 0 else base_amount
        settlements.append({
            "settlement_id": f"SET-DEMO60-{i:03d}",
            "payment_id": payment_id,
            "merchant_id": merchant_id,
            "amount": settlement_amount,
            "status": "SETTLED" if scenario == "EXACT_MATCH" else "PENDING",
            "settlement_timestamp": (base_time + timedelta(hours=i*3 + 1)).isoformat()
        })
    
    # Store case info
    cases.append({
        "case_id": case_id,
        "payment_id": payment_id,
        "merchant_id": merchant_id,
        "expected_amount": base_amount,
        "actual_amount": actual_amount,
        "difference": difference,
        "scenario": scenario,
        "risk_category": risk,
        "resolvable": resolvable,
        "status": status,
        "observation_timestamp": (base_time + timedelta(hours=i*3)).isoformat()
    })
    
    scenario_assignments[scenario].append(payment_id)

# Generate merchants list
merchants = [
    {"merchant_id": mid, "name": name, "status": "ACTIVE"}
    for mid, name in MERCHANTS.items()
]

# Create manifest
manifest = {
    "batch_id": "DEMO-60",
    "generated_at": datetime.utcnow().isoformat(),
    "case_count": len(cases),
    "payment_count": len(payments),
    "settlement_count": len(settlements),
    "refund_count": len(refunds),
    "fee_count": len(fees),
    "tax_count": len(taxes),
    "scenario_distribution": {k: len(v) for k, v in scenario_assignments.items()}
}

# Save all files
files = {
    "payments.json": payments,
    "settlements.json": settlements,
    "fees.json": fees,
    "refunds.json": refunds,
    "taxes.json": taxes,
    "cases.json": cases,
    "merchants.json": merchants,
    "manifest.json": manifest
}

for filename, data in files.items():
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

print(f"Generated DEMO-60 batch with:")
print(f"  - {len(payments)} payments")
print(f"  - {len(settlements)} settlements")
print(f"  - {len(fees)} fees")
print(f"  - {len(refunds)} refunds")
print(f"  - {len(taxes)} taxes")
print(f"  - {len(cases)} cases")
print(f"\nScenario distribution:")
for scenario, pids in sorted(scenario_assignments.items()):
    print(f"  - {scenario}: {len(pids)} records")
print(f"\nStatus distribution in cases:")
statuses = {}
for case in cases:
    statuses[case["status"]] = statuses.get(case["status"], 0) + 1
for status, count in sorted(statuses.items()):
    print(f"  - {status}: {count} records")
print(f"\nRisk distribution:")
risks = {}
for case in cases:
    risks[case["risk_category"]] = risks.get(case["risk_category"], 0) + 1
for risk, count in sorted(risks.items()):
    print(f"  - {risk}: {count} records")
print(f"\nFiles written to {OUTPUT_DIR}/")
