#!/usr/bin/env python3
"""
Process DEMO-60 batch through reconciliation and register in batch registry.
This script loads the batch data, runs reconciliation, and updates the in-memory registry.
"""

import json
import os
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, 'backend')

from app.api.services.batch_service import BatchService, _batch_registry
from app.reconciliation.engine import reconcile_batch
from app.schemas.financial import Payment, Settlement, Refund, Fee, Tax, Adjustment
from app.schemas.enums import MatchStatus


def load_json(filepath):
    """Load JSON file."""
    if not os.path.isfile(filepath):
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def dicts_to_payments(dicts):
    return [Payment(**d) for d in dicts]

def dicts_to_settlements(dicts):
    return [Settlement(**d) for d in dicts]

def dicts_to_refunds(dicts):
    return [Refund(**d) for d in dicts]

def dicts_to_fees(dicts):
    return [Fee(**d) for d in dicts]

def dicts_to_taxes(dicts):
    return [Tax(**d) for d in dicts]

def dicts_to_adjustments(dicts):
    return [Adjustment(**d) for d in dicts]


def main():
    data_dir = "backend/data"
    batch_id = "DEMO-60"
    generated_dir = os.path.join(data_dir, batch_id, "generated")
    
    print(f"Loading batch data from {generated_dir}...")
    
    # Load data files
    payments = dicts_to_payments(load_json(os.path.join(generated_dir, "payments.json")))
    settlements = dicts_to_settlements(load_json(os.path.join(generated_dir, "settlements.json")))
    refunds = dicts_to_refunds(load_json(os.path.join(generated_dir, "refunds.json")))
    fees = dicts_to_fees(load_json(os.path.join(generated_dir, "fees.json")))
    taxes = dicts_to_taxes(load_json(os.path.join(generated_dir, "taxes.json")))
    adjustments = dicts_to_adjustments(load_json(os.path.join(generated_dir, "adjustments.json")))
    
    # Load cases for case mapping
    cases = load_json(os.path.join(generated_dir, "cases.json"))
    case_mapping = {c.get("payment_id", ""): c.get("case_id", "") for c in cases}
    
    print(f"Loaded: {len(payments)} payments, {len(settlements)} settlements")
    print(f"        {len(refunds)} refunds, {len(fees)} fees, {len(taxes)} taxes")
    print(f"        {len(cases)} cases")
    
    # Run reconciliation
    print("\nRunning reconciliation...")
    results = reconcile_batch(
        payments=payments,
        settlements=settlements,
        refunds=refunds,
        fees=fees,
        taxes=taxes,
        adjustments=adjustments,
        case_mapping=case_mapping,
    )
    
    # Compute summary statistics
    total = len(results)
    matched = sum(1 for r in results if r.match_status == MatchStatus.MATCHED)
    exceptions = total - matched
    match_rate = matched / total if total > 0 else 0.0
    exception_rate = exceptions / total if total > 0 else 0.0
    
    print(f"\nReconciliation results:")
    print(f"  Total records: {total}")
    print(f"  Matched: {matched}")
    print(f"  Exceptions: {exceptions}")
    print(f"  Match rate: {match_rate:.2%}")
    print(f"  Exception rate: {exception_rate:.2%}")
    
    # Save reconciliation results
    results_path = os.path.join(generated_dir, "reconciliation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            [r.model_dump() if hasattr(r, "model_dump") else r.__dict__ for r in results],
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved reconciliation results to {results_path}")
    
    # Register batch in in-memory registry
    batch_meta = {
        "batch_id": batch_id,
        "name": "Demo Batch 60 - Hackathon",
        "description": "Realistic 60-record financial reconciliation demo with diverse scenarios",
        "source": "prebuilt",
        "status": "COMPLETED",
        "created_at": datetime.utcnow().isoformat(),
        "exception_count": exceptions,
        "total_records": total,
        "matched_records": matched,
        "success_count": total,
        "failure_count": 0,
        "processing_time_ms": 100,  # Simulated
        "match_rate": round(match_rate, 4),
        "exception_rate": round(exception_rate, 4),
        "auto_resolved": 0,
        "human_review": 0,
        "unresolved": exceptions,
        "verification_passed": 0,
        "verification_failed": 0,
        "financial_impact_paise": 0,
        "auto_decisions": 0,
        "human_decisions": 0,
        "unresolved_decisions": exceptions,
        "guardrail_blocks": 0,
        "high_value_blocks": sum(1 for r in results if getattr(r, 'risk_category', None) == "HIGH"),
        "conflict_blocks": 0,
        "novelty_blocks": 0,
        "verification_failures": 0,
    }
    
    _batch_registry[batch_id] = batch_meta
    print(f"\nRegistered batch {batch_id} in memory registry")
    
    # Print summary
    print("\n" + "="*60)
    print("DEMO-60 BATCH SUMMARY")
    print("="*60)
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch_meta['status']}")
    print(f"Total Records: {total}")
    print(f"Matched Records: {matched}")
    print(f"Exceptions: {exceptions}")
    print(f"Match Rate: {match_rate:.2%}")
    print(f"Exception Rate: {exception_rate:.2%}")
    print(f"High Risk Records: {batch_meta['high_value_blocks']}")
    print("\nException Scenarios:")
    scenario_counts = {}
    for r in results:
        if r.match_status != MatchStatus.MATCHED:
            scenario = r.exception_type or "UNKNOWN"
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  - {scenario}: {count}")
    
    return batch_meta


if __name__ == "__main__":
    main()
