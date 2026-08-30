#!/usr/bin/env python3
"""
Generate 3 batches of large-scale synthetic financial data.

This script generates the production-scale dataset with:
- Batch 1: 5000 cases (training)
- Batch 2: 3000 cases (validation)
- Batch 3: 2500 cases (test)

Total: 10,500 cases with all financial records.
"""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.generator.batch import BatchGenerator
from app.schemas.config import GeneratorConfig


def main():
    """Generate 3 batches of large-scale data."""
    print("=" * 60)
    print("RAZORPAY CLOSELOOP - Large Dataset Generation")
    print("=" * 60)
    print()

    start_time = time.time()

    # Create batch generator
    generator = BatchGenerator(
        base_seed=42,
        output_dir="data",
    )

    # Define batch configurations
    batch_configs = [
        # Batch 1: Training data (largest)
        {
            "batch_id": "batch_001",
            "num_merchants": 50,
            "num_cases": 5000,
            "seed_offset": 0,
        },
        # Batch 2: Validation data
        {
            "batch_id": "batch_002",
            "num_merchants": 30,
            "num_cases": 3000,
            "seed_offset": 1000,
        },
        # Batch 3: Test data
        {
            "batch_id": "batch_003",
            "num_merchants": 30,
            "num_cases": 2500,
            "seed_offset": 2000,
        },
    ]

    # Generate batches
    results = generator.generate_multiple_batches(batch_configs)

    # Get summary
    summary = generator.get_summary()

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Total Batches:     {summary['total_batches']}")
    print(f"  Total Cases:       {summary['total_cases']:,}")
    print(f"  Total Events:      {summary['total_events']:,}")
    print(f"  Total Time:        {summary['total_generation_time_seconds']:.2f}s")
    print(f"  Events/Second:     {summary['average_events_per_second']:,.0f}")
    print()

    print("Batch Details:")
    for batch in results:
        print(f"  {batch['batch_id']}:")
        print(f"    Cases:           {batch['num_cases']:,}")
        print(f"    Events:          {batch['total_events']:,}")
        print(f"    Time:            {batch['generation_time_seconds']:.2f}s")
        print(f"    Validation:      {'PASS' if batch['validation']['valid'] else 'FAIL'}")
        print()

    print("Files Created:")
    for batch in results:
        print(f"  data/{batch['batch_id']}/")
        print(f"    generated/")
        print(f"      merchants.json")
        print(f"      payments.json")
        print(f"      settlements.json")
        print(f"      refunds.json")
        print(f"      fees.json")
        print(f"      taxes.json")
        print(f"      adjustments.json")
        print(f"      cases.json")
        print(f"      manifest.json")
        print(f"    ground_truth/")
        print(f"      ground_truth.json")
    print()

    print("=" * 60)
    print("STOP - Do not begin reconciliation")
    print("=" * 60)


if __name__ == "__main__":
    main()
