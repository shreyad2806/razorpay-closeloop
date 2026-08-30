#!/usr/bin/env python3
"""
CLI entrypoint for synthetic financial data generation.

Usage:
    python scripts/generate_data.py [OPTIONS]

Examples:
    # Generate small dataset (100 cases) with default seed
    python scripts/generate_data.py --cases 100

    # Generate with specific seed
    python scripts/generate_data.py --cases 100 --seed 123

    # Generate to specific output directory
    python scripts/generate_data.py --cases 100 --output ./data
"""

import argparse
import json
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.generator.orchestrator import DatasetGenerator, save_dataset
from app.schemas.config import GeneratorConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial data for Razorpay CloseLoop"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=100,
        help="Number of reconciliation cases to generate (default: 100)",
    )
    parser.add_argument(
        "--merchants",
        type=int,
        default=10,
        help="Number of merchants to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for generated data (default: data)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed generation info",
    )
    return parser.parse_args()


def main():
    """Main entry point for data generation."""
    args = parse_args()

    print(f"Generating synthetic financial data...")
    print(f"  Seed: {args.seed}")
    print(f"  Cases: {args.cases}")
    print(f"  Merchants: {args.merchants}")
    print(f"  Output: {args.output}")
    print()

    # Create configuration
    config = GeneratorConfig(
        random_seed=args.seed,
        num_cases=args.cases,
        num_merchants=args.merchants,
    )

    # Generate dataset
    generator = DatasetGenerator(config)
    dataset = generator.generate()

    # Save to disk
    save_dataset(dataset, args.output)

    # Print summary
    manifest = dataset["manifest"]
    print("Generation complete!")
    print()
    print("Dataset Summary:")
    print(f"  Merchants:     {manifest['counts']['merchants']}")
    print(f"  Payments:      {manifest['counts']['payments']}")
    print(f"  Settlements:   {manifest['counts']['settlements']}")
    print(f"  Refunds:       {manifest['counts']['refunds']}")
    print(f"  Fees:          {manifest['counts']['fees']}")
    print(f"  Taxes:         {manifest['counts']['taxes']}")
    print(f"  Adjustments:   {manifest['counts']['adjustments']}")
    print(f"  Cases:         {manifest['counts']['cases']}")
    print()

    # Print validation results
    validation = manifest["validation"]
    if validation["valid"]:
        print("[OK] Validation passed")
    else:
        print(f"[FAIL] Validation failed with {validation['error_count']} errors")
        for error in validation["errors"]:
            print(f"  - {error}")

    if validation["warning_count"] > 0:
        print(f"  Warnings: {validation['warning_count']}")

    print()
    print(f"Data saved to: {args.output}/")

    if args.verbose:
        print()
        print("Scenario Distribution:")
        # Count scenarios in cases
        scenario_counts = {}
        for case in dataset["cases"]:
            s = case.scenario.value
            scenario_counts[s] = scenario_counts.get(s, 0) + 1
        for scenario, count in sorted(scenario_counts.items()):
            print(f"  {scenario}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
