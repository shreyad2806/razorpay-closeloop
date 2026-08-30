"""
Batch generation system for large-scale synthetic financial datasets.

Supports generating multiple independent batches with:
- Independent case IDs
- Deterministic seeds
- Configurable distributions
- Performance measurement
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.generator.orchestrator import DatasetGenerator, save_dataset
from app.schemas.config import GeneratorConfig


class BatchConfig:
    """Configuration for a single batch."""

    def __init__(
        self,
        batch_id: str,
        seed: int,
        num_merchants: int,
        num_cases: int,
        base_seed: int = 42,
    ):
        self.batch_id = batch_id
        self.seed = seed
        self.num_merchants = num_merchants
        self.num_cases = num_cases
        self.base_seed = base_seed

    def to_generator_config(self) -> GeneratorConfig:
        """Convert to GeneratorConfig."""
        return GeneratorConfig(
            random_seed=self.seed,
            num_merchants=self.num_merchants,
            num_cases=self.num_cases,
        )


class BatchGenerator:
    """
    Generates multiple independent batches of financial data.

    Each batch has:
    - Independent case IDs
    - Deterministic seed
    - Complete financial records and ground truth
    - Manifest with statistics
    """

    def __init__(
        self,
        base_seed: int = 42,
        output_dir: str = "data",
    ):
        self.base_seed = base_seed
        self.output_dir = Path(output_dir)
        self.results: List[Dict[str, Any]] = []

    def generate_batch(
        self,
        batch_id: str,
        num_merchants: int,
        num_cases: int,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a single batch.

        Args:
            batch_id: Unique batch identifier (e.g., "batch_001")
            num_merchants: Number of merchants for this batch
            num_cases: Number of cases for this batch
            seed_offset: Offset from base seed for this batch

        Returns:
            Dictionary with batch results and timing
        """
        seed = self.base_seed + seed_offset
        start_time = time.time()

        # Create batch config
        batch_config = BatchConfig(
            batch_id=batch_id,
            seed=seed,
            num_merchants=num_merchants,
            num_cases=num_cases,
            base_seed=self.base_seed,
        )

        # Generate dataset
        generator_config = batch_config.to_generator_config()
        generator = DatasetGenerator(generator_config)
        dataset = generator.generate()

        # Calculate timing
        generation_time = time.time() - start_time
        total_events = sum(dataset["manifest"]["counts"].values())

        # Add batch metadata to manifest
        dataset["manifest"]["batch_id"] = batch_id
        dataset["manifest"]["batch_seed"] = seed
        dataset["manifest"]["generation_time_seconds"] = round(generation_time, 3)
        dataset["manifest"]["events_per_second"] = round(
            total_events / generation_time if generation_time > 0 else 0, 2
        )

        # Save batch to separate directory
        batch_dir = self.output_dir / batch_id
        save_dataset(dataset, str(batch_dir))

        # Store results
        result = {
            "batch_id": batch_id,
            "seed": seed,
            "num_merchants": num_merchants,
            "num_cases": num_cases,
            "total_events": total_events,
            "generation_time_seconds": round(generation_time, 3),
            "events_per_second": round(
                total_events / generation_time if generation_time > 0 else 0, 2
            ),
            "counts": dataset["manifest"]["counts"],
            "scenario_distribution": dataset["manifest"]["scenario_distribution"],
            "risk_distribution": dataset["manifest"]["risk_distribution"],
            "resolvable_count": dataset["manifest"]["resolvable_count"],
            "unresolved_count": dataset["manifest"]["unresolved_count"],
            "validation": dataset["manifest"]["validation"],
        }
        self.results.append(result)

        return result

    def generate_multiple_batches(
        self,
        batch_configs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple batches.

        Args:
            batch_configs: List of batch configuration dicts with keys:
                - batch_id, num_merchants, num_cases, seed_offset

        Returns:
            List of batch results
        """
        results = []
        for config in batch_configs:
            result = self.generate_batch(
                batch_id=config["batch_id"],
                num_merchants=config["num_merchants"],
                num_cases=config["num_cases"],
                seed_offset=config.get("seed_offset", 0),
            )
            results.append(result)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all generated batches."""
        total_events = sum(r["total_events"] for r in self.results)
        total_cases = sum(r["num_cases"] for r in self.results)
        total_time = sum(r["generation_time_seconds"] for r in self.results)

        return {
            "total_batches": len(self.results),
            "total_events": total_events,
            "total_cases": total_cases,
            "total_generation_time_seconds": round(total_time, 3),
            "average_events_per_second": round(
                total_events / total_time if total_time > 0 else 0, 2
            ),
            "batches": self.results,
        }


def generate_large_dataset(
    base_seed: int = 42,
    output_dir: str = "data",
) -> Dict[str, Any]:
    """
    Generate a large dataset with multiple batches.

    Returns summary of all generated batches.
    """
    generator = BatchGenerator(base_seed=base_seed, output_dir=output_dir)

    # Define batch configurations
    batch_configs = [
        # Batch 1: Training data (larger)
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

    results = generator.generate_multiple_batches(batch_configs)
    return generator.get_summary()
