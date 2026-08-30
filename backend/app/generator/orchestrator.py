"""
Dataset orchestrator for synthetic financial data generation.

Coordinates all generators and produces a complete, validated dataset.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.generator.adjustments import generate_adjustments
from app.generator.cases import generate_cases
from app.generator.fees import generate_fees
from app.generator.merchants import generate_merchants
from app.generator.payments import generate_payments
from app.generator.refunds import generate_refunds
from app.generator.rng import DeterministicRNG
from app.generator.settlements import generate_settlements
from app.generator.taxes import generate_taxes
from app.generator.validation import validate_dataset
from app.schemas.config import GeneratorConfig


class DatasetGenerator:
    """
    Orchestrates the complete synthetic dataset generation pipeline.

    Coordinates merchant, payment, settlement, refund, fee, tax, adjustment,
    and case generators to produce a consistent, validated dataset.
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = DeterministicRNG(config.random_seed)

    def generate(self) -> Dict[str, Any]:
        """
        Generate the complete synthetic dataset.

        Returns:
            Dictionary containing all generated records and manifest
        """
        # Generate financial entities in dependency order
        merchants = generate_merchants(self.config, self.rng)
        payments = generate_payments(self.config, merchants, self.rng)
        settlements = generate_settlements(self.config, payments, merchants, self.rng)
        refunds = generate_refunds(self.config, payments, self.rng)
        fees = generate_fees(self.config, payments, self.rng)
        taxes = generate_taxes(self.config, payments, self.rng)
        adjustments = generate_adjustments(self.config, payments, self.rng)

        # Generate cases with scenario injection
        cases, ground_truth_records = generate_cases(
            self.config,
            payments,
            refunds,
            fees,
            taxes,
            adjustments,
            settlements,
            self.rng,
        )

        # Validate the dataset
        validation_results = validate_dataset(
            merchants=merchants,
            payments=payments,
            settlements=settlements,
            refunds=refunds,
            fees=fees,
            taxes=taxes,
            adjustments=adjustments,
            cases=cases,
            ground_truth_records=ground_truth_records,
        )

        # Generate manifest
        manifest = self._create_manifest(
            merchants=merchants,
            payments=payments,
            settlements=settlements,
            refunds=refunds,
            fees=fees,
            taxes=taxes,
            adjustments=adjustments,
            cases=cases,
            ground_truth_records=ground_truth_records,
            validation_results=validation_results,
        )

        return {
            "merchants": merchants,
            "payments": payments,
            "settlements": settlements,
            "refunds": refunds,
            "fees": fees,
            "taxes": taxes,
            "adjustments": adjustments,
            "cases": cases,
            "ground_truth": ground_truth_records,
            "manifest": manifest,
        }

    def _create_manifest(
        self,
        merchants: List,
        payments: List,
        settlements: List,
        refunds: List,
        fees: List,
        taxes: List,
        adjustments: List,
        cases: List,
        ground_truth_records: List,
        validation_results: Dict,
    ) -> Dict:
        """Create a dataset manifest with generation metadata."""
        # Compute scenario distribution
        scenario_counts = {}
        for case in cases:
            s = case.scenario.value
            scenario_counts[s] = scenario_counts.get(s, 0) + 1

        # Compute risk distribution
        risk_counts = {}
        for case in cases:
            r = case.risk_category.value
            risk_counts[r] = risk_counts.get(r, 0) + 1

        # Compute resolvable counts
        resolvable_count = sum(1 for case in cases if case.resolvable)
        unresolved_count = len(cases) - resolvable_count

        return {
            "seed": self.config.random_seed,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "num_merchants": self.config.num_merchants,
                "num_cases": self.config.num_cases,
                "currency": self.config.currency.value,
                "date_range_start": self.config.date_range_start.isoformat(),
                "date_range_end": self.config.date_range_end.isoformat(),
                "fee_rate_bps": self.config.fee_rate_bps,
                "tax_rate_bps": self.config.tax_rate_bps,
            },
            "counts": {
                "merchants": len(merchants),
                "payments": len(payments),
                "settlements": len(settlements),
                "refunds": len(refunds),
                "fees": len(fees),
                "taxes": len(taxes),
                "adjustments": len(adjustments),
                "cases": len(cases),
            },
            "scenario_distribution": scenario_counts,
            "risk_distribution": risk_counts,
            "resolvable_count": resolvable_count,
            "unresolved_count": unresolved_count,
            "validation": validation_results,
        }


def save_dataset(dataset: Dict[str, Any], output_dir: str) -> None:
    """
    Save the generated dataset to disk.

    Outputs:
        data/generated/merchants.json
        data/generated/payments.json
        data/generated/settlements.json
        data/generated/refunds.json
        data/generated/fees.json
        data/generated/taxes.json
        data/generated/adjustments.json
        data/generated/cases.json
        data/ground_truth/ground_truth.json
        data/generated/manifest.json
    """
    output_path = Path(output_dir)
    generated_path = output_path / "generated"
    ground_truth_path = output_path / "ground_truth"

    # Create directories if they don't exist
    generated_path.mkdir(parents=True, exist_ok=True)
    ground_truth_path.mkdir(parents=True, exist_ok=True)

    # Save financial records to generated/
    entity_files = {
        "merchants": "merchants.json",
        "payments": "payments.json",
        "settlements": "settlements.json",
        "refunds": "refunds.json",
        "fees": "fees.json",
        "taxes": "taxes.json",
        "adjustments": "adjustments.json",
        "cases": "cases.json",
    }

    for entity_key, filename in entity_files.items():
        filepath = generated_path / filename
        records = dataset[entity_key]
        # Serialize Pydantic models to dicts
        if records and hasattr(records[0], "model_dump"):
            data = [r.model_dump(mode="json") for r in records]
        else:
            data = records
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # Save ground truth separately using Pydantic model serialization
    gt_path = ground_truth_path / "ground_truth.json"
    gt_data = [gt.model_dump(mode="json") for gt in dataset["ground_truth"]]
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2, default=str)

    # Save manifest
    manifest_path = generated_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dataset["manifest"], f, indent=2, default=str)
