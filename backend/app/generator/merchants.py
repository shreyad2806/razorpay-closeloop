"""
Merchant generator for synthetic financial data.

Generates deterministic merchant records with realistic names and metadata.
"""

from datetime import datetime
from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import Currency
from app.schemas.financial import Merchant

# Realistic merchant name templates
_MERCHANT_PREFIXES = [
    "Tech", "Fresh", "Quick", "Smart", "Global", "Prime", "Elite", "Star",
    "Blue", "Green", "Red", "Silver", "Gold", "Crystal", "Diamond", "Royal",
]

_MERCHANT_SUFFIXES = [
    "Solutions", "Enterprises", "Trading", "Store", "Mart", "Shop",
    "Hub", "Market", "Goods", "Services", "Co", "Labs", "Digital",
]


def generate_merchants(config: GeneratorConfig, rng: DeterministicRNG) -> List[Merchant]:
    """
    Generate a list of deterministic merchant records.

    Args:
        config: Generator configuration
        rng: Deterministic random number generator

    Returns:
        List of Merchant objects with unique IDs and realistic names
    """
    merchants = []

    for i in range(config.num_merchants):
        merchant_id = f"MER-{i + 1:04d}"

        # Generate deterministic merchant name
        prefix = rng.choice(_MERCHANT_PREFIXES)
        suffix = rng.choice(_MERCHANT_SUFFIXES)
        name = f"{prefix} {suffix}"

        # Generate reference code (e.g., "MER-TECH-001")
        reference = f"MER-{prefix[:3].upper()}-{i + 1:03d}"

        # Merchant created within the date range
        created_at = rng.random_timestamp(
            config.date_range_start,
            config.date_range_end,
        )

        merchant = Merchant(
            merchant_id=merchant_id,
            name=name,
            reference=reference,
            currency=config.currency,
            created_at=created_at,
        )
        merchants.append(merchant)

    return merchants
