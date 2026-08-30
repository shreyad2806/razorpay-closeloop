"""
Generator configuration contract for deterministic synthetic dataset generation.

The same seed and configuration must always produce the same dataset.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.enums import Currency, ExceptionType


class ScenarioDistribution(BaseModel):
    """
    Distribution weights for exception scenarios.

    Weights are normalized internally; they do not need to sum to 1.0.
    Higher weight = more cases of that type generated.
    """

    weights: dict[ExceptionType, float] = Field(
        default_factory=lambda: {
            ExceptionType.EXACT_MATCH: 30.0,
            ExceptionType.FEE_DIFFERENCE: 15.0,
            ExceptionType.REFUND_ADJUSTMENT: 10.0,
            ExceptionType.TAX_ADJUSTMENT: 10.0,
            ExceptionType.TIMING_DIFFERENCE: 10.0,
            ExceptionType.PARTIAL_SETTLEMENT: 8.0,
            ExceptionType.DUPLICATE: 5.0,
            ExceptionType.MISSING_RECORD: 5.0,
            ExceptionType.COMPLEX_MULTI_ADJUSTMENT: 4.0,
            ExceptionType.UNKNOWN: 3.0,
        },
        description="Relative weights for each exception scenario type",
    )


class GeneratorConfig(BaseModel):
    """
    Configuration for deterministic synthetic dataset generation.

    The same (seed + config) pair always produces identical output.
    """

    random_seed: int = Field(
        default=42, description="Random seed for reproducible generation"
    )
    num_merchants: int = Field(
        default=10, description="Number of unique merchants to generate"
    )
    num_cases: int = Field(
        default=1000, description="Number of reconciliation cases to generate"
    )
    scenario_distribution: ScenarioDistribution = Field(
        default_factory=ScenarioDistribution,
        description="Distribution weights for exception scenarios",
    )
    date_range_start: datetime = Field(
        default_factory=lambda: datetime(2025, 1, 1),
        description="Start of the date range for generated events",
    )
    date_range_end: datetime = Field(
        default_factory=lambda: datetime(2025, 6, 30),
        description="End of the date range for generated events",
    )
    currency: Currency = Field(
        default=Currency.INR, description="Currency for all generated amounts"
    )
    min_payment_amount_paise: int = Field(
        default=10000,  # ₹100
        description="Minimum payment amount in paise",
    )
    max_payment_amount_paise: int = Field(
        default=10000000,  # ₹1,00,000
        description="Maximum payment amount in paise",
    )
    fee_rate_bps: int = Field(
        default=200,  # 2%
        description="Fee rate in basis points (200 = 2.00%)",
    )
    tax_rate_bps: int = Field(
        default=1800,  # 18%
        description="Tax rate in basis points (1800 = 18.00%)",
    )
    duplicate_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of generating a duplicate settlement for a payment",
    )
    missing_record_probability: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Probability of omitting an expected record",
    )
