"""
Payment generator for synthetic financial data.

Generates deterministic payment records linked to merchants.
"""

from datetime import datetime
from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import PaymentStatus
from app.schemas.financial import Merchant, Payment


def generate_payments(
    config: GeneratorConfig,
    merchants: List[Merchant],
    rng: DeterministicRNG,
) -> List[Payment]:
    """
    Generate deterministic payment records.

    Each payment is linked to a merchant and assigned a deterministic amount
    within the configured range.

    Args:
        config: Generator configuration
        merchants: List of merchants to generate payments for
        rng: Deterministic random number generator

    Returns:
        List of Payment objects with unique IDs and valid merchant references
    """
    payments = []

    for i in range(config.num_cases):
        payment_id = f"PAY-{i + 1:06d}"

        # Assign payment to a merchant (round-robin distribution)
        merchant = merchants[i % len(merchants)]

        # Generate deterministic payment amount in paise
        amount = rng.random_amount(
            config.min_payment_amount_paise,
            config.max_payment_amount_paise,
        )

        # Payment timestamp within the date range
        payment_timestamp = rng.random_timestamp(
            config.date_range_start,
            config.date_range_end,
        )

        # Status is typically CAPTURED for normal cases
        status = PaymentStatus.CAPTURED

        payment = Payment(
            payment_id=payment_id,
            merchant_id=merchant.merchant_id,
            amount=amount,
            currency=config.currency,
            status=status,
            payment_timestamp=payment_timestamp,
        )
        payments.append(payment)

    return payments
