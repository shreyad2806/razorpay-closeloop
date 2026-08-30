"""
Fee generator for synthetic financial data.

Generates deterministic fee records linked to payments.
"""

from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import FeeType
from app.schemas.financial import Fee, Payment


def generate_fees(
    config: GeneratorConfig,
    payments: List[Payment],
    rng: DeterministicRNG,
) -> List[Fee]:
    """
    Generate deterministic fee records.

    Every payment has at least one TRANSACTION fee (2% by default).
    Some payments may also have PLATFORM fees or GST_ON_FEES.

    Fee calculation uses integer arithmetic to avoid floating-point issues:
        fee_amount = payment_amount * fee_rate_bps // 10000

    Args:
        config: Generator configuration
        payments: List of payments to generate fees for
        rng: Deterministic random number generator

    Returns:
        List of Fee objects with unique IDs and valid payment references
    """
    fees = []
    fee_counter = 0

    for payment in payments:
        # Always generate a TRANSACTION fee
        fee_counter += 1
        fee_id = f"FEE-{fee_counter:06d}"

        # Integer arithmetic: amount * bps // 10000
        fee_amount = payment.amount * config.fee_rate_bps // 10000

        # Ensure minimum fee of 1 paise
        fee_amount = max(1, fee_amount)

        fee = Fee(
            fee_id=fee_id,
            payment_id=payment.payment_id,
            amount=fee_amount,
            fee_type=FeeType.TRANSACTION,
        )
        fees.append(fee)

        # ~30% of payments also have a PLATFORM fee (0.5%)
        if rng.should_trigger(0.30):
            fee_counter += 1
            platform_fee_id = f"FEE-{fee_counter:06d}"
            platform_fee_amount = payment.amount * 50 // 10000  # 50 bps = 0.5%
            platform_fee_amount = max(1, platform_fee_amount)

            platform_fee = Fee(
                fee_id=platform_fee_id,
                payment_id=payment.payment_id,
                amount=platform_fee_amount,
                fee_type=FeeType.PLATFORM,
            )
            fees.append(platform_fee)

    return fees
