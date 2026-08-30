"""
Settlement generator for synthetic financial data.

Generates deterministic settlement records linked to payments.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import SettlementStatus
from app.schemas.financial import Merchant, Payment, Settlement


def generate_settlements(
    config: GeneratorConfig,
    payments: List[Payment],
    merchants: List[Merchant],
    rng: DeterministicRNG,
) -> List[Settlement]:
    """
    Generate deterministic settlement records.

    Normally, settlement amount = payment amount - fees - taxes + adjustments.
    Exception scenarios will be injected later to create discrepancies.

    Args:
        config: Generator configuration
        payments: List of payments to generate settlements for
        merchants: List of merchants (for merchant_id lookup)
        rng: Deterministic random number generator

    Returns:
        List of Settlement objects with unique IDs and valid references
    """
    # Build merchant lookup for efficient reference
    merchant_map = {m.merchant_id: m for m in merchants}

    settlements = []
    settlement_counter = 0

    for payment in payments:
        # Check if this payment should be missing (for MISSING_RECORD scenarios)
        if rng.should_trigger(config.missing_record_probability):
            # Skip settlement for this payment - will be handled as MISSING_RECORD
            continue

        settlement_counter += 1
        settlement_id = f"SET-{settlement_counter:06d}"

        # Settlement amount initially equals payment amount (normal case)
        # Exception scenarios will modify this in the case generator
        amount = payment.amount

        # Settlement occurs 1-3 days after payment (T+1 or T+2 settlement cycle)
        days_offset = rng.randint(1, 3)
        settlement_timestamp = payment.payment_timestamp + timedelta(days=days_offset)

        settlement = Settlement(
            settlement_id=settlement_id,
            payment_id=payment.payment_id,
            merchant_id=payment.merchant_id,
            amount=amount,
            currency=config.currency,
            status=SettlementStatus.SETTLED,
            settlement_timestamp=settlement_timestamp,
        )
        settlements.append(settlement)

    return settlements
