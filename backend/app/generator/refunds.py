"""
Refund generator for synthetic financial data.

Generates deterministic refund records linked to payments.
"""

from datetime import datetime, timedelta
from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import RefundStatus
from app.schemas.financial import Payment, Refund


def generate_refunds(
    config: GeneratorConfig,
    payments: List[Payment],
    rng: DeterministicRNG,
) -> List[Refund]:
    """
    Generate deterministic refund records.

    Not all payments have refunds. A subset of payments (configured by probability)
    will have partial or full refunds.

    Args:
        config: Generator configuration
        payments: List of payments to potentially generate refunds for
        rng: Deterministic random number generator

    Returns:
        List of Refund objects with unique IDs and valid payment references
    """
    refunds = []
    refund_counter = 0

    # Approximately 15-25% of payments have refunds
    refund_probability = 0.20

    for payment in payments:
        if not rng.should_trigger(refund_probability):
            continue

        refund_counter += 1
        refund_id = f"REF-{refund_counter:06d}"

        # Refund is typically a portion of the payment (10-90%)
        refund_pct = rng.random_percentage(0.10, 0.90)
        refund_amount = int(payment.amount * refund_pct)

        # Ensure refund amount is at least 1 paise and at most payment amount
        refund_amount = max(1, min(refund_amount, payment.amount))

        # Refund occurs after payment
        days_offset = rng.randint(1, 7)
        refund_timestamp = payment.payment_timestamp + timedelta(days=days_offset)

        refund = Refund(
            refund_id=refund_id,
            payment_id=payment.payment_id,
            amount=refund_amount,
            status=RefundStatus.PROCESSED,
            refund_timestamp=refund_timestamp,
        )
        refunds.append(refund)

    return refunds
