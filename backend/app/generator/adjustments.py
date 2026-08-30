"""
Adjustment generator for synthetic financial data.

Generates deterministic adjustment records linked to payments.
"""

from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import AdjustmentType
from app.schemas.financial import Adjustment, Payment


def generate_adjustments(
    config: GeneratorConfig,
    payments: List[Payment],
    rng: DeterministicRNG,
) -> List[Adjustment]:
    """
    Generate deterministic adjustment records.

    Most payments have no adjustments. A small subset (~10%) may have
    credits, debits, or corrections.

    Adjustment amounts use integer paise:
        - Positive = credit (increases settlement)
        - Negative = debit (decreases settlement)

    Args:
        config: Generator configuration
        payments: List of payments to potentially generate adjustments for
        rng: Deterministic random number generator

    Returns:
        List of Adjustment objects with unique IDs and valid payment references
    """
    adjustments = []
    adjustment_counter = 0

    # ~10% of payments have adjustments
    adjustment_probability = 0.10

    for payment in payments:
        if not rng.should_trigger(adjustment_probability):
            continue

        adjustment_counter += 1
        adjustment_id = f"ADJ-{adjustment_counter:06d}"

        # Adjustment type
        adj_type = rng.choice([
            AdjustmentType.CREDIT,
            AdjustmentType.DEBIT,
            AdjustmentType.CORRECTION,
        ])

        # Adjustment amount: 1-5% of payment amount
        adj_pct = rng.random_percentage(0.01, 0.05)
        adj_amount = int(payment.amount * adj_pct)
        adj_amount = max(1, adj_amount)  # At least 1 paise

        # Credits are positive, debits are negative
        if adj_type in (AdjustmentType.DEBIT, AdjustmentType.PENALTY):
            adj_amount = -adj_amount

        adjustment = Adjustment(
            adjustment_id=adjustment_id,
            payment_id=payment.payment_id,
            amount=adj_amount,
            adjustment_type=adj_type,
        )
        adjustments.append(adjustment)

    return adjustments
