"""
Tax generator for synthetic financial data.

Generates deterministic tax records linked to payments.
"""

from typing import List

from app.generator.rng import DeterministicRNG
from app.schemas.config import GeneratorConfig
from app.schemas.enums import TaxType
from app.schemas.financial import Payment, Tax


def generate_taxes(
    config: GeneratorConfig,
    payments: List[Payment],
    rng: DeterministicRNG,
) -> List[Tax]:
    """
    Generate deterministic tax records.

    Every payment has GST applied (18% by default).
    Some payments may also have TDS.

    Tax calculation uses integer arithmetic to avoid floating-point issues:
        tax_amount = payment_amount * tax_rate_bps // 10000

    Args:
        config: Generator configuration
        payments: List of payments to generate taxes for
        rng: Deterministic random number generator

    Returns:
        List of Tax objects with unique IDs and valid payment references
    """
    taxes = []
    tax_counter = 0

    for payment in payments:
        # Always generate GST
        tax_counter += 1
        tax_id = f"TAX-{tax_counter:06d}"

        # Integer arithmetic: amount * bps // 10000
        tax_amount = payment.amount * config.tax_rate_bps // 10000

        # Ensure minimum tax of 1 paise
        tax_amount = max(1, tax_amount)

        tax = Tax(
            tax_id=tax_id,
            payment_id=payment.payment_id,
            amount=tax_amount,
            tax_type=TaxType.GST,
        )
        taxes.append(tax)

        # ~10% of payments also have TDS (1%)
        if rng.should_trigger(0.10):
            tax_counter += 1
            tds_id = f"TAX-{tax_counter:06d}"
            tds_amount = payment.amount * 100 // 10000  # 100 bps = 1%
            tds_amount = max(1, tds_amount)

            tds = Tax(
                tax_id=tds_id,
                payment_id=payment.payment_id,
                amount=tds_amount,
                tax_type=TaxType.TDS,
            )
            taxes.append(tds)

    return taxes
