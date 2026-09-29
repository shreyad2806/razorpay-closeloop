"""
Financial provider boundary (architecture section 14).

``PaymentProvider`` is the only door between CloseLoop and an external
financial system. Implementations return provider DTOs, never ORM models.
"""

from app.providers.base import PaymentProvider
from app.providers.dto import (
    ProviderChargeback,
    ProviderFee,
    ProviderMerchant,
    ProviderPayment,
    ProviderPaymentAttempt,
    ProviderRefund,
    ProviderSettlement,
    ProviderSettlementLine,
    ProviderTax,
)
from app.providers.mock_provider import MockProvider

__all__ = [
    "PaymentProvider",
    "MockProvider",
    "ProviderChargeback",
    "ProviderFee",
    "ProviderMerchant",
    "ProviderPayment",
    "ProviderPaymentAttempt",
    "ProviderRefund",
    "ProviderSettlement",
    "ProviderSettlementLine",
    "ProviderTax",
]
