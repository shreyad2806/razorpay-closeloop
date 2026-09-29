"""
PaymentProvider — the financial provider boundary.

Architecture section 14: CloseLoop never talks to an external financial system
directly. Everything CloseLoop learns about money arrives through a
``PaymentProvider`` implementation, and everything CloseLoop eventually *does*
to money will leave through the same boundary. Phase 2 uses the read/ingest
half of the boundary; the mutation half is intentionally left for the
execution phase so that later provider-side actions cannot bypass it.

Contract:

* every method returns provider DTOs (:mod:`app.providers.dto`), never ORM
  models — the provider represents the *external* system;
* every record reports ``provider`` and ``version`` so ingestion can be
  idempotent and version-aware;
* collections are returned newest-version-first per record is NOT promised —
  providers return a snapshot; ingestion resolves conflicts by version;
* a provider never touches a database session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


class PaymentProvider(ABC):
    """Read-side boundary to an external financial system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier of this provider (e.g. ``mock``, ``razorpay``)."""

    # -- merchants ---------------------------------------------------------

    @abstractmethod
    def get_merchants(self) -> list[ProviderMerchant]:
        """All merchants visible to CloseLoop on this provider."""

    # -- payments and their attempts ---------------------------------------

    @abstractmethod
    def get_payments(self) -> list[ProviderPayment]:
        """All payments in the provider snapshot."""

    @abstractmethod
    def get_payment_attempts(self) -> list[ProviderPaymentAttempt]:
        """Authorisation/capture attempts for the payments in the snapshot."""

    # -- settlements ---------------------------------------------------------

    @abstractmethod
    def get_settlements(self) -> list[ProviderSettlement]:
        """All settlements in the provider snapshot."""

    @abstractmethod
    def get_settlement_lines(self) -> list[ProviderSettlementLine]:
        """Component lines for the settlements in the snapshot."""

    # -- payment-level financial events --------------------------------------

    @abstractmethod
    def get_refunds(self) -> list[ProviderRefund]:
        """Refunds against the payments in the snapshot."""

    @abstractmethod
    def get_chargebacks(self) -> list[ProviderChargeback]:
        """Disputes against the payments in the snapshot."""

    @abstractmethod
    def get_fees(self) -> list[ProviderFee]:
        """Fees charged against the payments in the snapshot."""

    @abstractmethod
    def get_taxes(self) -> list[ProviderTax]:
        """Taxes applied to the payments in the snapshot."""
