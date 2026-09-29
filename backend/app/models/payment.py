"""
Payment - the logical payment.

REFACTORED in CloseLoop 2.0 (architecture section 5): the pre-2.0 table had only
``id``, ``merchant_id`` and ``amount``, which made it impossible to represent an
order, a currency, a provider status or an optimistic-locking version.

Lifecycle note (section 7.1): a Payment is a **read-only mirror of provider
state**. Only ingestion moves it, and only from a provider payload, upserted by
provider id + ``version``. CloseLoop never sets these states itself. The
transition table lives in :mod:`app.domain.state_machines` and is applied by
:meth:`Payment.transition_to`.

Money: ``amount`` is integer paise (``BIGINT``). Never a float.

Phase 2: ``merchant_id`` is NOT NULL. Ingestion validates every provider
payment against the snapshot's merchants before persisting, so a payment is
always attributable to a merchant. (Deferred in Phase 1; migration 0002
tightens the column.)
"""

from sqlalchemy import Column, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.domain import state_machines
from app.models.types import (
    CurrencyType,
    MoneyType,
    TimestampType,
    currency_check,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums.PaymentStatus by a Phase 1 drift test.
PAYMENT_STATUS_VALUES = (
    "PENDING",
    "CREATED",
    "AUTHORIZED",
    "CAPTURED",
    "SETTLED",
    "PARTIALLY_REFUNDED",
    "REFUNDED",
    "CHARGEBACK",
    "FAILED",
)


class Payment(Base):
    """A logical payment, mirrored from the provider."""

    __tablename__ = "payments"

    # Provider-native payment id (e.g. pay_*). Upsert target for ingestion.
    id = Column(String, primary_key=True)

    merchant_id = Column(
        String,
        ForeignKey("merchants.id", ondelete="RESTRICT"),
        nullable=False,
    )
    order_id = Column(String, nullable=True)

    # Integer paise. NEVER a float/double/real.
    amount = Column(MoneyType, nullable=False)
    currency = Column(
        CurrencyType, nullable=False, default="INR", server_default="INR"
    )

    status = Column(
        String(32), nullable=False, default="CREATED", server_default="CREATED"
    )
    method = Column(String(32), nullable=True)

    captured_at = Column(TimestampType, nullable=True)
    provider_created_at = Column(TimestampType, nullable=True)

    # Optimistic locking (architecture section 6).
    version = Column(Integer, nullable=False, default=1, server_default="1")

    # server_default matters: these columns did not exist on the pre-2.0
    # `payments` table, so migration 0001 adds them with ALTER TABLE. A NOT NULL
    # column can only be added to a populated table when the database has a
    # default to fall back on.
    created_at = Column(
        TimestampType, nullable=False, default=utcnow, server_default=func.now()
    )
    updated_at = Column(
        TimestampType,
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
    )

    attempts = relationship(
        "PaymentAttempt", back_populates="payment", cascade="save-update, merge"
    )
    refunds = relationship("Refund", back_populates="payment")
    chargebacks = relationship("Chargeback", back_populates="payment")
    settlements = relationship("Settlement", back_populates="payment")

    __table_args__ = (
        non_negative_check("amount", "ck_payments_amount_non_negative"),
        non_negative_check("version", "ck_payments_version"),
        currency_check(),
        enum_check("status", PAYMENT_STATUS_VALUES, "ck_payments_status"),
        Index(
            "ix_payments_merchant_provider_created",
            "merchant_id",
            "provider_created_at",
        ),
        Index("ix_payments_order_id", "order_id"),
        Index("ix_payments_status", "status"),
        Index("ix_payments_version", "version"),
    )

    # -- state machine ------------------------------------------------------

    def can_transition_to(self, new_status) -> bool:
        """True when ``self.status -> new_status`` is permitted."""
        return state_machines.is_valid_transition(
            state_machines.PAYMENT, self.status, new_status
        )

    def transition_to(self, new_status) -> str:
        """Validate and apply a provider status transition; raises when invalid."""
        self.status = state_machines.assert_transition(
            state_machines.PAYMENT, self.status, new_status
        )
        return self.status

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Payment id={self.id!r} amount={self.amount!r} "
            f"{self.currency} status={self.status!r}>"
        )
