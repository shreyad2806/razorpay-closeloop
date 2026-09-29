"""
Settlement - a settlement received from a provider.

REFACTORED in CloseLoop 2.0 (architecture section 5): the pre-2.0 table had only
``id``, ``payment_id`` and ``amount``, so there was no merchant linkage, no
currency, no provider batch reference and no settled timestamp.

Partial settlements remain separate rows (the reconciliation engine already sums
settlements per payment), so ``id`` stays the provider settlement id.

Lifecycle (section 7.1, mirror of provider state):
``CREATED/PENDING -> PROCESSING -> SETTLED | FAILED``, expressed as a CHECK
constraint on the status column. The full transition table for settlements is not
part of section 7, so no domain machine is registered for it in Phase 1.

Money: ``amount`` is integer paise (``BIGINT``).
"""

from sqlalchemy import Column, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.models.types import (
    CurrencyType,
    MoneyType,
    TimestampType,
    currency_check,
    enum_check,
    non_negative_check,
    utcnow,
)

# Kept in sync with app.schemas.enums.SettlementStatus by a Phase 1 drift test.
SETTLEMENT_STATUS_VALUES = ("PENDING", "PROCESSING", "SETTLED", "FAILED")


class Settlement(Base):
    """A provider settlement covering one or more payments."""

    __tablename__ = "settlements"

    id = Column(String, primary_key=True)  # provider settlement id, e.g. setl_*

    payment_id = Column(
        String,
        ForeignKey("payments.id", ondelete="RESTRICT"),
        nullable=True,
    )
    merchant_id = Column(
        String,
        ForeignKey("merchants.id", ondelete="RESTRICT"),
        nullable=True,
    )

    # Integer paise. NEVER a float/double/real.
    amount = Column(MoneyType, nullable=False)
    currency = Column(
        CurrencyType, nullable=False, default="INR", server_default="INR"
    )

    status = Column(String(16), nullable=True)
    settled_at = Column(TimestampType, nullable=True)

    provider_batch_id = Column(String, nullable=True)

    version = Column(Integer, nullable=False, default=1, server_default="1")

    # server_default matters: these columns did not exist on the pre-2.0
    # `settlements` table, so migration 0001 adds them with ALTER TABLE. A NOT
    # NULL column can only be added to a populated table when the database has a
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

    payment = relationship("Payment", back_populates="settlements")
    lines = relationship(
        "SettlementLine", back_populates="settlement", cascade="save-update, merge"
    )

    __table_args__ = (
        non_negative_check("amount", "ck_settlements_amount_non_negative"),
        non_negative_check("version", "ck_settlements_version"),
        currency_check(),
        enum_check("status", SETTLEMENT_STATUS_VALUES, "ck_settlements_status"),
        Index("ix_settlements_payment", "payment_id"),
        Index("ix_settlements_settled_at", "settled_at"),
        Index("ix_settlements_merchant_settled_at", "merchant_id", "settled_at"),
        Index("ix_settlements_version", "version"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Settlement id={self.id!r} amount={self.amount!r} "
            f"{self.currency} status={self.status!r}>"
        )
