"""
Chargeback - a disputed payment.

NEW entity in CloseLoop 2.0 (architecture section 5). The ``CHARGEBACK_FEE``
value already existed in ``FeeType`` but no chargeback record existed anywhere
in the schema, so dispute state, reason codes and evidence deadlines had no
home.

No dispute-handling workflow is implemented in Phase 1 - this is the
persistence model only.
"""

from sqlalchemy import Column, ForeignKey, Index, String, Text
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

# Kept in sync with app.schemas.enums.ChargebackStatus by a Phase 1 drift test.
CHARGEBACK_STATUS_VALUES = ("OPEN", "PRE_ARBITRATION", "WON", "LOST")


class Chargeback(Base):
    """A chargeback/dispute raised against a payment."""

    __tablename__ = "chargebacks"

    id = Column(String, primary_key=True)

    payment_id = Column(
        String,
        ForeignKey("payments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    merchant_id = Column(String, nullable=True)

    amount = Column(MoneyType, nullable=False)  # paise, unsigned
    currency = Column(CurrencyType, nullable=False, default="INR", server_default="INR")

    status = Column(
        String(32), nullable=False, default="OPEN", server_default="OPEN"
    )
    reason_code = Column(String(64), nullable=True)

    evidence_due_at = Column(TimestampType, nullable=True)
    resolved_at = Column(TimestampType, nullable=True)

    notes = Column(Text, nullable=True)

    payment = relationship("Payment", back_populates="chargebacks")

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        non_negative_check("amount"),
        currency_check(),
        enum_check("status", CHARGEBACK_STATUS_VALUES, "ck_chargebacks_status"),
        Index("ix_chargebacks_payment_status", "payment_id", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Chargeback id={self.id!r} payment_id={self.payment_id!r} "
            f"status={self.status!r} amount={self.amount!r}>"
        )
