"""
PaymentAttempt - authorisation/capture granularity beneath a Payment.

NEW in CloseLoop 2.0 (architecture section 5). The existing synthetic generator
produces one payment = one capture, so a Payment is usually backed by exactly
one CAPTURE attempt; the table exists so that retries, multi-capture and
auth-then-capture histories have somewhere to live without changing the Payment
row's meaning.

No provider-specific execution logic lives here (task section 4).
"""

from sqlalchemy import Column, ForeignKey, Index, Integer, String
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

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
PAYMENT_ATTEMPT_KIND_VALUES = ("AUTH", "CAPTURE")
PAYMENT_ATTEMPT_STATUS_VALUES = (
    "CREATED",
    "AUTHORIZED",
    "CAPTURED",
    "FAILED",
    "CANCELLED",
)


class PaymentAttempt(Base):
    """One attempt to authorise or capture funds for a Payment."""

    __tablename__ = "payment_attempts"

    id = Column(String, primary_key=True)

    payment_id = Column(
        String,
        ForeignKey("payments.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # 1-based sequence within the payment.
    attempt_no = Column(Integer, nullable=False, default=1)
    kind = Column(String(16), nullable=False)

    amount = Column(MoneyType, nullable=False)  # paise
    currency = Column(CurrencyType, nullable=False, default="INR", server_default="INR")

    status = Column(
        String(32), nullable=False, default="CREATED", server_default="CREATED"
    )

    provider_attempt_id = Column(String, nullable=True)

    payment = relationship("Payment", back_populates="attempts")

    occurred_at = Column(TimestampType, nullable=True)
    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        non_negative_check("amount"),
        currency_check(),
        enum_check("kind", PAYMENT_ATTEMPT_KIND_VALUES, "ck_payment_attempts_kind"),
        enum_check(
            "status", PAYMENT_ATTEMPT_STATUS_VALUES, "ck_payment_attempts_status"
        ),
        Index("ix_payment_attempts_payment", "payment_id"),
        Index("ix_payment_attempts_payment_attempt_no", "payment_id", "attempt_no"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<PaymentAttempt id={self.id!r} payment_id={self.payment_id!r} "
            f"kind={self.kind!r} status={self.status!r}>"
        )
