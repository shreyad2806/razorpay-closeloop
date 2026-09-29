"""
SettlementLine - a component breakdown of one settlement.

NEW in CloseLoop 2.0 (architecture section 5). A settlement is a total; the
evidence graph needs to attach to the *line items* (gross, fee, tax, refund,
adjustment, chargeback) rather than only to the total, so every component gets
its own row pointing back at the entity that produced it.

``amount`` is SIGNED by design: refunds, fees and chargebacks reduce the payout,
gross/credits increase it. A non-negative CHECK therefore must not be applied
here.
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
    utcnow,
)

# Kept in sync with app.schemas.enums.SettlementComponentType by a Phase 1 test.
SETTLEMENT_COMPONENT_VALUES = (
    "GROSS",
    "FEE",
    "TAX",
    "REFUND",
    "ADJUSTMENT",
    "CHARGEBACK",
)


class SettlementLine(Base):
    """One signed component of a settlement."""

    __tablename__ = "settlement_lines"

    id = Column(String, primary_key=True)

    settlement_id = Column(
        String,
        ForeignKey("settlements.id", ondelete="RESTRICT"),
        nullable=False,
    )

    component_type = Column(String(32), nullable=False)

    # SIGNED: positive increases the payout, negative decreases it.
    amount = Column(MoneyType, nullable=False)  # paise
    currency = Column(CurrencyType, nullable=False, default="INR", server_default="INR")

    # Traceability to the entity this line was derived from, e.g.
    # ("FEE", "FEE-000123") or ("SETTLEMENT_GROSS", None) for the gross line.
    source_entity_type = Column(String(64), nullable=True)
    source_entity_id = Column(String, nullable=True)

    description = Column(Text, nullable=True)

    settlement = relationship("Settlement", back_populates="lines")

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        currency_check(),
        enum_check(
            "component_type",
            SETTLEMENT_COMPONENT_VALUES,
            "ck_settlement_lines_component_type",
        ),
        Index("ix_settlement_lines_settlement", "settlement_id"),
        Index(
            "ix_settlement_lines_source_entity",
            "source_entity_type",
            "source_entity_id",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<SettlementLine id={self.id!r} settlement_id={self.settlement_id!r} "
            f"component_type={self.component_type!r} amount={self.amount!r}>"
        )
