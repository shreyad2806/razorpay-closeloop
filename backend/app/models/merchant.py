"""
Merchant - the aggregation root for merchant-scoped financial data.

NEW in CloseLoop 2.0 (architecture section 5). Before 2.0, payments carried a
bare ``merchant_id`` string with no merchant record and therefore no place to
hang merchant-level features or scoped access.

Deletion policy: merchants are the only entity with a soft delete
(``status = INACTIVE`` + ``deactivated_at``). Financial records are never
deleted.
"""

from sqlalchemy import Column, Index, String

from app.database.database import Base
from app.models.types import (
    JSONType,
    TimestampType,
    enum_check,
    utcnow,
)

# Kept in sync with app.schemas.enums.MerchantStatus by a Phase 1 drift test.
MERCHANT_STATUS_VALUES = ("ACTIVE", "INACTIVE", "SUSPENDED")


class Merchant(Base):
    """A merchant/business whose money CloseLoop reconciles."""

    __tablename__ = "merchants"

    # Provider-native merchant identifier (idempotency: PK = provider id).
    id = Column(String, primary_key=True)

    name = Column(String(255), nullable=False)
    status = Column(
        String(32),
        nullable=False,
        default="ACTIVE",
        server_default="ACTIVE",
    )
    category_code = Column(String(64), nullable=True)

    onboarded_at = Column(TimestampType, nullable=True)
    deactivated_at = Column(TimestampType, nullable=True)

    metadata_json = Column("metadata", JSONType, nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)
    updated_at = Column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        enum_check("status", MERCHANT_STATUS_VALUES, "ck_merchants_status"),
        Index("ix_merchants_status", "status"),
        Index("ix_merchants_name", "name"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Merchant id={self.id!r} status={self.status!r}>"
