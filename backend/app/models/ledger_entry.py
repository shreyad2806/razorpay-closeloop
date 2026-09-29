"""
LedgerEntry - the append-only event log of financial state changes.

NEW in CloseLoop 2.0 (architecture section 5). Every financial state change that
CloseLoop observes (INGEST) or causes (EXECUTION / ROLLBACK) writes one row
here. The re-reconciliation loop (later phase) compares a ledger-derived
expected amount against the provider-reported actual amount, which is only
possible if the financial history is an immutable event log rather than a
mutable balance.

**Append-only.** No UPDATE, no DELETE - enforced at the ORM level by
``app.models.immutability``. Reversals are new entries with entry_type
``REVERSAL``.

``amount`` is SIGNED: the direction column records the accounting side and the
sign records the effect.
"""

from sqlalchemy import Column, ForeignKey, Index, String

from app.database.database import Base
from app.models.types import (
    CurrencyType,
    JSONType,
    MoneyType,
    TimestampType,
    currency_check,
    enum_check,
    utcnow,
)
from app.models.immutability import register_append_only

# Kept in sync with app.schemas.enums by Phase 1 drift tests.
LEDGER_ENTRY_TYPE_VALUES = (
    "PAYMENT",
    "SETTLEMENT",
    "REFUND",
    "CHARGEBACK",
    "FEE",
    "TAX",
    "ADJUSTMENT",
    "REVERSAL",
)
LEDGER_DIRECTION_VALUES = ("DEBIT", "CREDIT")
LEDGER_SOURCE_VALUES = ("INGEST", "EXECUTION", "ROLLBACK")


class LedgerEntry(Base):
    """One immutable financial event."""

    __tablename__ = "ledger_entries"

    id = Column(String, primary_key=True)

    payment_id = Column(
        String,
        ForeignKey("payments.id", ondelete="RESTRICT"),
        nullable=True,
    )
    merchant_id = Column(String, nullable=True)

    entry_type = Column(String(32), nullable=False)

    # SIGNED by design - the ledger is not an unsigned-balance table.
    amount = Column(MoneyType, nullable=False)  # paise
    currency = Column(CurrencyType, nullable=False, default="INR", server_default="INR")
    direction = Column(String(16), nullable=False)
    source = Column(String(16), nullable=False)

    # Which record produced this entry, e.g. ("REFUND", "REF-000001").
    source_entity_type = Column(String(64), nullable=True)
    source_entity_id = Column(String, nullable=True)

    # When the financial event happened (provider time) vs when we saw it.
    occurred_at = Column(TimestampType, nullable=False)
    ingested_at = Column(TimestampType, nullable=False, default=utcnow)

    # Groups every entry produced by a single request/workflow.
    correlation_id = Column(String, nullable=True)

    entry_metadata = Column("metadata", JSONType, nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        currency_check(),
        enum_check("entry_type", LEDGER_ENTRY_TYPE_VALUES, "ck_ledger_entries_type"),
        enum_check(
            "direction", LEDGER_DIRECTION_VALUES, "ck_ledger_entries_direction"
        ),
        enum_check("source", LEDGER_SOURCE_VALUES, "ck_ledger_entries_source"),
        Index("ix_ledger_entries_payment_occurred", "payment_id", "occurred_at"),
        Index(
            "ix_ledger_entries_source_entity",
            "source_entity_type",
            "source_entity_id",
        ),
        Index("ix_ledger_entries_correlation_id", "correlation_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<LedgerEntry id={self.id!r} type={self.entry_type!r} "
            f"amount={self.amount!r} {self.direction!r}>"
        )


register_append_only(LedgerEntry)
