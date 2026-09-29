"""
Shared column types and constraint helpers for the CloseLoop domain models.

Production database is PostgreSQL 16 (see ``CLOSELOOP_2.0_ARCHITECTURE.md``
section 6). The repository's test harness runs on SQLite in memory, so every
type here is chosen to produce the *production* type on PostgreSQL and a
faithful stand-in on SQLite. Nothing in this module is PostgreSQL-specific
behaviour hidden behind a SQLite substitute: money is integer minor units on
both backends, currency is an explicit 3-character code on both, statuses are
strings with ``CHECK`` constraints on both.

Money rule (architecture section 6, task section 5): **integer minor units,
never floating point.** ``MoneyType`` is ``BIGINT``/``BigInteger``; no ``Float``,
``Numeric`` or ``REAL`` column is ever used for a financial amount. Fractional
values (confidence scores) use ``Numeric`` and are not money.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects import postgresql

__all__ = [
    "JSONType",
    "MoneyType",
    "CurrencyType",
    "TimestampType",
    "utcnow",
    "currency_check",
    "non_negative_check",
    "enum_check",
    "CURRENCY_VALUES",
    "DEFAULT_CURRENCY",
]


# -----------------------------------------------------------------------------
# Column types
# -----------------------------------------------------------------------------

#: JSONB on PostgreSQL, JSON on SQLite. Used for metadata / snapshots / counts.
JSONType = JSON().with_variant(postgresql.JSONB(), "postgresql")

#: Integer minor units (paise). NEVER replace this with a float type.
MoneyType = BigInteger

#: ISO-4217 currency code, stored explicitly on every amount-bearing table.
CurrencyType = String(3)

#: Timezone-aware timestamp -> TIMESTAMPTZ on PostgreSQL.
TimestampType = DateTime(timezone=True)

#: Free-text columns (failure reasons, explanations, comments).
TextType = Text

#: Default currency for records created without an explicit one.
DEFAULT_CURRENCY = "INR"

#: Kept in sync with ``app.schemas.enums.Currency`` by a Phase 1 drift test.
CURRENCY_VALUES = ("INR", "USD")


def utcnow() -> datetime:
    """Timezone-aware UTC now, used as the default for timestamp columns."""
    return datetime.now(timezone.utc)


# -----------------------------------------------------------------------------
# CHECK constraint helpers
# -----------------------------------------------------------------------------


def currency_check(
    column: str = "currency", name: Optional[str] = None
) -> CheckConstraint:
    """``CHECK (column IN ('INR','USD'))`` - currency is always explicit."""
    return enum_check(column, CURRENCY_VALUES, name or f"ck_{column}_currency")


def non_negative_check(
    column: str, name: Optional[str] = None
) -> CheckConstraint:
    """``CHECK (column >= 0)`` for amounts that are never negative.

    Do NOT apply this to entity columns that explicitly represent a negative
    financial movement: ``adjustments.amount``, ``settlement_lines.amount`` and
    ``ledger_entries.amount`` are signed by design.
    """
    return CheckConstraint(f"{column} >= 0", name=name or f"ck_{column}_non_negative")


def enum_check(
    column: str, values: Iterable[str], name: Optional[str] = None
) -> CheckConstraint:
    """``CHECK (column IN (...))`` for status/type strings.

    Architecture section 6 prefers CHECK constraints over native PostgreSQL
    ENUMs for migration ease; this mirrors that decision. Values are passed as
    literal tuples rather than imported from the enum classes so the model layer
    keeps no dependency on ``app.schemas``; a Phase 1 test asserts the literals
    match the enums so drift fails loudly.
    """
    rendered = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(
        f"{column} IN ({rendered})", name=name or f"ck_{column}_enum"
    )
