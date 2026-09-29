"""
Test database helper using SQLite in-memory.

Provides a test-friendly database setup that doesn't require PostgreSQL.
All tests that need database access should use this module.

Usage:
    from tests.db_test_helper import get_test_session, create_all_tables
"""

import os
import sys
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

# Create a test-specific Base (separate from production Base)
TestBase = declarative_base()

# SQLite in-memory engine
_test_engine = create_engine(
    "sqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
)


# Enable foreign key support for SQLite
@event.listens_for(_test_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_test_engine():
    """Get the SQLite test engine."""
    return _test_engine


def get_test_session():
    """Create a new test session."""
    TestSession = sessionmaker(bind=_test_engine)
    return TestSession()


def create_all_tables():
    """
    Create all tables in the SQLite test database.

    Uses TestBase.metadata.create_all() — safe for tests only.
    """
    # Import the full model registry so every table (and every foreign key
    # target) is registered before metadata.create_all runs.
    import app.models  # noqa: F401

    # The production models use `Base` from app.database.database.
    # For testing, we need to create those same tables in SQLite.
    # We use the production Base's metadata since all models are registered there.
    from app.database.database import Base as ProdBase
    ProdBase.metadata.create_all(bind=_test_engine)


def drop_all_tables():
    """Drop all tables (for test cleanup)."""
    from app.database.database import Base as ProdBase
    ProdBase.metadata.drop_all(bind=_test_engine)


def reset_database():
    """Drop and recreate all tables."""
    drop_all_tables()
    create_all_tables()


# ─────────────────────────────────────────────────────────────────────────────
# Parent row helpers
#
# CloseLoop 2.0 (Phase 1) added real foreign keys: a refund, fee, tax,
# adjustment or settlement can no longer be inserted without its Payment, and a
# Payment can no longer reference a Merchant that does not exist. Tests that use
# synthetic ids therefore need the parent rows created too. These helpers make
# that a single call instead of duplicating insert logic in every test file.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEST_MERCHANT_ID = "MER-TEST-01"


def ensure_merchant(session, merchant_id=DEFAULT_TEST_MERCHANT_ID, name=None):
    """Create the merchant if it is not already present. Idempotent."""
    from app.models.merchant import Merchant

    merchant = session.get(Merchant, merchant_id)
    if merchant is None:
        merchant = Merchant(
            id=merchant_id,
            name=name or f"Test Merchant {merchant_id}",
            status="ACTIVE",
        )
        session.add(merchant)
        session.flush()
    return merchant


def ensure_payment(
    session,
    payment_id,
    merchant_id=DEFAULT_TEST_MERCHANT_ID,
    amount=100000,
    currency="INR",
    status="CAPTURED",
):
    """Create the merchant and payment if they are not already present."""
    from app.models.payment import Payment

    payment = session.get(Payment, payment_id)
    if payment is None:
        ensure_merchant(session, merchant_id)
        payment = Payment(
            id=payment_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            status=status,
        )
        session.add(payment)
        session.flush()
    return payment


def ensure_financial_parents(session, payment_ids, merchant_id=DEFAULT_TEST_MERCHANT_ID):
    """Seed a merchant plus one payment per id in ``payment_ids``."""
    ensure_merchant(session, merchant_id)
    return [
        ensure_payment(session, payment_id, merchant_id)
        for payment_id in payment_ids
    ]
