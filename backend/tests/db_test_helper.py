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
    # Import all model modules so they register with TestBase
    # We remap the production Base to TestBase for test models
    import app.models.payment
    import app.models.settlement
    import app.models.refund
    import app.models.fee
    import app.models.tax
    import app.models.adjustment
    import app.models.exception
    import app.models.reconciliation
    import app.models.evidence_link
    import app.models.historical_resolution

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
