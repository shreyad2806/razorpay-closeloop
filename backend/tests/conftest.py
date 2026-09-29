"""
Shared pytest fixtures for database integration tests.

Provides isolated SQLite in-memory database sessions that:
- Reset between tests (no cross-test contamination)
- Are deterministic
- Clean up after themselves

CloseLoop 2.0 note: the full model registry is imported **at collection time**
(not inside a fixture). Several test modules build tables straight from
``Base.metadata`` using their own engine, and the tables now carry real foreign
keys, so a partially-registered metadata would fail with
``NoReferencedTableError``. Importing ``app.models`` here guarantees the complete
schema is available to every test.
"""

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep the tests off any real database: this must happen before app modules are
# imported, because app.database.database reads DATABASE_URL at import time.
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///file:test_prag?mode=memory&cache=shared&uri=true"
)

# Register every model with Base.metadata. Import the package, not individual
# modules: a model that is never imported never becomes a table.
import app.models  # noqa: E402,F401


@pytest.fixture(scope="session", autouse=True)
def set_database_url():
    """Set DATABASE_URL for the session so app.database.database doesn't crash."""
    os.environ.setdefault(
        "DATABASE_URL", "sqlite:///file:test_prag?mode=memory&cache=shared&uri=true"
    )


@pytest.fixture(scope="function")
def db_engine():
    """
    Create a fresh SQLite in-memory engine per test.
    Enables foreign keys and returns the engine.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Enable foreign key support for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Create a fresh database session with all tables created.
    Rolls back all changes after the test.
    """
    from app.database.database import Base

    # All models are already registered by the module-level import above; this
    # import is kept as a defensive no-op for readers.
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=db_engine)
    Session = sessionmaker(bind=db_engine)
    session = Session()

    yield session

    session.close()
    Base.metadata.drop_all(bind=db_engine)
