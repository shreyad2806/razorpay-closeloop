"""
Legacy database bootstrap script.

CloseLoop 2.0 (architecture section 6) makes **Alembic** the schema authority:
``alembic upgrade head`` is how a database is created or migrated. This module is
kept only because it existed before, and it is corrected in two ways:

1. It imported three models out of ten, so ``create_all`` silently created an
   incomplete schema. It now imports the full model registry (``app.models``).
2. It executed ``create_all`` at import time, so merely importing the module had
   a side effect. The DDL now runs only when the module is executed directly.

For anything other than a throwaway local database, use the migration::

    cd backend && .venv/Scripts/alembic upgrade head
"""

from app.database.database import Base, engine

# Import the registry so every model is registered before metadata is used.
import app.models  # noqa: F401  (side effect: registers all models)


def create_all() -> None:
    """Create any missing tables from the ORM metadata (no migrations applied)."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_all()
    print("Tables created from ORM metadata (prefer `alembic upgrade head`).")
