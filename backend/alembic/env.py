"""Alembic environment for CloseLoop.

Two things matter here:

1. **The target URL comes from the environment.** ``alembic.ini`` deliberately
   leaves ``sqlalchemy.url`` empty; the value is read from ``DATABASE_URL``
   (through ``app.database.database``, which also loads ``.env``). Migrations
   therefore always hit the same database the application hits, and no
   credential is committed.

2. **The metadata comes from the model registry.** ``import app.models``
   registers every table with ``Base.metadata``, which is what
   ``target_metadata`` needs for ``alembic revision --autogenerate``.

Autogenerate is supported from Phase 2 onward, but note the architecture's
warning (section 6): autogenerate does not detect CHECK constraints or
partial-index WHERE clauses, so those must be written by hand in each revision.
Revision 0001 is hand-frozen for exactly that reason.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the `app` package importable when alembic runs from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.database import Base  # noqa: E402
import app.models  # noqa: E402,F401  (side effect: registers every model)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# `app.database.database` raises if DATABASE_URL is unset, so reaching this line
# means a URL is available. It is read from there rather than re-read here so
# there is exactly one place that decides which database the app talks to.
from app.database.database import DATABASE_URL  # noqa: E402

target_metadata = Base.metadata


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL") or DATABASE_URL
    # Escape '%' so ConfigParser interpolation does not mangle passwords.
    return url.replace("%", "%%")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade --sql``)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
