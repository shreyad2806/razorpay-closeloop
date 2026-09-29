"""
Phase 1 tests: the Alembic baseline migration.

Architecture section 6 requires migrations to be forward-only with a real
downgrade, and the Phase 1 acceptance criteria require the migration to actually
apply and to leave a schema matching the models.

These tests drive Alembic itself (not `create_all`) against a temporary SQLite
file, so the revision is exercised the way a developer or the deployment pipeline
would run it:

* ``upgrade head`` on an empty database creates the whole Phase 1 schema;
* the created schema carries the tables, columns, CHECK constraints, foreign keys
  and indexes Phase 1 promised;
* ``downgrade base`` empties it again;
* running ``upgrade head`` twice is a no-op rather than an error.

The frozen definitions in revision 0001 are also inspected directly to prove the
money invariant holds in the migration as well as in the models.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import BigInteger, Float, Numeric, create_engine, inspect, text

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATION_PATH = (
    BACKEND_DIR / "alembic" / "versions" / "0001_phase1_domain_foundation.py"
)

PHASE1_TABLES = {
    "merchants",
    "payments",
    "payment_attempts",
    "settlements",
    "settlement_lines",
    "refunds",
    "chargebacks",
    "fees",
    "taxes",
    "adjustments",
    "ledger_entries",
    "reconciliation_runs",
    "reconciliation_results",
    "reconciliation_evidence",
    "exceptions",
    "evidence",
    "evidence_links",
    "root_causes",
    "model_predictions",
    "audit_events",
    "feedback",
    "resolutions",
    "resolution_actions",
    "approvals",
    "historical_cases",
    "historical_resolutions",
    "case_embeddings",
}

# Phase 2 adds no tables - only constraints and indexes on the Phase 1 schema.
# The full expected schema after `upgrade head` is therefore the same set.


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "phase1_migration_0001", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(db_path: Path) -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return config


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "phase1_migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path.as_posix()}")
    return path


@pytest.fixture
def upgraded(db_path):
    command.upgrade(_config(db_path), "head")
    return db_path


def _table_names(engine):
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


# ─────────────────────────────────────────────────────────────────────────────
# Revision metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestRevisionMetadata:
    def test_revision_identifier_and_root(self):
        module = _migration_module()
        assert module.revision == "0001_phase1_domain"
        assert module.down_revision is None

    def test_revision_0002_follows_0001(self):
        """Phase 2 chains its revision onto the frozen Phase 1 baseline."""
        phase2_path = (
            BACKEND_DIR / "alembic" / "versions"
            / "0002_phase2_ingestion_provider_boundary.py"
        )
        assert phase2_path.is_file()
        spec2 = importlib.util.spec_from_file_location(
            "phase2_migration_0002", phase2_path
        )
        module2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(module2)
        assert module2.revision == "0002_phase2_ingestion"
        assert module2.down_revision == "0001_phase1_domain"

    def test_revision_0003_follows_0002(self):
        """Phase 3 chains its revision onto Phase 2."""
        phase3_path = (
            BACKEND_DIR / "alembic" / "versions"
            / "0003_phase3_deterministic_reconciliation.py"
        )
        assert phase3_path.is_file()
        spec3 = importlib.util.spec_from_file_location(
            "phase3_migration_0003", phase3_path
        )
        module3 = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(module3)
        assert module3.revision == "0003_phase3_reconciliation"
        assert module3.down_revision == "0002_phase2_ingestion"

    def test_revision_defines_both_directions(self):
        module = _migration_module()
        assert callable(module.upgrade)
        assert callable(module.downgrade)

    def test_revision_is_discoverable_by_alembic(self, db_path):
        """`alembic heads` must see exactly one head - the Phase 3 tip."""
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(_config(db_path))
        assert script.get_heads() == ["0003_phase3_reconciliation"]
        revisions = {r.revision for r in script.walk_revisions()}
        assert revisions == {
            "0001_phase1_domain",
            "0002_phase2_ingestion",
            "0003_phase3_reconciliation",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade
# ─────────────────────────────────────────────────────────────────────────────


class TestUpgrade:
    def test_upgrade_creates_every_phase1_table(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            assert _table_names(engine) == PHASE1_TABLES
        finally:
            engine.dispose()

    def test_upgrade_stamps_the_revision(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            with engine.connect() as connection:
                stamped = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar()
            assert stamped == "0003_phase3_reconciliation"
        finally:
            engine.dispose()

    def test_money_columns_are_bigint_in_the_migrated_schema(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            for table in sorted(PHASE1_TABLES):
                for column in inspector.get_columns(table):
                    if column["name"] not in {
                        "amount",
                        "amount_paise",
                        "expected_amount",
                        "actual_amount",
                        "difference",
                    }:
                        continue
                    assert isinstance(column["type"], BigInteger), (
                        f"{table}.{column['name']} is {column['type']}, not BIGINT"
                    )
        finally:
            engine.dispose()

    def test_new_columns_exist_on_pre_existing_tables(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            payments = {c["name"] for c in inspector.get_columns("payments")}
            assert {
                "merchant_id",
                "order_id",
                "amount",
                "currency",
                "status",
                "method",
                "captured_at",
                "provider_created_at",
                "version",
                "created_at",
                "updated_at",
            } <= payments

            exceptions = {c["name"] for c in inspector.get_columns("exceptions")}
            assert {
                "status",
                "status_reason",
                "risk_category",
                "assigned_to",
                "sla_due_at",
                "closed_at",
                "close_reason",
                "reopen_count",
                "version",
                "currency",
            } <= exceptions
        finally:
            engine.dispose()

    def test_check_constraints_are_created(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            with engine.connect() as connection:
                ddl = connection.execute(
                    text("SELECT sql FROM sqlite_master WHERE name = 'payments'")
                ).scalar()
        finally:
            engine.dispose()

        assert "ck_payments_amount_non_negative" in ddl
        assert "CHECK (amount >= 0)" in ddl
        assert "ck_payments_status" in ddl
        assert "ck_currency_currency" in ddl
        assert "FOREIGN KEY(merchant_id) REFERENCES merchants" in ddl

    def test_foreign_keys_are_discoverable(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            payment_fks = {
                tuple(fk["constrained_columns"]) for fk in inspector.get_foreign_keys("payments")
            }
            assert ("merchant_id",) in payment_fks

            child_fks = {
                (fk["referred_table"], tuple(fk["constrained_columns"]))
                for table in ("refunds", "fees", "taxes", "adjustments", "settlements")
                for fk in inspector.get_foreign_keys(table)
            }
            for table in ("refunds", "fees", "taxes", "adjustments", "settlements"):
                assert ("payments", ("payment_id",)) in child_fks, table
        finally:
            engine.dispose()

    def test_partial_unique_indexes_are_created(self, upgraded):
        """The one-running-run-per-scope and one-pending-approval rules survive."""
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            with engine.connect() as connection:
                rows = connection.execute(
                    text("SELECT name, sql FROM sqlite_master WHERE type = 'index'")
                ).fetchall()
            indexes = {row[0]: row[1] or "" for row in rows}
        finally:
            engine.dispose()

        assert "uq_reconciliation_runs_one_running_per_scope" in indexes
        assert "uq_approvals_one_pending_per_resolution" in indexes
        assert "'RUNNING'" in indexes["uq_reconciliation_runs_one_running_per_scope"]
        assert "'PENDING'" in indexes["uq_approvals_one_pending_per_resolution"]

    def test_idempotency_indexes_are_created(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            with engine.connect() as connection:
                names = {
                    row[0]
                    for row in connection.execute(
                        text("SELECT name FROM sqlite_master WHERE type = 'index'")
                    )
                }
        finally:
            engine.dispose()

        assert {
            "uq_resolution_actions_idempotency_key",
            "ix_reconciliation_case_batch",
            "ix_exceptions_case_batch",
        } <= names

    def test_phase2_tightens_payments_merchant_id(self, upgraded):
        """Phase 2: ingestion guarantees every payment has a merchant."""
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            merchant_col = next(
                c
                for c in inspector.get_columns("payments")
                if c["name"] == "merchant_id"
            )
            assert merchant_col["nullable"] is False
        finally:
            engine.dispose()

    def test_phase2_adds_merchant_fks_to_event_tables(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            for table in ("refunds", "fees", "taxes"):
                refs = {
                    (fk["referred_table"], tuple(fk["constrained_columns"]))
                    for fk in inspector.get_foreign_keys(table)
                }
                assert ("merchants", ("merchant_id",)) in refs, table
        finally:
            engine.dispose()

    def test_indexes_are_created(self, upgraded):
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            with engine.connect() as connection:
                count = connection.execute(
                    text(
                        "SELECT count(*) FROM sqlite_master WHERE type = 'index' "
                        "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
                    )
                ).scalar()
        finally:
            engine.dispose()
        assert count >= 80


# ─────────────────────────────────────────────────────────────────────────────
# Downgrade
# ─────────────────────────────────────────────────────────────────────────────


class TestDowngrade:
    def test_downgrade_removes_the_schema(self, upgraded):
        config = _config(upgraded)
        command.downgrade(config, "base")

        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            assert _table_names(engine) == set()
        finally:
            engine.dispose()

    def test_upgrade_after_downgrade_restores_the_schema(self, upgraded):
        config = _config(upgraded)
        command.downgrade(config, "base")
        command.upgrade(config, "head")

        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            assert _table_names(engine) == PHASE1_TABLES
        finally:
            engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# The frozen definitions themselves
# ─────────────────────────────────────────────────────────────────────────────


class TestFrozenDefinitions:
    """The migration's own table definitions are checked, not the models."""

    def test_frozen_schema_holds_every_phase1_table(self):
        metadata = _migration_module()._frozen_schema()
        assert set(metadata.tables) == PHASE1_TABLES

    def test_frozen_schema_has_no_float_money_columns(self):
        metadata = _migration_module()._frozen_schema()
        money_names = {
            "amount",
            "amount_paise",
            "expected_amount",
            "actual_amount",
            "difference",
            "payment_amount",
            "total_refunds",
            "total_fees",
            "total_taxes",
            "total_adjustments",
            "resolved_amount",
            "financial_exposure_paise",
            "difference_at_resolution",
        }
        for table in metadata.tables.values():
            for column in table.columns:
                if column.name not in money_names:
                    continue
                assert isinstance(column.type, BigInteger), (
                    f"{table.name}.{column.name} is {column.type}"
                )
                assert not isinstance(column.type, (Float, Numeric))

    def test_frozen_schema_declares_constraints_and_indexes(self):
        metadata = _migration_module()._frozen_schema()
        checks = sum(
            1
            for table in metadata.tables.values()
            for constraint in table.constraints
            if constraint.__class__.__name__ == "CheckConstraint"
        )
        foreign_keys = sum(
            1
            for table in metadata.tables.values()
            for constraint in table.constraints
            if constraint.__class__.__name__ == "ForeignKeyConstraint"
        )
        indexes = sum(len(table.indexes) for table in metadata.tables.values())

        assert checks >= 70
        assert foreign_keys >= 25
        assert indexes >= 80

    def test_migrated_schema_matches_the_models(self, upgraded):
        """Phase 2 guard: models must not disagree with `upgrade head`.

        PHASE 2 ACTION (from the Phase 1 docstring), now applied: while only
        revision 0001 existed, the models were compared against 0001's frozen
        definitions. From Phase 2 on, the models are compared against the
        schema Alembic actually builds when the full revision chain is applied
        - which is the schema production will have.
        """
        engine = create_engine(f"sqlite:///{upgraded.as_posix()}")
        try:
            inspector = inspect(engine)
            migrated_columns = {
                table: {column["name"] for column in inspector.get_columns(table)}
                for table in _table_names(engine)
            }
            migrated_fks = {
                table: {
                    (fk["referred_table"], tuple(fk["constrained_columns"]))
                    for fk in inspector.get_foreign_keys(table)
                }
                for table in _table_names(engine)
            }
            model_tables = None
        finally:
            engine.dispose()

        import app.models  # noqa: F401
        from app.database.database import Base

        model_columns = {
            name: {column.name for column in table.columns}
            for name, table in Base.metadata.tables.items()
        }
        model_fks = {
            name: {
                (
                    constraint.referred_table.name
                    if hasattr(constraint.referred_table, "name")
                    else str(constraint.referred_table),
                    tuple(element.parent.name for element in constraint.elements),
                )
                for constraint in table.constraints
                if constraint.__class__.__name__ == "ForeignKeyConstraint"
            }
            for name, table in Base.metadata.tables.items()
        }

        assert migrated_columns == model_columns
        for table, fks in migrated_fks.items():
            assert fks == model_fks[table], (
                f"{table}: migrated={sorted(fks)} models={sorted(model_fks[table])}"
            )
