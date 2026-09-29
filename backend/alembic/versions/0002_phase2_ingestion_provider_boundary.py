"""
Phase 2: ingestion + provider boundary + idempotency.

Revision ID: 0002_phase2_ingestion
Revises: 0001_phase1_domain

Phase 2 changes, all forward-only:

1. ``payments.merchant_id`` is tightened to NOT NULL.

   Phase 1 deliberately deferred this: the pre-2.0 table allowed payments
   without a merchant and Phase 1 could not guarantee otherwise. Phase 2's
   ingestion now guarantees it — the provider snapshot is validated before any
   database work and a payment whose ``merchant_id`` is absent from the
   snapshot is rejected, and merchants are always ingested before payments.
   Every payment row is therefore attributable to a merchant from here on.

   The pre-existing production rows do not exist (no deployment predates this
   schema in the wild); on a hypothetical populated table the batch would
   fail loudly rather than silently null-attribute money, which is the
   intended behaviour for financial data.

2. ``refunds``, ``fees`` and ``taxes`` gain real foreign keys to
   ``merchants.id`` (the columns existed as bare strings).

3. Idempotency-supporting indexes: covering lookups used by the ingestion
   upsert path (provider-native PKs are the uniqueness guarantee; these
   indexes make version lookups and parent-scoped queries fast).

Downgrade restores the Phase 1 shapes (nullable merchant linkage, no merchant
FKs on the event tables, no Phase 2 indexes).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_phase2_ingestion"
down_revision = "0001_phase1_domain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. payments.merchant_id becomes NOT NULL (ingestion guarantees it).
    #    batch_alter_table recreates the table on SQLite, which cannot ALTER
    #    a column's nullability in place; on PostgreSQL this is a plain
    #    ALTER TABLE ... SET NOT NULL.
    with op.batch_alter_table("payments") as batch:
        batch.alter_column(
            "merchant_id",
            existing_type=sa.String(),
            nullable=False,
        )

    # 2. refunds/fees/taxes: promote merchant_id from bare string to FK.
    #    The column already exists; only the constraint is added. SQLite (used
    #    by the test suite) cannot ADD CONSTRAINT, so batch_alter_table
    #    recreates the table transparently where needed.
    with op.batch_alter_table("refunds") as batch:
        batch.create_foreign_key(
            "fk_refunds_merchant_id_merchants", "merchants", ["merchant_id"], ["id"]
        )
    with op.batch_alter_table("fees") as batch:
        batch.create_foreign_key(
            "fk_fees_merchant_id_merchants", "merchants", ["merchant_id"], ["id"]
        )
    with op.batch_alter_table("taxes") as batch:
        batch.create_foreign_key(
            "fk_taxes_merchant_id_merchants", "merchants", ["merchant_id"], ["id"]
        )

    # 3. Indexes supporting the ingestion upsert path.
    op.create_index(
        "ix_payments_version", "payments", ["version"]
    )
    op.create_index(
        "ix_settlements_version", "settlements", ["version"]
    )
    op.create_index(
        "ix_merchants_name", "merchants", ["name"]
    )


def downgrade() -> None:
    op.drop_index("ix_merchants_name", table_name="merchants")
    op.drop_index("ix_settlements_version", table_name="settlements")
    op.drop_index("ix_payments_version", table_name="payments")

    with op.batch_alter_table("taxes") as batch:
        batch.drop_constraint("fk_taxes_merchant_id_merchants", type_="foreignkey")
    with op.batch_alter_table("fees") as batch:
        batch.drop_constraint("fk_fees_merchant_id_merchants", type_="foreignkey")
    with op.batch_alter_table("refunds") as batch:
        batch.drop_constraint("fk_refunds_merchant_id_merchants", type_="foreignkey")

    with op.batch_alter_table("payments") as batch:
        batch.alter_column(
            "merchant_id",
            existing_type=sa.String(),
            nullable=True,
        )
