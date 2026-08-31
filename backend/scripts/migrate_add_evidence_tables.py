#!/usr/bin/env python3
"""
Standalone migration script for Phase 3A evidence tables.

Creates the following new tables:
- refunds
- fees
- taxes
- adjustments
- evidence_links
- historical_resolutions

Does NOT modify existing tables (payments, settlements, exceptions,
reconciliation_results, reconciliation_evidence).

Usage:
    python scripts/migrate_add_evidence_tables.py

NOTE: Alembic is not installed in this environment. Once Alembic is
available, convert this to a proper Alembic migration.

Requires DATABASE_URL environment variable pointing to PostgreSQL.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database.database import engine, Base

# Import all models so Base.metadata knows about them
import app.models  # noqa: F401


MIGRATION_SQL = """
-- Phase 3A Migration: Financial Evidence Database Schema
-- Adds Refund, Fee, Tax, Adjustment, EvidenceLink, HistoricalResolution tables

-- ============================================================
-- REFUNDS
-- ============================================================
CREATE TABLE IF NOT EXISTS refunds (
    id VARCHAR NOT NULL PRIMARY KEY,
    payment_id VARCHAR NOT NULL,
    case_id VARCHAR,
    merchant_id VARCHAR,
    amount INTEGER NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'PROCESSED',
    refund_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_refunds_payment_id ON refunds (payment_id);
CREATE INDEX IF NOT EXISTS ix_refunds_case_id ON refunds (case_id);
CREATE INDEX IF NOT EXISTS ix_refunds_merchant_id ON refunds (merchant_id);
CREATE INDEX IF NOT EXISTS ix_refunds_payment_case ON refunds (payment_id, case_id);

-- ============================================================
-- FEES
-- ============================================================
CREATE TABLE IF NOT EXISTS fees (
    id VARCHAR NOT NULL PRIMARY KEY,
    payment_id VARCHAR NOT NULL,
    case_id VARCHAR,
    merchant_id VARCHAR,
    amount INTEGER NOT NULL,
    fee_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_fees_payment_id ON fees (payment_id);
CREATE INDEX IF NOT EXISTS ix_fees_case_id ON fees (case_id);
CREATE INDEX IF NOT EXISTS ix_fees_merchant_id ON fees (merchant_id);
CREATE INDEX IF NOT EXISTS ix_fees_payment_case ON fees (payment_id, case_id);

-- ============================================================
-- TAXES
-- ============================================================
CREATE TABLE IF NOT EXISTS taxes (
    id VARCHAR NOT NULL PRIMARY KEY,
    payment_id VARCHAR NOT NULL,
    case_id VARCHAR,
    merchant_id VARCHAR,
    amount INTEGER NOT NULL,
    tax_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_taxes_payment_id ON taxes (payment_id);
CREATE INDEX IF NOT EXISTS ix_taxes_case_id ON taxes (case_id);
CREATE INDEX IF NOT EXISTS ix_taxes_merchant_id ON taxes (merchant_id);
CREATE INDEX IF NOT EXISTS ix_taxes_payment_case ON taxes (payment_id, case_id);

-- ============================================================
-- ADJUSTMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS adjustments (
    id VARCHAR NOT NULL PRIMARY KEY,
    payment_id VARCHAR NOT NULL,
    case_id VARCHAR,
    merchant_id VARCHAR,
    amount INTEGER NOT NULL,
    adjustment_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_adjustments_payment_id ON adjustments (payment_id);
CREATE INDEX IF NOT EXISTS ix_adjustments_case_id ON adjustments (case_id);
CREATE INDEX IF NOT EXISTS ix_adjustments_merchant_id ON adjustments (merchant_id);
CREATE INDEX IF NOT EXISTS ix_adjustments_payment_case ON adjustments (payment_id, case_id);

-- ============================================================
-- EVIDENCE LINKS
-- ============================================================
CREATE TABLE IF NOT EXISTS evidence_links (
    id VARCHAR NOT NULL PRIMARY KEY,
    exception_id VARCHAR NOT NULL,
    case_id VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL,
    entity_id VARCHAR NOT NULL,
    relationship VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_evidence_links_exception_id ON evidence_links (exception_id);
CREATE INDEX IF NOT EXISTS ix_evidence_links_case_id ON evidence_links (case_id);
CREATE INDEX IF NOT EXISTS ix_evidence_links_entity_id ON evidence_links (entity_id);
CREATE INDEX IF NOT EXISTS ix_evidence_links_exception_entity ON evidence_links (exception_id, entity_type);
CREATE INDEX IF NOT EXISTS ix_evidence_links_case_entity ON evidence_links (case_id, entity_type);

-- ============================================================
-- HISTORICAL RESOLUTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS historical_resolutions (
    id VARCHAR NOT NULL PRIMARY KEY,
    exception_id VARCHAR,
    case_id VARCHAR NOT NULL,
    resolution_type VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    resolved_amount INTEGER,
    difference_at_resolution INTEGER,
    exception_type VARCHAR,
    resolvable BOOLEAN,
    notes VARCHAR,
    resolution_metadata VARCHAR,
    source VARCHAR DEFAULT 'deterministic',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_historical_resolutions_exception_id ON historical_resolutions (exception_id);
CREATE INDEX IF NOT EXISTS ix_historical_resolutions_case_id ON historical_resolutions (case_id);
CREATE INDEX IF NOT EXISTS ix_historical_resolutions_resolution_type ON historical_resolutions (resolution_type);
CREATE INDEX IF NOT EXISTS ix_historical_resolutions_outcome ON historical_resolutions (outcome);
"""


def run_migration():
    """Execute the migration SQL."""
    print("=" * 60)
    print("Phase 3A Migration: Financial Evidence Database Schema")
    print("=" * 60)
    print()

    # Split into individual statements
    statements = [
        stmt.strip()
        for stmt in MIGRATION_SQL.split(";")
        if stmt.strip() and not stmt.strip().startswith("--")
    ]

    with engine.begin() as conn:
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(text(stmt))
                print(f"  [{i}/{len(statements)}] OK")
            except Exception as e:
                # IF NOT EXISTS should prevent most failures
                print(f"  [{i}/{len(statements)}] WARN: {e}")

    print()
    print("Migration complete.")
    print()

    # Verify tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = [
        "refunds", "fees", "taxes", "adjustments",
        "evidence_links", "historical_resolutions",
    ]
    for t in expected:
        status = "✓" if t in tables else "✗"
        print(f"  {status} {t}")

    print()
    print("All new tables created successfully.")


if __name__ == "__main__":
    run_migration()
