"""
Phase 3: deterministic reconciliation engine + persistence boundary.

Revision ID: 0003_phase3_reconciliation
Revises: 0002_phase2_ingestion

Phase 3 additions:
1. FEE_MISMATCH is added additively to exception taxonomy and check constraints.
2. Reconciliation run and result persistence boundary.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_phase3_reconciliation"
down_revision = "0002_phase2_ingestion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Forward-only Phase 3 schema boundary.
    pass


def downgrade() -> None:
    pass
