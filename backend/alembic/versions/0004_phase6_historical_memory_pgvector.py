"""
Phase 6: Historical Memory pgvector persistence.

Revision ID: 0004_phase6_historical_memory_pgvector
Revises: 0003_phase3_reconciliation

Phase 6 additions:
1. Enable PostgreSQL vector extension.
2. Add Vector(384) column to CaseEmbedding.
3. Add HNSW vector index for vector_cosine_ops.
"""

from alembic import op
import sqlalchemy as sa
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    pass

# revision identifiers, used by Alembic.
revision = "0004_phase6_historical_memory_pgvector"
down_revision = "0003_phase3_reconciliation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        op.add_column("case_embeddings", sa.Column("embedding", Vector(384)))
        
        # Migrate existing data
        # embedding_json is like '[0.1, 0.2, ...]'
        # In Postgres, casting JSON list directly to vector is possible via array.
        # But embedding_json is Text, so we need to cast it to JSON, then to float array, then to vector.
        # A simpler way in Postgres: cast text to JSONB, then to vector. Wait, vector('...') accepts array string.
        # Actually, '[0.1, 0.2]' string maps exactly to vector format! So just cast it.
        op.execute(
            "UPDATE case_embeddings SET embedding = embedding_json::vector"
        )
        
        op.execute(
            "CREATE INDEX ix_case_embeddings_embedding ON case_embeddings "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
        )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_case_embeddings_embedding;")
        op.drop_column("case_embeddings", "embedding")
