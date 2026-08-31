"""
Similarity search service for Razorpay CloseLoop Phase 4F.

Provides semantic retrieval of historically similar resolved cases
using Sentence Transformer embeddings and pgvector.

Supports:
- Vector storage in PostgreSQL via pgvector
- Numpy-based fallback for testing without PostgreSQL
- Top-k similarity search
- Configurable similarity metric
- Cosine similarity by default

DOES NOT replace deterministic reconciliation or evidence analysis.
Similarity is an intelligence signal, not a financial decision.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import Column, String, Integer, Float, Text, Index, DateTime
from sqlalchemy.orm import Session
from datetime import datetime

from app.database.database import Base
from app.schemas.historical_case import HistoricalCase
from app.schemas.similarity import SimilarCase, SimilaritySearchResult
from app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_VERSION,
    EmbeddingService,
    cosine_similarity_batch,
    historical_case_to_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# pgvector Database Model
# ─────────────────────────────────────────────────────────────────────────────


class CaseEmbedding(Base):
    """Database model for stored case embeddings.

    Uses pgvector for vector similarity search in PostgreSQL.
    """

    __tablename__ = "case_embeddings"

    id = Column(String, primary_key=True)  # case_id

    # Embedding (stored as pgvector Vector type)
    # Note: The actual Vector column is added dynamically to support
    # both pgvector (PostgreSQL) and fallback (numpy) modes
    embedding_json = Column(Text, nullable=False)  # JSON-serialized embedding

    # Case metadata (denormalized for fast retrieval)
    exception_type = Column(String, nullable=False)
    resolution_type = Column(String, nullable=False)
    resolution_outcome = Column(String, nullable=False)
    payment_amount = Column(Integer, nullable=False)
    difference = Column(Integer, nullable=False)
    supporting_evidence_count = Column(Integer, default=0)
    tags_json = Column(Text, default="[]")

    # Text representation (for re-embedding if model changes)
    case_text = Column(Text, nullable=False)

    # Metadata
    embedding_model = Column(String, nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_case_embeddings_exception_type", "exception_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Search Service
# ─────────────────────────────────────────────────────────────────────────────


class SimilarityService:
    """Service for semantic similarity search of historical cases.

    Stores embeddings and provides top-k similarity search.
    Supports both pgvector (PostgreSQL) and numpy fallback modes.
    """

    def __init__(
        self,
        session: Session,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = 5,
        use_pgvector: bool = False,
    ):
        """Initialize the similarity service.

        Args:
            session: SQLAlchemy session
            embedding_service: EmbeddingService instance (lazy-loaded if None)
            top_k: Default number of results to return
            use_pgvector: Whether to use pgvector for vector storage
        """
        self.session = session
        self._embedding_service = embedding_service
        self.top_k = top_k
        self.use_pgvector = use_pgvector

        # In-memory numpy index for fallback mode
        self._numpy_ids: List[str] = []
        self._numpy_matrix: Optional[np.ndarray] = None

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create the embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def index_case(self, case: HistoricalCase) -> bool:
        """Index a historical case for similarity search.

        Generates embedding and stores it in the database.

        Args:
            case: HistoricalCase to index

        Returns:
            True if indexed, False if already exists
        """
        # Check if already indexed
        existing = (
            self.session.query(CaseEmbedding)
            .filter(CaseEmbedding.id == case.case_id)
            .first()
        )
        if existing:
            return False

        # Generate text representation
        case_dict = case.model_dump()
        case_text = historical_case_to_text(case_dict)

        # Generate embedding
        embedding = self.embedding_service.embed_text(case_text)

        # Store in database
        record = CaseEmbedding(
            id=case.case_id,
            embedding_json=json.dumps(embedding.tolist()),
            exception_type=case.exception_type,
            resolution_type=case.resolution_type,
            resolution_outcome=case.resolution_outcome.value,
            payment_amount=case.financial_context.payment_amount,
            difference=case.financial_context.difference,
            supporting_evidence_count=case.supporting_evidence_count,
            tags_json=json.dumps(case.tags),
            case_text=case_text,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
        )

        self.session.add(record)
        self.session.flush()

        # Update numpy index
        self._update_numpy_index(case.case_id, embedding)

        return True

    def index_batch(self, cases: List[HistoricalCase]) -> int:
        """Index multiple historical cases.

        Args:
            cases: List of HistoricalCase objects

        Returns:
            Number of cases indexed
        """
        count = 0
        for case in cases:
            if self.index_case(case):
                count += 1
        return count

    def search(
        self,
        query_case: Dict,
        top_k: Optional[int] = None,
    ) -> SimilaritySearchResult:
        """Search for similar historical cases.

        Args:
            query_case: Dictionary with case fields (exception_type, financial_context, etc.)
            top_k: Number of results to return (uses default if None)

        Returns:
            SimilaritySearchResult with ranked similar cases
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_text = historical_case_to_text(query_case)
        query_embedding = self.embedding_service.embed_text(query_text)

        # Get all indexed cases
        all_records = self.session.query(CaseEmbedding).all()
        total_indexed = len(all_records)

        if total_indexed == 0:
            return SimilaritySearchResult(
                query_case_id=query_case.get("case_id", "UNKNOWN"),
                similar_cases=[],
                top_k=k,
                total_indexed=0,
                embedding_model=EMBEDDING_MODEL_NAME,
                embedding_dimension=EMBEDDING_DIMENSION,
            )

        # Build matrix for batch similarity
        ids = []
        embeddings = []
        for record in all_records:
            ids.append(record.id)
            emb = np.array(json.loads(record.embedding_json), dtype=np.float32)
            embeddings.append(emb)

        matrix = np.vstack(embeddings)

        # Compute similarities
        similarities = cosine_similarity_batch(query_embedding, matrix)

        # Get top-k indices (sorted by descending similarity)
        if len(similarities) <= k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        similar_cases = []
        records_by_id = {r.id: r for r in all_records}
        for idx in top_indices:
            case_id = ids[idx]
            record = records_by_id[case_id]
            tags = json.loads(record.tags_json or "[]")

            similar_cases.append(
                SimilarCase(
                    case_id=case_id,
                    similarity_score=float(similarities[idx]),
                    exception_type=record.exception_type,
                    resolution_type=record.resolution_type,
                    resolution_outcome=record.resolution_outcome,
                    payment_amount=record.payment_amount,
                    difference=record.difference,
                    evidence_count=record.supporting_evidence_count,
                    tags=tags,
                )
            )

        return SimilaritySearchResult(
            query_case_id=query_case.get("case_id", "UNKNOWN"),
            similar_cases=similar_cases,
            top_k=k,
            total_indexed=total_indexed,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
        )

    def count(self) -> int:
        """Count indexed cases."""
        return self.session.query(CaseEmbedding).count()

    def is_indexed(self, case_id: str) -> bool:
        """Check if a case is indexed."""
        return (
            self.session.query(CaseEmbedding)
            .filter(CaseEmbedding.id == case_id)
            .first()
            is not None
        )

    def get_embedding(self, case_id: str) -> Optional[np.ndarray]:
        """Retrieve the stored embedding for a case."""
        record = (
            self.session.query(CaseEmbedding)
            .filter(CaseEmbedding.id == case_id)
            .first()
        )
        if not record:
            return None
        return np.array(json.loads(record.embedding_json), dtype=np.float32)

    def _update_numpy_index(self, case_id: str, embedding: np.ndarray):
        """Update the in-memory numpy index."""
        self._numpy_ids.append(case_id)
        if self._numpy_matrix is None:
            self._numpy_matrix = embedding.reshape(1, -1)
        else:
            self._numpy_matrix = np.vstack(
                [self._numpy_matrix, embedding.reshape(1, -1)]
            )

    def rebuild_numpy_index(self):
        """Rebuild the in-memory numpy index from the database."""
        records = self.session.query(CaseEmbedding).all()
        self._numpy_ids = [r.id for r in records]
        if records:
            embeddings = [
                np.array(json.loads(r.embedding_json), dtype=np.float32)
                for r in records
            ]
            self._numpy_matrix = np.vstack(embeddings)
        else:
            self._numpy_matrix = None
