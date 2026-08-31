"""
Embedding service for Razorpay CloseLoop Phase 4F.

Generates deterministic textual representations of financial cases
and produces embeddings using a Sentence Transformer model.

Model: all-MiniLM-L6-v2
- 384 dimensions
- Fast on CPU
- Suitable for short-text semantic similarity

DOES NOT use external APIs — all inference is local.
"""

import re
from typing import Dict, List, Optional, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Case Text Representation
# ─────────────────────────────────────────────────────────────────────────────


def format_amount_paise(amount: int) -> str:
    """Format paise amount to human-readable string.

    100000 paise = ₹1,000.00
    """
    rupees = amount / 100
    if rupees == int(rupees):
        return f"Rs{int(rupees)}"
    return f"Rs{rupees:.2f}"


def case_to_text(
    exception_type: str,
    payment_amount: int,
    expected_amount: int,
    actual_amount: int,
    difference: int,
    total_refunds: int = 0,
    total_fees: int = 0,
    total_taxes: int = 0,
    total_adjustments: int = 0,
    evidence_count: int = 0,
    resolution_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Convert a financial case to a deterministic textual representation.

    This text is used as input to the Sentence Transformer model.
    It must NOT contain ground-truth labels that would not be available
    for a new (unresolved) case.

    For historical cases, the resolution_type is included as metadata
    (it represents what was done, not a hidden label).

    Args:
        exception_type: Engine-classified exception type
        payment_amount: Original payment in paise
        expected_amount: Expected settlement in paise
        actual_amount: Actual settlement in paise
        difference: expected - actual in paise
        total_refunds: Total refunds in paise
        total_fees: Total fees in paise
        total_taxes: Total taxes in paise
        total_adjustments: Net adjustments in paise
        evidence_count: Number of evidence records
        resolution_type: Resolution applied (None for unresolved cases)
        tags: Case tags

    Returns:
        Deterministic text representation
    """
    parts = []

    # Exception type
    parts.append(f"Exception type: {exception_type}.")

    # Financial context
    parts.append(f"Payment amount: {format_amount_paise(payment_amount)}.")
    parts.append(f"Expected settlement: {format_amount_paise(expected_amount)}.")
    parts.append(f"Actual settlement: {format_amount_paise(actual_amount)}.")

    # Discrepancy
    if difference != 0:
        direction = "over" if difference > 0 else "under"
        parts.append(
            f"Discrepancy: {format_amount_paise(abs(difference))} {direction} expected."
        )
    else:
        parts.append("No discrepancy.")

    # Financial components (only mention non-zero)
    components = []
    if total_refunds > 0:
        components.append(f"refunds of {format_amount_paise(total_refunds)}")
    if total_fees > 0:
        components.append(f"fees of {format_amount_paise(total_fees)}")
    if total_taxes > 0:
        components.append(f"taxes of {format_amount_paise(total_taxes)}")
    if total_adjustments != 0:
        adj_dir = "credit" if total_adjustments > 0 else "debit"
        components.append(
            f"{adj_dir} adjustment of {format_amount_paise(abs(total_adjustments))}"
        )

    if components:
        parts.append(f"Financial components: {', '.join(components)}.")

    # Evidence
    if evidence_count > 0:
        parts.append(f"Evidence records available: {evidence_count}.")
    else:
        parts.append("No evidence records available.")

    # Resolution (only for historical cases)
    if resolution_type:
        parts.append(f"Resolution applied: {resolution_type}.")

    # Tags
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")

    return " ".join(parts)


def historical_case_to_text(case_dict: Dict) -> str:
    """Convert a historical case dict to text for embedding.

    Args:
        case_dict: Dictionary with case fields (from HistoricalCase or similar)

    Returns:
        Deterministic text representation
    """
    fc = case_dict.get("financial_context", {})
    return case_to_text(
        exception_type=case_dict.get("exception_type", "UNKNOWN"),
        payment_amount=fc.get("payment_amount", 0),
        expected_amount=fc.get("expected_amount", 0),
        actual_amount=fc.get("actual_amount", 0),
        difference=fc.get("difference", 0),
        total_refunds=fc.get("total_refunds", 0),
        total_fees=fc.get("total_fees", 0),
        total_taxes=fc.get("total_taxes", 0),
        total_adjustments=fc.get("total_adjustments", 0),
        evidence_count=case_dict.get("supporting_evidence_count", 0),
        resolution_type=case_dict.get("resolution_type"),
        tags=case_dict.get("tags"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Service
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingService:
    """Generates embeddings for financial case text using Sentence Transformers.

    Uses all-MiniLM-L6-v2 (384 dimensions) for local inference.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model name for Sentence Transformer
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the Sentence Transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: Input text

        Returns:
            numpy array of shape (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def embed_case(self, case_dict: Dict) -> np.ndarray:
        """Generate an embedding for a case dictionary.

        Args:
            case_dict: Dictionary with case fields

        Returns:
            numpy array of shape (dimension,)
        """
        text = historical_case_to_text(case_dict)
        return self.embed_text(text)

    def embed_cases(self, case_dicts: List[Dict]) -> np.ndarray:
        """Generate embeddings for multiple case dictionaries.

        Args:
            case_dicts: List of case dictionaries

        Returns:
            numpy array of shape (len(case_dicts), dimension)
        """
        texts = [historical_case_to_text(cd) for cd in case_dicts]
        return self.embed_batch(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine Similarity (numpy fallback for testing without PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (-1.0 to 1.0)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a query and a matrix of vectors.

    Args:
        query: Query vector of shape (dimension,)
        matrix: Matrix of shape (n, dimension)

    Returns:
        Array of similarity scores of shape (n,)
    """
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    # Avoid division by zero for zero-norm vectors
    norms = np.where(norms == 0, 1.0, norms)
    similarities = np.dot(matrix, query) / (norms * query_norm)
    return similarities.astype(np.float32)
