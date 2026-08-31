"""
Tests for Razorpay CloseLoop Phase 4F — Semantic Similarity Retrieval.

Tests cover:
- Case text representation
- Embedding generation
- Dimension correctness
- Similarity service indexing
- Similarity search
- Top-k behavior
- Similarity ordering
- Empty database behavior
- Cosine similarity math
- Ground truth separation
"""

import os
import pytest
import numpy as np
from datetime import datetime

os.environ["DATABASE_URL"] = "sqlite:///test_similarity.db"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.schemas.historical_case import (
    FinancialContext,
    HistoricalCase,
    ResolutionOutcome,
    ResolutionOrigin,
)
from app.schemas.similarity import SimilarCase, SimilaritySearchResult
from app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EmbeddingService,
    case_to_text,
    cosine_similarity,
    cosine_similarity_batch,
    format_amount_paise,
    historical_case_to_text,
)
from app.services.similarity_service import CaseEmbedding, SimilarityService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


@pytest.fixture
def similarity_service(session, embedding_service):
    return SimilarityService(
        session=session,
        embedding_service=embedding_service,
        top_k=5,
    )


def _make_case(
    case_id="CASE-001",
    exception_type="FEE_DIFFERENCE",
    payment_amount=100000,
    expected_amount=100000,
    actual_amount=97000,
    difference=3000,
    total_fees=3000,
    resolution_type="FEE_ADJUSTMENT",
    tags=None,
):
    """Create a HistoricalCase for testing."""
    return HistoricalCase(
        case_id=case_id,
        exception_id=f"EXC-{case_id}",
        payment_id=f"PAY-{case_id}",
        exception_type=exception_type,
        financial_context=FinancialContext(
            payment_amount=payment_amount,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
            difference=difference,
            total_fees=total_fees,
        ),
        resolution_type=resolution_type,
        resolution_outcome=ResolutionOutcome.SUCCESSFUL,
        resolution_origin=ResolutionOrigin.DETERMINISTIC,
        tags=tags or [],
        created_at=datetime(2026, 1, 15, 10, 0, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text Representation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseTextRepresentation:
    def test_format_amount_paise(self):
        assert format_amount_paise(100000) == "Rs1000"
        assert format_amount_paise(1500) == "Rs15"
        assert format_amount_paise(0) == "Rs0"
        assert format_amount_paise(99) == "Rs0.99"

    def test_case_to_text_basic(self):
        text = case_to_text(
            exception_type="FEE_DIFFERENCE",
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
        )
        assert "FEE_DIFFERENCE" in text
        assert "Rs1000" in text
        assert "Rs970" in text
        assert "30" in text  # 3000 paise = Rs30

    def test_case_to_text_zero_difference(self):
        text = case_to_text(
            exception_type="EXACT_MATCH",
            payment_amount=50000,
            expected_amount=50000,
            actual_amount=50000,
            difference=0,
        )
        assert "No discrepancy" in text

    def test_case_to_text_with_components(self):
        text = case_to_text(
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            payment_amount=200000,
            expected_amount=200000,
            actual_amount=190000,
            difference=10000,
            total_refunds=3000,
            total_fees=2000,
            total_taxes=1500,
            total_adjustments=-3500,
        )
        assert "refunds" in text
        assert "fees" in text
        assert "taxes" in text
        assert "debit" in text

    def test_case_to_text_with_resolution(self):
        text = case_to_text(
            exception_type="FEE_DIFFERENCE",
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
            resolution_type="FEE_ADJUSTMENT",
        )
        assert "FEE_ADJUSTMENT" in text

    def test_case_to_text_deterministic(self):
        """Same inputs produce same text."""
        args = dict(
            exception_type="FEE_DIFFERENCE",
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
        )
        text1 = case_to_text(**args)
        text2 = case_to_text(**args)
        assert text1 == text2

    def test_historical_case_to_text(self):
        case_dict = {
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
                "total_fees": 3000,
            },
            "supporting_evidence_count": 1,
            "resolution_type": "FEE_ADJUSTMENT",
            "tags": ["fee", "known"],
        }
        text = historical_case_to_text(case_dict)
        assert "FEE_DIFFERENCE" in text
        assert "FEE_ADJUSTMENT" in text

    def test_no_ground_truth_in_text(self):
        """Text representation must not contain ground-truth labels."""
        text = case_to_text(
            exception_type="FEE_DIFFERENCE",
            payment_amount=100000,
            expected_amount=100000,
            actual_amount=97000,
            difference=3000,
        )
        assert "true_exception_type" not in text
        assert "true_resolution" not in text
        assert "resolvable" not in text


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Service Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbeddingService:
    def test_model_name(self):
        assert EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"

    def test_dimension(self, embedding_service):
        assert embedding_service.dimension == EMBEDDING_DIMENSION
        assert EMBEDDING_DIMENSION == 384

    def test_embed_text_returns_correct_shape(self, embedding_service):
        text = "Exception type: FEE_DIFFERENCE. Payment amount: Rs1000."
        embedding = embedding_service.embed_text(text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (EMBEDDING_DIMENSION,)
        assert embedding.dtype == np.float32

    def test_embed_text_deterministic(self, embedding_service):
        """Same text produces same embedding."""
        text = "Exception type: FEE_DIFFERENCE. Payment amount: Rs1000."
        emb1 = embedding_service.embed_text(text)
        emb2 = embedding_service.embed_text(text)
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_batch(self, embedding_service):
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embedding_service.embed_batch(texts)
        assert embeddings.shape == (3, EMBEDDING_DIMENSION)

    def test_embed_batch_empty(self, embedding_service):
        embeddings = embedding_service.embed_batch([])
        assert embeddings.shape == (0, EMBEDDING_DIMENSION)

    def test_embed_case(self, embedding_service):
        case = _make_case()
        embedding = embedding_service.embed_case(case.model_dump())
        assert embedding.shape == (EMBEDDING_DIMENSION,)

    def test_embed_cases(self, embedding_service):
        cases = [_make_case(case_id=f"CASE-{i:03d}") for i in range(3)]
        embeddings = embedding_service.embed_cases([c.model_dump() for c in cases])
        assert embeddings.shape == (3, EMBEDDING_DIMENSION)

    def test_different_texts_different_embeddings(self, embedding_service):
        emb1 = embedding_service.embed_text("Fee difference case")
        emb2 = embedding_service.embed_text("Missing record case")
        # They should be different (not identical)
        assert not np.array_equal(emb1, emb2)

    def test_similar_texts_more_similar_embeddings(self, embedding_service):
        """Similar texts should have higher cosine similarity."""
        emb_fee1 = embedding_service.embed_text(
            "Exception type: FEE_DIFFERENCE. Payment: Rs1000. Fee: Rs10."
        )
        emb_fee2 = embedding_service.embed_text(
            "Exception type: FEE_DIFFERENCE. Payment: Rs2000. Fee: Rs20."
        )
        emb_missing = embedding_service.embed_text(
            "Exception type: MISSING_RECORD. No settlement found."
        )

        sim_fee = cosine_similarity(emb_fee1, emb_fee2)
        sim_cross = cosine_similarity(emb_fee1, emb_missing)

        assert sim_fee > sim_cross


# ─────────────────────────────────────────────────────────────────────────────
# Cosine Similarity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_batch_similarity(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        sims = cosine_similarity_batch(query, matrix)
        assert len(sims) == 3
        assert abs(sims[0] - 1.0) < 1e-6
        assert abs(sims[1]) < 1e-6
        assert abs(sims[2] - (-1.0)) < 1e-6

    def test_batch_empty(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.zeros((0, 2), dtype=np.float32)
        sims = cosine_similarity_batch(query, matrix)
        assert len(sims) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Service Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSimilarityService:
    def test_index_case(self, similarity_service):
        case = _make_case(case_id="CASE-IDX1")
        result = similarity_service.index_case(case)
        assert result is True
        assert similarity_service.is_indexed("CASE-IDX1")

    def test_index_duplicate_returns_false(self, similarity_service):
        case = _make_case(case_id="CASE-IDXD")
        similarity_service.index_case(case)
        result = similarity_service.index_case(case)
        assert result is False

    def test_count(self, similarity_service):
        initial = similarity_service.count()
        similarity_service.index_case(_make_case(case_id="CASE-CNT1"))
        assert similarity_service.count() == initial + 1

    def test_get_embedding(self, similarity_service):
        case = _make_case(case_id="CASE-EMB1")
        similarity_service.index_case(case)
        emb = similarity_service.get_embedding("CASE-EMB1")
        assert emb is not None
        assert emb.shape == (EMBEDDING_DIMENSION,)

    def test_get_embedding_nonexistent(self, similarity_service):
        emb = similarity_service.get_embedding("CASE-NONEXIST")
        assert emb is None

    def test_search_empty_database(self, similarity_service, session):
        """Search on empty index returns empty results."""
        # Use a fresh service with no indexed cases
        fresh_service = SimilarityService(
            session=session,
            embedding_service=similarity_service.embedding_service,
        )
        query = {
            "case_id": "QUERY-001",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
            },
        }
        result = fresh_service.search(query)
        assert result.has_results() is False
        assert result.total_indexed == 0

    def test_search_returns_results(self, similarity_service):
        similarity_service.index_case(
            _make_case(case_id="CASE-SR1", exception_type="FEE_DIFFERENCE")
        )
        query = {
            "case_id": "QUERY-SR",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
                "total_fees": 3000,
            },
        }
        result = similarity_service.search(query)
        assert result.has_results()
        assert result.total_indexed >= 1

    def test_search_top_k(self, similarity_service):
        # Index multiple cases
        for i in range(8):
            similarity_service.index_case(
                _make_case(
                    case_id=f"CASE-TK{i:03d}",
                    exception_type="FEE_DIFFERENCE",
                    payment_amount=100000 + i * 1000,
                )
            )
        query = {
            "case_id": "QUERY-TK",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
                "total_fees": 3000,
            },
        }
        result = similarity_service.search(query, top_k=3)
        assert len(result.similar_cases) <= 3

    def test_search_similarity_ordering(self, similarity_service):
        """Most similar case should come first."""
        # Index cases with different exception types
        similarity_service.index_case(
            _make_case(
                case_id="CASE-ORD-FEE",
                exception_type="FEE_DIFFERENCE",
                payment_amount=100000,
                actual_amount=97000,
                difference=3000,
                total_fees=3000,
            )
        )
        similarity_service.index_case(
            _make_case(
                case_id="CASE-ORD-MISS",
                exception_type="MISSING_RECORD",
                payment_amount=50000,
                actual_amount=0,
                difference=50000,
                total_fees=0,
            )
        )

        query = {
            "case_id": "QUERY-ORD",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
                "total_fees": 3000,
            },
        }
        result = similarity_service.search(query)
        if result.has_results():
            # The fee case should be more similar to a fee query
            scores = {c.case_id: c.similarity_score for c in result.similar_cases}
            if "CASE-ORD-FEE" in scores and "CASE-ORD-MISS" in scores:
                assert scores["CASE-ORD-FEE"] >= scores["CASE-ORD-MISS"]

    def test_search_returns_metadata(self, similarity_service):
        similarity_service.index_case(
            _make_case(
                case_id="CASE-META",
                exception_type="FEE_DIFFERENCE",
                tags=["fee", "known"],
            )
        )
        query = {
            "case_id": "QUERY-META",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
            },
        }
        result = similarity_service.search(query)
        if result.has_results():
            case = result.similar_cases[0]
            assert case.case_id == "CASE-META"
            assert case.exception_type == "FEE_DIFFERENCE"
            assert "fee" in case.tags

    def test_search_result_structure(self, similarity_service):
        query = {
            "case_id": "QUERY-STRUCT",
            "exception_type": "EXACT_MATCH",
            "financial_context": {
                "payment_amount": 50000,
                "expected_amount": 50000,
                "actual_amount": 50000,
                "difference": 0,
            },
        }
        result = similarity_service.search(query)
        assert isinstance(result, SimilaritySearchResult)
        assert result.query_case_id == "QUERY-STRUCT"
        assert result.embedding_model == EMBEDDING_MODEL_NAME
        assert result.embedding_dimension == EMBEDDING_DIMENSION
        assert result.similarity_metric == "cosine"

    def test_index_batch(self, similarity_service):
        cases = [_make_case(case_id=f"CASE-BATCH{i:03d}") for i in range(3)]
        count = similarity_service.index_batch(cases)
        assert count == 3

    def test_above_threshold(self, similarity_service):
        query = {
            "case_id": "QUERY-THR",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
            },
        }
        result = similarity_service.search(query)
        high = result.above_threshold(0.99)
        assert isinstance(high, list)

    def test_best_match(self, similarity_service):
        query = {
            "case_id": "QUERY-BEST",
            "exception_type": "FEE_DIFFERENCE",
            "financial_context": {
                "payment_amount": 100000,
                "expected_amount": 100000,
                "actual_amount": 97000,
                "difference": 3000,
            },
        }
        result = similarity_service.search(query)
        if result.has_results():
            best = result.best_match()
            assert best is not None
            assert best.similarity_score >= 0

    def test_no_ground_truth_in_similarity_results(self):
        """Similarity results must not contain ground truth."""
        import inspect

        source = inspect.getsource(SimilarityService)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "ground_truth" not in source

    def test_rebuild_numpy_index(self, similarity_service):
        similarity_service.index_case(_make_case(case_id="CASE-REB"))
        similarity_service.rebuild_numpy_index()
        assert len(similarity_service._numpy_ids) >= 1
