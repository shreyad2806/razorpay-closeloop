"""
Phase 6: Historical Memory pgvector integration tests.
"""
import os
import pytest
from datetime import datetime, timezone
import json
import numpy as np

from app.schemas.historical_case import HistoricalCase, FinancialContext, ResolutionOutcome, ResolutionOrigin
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService, CaseEmbedding
from app.database.database import Base

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

HAS_PGVECTOR = True
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    HAS_PGVECTOR = False

# Only run if POSTGRES_URL is set, otherwise skip
POSTGRES_URL = os.environ.get("POSTGRES_TEST_URL")

@pytest.fixture(scope="module")
def engine():
    if not POSTGRES_URL:
        pytest.skip("POSTGRES_TEST_URL not set, skipping pgvector integration tests.")
    engine = create_engine(POSTGRES_URL)
    
    # Try creating vector extension
    with engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        except sqlalchemy.exc.ProgrammingError as e:
            pytest.skip(f"pgvector extension not available on this Postgres server: {e}")
            
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

@pytest.fixture
def similarity_service(session):
    return SimilarityService(session=session, use_pgvector=True)

def test_pgvector_persistence(similarity_service, session):
    case = HistoricalCase(
        case_id="PG-001",
        exception_id="EX-001",
        payment_id="PAY-001",
        exception_type="FEE_MISMATCH",
        financial_context=FinancialContext(
            payment_amount=1000, expected_amount=950, actual_amount=900, difference=50
        ),
        resolution_type="FEE_ADJUSTMENT",
        resolution_outcome=ResolutionOutcome.SUCCESSFUL,
        resolution_origin=ResolutionOrigin.DETERMINISTIC
    )
    
    indexed = similarity_service.index_case(case)
    assert indexed is True
    
    record = session.query(CaseEmbedding).filter_by(id="PG-001").first()
    assert record is not None
    assert record.embedding is not None

def test_golden_fee_mismatch(similarity_service, session):
    case = HistoricalCase(
        case_id="GOLDEN-001",
        exception_id="EX-GOLDEN",
        payment_id="PAY-GOLDEN",
        exception_type="FEE_MISMATCH",
        financial_context=FinancialContext(
            payment_amount=1000000,
            expected_amount=950000,
            actual_amount=925000,
            difference=25000
        ),
        resolution_type="FEE_ADJUSTMENT",
        resolution_outcome=ResolutionOutcome.SUCCESSFUL,
        resolution_origin=ResolutionOrigin.DETERMINISTIC
    )
    
    similarity_service.index_case(case)
    
    query = {
        "case_id": "QUERY-GOLDEN",
        "exception_type": "FEE_MISMATCH",
        "financial_context": {
            "payment_amount": 1000000,
            "expected_amount": 950000,
            "actual_amount": 925000,
            "difference": 25000,
        }
    }
    
    result = similarity_service.search(query, top_k=1)
    assert len(result.similar_cases) > 0
    assert result.similar_cases[0].case_id == "GOLDEN-001"
    assert result.similar_cases[0].exception_type == "FEE_MISMATCH"
    
def test_self_match_exclusion(similarity_service):
    # Tests that query_case_id is excluded
    case = HistoricalCase(
        case_id="SELF-001",
        exception_id="EX-SELF",
        payment_id="PAY-SELF",
        exception_type="FEE_MISMATCH",
        financial_context=FinancialContext(
            payment_amount=1000, expected_amount=900, actual_amount=900, difference=0
        ),
        resolution_type="FEE_ADJUSTMENT",
        resolution_outcome=ResolutionOutcome.SUCCESSFUL,
        resolution_origin=ResolutionOrigin.DETERMINISTIC
    )
    similarity_service.index_case(case)
    
    query = {
        "case_id": "SELF-001",
        "exception_type": "FEE_MISMATCH",
        "financial_context": {
            "payment_amount": 1000,
            "expected_amount": 900,
            "actual_amount": 900,
            "difference": 0,
        }
    }
    
    result = similarity_service.search(query)
    for c in result.similar_cases:
        assert c.case_id != "SELF-001"
