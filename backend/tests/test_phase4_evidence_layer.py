"""
Phase 4 — Evidence Layer Integration & Verification Test Suite.

Verifies:
1. Evidence creation, normalization, persistence in DB ('evidence' and 'evidence_links' tables).
2. Deterministic content hashing SHA-256 (canonical JSON, transient-key stripping, key-order invariance).
3. Evidence coverage calculation and missing evidence detection without fake fabrication.
4. Evidence deduplication (idempotent collection).
5. Snapshot stability (persisted factual observation remains unchanged upon underlying DB mutation).
6. Graph persistence and NetworkX DiGraph reconstruction from DB.
7. Golden FEE_MISMATCH end-to-end flow:
   MockProvider -> Ingestion -> DB -> Reconciliation -> Run -> Result -> Exception -> Evidence -> Graph -> Hashes -> Coverage.
"""

from datetime import datetime, timezone
import json
import pytest

from app.ingestion.service import IngestionService
from app.models.evidence import Evidence as DBEvidence
from app.models.evidence_link import EvidenceLink as DBEvidenceLink
from app.models.exception import FinancialException
from app.models.payment import Payment as DBPayment
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
from app.models.settlement import Settlement as DBSettlement
from app.providers.mock_provider import MockProvider
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import ExceptionType, FeeType, MatchStatus
from app.schemas.financial import Fee, Payment, Settlement
from app.services.evidence_graph import EvidenceGraphBuilder
from app.services.evidence_integrity import compute_canonical_hash
from app.services.evidence_retrieval import EvidenceRetrievalService
from app.services.persistence import PersistenceService


def _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-001", batch_id="BATCH-P4-001"):
    """Helper to run MockProvider -> Ingestion -> Reconciliation pipeline and return the created FinancialException."""
    NOW = datetime.now(timezone.utc)
    provider = MockProvider(seed=100)
    provider.add_merchant("MER-P4-001", name="Phase 4 Merchant")
    provider.add_payment("PAY-P4-001", merchant_id="MER-P4-001", amount=1_000_000, captured_at=NOW)
    provider.add_fee("FEE-P4-001", payment_id="PAY-P4-001", amount=50_000, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    provider.add_settlement("SET-P4-001", payment_id="PAY-P4-001", merchant_id="MER-P4-001", amount=925_000, settled_at=NOW)

    ingestion_service = IngestionService(db_session)
    ingestion_service.ingest(provider)

    run = DBReconciliationRun(
        id=f"RUN-{case_id}",
        scope_type="BATCH",
        scope_ref=batch_id,
        status="PENDING",
        trigger="MANUAL",
        started_at=NOW,
        engine_version="4.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    payment_db = db_session.get(DBPayment, "PAY-P4-001")
    settlement_db = db_session.get(DBSettlement, "SET-P4-001")
    from app.models.fee import Fee as DBFee
    fee_db = db_session.get(DBFee, "FEE-P4-001")

    p_domain = Payment(payment_id=payment_db.id, merchant_id=payment_db.merchant_id, amount=payment_db.amount, payment_timestamp=NOW)
    f_domain = Fee(fee_id=fee_db.id, payment_id=fee_db.payment_id, amount=fee_db.amount, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    s_domain = Settlement(settlement_id=settlement_db.id, payment_id=settlement_db.payment_id, merchant_id=settlement_db.merchant_id, amount=settlement_db.amount, settlement_timestamp=NOW)

    result_schema = calculate_reconciliation(
        payment=p_domain,
        settlements=[s_domain],
        refunds=[],
        fees=[f_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id=f"REC-{case_id}",
    )

    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(result_schema, batch_id=batch_id)
    db_result.reconciliation_run_id = run.id
    db_exception = persistence.persist_exception(result_schema, batch_id=batch_id)

    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = datetime.now(timezone.utc)
    run.transition_to("COMPLETED")
    db_session.commit()

    return db_exception


def test_evidence_creation(db_session):
    """Test 1: Evidence package creation from FinancialException."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-EXC1")
    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_exception_id(exc.id, persist_links=False)

    assert package is not None
    assert package.exception_id == exc.id
    assert package.payment is not None
    assert package.payment.record_id == "PAY-P4-001"
    assert len(package.settlements) == 1
    assert package.settlements[0].record_id == "SET-P4-001"
    assert len(package.fees) == 1
    assert package.fees[0].record_id == "FEE-P4-001"


def test_evidence_persistence(db_session):
    """Test 2: Evidence nodes and links exist in DB after retrieval with persist_links=True."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-EXC2")
    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_exception_id(exc.id, persist_links=True)

    db_session.commit()

    ev_nodes = db_session.query(DBEvidence).filter_by(exception_id=exc.id).all()
    ev_links = db_session.query(DBEvidenceLink).filter_by(exception_id=exc.id).all()

    assert len(ev_nodes) >= 3  # Payment, Settlement, Fee
    assert len(ev_links) >= 3
    for node in ev_nodes:
        assert node.payload is not None
        assert "content_hash" in node.payload


def test_evidence_links(db_session):
    """Test 3: Evidence links persist entity relationships correctly."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-EXC3")
    retrieval = EvidenceRetrievalService(db_session)
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    db_session.commit()

    links = db_session.query(DBEvidenceLink).filter_by(exception_id=exc.id).all()
    entity_types = {link.entity_type for link in links}
    assert {"PAYMENT", "SETTLEMENT", "FEE"} <= entity_types


def test_evidence_hash_determinism():
    """Test 4: Content hash is deterministic (same content -> same hash, key order invariant)."""
    p1 = {"amount": 1000000, "currency": "INR", "entity_id": "PAY-001", "status": "CAPTURED"}
    p2 = {"status": "CAPTURED", "entity_id": "PAY-001", "currency": "INR", "amount": 1000000}

    h1 = compute_canonical_hash(p1)
    h2 = compute_canonical_hash(p2)

    assert h1 == h2
    assert len(h1) == 64


def test_evidence_hash_changed_content():
    """Test 5: Content hash changes when factual financial content changes."""
    p1 = {"amount": 950000, "currency": "INR", "entity_id": "PAY-001"}
    p2 = {"amount": 925000, "currency": "INR", "entity_id": "PAY-001"}

    assert compute_canonical_hash(p1) != compute_canonical_hash(p2)


def test_evidence_coverage_complete(db_session):
    """Test 6: Complete evidence produces 100% coverage."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-COV1")
    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_exception_id(exc.id, persist_links=True)

    cov = retrieval.calculate_coverage(package)
    assert cov["coverage_pct"] == 100.0
    assert cov["is_complete"] is True
    assert len(cov["missing_required"]) == 0


def test_evidence_coverage_missing(db_session):
    """Test 7: Incomplete evidence produces <100% coverage without fabricating fake records."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-COV2")
    # Delete the fee record from DB to simulate missing fee evidence
    from app.models.fee import Fee as DBFee
    db_session.query(DBFee).filter_by(payment_id="PAY-P4-001").delete()
    db_session.commit()

    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_exception_id(exc.id, persist_links=False)

    cov = retrieval.calculate_coverage(package)
    assert cov["coverage_pct"] < 100.0
    assert cov["is_complete"] is False
    assert "FEE" in cov["missing_required"]


def test_evidence_deduplication(db_session):
    """Test 8: Repeated evidence collection calls are idempotent and do not duplicate rows."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-DEDUP")
    retrieval = EvidenceRetrievalService(db_session)

    # Collect 3 times
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    db_session.commit()

    nodes = db_session.query(DBEvidence).filter_by(exception_id=exc.id).all()
    links = db_session.query(DBEvidenceLink).filter_by(exception_id=exc.id).all()

    # Uniqueness check
    node_tuples = {(n.entity_type, n.entity_id, n.relationship) for n in nodes}
    link_tuples = {(l.entity_type, l.entity_id) for l in links}

    assert len(nodes) == len(node_tuples)
    assert len(links) == len(link_tuples)


def test_evidence_snapshot_stability(db_session):
    """Test 9: Historical evidence payload in DB remains unchanged when underlying payment is mutated."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-SNAP")
    retrieval = EvidenceRetrievalService(db_session)
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    db_session.commit()

    # Get original evidence node from DB
    node_before = (
        db_session.query(DBEvidence)
        .filter_by(exception_id=exc.id, entity_type="PAYMENT")
        .first()
    )
    original_hash = node_before.payload["content_hash"]
    original_amount = node_before.payload["amount"]

    # Mutate underlying DB payment
    payment_db = db_session.get(DBPayment, "PAY-P4-001")
    payment_db.amount = 999_999
    db_session.commit()

    # Query existing persisted evidence node from DB again
    node_after = (
        db_session.query(DBEvidence)
        .filter_by(exception_id=exc.id, entity_type="PAYMENT")
        .first()
    )

    assert node_after.payload["amount"] == original_amount
    assert node_after.payload["content_hash"] == original_hash


def test_evidence_graph_reconstruction(db_session):
    """Test 10: Persisted DB evidence can be reconstructed into a NetworkX DiGraph."""
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-P4-GRAPH")
    retrieval = EvidenceRetrievalService(db_session)
    retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    db_session.commit()

    graph = retrieval.reconstruct_graph(exc.id)

    assert graph is not None
    assert exc.id in graph.nodes
    assert "PAY-P4-001" in graph.nodes
    assert "SET-P4-001" in graph.nodes
    assert "FEE-P4-001" in graph.nodes


def test_golden_fee_mismatch_phase4_end_to_end(db_session):
    """
    Test 11: Full Golden FEE_MISMATCH workflow integration through Phase 4.

    Pipeline:
    MockProvider -> Ingestion -> DB -> Reconciliation -> Exception -> Evidence -> Graph -> Hashes -> Coverage

    Canonical assertions:
    - expected = 950000
    - actual = 925000
    - difference = 25000
    - exception = FEE_MISMATCH
    - evidence persisted with content hashes
    - coverage = 100%
    """
    exc = _setup_golden_fee_mismatch_exception(db_session, case_id="CASE-GOLDEN-P4")
    assert exc.expected_amount == 950_000
    assert exc.actual_amount == 925_000
    assert exc.difference == 25_000
    assert exc.exception_type == ExceptionType.FEE_MISMATCH.value

    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_exception_id(exc.id, persist_links=True)
    db_session.commit()

    assert package is not None
    assert package.expected_amount == 950_000
    assert package.actual_amount == 925_000
    assert package.difference == 25_000
    assert package.exception_type == "FEE_MISMATCH"

    cov = retrieval.calculate_coverage(package)
    assert cov["coverage_pct"] == 100.0
    assert cov["is_complete"] is True

    graph = retrieval.reconstruct_graph(exc.id)
    assert graph.has_node(exc.id)
    assert graph.has_node("PAY-P4-001")
    assert graph.has_node("SET-P4-001")
    assert graph.has_node("FEE-P4-001")


def test_golden_matched_case_no_exception_evidence(db_session):
    """Test 12: Golden MATCHED case does not generate exception evidence."""
    NOW = datetime.now(timezone.utc)
    provider = MockProvider(seed=101)
    provider.add_merchant("MER-P4-M1", name="Match Merchant")
    provider.add_payment("PAY-P4-M1", merchant_id="MER-P4-M1", amount=1_000_000, captured_at=NOW)
    provider.add_fee("FEE-P4-M1", payment_id="PAY-P4-M1", amount=50_000, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    provider.add_settlement("SET-P4-M1", payment_id="PAY-P4-M1", merchant_id="MER-P4-M1", amount=950_000, settled_at=NOW)

    ingestion_service = IngestionService(db_session)
    ingestion_service.ingest(provider)

    p_domain = Payment(payment_id="PAY-P4-M1", merchant_id="MER-P4-M1", amount=1_000_000, payment_timestamp=NOW)
    f_domain = Fee(fee_id="FEE-P4-M1", payment_id="PAY-P4-M1", amount=50_000, fee_type=FeeType.TRANSACTION, processed_at=NOW)
    s_domain = Settlement(settlement_id="SET-P4-M1", payment_id="PAY-P4-M1", merchant_id="MER-P4-M1", amount=950_000, settlement_timestamp=NOW)

    result_schema = calculate_reconciliation(
        payment=p_domain,
        settlements=[s_domain],
        refunds=[],
        fees=[f_domain],
        taxes=[],
        adjustments=[],
        case_id="CASE-P4-MATCH",
        reconciliation_id="REC-P4-MATCH",
    )

    persistence = PersistenceService(db_session)
    persistence.persist_reconciliation_result(result_schema, batch_id="BATCH-P4-MATCH")
    db_exception = persistence.persist_exception(result_schema, batch_id="BATCH-P4-MATCH")
    db_session.commit()

    assert db_exception is None
    retrieval = EvidenceRetrievalService(db_session)
    package = retrieval.retrieve_by_case_id("CASE-P4-MATCH")
    assert package is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
