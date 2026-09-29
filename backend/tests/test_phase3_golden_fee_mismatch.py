"""
Golden end-to-end integration tests for Phase 3 — Canonical FEE_MISMATCH scenario.

Exercises the full pipeline:
MockProvider -> IngestionService -> Test Database -> ReconciliationEngine -> PersistenceService
(ReconciliationRun, ReconciliationResult, FinancialException).
"""

from datetime import datetime, timezone

import pytest

from app.ingestion.service import IngestionService
from app.models.exception import FinancialException
from app.models.fee import Fee as DBFee
from app.models.merchant import Merchant as DBMerchant
from app.models.payment import Payment as DBPayment
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
from app.models.settlement import Settlement as DBSettlement
from app.providers.mock_provider import MockProvider
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import ExceptionType, FeeType, MatchStatus
from app.schemas.financial import Adjustment, Fee, Payment, Refund, Settlement, Tax
from app.services.persistence import PersistenceService


def test_golden_fee_mismatch_end_to_end(db_session):
    """
    Canonical Phase 3 golden integration test: FEE_MISMATCH detection.

    Canonical scenario (paise integers):
    - Payment: ₹10,000 = 1,000,000 paise
    - Fee: ₹500 = 50,000 paise
    - Expected settlement: payment - fee = ₹9,500 = 950,000 paise
    - Actual settlement: ₹9,250 = 925,000 paise
    - Difference: expected - actual = ₹250 = 25,000 paise
    - Exception type: FEE_MISMATCH

    Verifies the complete pipeline from MockProvider to persisted database models.
    """
    NOW = datetime.now(timezone.utc)
    batch_id = "BATCH-GOLDEN-001"
    case_id = "CASE-GOLDEN-001"

    # Step 1: MockProvider
    provider = MockProvider(seed=42)
    provider.add_merchant("MER-GOLDEN-001", name="Golden Merchant")
    provider.add_payment(
        "PAY-GOLDEN-001",
        merchant_id="MER-GOLDEN-001",
        amount=1_000_000,
        captured_at=NOW,
    )
    provider.add_fee(
        "FEE-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        amount=50_000,
        fee_type=FeeType.TRANSACTION,
        processed_at=NOW,
    )
    provider.add_settlement(
        "SET-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        merchant_id="MER-GOLDEN-001",
        amount=925_000,
        settled_at=NOW,
    )

    # Step 2: IngestionService -> Database
    ingestion_service = IngestionService(db_session)
    ingestion_result = ingestion_service.ingest(provider)
    assert ingestion_result.status == "COMPLETED"

    # Step 3: Verify ingested DB records
    merchant_db = db_session.get(DBMerchant, "MER-GOLDEN-001")
    payment_db = db_session.get(DBPayment, "PAY-GOLDEN-001")
    fee_db = db_session.get(DBFee, "FEE-GOLDEN-001")
    settlement_db = db_session.get(DBSettlement, "SET-GOLDEN-001")

    assert merchant_db is not None
    assert payment_db is not None and payment_db.amount == 1_000_000
    assert fee_db is not None and fee_db.amount == 50_000
    assert settlement_db is not None and settlement_db.amount == 925_000

    # Step 4: Run Reconciliation under a ReconciliationRun
    run = DBReconciliationRun(
        id="RUN-GOLDEN-001",
        scope_type="BATCH",
        scope_ref=batch_id,
        status="PENDING",
        trigger="MANUAL",
        started_at=NOW,
        engine_version="3.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    # Domain objects for calculation
    payment_domain = Payment(
        payment_id=payment_db.id,
        merchant_id=payment_db.merchant_id,
        amount=payment_db.amount,
        payment_timestamp=payment_db.created_at or NOW,
    )
    fee_domain = Fee(
        fee_id=fee_db.id,
        payment_id=fee_db.payment_id,
        amount=fee_db.amount,
        fee_type=FeeType.TRANSACTION,
        processed_at=fee_db.created_at or NOW,
    )
    settlement_domain = Settlement(
        settlement_id=settlement_db.id,
        payment_id=settlement_db.payment_id,
        merchant_id=settlement_db.merchant_id,
        amount=settlement_db.amount,
        settlement_timestamp=settlement_db.created_at or NOW,
    )

    result_schema = calculate_reconciliation(
        payment=payment_domain,
        settlements=[settlement_domain],
        refunds=[],
        fees=[fee_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id="REC-GOLDEN-001",
    )

    # Step 5: PersistenceService -> Database
    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(result_schema, batch_id=batch_id)
    db_result.reconciliation_run_id = run.id
    db_exception = persistence.persist_exception(result_schema, batch_id=batch_id)

    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = datetime.now(timezone.utc)
    run.transition_to("COMPLETED")
    db_session.commit()

    # Step 6: ReconciliationRun Assertions
    persisted_run = db_session.get(DBReconciliationRun, "RUN-GOLDEN-001")
    assert persisted_run is not None
    assert persisted_run.status == "COMPLETED"
    assert persisted_run.exception_count == 1
    assert persisted_run.started_at is not None
    assert persisted_run.finished_at is not None

    # Step 7: ReconciliationResult Assertions (Exact canonical values)
    persisted_result = (
        db_session.query(DBReconciliationResult)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .first()
    )
    assert persisted_result is not None
    assert persisted_result.payment_amount == 1_000_000
    assert persisted_result.expected_amount == 950_000
    assert persisted_result.actual_amount == 925_000
    assert persisted_result.difference == 25_000
    assert persisted_result.match_status == MatchStatus.EXCEPTION.value
    assert persisted_result.exception_type == ExceptionType.FEE_MISMATCH.value
    assert persisted_result.currency == "INR"

    # Step 8: FinancialException Assertions
    persisted_exception = (
        db_session.query(FinancialException)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .first()
    )
    assert persisted_exception is not None
    assert persisted_exception.expected_amount == 950_000
    assert persisted_exception.actual_amount == 925_000
    assert persisted_exception.difference == 25_000
    assert persisted_exception.exception_type == ExceptionType.FEE_MISMATCH.value
    assert persisted_exception.reconciliation_id == persisted_result.id
    assert persisted_exception.currency == "INR"

    # Step 9: Integer Arithmetic Assertions
    assert isinstance(persisted_result.payment_amount, int)
    assert isinstance(persisted_result.expected_amount, int)
    assert isinstance(persisted_result.actual_amount, int)
    assert isinstance(persisted_result.difference, int)
    assert isinstance(persisted_exception.difference, int)


def test_golden_exact_match_end_to_end(db_session):
    """
    Golden end-to-end test for exact match scenario.

    Canonical exact match:
    - Payment: ₹10,000 (1,000,000 paise)
    - Fee: ₹500 (50,000 paise)
    - Expected settlement: ₹9,500 (950,000 paise)
    - Actual settlement: ₹9,500 (950,000 paise)
    - Difference: 0
    - MATCHED status, no exception persisted.
    """
    NOW = datetime.now(timezone.utc)
    batch_id = "BATCH-MATCH-001"
    case_id = "CASE-MATCH-001"

    provider = MockProvider(seed=43)
    provider.add_merchant("MER-MATCH-001", name="Match Merchant")
    provider.add_payment(
        "PAY-MATCH-001",
        merchant_id="MER-MATCH-001",
        amount=1_000_000,
        captured_at=NOW,
    )
    provider.add_fee(
        "FEE-MATCH-001",
        payment_id="PAY-MATCH-001",
        amount=50_000,
        fee_type=FeeType.TRANSACTION,
        processed_at=NOW,
    )
    provider.add_settlement(
        "SET-MATCH-001",
        payment_id="PAY-MATCH-001",
        merchant_id="MER-MATCH-001",
        amount=950_000,
        settled_at=NOW,
    )

    ingestion_service = IngestionService(db_session)
    ingestion_result = ingestion_service.ingest(provider)
    assert ingestion_result.status == "COMPLETED"

    payment_db = db_session.get(DBPayment, "PAY-MATCH-001")
    fee_db = db_session.get(DBFee, "FEE-MATCH-001")
    settlement_db = db_session.get(DBSettlement, "SET-MATCH-001")

    run = DBReconciliationRun(
        id="RUN-MATCH-001",
        scope_type="BATCH",
        scope_ref=batch_id,
        status="PENDING",
        trigger="MANUAL",
        started_at=NOW,
        engine_version="3.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    payment_domain = Payment(
        payment_id=payment_db.id,
        merchant_id=payment_db.merchant_id,
        amount=payment_db.amount,
        payment_timestamp=payment_db.created_at or NOW,
    )
    fee_domain = Fee(
        fee_id=fee_db.id,
        payment_id=fee_db.payment_id,
        amount=fee_db.amount,
        fee_type=FeeType.TRANSACTION,
        processed_at=fee_db.created_at or NOW,
    )
    settlement_domain = Settlement(
        settlement_id=settlement_db.id,
        payment_id=settlement_db.payment_id,
        merchant_id=settlement_db.merchant_id,
        amount=settlement_db.amount,
        settlement_timestamp=settlement_db.created_at or NOW,
    )

    result_schema = calculate_reconciliation(
        payment=payment_domain,
        settlements=[settlement_domain],
        refunds=[],
        fees=[fee_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id="REC-MATCH-001",
    )

    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(result_schema, batch_id=batch_id)
    db_result.reconciliation_run_id = run.id
    db_exception = persistence.persist_exception(result_schema, batch_id=batch_id)

    run.matched_count = 1
    run.exception_count = 0
    run.finished_at = datetime.now(timezone.utc)
    run.transition_to("COMPLETED")
    db_session.commit()

    persisted_run = db_session.get(DBReconciliationRun, "RUN-MATCH-001")
    assert persisted_run.status == "COMPLETED"

    persisted_result = (
        db_session.query(DBReconciliationResult)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .first()
    )
    assert persisted_result.match_status == MatchStatus.MATCHED.value
    assert persisted_result.exception_type == ExceptionType.EXACT_MATCH.value
    assert persisted_result.expected_amount == 950_000
    assert persisted_result.actual_amount == 950_000
    assert persisted_result.difference == 0

    assert db_exception is None
    persisted_exception = (
        db_session.query(FinancialException)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .first()
    )
    assert persisted_exception is None


def test_golden_fee_mismatch_idempotency(db_session):
    """
    Test that re-running reconciliation over the same input is idempotent.
    """
    NOW = datetime.now(timezone.utc)
    batch_id = "BATCH-IDEM-001"
    case_id = "CASE-IDEM-001"

    provider = MockProvider(seed=44)
    provider.add_merchant("MER-IDEM-001", name="Idempotent Merchant")
    provider.add_payment(
        "PAY-IDEM-001",
        merchant_id="MER-IDEM-001",
        amount=1_000_000,
        captured_at=NOW,
    )
    provider.add_fee(
        "FEE-IDEM-001",
        payment_id="PAY-IDEM-001",
        amount=50_000,
        fee_type=FeeType.TRANSACTION,
        processed_at=NOW,
    )
    provider.add_settlement(
        "SET-IDEM-001",
        payment_id="PAY-IDEM-001",
        merchant_id="MER-IDEM-001",
        amount=925_000,
        settled_at=NOW,
    )

    ingestion_service = IngestionService(db_session)
    ingestion_service.ingest(provider)

    payment_db = db_session.get(DBPayment, "PAY-IDEM-001")
    fee_db = db_session.get(DBFee, "FEE-IDEM-001")
    settlement_db = db_session.get(DBSettlement, "SET-IDEM-001")

    payment_domain = Payment(
        payment_id=payment_db.id,
        merchant_id=payment_db.merchant_id,
        amount=payment_db.amount,
        payment_timestamp=NOW,
    )
    fee_domain = Fee(
        fee_id=fee_db.id,
        payment_id=fee_db.payment_id,
        amount=fee_db.amount,
        fee_type=FeeType.TRANSACTION,
        processed_at=NOW,
    )
    settlement_domain = Settlement(
        settlement_id=settlement_db.id,
        payment_id=settlement_db.payment_id,
        merchant_id=settlement_db.merchant_id,
        amount=settlement_db.amount,
        settlement_timestamp=NOW,
    )

    persistence = PersistenceService(db_session)

    # First run
    result1 = calculate_reconciliation(
        payment=payment_domain,
        settlements=[settlement_domain],
        refunds=[],
        fees=[fee_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id="REC-IDEM-001",
    )
    persistence.persist_reconciliation_result(result1, batch_id=batch_id)
    persistence.persist_exception(result1, batch_id=batch_id)
    db_session.commit()

    # Second run (replaying same batch/case)
    result2 = calculate_reconciliation(
        payment=payment_domain,
        settlements=[settlement_domain],
        refunds=[],
        fees=[fee_domain],
        taxes=[],
        adjustments=[],
        case_id=case_id,
        reconciliation_id="REC-IDEM-001",
    )
    persistence.persist_reconciliation_result(result2, batch_id=batch_id)
    persistence.persist_exception(result2, batch_id=batch_id)
    db_session.commit()

    # Verify single persisted result and exception row exist for (case_id, batch_id)
    results = (
        db_session.query(DBReconciliationResult)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .all()
    )
    exceptions = (
        db_session.query(FinancialException)
        .filter_by(case_id=case_id, batch_id=batch_id)
        .all()
    )

    assert len(results) == 1
    assert len(exceptions) == 1
    assert results[0].expected_amount == 950_000
    assert results[0].actual_amount == 925_000
    assert exceptions[0].exception_type == ExceptionType.FEE_MISMATCH.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
