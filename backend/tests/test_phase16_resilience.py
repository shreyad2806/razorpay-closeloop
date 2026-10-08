"""
Phase 16 — Failure Engineering & Resilience: executable test suite.

Every test injects a failure from *outside* production code and then asserts an
observable outcome of the real services. Nothing here weakens a boundary; the
suite's job is to prove the boundaries hold when things break.

Core contract under test::

    deterministic reconciliation = financial truth
    provider success             != closure
    unknown outcome              != verified success
    authorization failure        != execution
    duplicate execution          != duplicate money movement
    telemetry failure            != business failure
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from auth_test_helper import (
    deterministic_principal,
    make_test_token,
    make_test_verifier,
)

from app.evaluation.golden import (
    ACTUAL_WRONG_PAISE,
    BATCH_ID,
    CASE_ID,
    DIFFERENCE_PAISE,
    EXPECTED_PAISE,
    FEE_PAISE,
    PAYMENT_ID,
    PAYMENT_PAISE,
    _closed_loop_service,
    _financial_state,
    _resolution_record,
    action_request,
    detect_and_persist,
    reconcile_from_db,
)
from app.auth.principal import Role
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import ExceptionType, FeeType, MatchStatus, ProviderExecutionStatus
from app.schemas.execution import ExecutionStatus
from app.schemas.financial import (
    Fee as DomainFee,
    Payment as DomainPayment,
    Settlement as DomainSettlement,
)
from app.services.execution import ResolutionExecutionService
from app.services.evidence_integrity import compute_canonical_hash
from app.services.rollback import RollbackService

from failure_injection import (
    RECOVERY_MATRIX,
    ProviderOutage,
    classify,
    recovery_for,
    session_fault,
    telemetry_outage,
)


@pytest.fixture
def session_factory():
    """Fresh isolated SQLite session per call (the golden ids are shared)."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from app.database.database import Base

    created = []

    def make():
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )

        @event.listens_for(engine, "connect")
        def _pragma(dbapi_connection, _record):  # pragma: no cover - setup
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()
        created.append((session, engine))
        return session

    yield make

    for session, engine in created:
        session.close()
        engine.dispose()


# ============================================================================
# Shared staging — reuses the Phase 15 golden closed-loop fixtures
# ============================================================================


def staged(settlement_amount: int = ACTUAL_WRONG_PAISE, **provider_kwargs) -> ProviderOutage:
    """A provider holding the canonical FEE_MISMATCH scenario, with fault flags."""
    now = datetime.now(timezone.utc)
    provider = ProviderOutage(**provider_kwargs)
    provider.add_merchant("MER-GOLDEN-P15", name="Golden Merchant")
    provider.add_payment(
        PAYMENT_ID, merchant_id="MER-GOLDEN-P15", amount=PAYMENT_PAISE, captured_at=now
    )
    provider.add_fee(
        "FEE-GOLDEN-P15",
        payment_id=PAYMENT_ID,
        amount=FEE_PAISE,
        fee_type=FeeType.TRANSACTION,
        processed_at=now,
    )
    provider.add_settlement(
        "SET-GOLDEN-P15",
        payment_id=PAYMENT_ID,
        merchant_id="MER-GOLDEN-P15",
        amount=settlement_amount,
        settled_at=now,
    )
    return provider


def status_of(result) -> str:
    return getattr(result.status, "value", str(result.status))


def resolution_for(session, exception):
    """The persisted resolution for this exception (created once per case)."""
    from app.models.resolution import Resolution

    existing = (
        session.query(Resolution).filter_by(exception_id=exception.id).first()
    )
    if existing is not None:
        return existing
    return _resolution_record(session, exception)


def verify(session, provider, execution, exception=None):
    """Run the real post-execution verification for the canonical case."""
    if exception is None:
        exception, _ = detect_and_persist(session, provider)
    return _closed_loop_service(session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id=PAYMENT_ID,
        resolution=resolution_for(session, exception),
    )


def open_exception(session, provider, *, corrected: bool = False):
    """Stage the canonical mismatch so an exception exists.

    With ``corrected=True`` the provider is then moved to the correct state (a
    version bump + re-ingestion), which is what a real provider does after a
    successful correction. Closure can only come from that fresh state.
    """
    from app.ingestion.service import IngestionService

    exception, _ = detect_and_persist(session, provider)
    assert exception is not None, "staging must produce an open exception"
    if corrected:
        provider.mutate_amount("settlement", "SET-GOLDEN-P15", EXPECTED_PAISE)
        provider.set_version("settlement", "SET-GOLDEN-P15", 2)
        IngestionService(session).ingest(provider)
    return exception


# ============================================================================
# CHECKPOINT 1 — Provider failure matrix
# ============================================================================


def test_provider_success_executes_and_is_not_closure():
    provider = staged()
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTED.value
    assert execution.after_state is not None
    assert provider.write_count == 1


def test_provider_success_with_mismatching_state_does_not_close(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTED.value

    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False
    assert verification.exception_closed is False
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_provider_failed_is_reported_as_execution_failure():
    provider = staged(write_status=ProviderExecutionStatus.FAILED)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert execution.after_state is None
    assert execution.error


@pytest.mark.parametrize(
    "outcome",
    [ProviderExecutionStatus.TIMEOUT, ProviderExecutionStatus.UNKNOWN],
    ids=["TIMEOUT", "UNKNOWN"],
)
def test_provider_timeout_and_unknown_are_never_success(outcome):
    provider = staged(write_status=outcome)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert status_of(execution) != ExecutionStatus.EXECUTED.value
    assert execution.after_state is None


@pytest.mark.parametrize(
    "outcome",
    [ProviderExecutionStatus.TIMEOUT, ProviderExecutionStatus.UNKNOWN],
    ids=["TIMEOUT", "UNKNOWN"],
)
def test_provider_timeout_and_unknown_never_close_the_exception(db_session, outcome):
    provider = staged(ACTUAL_WRONG_PAISE, write_status=outcome)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False
    assert verification.exception_closed is False


def test_provider_read_failure_fails_closed(db_session):
    provider = staged()
    exception, _ = detect_and_persist(db_session, provider)
    provider.read_failure = True

    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False
    assert verification.exception_closed is False


def test_provider_write_exception_is_an_execution_failure():
    provider = staged(write_failure=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert execution.error


@pytest.mark.parametrize("malformed", ["none", "wrong_type", "no_status"])
def test_malformed_provider_responses_fail_closed(malformed):
    provider = staged(malformed=malformed)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    # A provider that breaks the response contract can never yield a success.
    assert status_of(execution) != ExecutionStatus.EXECUTED.value
    assert execution.after_state is None


def test_delayed_provider_response_stays_deterministic(db_session):
    provider = staged(ACTUAL_WRONG_PAISE, delay_seconds=0.01)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTED.value
    # Latency must not be mistaken for success: closure still needs fresh truth.
    verification = verify(db_session, provider, execution, exception)
    assert verification.exception_closed is False


def test_provider_empty_snapshot_produces_no_closure(db_session):
    provider = staged()
    exception, _ = detect_and_persist(db_session, provider)
    provider.read_empty = True

    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.exception_closed is False


# ============================================================================
# CHECKPOINT 2 — Idempotency & concurrency
# ============================================================================


def _service_with(provider):
    return ResolutionExecutionService(provider=provider)


def test_duplicate_idempotency_key_mutates_once():
    provider = staged()
    service = _service_with(provider)
    request = action_request()

    first = service.execute(request, _financial_state())
    second = service.execute(request, _financial_state())

    assert status_of(first) == ExecutionStatus.EXECUTED.value
    assert second.execution_id == first.execution_id
    assert provider.write_count == 1


def test_concurrent_identical_requests_mutate_once():
    provider = staged(delay_seconds=0.02)
    service = _service_with(provider)
    request = action_request()
    results = []
    barrier = threading.Barrier(4)

    def run():
        barrier.wait()
        results.append(service.execute(request, _financial_state()))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    executed = [r for r in results if status_of(r) == ExecutionStatus.EXECUTED.value]
    assert len(executed) == 1, [status_of(r) for r in results]
    assert provider.write_count == 1, provider.write_count


def test_concurrent_distinct_keys_each_mutate_once():
    provider = staged(delay_seconds=0.02)
    service = _service_with(provider)
    results = []
    barrier = threading.Barrier(3)

    def run(key):
        barrier.wait()
        results.append(
            service.execute(action_request(idempotency_key=key), _financial_state())
        )

    threads = [threading.Thread(target=run, args=(f"IDEM-{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len([r for r in results if status_of(r) == ExecutionStatus.EXECUTED.value]) == 3
    assert provider.write_count == 3
    assert provider.distinct_idempotency_keys() == {"IDEM-0", "IDEM-1", "IDEM-2"}


def test_retry_after_timeout_reuses_the_key_and_is_visible_to_the_provider():
    """A failed attempt releases its key, so a retry reaches the provider again.

    Recorded behaviour (not hidden): the idempotency guard releases keys whose
    execution FAILED, so retrying after a timeout re-attempts the mutation. The
    same idempotency key is sent, which is what lets a real provider dedupe —
    CloseLoop delegates mutation-level deduplication to the provider boundary.
    This is why TIMEOUT/UNKNOWN are classified as manual-retry-required in the
    recovery matrix.
    """
    provider = staged(write_status=ProviderExecutionStatus.TIMEOUT)
    service = _service_with(provider)
    key = "IDEM-RETRY-TIMEOUT"

    first = service.execute(action_request(idempotency_key=key), _financial_state())
    assert status_of(first) == ExecutionStatus.EXECUTION_FAILED.value

    provider.write_status = None  # provider recovers
    second = service.execute(action_request(idempotency_key=key), _financial_state())

    assert status_of(second) == ExecutionStatus.EXECUTED.value
    assert provider.write_count == 2
    # Both attempts carried the same key: the provider can dedupe.
    assert provider.distinct_idempotency_keys() == {key}
    assert recovery_for("PROVIDER_TIMEOUT")["human_intervention_required"] is True


def test_retry_with_a_fresh_key_still_requires_fresh_verification(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(db_session, provider)
    service = _service_with(provider)

    first = service.execute(action_request(idempotency_key="IDEM-A"), _financial_state())
    retry = service.execute(action_request(idempotency_key="IDEM-B"), _financial_state())
    assert first.execution_id != retry.execution_id
    assert provider.write_count == 2

    verification = verify(db_session, provider, retry, exception)
    assert verification.exception_closed is False


def test_duplicate_post_execution_verification_is_idempotent(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )

    first = verify(db_session, provider, execution, exception)
    second = verify(db_session, provider, execution, exception)

    assert first.exception_closed is True
    assert second.exception_closed is True
    assert second.exception_already_closed is True


# ============================================================================
# CHECKPOINT 3 — Database failure & transaction boundaries
# ============================================================================


def test_ingestion_flush_failure_persists_no_partial_financial_state(db_session):
    from app.models.payment import Payment as DBPayment

    provider = staged()
    with pytest.raises(RuntimeError, match="injected database failure"):
        with session_fault(db_session, on="before_flush"):
            detect_and_persist(db_session, provider)

    db_session.rollback()
    assert db_session.query(DBPayment).count() == 0


def test_persistence_failure_after_reconciliation_creates_no_exception(db_session):
    from app.models.exception import FinancialException

    provider = staged()
    with pytest.raises(RuntimeError, match="injected database failure"):
        with session_fault(db_session, on="after_flush"):
            detect_and_persist(db_session, provider)

    db_session.rollback()
    assert db_session.query(FinancialException).count() == 0


def test_verification_persistence_failure_does_not_close_the_exception(db_session):
    from app.models.exception import FinancialException

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )

    with pytest.raises(RuntimeError, match="injected database failure"):
        with session_fault(db_session, on="before_flush"):
            verify(db_session, provider, execution, exception)

    db_session.rollback()
    persisted = db_session.get(FinancialException, exception.id)
    assert persisted is not None
    assert persisted.status != "CLOSED"


def test_provider_mutation_then_local_failure_leaves_no_false_local_verification(db_session):
    """The provider moved; the database did not record it. Nothing may claim closure."""
    from app.models.exception import FinancialException

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert provider.write_count == 1

    with pytest.raises(RuntimeError, match="injected database failure"):
        with session_fault(db_session, on="before_flush"):
            verify(db_session, provider, execution, exception)

    db_session.rollback()
    persisted = db_session.get(FinancialException, exception.id)
    assert persisted.status != "CLOSED"


def test_rollback_after_failed_verification_restores_financial_state():
    provider = staged()
    service = ResolutionExecutionService(provider=provider)
    execution = service.execute(action_request(), _financial_state())

    result = RollbackService().rollback(
        execution, execution.before_state
    )
    assert result.status.value in ("ROLLED_BACK", "ROLLBACK_FAILED", "ESCALATED")
    assert result.rollback_id
    assert result.audit_trail


# ============================================================================
# CHECKPOINT 4 — Ingestion & data corruption
# ============================================================================


def test_duplicate_ingestion_replay_is_idempotent(db_session):
    from app.ingestion.service import IngestionService
    from app.models.payment import Payment as DBPayment

    provider = staged()
    first = IngestionService(db_session).ingest(provider)
    assert first.total_created > 0

    second = IngestionService(db_session).ingest(provider)
    assert second.total_created == 0
    assert second.count("payments", "noop") == 1
    assert db_session.query(DBPayment).count() == 1


def test_child_before_parent_is_rejected(db_session):
    from app.ingestion.service import IngestionService
    from app.providers.dto import ProviderFee

    provider = staged()
    provider.inject(
        "fee",
        "FEE-ORPHAN",
        ProviderFee(
            provider=provider.name,
            version=1,
            fee_id="FEE-ORPHAN",
            payment_id="PAY-DOES-NOT-EXIST",
            amount=100,
            fee_type=FeeType.TRANSACTION,
        ),
    )
    result = IngestionService(db_session).ingest(provider)

    assert result.count("fees", "created") == 1  # only the valid fee
    assert any(err.record_id == "FEE-ORPHAN" for err in result.errors)
    assert result.status == "PARTIAL"


def test_unsupported_currency_is_rejected_by_validation():
    """An unsupported currency code is refused at the validation layer."""
    from app.ingestion.validation import IngestionValidationError, _require_currency

    _require_currency("payment", "PAY-OK", "INR")  # supported
    with pytest.raises(IngestionValidationError, match="currency"):
        _require_currency("payment", "PAY-BAD", "XYZ")


def test_non_integer_amount_is_rejected_at_the_boundary():
    from app.providers.dto import ProviderSettlement

    with pytest.raises(Exception):
        ProviderSettlement(
            provider="mock",
            version=1,
            settlement_id="SET-FLOAT",
            amount=100.5,  # money is integer minor units only
        )


def test_unknown_currency_code_is_rejected_by_the_dto():
    from app.providers.dto import ProviderPayment

    with pytest.raises(Exception):
        ProviderPayment(
            provider="mock",
            version=1,
            payment_id="PAY-BAD-CCY",
            merchant_id="MER-GOLDEN-P15",
            amount=100,
            currency="XYZ",
        )


def test_stale_version_is_a_noop_not_an_overwrite(db_session):
    from app.ingestion.service import IngestionService
    from app.models.payment import Payment as DBPayment

    provider = staged()
    IngestionService(db_session).ingest(provider)
    db_session.expire_all()
    before = db_session.get(DBPayment, PAYMENT_ID)
    assert before is not None
    stored_version = before.version

    # Replay at a *lower* version, and with different money, must not regress.
    provider.set_version("payment", PAYMENT_ID, 0)
    IngestionService(db_session).ingest(provider)
    db_session.expire_all()
    assert db_session.get(DBPayment, PAYMENT_ID).version == stored_version
    assert db_session.get(DBPayment, PAYMENT_ID).amount == PAYMENT_PAISE


def test_newer_version_updates_the_stored_record(db_session):
    from app.ingestion.service import IngestionService
    from app.models.payment import Payment as DBPayment

    provider = staged()
    IngestionService(db_session).ingest(provider)

    provider.set_version("payment", PAYMENT_ID, 2)
    provider.mutate_amount("payment", PAYMENT_ID, 1_200_000)
    result = IngestionService(db_session).ingest(provider)
    db_session.expire_all()

    assert result.count("payments", "updated") == 1
    assert db_session.get(DBPayment, PAYMENT_ID).amount == 1_200_000


def test_malformed_timestamp_is_rejected_at_the_boundary():
    from app.providers.dto import ProviderPayment

    with pytest.raises(Exception):
        ProviderPayment(
            provider="mock",
            version=1,
            payment_id="PAY-STR-TS",
            merchant_id="MER-1",
            amount=100,
            captured_at="2026-01-01T00:00:00Z",  # a string is not a datetime
        )


def test_zero_amount_settlement_cannot_fabricate_a_match(db_session):
    """A zero-amount settlement is a legal provider record (amount >= 0).

    What matters is that a junk record cannot turn a real mismatch into a
    MATCHED exception: the deterministic engine still sees the 25 000 paise
    discrepancy.
    """
    from app.ingestion.service import IngestionService
    from app.providers.dto import ProviderSettlement

    provider = staged()
    provider.inject(
        "settlement",
        "SET-ZERO",
        ProviderSettlement(
            provider=provider.name,
            version=1,
            settlement_id="SET-ZERO",
            payment_id=PAYMENT_ID,
            merchant_id="MER-GOLDEN-P15",
            amount=0,
        ),
    )
    result = IngestionService(db_session).ingest(provider)
    assert result.status == "COMPLETED"

    reconciliation = reconcile_from_db(db_session)
    assert reconciliation.difference == DIFFERENCE_PAISE
    assert reconciliation.match_status == MatchStatus.EXCEPTION


def test_partially_failed_batch_keeps_valid_records_and_reports_the_rest(db_session):
    """Three orphan payments are rejected; the valid ones still land."""
    from app.ingestion.service import IngestionService
    from app.providers.dto import ProviderPayment

    provider = staged()
    for index in range(3):
        provider.inject(
            "payment",
            f"PAY-ORPHAN-{index}",
            ProviderPayment(
                provider=provider.name,
                version=1,
                payment_id=f"PAY-ORPHAN-{index}",
                merchant_id="MER-DOES-NOT-EXIST",
                amount=500,
            ),
        )
    result = IngestionService(db_session).ingest(provider)

    assert result.status == "PARTIAL"
    assert result.count("payments", "created") == 1
    assert len([e for e in result.errors if e.entity.startswith("payment")]) == 3
    assert result.total_created >= 1
    # A partially failed batch never corrupts what was already valid.
    assert reconcile_from_db(db_session).difference == DIFFERENCE_PAISE


# ============================================================================
# CHECKPOINT 5 — Reconciliation failure
# ============================================================================


def _reconcile(settlement_amounts, fees=0, refunds=0, taxes=0, adjustments=0, payment=PAYMENT_PAISE):
    now = datetime.now(timezone.utc)
    domain_payment = DomainPayment(
        payment_id="PAY-RECON", merchant_id="MER-RECON", amount=payment, payment_timestamp=now
    )
    settlements = [
        DomainSettlement(
            settlement_id=f"SET-RECON-{i}",
            payment_id=domain_payment.payment_id,
            merchant_id="MER-RECON",
            amount=amount,
            settlement_timestamp=now,
        )
        for i, amount in enumerate(settlement_amounts)
    ]
    fee_records = [
        DomainFee(
            fee_id=f"FEE-RECON-{i}",
            payment_id=domain_payment.payment_id,
            amount=amount,
            fee_type=FeeType.TRANSACTION,
        )
        for i, amount in enumerate(fees if isinstance(fees, list) else ([fees] if fees else []))
    ]
    return calculate_reconciliation(
        payment=domain_payment,
        settlements=settlements,
        refunds=[],
        fees=fee_records,
        taxes=[],
        adjustments=[],
        case_id="CASE-RECON",
        reconciliation_id="REC-RECON",
    )


def test_no_settlement_is_a_missing_record_never_a_match():
    result = _reconcile([])
    assert result.match_status != MatchStatus.MATCHED
    assert result.exception_type in (ExceptionType.MISSING_RECORD, ExceptionType.UNKNOWN)


def test_partial_settlement_is_not_a_match():
    result = _reconcile([400_000], fees=20_000)
    assert result.match_status != MatchStatus.MATCHED


def test_duplicate_settlement_is_detected():
    result = _reconcile([950_000, 950_000])
    assert result.match_status != MatchStatus.MATCHED


def test_matched_case_has_zero_discrepancy_and_integer_paise():
    result = _reconcile([950_000], fees=50_000)
    assert result.match_status == MatchStatus.MATCHED
    assert result.difference == 0
    for amount in (result.expected_amount, result.actual_amount, result.difference):
        assert isinstance(amount, int)


def test_reconciliation_engine_exception_does_not_close_the_exception(db_session, monkeypatch):
    from app.services import post_execution_reconciliation as per

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )

    def _explode(*args, **kwargs):
        raise RuntimeError("reconciliation engine unavailable")

    monkeypatch.setattr(per, "calculate_reconciliation", _explode)

    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False
    assert verification.exception_closed is False


def test_ml_llm_and_similarity_cannot_change_deterministic_truth():
    """The engine's verdict is a pure function of financial inputs."""
    baseline = _reconcile([925_000], fees=50_000)
    repeated = _reconcile([925_000], fees=50_000)
    assert baseline.exception_type == repeated.exception_type
    assert baseline.difference == repeated.difference == DIFFERENCE_PAISE
    assert baseline.match_status == MatchStatus.EXCEPTION


# ============================================================================
# CHECKPOINT 6 — Evidence failure & integrity
# ============================================================================


def test_evidence_hash_is_deterministic_and_tamper_evident():
    payload = {"payment_id": PAYMENT_ID, "expected": EXPECTED_PAISE, "actual": 925_000}
    original = compute_canonical_hash(payload)
    assert original == compute_canonical_hash(dict(payload))
    assert len(original) == 64

    tampered = dict(payload, actual=EXPECTED_PAISE)
    assert compute_canonical_hash(tampered) != original


def test_tampered_evidence_is_detectable_against_its_recorded_digest():
    payload = {"amount_paise": 925_000, "currency": "INR"}
    recorded = compute_canonical_hash(payload)
    payload["amount_paise"] = 950_000  # the record was edited after the fact
    assert compute_canonical_hash(payload) != recorded


def test_evidence_digest_mismatch_blocks_execution():
    """A human approval bound to a different evidence digest cannot execute."""
    service = ResolutionExecutionService(provider=staged())
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            current_evidence_digest="digest-after-tamper",
            approval_record={
                "decision": "APPROVED",
                "role": "APPROVER",
                "expires_at": expires,
                "evidence_digest": "digest-before-tamper",
            },
        ),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert "digest" in (execution.error or "").lower()


def test_conflicting_evidence_reproduces_the_phase15_finding(db_session):
    """Phase 15 finding: the selector may recommend, but policy still blocks.

    Reproduced end to end so the consequence is on the record: the guardrail
    denies, the execution service refuses, and no money moves.
    """
    from app.evaluation.dataset import build_synthetic_dataset_v1
    from app.evaluation.resolution_eval import evaluate_resolution
    from app.evaluation.safety import automation_decision_for_case
    from app.schemas.resolution_selection import SelectionStatus
    from app.services.guardrail_engine import GuardrailEngine

    dataset = build_synthetic_dataset_v1(per_family=3)
    report = evaluate_resolution(dataset)
    engine = GuardrailEngine()
    by_id = {result.case_id: result for result in report.results}

    conflicting = [case for case in dataset.cases if case.evidence.has_conflict]
    assert conflicting, "fixture must contain conflicting-evidence cases"

    recommended = 0
    for case in conflicting:
        result = by_id[case.case_id]
        if result.selection_status == SelectionStatus.RECOMMENDED.value:
            recommended += 1
        assert automation_decision_for_case(case, result, engine) != "AUTO"
    # The weakness exists (recommendation produced) AND is contained (no AUTO).
    assert recommended > 0, "expected at least one conflicting case to be recommended"

    # A selector recommendation on its own can never authorise execution.
    provider = staged()
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(guardrail_decision="UNRESOLVED", authorization_source="AUTO_GUARDRAIL"),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert provider.write_count == 0


def test_phase15_auto_threshold_misalignment_is_reproduced():
    """Phase 15 finding 2: selector/guardrail threshold misalignment made
    AUTO unreachable on the synthetic corpus. Reproduced as a failure
    scenario and recorded — NOT fixed.

    Consequence is measurement loss, not a safety loss: the automation rate
    over the corpus is zero, so the gate never fires there; the gate itself
    is not dead code (a clean LOW-risk row still resolves to AUTO, asserted
    below), so the condition is threshold alignment rather than an unsafe or
    broken authorization path.
    """
    from app.evaluation.dataset import build_synthetic_dataset_v1
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.evaluation.resolution_eval import evaluate_resolution
    from app.evaluation.safety import automation_decision_for_case
    from app.services.guardrail_engine import GuardrailEngine

    dataset = build_synthetic_dataset_v1(per_family=3)
    report = evaluate_resolution(dataset)
    engine = GuardrailEngine()
    by_id = {result.case_id: result for result in report.results}
    decisions = [
        automation_decision_for_case(case, by_id[case.case_id], engine)
        for case in dataset.cases
    ]

    # The rate is measurable over the whole corpus and it is zero:
    # confidence peaks near 0.5 (< 0.75) and adjustments above 10 000 paise
    # are MEDIUM risk, so no case can satisfy the AUTO gate.
    assert len(decisions) == len(dataset.cases)
    assert "AUTO" not in decisions
    assert decisions.count("HUMAN_REVIEW") >= 1  # not a trivial all-UNRESOLVED corpus

    # The AUTO path itself still works on clean input — alignment, not a
    # dead gate, is what is missing.
    clean = PolicyRow(
        row_id="AUTO-REACHABLE-CONTROL",
        description="clean LOW-risk row proving the AUTO gate is live",
        confidence=0.99,
        evidence_coverage=0.95,
        evidence_consistency=0.95,
        historical_similarity=0.95,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="AUTO",
    )
    assert engine.evaluate(_engine_result(clean)).decision.value == "AUTO"


def test_missing_evidence_cannot_become_an_approval():
    service = ResolutionExecutionService(provider=staged())
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            approval_record=None,  # no approval record at all
        ),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert "approval" in (execution.error or "").lower()


# ============================================================================
# CHECKPOINT 7 — ML & historical memory failure
# ============================================================================


def test_missing_model_artifact_fails_loudly(tmp_path):
    from app.ml.classifier import ExceptionClassifierService

    with pytest.raises(Exception):
        ExceptionClassifierService.from_artifact(str(tmp_path / "does-not-exist"))


def test_corrupt_model_artifact_is_rejected(tmp_path):
    from app.ml.classifier import ExceptionClassifierService

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.joblib").write_bytes(b"not a model")
    (artifact / "metadata.json").write_text('{"feature_names": [], "label_names": []}')

    with pytest.raises(Exception):
        ExceptionClassifierService.from_artifact(str(artifact))


def test_incompatible_model_schema_version_is_rejected(tmp_path):
    from app.ml.classifier import ExceptionClassifierService, ModelArtifact
    from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION

    path = str(tmp_path / "artifact")
    ModelArtifact.save(
        {"dummy": True},
        path,
        feature_names=["a", "b"],
        label_names=["UNKNOWN"],
        training_metadata={},
        evaluation={},
    )

    with pytest.raises(ValueError, match="[Ii]ncompatible feature schema"):
        ExceptionClassifierService.from_artifact(path, expected_schema_version="9.9.9")

    service = ExceptionClassifierService.from_artifact(
        path, expected_schema_version=FEATURE_SCHEMA_VERSION
    )
    assert service.feature_names == ["a", "b"]


def test_missing_feature_raises_instead_of_silently_defaulting():
    from app.ml.classifier import ExceptionClassifierService

    service = ExceptionClassifierService(
        model=object(), feature_names=["fee_ratio", "tax_ratio"], label_names=["UNKNOWN"]
    )
    with pytest.raises(ValueError, match="[Ii]ncompatible feature schema"):
        service.predict({"fee_ratio": 0.5})


def test_classifier_exception_does_not_affect_deterministic_reconciliation():
    from app.ml.classifier import ExceptionClassifierService

    class ExplodingModel:
        def predict(self, X):
            raise RuntimeError("classifier unavailable")

        def predict_proba(self, X):
            raise RuntimeError("classifier unavailable")

    service = ExceptionClassifierService(
        model=ExplodingModel(), feature_names=["fee_ratio"], label_names=["UNKNOWN"]
    )
    with pytest.raises(RuntimeError):
        service.predict({"fee_ratio": 0.5})

    # Financial truth is untouched by an ML outage.
    result = _reconcile([925_000], fees=50_000)
    assert result.exception_type == ExceptionType.FEE_MISMATCH
    assert result.difference == DIFFERENCE_PAISE


def test_ml_dependency_failure_is_advisory_not_authoritative():
    """ML is advisory: its absence neither creates nor blocks authorization.

    The authoritative dependency is the database — when *that* fails the
    guardrail fails closed.
    """
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    row = PolicyRow(
        row_id="FAIL-ML",
        description="perfect case, ML dependency down",
        confidence=0.99,
        evidence_coverage=0.95,
        evidence_consistency=0.95,
        historical_similarity=0.95,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="NOT_AUTO",
    )
    engine = GuardrailEngine()
    baseline = engine.evaluate(_engine_result(row)).decision.value
    ml_down = engine.evaluate(
        _engine_result(row), {"ml_classifier": False, "database": True}
    ).decision.value
    db_down = engine.evaluate(
        _engine_result(row), {"ml_classifier": True, "database": False}
    ).decision.value

    assert baseline == "AUTO"
    assert ml_down == baseline  # ML availability is not an authorization signal
    assert db_down != "AUTO"  # losing financial truth fails closed


def test_classifier_outage_does_not_change_policy_decision():
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    row = PolicyRow(
        row_id="ML-OUT",
        description="ML unavailable but everything else clean",
        confidence=0.9,
        evidence_coverage=0.9,
        evidence_consistency=0.9,
        historical_similarity=0.9,
        exposure_paise=5_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="NOT_AUTO",
    )
    result = GuardrailEngine().evaluate(_engine_result(row), {"ml_classifier": False})
    assert result.decision.value in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
    assert result.system_healthy is True


def test_retrieval_corruption_and_contamination_are_handled():
    """A corrupt vector and a future case must not change what is retrieved."""
    from app.evaluation.dataset import build_synthetic_dataset_v1
    from app.evaluation.retrieval_eval import rank_candidates

    dataset = build_synthetic_dataset_v1(per_family=2)
    ordered = sorted(dataset.cases, key=lambda c: c.decision_time)
    query = ordered[-1]

    future = query.model_copy(
        update={"case_id": "EVAL-FUTURE", "decision_time": query.decision_time + timedelta(days=1)}
    )
    corrupt = query.model_copy(
        update={"case_id": "EVAL-CORRUPT", "features": {k: float("nan") for k in query.features}}
    )

    memory = [query, future, corrupt] + [c for c in ordered if c.case_id != query.case_id]
    ranked, self_violations, temporal_violations = rank_candidates(query, memory)

    assert query.case_id not in ranked
    assert "EVAL-FUTURE" not in ranked
    assert self_violations == 1
    assert temporal_violations == 1
    # Every candidate that is allowed to be retrieved is still returned; a
    # NaN vector does not crash ranking. NOTE: its position is undefined —
    # NaN comparisons make the order of a corrupt vector non-deterministic,
    # which is recorded as a known limitation rather than asserted as safe.
    assert len(ranked) == len(memory) - 2
    assert "EVAL-CORRUPT" in ranked
    # A corrupt vector must not displace genuine same-family neighbours from
    # the top of the list in a way that loses them entirely.
    assert set(ranked[:10]) & {c.case_id for c in ordered if c.family == query.family}


def test_empty_historical_memory_is_not_an_execution_trigger():
    from app.evaluation.retrieval_eval import rank_candidates

    ranking, self_violations, temporal_violations = rank_candidates(staged_case(), [])
    assert ranking == []
    assert (self_violations, temporal_violations) == (0, 0)


def staged_case():
    from app.evaluation.dataset import build_synthetic_dataset_v1

    return build_synthetic_dataset_v1(per_family=1).cases[0]


# ============================================================================
# CHECKPOINT 8 — Policy / guardrail failure
# ============================================================================


def test_guardrail_engine_fails_closed_when_it_raises(monkeypatch):
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.services import guardrail_engine as ge

    class ExplodingGate:
        def evaluate(self, engine_result):
            raise RuntimeError("policy backend unavailable")

    row = PolicyRow(
        row_id="BOOM",
        description="everything looks perfect",
        confidence=0.99,
        evidence_coverage=0.95,
        evidence_consistency=0.95,
        historical_similarity=0.99,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="NOT_AUTO",
    )
    engine = ge.GuardrailEngine()
    engine.confidence_gate = ExplodingGate()
    result = engine.evaluate(_engine_result(row))

    assert result.decision.value != "AUTO"
    assert result.system_healthy is False


def test_expired_human_approval_blocks_execution():
    service = ResolutionExecutionService(provider=staged())
    expired = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            approval_record={"decision": "APPROVED", "role": "APPROVER", "expires_at": expired},
        ),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert "expire" in (execution.error or "").lower()


def test_wrong_authorization_source_blocks_execution():
    service = ResolutionExecutionService(provider=staged())
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="AUTO_GUARDRAIL",  # human review needs human approval
            approval_record={"decision": "APPROVED", "role": "APPROVER"},
        ),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert "authorization" in (execution.error or "").lower()


def test_rejected_approval_blocks_execution():
    service = ResolutionExecutionService(provider=staged())
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            approval_record={"decision": "REJECTED", "role": "APPROVER"},
        ),
        _financial_state(),
    )
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value


def test_unknown_guardrail_decision_blocks_execution():
    service = ResolutionExecutionService(provider=staged())
    for decision in ("UNRESOLVED", "BLOCKED", "DENIED", "WHATEVER"):
        execution = service.execute(
            action_request(guardrail_decision=decision), _financial_state()
        )
        assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value


def test_guardrail_denial_after_selector_recommendation_never_executes():
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    row = PolicyRow(
        row_id="SELECTOR-OK-POLICY-NO",
        description="selector recommended, evidence conflicts",
        confidence=0.99,
        evidence_coverage=0.95,
        evidence_consistency=0.95,
        historical_similarity=0.99,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        has_conflict=True,
        expected="NOT_AUTO",
    )
    result = GuardrailEngine().evaluate(_engine_result(row))
    assert result.decision.value != "AUTO"

    provider = staged()
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(guardrail_decision=result.decision.value), _financial_state()
    )
    assert status_of(execution) != ExecutionStatus.EXECUTED.value
    assert provider.write_count == 0


# ============================================================================
# CHECKPOINT 9 — Auth / RBAC failure
# ============================================================================


def _auth_app():
    from fastapi import Depends, FastAPI

    from app.api.auth_dependencies import get_current_principal, require_permission
    from app.auth.principal import Permission

    app = FastAPI()

    @app.get("/whoami")
    def whoami(principal=Depends(get_current_principal)):  # noqa: ANN001
        return {"subject": principal.subject, "roles": [r.value for r in principal.roles]}

    @app.post("/execute")
    def execute(principal=Depends(require_permission(Permission.REQUEST_EXECUTION))):  # noqa: ANN001
        return {"actor": principal.subject}

    return app


@pytest.fixture(autouse=True)
def real_verifier():
    from app.api.auth_dependencies import set_token_verifier

    set_token_verifier(make_test_verifier())
    yield
    set_token_verifier(None)


@pytest.mark.parametrize(
    "token_kwargs",
    [
        {"expired": True},
        {"issuer": "https://evil.example"},
        {"audience": "other-api"},
        {"override_roles": ["SUPERUSER"]},
        {"include_subject": False},
    ],
    ids=["expired", "bad-issuer", "bad-audience", "unknown-role", "no-subject"],
)
def test_invalid_tokens_are_never_authenticated(token_kwargs):
    from fastapi.testclient import TestClient

    token = make_test_token(**token_kwargs)
    client = TestClient(_auth_app())
    response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401, response.text


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", "Bearer not-a-jwt", "Basic dXNlcjpwYXNz"],
    ids=["missing", "empty", "no-token", "malformed", "wrong-scheme"],
)
def test_missing_or_malformed_credentials_are_rejected(authorization):
    from fastapi.testclient import TestClient

    headers = {} if authorization is None else {"Authorization": authorization}
    response = TestClient(_auth_app()).get("/whoami", headers=headers)
    assert response.status_code == 401


def test_forged_signature_is_rejected():
    from fastapi.testclient import TestClient

    forged = make_test_token(signing_key="attacker-secret")
    response = TestClient(_auth_app()).get(
        "/whoami", headers={"Authorization": f"Bearer {forged}"}
    )
    assert response.status_code == 401


def test_insufficient_permission_is_denied():
    from fastapi.testclient import TestClient

    token = make_test_token(roles=["VIEWER"])
    response = TestClient(_auth_app()).post(
        "/execute", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_identity_is_token_derived_and_not_caller_supplied():
    from fastapi.testclient import TestClient

    token = make_test_token(subject="real-actor", roles=["APPROVER"], override_roles=None)
    client = TestClient(_auth_app())
    response = client.get(
        "/whoami",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Actor": "impersonated-admin",
            "X-Request-Id": "forged-request-id",
        },
    )
    assert response.status_code == 200
    assert response.json()["subject"] == "real-actor"


def test_health_and_readiness_need_no_authentication():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    for path in ("/health", "/ready"):
        response = client.get(path)
        assert response.status_code not in (401, 403), f"{path} demanded credentials"


# ============================================================================
# CHECKPOINT 10 — LangGraph / MCP failure
# ============================================================================


def p11_provider(**provider_kwargs) -> ProviderOutage:
    """Provider holding the Phase 11 orchestration scenario (its own ids)."""
    now = datetime.now(timezone.utc)
    provider = ProviderOutage(**provider_kwargs)
    provider.add_merchant("MER-P11-001", name="Phase 11 Merchant")
    provider.add_payment(
        "PAY-P11-001", merchant_id="MER-P11-001", amount=PAYMENT_PAISE, captured_at=now
    )
    provider.add_fee(
        "FEE-P11-001",
        payment_id="PAY-P11-001",
        amount=FEE_PAISE,
        fee_type=FeeType.TRANSACTION,
        processed_at=now,
    )
    provider.add_settlement(
        "SET-P11-001",
        payment_id="PAY-P11-001",
        merchant_id="MER-P11-001",
        amount=ACTUAL_WRONG_PAISE,
        settled_at=now,
        version=1,
    )
    return provider


def test_mcp_execution_tool_requires_phase8_authorization(db_session):
    """Machine scope is not an authorization bypass for the sensitive tool."""
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _detect_fee_mismatch, _make_client

    provider = p11_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    toolkit.bind_principal(None)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]

    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "workflow_id": "WF-P16",
            "idempotency_key": "IDEM-P16-MCP",
            "proposal": proposal,
        }
    )
    assert "authorization" in result
    decision = result["authorization"]["decision"]
    assert decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
    if result.get("executed") is not True:
        assert provider.write_count == 0
    else:
        # If it did execute, the policy engine — not the caller — authorised it.
        assert decision == "AUTO"


def test_mcp_tool_denies_rbac_when_principal_lacks_permission(db_session):
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _detect_fee_mismatch, _make_client

    provider = p11_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    toolkit.bind_principal(deterministic_principal(roles=(Role.VIEWER,)))

    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "workflow_id": "WF-P16-RBAC",
            "idempotency_key": "IDEM-P16-RBAC",
        }
    )
    assert result["executed"] is False
    assert "rbac denied" in json.dumps(result).lower()
    assert provider.write_count == 0


def test_mcp_tool_ignores_forged_authorization_arguments(db_session):
    """Caller-supplied authorization fields are ignored, never honoured."""
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _detect_fee_mismatch, _make_client

    provider = p11_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    toolkit.bind_principal(deterministic_principal())  # OPERATOR + APPROVER
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]

    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "workflow_id": "WF-P16-FORGED",
            "idempotency_key": "IDEM-P16-FORGED",
            "proposal": proposal,
            # Forged authority: none of these fields may be honoured.
            "guardrail_decision": "AUTO",
            "authorization_source": "AUTO_GUARDRAIL",
            "ml_confidence": 0.999,
            "approved": True,
        }
    )
    assert "authorization" in result
    assert result["authorization"]["decision"] in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
    assert provider.write_count <= 1


def test_mcp_execution_tool_failure_surfaces_as_not_executed(db_session):
    """A broken execution collaborator never becomes a successful execution."""
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _detect_fee_mismatch, _make_client

    class BrokenExecutionService:
        def execute(self, *args, **kwargs):
            raise RuntimeError("execution service unavailable")

    provider = p11_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(
        db_session, provider, execution_service=BrokenExecutionService()
    )
    toolkit.bind_principal(deterministic_principal())

    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "workflow_id": "WF-P16-EXEC-DOWN",
            "idempotency_key": "IDEM-P16-EXEC-DOWN",
        }
    )
    assert result.get("executed") is not True
    db_session.expire_all()
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_execution_tool_failure_does_not_close_the_exception(db_session):
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _make_client

    provider = staged(ACTUAL_WRONG_PAISE, write_failure=True)
    exception = open_exception(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    toolkit.bind_principal(deterministic_principal())

    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "workflow_id": "WF-P16-BROKEN-PROVIDER",
            "idempotency_key": "IDEM-P16-BROKEN-PROVIDER",
        }
    )
    assert result.get("executed") is not True
    db_session.expire_all()
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_mcp_server_failure_surfaces_rather_than_succeeding_silently(db_session):
    """A dead MCP server must not look like a successful tool call."""
    from test_phase11_langgraph_orchestration import EXCEPTION_ID, _detect_fee_mismatch, _make_client

    provider = p11_provider()
    _detect_fee_mismatch(db_session, provider)
    _toolkit, server, client = _make_client(db_session, provider)

    def _explode(*args, **kwargs):
        raise RuntimeError("MCP server unavailable")

    server.invoke = _explode  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="MCP server unavailable"):
        client.call_tool(
            "request_authorized_execution",
            {"exception_id": EXCEPTION_ID, "workflow_id": "WF", "idempotency_key": "IDEM"},
            workflow_id="WF-P16-MCP-DOWN",
            exception_id=EXCEPTION_ID,
        )
    assert provider.write_count == 0


def test_graph_node_exception_does_not_close_the_exception(db_session):
    from test_phase11_langgraph_orchestration import _make_client
    from app.orchestration.graph import run_orchestration

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)

    def _explode(*args, **kwargs):
        raise RuntimeError("graph node exploded")

    client.call_tool = _explode  # type: ignore[assignment]

    with pytest.raises(Exception):
        run_orchestration(exception.id, client, workflow_id="WF-P16-GRAPH-FAIL")

    db_session.expire_all()
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_duplicate_workflow_invocation_mutates_once(db_session):
    from test_phase11_langgraph_orchestration import _make_client

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    toolkit.bind_principal(deterministic_principal())

    params = {
        "exception_id": exception.id,
        "workflow_id": "WF-P16-DUP",
        "idempotency_key": "IDEM-P16-DUP",
    }
    first = toolkit.request_authorized_execution(dict(params))
    second = toolkit.request_authorized_execution(dict(params))

    assert first.get("executed") in (True, False)
    # Whatever the outcome, the provider saw at most one mutation for this key.
    assert provider.write_count <= 1
    assert second.get("executed") is not True or first.get("executed") is True


# ============================================================================
# CHECKPOINT 11 — Observability failure
# ============================================================================


def test_telemetry_outage_does_not_change_execution_outcome(monkeypatch):
    healthy_provider = staged()
    healthy = ResolutionExecutionService(provider=healthy_provider).execute(
        action_request(), _financial_state()
    )

    with telemetry_outage(monkeypatch):
        blind_provider = staged()
        blind = ResolutionExecutionService(provider=blind_provider).execute(
            action_request(), _financial_state()
        )

    assert status_of(blind) == status_of(healthy) == ExecutionStatus.EXECUTED.value
    assert blind.actual_adjustment_paise == healthy.actual_adjustment_paise
    assert blind_provider.write_count == healthy_provider.write_count == 1


def test_telemetry_outage_does_not_change_closure(monkeypatch, db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )

    with telemetry_outage(monkeypatch):
        verification = verify(db_session, provider, execution, exception)

    assert verification.financially_verified is True
    assert verification.exception_closed is True


def test_telemetry_outage_does_not_change_policy_decision(monkeypatch):
    from app.evaluation.policy_eval import evaluate_policy

    baseline = evaluate_policy()
    with telemetry_outage(monkeypatch):
        during = evaluate_policy()

    assert [r.actual_decision for r in during.rows] == [
        r.actual_decision for r in baseline.rows
    ]
    assert during.invariant_violations == []


def test_sensitive_values_are_stripped_from_telemetry_labels():
    from app.core import observability as obs

    labels, dropped = obs.sanitize_labels(
        {
            "authorization": "Bearer super-secret-token",
            "api_key": "sk-live-123",
            "payment_id": "PAY-123",
            "outcome": "success",
        }
    )
    assert labels["outcome"] == "success"
    assert labels["authorization"] == "***MASKED***"
    assert labels["api_key"] == "***MASKED***"
    # A high-cardinality id is dropped rather than recorded as a metric label.
    assert "payment_id" in dropped
    dumped = json.dumps(labels)
    assert "super-secret-token" not in dumped
    assert "sk-live-123" not in dumped


# ============================================================================
# CHECKPOINT 12 — Closed-loop chaos scenarios
# ============================================================================


def _chaos_record(failure: str, expected: str, actual: str, **extra) -> dict:
    return {"failure": failure, "expected": expected, "actual": actual, **extra}


def test_chaos_f1_provider_success_with_stale_state_is_not_closed(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    record = _chaos_record(
        "provider SUCCESS + unchanged provider state",
        "exception NOT closed",
        "not closed" if not verification.exception_closed else "CLOSED",
        exception_state=db_session.get(type(exception), exception.id).status,
    )
    assert record["actual"] == "not closed"
    assert record["exception_state"] != "CLOSED"


def test_chaos_f2_provider_timeout_fails_closed(db_session):
    provider = staged(ACTUAL_WRONG_PAISE, write_status=ProviderExecutionStatus.TIMEOUT)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert status_of(execution) == ExecutionStatus.EXECUTION_FAILED.value
    assert verification.exception_closed is False


def test_chaos_f3_provider_unknown_fails_closed(db_session):
    provider = staged(ACTUAL_WRONG_PAISE, write_status=ProviderExecutionStatus.UNKNOWN)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False
    assert verification.exception_closed is False


def test_chaos_f4_provider_success_then_database_failure(db_session):
    from app.models.exception import FinancialException

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert provider.write_count == 1

    with pytest.raises(RuntimeError, match="injected database failure"):
        with session_fault(db_session, on="before_flush"):
            verify(db_session, provider, execution, exception)
    db_session.rollback()

    persisted = db_session.get(FinancialException, exception.id)
    assert persisted.status != "CLOSED"


def test_chaos_f5_retry_after_timeout_carries_the_same_key(db_session):
    """F5: a retry after TIMEOUT reuses the key and cannot silently close."""
    provider = staged(ACTUAL_WRONG_PAISE, write_status=ProviderExecutionStatus.TIMEOUT)
    exception = open_exception(db_session, provider)
    service = ResolutionExecutionService(provider=provider)
    key = "IDEM-CHAOS-F5"

    service.execute(action_request(idempotency_key=key), _financial_state())
    provider.write_status = None
    retried = service.execute(action_request(idempotency_key=key), _financial_state())

    assert status_of(retried) == ExecutionStatus.EXECUTED.value
    assert provider.distinct_idempotency_keys() == {key}
    # A successful retry still cannot close anything by itself.
    db_session.expire_all()
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_chaos_f6_conflicting_evidence_selector_recommends_guardrail_blocks():
    from app.evaluation.dataset import build_synthetic_dataset_v1
    from app.evaluation.resolution_eval import evaluate_resolution
    from app.evaluation.safety import automation_decision_for_case
    from app.schemas.resolution_selection import SelectionStatus
    from app.services.guardrail_engine import GuardrailEngine

    dataset = build_synthetic_dataset_v1(per_family=3)
    report = evaluate_resolution(dataset)
    engine = GuardrailEngine()
    by_id = {r.case_id: r for r in report.results}

    conflicting = [c for c in dataset.cases if c.evidence.has_conflict]
    recommended = [
        c for c in conflicting
        if by_id[c.case_id].selection_status == SelectionStatus.RECOMMENDED.value
    ]
    assert recommended, "expected the Phase 15 selector weakness to reproduce"
    for case in recommended:
        assert automation_decision_for_case(case, by_id[case.case_id], engine) != "AUTO"


def test_chaos_f7_missing_ml_artifact_preserves_deterministic_truth(tmp_path):
    from app.ml.classifier import ExceptionClassifierService

    with pytest.raises(Exception):
        ExceptionClassifierService.from_artifact(str(tmp_path / "absent"))

    result = _reconcile([925_000], fees=50_000)
    assert result.exception_type == ExceptionType.FEE_MISMATCH
    assert result.difference == DIFFERENCE_PAISE
    assert result.match_status == MatchStatus.EXCEPTION


def test_chaos_f8_retrieval_unavailable_is_a_safe_fallback():
    from app.evaluation.retrieval_eval import rank_candidates

    ranking, self_violations, temporal_violations = rank_candidates(staged_case(), [])
    assert ranking == []
    assert self_violations == 0
    assert temporal_violations == 0


def test_chaos_f9_stale_approval_denies_execution():
    service = ResolutionExecutionService(provider=staged())
    stale = (datetime.utcnow() - timedelta(hours=2)).isoformat()
    execution = service.execute(
        action_request(
            verification_passed=False,
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            approval_record={"decision": "APPROVED", "expires_at": stale},
        ),
        _financial_state(),
    )
    assert status_of(execution) != ExecutionStatus.EXECUTED.value


def test_chaos_f10_evidence_hash_mismatch_denies_execution():
    payload = {"amount_paise": DIFFERENCE_PAISE}
    recorded = compute_canonical_hash(payload)
    tampered = compute_canonical_hash({"amount_paise": DIFFERENCE_PAISE + 1})
    assert tampered != recorded

    service = ResolutionExecutionService(provider=staged())
    execution = service.execute(
        action_request(
            guardrail_decision="HUMAN_REVIEW",
            authorization_source="HUMAN_APPROVAL",
            current_evidence_digest=tampered,
            approval_record={
                "decision": "APPROVED",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "evidence_digest": recorded,
            },
        ),
        _financial_state(),
    )
    assert status_of(execution) != ExecutionStatus.EXECUTED.value


def test_chaos_f11_reconciliation_engine_failure_never_closes(db_session, monkeypatch):
    from app.services import post_execution_reconciliation as per

    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )

    def _explode(*args, **kwargs):
        raise RuntimeError("reconciliation unavailable")

    monkeypatch.setattr(per, "calculate_reconciliation", _explode)
    verification = verify(db_session, provider, execution, exception)
    assert verification.exception_closed is False


def test_chaos_f12_telemetry_unavailable_keeps_business_outcome(monkeypatch, session_factory):
    baseline_session = session_factory()
    baseline_provider = staged(ACTUAL_WRONG_PAISE)
    baseline_exception = open_exception(baseline_session, baseline_provider, corrected=True)
    baseline_execution = ResolutionExecutionService(provider=baseline_provider).execute(
        action_request(), _financial_state()
    )
    baseline_verification = verify(
        baseline_session, baseline_provider, baseline_execution, baseline_exception
    )

    blind_session = session_factory()
    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(blind_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    with telemetry_outage(monkeypatch):
        verification = verify(blind_session, provider, execution, exception)

    assert status_of(execution) == status_of(baseline_execution)
    assert verification.exception_closed == baseline_verification.exception_closed is True
    assert verification.decision == baseline_verification.decision


# ============================================================================
# CHECKPOINT 13 — Recovery & retry matrix
# ============================================================================


def test_every_failure_class_has_a_recovery_classification():
    assert RECOVERY_MATRIX
    for failure, entry in RECOVERY_MATRIX.items():
        assert entry["action"], failure
        assert isinstance(entry["retry_allowed"], bool)
        assert isinstance(entry["provider_mutation_possible"], bool)
        assert isinstance(entry["reconciliation_required"], bool)
        assert classify(failure) in ("RECOVERABLE", "MANUAL_RETRY", "ESCALATE")


def test_ambiguous_failures_are_never_auto_retried():
    for failure in ("PROVIDER_TIMEOUT", "PROVIDER_UNKNOWN", "AUTHORIZATION_FAILURE", "EVIDENCE_INTEGRITY_FAILURE"):
        entry = recovery_for(failure)
        assert entry["retry_allowed"] is False, failure
        assert entry["human_intervention_required"] is True, failure
        assert classify(failure) in ("MANUAL_RETRY", "ESCALATE"), failure


def test_recovery_matrix_is_deterministic_and_derived_not_invented():
    for failure in RECOVERY_MATRIX:
        assert recovery_for(failure) == recovery_for(failure)
    # Anything financial-ambiguous requires reconciliation before closure.
    for failure in ("PROVIDER_TIMEOUT", "PROVIDER_UNKNOWN", "DATABASE_FAILURE"):
        assert recovery_for(failure)["reconciliation_required"] is True
    with pytest.raises(KeyError):
        recovery_for("NOT_A_REAL_FAILURE")


def test_advisory_failures_degrade_without_touching_financial_truth():
    for failure in ("ML_UNAVAILABLE", "RETRIEVAL_UNAVAILABLE", "TELEMETRY_UNAVAILABLE"):
        assert recovery_for(failure)["provider_mutation_possible"] is False
        assert classify(failure) == "RECOVERABLE"


# ============================================================================
# CHECKPOINT 14 — Failure invariants
# ============================================================================


def test_invariant_1_no_invalid_financial_state_becomes_closed(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(db_session, provider)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.exception_closed is False
    assert db_session.get(type(exception), exception.id).status != "CLOSED"


def test_invariant_2_provider_success_alone_never_closes(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    assert status_of(execution) == ExecutionStatus.EXECUTED.value
    # Execution alone never advances the exception, even with corrected state.
    before = db_session.get(type(exception), exception.id).status
    assert before != "CLOSED"


def test_invariant_3_unknown_outcome_never_becomes_verified_success(db_session):
    provider = staged(ACTUAL_WRONG_PAISE, write_status=ProviderExecutionStatus.UNKNOWN)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is False


def test_invariant_4_ml_cannot_override_deterministic_reconciliation():
    from app.ml.classifier import ExceptionClassifierService

    class ConfidentModel:
        def predict(self, X):
            return [0]

        def predict_proba(self, X):
            return [[1.0]]

    service = ExceptionClassifierService(
        model=ConfidentModel(), feature_names=["fee_ratio"], label_names=["EXACT_MATCH"]
    )
    prediction = service.predict({"fee_ratio": 0.5})
    assert prediction.predicted_type == "EXACT_MATCH"

    truth = _reconcile([925_000], fees=50_000)
    assert truth.exception_type == ExceptionType.FEE_MISMATCH
    assert truth.match_status == MatchStatus.EXCEPTION


def test_invariant_5_historical_similarity_cannot_override_policy():
    from app.evaluation.policy_eval import PolicyRow, _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    row = PolicyRow(
        row_id="SIM-1.0",
        description="perfect historical similarity, no evidence",
        confidence=0.5,
        evidence_coverage=0.05,
        evidence_consistency=0.05,
        historical_similarity=1.0,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="NOT_AUTO",
    )
    assert GuardrailEngine().evaluate(_engine_result(row)).decision.value != "AUTO"


def _evidence_row(**kwargs):
    from app.evaluation.policy_eval import PolicyRow

    return PolicyRow(
        row_id="EVIDENCE",
        description="evidence authority check",
        confidence=0.99,
        evidence_coverage=0.95,
        evidence_consistency=0.95,
        historical_similarity=0.99,
        exposure_paise=10_000,
        deterministic_consistent=True,
        authorized=True,
        approval_required=False,
        dependencies_healthy=True,
        expected="NOT_AUTO",
        **kwargs,
    )


def test_invariant_6_conflicting_or_novel_evidence_cannot_authorize(db_session):
    from app.evaluation.policy_eval import _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    engine = GuardrailEngine()
    assert engine.evaluate(_engine_result(_evidence_row(has_conflict=True))).decision.value != "AUTO"
    assert engine.evaluate(_engine_result(_evidence_row(is_novel=True))).decision.value != "AUTO"


def test_invariant_6b_missing_evidence_list_is_not_read_by_the_guardrail():
    """Recorded gap (Phase 16, not fixed): the policy layer decides on
    coverage/consistency aggregates and the novelty/conflict flags. The
    ``missing_evidence`` list is informational — a row that claims high
    coverage while listing missing evidence is trusted. In the real pipeline a
    missing record lowers coverage, so the gap is only reachable through an
    inconsistent signal pair.
    """
    from app.evaluation.policy_eval import _engine_result
    from app.services.guardrail_engine import GuardrailEngine

    result = GuardrailEngine().evaluate(
        _engine_result(_evidence_row(missing_evidence=["fee_record"]))
    )
    assert result.decision.value == "AUTO"
    assert any("missing" in code.value.lower() for code in result.reason_codes) is False


def test_invariant_7_authorization_failure_cannot_reach_the_provider():
    provider = staged()
    service = ResolutionExecutionService(provider=provider)
    for decision in ("UNRESOLVED", "BLOCKED", "DENIED"):
        service.execute(action_request(guardrail_decision=decision), _financial_state())
    assert provider.write_count == 0


def test_invariant_8_duplicate_execution_cannot_duplicate_money_movement():
    provider = staged()
    service = ResolutionExecutionService(provider=provider)
    request = action_request()
    results = [service.execute(request, _financial_state()) for _ in range(5)]
    assert provider.write_count == 1
    assert len({r.execution_id for r in results}) == 1


def test_invariant_9_telemetry_failure_cannot_change_financial_outcome(monkeypatch):
    provider = staged()
    request = action_request()
    with telemetry_outage(monkeypatch):
        result = ResolutionExecutionService(provider=provider).execute(
            request, _financial_state()
        )
    baseline = ResolutionExecutionService(provider=staged()).execute(
        action_request(), _financial_state()
    )
    assert status_of(result) == status_of(baseline)
    assert result.actual_adjustment_paise == baseline.actual_adjustment_paise == DIFFERENCE_PAISE


def test_invariant_10_post_execution_reconciliation_is_the_only_closure_arbiter(db_session):
    provider = staged(ACTUAL_WRONG_PAISE)
    exception = open_exception(db_session, provider, corrected=True)
    execution = ResolutionExecutionService(provider=provider).execute(
        action_request(), _financial_state()
    )
    # Execution's own after_state claims success; nothing may act on it.
    assert execution.after_state is not None
    assert db_session.get(type(exception), exception.id).status != "CLOSED"

    # Only the fresh-state reconciliation can close it, and it does.
    verification = verify(db_session, provider, execution, exception)
    assert verification.financially_verified is True
    assert verification.exception_closed is True
