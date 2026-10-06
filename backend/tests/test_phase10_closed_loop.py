"""
Phase 10 tests — post-execution reconciliation, verification and closed-loop closure.

The invariant under test:

    PROVIDER SUCCESS != FINANCIAL VERIFICATION.

An exception may be closed only when the existing deterministic reconciliation
engine reports MATCHED on *fresh post-execution provider state*. Provider
failure, timeout, unknown, missing fresh state, reconciliation errors, a
fabricated execution ``after_state``, ML confidence, historical similarity and
raw provider SUCCESS must all leave the exception NOT CLOSED.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.state_machines import EXCEPTION, is_valid_transition
from app.ingestion.service import IngestionService
from app.models.exception import FinancialException
from app.models.fee import Fee as DBFee
from app.models.payment import Payment as DBPayment
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
from app.models.settlement import Settlement as DBSettlement
from app.providers.mock_provider import MockProvider
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.closed_loop import VerificationDecision
from app.schemas.enums import ExceptionType, FeeType, MatchStatus
from app.schemas.execution import ExecutionStatus, FinancialStateSnapshot
from app.schemas.financial import (
    Fee as DomainFee,
    Payment as DomainPayment,
    Settlement as DomainSettlement,
)
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.services import post_execution_reconciliation as per_module
from app.services.audit_log import AuditLogService
from app.services.authorization import AuthorizationService
from app.services.execution import ResolutionExecutionService
from app.services.persistence import PersistenceService
from app.services.post_execution_reconciliation import (
    PostExecutionReconciliationService,
)

# Canonical FEE_MISMATCH numbers (integer paise)
PAYMENT_PAISE = 1_000_000
FEE_PAISE = 50_000
EXPECTED_PAISE = 950_000
ACTUAL_WRONG_PAISE = 925_000
DIFFERENCE_PAISE = 25_000

CASE_ID = "CASE-P10-001"
BATCH_ID = "BATCH-P10-001"


# --------------------------------------------------------------------- helpers


def _stage_fee_mismatch_provider(
    settlement_amount: int = ACTUAL_WRONG_PAISE,
    settlement_version: int = 1,
) -> MockProvider:
    now = datetime.now(timezone.utc)
    provider = MockProvider(seed=42)
    provider.add_merchant("MER-P10-001", name="Phase 10 Merchant")
    provider.add_payment(
        "PAY-P10-001", merchant_id="MER-P10-001", amount=PAYMENT_PAISE, captured_at=now
    )
    provider.add_fee(
        "FEE-P10-001",
        payment_id="PAY-P10-001",
        amount=FEE_PAISE,
        fee_type=FeeType.TRANSACTION,
        processed_at=now,
    )
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=settlement_amount,
        settled_at=now,
        version=settlement_version,
    )
    return provider


def _detect_fee_mismatch(db_session, provider):
    """Phase 3 detection: ingest -> reconcile -> persist run/result/exception."""
    now = datetime.now(timezone.utc)
    IngestionService(db_session).ingest(provider)

    run = DBReconciliationRun(
        id="RUN-P10-DETECT",
        scope_type="BATCH",
        scope_ref=BATCH_ID,
        status="PENDING",
        trigger="MANUAL",
        started_at=now,
        engine_version="3.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    result_schema = _reconcile_from_db(db_session, reconciliation_id="REC-P10-DETECT")
    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(
        result_schema, batch_id=BATCH_ID
    )
    db_result.reconciliation_run_id = run.id
    exception = persistence.persist_exception(result_schema, batch_id=BATCH_ID)
    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = now
    run.transition_to("COMPLETED")
    db_session.commit()
    return exception, result_schema


def _reconcile_from_db(db_session, reconciliation_id):
    payment_db = db_session.get(DBPayment, "PAY-P10-001")
    fee_db = db_session.get(DBFee, "FEE-P10-001")
    settlement_db = db_session.get(DBSettlement, "SET-P10-001")
    now = datetime.now(timezone.utc)
    payment = DomainPayment(
        payment_id=payment_db.id,
        merchant_id=payment_db.merchant_id,
        amount=payment_db.amount,
        payment_timestamp=payment_db.captured_at or now,
    )
    fee = DomainFee(
        fee_id=fee_db.id,
        payment_id=fee_db.payment_id,
        amount=fee_db.amount,
        fee_type=FeeType(fee_db.fee_type),
    )
    settlement = DomainSettlement(
        settlement_id=settlement_db.id,
        payment_id=settlement_db.payment_id,
        merchant_id=settlement_db.merchant_id,
        amount=settlement_db.amount,
        settlement_timestamp=settlement_db.settled_at or now,
    )
    return calculate_reconciliation(
        payment=payment,
        settlements=[settlement],
        refunds=[],
        fees=[fee],
        taxes=[],
        adjustments=[],
        case_id=CASE_ID,
        reconciliation_id=reconciliation_id,
    )


def _financial_state():
    return {
        "payment_amount": PAYMENT_PAISE,
        "expected_amount": EXPECTED_PAISE,
        "actual_amount": ACTUAL_WRONG_PAISE,
        "difference": DIFFERENCE_PAISE,
        "total_refunds": 0,
        "total_fees": FEE_PAISE,
        "total_taxes": 0,
        "total_adjustments": 0,
        "settlement_count": 1,
        "refund_count": 0,
        "fee_count": 1,
        "tax_count": 0,
        "adjustment_count": 0,
    }


def _no_action_request(**overrides):
    base = {
        "action_id": "ACT-P10-001",
        "idempotency_key": "IDEM-P10-001",
        "workflow_id": "WF-P10-001",
        "exception_id": "EXC-P10-DETECT",
        "case_id": CASE_ID,
        "candidate_id": "CAND-P10-001",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "financial_adjustment_paise": DIFFERENCE_PAISE,
        "currency": "INR",
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
        "guardrail_confidence": 0.95,
        "metadata": {},
        "current_evidence_digest": "digest-p10",
    }
    base.update(overrides)
    return base


def _execute_action(
    adjustment_paise: int = DIFFERENCE_PAISE,
    idempotency_key: str = "IDEM-P10-001",
    metadata: dict | None = None,
    authorization_source: str = "AUTO_GUARDRAIL",
    guardrail_decision: str = "AUTO",
):
    service = ResolutionExecutionService()
    request = _no_action_request(
        idempotency_key=idempotency_key,
        financial_adjustment_paise=adjustment_paise,
        metadata=metadata or {},
        authorization_source=authorization_source,
        guardrail_decision=guardrail_decision,
    )
    return service.execute(request, _financial_state())


def _service(db_session, execution_service=None):
    return PostExecutionReconciliationService(
        session=db_session,
        execution_service=execution_service,
        audit_log=AuditLogService(),
    )


def _make_proposal():
    return ResolutionProposal(
        candidate_id="CAND-P10-001",
        exception_id="EXC-P10-DETECT",
        case_id=CASE_ID,
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Correct the under-settled fee difference",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=DIFFERENCE_PAISE,
            direction="CREDIT",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-P10-001", "PAY-P10-001"],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record explains the discrepancy",
        sources=["deterministic_evidence"],
        ranking=CandidateRanking(rank=1, confidence_score=0.92, evidence_support=0.95),
        rationale="Deterministic fee mismatch",
    )


class _BrokenProvider(MockProvider):
    """A provider whose read side fails — fresh state cannot be established."""

    def get_settlements(self):  # pragma: no cover - raised on purpose
        raise RuntimeError("provider read side unavailable")


# =============================================================================
# A. fresh post-execution state is retrieved from the provider boundary
# =============================================================================


def test_fresh_post_execution_state_is_retrieved(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)

    execution = _execute_action()
    assert execution.status == ExecutionStatus.EXECUTED

    # The external system applies the correction: settlement now fully settled.
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        idempotency_key=execution.idempotency_key,
    )

    assert result.fresh_state_used is True
    assert result.expected_amount == EXPECTED_PAISE
    assert result.actual_amount == EXPECTED_PAISE
    # The database itself was refreshed from provider state.
    assert db_session.get(DBSettlement, "SET-P10-001").amount == EXPECTED_PAISE


# =============================================================================
# B / M. successful reconciliation -> verified -> closed
# =============================================================================


def test_successful_reconciliation_verifies_and_closes(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.VERIFIED
    assert result.financially_verified is True
    assert result.match_status == MatchStatus.MATCHED.value
    assert result.difference == 0
    assert result.exception_closed is True
    assert exception.status == "CLOSED"
    assert exception.closed_at is not None
    assert result.execution_status == ExecutionStatus.VERIFIED.value


# =============================================================================
# C / N. provider SUCCESS but NOT resolved -> never closed
# =============================================================================


def test_provider_success_but_mismatch_is_not_closed(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    assert execution.status == ExecutionStatus.EXECUTED

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.FAILED
    assert result.financially_verified is False
    assert result.exception_closed is False
    assert exception.status != "CLOSED"
    assert result.difference == DIFFERENCE_PAISE
    assert result.actual_amount == ACTUAL_WRONG_PAISE


# =============================================================================
# D / E / F. provider failure / timeout / unknown
# =============================================================================


@pytest.mark.parametrize(
    "outcome",
    ["FAILED", "TIMEOUT", "UNKNOWN"],
)
def test_provider_non_success_cannot_close(db_session, outcome):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action(metadata={"mock_outcome": outcome})
    assert execution.status == ExecutionStatus.EXECUTION_FAILED

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.UNRESOLVED
    assert result.financially_verified is False
    assert result.exception_closed is False
    assert exception.status != "CLOSED"
    # No fabricated financial state was produced.
    assert result.expected_amount == 0
    assert result.actual_amount == 0


# =============================================================================
# G. post-execution state retrieval failure
# =============================================================================


def test_post_execution_state_retrieval_failure_fails_closed(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=_BrokenProvider(),
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.UNRESOLVED
    assert result.fresh_state_used is False
    assert result.exception_closed is False
    assert exception.status != "CLOSED"
    assert "state" in (result.failure_reason or "").lower()


# =============================================================================
# H. reconciliation failure is never interpreted as MATCH
# =============================================================================


def test_reconciliation_failure_cannot_close(db_session, monkeypatch):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(per_module, "calculate_reconciliation", _boom)

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.FAILED
    assert result.financially_verified is False
    assert result.exception_closed is False
    assert exception.status != "CLOSED"
    assert "reconciliation failed" in (result.failure_reason or "")


# =============================================================================
# I. before/after state correctness
# =============================================================================


def test_before_and_after_state_are_distinguished(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, detection = _detect_fee_mismatch(db_session, provider)
    assert detection.expected_amount == EXPECTED_PAISE
    assert detection.actual_amount == ACTUAL_WRONG_PAISE
    assert detection.difference == DIFFERENCE_PAISE

    execution = _execute_action()
    assert execution.before_state.difference == DIFFERENCE_PAISE
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert (result.expected_amount, result.actual_amount, result.difference) == (
        EXPECTED_PAISE,
        EXPECTED_PAISE,
        0,
    )


# =============================================================================
# J / K. legal transitions and illegal closure
# =============================================================================


def test_closure_uses_only_legal_transitions(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    # CLOSED is legal only from RECONCILING.
    assert is_valid_transition(EXCEPTION, "RECONCILING", "CLOSED") is True
    assert is_valid_transition(EXCEPTION, "OPEN", "CLOSED") is False
    assert exception.status == "CLOSED"


def test_illegal_closure_is_blocked_by_the_state_machine(db_session):
    from app.domain.state_machines import InvalidStateTransitionError

    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    assert exception.status == "OPEN"
    with pytest.raises(InvalidStateTransitionError):
        exception.transition_to("CLOSED")
    assert exception.status == "OPEN"


# =============================================================================
# L. repeated verification is deterministic and does not duplicate closure
# =============================================================================


def test_repeated_verification_is_deterministic(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    service = _service(db_session)
    first = service.verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        idempotency_key=execution.idempotency_key,
    )
    second = service.verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        idempotency_key=execution.idempotency_key,
    )

    assert first.decision == second.decision == VerificationDecision.VERIFIED
    assert (first.expected_amount, first.actual_amount, first.difference) == (
        second.expected_amount,
        second.actual_amount,
        second.difference,
    )
    assert (second.expected_amount, second.actual_amount, second.difference) == (
        EXPECTED_PAISE,
        EXPECTED_PAISE,
        0,
    )
    assert second.exception_already_closed is True
    assert second.exception_closed is True
    assert exception.status == "CLOSED"

    runs = (
        db_session.query(DBReconciliationRun)
        .filter_by(scope_ref=exception.id, trigger="POST_RESOLUTION")
        .all()
    )
    results = (
        db_session.query(DBReconciliationResult)
        .filter_by(batch_id=f"{BATCH_ID}::POST_EXECUTION")
        .all()
    )
    assert len(runs) == 1
    assert len(results) == 1


# =============================================================================
# O / P / Q. no AI, ML, similarity or raw provider success can close
# =============================================================================


def test_ml_confidence_cannot_close(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    # A maximally confident automated decision must not influence truth.
    execution.decision = "AUTO"
    execution.confidence = 1.0
    execution.guardrail_reason_codes = ["ML_HIGH_CONFIDENCE"]

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.FAILED
    assert exception.status != "CLOSED"


def test_historical_similarity_cannot_close(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    execution.candidate_id = "CAND-SIMILAR-HISTORICAL-001"
    execution.evidence_references = ["HIST-001", "HIST-002"]
    exception.status_reason = "historical similarity 0.99 says resolved"

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.decision == VerificationDecision.FAILED
    assert exception.status != "CLOSED"


def test_provider_success_alone_cannot_close(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    assert execution.status == ExecutionStatus.EXECUTED

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    # Provider SUCCESS was real, yet verification still failed closed.
    assert result.decision == VerificationDecision.FAILED
    assert result.financially_verified is False
    assert result.exception_closed is False
    assert exception.status != "CLOSED"


def test_service_has_no_ml_llm_or_similarity_dependencies():
    source = inspect.getsource(per_module)
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("app.ml", "app.llm", "similarity", "embedding", "xgboost", "openai")
    for line in import_lines:
        for bad in forbidden:
            assert bad not in line.lower(), f"forbidden dependency: {line}"


# =============================================================================
# R. no fake after-state
# =============================================================================


def test_fabricated_execution_after_state_is_ignored(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()

    # Fabricate a "resolved" after-state on the execution result.
    execution.after_state = FinancialStateSnapshot(
        exception_id="EXC-P10-DETECT",
        payment_amount=PAYMENT_PAISE,
        expected_amount=EXPECTED_PAISE,
        actual_amount=EXPECTED_PAISE,
        difference=0,
        total_adjustments=DIFFERENCE_PAISE,
        adjustment_count=1,
        snapshot_reason="fabricated_post_execution",
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    # Provider truth (still 925,000) wins over the fabricated after-state.
    assert result.actual_amount == ACTUAL_WRONG_PAISE
    assert result.difference == DIFFERENCE_PAISE
    assert result.decision == VerificationDecision.FAILED
    assert exception.status != "CLOSED"


def test_provider_value_wins_over_requested_adjustment(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action(adjustment_paise=DIFFERENCE_PAISE)

    # Provider reports a partially-corrected settlement (940,000), not the
    # 950,000 that before_state + requested adjustment would imply.
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=940_000,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.expected_amount == EXPECTED_PAISE
    assert result.actual_amount == 940_000
    assert result.difference == 10_000
    assert result.decision == VerificationDecision.FAILED
    assert exception.status != "CLOSED"


# =============================================================================
# S. no provider/DB financial mutation bypass
# =============================================================================


def test_verification_does_not_mutate_provider_state(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()

    before = [
        (s.settlement_id, s.amount, s.version) for s in provider.get_settlements()
    ]
    _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )
    after = [
        (s.settlement_id, s.amount, s.version) for s in provider.get_settlements()
    ]
    assert before == after


def test_service_does_not_call_provider_execution():
    source = inspect.getsource(per_module)
    assert "provider.execute" not in source
    assert ".execute(request" not in source


# =============================================================================
# T. reconciliation persistence
# =============================================================================


def test_verification_reconciliation_is_persisted(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.reconciliation_persisted is True
    run = (
        db_session.query(DBReconciliationRun)
        .filter_by(scope_ref=exception.id, trigger="POST_RESOLUTION")
        .one()
    )
    assert run.status == "COMPLETED"
    assert run.matched_count == 1
    assert run.exception_count == 0
    assert run.engine_version == per_module.ENGINE_VERSION

    db_result = (
        db_session.query(DBReconciliationResult)
        .filter_by(batch_id=f"{BATCH_ID}::POST_EXECUTION")
        .one()
    )
    assert db_result.reconciliation_run_id == run.id
    assert db_result.match_status == MatchStatus.MATCHED.value
    assert db_result.difference == 0
    assert db_result.actual_amount == EXPECTED_PAISE


def test_mismatch_is_also_persisted_and_audited(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
    )

    assert result.reconciliation_persisted is True
    run = (
        db_session.query(DBReconciliationRun)
        .filter_by(scope_ref=exception.id, trigger="POST_RESOLUTION")
        .one()
    )
    assert run.status == "COMPLETED"
    assert run.exception_count == 1
    db_result = (
        db_session.query(DBReconciliationResult)
        .filter_by(batch_id=f"{BATCH_ID}::POST_EXECUTION")
        .one()
    )
    assert db_result.match_status == MatchStatus.EXCEPTION.value
    assert db_result.difference == DIFFERENCE_PAISE
    assert len(result.audit_event_ids) >= 1


# =============================================================================
# Auditability
# =============================================================================


def test_verification_is_auditable(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )
    audit_log = AuditLogService()
    service = PostExecutionReconciliationService(
        session=db_session, audit_log=audit_log
    )

    result = service.verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        workflow_id="WF-P10-001",
    )

    assert result.audit_event_ids
    events = audit_log.get_workflow_events("WF-P10-001")
    types = {e.event_type.value for e in events}
    assert "VERIFICATION_PERFORMED" in types
    assert "RESOLUTION_VERIFIED" in types
    verification_event = next(
        e for e in events if e.event_type.value == "VERIFICATION_PERFORMED"
    )
    assert verification_event.verification_metadata.difference_after == 0
    assert verification_event.verification_metadata.discrepancy_eliminated is True


# =============================================================================
# Resolution / ResolutionAction legal transitions
# =============================================================================


def test_resolution_and_action_are_verified_on_match(db_session):
    from app.models.resolution import Resolution
    from app.models.resolution_action import ResolutionAction

    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)

    resolution = Resolution(
        id="RES-P10-001",
        exception_id=exception.id,
        resolution_type="FEE_ADJUSTMENT",
        amount_paise=DIFFERENCE_PAISE,
        status="EXECUTED",
    )
    action = ResolutionAction(
        id="RACT-P10-001",
        resolution_id="RES-P10-001",
        action_type="ADJUSTMENT",
        amount_paise=DIFFERENCE_PAISE,
        idempotency_key="RACT-KEY-P10-001",
        status="EXECUTED",
    )
    db_session.add_all([resolution, action])
    db_session.flush()

    execution = _execute_action()
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        resolution=resolution,
        resolution_action=action,
    )

    assert result.decision == VerificationDecision.VERIFIED
    assert resolution.status == "VERIFIED"
    assert action.status == "VERIFIED"
    assert result.resolution_verified is True
    assert result.action_verified is True


def test_resolution_not_verified_on_mismatch(db_session):
    from app.models.resolution import Resolution

    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    resolution = Resolution(
        id="RES-P10-002",
        exception_id=exception.id,
        resolution_type="FEE_ADJUSTMENT",
        amount_paise=DIFFERENCE_PAISE,
        status="EXECUTED",
    )
    db_session.add(resolution)
    db_session.flush()

    execution = _execute_action()
    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        resolution=resolution,
    )

    assert result.decision == VerificationDecision.FAILED
    assert resolution.status != "VERIFIED"
    assert result.resolution_verified is None or result.resolution_verified is False


# =============================================================================
# GOLDEN — full closed-loop vertical slice (FEE_MISMATCH)
# =============================================================================


def test_golden_fee_mismatch_successful_closed_loop(db_session):
    """Phase 3 detection -> Phase 7 proposal -> Phase 8 authorization ->
    Phase 9 execution -> provider state updated -> fresh state -> reconciliation
    MATCH -> VERIFIED -> CLOSED.
    """
    provider = _stage_fee_mismatch_provider()
    exception, detection = _detect_fee_mismatch(db_session, provider)
    assert detection.exception_type == ExceptionType.FEE_MISMATCH
    assert detection.difference == DIFFERENCE_PAISE

    # Phase 7: deterministic proposal.
    proposal = _make_proposal()

    # Phase 8: policy authorization (reuses the guardrail engine).
    authorization = AuthorizationService(session=None).authorize_proposal(proposal)
    assert authorization.decision.value in ("AUTO", "HUMAN_REVIEW")

    # Phase 9: controlled provider execution.
    future = datetime.now(timezone.utc) + timedelta(days=1)
    if authorization.is_auto():
        execution = _execute_action()
    else:
        execution = _execute_action(
            authorization_source="HUMAN_APPROVAL",
            guardrail_decision="HUMAN_REVIEW",
        )
        # HUMAN_REVIEW requires an explicit APPROVED approval record.
        execution_service = ResolutionExecutionService()
        execution = execution_service.execute(
            _no_action_request(
                idempotency_key="IDEM-P10-GOLDEN",
                authorization_source="HUMAN_APPROVAL",
                guardrail_decision="HUMAN_REVIEW",
                approval_record={
                    "decision": "APPROVED",
                    "expires_at": future,
                    "evidence_digest": "digest-p10",
                },
            ),
            _financial_state(),
        )
    assert execution.status == ExecutionStatus.EXECUTED

    # External system applies the correction.
    provider.add_settlement(
        "SET-P10-001",
        payment_id="PAY-P10-001",
        merchant_id="MER-P10-001",
        amount=EXPECTED_PAISE,
        version=2,
    )

    # Phase 10: fresh state -> existing engine -> MATCH -> VERIFIED -> CLOSED.
    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        idempotency_key=execution.idempotency_key,
    )

    assert result.financially_verified is True
    assert result.difference == 0
    assert result.decision == VerificationDecision.VERIFIED
    assert result.exception_closed is True
    assert exception.status == "CLOSED"
    assert exception.close_reason == "POST_EXECUTION_RECONCILED"
    assert execution.status == ExecutionStatus.VERIFIED


def test_golden_fee_mismatch_negative_provider_success_but_unresolved(db_session):
    """Provider SUCCESS, but the financial state stays wrong -> NOT CLOSED."""
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    proposal = _make_proposal()
    authorization = AuthorizationService(session=None).authorize_proposal(proposal)
    assert authorization.decision.value in ("AUTO", "HUMAN_REVIEW")

    execution = _execute_action(idempotency_key="IDEM-P10-GOLDEN-NEG")
    assert execution.status == ExecutionStatus.EXECUTED

    result = _service(db_session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P10-001",
        idempotency_key=execution.idempotency_key,
    )

    assert result.decision == VerificationDecision.FAILED
    assert result.financially_verified is False
    assert result.difference == DIFFERENCE_PAISE
    assert result.exception_closed is False
    assert exception.status != "CLOSED"
    # The execution side legally records the verification failure.
    assert execution.status == ExecutionStatus.VERIFICATION_FAILED


def test_verification_decision_has_no_verified_default():
    """A default-constructed denial can never look verified."""
    from app.schemas.closed_loop import ClosedLoopVerificationResult

    result = ClosedLoopVerificationResult(
        verification_id="CLV-X",
        execution_id="EXE-X",
        exception_id="EXC-X",
        reconciliation_id="REC-X",
        decision=VerificationDecision.FAILED,
    )
    assert result.financially_verified is False
    assert result.exception_closed is False
