"""
Golden end-to-end scenarios (Phase 15).

Ten deterministic scenarios, each asserting an outcome that must hold for the
closed-loop contract. Every stage is driven through the **real** production
component:

    deterministic reconciliation → policy/guardrail → controlled execution →
    post-execution reconciliation → closure decision

Canonical FEE_MISMATCH numbers (integer paise):
    payment 1_000_000, fee 50_000, expected settlement 950_000,
    actual settlement 925_000, difference 25_000.

Scenarios 6, 7, 9 and 10 exercise the real Phase 10 closed-loop service and
therefore require a database session (provided by the caller).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ingestion.service import IngestionService
from app.providers.mock_provider import MockProvider
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import ExceptionType, FeeType, MatchStatus
from app.schemas.execution import ExecutionStatus
from app.schemas.financial import (
    Fee as DomainFee,
    Payment as DomainPayment,
    Settlement as DomainSettlement,
)
from app.services.execution import ResolutionExecutionService
from app.services.post_execution_reconciliation import PostExecutionReconciliationService

from app.evaluation.policy_eval import evaluate_policy, EXPECT_AUTO, EXPECT_NOT_AUTO
from app.evaluation.versioning import EVALUATOR_VERSION

PAYMENT_PAISE = 1_000_000
FEE_PAISE = 50_000
EXPECTED_PAISE = 950_000
ACTUAL_WRONG_PAISE = 925_000
DIFFERENCE_PAISE = 25_000
CASE_ID = "CASE-GOLDEN-P15"
BATCH_ID = "BATCH-GOLDEN-P15"
PAYMENT_ID = "PAY-GOLDEN-P15"


class GoldenResult(BaseModel):
    """Outcome of one golden scenario."""

    scenario_id: str
    title: str
    passed: bool
    expected: str
    observed: Dict[str, Any] = Field(default_factory=dict)
    failures: List[str] = Field(default_factory=list)
    #: ``True``/``False`` when the scenario reaches a closure decision, ``None``
    #: for scenarios that never get that far (no closure to judge).
    expect_closed: Optional[bool] = None


# ============================================================================
# Stage helpers
# ============================================================================


def canonical_reconciliation(settlement_amount: int = ACTUAL_WRONG_PAISE, settlement_count: int = 1):
    """Run the real deterministic engine on the canonical scenario."""
    now = datetime.now(timezone.utc)
    payment = DomainPayment(
        payment_id=PAYMENT_ID,
        merchant_id="MER-GOLDEN-P15",
        amount=PAYMENT_PAISE,
        payment_timestamp=now,
    )
    settlements = [
        DomainSettlement(
            settlement_id=f"SET-GOLDEN-P15-{i}",
            payment_id=PAYMENT_ID,
            merchant_id="MER-GOLDEN-P15",
            amount=settlement_amount,
            settlement_timestamp=now,
        )
        for i in range(settlement_count)
    ]
    fee = DomainFee(
        fee_id="FEE-GOLDEN-P15",
        payment_id=PAYMENT_ID,
        amount=FEE_PAISE,
        fee_type=FeeType.TRANSACTION,
    )
    return calculate_reconciliation(
        payment=payment,
        settlements=settlements,
        refunds=[],
        fees=[fee],
        taxes=[],
        adjustments=[],
        case_id=CASE_ID,
        reconciliation_id="REC-GOLDEN-P15",
    )


def action_request(**overrides) -> Dict[str, Any]:
    base = {
        "action_id": "ACT-GOLDEN-P15",
        "idempotency_key": "IDEM-GOLDEN-P15",
        "workflow_id": "WF-GOLDEN-P15",
        "exception_id": "EXC-GOLDEN-P15",
        "case_id": CASE_ID,
        "candidate_id": "CAND-GOLDEN-P15",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "financial_adjustment_paise": DIFFERENCE_PAISE,
        "currency": "INR",
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
        "guardrail_confidence": 0.95,
        "metadata": {},
        "current_evidence_digest": "digest-golden-p15",
    }
    base.update(overrides)
    return base


def _financial_state(actual: int = ACTUAL_WRONG_PAISE) -> Dict[str, Any]:
    return {
        "payment_amount": PAYMENT_PAISE,
        "expected_amount": EXPECTED_PAISE,
        "actual_amount": actual,
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


def stage_fee_mismatch_provider(settlement_amount: int = ACTUAL_WRONG_PAISE) -> MockProvider:
    now = datetime.now(timezone.utc)
    provider = MockProvider(seed=42)
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


class BrokenReadProvider(MockProvider):
    """Provider whose read side fails — fresh state cannot be established."""

    def get_settlements(self):  # pragma: no cover - raised on purpose
        raise RuntimeError("provider read side unavailable")


def detect_and_persist(session, provider) -> tuple:
    """Phase 3 detection: ingest → reconcile → persist run/result/exception."""
    from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
    from app.services.persistence import PersistenceService

    now = datetime.now(timezone.utc)
    IngestionService(session).ingest(provider)

    run = DBReconciliationRun(
        id="RUN-GOLDEN-P15",
        scope_type="BATCH",
        scope_ref=BATCH_ID,
        status="PENDING",
        trigger="MANUAL",
        started_at=now,
        engine_version="3.0.0",
    )
    session.add(run)
    run.transition_to("RUNNING")
    session.flush()

    result_schema = reconcile_from_db(session, reconciliation_id="REC-GOLDEN-P15")
    persistence = PersistenceService(session)
    db_result = persistence.persist_reconciliation_result(result_schema, batch_id=BATCH_ID)
    db_result.reconciliation_run_id = run.id
    exception = persistence.persist_exception(result_schema, batch_id=BATCH_ID)
    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = now
    run.transition_to("COMPLETED")
    session.commit()
    return exception, result_schema


def reconcile_from_db(session, reconciliation_id: str = "REC-GOLDEN-P15"):
    """Rebuild domain records from the ingested database and reconcile them."""
    from app.models.fee import Fee as DBFee
    from app.models.payment import Payment as DBPayment
    from app.models.settlement import Settlement as DBSettlement

    now = datetime.now(timezone.utc)
    payment_db = session.get(DBPayment, PAYMENT_ID)
    fee_db = session.get(DBFee, "FEE-GOLDEN-P15")
    settlement_db = session.get(DBSettlement, "SET-GOLDEN-P15")

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


# ============================================================================
# Golden scenarios 1–5 and 8 (no database required)
# ============================================================================


def golden_1_fee_mismatch_full_chain() -> GoldenResult:
    """G1 — canonical FEE_MISMATCH through detection, policy and execution."""
    failures: List[str] = []
    result = canonical_reconciliation()

    if result.exception_type != ExceptionType.FEE_MISMATCH:
        failures.append(f"engine classified {result.exception_type.value}")
    if result.match_status != MatchStatus.EXCEPTION:
        failures.append(f"match status {result.match_status.value}")
    if result.difference != DIFFERENCE_PAISE:
        failures.append(f"difference {result.difference} != {DIFFERENCE_PAISE}")

    policy = evaluate_policy()
    if not policy.rows or policy.rows[0].actual_decision != "AUTO":
        failures.append("policy did not allow the canonical low-exposure case")

    execution = ResolutionExecutionService().execute(action_request(), _financial_state())
    execution_status = getattr(execution.status, "value", str(execution.status))

    # Execution success must never be treated as closure.
    closure_from_execution = execution_status == ExecutionStatus.EXECUTED.value

    return GoldenResult(
        scenario_id="G1",
        title="FEE_MISMATCH full chain",
        passed=not failures,
        expected=(
            "FEE_MISMATCH → policy AUTO → execution attempted; execution success "
            "alone never closes"
        ),
        observed={
            "exception_type": result.exception_type.value,
            "match_status": result.match_status.value,
            "expected_amount": result.expected_amount,
            "actual_amount": result.actual_amount,
            "difference": result.difference,
            "policy_decision": policy.rows[0].actual_decision if policy.rows else None,
            "execution_status": execution_status,
            "execution_is_not_closure": closure_from_execution,
        },
        failures=failures,
    )


def golden_2_already_matched() -> GoldenResult:
    """G2 — an already-matched case produces no financial exception."""
    failures: List[str] = []
    result = canonical_reconciliation(settlement_amount=EXPECTED_PAISE)

    if result.match_status != MatchStatus.MATCHED:
        failures.append(f"match status {result.match_status.value}")
    if result.difference != 0:
        failures.append(f"difference {result.difference} != 0")
    if result.exception_type != ExceptionType.EXACT_MATCH:
        failures.append(f"exception type {result.exception_type.value}")

    return GoldenResult(
        scenario_id="G2",
        title="Already matched",
        passed=not failures,
        expected="no financial exception, no resolution",
        observed={
            "match_status": result.match_status.value,
            "exception_type": result.exception_type.value,
            "difference": result.difference,
            "resolution": "NO_ACTION",
        },
        failures=failures,
    )


def golden_3_missing_evidence() -> GoldenResult:
    """G3 — missing evidence must not produce an unsafe automatic resolution."""
    failures: List[str] = []
    policy = evaluate_policy()
    weak = [r for r in policy.rows if r.row_id in ("R2", "R8")]

    for row in weak:
        if row.auto:
            failures.append(f"{row.row_id} auto-resolved without sufficient evidence")
    if policy.evidence_bypass_detected:
        failures.append("evidence bypass detected")

    return GoldenResult(
        scenario_id="G3",
        title="Missing evidence",
        passed=not failures,
        expected="no unsafe automatic resolution",
        observed={
            "rows_checked": [r.row_id for r in weak],
            "decisions": {r.row_id: r.actual_decision for r in weak},
        },
        failures=failures,
    )


def golden_4_conflicting_evidence() -> GoldenResult:
    """G4 — conflicting evidence must be deferred or escalated."""
    failures: List[str] = []
    policy = evaluate_policy()
    conflict_rows = [r for r in policy.rows if r.has_conflict]

    for row in conflict_rows:
        if row.auto:
            failures.append(f"{row.row_id} auto-resolved despite conflicting evidence")
        if row.actual_decision not in ("HUMAN_REVIEW", "UNRESOLVED"):
            failures.append(f"{row.row_id} decision {row.actual_decision}")

    return GoldenResult(
        scenario_id="G4",
        title="Conflicting evidence",
        passed=not failures,
        expected="human review / escalation",
        observed={"decisions": {r.row_id: r.actual_decision for r in conflict_rows}},
        failures=failures,
    )


def golden_5_high_confidence_policy_denial() -> GoldenResult:
    """G5 — maximum ML confidence must not bypass policy."""
    failures: List[str] = []
    policy = evaluate_policy()
    high_conf = [r for r in policy.rows if r.confidence >= 0.99 and r.expected == EXPECT_NOT_AUTO]

    for row in high_conf:
        if row.auto:
            failures.append(f"{row.row_id} auto-resolved on confidence alone")
    if policy.confidence_bypass_detected:
        failures.append("confidence bypass detected")

    return GoldenResult(
        scenario_id="G5",
        title="High ML confidence + policy denial",
        passed=not failures,
        expected="DENY / HUMAN_REVIEW — ML never bypasses policy",
        observed={"decisions": {r.row_id: r.actual_decision for r in high_conf}},
        failures=failures,
    )


def golden_8_stale_approval() -> GoldenResult:
    """G8 — an execution request with a stale/failed verification is blocked."""
    failures: List[str] = []
    service = ResolutionExecutionService()

    stale = service.execute(
        action_request(
            idempotency_key="IDEM-GOLDEN-STALE",
            verification_passed=False,
        ),
        _financial_state(),
    )
    stale_status = getattr(stale.status, "value", str(stale.status))

    if stale_status == ExecutionStatus.EXECUTED.value:
        failures.append("stale approval executed successfully")

    return GoldenResult(
        scenario_id="G8",
        title="Stale approval",
        passed=not failures,
        expected="execution blocked",
        observed={"execution_status": stale_status, "error": getattr(stale, "error", None)},
        failures=failures,
    )


# ============================================================================
# Golden scenarios 6, 7, 9, 10 (database-backed closed loop)
# ============================================================================


def _closed_loop_service(session, execution_service=None) -> PostExecutionReconciliationService:
    from app.services.audit_log import AuditLogService

    return PostExecutionReconciliationService(
        session=session,
        execution_service=execution_service,
        audit_log=AuditLogService(),
    )


def _proposal():
    from app.schemas.resolution_candidate import (
        CandidateRanking,
        FinancialAdjustment,
        ResolutionProposal,
    )

    return ResolutionProposal(
        candidate_id="CAND-GOLDEN-P15",
        exception_id="EXC-GOLDEN-P15",
        case_id=CASE_ID,
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Correct the under-settled fee difference",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=DIFFERENCE_PAISE,
            direction="CREDIT",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-GOLDEN-P15", PAYMENT_ID],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record explains the discrepancy",
        sources=["deterministic_evidence"],
        ranking=CandidateRanking(rank=1, confidence_score=0.92, evidence_support=0.95),
        rationale="Deterministic fee mismatch",
    )


def _resolution_record(session, exception, status: str = "EXECUTED"):
    """Persist the resolution the closed-loop service verifies.

    ``verify_and_close`` takes the persisted Phase-10 ``Resolution`` row (not a
    transient proposal) so its legal ``EXECUTED -> VERIFIED`` transition can be
    exercised.
    """
    from app.models.resolution import Resolution

    resolution = Resolution(
        id="RES-GOLDEN-P15",
        exception_id=exception.id,
        resolution_type="FEE_ADJUSTMENT",
        amount_paise=DIFFERENCE_PAISE,
        status=status,
    )
    session.add(resolution)
    session.flush()
    return resolution


def _run_closed_loop(session, settlement_amount: int, adjustment_paise: int = DIFFERENCE_PAISE, provider=None):
    """Stage, execute and verify.

    Returns ``(execution_result, verification_result, status_after_execute)``.
    The status is captured *before* verification because verification may
    legally transition the execution to ``VERIFICATION_FAILED``; asserting on
    the post-verification status would confuse the two contracts.
    """
    provider = provider or stage_fee_mismatch_provider(settlement_amount)
    exception, _ = detect_and_persist(session, provider)

    execution = ResolutionExecutionService().execute(
        action_request(financial_adjustment_paise=adjustment_paise), _financial_state()
    )
    status_after_execute = getattr(execution.status, "value", str(execution.status))
    verification = _closed_loop_service(session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id=PAYMENT_ID,
        resolution=_resolution_record(session, exception),
    )
    return execution, verification, status_after_execute


def golden_6_provider_success_reconciliation_failure(session) -> GoldenResult:
    """G6 — execution success + unchanged provider state must NOT close."""
    failures: List[str] = []
    # The provider still reports the wrong settlement, so post-execution
    # reconciliation must still mismatch.
    execution, verification, execution_status = _run_closed_loop(session, ACTUAL_WRONG_PAISE)

    if execution_status != ExecutionStatus.EXECUTED.value:
        failures.append(f"execution did not succeed: {execution_status}")
    if verification.financially_verified:
        failures.append("verification claimed success on mismatched state")
    if verification.exception_closed:
        failures.append("exception was closed despite failed verification")

    return GoldenResult(
        scenario_id="G6",
        title="Provider success + reconciliation failure",
        passed=not failures,
        expect_closed=False,
        expected="execution success → verification failure → NOT CLOSED → escalation",
        observed={
            "execution_status": execution_status,
            "financially_verified": verification.financially_verified,
            "exception_closed": verification.exception_closed,
            "exception_escalated": verification.exception_escalated,
            "decision": getattr(verification.decision, "value", str(verification.decision)),
        },
        failures=failures,
    )


def golden_7_provider_timeout(session) -> GoldenResult:
    """G7 — a provider read failure must fail closed."""
    failures: List[str] = []
    provider = BrokenReadProvider(seed=42)
    # Stage the data first with a working provider, then swap in the broken one.
    working = stage_fee_mismatch_provider(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(session, working)

    execution = ResolutionExecutionService().execute(action_request(), _financial_state())
    verification = _closed_loop_service(session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id=PAYMENT_ID,
        resolution=_resolution_record(session, exception),
    )

    if verification.financially_verified:
        failures.append("verified despite provider failure")
    if verification.exception_closed:
        failures.append("closed despite provider failure")

    return GoldenResult(
        scenario_id="G7",
        title="Provider timeout / read failure",
        passed=not failures,
        expect_closed=False,
        expected="fail closed — not verified, not closed",
        observed={
            "financially_verified": verification.financially_verified,
            "exception_closed": verification.exception_closed,
            "decision": getattr(verification.decision, "value", str(verification.decision)),
            "failure_reason": verification.failure_reason,
        },
        failures=failures,
    )


def golden_9_unknown_provider_result(session) -> GoldenResult:
    """G9 — an unknown/non-success execution status cannot close."""
    failures: List[str] = []
    # The exception must exist so the closed-loop service has something to
    # refuse to close; the settlement therefore stays wrong.
    provider = stage_fee_mismatch_provider(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(session, provider)

    # A synthetic non-success execution result: provider reported neither
    # success nor a definitive failure.
    failed = ResolutionExecutionService().execute(
        action_request(idempotency_key="IDEM-GOLDEN-UNKNOWN", verification_passed=False),
        _financial_state(),
    )
    status = getattr(failed.status, "value", str(failed.status))
    if status == ExecutionStatus.EXECUTED.value:
        failures.append("expected a non-success execution result to build G9")

    verification = _closed_loop_service(session).verify_and_close(
        failed,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id=PAYMENT_ID,
        resolution=_resolution_record(session, exception),
    )

    if verification.financially_verified:
        failures.append("verified a non-success execution")
    if verification.exception_closed:
        failures.append("closed a non-success execution")

    return GoldenResult(
        scenario_id="G9",
        title="Unknown provider result",
        passed=not failures,
        expect_closed=False,
        expected="not verified / escalation",
        observed={
            "execution_status": status,
            "financially_verified": verification.financially_verified,
            "exception_closed": verification.exception_closed,
            "decision": getattr(verification.decision, "value", str(verification.decision)),
        },
        failures=failures,
    )


def golden_10_successful_closed_loop(session) -> GoldenResult:
    """G10 — the provider state is corrected, so the loop closes."""
    failures: List[str] = []
    # Provider initially wrong; after execution the settlement is corrected to
    # the expected amount, so fresh post-execution state reconciles MATCHED.
    provider = stage_fee_mismatch_provider(ACTUAL_WRONG_PAISE)
    exception, _ = detect_and_persist(session, provider)

    # Simulate the provider applying the fee correction.
    from app.models.settlement import Settlement as DBSettlement

    settlement = session.get(DBSettlement, "SET-GOLDEN-P15")
    settlement.amount = EXPECTED_PAISE
    session.commit()

    resolution = _resolution_record(session, exception)
    execution = ResolutionExecutionService().execute(action_request(), _financial_state())
    verification = _closed_loop_service(session).verify_and_close(
        execution,
        exception=exception,
        provider=provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id=PAYMENT_ID,
        resolution=resolution,
    )

    if not verification.financially_verified:
        failures.append(f"not verified: {verification.failure_reason}")
    if not verification.exception_closed:
        failures.append("exception was not closed after successful verification")

    return GoldenResult(
        scenario_id="G10",
        title="Correct resolution → verified → closed",
        passed=not failures,
        expect_closed=True,
        expected="execution → re-ingestion → deterministic reconciliation → verified → closure",
        observed={
            "financially_verified": verification.financially_verified,
            "exception_closed": verification.exception_closed,
            "decision": getattr(verification.decision, "value", str(verification.decision)),
            "difference": getattr(verification, "difference", None),
            "resolution_status": resolution.status,
            "resolved_through_verification": verification.resolution_verified,
            "match_status": getattr(verification, "match_status", None),
        },
        failures=failures,
    )


STAGE_SCENARIOS = (
    golden_1_fee_mismatch_full_chain,
    golden_2_already_matched,
    golden_3_missing_evidence,
    golden_4_conflicting_evidence,
    golden_5_high_confidence_policy_denial,
    golden_8_stale_approval,
)

DB_SCENARIOS = (
    golden_6_provider_success_reconciliation_failure,
    golden_7_provider_timeout,
    golden_9_unknown_provider_result,
    golden_10_successful_closed_loop,
)


def run_stage_scenarios() -> List[GoldenResult]:
    """Run the database-free golden scenarios."""
    return [scenario() for scenario in STAGE_SCENARIOS]


def closure_metrics(results: List[GoldenResult]) -> Dict[str, Any]:
    """Closure accuracy over the scenarios that reach a closure decision.

    Denominators are explicit: if no scenario reaches a closure decision the
    accuracy is reported as ``None`` ("N/A — no applicable cases") rather than
    as a measured zero.
    """

    evaluated = [r for r in results if r.expect_closed is not None]
    if not evaluated:
        return {
            "closure_attempts": 0,
            "closed_exceptions": 0,
            "correctly_closed": 0,
            "closure_accuracy": None,
            "closure_precision": None,
        }
    closed = sum(1 for r in evaluated if bool(r.observed.get("exception_closed")))
    correct = sum(
        1 for r in evaluated if bool(r.observed.get("exception_closed")) == r.expect_closed
    )
    closed_and_correct = sum(
        1
        for r in evaluated
        if r.expect_closed and bool(r.observed.get("exception_closed"))
    )
    return {
        "closure_attempts": len(evaluated),
        "closed_exceptions": closed,
        "correctly_closed": correct,
        # Decision accuracy: did the closed-loop service close exactly the
        # scenarios that should have been closed?
        "closure_accuracy": round(correct / len(evaluated), 4),
        # Literal "correctly closed / all closed" — N/A when nothing closed.
        "closure_precision": round(closed_and_correct / closed, 4) if closed else None,
    }


def golden_summary(results: List[GoldenResult]) -> Dict[str, Any]:
    """Summary of golden scenario outcomes."""
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [r.model_dump() for r in results],
    }
