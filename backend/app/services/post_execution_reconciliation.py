"""
Post-execution reconciliation + closed-loop verification (Phase 10).

Rule:

    PROVIDER SUCCESS != FINANCIAL VERIFICATION.

A financial exception may be closed **only** when post-execution *deterministic
reconciliation* proves the discrepancy is gone. The provider's execution
response, the execution result's ``after_state``, an ML prediction, a
similarity lookup, an LLM explanation and an agent's claim of success are all
insufficient on their own.

This module is an orchestrator, not a new source of truth. It reuses:

* the provider boundary (:mod:`app.providers`) to re-fetch *fresh*
  post-execution financial state;
* the ingestion service (:mod:`app.ingestion`) to normalize that snapshot into
  the database idempotently;
* the existing deterministic reconciliation engine
  (:mod:`app.reconciliation.engine`) as the sole arbiter of financial truth —
  integer paise, no floating point, no ML, no LLM;
* the existing state machines (:mod:`app.domain.state_machines`) so every
  transition is legal (a direct string assignment is never used);
* the existing :class:`~app.services.persistence.PersistenceService` to persist
  the verification as a ``ReconciliationRun`` (trigger ``POST_RESOLUTION``) and
  a ``ReconciliationResult``;
* the existing :class:`~app.services.audit_log.AuditLogService` for an
  append-only audit trail.

Every uncertainty fails closed: provider failure/timeout/unknown, missing fresh
state, malformed engine output, a reconciliation error, or a mismatch all leave
the exception **NOT CLOSED**.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from app.domain.state_machines import EXCEPTION, allowed_transitions
from app.core import observability as obs


def _closed_loop_outcome(result: Any) -> Dict[str, Any]:
    """Telemetry-only view of a post-execution verification result.

    Records what the existing Phase 10 decision *was* — never computes it.
    """
    verified = bool(getattr(result, "financially_verified", False))
    decision = getattr(result, "decision", None)
    decision_value = str(getattr(decision, "value", decision) or "UNKNOWN")
    if verified:
        obs.record_counter(obs.MetricName.POST_EXECUTION_VERIFIED, 1)
    else:
        obs.record_counter(obs.MetricName.POST_EXECUTION_FAILED, 1, {"decision": decision_value})
    return {
        "verification.decision": decision_value,
        "verification.financially_verified": verified,
        "verification.exception_closed": bool(getattr(result, "exception_closed", False)),
        "verification.escalated": bool(getattr(result, "exception_escalated", False)),
    }
from app.ingestion.service import IngestionService
from app.models.adjustment import Adjustment as DBAdjustment
from app.models.exception import FinancialException
from app.models.fee import Fee as DBFee
from app.models.payment import Payment as DBPayment
from app.models.reconciliation_run import ReconciliationRun
from app.models.refund import Refund as DBRefund
from app.models.resolution import Resolution
from app.models.resolution_action import ResolutionAction
from app.models.settlement import Settlement as DBSettlement
from app.models.tax import Tax as DBTax
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.audit import (
    ActionMetadata,
    AuditEventType,
    FinalOutcome,
    VerificationMetadata,
)
from app.schemas.closed_loop import (
    ClosedLoopVerificationResult,
    VerificationDecision,
)
from app.schemas.enums import (
    AdjustmentType,
    Currency,
    FeeType,
    MatchStatus,
    PaymentStatus,
    RefundStatus,
    SettlementStatus,
    TaxType,
)
from app.schemas.execution import ExecutionResult, ExecutionStatus
from app.schemas.financial import (
    Adjustment as DomainAdjustment,
    Fee as DomainFee,
    Payment as DomainPayment,
    Refund as DomainRefund,
    Settlement as DomainSettlement,
    Tax as DomainTax,
)
from app.services.audit_log import AuditLogService
from app.services.execution import ResolutionExecutionService
from app.services.persistence import PersistenceService

#: Version of the Phase 10 verification orchestration, recorded on each run.
ENGINE_VERSION = "10.0.0"

#: Batch suffix keeping the post-execution result distinct from the
#: exception-time reconciliation result (PersistenceService keys results on
#: ``(case_id, batch_id)``).
POST_EXECUTION_BATCH_SUFFIX = "::POST_EXECUTION"


class PostExecutionStateError(RuntimeError):
    """Raised when fresh post-execution financial state cannot be established."""


class ClosedLoopClosureError(RuntimeError):
    """Raised when closure was demanded but the deterministic facts refuse it."""


class PostExecutionReconciliationService:
    """Orchestrates execution result -> fresh state -> reconciliation -> closure."""

    def __init__(
        self,
        session: Optional[Session] = None,
        execution_service: Optional[ResolutionExecutionService] = None,
        audit_log: Optional[AuditLogService] = None,
        ingestion_service: Optional[IngestionService] = None,
        persistence: Optional[PersistenceService] = None,
    ) -> None:
        self.session = session
        self.execution_service = execution_service or ResolutionExecutionService()
        self.audit_log = audit_log or AuditLogService()
        self._ingestion_factory = ingestion_service
        self._persistence = persistence

    # ------------------------------------------------------------------ public

    @obs.observed(
        obs.SpanName.POST_EXECUTION_RECONCILE,
        metric=obs.MetricName.POST_EXECUTION_RUNS,
        error_metric=obs.MetricName.POST_EXECUTION_FAILED,
        duration_metric=obs.MetricName.POST_EXECUTION_DURATION,
        attributes={"operation": "verify_and_close"},
        outcome_resolver=_closed_loop_outcome,
    )
    def verify_and_close(
        self,
        execution_result: ExecutionResult,
        *,
        exception: FinancialException,
        provider,
        case_id: str,
        batch_id: str,
        payment_id: str,
        run_id: Optional[str] = None,
        reconciliation_id: Optional[str] = None,
        resolution: Optional[Resolution] = None,
        resolution_action: Optional[ResolutionAction] = None,
        idempotency_key: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ClosedLoopVerificationResult:
        """Verify a Phase 9 execution against fresh provider state.

        Closure happens only when the existing reconciliation engine reports
        ``MATCHED`` with a zero difference on the *fresh* post-execution state.
        """
        verification_id = f"CLV-{uuid.uuid4().hex[:8].upper()}"
        reconciliation_id = reconciliation_id or f"REC-POST-{uuid.uuid4().hex[:8].upper()}"
        run_id = run_id or f"RUN-POST-{uuid.uuid4().hex[:8].upper()}"
        run_key = idempotency_key or f"{exception.id}:{execution_result.execution_id}"
        post_batch_id = f"{batch_id}{POST_EXECUTION_BATCH_SUFFIX}"

        base = dict(
            verification_id=verification_id,
            execution_id=execution_result.execution_id,
            action_id=execution_result.action_id,
            exception_id=execution_result.exception_id,
            case_id=case_id,
            run_id=run_id,
            reconciliation_id=reconciliation_id,
            exception_already_closed=exception.status == "CLOSED",
        )

        # ── Gate 1: provider success is required, and is not sufficient ──
        if execution_result.status not in (
            ExecutionStatus.EXECUTED,
            ExecutionStatus.VERIFICATION_PENDING,
            ExecutionStatus.VERIFIED,
        ):
            return self._deny(
                base,
                VerificationDecision.UNRESOLVED,
                f"execution not successful: {execution_result.status.value}",
                execution_result=execution_result,
            )

        if execution_result.error and execution_result.status is ExecutionStatus.EXECUTED:
            # An executed result that carries an error is not trustworthy.
            return self._deny(
                base,
                VerificationDecision.FAILED,
                f"execution reported an error: {execution_result.error}",
                execution_result=execution_result,
            )

        # ── Gate 2: fresh post-execution state (never after_state) ──
        try:
            self._refresh_provider_state(provider)
            inputs = self._load_reconciliation_inputs(payment_id)
        except PostExecutionStateError as error:
            return self._deny(
                base,
                VerificationDecision.UNRESOLVED,
                f"post-execution state unavailable: {error}",
                execution_result=execution_result,
            )
        except Exception as error:  # pragma: no cover - defensive
            return self._deny(
                base,
                VerificationDecision.UNRESOLVED,
                f"post-execution state retrieval failed: {error}",
                execution_result=execution_result,
            )

        # ── Gate 3: deterministic reconciliation is the sole arbiter ──
        try:
            engine_result = calculate_reconciliation(
                payment=inputs["payment"],
                settlements=inputs["settlements"],
                refunds=inputs["refunds"],
                fees=inputs["fees"],
                taxes=inputs["taxes"],
                adjustments=inputs["adjustments"],
                case_id=case_id,
                reconciliation_id=reconciliation_id,
            )
        except Exception as error:
            return self._deny(
                base,
                VerificationDecision.FAILED,
                f"reconciliation failed: {error}",
                execution_result=execution_result,
            )

        matched = (
            engine_result.match_status == MatchStatus.MATCHED
            and engine_result.difference == 0
        )

        facts = dict(
            expected_amount=engine_result.expected_amount,
            actual_amount=engine_result.actual_amount,
            difference=engine_result.difference,
            match_status=engine_result.match_status.value,
            exception_type=engine_result.exception_type.value,
            fresh_state_used=True,
        )

        if not matched:
            # Provider SUCCESS + reconciliation MISMATCH: never close.
            decision = VerificationDecision.FAILED
            reason = (
                f"reconciliation mismatch after execution: expected "
                f"{engine_result.expected_amount}, actual "
                f"{engine_result.actual_amount}, difference "
                f"{engine_result.difference}"
            )
            execution = self._transition_execution(
                execution_result, ExecutionStatus.VERIFICATION_FAILED
            )
            escalated = self._escalate_exception(exception, reason)
            persisted = self._persist_verification(
                engine_result=engine_result,
                post_batch_id=post_batch_id,
                scope_ref=exception.id,
                run_id=run_id,
                run_key=run_key,
                matched=False,
            )
            audit_ids = self._audit(
                exception=exception,
                case_id=case_id,
                workflow_id=workflow_id,
                decision=decision,
                final_outcome=FinalOutcome.VERIFICATION_FAILED,
                engine_result=engine_result,
                execution_result=execution_result,
                reason=reason,
            )
            return ClosedLoopVerificationResult(
                **base,
                **facts,
                decision=decision,
                financially_verified=False,
                execution_status=execution.status.value,
                exception_closed=False,
                exception_escalated=escalated,
                reconciliation_persisted=persisted,
                audit_event_ids=audit_ids,
                failure_reason=reason,
            )

        # ── MATCH: the only path to verification and closure ──
        try:
            action_verified = self._verify_action(resolution_action)
            resolution_verified = self._verify_resolution(resolution)
            closed = self._close_exception(exception, engine_result)
        except ClosedLoopClosureError as error:
            # Reconciliation matched but closure is not legally reachable.
            # Fail closed rather than force an illegal transition.
            execution = self._transition_execution(
                execution_result, ExecutionStatus.VERIFICATION_FAILED
            )
            reason = f"reconciliation matched but closure blocked: {error}"
            persisted = self._persist_verification(
                engine_result=engine_result,
                post_batch_id=post_batch_id,
                scope_ref=exception.id,
                run_id=run_id,
                run_key=run_key,
                matched=True,
            )
            audit_ids = self._audit(
                exception=exception,
                case_id=case_id,
                workflow_id=workflow_id,
                decision=VerificationDecision.HUMAN_REVIEW,
                final_outcome=FinalOutcome.VERIFICATION_FAILED,
                engine_result=engine_result,
                execution_result=execution_result,
                reason=reason,
            )
            return ClosedLoopVerificationResult(
                **base,
                **facts,
                decision=VerificationDecision.HUMAN_REVIEW,
                financially_verified=False,
                execution_status=execution.status.value,
                exception_closed=False,
                reconciliation_persisted=persisted,
                audit_event_ids=audit_ids,
                failure_reason=reason,
            )

        execution = self._transition_execution(
            execution_result, ExecutionStatus.VERIFIED
        )
        persisted = self._persist_verification(
            engine_result=engine_result,
            post_batch_id=post_batch_id,
            scope_ref=exception.id,
            run_id=run_id,
            run_key=run_key,
            matched=True,
        )
        audit_ids = self._audit(
            exception=exception,
            case_id=case_id,
            workflow_id=workflow_id,
            decision=VerificationDecision.VERIFIED,
            final_outcome=FinalOutcome.VERIFIED_SUCCESS,
            engine_result=engine_result,
            execution_result=execution_result,
            reason=None,
        )

        return ClosedLoopVerificationResult(
            **base,
            **facts,
            decision=VerificationDecision.VERIFIED,
            financially_verified=True,
            execution_status=execution.status.value,
            exception_closed=closed,
            exception_escalated=False,
            resolution_verified=resolution_verified,
            action_verified=action_verified,
            reconciliation_persisted=persisted,
            audit_event_ids=audit_ids,
            failure_reason=None,
        )

    # --------------------------------------------------------------- internals

    def _refresh_provider_state(self, provider) -> None:
        """Re-ingest the current provider snapshot so the DB mirrors reality.

        The provider — not the execution response — is authoritative. If the
        session or an ingestion run is unavailable this raises
        :class:`PostExecutionStateError` and the caller fails closed.
        """
        if self.session is None:
            raise PostExecutionStateError("no database session available")
        if provider is None:
            raise PostExecutionStateError("no provider available")
        # A provider whose read side fails mid-flight must not be trusted.
        for getter in (
            "get_merchants",
            "get_payments",
            "get_settlements",
            "get_fees",
            "get_refunds",
            "get_taxes",
            "get_adjustments",
        ):
            if hasattr(provider, getter):
                getattr(provider, getter)()
        ingestion = self._ingestion_factory or IngestionService(self.session)
        result = ingestion.ingest(provider)
        if getattr(result, "status", "COMPLETED") == "FAILED":
            raise PostExecutionStateError("ingestion reported FAILED")

    def _load_reconciliation_inputs(self, payment_id: str) -> dict:
        """Read current DB state and convert it to the engine's domain inputs."""
        payment_db = self.session.get(DBPayment, payment_id)
        if payment_db is None:
            raise PostExecutionStateError(
                f"payment {payment_id!r} not found in current state"
            )

        payment = DomainPayment(
            payment_id=payment_db.id,
            merchant_id=payment_db.merchant_id,
            amount=payment_db.amount,
            currency=Currency(payment_db.currency or "INR"),
            status=_safe_enum(PaymentStatus, payment_db.status, PaymentStatus.CAPTURED),
            payment_timestamp=_ts(payment_db.captured_at or payment_db.created_at),
        )

        settlements: List[DomainSettlement] = []
        for row in self.session.query(DBSettlement).filter_by(payment_id=payment_id):
            settlements.append(
                DomainSettlement(
                    settlement_id=row.id,
                    payment_id=row.payment_id,
                    merchant_id=row.merchant_id or payment.merchant_id,
                    amount=row.amount,
                    currency=Currency(row.currency or "INR"),
                    status=_safe_enum(
                        SettlementStatus, row.status, SettlementStatus.SETTLED
                    ),
                    settlement_timestamp=_ts(row.settled_at or row.created_at),
                )
            )

        refunds: List[DomainRefund] = []
        for row in self.session.query(DBRefund).filter_by(payment_id=payment_id):
            refunds.append(
                DomainRefund(
                    refund_id=row.id,
                    payment_id=row.payment_id,
                    amount=row.amount,
                    status=_safe_enum(RefundStatus, row.status, RefundStatus.PROCESSED),
                    refund_timestamp=_ts(row.refund_timestamp or row.created_at),
                )
            )

        fees: List[DomainFee] = []
        for row in self.session.query(DBFee).filter_by(payment_id=payment_id):
            fees.append(
                DomainFee(
                    fee_id=row.id,
                    payment_id=row.payment_id,
                    amount=row.amount,
                    fee_type=_safe_enum(FeeType, row.fee_type, FeeType.TRANSACTION),
                )
            )

        taxes: List[DomainTax] = []
        for row in self.session.query(DBTax).filter_by(payment_id=payment_id):
            taxes.append(
                DomainTax(
                    tax_id=row.id,
                    payment_id=row.payment_id,
                    amount=row.amount,
                    tax_type=_safe_enum(TaxType, row.tax_type, TaxType.GST),
                )
            )

        adjustments: List[DomainAdjustment] = []
        for row in self.session.query(DBAdjustment).filter_by(payment_id=payment_id):
            adjustments.append(
                DomainAdjustment(
                    adjustment_id=row.id,
                    payment_id=row.payment_id,
                    case_id=row.case_id,
                    amount=row.amount,
                    adjustment_type=_safe_enum(
                        AdjustmentType, row.adjustment_type, AdjustmentType.CORRECTION
                    ),
                )
            )

        return {
            "payment": payment,
            "settlements": settlements,
            "refunds": refunds,
            "fees": fees,
            "taxes": taxes,
            "adjustments": adjustments,
        }

    def _persist_verification(
        self,
        *,
        engine_result,
        post_batch_id: str,
        scope_ref: str,
        run_id: str,
        run_key: str,
        matched: bool,
    ) -> bool:
        """Persist the verification as a real ReconciliationRun + result."""
        if self.session is None:
            return False

        persistence = self._persistence or PersistenceService(self.session)
        now = datetime.utcnow()

        run = (
            self.session.query(ReconciliationRun)
            .filter_by(
                scope_type="EXCEPTION",
                scope_ref=scope_ref,
                trigger="POST_RESOLUTION",
                run_key=run_key,
            )
            .first()
        )
        if run is None:
            run = ReconciliationRun(
                id=run_id,
                scope_type="EXCEPTION",
                scope_ref=scope_ref,
                status="PENDING",
                trigger="POST_RESOLUTION",
                run_key=run_key,
                engine_version=ENGINE_VERSION,
                started_at=now,
                created_by="post_execution_reconciliation",
            )
            self.session.add(run)
            self.session.flush()

        if run.status == "PENDING":
            run.transition_to("RUNNING")
        # RUNNING (crashed mid-flight) and COMPLETED are left as-is: a
        # re-verification never restarts or rewrites a finished run.
        db_result = persistence.persist_reconciliation_result(
            engine_result, batch_id=post_batch_id
        )
        db_result.reconciliation_run_id = run.id

        run.matched_count = 1 if matched else 0
        run.exception_count = 0 if matched else 1
        run.counts = {"checked": 1, "matched": run.matched_count, "exceptions": run.exception_count}
        run.finished_at = now
        if run.status == "RUNNING":
            run.transition_to("COMPLETED")

        self.session.commit()
        return True

    def _transition_execution(
        self, execution_result: ExecutionResult, target: ExecutionStatus
    ) -> ExecutionResult:
        """Move the Phase 9 execution result to ``target`` legally.

        ``EXECUTED`` may not jump straight to ``VERIFIED`` — the execution
        machine requires ``EXECUTED -> VERIFICATION_PENDING -> VERIFIED |
        VERIFICATION_FAILED``. When the target is unreachable the current
        result is left untouched (never forced).
        """
        service = self.execution_service
        if execution_result.status == target:
            return execution_result
        if execution_result.status == ExecutionStatus.EXECUTED:
            try:
                service.transition_status(
                    execution_result, ExecutionStatus.VERIFICATION_PENDING
                )
            except Exception:
                return execution_result
        try:
            service.transition_status(execution_result, target)
        except Exception:
            pass
        return execution_result

    def _verify_action(self, action: Optional[ResolutionAction]) -> Optional[bool]:
        """ResolutionAction EXECUTED -> VERIFIED only when reconciliation passed."""
        if action is None:
            return None
        if action.status == "VERIFIED":
            return True
        if action.can_transition_to("VERIFIED"):
            action.transition_to("VERIFIED")
            action.verified_at = datetime.utcnow()
            return True
        return False

    def _verify_resolution(self, resolution: Optional[Resolution]) -> Optional[bool]:
        """Resolution EXECUTED -> VERIFIED only when reconciliation passed."""
        if resolution is None:
            return None
        if resolution.status == "VERIFIED":
            return True
        if resolution.can_transition_to("VERIFIED"):
            resolution.transition_to("VERIFIED")
            return True
        return False

    def _close_exception(self, exception: FinancialException, engine_result) -> bool:
        """Advance the exception to CLOSED along legal transitions only.

        ``CLOSED`` is reachable exclusively from ``RECONCILING`` (section 7.3),
        so the path is discovered from the state machine itself — never forced
        with a string assignment. An already-CLOSED exception is left alone
        (idempotent re-verification).
        """
        if exception.status == "CLOSED":
            return True

        path = _path_to_closed(exception.status)
        if path is None:
            raise ClosedLoopClosureError(
                f"no legal transition path from {exception.status!r} to CLOSED"
            )
        for target in path:
            exception.transition_to(target)

        exception.closed_at = datetime.utcnow()
        exception.close_reason = "POST_EXECUTION_RECONCILED"
        exception.status_reason = (
            f"post-execution reconciliation matched: expected "
            f"{engine_result.expected_amount} == actual "
            f"{engine_result.actual_amount}"
        )
        if self.session is not None:
            self.session.flush()
        return True

    def _escalate_exception(self, exception: FinancialException, reason: str) -> bool:
        """Escalate legally after a mismatch; never reopen a CLOSED exception."""
        exception.status_reason = reason
        if exception.status == "CLOSED":
            return False
        if exception.can_transition_to("ESCALATED"):
            exception.transition_to("ESCALATED")
            if self.session is not None:
                self.session.flush()
            return True
        return False

    def _audit(
        self,
        *,
        exception: FinancialException,
        case_id: str,
        workflow_id: Optional[str],
        decision: VerificationDecision,
        final_outcome: FinalOutcome,
        engine_result,
        execution_result: ExecutionResult,
        reason: Optional[str],
    ) -> List[str]:
        """Record structured verification facts through the existing audit log."""
        workflow = workflow_id or f"WF-CLOSED-LOOP-{exception.id}"
        verification_metadata = VerificationMetadata(
            difference_before=execution_result.before_state.difference,
            difference_after=engine_result.difference,
            discrepancy_eliminated=engine_result.difference == 0,
            verification_status=decision.value,
            verification_failure_reason=reason,
            expected_result={"expected_amount": engine_result.expected_amount},
            actual_result={"actual_amount": engine_result.actual_amount},
        )
        action_metadata = ActionMetadata(
            resolution_type=execution_result.resolution_type,
            requested_adjustment_paise=execution_result.requested_adjustment_paise,
            actual_adjustment_paise=execution_result.actual_adjustment_paise,
            execution_status=execution_result.status.value,
            idempotency_key=execution_result.idempotency_key,
            execution_id=execution_result.execution_id,
        )
        event = self.audit_log.create_event(
            AuditEventType.VERIFICATION_PERFORMED,
            workflow_id=workflow,
            exception_id=exception.id,
            case_id=case_id,
            candidate_id=execution_result.candidate_id,
            actor="post_execution_reconciliation",
            decision=decision.value,
            verification_metadata=verification_metadata,
            action_metadata=action_metadata,
            final_outcome=final_outcome,
            error=reason,
        )
        event_ids = [event.event_id]

        follow_up = (
            AuditEventType.RESOLUTION_VERIFIED
            if decision == VerificationDecision.VERIFIED
            else (
                AuditEventType.CASE_ESCALATED
                if exception.status == "ESCALATED"
                else AuditEventType.RESOLUTION_FAILED
            )
        )
        second = self.audit_log.create_event(
            follow_up,
            workflow_id=workflow,
            exception_id=exception.id,
            case_id=case_id,
            actor="post_execution_reconciliation",
            decision=decision.value,
            final_outcome=final_outcome,
            error=reason,
        )
        event_ids.append(second.event_id)
        return event_ids

    def _deny(
        self,
        base: dict,
        decision: VerificationDecision,
        reason: str,
        *,
        execution_result: ExecutionResult,
    ) -> ClosedLoopVerificationResult:
        """Fail closed: no verification, no closure, no fabricated state."""
        audit_ids: List[str] = []
        try:
            exception_id = base.get("exception_id") or ""
            workflow = f"WF-CLOSED-LOOP-{exception_id}"
            event = self.audit_log.create_event(
                AuditEventType.VERIFICATION_PERFORMED,
                workflow_id=workflow,
                exception_id=exception_id,
                case_id=base.get("case_id"),
                actor="post_execution_reconciliation",
                decision=decision.value,
                action_metadata=ActionMetadata(
                    resolution_type=execution_result.resolution_type,
                    execution_status=execution_result.status.value,
                    execution_id=execution_result.execution_id,
                    idempotency_key=execution_result.idempotency_key,
                ),
                final_outcome=FinalOutcome.UNRESOLVED,
                error=reason,
            )
            audit_ids.append(event.event_id)
        except Exception:  # pragma: no cover - audit must never mask a denial
            pass

        return ClosedLoopVerificationResult(
            **base,
            decision=decision,
            financially_verified=False,
            execution_status=execution_result.status.value,
            exception_closed=False,
            exception_escalated=False,
            reconciliation_persisted=False,
            audit_event_ids=audit_ids,
            failure_reason=reason,
        )


# --------------------------------------------------------------------- helpers


def _path_to_closed(status: str) -> Optional[List[str]]:
    """Shortest legal path from ``status`` to ``CLOSED`` (inclusive of CLOSED).

    Returns ``[]`` when already CLOSED, or ``None`` when no legal path exists.
    Prefers paths that do not route through ``ESCALATED``; the state machine
    remains the source of truth for what is legal.
    """
    if status == "CLOSED":
        return []
    path = _bfs(status, avoid={"ESCALATED"})
    if path is None:
        path = _bfs(status, avoid=set())
    return path


def _bfs(status: str, avoid: Iterable[str]) -> Optional[List[str]]:
    avoid = set(avoid)
    queue = deque([[status]])
    seen = {status}
    while queue:
        path = queue.popleft()
        current = path[-1]
        for nxt in sorted(allowed_transitions(EXCEPTION, current)):
            if nxt in avoid or nxt in seen:
                continue
            if nxt == "CLOSED":
                return path[1:] + ["CLOSED"]
            seen.add(nxt)
            queue.append(path + [nxt])
    return None


def _safe_enum(enum_cls, value, default):
    """Convert a stored string to its enum, tolerating legacy/unset values."""
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _ts(value) -> datetime:
    return value or datetime.utcnow()


__all__ = [
    "PostExecutionReconciliationService",
    "PostExecutionStateError",
    "ClosedLoopClosureError",
    "ENGINE_VERSION",
]
