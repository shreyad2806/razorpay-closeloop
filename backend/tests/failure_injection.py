"""
Phase 16 — test-only failure injection infrastructure.

Every failure in Phase 16 is injected from *outside* production code: by
subclassing the provider boundary, by installing SQLAlchemy session events, by
monkeypatching the observability layer, or by handing a service a broken
collaborator. No production module contains a test switch.

Nothing in this module is imported by the application.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import event

from app.providers.dto import ProviderExecutionRequest, ProviderExecutionResult
from app.providers.mock_provider import MockProvider
from app.schemas.enums import ProviderExecutionStatus


# ============================================================================
# Provider boundary faults
# ============================================================================


class ProviderOutage(MockProvider):
    """A provider whose read or write side can be made to fail on demand.

    ``read_failure``  — every read call raises (the provider is unreachable).
    ``read_empty``    — every read returns an empty snapshot (data loss).
    ``write_failure`` — ``execute`` raises instead of reporting a status.
    ``write_status``  — force a specific terminal status for every mutation.
    ``malformed``     — ``execute`` returns a value that breaks the contract.
    ``delay_seconds`` — ``execute`` stalls before answering.
    ``write_calls``   — every mutation attempt, so tests can count mutations.
    """

    def __init__(
        self,
        seed: int = 42,
        *,
        read_failure: bool = False,
        read_empty: bool = False,
        write_failure: bool = False,
        write_status: Optional[ProviderExecutionStatus] = None,
        malformed: str = "",
        delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(seed=seed)
        self.read_failure = read_failure
        self.read_empty = read_empty
        self.write_failure = write_failure
        self.write_status = write_status
        self.malformed = malformed
        self.delay_seconds = delay_seconds
        self.write_calls: List[ProviderExecutionRequest] = []

    # -- read side -------------------------------------------------------

    def _read(self, records: list) -> list:
        if self.read_failure:
            raise RuntimeError("provider read side unavailable")
        if self.read_empty:
            return []
        return records

    def get_merchants(self):
        return self._read(super().get_merchants())

    def get_payments(self):
        return self._read(super().get_payments())

    def get_settlements(self):
        return self._read(super().get_settlements())

    def get_fees(self):
        return self._read(super().get_fees())

    def get_taxes(self):
        return self._read(super().get_taxes())

    def get_refunds(self):
        return self._read(super().get_refunds())

    # -- write side ------------------------------------------------------

    def execute(self, request: ProviderExecutionRequest) -> ProviderExecutionResult:
        self.write_calls.append(request)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.write_failure:
            raise RuntimeError("provider write side unavailable")
        if self.malformed == "none":
            return None  # type: ignore[return-value]
        if self.malformed == "wrong_type":
            return "SUCCESS"  # type: ignore[return-value]
        if self.malformed == "no_status":
            return type("Result", (), {"provider_reference_id": "x"})()  # type: ignore[return-value]
        if self.write_status is not None:
            return ProviderExecutionResult(
                status=self.write_status,
                amount_paise=0,
                error_code=f"FORCED_{self.write_status.value}",
                error_message=f"Forced {self.write_status.value} outcome",
                idempotency_key=request.idempotency_key,
                executed_at=datetime.now(timezone.utc),
            )
        return super().execute(request)

    # -- raw snapshot manipulation (corruption / version faults) ----------

    _TABLES = {
        "merchant": "_merchants",
        "payment": "_payments",
        "attempt": "_attempts",
        "settlement": "_settlements",
        "settlement_line": "_settlement_lines",
        "refund": "_refunds",
        "chargeback": "_chargebacks",
        "fee": "_fees",
        "tax": "_taxes",
    }

    def _table(self, kind: str) -> Dict[str, Any]:
        if kind not in self._TABLES:
            raise ValueError(f"unknown provider record kind: {kind}")
        return getattr(self, self._TABLES[kind])

    def inject(self, kind: str, key: str, record: Any) -> None:
        """Put a hand-made (often invalid) record into the provider snapshot."""
        self._table(kind)[key] = record

    def set_version(self, kind: str, key: str, version: int) -> None:
        """Re-report an existing record at a chosen version (stale/conflict)."""
        self._table(kind)[key].version = version

    def mutate_amount(self, kind: str, key: str, amount: int) -> None:
        """Change money behind CloseLoop's back (the external world moved)."""
        self._table(kind)[key].amount = amount

    # -- assertion helpers ------------------------------------------------

    @property
    def write_count(self) -> int:
        """How many mutation attempts reached the provider boundary."""
        return len(self.write_calls)

    def distinct_idempotency_keys(self) -> set:
        return {call.idempotency_key for call in self.write_calls}


# ============================================================================
# Database faults
# ============================================================================


class SessionFault:
    """Raise inside a SQLAlchemy session at a chosen point.

    Installed as a session event listener so the failure happens *inside* the
    unit of work — exactly where a real database error would surface.
    """

    #: Real SQLAlchemy SessionEvents — not invented hook names.
    POINTS = (
        "before_flush",
        "after_flush",
        "after_flush_postexec",
        "before_commit",
        "after_commit",
        "after_rollback",
    )

    def __init__(
        self,
        session,
        *,
        on: str = "before_flush",
        message: str = "injected database failure",
        times: int = 1,
    ) -> None:
        if on not in self.POINTS:
            raise ValueError(f"unsupported injection point: {on}")
        self.session = session
        self.on = on
        self.message = message
        self.remaining = times
        self.fired = 0

    def _maybe_raise(self, *args, **kwargs) -> None:
        if self.remaining <= 0:
            return
        self.remaining -= 1
        self.fired += 1
        raise RuntimeError(self.message)

    def __enter__(self) -> "SessionFault":
        event.listen(self.session, self.on, self._maybe_raise)
        return self

    def __exit__(self, *exc) -> None:
        event.remove(self.session, self.on, self._maybe_raise)


def session_fault(session, *, on: str = "before_flush", message: str = "injected database failure") -> SessionFault:
    """Context manager injecting one failure at ``on`` inside ``session``."""
    return SessionFault(session, on=on, message=message)


# ============================================================================
# Telemetry faults
# ============================================================================


@contextlib.contextmanager
def telemetry_outage(monkeypatch):
    """Simulate a telemetry backend that is completely unavailable.

    The injection point is the **registry**, which is exactly where the
    observability layer swallows failures (``_safe_record`` and ``_record``
    both guard their registry calls). The public helpers therefore keep their
    contract — they return a falsy "not recorded" value instead of raising —
    while every measurement is lost.
    """
    from app.core import observability as obs

    def _boom(*args, **kwargs):
        raise RuntimeError("telemetry backend unavailable")

    monkeypatch.setattr(obs.REGISTRY, "record_span", _boom, raising=True)
    monkeypatch.setattr(obs.REGISTRY, "record_metric", _boom, raising=True)
    yield obs


# ============================================================================
# Recovery classification
# ============================================================================

#: Failure classes derived from the existing state machines and service
#: boundaries — not invented retry policy.
RECOVERY_MATRIX: Dict[str, Dict[str, Any]] = {
    "PROVIDER_TIMEOUT": {
        "recoverable": False,
        "action": "MANUAL_RETRY_REQUIRED",
        "retry_allowed": False,
        "provider_mutation_possible": True,
        "reconciliation_required": True,
        "human_intervention_required": True,
        "rationale": "The provider may have mutated money; state is unknown until it is re-read.",
    },
    "PROVIDER_UNKNOWN": {
        "recoverable": False,
        "action": "ESCALATE",
        "retry_allowed": False,
        "provider_mutation_possible": True,
        "reconciliation_required": True,
        "human_intervention_required": True,
        "rationale": "Unknown outcomes must never be retried blindly.",
    },
    "PROVIDER_FAILED": {
        "recoverable": True,
        "action": "MANUAL_RETRY_REQUIRED",
        "retry_allowed": True,
        "provider_mutation_possible": True,
        "reconciliation_required": True,
        "human_intervention_required": True,
        "rationale": "A definitive rejection moved no money, but execution is not automatic.",
    },
    "PROVIDER_READ_FAILURE": {
        "recoverable": True,
        "action": "RETRY_LATER",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": True,
        "human_intervention_required": False,
        "rationale": "No mutation occurred; fresh state is required before closure.",
    },
    "DATABASE_FAILURE": {
        "recoverable": True,
        "action": "RETRY_TRANSACTION",
        "retry_allowed": True,
        "provider_mutation_possible": True,
        "reconciliation_required": True,
        "human_intervention_required": True,
        "rationale": "A provider-side mutation followed by a local failure is a "
        "distributed boundary; the database may be behind the provider.",
    },
    "AUTHORIZATION_FAILURE": {
        "recoverable": False,
        "action": "ESCALATE",
        "retry_allowed": False,
        "provider_mutation_possible": False,
        "reconciliation_required": False,
        "human_intervention_required": True,
        "rationale": "Authorization is a policy decision, not a transient fault.",
    },
    "EVIDENCE_INTEGRITY_FAILURE": {
        "recoverable": False,
        "action": "ESCALATE",
        "retry_allowed": False,
        "provider_mutation_possible": False,
        "reconciliation_required": False,
        "human_intervention_required": True,
        "rationale": "Corrupted evidence must be investigated, never re-fetched and trusted.",
    },
    "RECONCILIATION_FAILURE": {
        "recoverable": True,
        "action": "RETRY_LATER",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": True,
        "human_intervention_required": False,
        "rationale": "Financial truth was not established; the exception stays open.",
    },
    "ML_UNAVAILABLE": {
        "recoverable": True,
        "action": "DEGRADE_AND_CONTINUE",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": True,
        "human_intervention_required": False,
        "rationale": "ML is advisory; deterministic reconciliation remains authoritative.",
    },
    "RETRIEVAL_UNAVAILABLE": {
        "recoverable": True,
        "action": "DEGRADE_AND_CONTINUE",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": False,
        "human_intervention_required": False,
        "rationale": "Historical memory is advisory evidence, not financial truth.",
    },
    "TELEMETRY_UNAVAILABLE": {
        "recoverable": True,
        "action": "CONTINUE",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": True,
        "human_intervention_required": False,
        "rationale": "Telemetry is non-authoritative; business outcome is unchanged.",
    },
    "INGESTION_REJECTION": {
        "recoverable": True,
        "action": "REPLAY_INGESTION",
        "retry_allowed": True,
        "provider_mutation_possible": False,
        "reconciliation_required": True,
        "human_intervention_required": False,
        "rationale": "Invalid records are isolated; valid records are already stored.",
    },
}


def recovery_for(failure: str) -> Dict[str, Any]:
    """Recovery classification for a failure class (KeyError if unknown)."""
    return dict(RECOVERY_MATRIX[failure])


def classify(failure: str) -> str:
    """One of RECOVERABLE / MANUAL_RETRY / ESCALATE / REJECTED."""
    entry = recovery_for(failure)
    if entry["action"] == "CONTINUE" or entry["action"] == "DEGRADE_AND_CONTINUE":
        return "RECOVERABLE"
    if entry["action"] == "RETRY_LATER" or entry["action"] == "RETRY_TRANSACTION":
        return "RECOVERABLE"
    if entry["action"] == "REPLAY_INGESTION":
        return "RECOVERABLE"
    if entry["action"] == "MANUAL_RETRY_REQUIRED":
        return "MANUAL_RETRY"
    return "ESCALATE"
