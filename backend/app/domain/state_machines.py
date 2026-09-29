"""
Explicit state machines for CloseLoop financial entities.

Source of truth: ``CLOSELOOP_2.0_ARCHITECTURE.md`` section 7. This module is a
deliberately small, explicit implementation - one transition table per stateful
entity plus five tiny helpers. There is no generic framework, no registry
metaclass and no plugin system: the transition policy for each entity is a
literal dictionary that can be read top to bottom and asserted against in
tests.

Section 7 requires that invalid transitions raise. The domain layer must not
depend on FastAPI, so the error raised here is a plain
:class:`InvalidStateTransitionError` (a ``ValueError``). The existing API layer
already maps its own ``InvalidStateException`` to HTTP 409; a later phase that
wires these machines into the API can translate this error into that one
without the domain importing HTTP machinery.

Statuses are stored in the database as strings with ``CHECK`` constraints
(architecture section 6), so the tables below are keyed by plain strings. Enum
members from :mod:`app.schemas.enums` are accepted anywhere a status is expected
and are normalized to their string value.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Union

__all__ = [
    "InvalidStateTransitionError",
    "PAYMENT",
    "RECONCILIATION_RUN",
    "EXCEPTION",
    "RESOLUTION",
    "RESOLUTION_ACTION",
    "APPROVAL",
    "MACHINES",
    "TERMINAL_STATES",
    "machine_names",
    "known_states",
    "is_known_state",
    "allowed_transitions",
    "is_valid_transition",
    "assert_transition",
    "is_terminal",
    "next_status",
]

StatusLike = Union[str, Enum]


class InvalidStateTransitionError(ValueError):
    """Raised when a state transition is not permitted.

    Architecture section 7 names this failure mode ``InvalidStateException``;
    the API layer keeps its own HTTP-flavoured class of that name in
    ``app.api.errors``. This is the domain-layer equivalent so that models and
    future services can reject transitions without importing HTTP concerns.
    """

    def __init__(self, machine: str, current: str, new: str) -> None:
        self.machine = machine
        self.current = current
        self.new = new
        allowed = sorted(allowed_transitions(machine, current))
        super().__init__(
            f"Invalid {machine} transition: {current} -> {new} "
            f"(allowed from {current}: {allowed or 'none - terminal state'})"
        )


# =============================================================================
# Machine names
# =============================================================================

PAYMENT = "PAYMENT"
RECONCILIATION_RUN = "RECONCILIATION_RUN"
EXCEPTION = "EXCEPTION"
RESOLUTION = "RESOLUTION"
RESOLUTION_ACTION = "RESOLUTION_ACTION"
APPROVAL = "APPROVAL"


# =============================================================================
# Payment (architecture section 7.1 - a read-only mirror of provider state)
#
#   CREATED -> AUTHORIZED -> CAPTURED -> SETTLED
#   CAPTURED -> PARTIALLY_REFUNDED -> REFUNDED
#   any -> FAILED
#
# The frozen diagram states "any -> FAILED", so FAILED is reachable from every
# non-terminal state and is applied here literally.
# =============================================================================

_PAYMENT_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "PENDING": frozenset({"CREATED", "AUTHORIZED", "CAPTURED", "FAILED"}),
    "CREATED": frozenset({"PENDING", "AUTHORIZED", "CAPTURED", "FAILED"}),
    "AUTHORIZED": frozenset({"CAPTURED", "FAILED"}),
    "CAPTURED": frozenset(
        {
            "SETTLED",
            "PARTIALLY_REFUNDED",
            "REFUNDED",
            "CHARGEBACK",
            "FAILED",
        }
    ),
    "SETTLED": frozenset({"PARTIALLY_REFUNDED", "REFUNDED", "CHARGEBACK", "FAILED"}),
    "PARTIALLY_REFUNDED": frozenset({"REFUNDED", "CHARGEBACK", "FAILED"}),
    "REFUNDED": frozenset(),
    "CHARGEBACK": frozenset(),
    "FAILED": frozenset(),
}


# =============================================================================
# ReconciliationRun (architecture section 7.2)
#
#   PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
# =============================================================================

_RECONCILIATION_RUN_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "PENDING": frozenset({"RUNNING", "CANCELLED"}),
    "RUNNING": frozenset({"COMPLETED", "FAILED", "CANCELLED"}),
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
    "CANCELLED": frozenset(),
}


# =============================================================================
# Exception (architecture section 7.3)
#
#   DETECTED -> INVESTIGATING -> ANALYZED -> RESOLUTION_PROPOSED
#      -> AUTO_APPROVED -> EXECUTING -> VERIFYING -> RECONCILING -> CLOSED
#      -> HUMAN_REVIEW -> (APPROVED|REJECTED) -> EXECUTING -> ... -> CLOSED
#   Any pre-CLOSED state -> ESCALATED -> (re-enters INVESTIGATING on triage)
#   EXECUTING/VERIFYING failure -> FAILED -> (rollback) -> ROLLED_BACK -> HUMAN_REVIEW
#   RESOLUTION_PROPOSED blocked -> UNRESOLVED -> HUMAN_REVIEW
#   CLOSED is reachable *only* from RECONCILING.
#
# Legacy values retained for the pre-2.0 codebase: ``OPEN`` is the
# persistence-time synonym for ``DETECTED`` (it is the current column default
# and is written by existing reconciliation/evidence code), while ``MATCHED``
# and ``RESOLVED`` are legacy terminal outcomes.
# =============================================================================

_ESCALATABLE = {"ESCALATED"}

_EXCEPTION_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "OPEN": frozenset({"INVESTIGATING", "DETECTED", "ESCALATED"}),
    "DETECTED": frozenset({"INVESTIGATING", "ESCALATED"}),
    "INVESTIGATING": frozenset({"ANALYZED", "ESCALATED"}),
    "ANALYZED": frozenset({"RESOLUTION_PROPOSED", "ESCALATED"}),
    "RESOLUTION_PROPOSED": frozenset(
        {"AUTO_APPROVED", "HUMAN_REVIEW", "UNRESOLVED", "ESCALATED"}
    ),
    "AUTO_APPROVED": frozenset({"EXECUTING", "ESCALATED"}),
    "HUMAN_REVIEW": frozenset({"APPROVED", "REJECTED", "ESCALATED"}),
    "APPROVED": frozenset({"EXECUTING", "ESCALATED"}),
    "REJECTED": frozenset({"ESCALATED"}),
    "EXECUTING": frozenset({"VERIFYING", "FAILED", "ESCALATED"}),
    "VERIFYING": frozenset({"RECONCILING", "FAILED", "ESCALATED"}),
    "RECONCILING": frozenset({"CLOSED", "FAILED", "ESCALATED"}),
    "FAILED": frozenset({"ROLLED_BACK", "ESCALATED"}),
    "ROLLED_BACK": frozenset({"HUMAN_REVIEW", "ESCALATED"}),
    "UNRESOLVED": frozenset({"HUMAN_REVIEW", "ESCALATED"}),
    "ESCALATED": frozenset({"INVESTIGATING"}),
    # CLOSED is reachable only from RECONCILING - nothing may jump straight here.
    "CLOSED": frozenset(),
    # Legacy terminal states kept for rows written before CloseLoop 2.0.
    "MATCHED": frozenset(),
    "RESOLVED": frozenset(),
}


# =============================================================================
# Resolution (architecture section 7.4)
#
#   DRAFT -> PROPOSED -> POLICY_EVALUATED
#      -> AUTO_APPROVED | HUMAN_REVIEW | BLOCKED
#   AUTO_APPROVED / HUMAN_REVIEW+APPROVED -> EXECUTING -> EXECUTED -> VERIFIED | FAILED
#   FAILED -> ROLLBACK_PENDING -> ROLLED_BACK | ROLLBACK_FAILED
#   BLOCKED -> WITHDRAWN
# =============================================================================

_RESOLUTION_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "DRAFT": frozenset({"PROPOSED"}),
    "PROPOSED": frozenset({"POLICY_EVALUATED", "WITHDRAWN"}),
    "POLICY_EVALUATED": frozenset({"AUTO_APPROVED", "HUMAN_REVIEW", "BLOCKED"}),
    "AUTO_APPROVED": frozenset({"EXECUTING", "WITHDRAWN"}),
    "HUMAN_REVIEW": frozenset({"APPROVED", "REJECTED", "WITHDRAWN"}),
    "APPROVED": frozenset({"EXECUTING", "WITHDRAWN"}),
    "REJECTED": frozenset({"WITHDRAWN"}),
    # A blocked resolution can only be withdrawn.
    "BLOCKED": frozenset({"WITHDRAWN"}),
    "EXECUTING": frozenset({"EXECUTED", "FAILED"}),
    "EXECUTED": frozenset({"VERIFIED", "FAILED"}),
    "FAILED": frozenset({"ROLLBACK_PENDING", "WITHDRAWN"}),
    "ROLLBACK_PENDING": frozenset({"ROLLED_BACK", "ROLLBACK_FAILED"}),
    "VERIFIED": frozenset(),
    "WITHDRAWN": frozenset(),
    "ROLLED_BACK": frozenset(),
    "ROLLBACK_FAILED": frozenset(),
}


# =============================================================================
# ResolutionAction (architecture section 7.5)
#
#   PROPOSED -> VALIDATED -> APPROVED -> EXECUTING -> EXECUTED -> VERIFIED
#                           FAILED -> ROLLBACK_PENDING -> ROLLED_BACK | ROLLBACK_FAILED
# =============================================================================

_RESOLUTION_ACTION_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "PROPOSED": frozenset({"VALIDATED"}),
    "VALIDATED": frozenset({"APPROVED"}),
    "APPROVED": frozenset({"EXECUTING"}),
    "EXECUTING": frozenset({"EXECUTED", "FAILED"}),
    "EXECUTED": frozenset({"VERIFIED"}),
    "FAILED": frozenset({"ROLLBACK_PENDING"}),
    "ROLLBACK_PENDING": frozenset({"ROLLED_BACK", "ROLLBACK_FAILED"}),
    "VERIFIED": frozenset(),
    "ROLLED_BACK": frozenset(),
    "ROLLBACK_FAILED": frozenset(),
}


# =============================================================================
# Approval (architecture section 7.6)
#
#   PENDING -> APPROVED | REJECTED | EXPIRED | REVOKED
# =============================================================================

_APPROVAL_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "PENDING": frozenset({"APPROVED", "REJECTED", "EXPIRED", "REVOKED"}),
    "APPROVED": frozenset(),
    "REJECTED": frozenset(),
    "EXPIRED": frozenset(),
    "REVOKED": frozenset(),
}


# =============================================================================
# Registry
# =============================================================================

MACHINES: Dict[str, Dict[str, FrozenSet[str]]] = {
    PAYMENT: _PAYMENT_TRANSITIONS,
    RECONCILIATION_RUN: _RECONCILIATION_RUN_TRANSITIONS,
    EXCEPTION: _EXCEPTION_TRANSITIONS,
    RESOLUTION: _RESOLUTION_TRANSITIONS,
    RESOLUTION_ACTION: _RESOLUTION_ACTION_TRANSITIONS,
    APPROVAL: _APPROVAL_TRANSITIONS,
}

#: Derived: states with no outgoing transitions. Used by tests and by the
#: service layer to avoid retry loops against a finished entity.
TERMINAL_STATES: Dict[str, FrozenSet[str]] = {
    name: frozenset(state for state, nxt in table.items() if not nxt)
    for name, table in MACHINES.items()
}


# =============================================================================
# Helpers
# =============================================================================


def _norm(status: StatusLike) -> str:
    """Normalize an enum member or string to the stored string value."""
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)


def _table(machine: str) -> Dict[str, FrozenSet[str]]:
    try:
        return MACHINES[machine]
    except KeyError:
        raise KeyError(
            f"Unknown state machine {machine!r}. Known machines: "
            f"{sorted(MACHINES)}"
        ) from None


def machine_names() -> FrozenSet[str]:
    """All registered machine names."""
    return frozenset(MACHINES)


def known_states(machine: str) -> FrozenSet[str]:
    """Every state defined for ``machine``."""
    return frozenset(_table(machine))


def is_known_state(machine: str, status: StatusLike) -> bool:
    """True when ``status`` is a state of ``machine``."""
    return _norm(status) in _table(machine)


def allowed_transitions(machine: str, current: StatusLike) -> FrozenSet[str]:
    """Transitions permitted from ``current``; empty for terminal/unknown states."""
    return _table(machine).get(_norm(current), frozenset())


def is_valid_transition(
    machine: str, current: StatusLike, new: StatusLike
) -> bool:
    """True when ``current -> new`` is permitted for ``machine``."""
    return _norm(new) in allowed_transitions(machine, current)


def is_terminal(machine: str, status: StatusLike) -> bool:
    """True when no further transition is possible from ``status``."""
    current = _norm(status)
    if current not in _table(machine):
        return False
    return not _table(machine)[current]


def assert_transition(machine: str, current: StatusLike, new: StatusLike) -> str:
    """Validate ``current -> new`` and return the normalized new status.

    Raises:
        InvalidStateTransitionError: when the transition is not permitted.
    """
    current_s = _norm(current)
    new_s = _norm(new)
    if new_s not in allowed_transitions(machine, current_s):
        raise InvalidStateTransitionError(machine, current_s, new_s)
    return new_s


def next_status(machine: str, current: StatusLike, new: StatusLike) -> StatusLike:
    """Validate a transition and return ``new`` unchanged.

    Convenience for model methods that want to keep the caller's type::

        self.status = next_status(RECONCILIATION_RUN, self.status, status)
    """
    assert_transition(machine, current, new)
    return new
