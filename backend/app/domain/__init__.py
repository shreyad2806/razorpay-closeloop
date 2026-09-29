"""
CloseLoop domain layer.

Phase 1 (domain + database foundation) introduces exactly one concern here:
the explicit, testable state machines that guard financial state transitions
(see CLOSELOOP_2.0_ARCHITECTURE.md section 7).

Nothing in this package performs I/O. It is pure validation logic so that the
service layer, the API layer and the models can all share one definition of
what a legal transition is.
"""

from app.domain import state_machines

__all__ = ["state_machines"]
