"""
Append-only enforcement for immutable CloseLoop records.

Architecture section 6 / section 20: ``audit_events`` is append-only (the app
role gets INSERT only) and corrections are *new* rows linked via
``correction_of``; ``ledger_entries`` are append-only and reversals are new
entries; ``feedback`` rows are immutable and corrections are new rows.

Phase 1 does not build the full immutability architecture (no DB roles, no
triggers). What it does establish is the foundational guarantee, enforced where
it is cheap and testable: SQLAlchemy refuses to UPDATE or DELETE these rows, so
no code path in the application can mutate history silently. The database-level
counterpart (INSERT-only role) lands with the production migration story.
"""

from __future__ import annotations

from typing import Type

from sqlalchemy import event

__all__ = ["ImmutableRecordError", "register_append_only"]


class ImmutableRecordError(RuntimeError):
    """Raised when code attempts to modify or delete an append-only record."""

    def __init__(self, model_name: str, operation: str) -> None:
        self.model_name = model_name
        self.operation = operation
        super().__init__(
            f"{model_name} is append-only: {operation} is not permitted. "
            f"Write a new row instead (use the correction/rollback linkage "
            f"column to reference the original record)."
        )


def register_append_only(*model_classes: Type[object]) -> None:
    """Block UPDATE and DELETE on ``model_classes`` at the ORM level.

    INSERT is untouched. ``Session.rollback`` is unaffected because a rollback
    discards the whole transaction rather than emitting an UPDATE/DELETE.
    """
    for model_class in model_classes:
        model_name = model_class.__name__

        @event.listens_for(model_class, "before_update")
        def _block_update(mapper, connection, target, _name=model_name):  # noqa: ANN001
            raise ImmutableRecordError(_name, "UPDATE")

        @event.listens_for(model_class, "before_delete")
        def _block_delete(mapper, connection, target, _name=model_name):  # noqa: ANN001
            raise ImmutableRecordError(_name, "DELETE")
