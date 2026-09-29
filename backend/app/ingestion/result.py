"""
Structured outcome of an ingestion run.

``IngestionResult`` is the smallest reliable reporting mechanism the task
requires: per-entity created/updated/no-op counts, per-record validation
errors, and overall status. It is deliberately not a database model — Phase 2
does not add an ingestion-audit table (the counts are cheap to recompute by
re-running an idempotent ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class IngestionError:
    """One rejected provider record."""

    entity: str
    record_id: Optional[str]
    reason: str

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.entity}[{self.record_id}]: {self.reason}"


@dataclass
class IngestionResult:
    """Counts and errors from one ingestion run."""

    provider: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None

    # Per-entity counters: created (INSERT), updated (newer version), and
    # noop (same version already present). Older versions count as noop too —
    # the newer state was preserved.
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[IngestionError] = field(default_factory=list)

    # ``COMPLETED``: every record validated. ``PARTIAL``: some records were
    # rejected but valid records persisted. ``FAILED``: the run aborted.
    status: str = "COMPLETED"

    # -- counter helpers ------------------------------------------------------

    def count(self, entity: str, action: str) -> int:
        return self.counts.get(entity, {}).get(action, 0)

    def _bump(self, entity: str, action: str) -> None:
        bucket = self.counts.setdefault(
            entity, {"created": 0, "updated": 0, "noop": 0}
        )
        bucket[action] += 1

    def record_created(self, entity: str) -> None:
        self._bump(entity, "created")

    def record_updated(self, entity: str) -> None:
        self._bump(entity, "updated")

    def record_noop(self, entity: str) -> None:
        self._bump(entity, "noop")

    def record_error(self, entity: str, record_id: Optional[str], reason: str) -> None:
        self.errors.append(IngestionError(entity, record_id, reason))

    # -- status helpers ---------------------------------------------------------

    def finish(self, status: Optional[str] = None) -> "IngestionResult":
        self.completed_at = datetime.now(timezone.utc)
        if status is not None:
            self.status = status
        elif self.errors:
            self.status = "PARTIAL"
        return self

    @property
    def total_created(self) -> int:
        return sum(bucket["created"] for bucket in self.counts.values())

    @property
    def total_updated(self) -> int:
        return sum(bucket["updated"] for bucket in self.counts.values())

    @property
    def total_noop(self) -> int:
        return sum(bucket["noop"] for bucket in self.counts.values())

    def summary(self) -> dict:
        """Flat summary suitable for logs and API responses."""
        return {
            "provider": self.provider,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "counts": {entity: dict(bucket) for entity, bucket in self.counts.items()},
            "errors": [str(error) for error in self.errors],
            "total_created": self.total_created,
            "total_updated": self.total_updated,
            "total_noop": self.total_noop,
        }
