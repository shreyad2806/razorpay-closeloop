"""
Idempotency Store and Concurrency Guard for Phase 8H.

Provides atomic idempotency operations and concurrency protection
to prevent duplicate financial adjustments.

Uses a simulated database with locking semantics.
In production, this would use database-level constraints and transactions.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.schemas.execution import ExecutionResult, ExecutionStatus
from app.schemas.idempotency import (
    ConcurrencyLock,
    ExecutionDeduplicationResult,
    IdempotencyRecord,
    IdempotencyStatus,
    LockResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Store
# ─────────────────────────────────────────────────────────────────────────────


class IdempotencyStore:
    """Atomic idempotency store with locking semantics.

    Simulates database-level idempotency constraints.
    Uses threading locks for concurrent access protection.
    """

    def __init__(self):
        self._records: Dict[str, IdempotencyRecord] = {}
        self._locks: Dict[str, ConcurrencyLock] = {}
        self._lock = threading.Lock()  # Global lock for store operations
        self._key_locks: Dict[str, threading.Lock] = {}  # Per-key locks

    def _get_key_lock(self, key: str) -> threading.Lock:
        """Get or create a per-key lock."""
        if key not in self._key_locks:
            with self._lock:
                if key not in self._key_locks:
                    self._key_locks[key] = threading.Lock()
        return self._key_locks[key]

    def check_idempotency(self, key: str) -> Optional[IdempotencyRecord]:
        """Check if an idempotency key already has a result.

        Returns the record if exists, None if available.
        """
        with self._get_key_lock(key):
            record = self._records.get(key)
            if record and record.status == IdempotencyStatus.COMPLETED:
                return record
            return None

    def claim_key(
        self,
        key: str,
        worker_id: str,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Atomically claim an idempotency key.

        Returns True if claimed successfully, False if already claimed.
        """
        with self._get_key_lock(key):
            record = self._records.get(key)

            # If already completed, cannot claim
            if record and record.status == IdempotencyStatus.COMPLETED:
                return False

            # If already claimed by another worker, check expiry
            if record and record.status == IdempotencyStatus.CLAIMED:
                if record.claimed_at and record.claimed_at + timedelta(seconds=timeout_seconds) > datetime.utcnow():
                    return False  # Still locked
                # Lock expired, allow re-claim

            # Claim the key
            self._records[key] = IdempotencyRecord(
                key=key,
                status=IdempotencyStatus.CLAIMED,
                claimed_by=worker_id,
                claimed_at=datetime.utcnow(),
            )
            return True

    def complete_key(
        self,
        key: str,
        execution_result: ExecutionResult,
    ) -> None:
        """Mark an idempotency key as completed with the result."""
        with self._get_key_lock(key):
            record = self._records.get(key)
            if record:
                record.status = IdempotencyStatus.COMPLETED
                record.execution_id = execution_result.execution_id
                record.result = execution_result.model_dump(mode="json")
                record.completed_at = datetime.utcnow()

    def fail_key(self, key: str, error: str) -> None:
        """Mark an idempotency key as failed."""
        with self._get_key_lock(key):
            record = self._records.get(key)
            if record:
                record.status = IdempotencyStatus.FAILED
                record.result = {"error": error}
                record.completed_at = datetime.utcnow()

    def release_key(self, key: str) -> None:
        """Release a claimed key (for retry)."""
        with self._get_key_lock(key):
            record = self._records.get(key)
            if record and record.status == IdempotencyStatus.CLAIMED:
                record.status = IdempotencyStatus.AVAILABLE
                record.claimed_by = None
                record.claimed_at = None

    def get_record(self, key: str) -> Optional[IdempotencyRecord]:
        """Get the idempotency record for a key."""
        return self._records.get(key)

    def get_completed_count(self) -> int:
        """Get count of completed keys."""
        return sum(1 for r in self._records.values() if r.status == IdempotencyStatus.COMPLETED)


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency Guard
# ─────────────────────────────────────────────────────────────────────────────


class ConcurrencyGuard:
    """Guards against concurrent execution of the same resolution.

    Uses idempotency store for atomic claim/complete semantics.
    """

    def __init__(self, store: Optional[IdempotencyStore] = None):
        self.store = store or IdempotencyStore()

    def deduplicate(
        self,
        idempotency_key: str,
        worker_id: str,
        execution_result: Optional[ExecutionResult] = None,
    ) -> ExecutionDeduplicationResult:
        """Check for duplicates and acquire lock atomically.

        Returns deduplication result with whether to proceed.
        """
        # Check if already completed
        existing = self.store.check_idempotency(idempotency_key)
        if existing and existing.result:
            return ExecutionDeduplicationResult(
                is_duplicate=True,
                existing_result=existing.result,
                lock_acquired=False,
                worker_id=existing.claimed_by,
            )

        # Try to claim the key
        claimed = self.store.claim_key(idempotency_key, worker_id)
        if not claimed:
            # Could not claim — either locked or completed
            existing = self.store.check_idempotency(idempotency_key)
            if existing and existing.result:
                return ExecutionDeduplicationResult(
                    is_duplicate=True,
                    existing_result=existing.result,
                    lock_acquired=False,
                    worker_id=existing.claimed_by,
                )
            return ExecutionDeduplicationResult(
                is_duplicate=True,
                lock_acquired=False,
                worker_id=None,
            )

        return ExecutionDeduplicationResult(
            is_duplicate=False,
            lock_acquired=True,
            worker_id=worker_id,
        )

    def complete(
        self,
        idempotency_key: str,
        execution_result: ExecutionResult,
    ) -> None:
        """Mark execution as completed."""
        self.store.complete_key(idempotency_key, execution_result)

    def fail(
        self,
        idempotency_key: str,
        error: str,
    ) -> None:
        """Mark execution as failed."""
        self.store.fail_key(idempotency_key, error)

    def release(self, idempotency_key: str) -> None:
        """Release lock for retry."""
        self.store.release_key(idempotency_key)
