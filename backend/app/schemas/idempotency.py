"""
Idempotency and Concurrency schemas for Razorpay CloseLoop Phase 8H.

Defines atomic idempotency operations and concurrency protection
to prevent duplicate financial adjustments.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Enums
# ─────────────────────────────────────────────────────────────────────────────


class IdempotencyStatus(str, Enum):
    """Status of an idempotency key."""
    AVAILABLE = "AVAILABLE"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class LockResult(str, Enum):
    """Result of a lock attempt."""
    ACQUIRED = "ACQUIRED"
    ALREADY_LOCKED = "ALREADY_LOCKED"
    CONFLICT = "CONFLICT"


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Record
# ─────────────────────────────────────────────────────────────────────────────


class IdempotencyRecord(BaseModel):
    """Record of an idempotency key's state."""
    key: str = Field(..., description="Idempotency key")
    status: IdempotencyStatus = Field(default=IdempotencyStatus.AVAILABLE)
    execution_id: Optional[str] = Field(default=None, description="Execution ID if completed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Cached result if completed")
    claimed_by: Optional[str] = Field(default=None, description="Worker/process that claimed the key")
    claimed_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConcurrencyLock(BaseModel):
    """A concurrency lock for a specific key."""
    key: str = Field(..., description="Lock key")
    owner: str = Field(..., description="Lock owner")
    acquired_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)


class ExecutionDeduplicationResult(BaseModel):
    """Result of deduplication check."""
    is_duplicate: bool = Field(..., description="Whether this is a duplicate request")
    existing_result: Optional[Dict[str, Any]] = Field(default=None, description="Existing result if duplicate")
    lock_acquired: bool = Field(default=False, description="Whether lock was acquired")
    worker_id: Optional[str] = Field(default=None, description="Worker that won the race")
