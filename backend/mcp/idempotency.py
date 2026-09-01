"""
MCP Idempotency + Failure Safety for Razorpay CloseLoop Phase 11F.

Provides:
- Idempotent write operations (no duplicate financial actions)
- Timeout safety (query status, don't blindly retry)
- Partial failure handling
- Structured retry behavior

Safety principle:
  Idempotency is a HARD SAFETY GATE.
  Repeated requests with the same key must NEVER create duplicate actions.
  Timeout does NOT mean failure.
  No response does NOT mean not executed.
"""

import time
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
from typing import Any, Callable, Dict, Optional


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Operation Status
# ─────────────────────────────────────────────────────────────────────────────


class MCPOperationStatus(str, Enum):
    """Status of an MCP operation."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Operation Record
# ─────────────────────────────────────────────────────────────────────────────


class MCPOperationRecord:
    """Tracks the state of an MCP operation for idempotency."""

    def __init__(
        self,
        operation_id: str,
        idempotency_key: str,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> None:
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key
        self.tool_name = tool_name
        self.parameters = parameters
        self.status = MCPOperationStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.retry_count = 0
        self.last_retry_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# MCP Idempotency Store
# ─────────────────────────────────────────────────────────────────────────────


class MCPOperationsStore:
    """In-memory store for MCP operation idempotency.

    Tracks operations by idempotency key.
    Provides duplicate detection and status query.
    """

    def __init__(self) -> None:
        self._operations: Dict[str, MCPOperationRecord] = {}
        self._by_operation_id: Dict[str, str] = {}  # operation_id → idempotency_key

    def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[MCPOperationRecord]:
        """Get an operation by its idempotency key."""
        return self._operations.get(idempotency_key)

    def get_by_operation_id(
        self, operation_id: str
    ) -> Optional[MCPOperationRecord]:
        """Get an operation by its operation ID."""
        key = self._by_operation_id.get(operation_id)
        if key:
            return self._operations.get(key)
        return None

    def create_operation(
        self,
        idempotency_key: str,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> MCPOperationRecord:
        """Create a new operation record."""
        record = MCPOperationRecord(
            operation_id=_gen_id("OP"),
            idempotency_key=idempotency_key,
            tool_name=tool_name,
            parameters=parameters,
        )
        self._operations[idempotency_key] = record
        self._by_operation_id[record.operation_id] = idempotency_key
        return record

    def mark_in_progress(self, idempotency_key: str) -> None:
        """Mark operation as in progress."""
        record = self._operations.get(idempotency_key)
        if record:
            record.status = MCPOperationStatus.IN_PROGRESS
            record.started_at = datetime.now(timezone.utc)

    def mark_completed(
        self, idempotency_key: str, result: Dict[str, Any]
    ) -> None:
        """Mark operation as completed with result."""
        record = self._operations.get(idempotency_key)
        if record:
            record.status = MCPOperationStatus.COMPLETED
            record.result = result
            record.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, idempotency_key: str, error: str) -> None:
        """Mark operation as failed."""
        record = self._operations.get(idempotency_key)
        if record:
            record.status = MCPOperationStatus.FAILED
            record.error = error
            record.completed_at = datetime.now(timezone.utc)

    def mark_timed_out(self, idempotency_key: str) -> None:
        """Mark operation as timed out."""
        record = self._operations.get(idempotency_key)
        if record:
            record.status = MCPOperationStatus.TIMED_OUT
            record.completed_at = datetime.now(timezone.utc)

    def is_duplicate(self, idempotency_key: str) -> bool:
        """Check if a request with this key is already known."""
        record = self._operations.get(idempotency_key)
        if record is None:
            return False
        # If completed or failed, it's a duplicate (return existing result)
        # If in progress, it's a duplicate (wait for result)
        return record.status in (
            MCPOperationStatus.COMPLETED,
            MCPOperationStatus.FAILED,
            MCPOperationStatus.IN_PROGRESS,
        )

    def is_in_progress(self, idempotency_key: str) -> bool:
        """Check if a request is currently in progress."""
        record = self._operations.get(idempotency_key)
        return record is not None and record.status == MCPOperationStatus.IN_PROGRESS

    def remove(self, idempotency_key: str) -> None:
        """Remove an operation record (used for retry after failure)."""
        record = self._operations.pop(idempotency_key, None)
        if record:
            self._by_operation_id.pop(record.operation_id, None)

    @property
    def operation_count(self) -> int:
        return len(self._operations)


# ─────────────────────────────────────────────────────────────────────────────
# MCP Idempotent Executor
# ─────────────────────────────────────────────────────────────────────────────


class MCPOperationExecutor:
    """Wraps write tool handlers with idempotency.

    Ensures:
    - Same idempotency key → same result (no duplicate execution)
    - Timeout → status query, not blind retry
    - Partial failures handled safely
    """

    def __init__(self) -> None:
        self._store = MCPOperationsStore()

    @property
    def store(self) -> MCPOperationsStore:
        return self._store

    def execute_idempotent(
        self,
        idempotency_key: str,
        tool_name: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a write operation with idempotency.

        Behavior:
        1. If key already completed → return cached result
        2. If key already in progress → return existing result
        3. Otherwise → execute and store result
        """
        # Check for duplicate
        existing = self._store.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing.status == MCPOperationStatus.COMPLETED:
                # Return cached result
                return {
                    **(existing.result or {}),
                    "_idempotent": True,
                    "_cached": True,
                    "_operation_id": existing.operation_id,
                }
            elif existing.status == MCPOperationStatus.FAILED:
                return {
                    "error": existing.error or "Previous execution failed",
                    "_idempotent": True,
                    "_cached": True,
                    "_operation_id": existing.operation_id,
                }
            elif existing.status == MCPOperationStatus.IN_PROGRESS:
                return {
                    "status": "IN_PROGRESS",
                    "_idempotent": True,
                    "_cached": True,
                    "_operation_id": existing.operation_id,
                    "message": "Operation is already in progress. Query status to check.",
                }

        # Create operation record
        record = self._store.create_operation(idempotency_key, tool_name, parameters)
        self._store.mark_in_progress(idempotency_key)

        # Execute
        try:
            result = handler(parameters)
            self._store.mark_completed(idempotency_key, result)
            return {
                **result,
                "_idempotent": False,
                "_cached": False,
                "_operation_id": record.operation_id,
            }
        except Exception as e:
            self._store.mark_failed(idempotency_key, str(e))
            return {
                "error": f"Execution failed: {str(e)}",
                "_idempotent": False,
                "_cached": False,
                "_operation_id": record.operation_id,
            }

    def query_status(self, idempotency_key: str) -> Dict[str, Any]:
        """Query the status of an operation.

        CRITICAL SAFETY:
        - timeout = unknown status, NOT failure
        - no response = status unknown, NOT not-executed
        """
        record = self._store.get_by_idempotency_key(idempotency_key)
        if record is None:
            return {
                "status": "UNKNOWN",
                "message": "No operation found with this idempotency key. "
                           "The operation may not have been submitted.",
            }

        return {
            "operation_id": record.operation_id,
            "idempotency_key": record.idempotency_key,
            "tool_name": record.tool_name,
            "status": record.status.value,
            "result": record.result,
            "error": record.error,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "retry_count": record.retry_count,
        }

    def safe_retry(
        self,
        idempotency_key: str,
        tool_name: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """Safely retry an operation.

        CRITICAL SAFETY RULES:
        - timeout ≠ failure
        - no response ≠ not executed
        - Before retrying, check if operation actually completed
        """
        record = self._store.get_by_idempotency_key(idempotency_key)

        if record and record.status == MCPOperationStatus.COMPLETED:
            # Already completed — return cached result
            return {
                **(record.result or {}),
                "_idempotent": True,
                "_cached": True,
                "_operation_id": record.operation_id,
            }

        if record and record.status == MCPOperationStatus.IN_PROGRESS:
            # Still in progress — don't retry, just query
            return {
                "status": "IN_PROGRESS",
                "_idempotent": True,
                "_cached": True,
                "_operation_id": record.operation_id,
                "message": "Operation still in progress. Do NOT retry blindly.",
            }

        if record and record.status == MCPOperationStatus.TIMED_OUT:
            # Timed out — this means status is UNKNOWN
            # Do NOT blindly retry the financial action
            return {
                "status": "TIMED_OUT",
                "message": "Operation timed out. Status is UNKNOWN. "
                           "Query the backend service directly to determine actual state.",
                "_operation_id": record.operation_id,
            }

        if record and record.status == MCPOperationStatus.FAILED:
            # Previously failed — only retry if allowed
            if record.retry_count >= max_retries:
                return {
                    "error": f"Max retries ({max_retries}) exceeded",
                    "_operation_id": record.operation_id,
                }
            # Remove old record so execute_idempotent creates a fresh one
            self._store.remove(idempotency_key)

        # Execute
        return self.execute_idempotent(
            idempotency_key, tool_name, parameters, handler
        )
