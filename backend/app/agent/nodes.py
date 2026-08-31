"""
Workflow nodes for Razorpay CloseLoop Phase 7B.

Each node performs work and returns state updates.
Nodes do not contain business logic — they delegate to services.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import (
    AgentState,
    NodeStatus,
    WorkflowMetadata,
    WorkflowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Node Result
# ─────────────────────────────────────────────────────────────────────────────


def _record_node_execution(
    state: AgentState,
    node_name: str,
    success: bool,
    error: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Record node execution metadata.

    Returns state updates for metadata fields.
    """
    elapsed_ms = None
    if start_time:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    log_entry = {
        "node": node_name,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms else None,
        "error": error,
    }

    # Update metadata
    metadata = state.metadata.model_dump()
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + [node_name]
    metadata["execution_log"] = list(state.metadata.execution_log) + [log_entry]

    if error:
        metadata["errors"] = list(state.metadata.errors) + [error]

    return {"metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# Load Exception Node
# ─────────────────────────────────────────────────────────────────────────────


def load_exception(state: AgentState) -> Dict[str, Any]:
    """Load exception data into agent state.

    Responsibilities:
    - Receive exception ID from state
    - Retrieve exception data (simulated for now)
    - Place exception into agent state
    - Record node execution metadata

    Does NOT:
    - Perform reconciliation
    - Perform ML
    - Generate resolution
    """
    start_time = time.perf_counter()
    node_name = "load_exception"

    exception_id = state.metadata.exception_id

    # Validate exception ID
    if not exception_id or not exception_id.strip():
        error_msg = "Missing or empty exception ID"
        updates = _record_node_execution(
            state, node_name, success=False, error=error_msg, start_time=start_time
        )
        updates["metadata"]["workflow_status"] = WorkflowStatus.FAILED.value
        return updates

    # Validate exception ID format
    if not exception_id.startswith("EXC-"):
        error_msg = f"Invalid exception ID format: {exception_id}"
        updates = _record_node_execution(
            state, node_name, success=False, error=error_msg, start_time=start_time
        )
        updates["metadata"]["workflow_status"] = WorkflowStatus.FAILED.value
        return updates

    # Simulate exception retrieval
    # In production, this would query the database
    exception_data = _simulate_exception_retrieval(exception_id)

    if exception_data is None:
        error_msg = f"Exception not found: {exception_id}"
        updates = _record_node_execution(
            state, node_name, success=False, error=error_msg, start_time=start_time
        )
        updates["metadata"]["workflow_status"] = WorkflowStatus.FAILED.value
        return updates

    # Success — place exception data into state
    updates = _record_node_execution(
        state, node_name, success=True, start_time=start_time
    )
    updates["metadata"]["workflow_status"] = WorkflowStatus.RUNNING.value
    updates["metadata"]["current_node"] = node_name

    # Store exception data (would be reconciliation input in production)
    updates["reconciliation_result"] = {
        "exception_id": exception_id,
        "case_id": exception_data.get("case_id"),
        "payment_id": exception_data.get("payment_id"),
        "merchant_id": exception_data.get("merchant_id"),
        "status": "LOADED",
    }

    return updates


def _simulate_exception_retrieval(exception_id: str) -> Optional[Dict[str, Any]]:
    """Simulate exception retrieval from database.

    In production, this would query PostgreSQL.
    """
    # Known test exceptions
    known_exceptions = {
        "EXC-001": {
            "case_id": "CASE-001",
            "payment_id": "PAY-001",
            "merchant_id": "MER-001",
            "exception_type": "FEE_DIFFERENCE",
        },
        "EXC-002": {
            "case_id": "CASE-002",
            "payment_id": "PAY-002",
            "merchant_id": "MER-001",
            "exception_type": "REFUND_ADJUSTMENT",
        },
        "EXC-003": {
            "case_id": "CASE-003",
            "payment_id": "PAY-003",
            "merchant_id": "MER-002",
            "exception_type": "UNKNOWN",
        },
    }
    return known_exceptions.get(exception_id)
