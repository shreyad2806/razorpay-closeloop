"""
MCP Audit for Razorpay CloseLoop Phase 11D.

Provides comprehensive audit trail for all MCP tool invocations.

Safety principle:
  Audit records are append-only.
  Sensitive fields are masked.
  They never influence financial decisions.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from mcp.schemas import MCPAuditRecord, MCPToolStatus


# ─────────────────────────────────────────────────────────────────────────────
# Sensitive Field Masking
# ─────────────────────────────────────────────────────────────────────────────

# Field names that must be masked in audit logs
SENSITIVE_FIELD_NAMES: Set[str] = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "private_key", "credential", "connection_string", "database_url",
    "auth_token", "access_token", "refresh_token", "session_token",
    "secret_key", "encryption_key", "signing_key",
}

# Pattern to detect sensitive-looking values (long alphanumeric strings)
SENSITIVE_VALUE_PATTERN = re.compile(
    r"^(sk_live_|sk_test_|pk_live_|pk_test_|"  # Stripe-like keys
    r"ghp_|gho_|github_pat_|"                    # GitHub tokens
    r"AKIA[A-Z0-9]{16})"                         # AWS access keys
)

MASKED_VALUE = "***MASKED***"


def mask_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive fields in a parameters dictionary.

    Returns a new dict with sensitive values replaced.
    Original dict is NOT modified.
    """
    masked: Dict[str, Any] = {}
    for key, value in params.items():
        if key.lower() in SENSITIVE_FIELD_NAMES:
            masked[key] = MASKED_VALUE
        elif isinstance(value, str) and SENSITIVE_VALUE_PATTERN.match(value):
            masked[key] = MASKED_VALUE
        elif isinstance(value, dict):
            masked[key] = mask_parameters(value)
        elif isinstance(value, list):
            masked[key] = [
                mask_parameters(item) if isinstance(item, dict)
                else MASKED_VALUE if isinstance(item, str) and SENSITIVE_VALUE_PATTERN.match(item)
                else item
                for item in value
            ]
        else:
            masked[key] = value
    return masked


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────────────────────


class MCPAuditLogger:
    """Append-only audit logger for MCP invocations.

    Features:
    - All records are append-only (never modified after creation)
    - Sensitive fields are masked before storage
    - Supports correlation via request_id and workflow_id
    - Supports agent_id tracking
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._records: List[MCPAuditRecord] = []
        self._max_records = max_records

    def record(
        self,
        request_id: str,
        tool_name: str,
        category: str,
        parameters: Dict[str, Any],
        status: MCPToolStatus,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        guardrail_checked: bool = False,
        guardrail_passed: bool = True,
        is_financial: bool = False,
        is_read_only: bool = True,
        workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        duration_ms: Optional[float] = None,
        authorization_context: Optional[Dict[str, Any]] = None,
        guardrail_result: Optional[Dict[str, Any]] = None,
        execution_result: Optional[str] = None,
        verification_result: Optional[str] = None,
    ) -> MCPAuditRecord:
        """Record an audit entry with sensitive field masking.

        Sensitive parameters are masked before storage.
        """
        # Mask sensitive fields in parameters
        masked_params = mask_parameters(parameters)

        # Mask sensitive fields in authorization context
        masked_auth = (
            mask_parameters(authorization_context) if authorization_context else None
        )

        record = MCPAuditRecord(
            record_id=f"MCPAUD-{uuid4().hex[:12].upper()}",
            request_id=request_id,
            tool_name=tool_name,
            category=category,
            is_read_only=is_read_only,
            parameters=masked_params,
            status=status,
            result_summary=result_summary,
            error=error,
            guardrail_checked=guardrail_checked,
            guardrail_passed=guardrail_passed,
            is_financial=is_financial,
            workflow_id=workflow_id,
            agent_id=agent_id,
            exception_id=exception_id,
            idempotency_key=idempotency_key,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            authorization_context=masked_auth,
            guardrail_result=guardrail_result,
            execution_result=execution_result,
            verification_result=verification_result,
        )
        self._records.append(record)

        # Trim old records if at capacity
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        return record

    def query(
        self,
        tool_name: Optional[str] = None,
        status: Optional[MCPToolStatus] = None,
        workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MCPAuditRecord]:
        """Query audit records with optional filters.

        Supports correlation via request_id and workflow_id.
        """
        records = self._records

        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        if status:
            records = [r for r in records if r.status == status]
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]
        if exception_id:
            records = [r for r in records if r.exception_id == exception_id]
        if request_id:
            records = [r for r in records if r.request_id == request_id]

        return records[-limit:]

    @property
    def record_count(self) -> int:
        return len(self._records)

    def get_financial_actions(self, limit: int = 100) -> List[MCPAuditRecord]:
        """Get all audit records for financial actions."""
        return [r for r in self._records if r.is_financial][-limit:]

    def get_failed_requests(self, limit: int = 100) -> List[MCPAuditRecord]:
        """Get all failed audit records."""
        return [
            r for r in self._records
            if r.status in (MCPToolStatus.ERROR, MCPToolStatus.GUARDRAIL_BLOCKED)
        ][-limit:]

    def get_workflow_trace(self, workflow_id: str) -> List[MCPAuditRecord]:
        """Get all audit records for a specific workflow (correlation)."""
        return [r for r in self._records if r.workflow_id == workflow_id]

    def get_request_trace(self, request_id: str) -> List[MCPAuditRecord]:
        """Get all audit records for a specific request (correlation)."""
        return [r for r in self._records if r.request_id == request_id]

    def get_agent_trace(self, agent_id: str, limit: int = 100) -> List[MCPAuditRecord]:
        """Get all audit records for a specific agent."""
        return [r for r in self._records if r.agent_id == agent_id][-limit:]

    def get_exception_trace(self, exception_id: str) -> List[MCPAuditRecord]:
        """Get all audit records for a specific exception."""
        return [r for r in self._records if r.exception_id == exception_id]
