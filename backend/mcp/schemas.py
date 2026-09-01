"""
MCP Schemas for Razorpay CloseLoop Phase 11.

Defines the request/response types for MCP tool invocations.

Safety principle:
  MCP schemas describe the interface.
  They never authorize execution or bypass Phase 6 guardrails.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Tool Status
# ─────────────────────────────────────────────────────────────────────────────


class MCPToolStatus(str, Enum):
    """Status of an MCP tool invocation."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definition
# ─────────────────────────────────────────────────────────────────────────────


class MCPToolParameter(BaseModel):
    """Definition of a single tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, object)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Whether required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[str]] = Field(None, description="Allowed values")


class MCPToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str = Field(..., description="Tool name (unique)")
    description: str = Field(..., description="What the tool does")
    category: str = Field(..., description="Tool category")
    parameters: List[MCPToolParameter] = Field(
        default_factory=list, description="Tool parameters"
    )
    requires_guardrail: bool = Field(
        default=False, description="Whether this tool requires guardrail approval"
    )
    requires_verification: bool = Field(
        default=False, description="Whether this tool requires post-execution verification"
    )
    is_financial: bool = Field(
        default=False, description="Whether this tool performs financial actions"
    )
    idempotent: bool = Field(
        default=True, description="Whether this tool is idempotent"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────────────────────────────────────────


class MCPToolRequest(BaseModel):
    """Request to invoke an MCP tool."""
    tool_name: str = Field(..., description="Tool to invoke")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    request_id: Optional[str] = Field(None, description="Client-provided request ID")
    workflow_id: Optional[str] = Field(None, description="Calling workflow ID")
    agent_id: Optional[str] = Field(None, description="Agent/workflow identity")
    exception_id: Optional[str] = Field(None, description="Exception ID where available")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    timeout_seconds: Optional[float] = Field(None, description="Request timeout override")


class MCPToolResponse(BaseModel):
    """Response from an MCP tool invocation."""
    request_id: str = Field(..., description="Request ID")
    tool_name: str = Field(..., description="Tool that was invoked")
    status: MCPToolStatus = Field(..., description="Invocation status")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message if failed")
    guardrail_result: Optional[Dict[str, Any]] = Field(
        None, description="Guardrail check result if applicable"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp",
    )
    duration_ms: Optional[float] = Field(None, description="Invocation duration in ms")


# ─────────────────────────────────────────────────────────────────────────────
# MCP Audit Record
# ─────────────────────────────────────────────────────────────────────────────


class MCPAuditRecord(BaseModel):
    """Audit record for an MCP tool invocation.

    Every financial tool invocation must be traceable.
    Sensitive fields are masked before storage.
    """
    record_id: str = Field(..., description="Unique audit record ID")
    request_id: str = Field(..., description="Request ID")
    tool_name: str = Field(..., description="Tool invoked")
    category: str = Field(default="", description="Tool category")
    is_read_only: bool = Field(default=True, description="Whether tool is read-only")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Request parameters (sensitive fields masked)"
    )
    status: MCPToolStatus = Field(..., description="Result status")
    result_summary: Optional[str] = Field(None, description="Result summary")
    error: Optional[str] = Field(None, description="Error if failed")
    guardrail_checked: bool = Field(default=False, description="Whether guardrails were checked")
    guardrail_passed: bool = Field(default=True, description="Whether guardrails passed")
    is_financial: bool = Field(default=False, description="Whether financial action")
    # Identity fields
    workflow_id: Optional[str] = Field(None, description="Calling workflow ID")
    agent_id: Optional[str] = Field(None, description="Agent/workflow identity")
    exception_id: Optional[str] = Field(None, description="Exception ID where available")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Audit timestamp",
    )
    duration_ms: Optional[float] = Field(None, description="Duration in ms")
    # Write-specific fields (None for read-only tools)
    authorization_context: Optional[Dict[str, Any]] = Field(
        None, description="Authorization context (write tools only)"
    )
    guardrail_result: Optional[Dict[str, Any]] = Field(
        None, description="Guardrail result (write tools only)"
    )
    execution_result: Optional[str] = Field(
        None, description="Execution result (write tools only)"
    )
    verification_result: Optional[str] = Field(
        None, description="Verification result (write tools only)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Server Info
# ─────────────────────────────────────────────────────────────────────────────


class MCPServerInfo(BaseModel):
    """Information about the MCP server."""
    server_name: str = Field(..., description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    mode: str = Field(default="embedded", description="Operating mode")
    tool_count: int = Field(default=0, description="Registered tools")
    enabled_categories: List[str] = Field(
        default_factory=list, description="Enabled categories"
    )
    uptime_seconds: Optional[float] = Field(None, description="Server uptime")
