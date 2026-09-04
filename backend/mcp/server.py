"""
MCP Server for Razorpay CloseLoop Phase 11.

The MCP server exposes controlled financial tools to the LangGraph agent.

Architecture:
  LangGraph Agent
      ↓
  MCP Client (tool invocation)
      ↓
  MCP Server (routing + validation)
      ↓
  Tool Handlers → Internal Finance Services

Safety principle:
  MCP server is an INTEGRATION LAYER.
  It delegates to existing backend services.
  It does NOT contain financial business logic.
  Phase 6 guardrails remain the final safety authority.
"""

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.core.structured_logging import (
    WorkflowEvent, mcp_logger, set_correlation_ids,
)
from mcp.audit import MCPAuditLogger
from mcp.config import MCPServerConfig, MCPServerMode, MCPToolCategory
from mcp.idempotency import MCPOperationExecutor
from mcp.schemas import (
    MCPAuditRecord,
    MCPServerInfo,
    MCPToolDefinition,
    MCPToolRequest,
    MCPToolResponse,
    MCPToolStatus,
)
from mcp.tools.registry import MCPToolRegistry


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────────────────────


class MCPServer:
    """MCP server that exposes controlled financial tools.

    Responsibilities:
    - Accept tool invocations from LangGraph agent
    - Validate requests
    - Route to registered tool handlers
    - Record audit entries
    - Enforce access control (enabled categories, disabled tools)

    Does NOT:
    - Contain financial business logic
    - Bypass Phase 6 guardrails
    - Execute financial actions directly
    """

    def __init__(self, config: Optional[MCPServerConfig] = None) -> None:
        self._config = config or MCPServerConfig()
        self._registry = MCPToolRegistry()
        self._audit_logger = MCPAuditLogger()
        self._idempotency_executor = MCPOperationExecutor()
        self._audit_log: List[MCPAuditRecord] = []
        self._start_time = datetime.now(timezone.utc)
        self._request_count = 0
        self._error_count = 0
        mcp_logger.info(
            WorkflowEvent.MCP_SERVER_STARTED.value,
            f"MCP server initialized: {self._config.server_name}",
            server_name=self._config.server_name,
            mode=self._config.mode.value,
            enabled_categories=[c.value for c in self._config.enabled_categories],
        )

    @property
    def idempotency_executor(self) -> MCPOperationExecutor:
        return self._idempotency_executor

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def registry(self) -> MCPToolRegistry:
        return self._registry

    # ─────────────────────────────────────────────────────────────────────
    # Tool Registration
    # ─────────────────────────────────────────────────────────────────────

    def register_tool(
        self,
        definition: MCPToolDefinition,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Register a tool if its category is enabled and not disabled."""
        # Check category is enabled
        try:
            category = MCPToolCategory(definition.category)
            if category not in self._config.enabled_categories:
                return
        except ValueError:
            # Unknown category — allow registration but warn
            pass

        # Check tool is not disabled
        if definition.name in self._config.disabled_tools:
            return

        self._registry.register_tool(definition, handler)
        mcp_logger.debug(WorkflowEvent.STARTUP.value,
                       f"MCP tool registered: {definition.name}",
                       tool_name=definition.name,
                       category=definition.category)

    def register_tools(
        self,
        definitions_and_handlers: List[tuple],
    ) -> int:
        """Register multiple tools. Returns count registered."""
        count = 0
        for definition, handler in definitions_and_handlers:
            self.register_tool(definition, handler)
            count += 1
        return count

    # ─────────────────────────────────────────────────────────────────────
    # Tool Invocation
    # ─────────────────────────────────────────────────────────────────────

    def invoke(self, request: MCPToolRequest) -> MCPToolResponse:
        """Invoke a tool through the MCP server.

        Handles:
        - Request validation
        - Access control (category, disabled tools)
        - Delegation to registry
        - Audit logging
        - Error handling

        Structured log sequence per call:
          MCP_TOOL_CALLED → MCP_TOOL_COMPLETED | MCP_TOOL_FAILED
        """
        start_time = time.time()
        self._request_count += 1

        # Generate request ID if not provided
        if not request.request_id:
            request.request_id = _gen_id("REQ")

        set_correlation_ids(workflow_id=request.workflow_id or "",
                           exception_id=request.exception_id or "",
                           request_id=request.request_id)

        # ── MCP_TOOL_CALLED ──
        mcp_logger.info(
            WorkflowEvent.MCP_TOOL_CALLED.value,
            f"MCP tool called: {request.tool_name}",
            tool_name=request.tool_name,
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            exception_id=request.exception_id,
            parameter_keys=list(request.parameters.keys()) if request.parameters else [],
        )

        # Check tool exists
        definition = self._registry.get_definition(request.tool_name)
        if definition is None:
            self._error_count += 1
            response = MCPToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=MCPToolStatus.ERROR,
                error=f"Tool '{request.tool_name}' not registered",
            )
            duration_ms = (time.time() - start_time) * 1000
            mcp_logger.error(
                WorkflowEvent.MCP_TOOL_FAILED.value,
                f"MCP tool not registered: {request.tool_name}",
                tool_name=request.tool_name,
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
                duration_ms=round(duration_ms, 2),
                error_type="tool_not_registered",
                error_message=f"Tool '{request.tool_name}' not registered",
            )
            self._audit_request(request, response, start_time)
            return response

        # Check category is enabled
        try:
            category = MCPToolCategory(definition.category)
            if category not in self._config.enabled_categories:
                response = MCPToolResponse(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    status=MCPToolStatus.ERROR,
                    error=f"Category '{definition.category}' is disabled",
                )
                self._error_count += 1
                duration_ms = (time.time() - start_time) * 1000
                mcp_logger.error(
                    WorkflowEvent.MCP_TOOL_FAILED.value,
                    f"MCP tool category disabled: {request.tool_name}",
                    tool_name=request.tool_name,
                    request_id=request.request_id,
                    workflow_id=request.workflow_id,
                    exception_id=request.exception_id,
                    duration_ms=round(duration_ms, 2),
                    error_type="category_disabled",
                    error_message=f"Category '{definition.category}' is disabled",
                )
                self._audit_request(request, response, start_time)
                return response
        except ValueError:
            pass  # Unknown category, allow

        # Check tool not disabled
        if request.tool_name in self._config.disabled_tools:
            self._error_count += 1
            response = MCPToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=MCPToolStatus.ERROR,
                error=f"Tool '{request.tool_name}' is disabled",
            )
            duration_ms = (time.time() - start_time) * 1000
            mcp_logger.error(
                WorkflowEvent.MCP_TOOL_FAILED.value,
                f"MCP tool disabled: {request.tool_name}",
                tool_name=request.tool_name,
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
                duration_ms=round(duration_ms, 2),
                error_type="tool_disabled",
                error_message=f"Tool '{request.tool_name}' is disabled",
            )
            self._audit_request(request, response, start_time)
            return response

        # Delegate to registry
        response = self._registry.invoke_tool(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        response.duration_ms = duration_ms

        if response.status == MCPToolStatus.ERROR:
            self._error_count += 1
            # ── MCP_TOOL_FAILED ──
            mcp_logger.error(
                WorkflowEvent.MCP_TOOL_FAILED.value,
                f"MCP tool failed: {request.tool_name} — {response.error or 'unknown error'}",
                tool_name=request.tool_name,
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
                duration_ms=round(duration_ms, 2),
                error_type="tool_execution_error",
                error_message=str(response.error or "")[:300],
            )
        else:
            # ── MCP_TOOL_COMPLETED ──
            mcp_logger.success(
                WorkflowEvent.MCP_TOOL_COMPLETED.value,
                f"MCP tool completed: {request.tool_name}",
                tool_name=request.tool_name,
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
                duration_ms=round(duration_ms, 2),
                result_summary="success",
            )

        # Audit
        self._audit_request(request, response, start_time)

        return response

    # ─────────────────────────────────────────────────────────────────────
    # Audit
    # ─────────────────────────────────────────────────────────────────────

    def _audit_request(
        self,
        request: MCPToolRequest,
        response: MCPToolResponse,
        start_time: float,
    ) -> None:
        """Record an audit entry for a tool invocation.

        Sensitive fields are masked by the audit logger.
        Every tool call (read + write) generates an audit entry.
        """
        if not self._config.audit_all_requests:
            return

        definition = self._registry.get_definition(request.tool_name)
        is_read_only = not (definition.is_financial if definition else False)

        record = self._audit_logger.record(
            request_id=request.request_id or "unknown",
            tool_name=request.tool_name,
            category=definition.category if definition else "",
            parameters=request.parameters,
            status=response.status,
            result_summary=(
                "success"
                if response.status == MCPToolStatus.SUCCESS
                else response.error
            ),
            error=response.error,
            guardrail_checked=definition.requires_guardrail if definition else False,
            guardrail_passed=(
                response.guardrail_result is None
                or response.guardrail_result.get("passed", True)
            ) if response.guardrail_result is not None else True,
            is_financial=definition.is_financial if definition else False,
            is_read_only=is_read_only,
            workflow_id=request.workflow_id,
            agent_id=request.agent_id,
            exception_id=request.exception_id,
            idempotency_key=request.idempotency_key,
            duration_ms=(time.time() - start_time) * 1000,
            guardrail_result=response.guardrail_result,
        )
        self._audit_log.append(record)

    def get_audit_log(
        self,
        tool_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MCPAuditRecord]:
        """Get audit records with optional filters.

        Supports correlation via workflow_id, agent_id, exception_id, request_id.
        """
        return self._audit_logger.query(
            tool_name=tool_name,
            workflow_id=workflow_id,
            agent_id=agent_id,
            exception_id=exception_id,
            request_id=request_id,
            limit=limit,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Server Info
    # ─────────────────────────────────────────────────────────────────────

    def get_server_info(self) -> MCPServerInfo:
        """Get MCP server information."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return MCPServerInfo(
            server_name=self._config.server_name,
            version="1.0.0",
            mode=self._config.mode.value,
            tool_count=self._registry.tool_count,
            enabled_categories=[c.value for c in self._config.enabled_categories],
            uptime_seconds=uptime,
        )

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def error_count(self) -> int:
        return self._error_count
