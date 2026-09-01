"""
MCP Client for Razorpay CloseLoop Phase 11G.

Wraps the MCPServer to provide a clean interface for the LangGraph agent.

Architecture:
  LangGraph Node
      ↓
  MCPClient.call_tool(name, params)
      ↓
  MCPServer.invoke(MCPToolRequest)
      ↓
  Tool Handler → Backend Service
      ↓
  MCPToolResponse → caller

Safety principle:
  The MCP Client is a delegation layer.
  It does NOT contain financial business logic.
  It does NOT bypass Phase 6 guardrails.
  All invocations are audited.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mcp.schemas import MCPToolRequest, MCPToolResponse, MCPToolStatus
from mcp.server import MCPServer


# ─────────────────────────────────────────────────────────────────────────────
# MCP Client
# ─────────────────────────────────────────────────────────────────────────────


class MCPClient:
    """Client for invoking MCP tools from LangGraph nodes.

    Provides a clean interface that wraps MCPServer.invoke().

    Responsibilities:
    - Convert simple call_tool() into MCPToolRequest
    - Route to MCPServer
    - Extract results from MCPToolResponse
    - Track invocation history for audit

    Does NOT:
    - Contain financial business logic
    - Bypass Phase 6 guardrails
    - Make authorization decisions
    """

    def __init__(self, server: Optional[MCPServer] = None) -> None:
        self._server = server or MCPServer()
        self._invocation_history: List[Dict[str, Any]] = []

    @property
    def server(self) -> MCPServer:
        return self._server

    @property
    def invocation_history(self) -> List[Dict[str, Any]]:
        return list(self._invocation_history)

    def call_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke an MCP tool and return the result.

        Args:
            tool_name: Name of the MCP tool to invoke
            parameters: Tool parameters
            workflow_id: Calling workflow ID (for audit)
            agent_id: Agent identity (for audit)
            exception_id: Exception being processed (for audit)
            request_id: Optional request ID for correlation

        Returns:
            Dict with tool result or error.
            Always includes 'success' and 'tool_name' keys.
        """
        start_time = time.time()
        parameters = parameters or {}

        request = MCPToolRequest(
            tool_name=tool_name,
            parameters=parameters,
            workflow_id=workflow_id,
            agent_id=agent_id,
            exception_id=exception_id,
            request_id=request_id or f"REQ-{uuid4().hex[:8].upper()}",
        )

        response = self._server.invoke(request)
        duration_ms = (time.time() - start_time) * 1000

        # Track invocation
        self._invocation_history.append({
            "tool_name": tool_name,
            "request_id": request.request_id,
            "workflow_id": workflow_id,
            "exception_id": exception_id,
            "status": response.status.value,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_error": response.status == MCPToolStatus.ERROR,
        })

        # Convert to simple dict
        return self._response_to_dict(response, duration_ms)

    def search_financial_records(
        self,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        payment_id: Optional[str] = None,
        settlement_id: Optional[str] = None,
        case_id: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Search financial records through MCP."""
        params: Dict[str, Any] = {"limit": limit}
        if merchant_id:
            params["merchant_id"] = merchant_id
        if payment_id:
            params["payment_id"] = payment_id
        if settlement_id:
            params["settlement_id"] = settlement_id
        if case_id:
            params["case_id"] = case_id
        if record_type:
            params["record_type"] = record_type

        return self.call_tool(
            "search_financial_records",
            parameters=params,
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_payment(
        self,
        payment_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get payment record through MCP."""
        return self.call_tool(
            "get_payment",
            parameters={"payment_id": payment_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_settlement(
        self,
        settlement_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get settlement record through MCP."""
        return self.call_tool(
            "get_settlement",
            parameters={"settlement_id": settlement_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_refund(
        self,
        refund_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get refund record through MCP."""
        return self.call_tool(
            "get_refund",
            parameters={"refund_id": refund_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_fee(
        self,
        fee_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get fee record through MCP."""
        return self.call_tool(
            "get_fee",
            parameters={"fee_id": fee_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_adjustment(
        self,
        adjustment_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get adjustment record through MCP."""
        return self.call_tool(
            "get_adjustment",
            parameters={"adjustment_id": adjustment_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_similar_exception(
        self,
        exception_id: str,
        top_k: int = 5,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get similar exceptions through MCP."""
        return self.call_tool(
            "get_similar_exception",
            parameters={"exception_id": exception_id, "top_k": top_k},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def create_resolution(
        self,
        exception_id: str,
        resolution_type: str,
        financial_adjustment_paise: int,
        workflow_id: str,
        guardrail_decision: str,
        authorization_source: str,
        idempotency_key: str,
        candidate_id: Optional[str] = None,
        exception_id_param: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a resolution through MCP (write operation)."""
        params: Dict[str, Any] = {
            "exception_id": exception_id,
            "resolution_type": resolution_type,
            "financial_adjustment_paise": financial_adjustment_paise,
            "workflow_id": workflow_id,
            "guardrail_decision": guardrail_decision,
            "authorization_source": authorization_source,
            "idempotency_key": idempotency_key,
        }
        if candidate_id:
            params["candidate_id"] = candidate_id

        return self.call_tool(
            "create_resolution",
            parameters=params,
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def verify_resolution(
        self,
        execution_id: str,
        workflow_id: str,
        exception_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a resolution through MCP."""
        return self.call_tool(
            "verify_resolution",
            parameters={"execution_id": execution_id, "workflow_id": workflow_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def record_feedback(
        self,
        workflow_id: str,
        exception_id: str,
        feedback_type: str,
        reviewer: str,
        system_prediction: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record feedback through MCP."""
        params: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "exception_id": exception_id,
            "feedback_type": feedback_type,
            "reviewer": reviewer,
            "system_prediction": system_prediction,
        }
        if reason:
            params["reason"] = reason

        return self.call_tool(
            "record_feedback",
            parameters=params,
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _response_to_dict(
        self, response: MCPToolResponse, duration_ms: float
    ) -> Dict[str, Any]:
        """Convert MCPToolResponse to simple dict."""
        result: Dict[str, Any] = {
            "success": response.status == MCPToolStatus.SUCCESS,
            "tool_name": response.tool_name,
            "request_id": response.request_id,
            "status": response.status.value,
            "duration_ms": round(duration_ms, 2),
        }

        if response.status == MCPToolStatus.SUCCESS:
            result["data"] = response.result
        elif response.status == MCPToolStatus.ERROR:
            result["error"] = response.error
        elif response.status == MCPToolStatus.VALIDATION_FAILED:
            result["error"] = response.error

        return result

    def get_invocation_count(self) -> int:
        """Get total number of tool invocations."""
        return len(self._invocation_history)

    def get_invocations_by_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get invocations for a specific tool."""
        return [h for h in self._invocation_history if h["tool_name"] == tool_name]

    def get_error_count(self) -> int:
        """Get total error count."""
        return sum(1 for h in self._invocation_history if h["is_error"])

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all invocations for audit."""
        return {
            "total_invocations": len(self._invocation_history),
            "error_count": self.get_error_count(),
            "success_count": len(self._invocation_history) - self.get_error_count(),
            "tools_called": list(set(h["tool_name"] for h in self._invocation_history)),
            "workflows_affected": list(set(
                h["workflow_id"] for h in self._invocation_history
                if h.get("workflow_id")
            )),
            "avg_duration_ms": (
                round(
                    sum(h["duration_ms"] for h in self._invocation_history)
                    / len(self._invocation_history),
                    2,
                )
                if self._invocation_history
                else 0
            ),
        }
