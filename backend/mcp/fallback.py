"""
MCP Fallback Router for Razorpay CloseLoop Phase 11H.

Provides fallback behavior when MCP is unavailable.

Architecture:
  PRIMARY PATH (MCP available):
    LangGraph → MCPClient → MCPServer → Tool Handler → Backend Service

  FALLBACK PATH (MCP unavailable):
    LangGraph → FallbackRouter → InternalServiceAdapter → Backend Service

CRITICAL SAFETY RULE:
  Both paths MUST use the same underlying business services.
  Fallback does NOT create new finance logic.

  Read operations:  Fallback to internal adapter (same FinancialDataAdapter)
  Write operations: Fallback to ESCALATION (never uncontrolled direct-write)

Execution paths:
  MCP     — routed through MCP server
  INTERNAL — routed through internal adapter (fallback)

Write operations NEVER fall back to direct database writes.
When MCP is unavailable for writes, the system escalates safely.
"""

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from mcp.adapters.financial_data import FinancialDataAdapter
from mcp.client import MCPClient


# ─────────────────────────────────────────────────────────────────────────────
# Execution Path
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionPath(str, Enum):
    """How a tool invocation was executed."""
    MCP = "MCP"
    INTERNAL = "INTERNAL"
    ESCALATED = "ESCALATED"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Result
# ─────────────────────────────────────────────────────────────────────────────


class FallbackResult:
    """Result of a tool invocation with fallback tracking."""

    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_path: ExecutionPath = ExecutionPath.MCP,
        fallback_used: bool = False,
        tool_name: str = "",
        duration_ms: float = 0.0,
        request_id: str = "",
    ) -> None:
        self.success = success
        self.data = data
        self.error = error
        self.execution_path = execution_path
        self.fallback_used = fallback_used
        self.tool_name = tool_name
        self.duration_ms = duration_ms
        self.request_id = request_id

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": self.success,
            "tool_name": self.tool_name,
            "execution_path": self.execution_path.value,
            "fallback_used": self.fallback_used,
            "duration_ms": round(self.duration_ms, 2),
            "request_id": self.request_id,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal Service Adapter
# ─────────────────────────────────────────────────────────────────────────────


class InternalServiceAdapter:
    """Direct adapter to FinancialDataAdapter for fallback.

    Uses the SAME FinancialDataAdapter that the MCP tools delegate to.
    This ensures the fallback path uses identical business logic.

    Does NOT:
    - Execute SQL
    - Modify data
    - Bypass guardrails
    - Perform writes
    """

    def __init__(self, adapter: Optional[FinancialDataAdapter] = None) -> None:
        self._adapter = adapter or FinancialDataAdapter()
        # Auto-load default batch
        self._adapter.load_batch()

    @property
    def adapter(self) -> FinancialDataAdapter:
        return self._adapter

    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """Get payment — same adapter as MCP tools use."""
        return self._adapter.get_payment(payment_id)

    def get_settlement(self, settlement_id: str) -> Optional[Dict[str, Any]]:
        """Get settlement — same adapter as MCP tools use."""
        return self._adapter.get_settlement(settlement_id)

    def get_refund(self, refund_id: str) -> Optional[Dict[str, Any]]:
        """Get refund — same adapter as MCP tools use."""
        return self._adapter.get_refund(refund_id)

    def get_fee(self, fee_id: str) -> Optional[Dict[str, Any]]:
        """Get fee — same adapter as MCP tools use."""
        return self._adapter.get_fee(fee_id)

    def get_adjustment(self, adjustment_id: str) -> Optional[Dict[str, Any]]:
        """Get adjustment — same adapter as MCP tools use."""
        return self._adapter.get_adjustment(adjustment_id)

    def search_records(
        self,
        case_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        payment_id: Optional[str] = None,
        settlement_id: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search records — same adapter as MCP tools use."""
        return self._adapter.search_records(
            case_id=case_id,
            merchant_id=merchant_id,
            payment_id=payment_id,
            settlement_id=settlement_id,
            record_type=record_type,
            limit=limit,
        )

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get case — same adapter as MCP tools use."""
        return self._adapter.get_case(case_id)


# ─────────────────────────────────────────────────────────────────────────────
# MCP Fallback Router
# ─────────────────────────────────────────────────────────────────────────────


class MCPFallbackRouter:
    """Routes tool invocations through MCP primary or internal fallback.

    Read operations:
      1. Try MCP server
      2. If MCP fails → fallback to InternalServiceAdapter
      3. Both paths use the same FinancialDataAdapter

    Write operations:
      1. Try MCP server
      2. If MCP fails → ESCALATE (never direct-write)

    All invocations are recorded with execution_path for audit.
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        internal_adapter: Optional[InternalServiceAdapter] = None,
        mcp_available: bool = True,
    ) -> None:
        self._mcp_client = mcp_client or MCPClient()
        self._internal = internal_adapter or InternalServiceAdapter()
        self._mcp_available = mcp_available
        self._fallback_log: List[Dict[str, Any]] = []

    @property
    def mcp_client(self) -> MCPClient:
        return self._mcp_client

    @property
    def internal_adapter(self) -> InternalServiceAdapter:
        return self._internal

    @property
    def mcp_available(self) -> bool:
        return self._mcp_available

    @mcp_available.setter
    def mcp_available(self, value: bool) -> None:
        self._mcp_available = value

    @property
    def fallback_log(self) -> List[Dict[str, Any]]:
        return list(self._fallback_log)

    # ─────────────────────────────────────────────────────────────────────
    # Read Operations (fallback to internal adapter)
    # ─────────────────────────────────────────────────────────────────────

    def search_financial_records(
        self,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
        case_id: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 50,
    ) -> FallbackResult:
        """Search records with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="search_financial_records",
            mcp_params={
                "limit": limit,
                **({"case_id": case_id} if case_id else {}),
                **({"record_type": record_type} if record_type else {}),
            },
            fallback_fn=lambda: {
                "records": self._internal.search_records(
                    case_id=case_id,
                    record_type=record_type,
                    limit=limit,
                ),
                "count": len(
                    self._internal.search_records(
                        case_id=case_id,
                        record_type=record_type,
                        limit=limit,
                    )
                ),
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_payment(
        self,
        payment_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Get payment with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="get_payment",
            mcp_params={"payment_id": payment_id},
            fallback_fn=lambda: {
                "record": self._internal.get_payment(payment_id),
                "found": self._internal.get_payment(payment_id) is not None,
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_settlement(
        self,
        settlement_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Get settlement with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="get_settlement",
            mcp_params={"settlement_id": settlement_id},
            fallback_fn=lambda: {
                "record": self._internal.get_settlement(settlement_id),
                "found": self._internal.get_settlement(settlement_id) is not None,
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_refund(
        self,
        refund_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Get refund with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="get_refund",
            mcp_params={"refund_id": refund_id},
            fallback_fn=lambda: {
                "record": self._internal.get_refund(refund_id),
                "found": self._internal.get_refund(refund_id) is not None,
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_fee(
        self,
        fee_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Get fee with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="get_fee",
            mcp_params={"fee_id": fee_id},
            fallback_fn=lambda: {
                "record": self._internal.get_fee(fee_id),
                "found": self._internal.get_fee(fee_id) is not None,
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_adjustment(
        self,
        adjustment_id: str,
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Get adjustment with MCP-primary, internal-fallback."""
        return self._read_with_fallback(
            tool_name="get_adjustment",
            mcp_params={"adjustment_id": adjustment_id},
            fallback_fn=lambda: {
                "record": self._internal.get_adjustment(adjustment_id),
                "found": self._internal.get_adjustment(adjustment_id) is not None,
                "source": "internal_adapter",
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )

    def get_similar_exception(
        self,
        exception_id: str,
        workflow_id: Optional[str] = None,
        top_k: int = 5,
    ) -> FallbackResult:
        """Get similar exceptions — no internal fallback for ML retrieval."""
        if not self._mcp_available:
            return FallbackResult(
                success=False,
                error="MCP unavailable and no internal similarity service for fallback",
                execution_path=ExecutionPath.ESCALATED,
                fallback_used=True,
                tool_name="get_similar_exception",
                request_id=f"FALLBACK-{uuid4().hex[:8].upper()}",
            )

        result = self._mcp_client.call_tool(
            "get_similar_exception",
            parameters={"exception_id": exception_id, "top_k": top_k},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )
        return self._mcp_result_to_fallback(result)

    # ─────────────────────────────────────────────────────────────────────
    # Write Operations (escalate on MCP failure, never direct-write)
    # ─────────────────────────────────────────────────────────────────────

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
    ) -> FallbackResult:
        """Create resolution — escalate if MCP unavailable."""
        if not self._mcp_available:
            return self._escalate_write(
                tool_name="create_resolution",
                reason="MCP unavailable for write operation — cannot safely execute without MCP validation layer",
                workflow_id=workflow_id,
                exception_id=exception_id,
            )

        result = self._mcp_client.call_tool(
            "create_resolution",
            parameters={
                "exception_id": exception_id,
                "resolution_type": resolution_type,
                "financial_adjustment_paise": financial_adjustment_paise,
                "workflow_id": workflow_id,
                "guardrail_decision": guardrail_decision,
                "authorization_source": authorization_source,
                "idempotency_key": idempotency_key,
                **({"candidate_id": candidate_id} if candidate_id else {}),
            },
            workflow_id=workflow_id,
            exception_id=exception_id,
        )
        return self._mcp_result_to_fallback(result)

    def verify_resolution(
        self,
        execution_id: str,
        workflow_id: str,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Verify resolution — escalate if MCP unavailable."""
        if not self._mcp_available:
            return self._escalate_write(
                tool_name="verify_resolution",
                reason="MCP unavailable for verification — cannot verify without MCP validation layer",
                workflow_id=workflow_id,
                exception_id=exception_id,
            )

        result = self._mcp_client.call_tool(
            "verify_resolution",
            parameters={"execution_id": execution_id, "workflow_id": workflow_id},
            workflow_id=workflow_id,
            exception_id=exception_id,
        )
        return self._mcp_result_to_fallback(result)

    def record_feedback(
        self,
        workflow_id: str,
        exception_id: str,
        feedback_type: str,
        reviewer: str,
        system_prediction: str,
        reason: Optional[str] = None,
    ) -> FallbackResult:
        """Record feedback — escalate if MCP unavailable."""
        if not self._mcp_available:
            return self._escalate_write(
                tool_name="record_feedback",
                reason="MCP unavailable for feedback recording — cannot record without MCP validation layer",
                workflow_id=workflow_id,
                exception_id=exception_id,
            )

        params: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "exception_id": exception_id,
            "feedback_type": feedback_type,
            "reviewer": reviewer,
            "system_prediction": system_prediction,
        }
        if reason:
            params["reason"] = reason

        result = self._mcp_client.call_tool(
            "record_feedback",
            parameters=params,
            workflow_id=workflow_id,
            exception_id=exception_id,
        )
        return self._mcp_result_to_fallback(result)

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Read with Fallback
    # ─────────────────────────────────────────────────────────────────────

    def _read_with_fallback(
        self,
        tool_name: str,
        mcp_params: Dict[str, Any],
        fallback_fn: Callable[[], Dict[str, Any]],
        workflow_id: Optional[str] = None,
        exception_id: Optional[str] = None,
    ) -> FallbackResult:
        """Execute a read operation with MCP-primary, internal-fallback.

        1. If MCP available → try MCP
        2. If MCP fails → fallback to internal adapter
        3. Both use the same underlying FinancialDataAdapter
        """
        request_id = f"REQ-{uuid4().hex[:8].upper()}"
        start_time = time.time()

        # Try MCP first
        if self._mcp_available:
            try:
                result = self._mcp_client.call_tool(
                    tool_name=tool_name,
                    parameters=mcp_params,
                    workflow_id=workflow_id,
                    exception_id=exception_id,
                    request_id=request_id,
                )
                duration_ms = (time.time() - start_time) * 1000

                if result.get("success"):
                    return FallbackResult(
                        success=True,
                        data=result.get("data"),
                        execution_path=ExecutionPath.MCP,
                        fallback_used=False,
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        request_id=request_id,
                    )
                else:
                    # MCP failed → fallback
                    return self._execute_fallback(
                        tool_name=tool_name,
                        fallback_fn=fallback_fn,
                        mcp_error=result.get("error", "MCP returned error"),
                        workflow_id=workflow_id,
                        exception_id=exception_id,
                        request_id=request_id,
                        start_time=start_time,
                    )
            except Exception as e:
                # MCP threw exception → fallback
                return self._execute_fallback(
                    tool_name=tool_name,
                    fallback_fn=fallback_fn,
                    mcp_error=str(e),
                    workflow_id=workflow_id,
                    exception_id=exception_id,
                    request_id=request_id,
                    start_time=start_time,
                )

        # MCP unavailable → direct fallback
        return self._execute_fallback(
            tool_name=tool_name,
            fallback_fn=fallback_fn,
            mcp_error="MCP unavailable",
            workflow_id=workflow_id,
            exception_id=exception_id,
            request_id=request_id,
            start_time=start_time,
        )

    def _execute_fallback(
        self,
        tool_name: str,
        fallback_fn: Callable[[], Dict[str, Any]],
        mcp_error: str,
        workflow_id: Optional[str],
        exception_id: Optional[str],
        request_id: str,
        start_time: float,
    ) -> FallbackResult:
        """Execute the internal fallback path."""
        try:
            data = fallback_fn()
            duration_ms = (time.time() - start_time) * 1000

            # Record fallback event
            self._record_fallback(
                tool_name=tool_name,
                mcp_error=mcp_error,
                success=True,
                workflow_id=workflow_id,
                exception_id=exception_id,
            )

            return FallbackResult(
                success=True,
                data=data,
                execution_path=ExecutionPath.INTERNAL,
                fallback_used=True,
                tool_name=tool_name,
                duration_ms=duration_ms,
                request_id=request_id,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self._record_fallback(
                tool_name=tool_name,
                mcp_error=mcp_error,
                success=False,
                workflow_id=workflow_id,
                exception_id=exception_id,
                fallback_error=str(e),
            )

            return FallbackResult(
                success=False,
                error=f"MCP failed ({mcp_error}) and internal fallback failed ({e})",
                execution_path=ExecutionPath.ESCALATED,
                fallback_used=True,
                tool_name=tool_name,
                duration_ms=duration_ms,
                request_id=request_id,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Write Escalation
    # ─────────────────────────────────────────────────────────────────────

    def _escalate_write(
        self,
        tool_name: str,
        reason: str,
        workflow_id: Optional[str],
        exception_id: Optional[str],
    ) -> FallbackResult:
        """Escalate a write operation when MCP is unavailable.

        CRITICAL SAFETY RULE:
        Write operations NEVER fall back to direct database writes.
        They escalate to human review.
        """
        request_id = f"ESCALATE-{uuid4().hex[:8].upper()}"

        self._record_fallback(
            tool_name=tool_name,
            mcp_error="MCP unavailable for write operation",
            success=False,
            workflow_id=workflow_id,
            exception_id=exception_id,
            fallback_error=reason,
            escalated=True,
        )

        return FallbackResult(
            success=False,
            error=reason,
            execution_path=ExecutionPath.ESCALATED,
            fallback_used=True,
            tool_name=tool_name,
            request_id=request_id,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _mcp_result_to_fallback(self, result: Dict[str, Any]) -> FallbackResult:
        """Convert MCP client result to FallbackResult."""
        return FallbackResult(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error"),
            execution_path=ExecutionPath.MCP,
            fallback_used=False,
            tool_name=result.get("tool_name", ""),
            duration_ms=result.get("duration_ms", 0.0),
            request_id=result.get("request_id", ""),
        )

    def _record_fallback(
        self,
        tool_name: str,
        mcp_error: str,
        success: bool,
        workflow_id: Optional[str],
        exception_id: Optional[str],
        fallback_error: Optional[str] = None,
        escalated: bool = False,
    ) -> None:
        """Record a fallback event for audit."""
        self._fallback_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "mcp_error": mcp_error,
            "fallback_success": success,
            "fallback_error": fallback_error,
            "escalated": escalated,
            "workflow_id": workflow_id,
            "exception_id": exception_id,
        })

    # ─────────────────────────────────────────────────────────────────────
    # Audit
    # ─────────────────────────────────────────────────────────────────────

    def get_fallback_summary(self) -> Dict[str, Any]:
        """Get summary of all fallback events."""
        total = len(self._fallback_log)
        escalations = sum(1 for f in self._fallback_log if f["escalated"])
        successes = sum(1 for f in self._fallback_log if f["fallback_success"])

        return {
            "total_fallbacks": total,
            "successful_fallbacks": successes,
            "failed_fallbacks": total - successes,
            "escalations": escalations,
            "mcp_available": self._mcp_available,
            "tools_with_fallback": list(set(
                f["tool_name"] for f in self._fallback_log
            )),
        }
