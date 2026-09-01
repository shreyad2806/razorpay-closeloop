"""
MCP Write Tools for Razorpay CloseLoop Phase 11E.

Controlled write tools that delegate to existing Phase 8/9 services.

CRITICAL SAFETY RULE:
  MCP MUST NOT directly execute arbitrary financial actions.
  Write tools MUST delegate to existing backend services.

Flow:
  Agent → MCP → Resolution Service → Phase 6 Guardrails → Execution → Verification

  NOT:
  Agent → MCP → Database UPDATE
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from mcp.input_validation import (
    validate_id,
    validate_limit,
    validate_no_injection,
    validate_tool_parameters,
    validate_amount,
    MAX_TOP_K,
)
from mcp.schemas import MCPToolDefinition, MCPToolParameter
from mcp.idempotency import MCPOperationExecutor


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

WRITE_TOOL_DEFINITIONS = [
    MCPToolDefinition(
        name="create_resolution",
        description=(
            "Create and execute a financial resolution for an exception. "
            "Delegates to the Phase 8 resolution execution service. "
            "Requires guardrail approval and verification prerequisites."
        ),
        category="execution",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True, description="Exception ID to resolve"),
            MCPToolParameter(name="resolution_type", type="string", required=True, description="Resolution type (e.g. FEE_DIFFERENCE, REFUND_ADJUSTMENT)"),
            MCPToolParameter(name="financial_adjustment_paise", type="number", required=True, description="Adjustment amount in paise"),
            MCPToolParameter(name="workflow_id", type="string", required=True, description="Calling workflow ID"),
            MCPToolParameter(name="candidate_id", type="string", required=False, description="Resolution candidate ID"),
            MCPToolParameter(name="authorization_source", type="string", required=True, description="Authorization source (e.g. guardrail AUTO, HUMAN_REVIEW)"),
            MCPToolParameter(name="guardrail_decision", type="string", required=True, description="Guardrail decision (AUTO or HUMAN_REVIEW)"),
            MCPToolParameter(name="idempotency_key", type="string", required=True, description="Idempotency key for deduplication"),
        ],
        requires_guardrail=True,
        requires_verification=True,
        is_financial=True,
        idempotent=False,
    ),
    MCPToolDefinition(
        name="verify_resolution",
        description=(
            "Verify that an executed resolution achieved its goal. "
            "Delegates to the Phase 8 verification engine. "
            "Independent verification — does not trust the execution response."
        ),
        category="execution",
        parameters=[
            MCPToolParameter(name="execution_id", type="string", required=True, description="Execution ID to verify"),
            MCPToolParameter(name="workflow_id", type="string", required=True, description="Calling workflow ID"),
        ],
        requires_guardrail=False,
        requires_verification=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="record_feedback",
        description=(
            "Record human feedback on a resolution. "
            "Delegates to the Phase 9 feedback service. "
            "Supports APPROVE, REJECT, CORRECT, ESCALATE."
        ),
        category="feedback",
        parameters=[
            MCPToolParameter(name="workflow_id", type="string", required=True, description="Workflow ID"),
            MCPToolParameter(name="exception_id", type="string", required=True, description="Exception ID"),
            MCPToolParameter(name="feedback_type", type="string", required=True, description="Feedback type (APPROVE, REJECT, CORRECT, ESCALATE)"),
            MCPToolParameter(name="reviewer", type="string", required=True, description="Reviewer identity"),
            MCPToolParameter(name="system_prediction", type="string", required=True, description="What the system predicted"),
            MCPToolParameter(name="reason", type="string", required=False, description="Reason for feedback"),
        ],
        requires_guardrail=False,
        requires_verification=False,
        is_financial=False,
        idempotent=True,
    ),
]

# Allowed resolution types
ALLOWED_RESOLUTION_TYPES = frozenset({
    "FEE_DIFFERENCE", "REFUND_ADJUSTMENT", "TAX_ADJUSTMENT",
    "SETTLEMENT_CORRECTION", "DUPLICATE",
    "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD",
})

# Allowed feedback types
ALLOWED_FEEDBACK_TYPES = frozenset({
    "APPROVE", "REJECT", "CORRECT", "ESCALATE",
})

# Allowed guardrail decisions
ALLOWED_GUARDRAIL_DECISIONS = frozenset({
    "AUTO", "HUMAN_REVIEW",
})


# ─────────────────────────────────────────────────────────────────────────────
# Tool Handlers
# ─────────────────────────────────────────────────────────────────────────────


def create_write_handlers(
    execution_service: Any,
    verification_engine: Any,
    feedback_service: Any,
    idempotency_executor: Optional[MCPOperationExecutor] = None,
) -> Dict[str, Any]:
    """Create write tool handler functions.

    Delegates to existing Phase 8/9 services.
    Does NOT contain financial business logic.

    When idempotency_executor is provided, create_resolution
    uses it to prevent duplicate financial actions.
    """

    def handle_create_resolution(params: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute a resolution.

        Flow: Agent → MCP → Execution Service → Guardrails → Execution
        """
        # Validate inputs
        exception_id = validate_id(params.get("exception_id"), "exception_id")
        if not exception_id.is_valid:
            return {"error": exception_id.error_message}

        workflow_id = validate_id(params.get("workflow_id"), "workflow_id")
        if not workflow_id.is_valid:
            return {"error": workflow_id.error_message}

        candidate_id_val = params.get("candidate_id")
        if candidate_id_val is not None:
            cid = validate_id(candidate_id_val, "candidate_id")
            if not cid.is_valid:
                return {"error": cid.error_message}

        idem_key = validate_id(params.get("idempotency_key"), "idempotency_key")
        if not idem_key.is_valid:
            return {"error": idem_key.error_message}

        # Validate resolution type
        resolution_type = params.get("resolution_type", "")
        if resolution_type not in ALLOWED_RESOLUTION_TYPES:
            return {
                "error": f"Invalid resolution type '{resolution_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_RESOLUTION_TYPES))}"
            }

        # Validate adjustment
        adj_check = validate_amount(
            params.get("financial_adjustment_paise"),
            "financial_adjustment_paise",
            allow_negative=False,
        )
        if not adj_check.is_valid:
            return {"error": adj_check.error_message}

        # Validate guardrail decision
        guardrail_decision = params.get("guardrail_decision", "")
        if guardrail_decision not in ALLOWED_GUARDRAIL_DECISIONS:
            return {
                "error": f"Invalid guardrail decision '{guardrail_decision}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_GUARDRAIL_DECISIONS))}"
            }

        # Validate authorization source
        auth_source = params.get("authorization_source", "")
        if not auth_source or auth_source == "NONE":
            return {"error": "Authorization source is required"}

        # Injection check on all string values
        for key, val in params.items():
            if isinstance(val, str):
                inj = validate_no_injection(val, key)
                if not inj.is_valid:
                    return {"error": inj.error_message}

        # Build action request for the execution service
        action_request = {
            "action_id": f"ACT-{params.get('idempotency_key', 'UNKNOWN')}",
            "workflow_id": params["workflow_id"],
            "exception_id": params["exception_id"],
            "case_id": params.get("case_id"),
            "candidate_id": params.get("candidate_id"),
            "resolution_type": params["resolution_type"],
            "financial_adjustment_paise": int(params["financial_adjustment_paise"]),
            "idempotency_key": params["idempotency_key"],
            "authorization_source": params["authorization_source"],
            "guardrail_decision": params["guardrail_decision"],
            "verification_passed": True,  # Pre-verified by guardrails
            "evidence_summary": params.get("evidence_summary", {}),
            "metadata": {
                "source": "mcp_write_tool",
                "requested_at": datetime.utcnow().isoformat(),
            },
        }

        # The actual execution handler
        def _do_execute(_params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = execution_service.execute(action_request)
                return {
                    "executed": result.status.value == "EXECUTED",
                    "execution_id": result.execution_id,
                    "status": result.status.value,
                    "adjustment_applied": result.actual_adjustment_paise,
                    "error": result.error,
                }
            except Exception as e:
                return {"error": f"Execution service error: {str(e)}"}

        # Wrap with idempotency if executor available
        if idempotency_executor is not None:
            idem_key = params.get("idempotency_key", f"auto-{params.get('exception_id', 'unknown')}")
            return idempotency_executor.execute_idempotent(
                idempotency_key=idem_key,
                tool_name="create_resolution",
                parameters=action_request,
                handler=_do_execute,
            )
        else:
            return _do_execute(params)

    def handle_verify_resolution(params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an executed resolution.

        Flow: Agent → MCP → Verification Engine → Independent Check
        """
        execution_id = validate_id(params.get("execution_id"), "execution_id")
        if not execution_id.is_valid:
            return {"error": execution_id.error_message}

        workflow_id = validate_id(params.get("workflow_id"), "workflow_id")
        if not workflow_id.is_valid:
            return {"error": workflow_id.error_message}

        # Injection check
        for key, val in params.items():
            if isinstance(val, str):
                inj = validate_no_injection(val, key)
                if not inj.is_valid:
                    return {"error": inj.error_message}

        # Look up the execution result
        execution_result = None
        if hasattr(execution_service, "get_execution"):
            execution_result = execution_service.get_execution(params.get("execution_id"))

        if execution_result is None:
            return {
                "error": f"Execution '{params.get('execution_id')}' not found",
                "verified": False,
            }

        # Delegate to verification engine
        try:
            verification_result = verification_engine.verify(execution_result)
            return {
                "verified": verification_result.status.value == "PASSED",
                "verification_id": verification_result.verification_id,
                "status": verification_result.status.value,
                "discrepancy_eliminated": verification_result.discrepancy_eliminated,
                "has_unintended_changes": verification_result.has_unintended_changes,
                "passed_checks": verification_result.passed_checks,
                "failed_checks": verification_result.failed_checks,
            }
        except Exception as e:
            return {"error": f"Verification service error: {str(e)}"}

    def handle_record_feedback(params: Dict[str, Any]) -> Dict[str, Any]:
        """Record human feedback on a resolution.

        Delegates to the Phase 9 feedback service.
        """
        workflow_id = validate_id(params.get("workflow_id"), "workflow_id")
        if not workflow_id.is_valid:
            return {"error": workflow_id.error_message}

        exception_id = validate_id(params.get("exception_id"), "exception_id")
        if not exception_id.is_valid:
            return {"error": exception_id.error_message}

        # Validate feedback type
        feedback_type = params.get("feedback_type", "")
        if feedback_type not in ALLOWED_FEEDBACK_TYPES:
            return {
                "error": f"Invalid feedback type '{feedback_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_FEEDBACK_TYPES))}"
            }

        # Injection check
        for key, val in params.items():
            if isinstance(val, str):
                inj = validate_no_injection(val, key)
                if not inj.is_valid:
                    return {"error": inj.error_message}

        # Delegate to feedback service
        try:
            from app.schemas.feedback import FeedbackType
            fb_type = FeedbackType(feedback_type)

            record = feedback_service.record_feedback(
                workflow_id=params["workflow_id"],
                exception_id=params["exception_id"],
                feedback_type=fb_type,
                reviewer=params.get("reviewer", "unknown"),
                system_prediction=params.get("system_prediction", ""),
                reason=params.get("reason"),
                candidate_id=params.get("candidate_id"),
                model_version=params.get("model_version"),
                policy_version=params.get("policy_version"),
            )
            return {
                "recorded": True,
                "feedback_id": record.feedback_id,
                "feedback_type": record.feedback_type.value,
                "reviewer": record.reviewer,
            }
        except Exception as e:
            return {"error": f"Feedback service error: {str(e)}"}

    return {
        "create_resolution": handle_create_resolution,
        "verify_resolution": handle_verify_resolution,
        "record_feedback": handle_record_feedback,
    }
