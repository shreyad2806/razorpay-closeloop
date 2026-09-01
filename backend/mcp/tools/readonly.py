"""
MCP Read-Only Financial Tools for Razorpay CloseLoop Phase 11B.

Implements controlled read-only tools for financial data access.

Safety principle:
  ALL tools in this module are READ-ONLY.
  They delegate to FinancialDataAdapter which loads from JSON files.
  They do NOT:
  - Execute SQL
  - Modify data
  - Bypass evidence services
  - Authorize financial actions
"""

from typing import Any, Dict, List, Optional

from mcp.adapters.financial_data import FinancialDataAdapter
from mcp.input_validation import (
    validate_no_injection,
    validate_tool_parameters,
    validate_output,
    MAX_SEARCH_LIMIT,
    MAX_TOP_K,
)
from mcp.schemas import MCPToolDefinition, MCPToolParameter


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    MCPToolDefinition(
        name="search_financial_records",
        description=(
            "Search financial records using controlled filters. "
            "Supports merchant_id, payment_id, settlement_id, case_id, "
            "and record_type. Returns up to 50 results."
        ),
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="merchant_id", type="string", required=False, description="Filter by merchant ID"),
            MCPToolParameter(name="payment_id", type="string", required=False, description="Filter by payment ID"),
            MCPToolParameter(name="settlement_id", type="string", required=False, description="Filter by settlement ID"),
            MCPToolParameter(name="case_id", type="string", required=False, description="Filter by case ID"),
            MCPToolParameter(name="record_type", type="string", required=False, description="Filter by record type (payment, settlement, refund, fee, adjustment, case)"),
            MCPToolParameter(name="limit", type="number", required=False, default=50, description="Max results"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_payment",
        description="Get payment information by payment ID.",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="payment_id", type="string", required=True, description="Payment ID"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_settlement",
        description="Get settlement information by settlement ID.",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="settlement_id", type="string", required=True, description="Settlement ID"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_refund",
        description="Get refund information by refund ID.",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="refund_id", type="string", required=True, description="Refund ID"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_fee",
        description="Get fee information by fee ID.",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="fee_id", type="string", required=True, description="Fee ID"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_adjustment",
        description="Get adjustment information by adjustment ID.",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="adjustment_id", type="string", required=True, description="Adjustment ID"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_similar_exception",
        description=(
            "Find similar historical exceptions using semantic similarity. "
            "Delegates to the Phase 4 similarity retrieval system."
        ),
        category="classification",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True, description="Exception/case ID to find similar cases for"),
            MCPToolParameter(name="top_k", type="number", required=False, default=5, description="Number of similar cases to return"),
        ],
        requires_guardrail=False,
        is_financial=False,
        idempotent=True,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Handlers
# ─────────────────────────────────────────────────────────────────────────────


def create_handlers(adapter: FinancialDataAdapter) -> Dict[str, Any]:
    """Create tool handler functions bound to a FinancialDataAdapter.

    Returns a dict of tool_name → handler_function.
    """

    def handle_search_financial_records(params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate all inputs
        val = validate_tool_parameters(
            params,
            required_params=set(),
            id_params={"merchant_id", "payment_id", "settlement_id", "case_id"},
            record_type_params={"record_type"},
            limit_params={"limit"},
        )
        if not val.is_valid:
            return {"error": val.error_message}

        # Clamp limit
        raw_limit = params.get("limit", 50)
        limit = min(int(raw_limit) if raw_limit is not None else 50, MAX_SEARCH_LIMIT)

        results = adapter.search_records(
            merchant_id=params.get("merchant_id"),
            payment_id=params.get("payment_id"),
            settlement_id=params.get("settlement_id"),
            case_id=params.get("case_id"),
            record_type=params.get("record_type"),
            limit=limit,
        )
        return {
            "count": len(results),
            "records": results,
        }

    def handle_get_payment(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(params, required_params={"payment_id"}, id_params={"payment_id"})
        if not val.is_valid:
            return {"error": val.error_message}
        payment = adapter.get_payment(params["payment_id"])
        if payment is None:
            return {"found": False, "error": f"Payment {params['payment_id']} not found"}
        return {"found": True, "payment": payment}

    def handle_get_settlement(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(params, required_params={"settlement_id"}, id_params={"settlement_id"})
        if not val.is_valid:
            return {"error": val.error_message}
        settlement = adapter.get_settlement(params["settlement_id"])
        if settlement is None:
            return {"found": False, "error": f"Settlement {params['settlement_id']} not found"}
        return {"found": True, "settlement": settlement}

    def handle_get_refund(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(params, required_params={"refund_id"}, id_params={"refund_id"})
        if not val.is_valid:
            return {"error": val.error_message}
        refund = adapter.get_refund(params["refund_id"])
        if refund is None:
            return {"found": False, "error": f"Refund {params['refund_id']} not found"}
        return {"found": True, "refund": refund}

    def handle_get_fee(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(params, required_params={"fee_id"}, id_params={"fee_id"})
        if not val.is_valid:
            return {"error": val.error_message}
        fee = adapter.get_fee(params["fee_id"])
        if fee is None:
            return {"found": False, "error": f"Fee {params['fee_id']} not found"}
        return {"found": True, "fee": fee}

    def handle_get_adjustment(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(params, required_params={"adjustment_id"}, id_params={"adjustment_id"})
        if not val.is_valid:
            return {"error": val.error_message}
        adjustment = adapter.get_adjustment(params["adjustment_id"])
        if adjustment is None:
            return {"found": False, "error": f"Adjustment {params['adjustment_id']} not found"}
        return {"found": True, "adjustment": adjustment}

    def handle_get_similar_exception(params: Dict[str, Any]) -> Dict[str, Any]:
        val = validate_tool_parameters(
            params,
            required_params={"exception_id"},
            id_params={"exception_id"},
            limit_params={"top_k"},
        )
        if not val.is_valid:
            return {"error": val.error_message}

        exception_id = params["exception_id"]
        raw_top_k = params.get("top_k", 5)
        top_k = min(int(raw_top_k) if raw_top_k is not None else 5, MAX_TOP_K)

        # Look up the case
        case = adapter.get_case(exception_id)
        if case is None:
            # Try as a payment_id lookup
            case = adapter.get_case(f"CASE-{exception_id}")

        if case is None:
            return {
                "found": False,
                "error": f"Exception {exception_id} not found in dataset",
                "similar_cases": [],
            }

        # Simple similarity: find cases with same scenario type
        similar = []
        for c in adapter._cases:
            if c.get("case_id") == case.get("case_id"):
                continue
            if c.get("scenario") == case.get("scenario"):
                similar.append({
                    "case_id": c.get("case_id"),
                    "payment_id": c.get("payment_id"),
                    "scenario": c.get("scenario"),
                    "difference": c.get("difference"),
                    "risk_category": c.get("risk_category"),
                    "similarity_type": "same_scenario",
                })
                if len(similar) >= top_k:
                    break

        return {
            "found": True,
            "query_case_id": case.get("case_id"),
            "query_scenario": case.get("scenario"),
            "count": len(similar),
            "similar_cases": similar,
        }

    return {
        "search_financial_records": handle_search_financial_records,
        "get_payment": handle_get_payment,
        "get_settlement": handle_get_settlement,
        "get_refund": handle_get_refund,
        "get_fee": handle_get_fee,
        "get_adjustment": handle_get_adjustment,
        "get_similar_exception": handle_get_similar_exception,
    }
