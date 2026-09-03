"""
Exception Management Routes for Razorpay CloseLoop.

Thin API layer over existing exception services.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- GET  /exceptions                        — List exceptions with filtering
- GET  /exceptions/{id}                   — Get exception details
- POST /exceptions/{id}/resolve           — Submit a resolution
- POST /exceptions/{id}/approve           — Approve a resolution
- POST /exceptions/{id}/reject            — Reject a resolution
- POST /exceptions/{id}/escalate          — Escalate for human review
"""

from typing import Optional

from fastapi import APIRouter, Query

from app.api.dependencies import get_exception_service
from app.api.errors import ConflictException, NotFoundException, ValidationException
from app.api.schemas import (
    ApiResponse,
    ApproveRequest,
    EscalateRequest,
    ExceptionStatus,
    ExceptionType,
    RejectRequest,
    ResolveRequest,
    ResolveResponse,
    RiskCategory,
)
from app.api.errors import ErrorResponse

router = APIRouter(prefix="/exceptions", tags=["Exceptions"])

# Shared error responses
_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Exception not found"}}
_ERRORS_409 = {409: {"model": ErrorResponse, "description": "Conflict — invalid state transition or duplicate operation"}}
_ERRORS_422 = {422: {"model": ErrorResponse, "description": "Validation error"}}


@router.get(
    "",
    summary="List exceptions",
    description=(
        "Retrieve a paginated list of financial exceptions. "
        "Supports filtering by exception type, status, risk category, and batch ID. "
        "Results include financial discrepancy summary, risk level, and current status."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_422,
)
async def list_exceptions(
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    exception_type: Optional[ExceptionType] = Query(
        None, description="Filter by exception type (FEE_DIFFERENCE, REFUND_ADJUSTMENT, etc.)"
    ),
    status: Optional[ExceptionStatus] = Query(
        None, description="Filter by status (PENDING, RESOLVED, ESCALATED, etc.)"
    ),
    risk_category: Optional[RiskCategory] = Query(
        None, description="Filter by risk category (LOW, MEDIUM, HIGH, CRITICAL)"
    ),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
):
    """List financial exceptions with optional filtering.

    **Filtering**: All filter parameters are optional and can be combined.
    When multiple filters are provided, only exceptions matching ALL criteria are returned.

    **Pagination**: Use `limit` (max 500) and `offset` for large result sets.
    """
    svc = get_exception_service()
    exceptions = svc.list_exceptions(
        limit=limit,
        offset=offset,
        exception_type=exception_type.value if exception_type else None,
        status=status.value if status else None,
        risk_category=risk_category.value if risk_category else None,
        batch_id=batch_id,
    )
    return ApiResponse(
        success=True,
        data=exceptions,
        count=len(exceptions),
    )


@router.get(
    "/{exception_id}",
    summary="Get exception details",
    description=(
        "Retrieve detailed information about a specific financial exception, "
        "including financial discrepancy, classification, risk assessment, "
        "guardrail decision, and resolution status."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_exception(exception_id: str):
    """Get details of a specific exception.

    Returns comprehensive exception data including:
    - Financial discrepancy (expected vs actual amounts)
    - Exception type and classification confidence
    - Risk category and guardrail decision
    - Current status and resolution history
    """
    svc = get_exception_service()
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)
    return ApiResponse(success=True, data=exc)


@router.post(
    "/{exception_id}/resolve",
    summary="Resolve an exception",
    description=(
        "Submit a resolution PROPOSAL for an exception. The resolution includes "
        "a resolution type, financial adjustment amount, and supporting reason. "
        "CRITICAL: This records a proposal that must go through Phase 6 guardrails, "
        "execution, and verification before being considered a final resolution. "
        "The server-computed decision and verification are authoritative."
    ),
    response_model=ApiResponse,
    responses={
        **_ERRORS_404,
        **_ERRORS_409,
        **_ERRORS_422,
        403: {"model": ErrorResponse, "description": "Guardrail rejected the resolution"},
    },
)
async def resolve_exception(exception_id: str, request: ResolveRequest):
    """Submit a resolution proposal (does NOT bypass guardrails).

    **Resolution types**: REFUND_ADJUSTMENT, FEE_REVERSAL, SETTLEMENT_CORRECTION, etc.

    **CRITICAL SAFETY**: This endpoint records a PROPOSAL only. The server does NOT
    declare the resolution safe. Guardrail evaluation, execution, and verification
    must run before the resolution is considered successful.

    **Amount limit**: Adjustments are capped at ₹100,000 (10,000,000 paise).
    """
    svc = get_exception_service()
    # Check if exception exists first
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)

    # CRITICAL #1 FIX: Block re-proposal only for already-RESOLVED exceptions.
    # PENDING proposals can be replaced (overwrite previous proposal).
    if exc.get("status") == "RESOLVED":
        raise ConflictException(f"Exception '{exception_id}' is already resolved")
    # Note: Existing PENDING proposals are replaced with the new proposal.

    result = svc.resolve_exception(exception_id, request.model_dump())
    if "error" in result:
        raise ValidationException(result["error"])
    return ApiResponse(success=True, data=result)


@router.post(
    "/{exception_id}/approve",
    summary="Approve a resolution",
    description=(
        "Approve a pending resolution for an exception. This records the reviewer's "
        "approval through the Phase 9 feedback system. Only PENDING or RESOLVED "
        "exceptions can be approved."
    ),
    response_model=ApiResponse,
    responses={
        **_ERRORS_404,
        **_ERRORS_409,
    },
)
async def approve_exception(exception_id: str, request: ApproveRequest):
    """Approve a resolution.

    Records the approval with the reviewer identity and optional comments.
    The approval is linked to the exception for audit trail purposes.

    **State transitions**:
    - PENDING → APPROVED
    - RESOLVED → APPROVED
    """
    svc = get_exception_service()
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)

    if exc.get("status") not in ("RESOLVED", "PENDING"):
        raise ConflictException(
            f"Cannot approve exception in '{exc.get('status')}' status"
        )

    result = svc.approve_exception(exception_id, request.model_dump())
    if "error" in result:
        raise ValidationException(result["error"])
    return ApiResponse(success=True, data=result)


@router.post(
    "/{exception_id}/reject",
    summary="Reject a resolution",
    description=(
        "Reject a pending resolution for an exception. Requires a rejection reason. "
        "This records the rejection through the Phase 9 feedback system."
    ),
    response_model=ApiResponse,
    responses={
        **_ERRORS_404,
        **_ERRORS_409,
        **_ERRORS_422,
    },
)
async def reject_exception(exception_id: str, request: RejectRequest):
    """Reject a resolution.

    Records the rejection with the reviewer identity and mandatory reason.
    The rejection is linked to the exception for audit and learning purposes.

    **State transitions**:
    - PENDING → REJECTED
    - RESOLVED → REJECTED
    """
    svc = get_exception_service()
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)

    if exc.get("status") not in ("RESOLVED", "PENDING"):
        raise ConflictException(
            f"Cannot reject exception in '{exc.get('status')}' status"
        )

    result = svc.reject_exception(exception_id, request.model_dump())
    if "error" in result:
        raise ValidationException(result["error"])
    return ApiResponse(success=True, data=result)


@router.post(
    "/{exception_id}/escalate",
    summary="Escalate for human review",
    description=(
        "Escalate an exception for manual human review. Records the escalation "
        "reason and optional priority level. The exception is routed to the "
        "human review queue through the Phase 9 feedback system."
    ),
    response_model=ApiResponse,
    responses={
        **_ERRORS_404,
        **_ERRORS_422,
    },
)
async def escalate_exception(exception_id: str, request: EscalateRequest):
    """Escalate an exception for manual human review.

    Records the escalation with:
    - Mandatory reason explaining why human review is needed
    - Optional reviewer identity
    - Optional priority level (NORMAL, HIGH, URGENT)

    **State transitions**:
    - PENDING → ESCALATED
    - RESOLVED → ESCALATED
    """
    svc = get_exception_service()
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)

    result = svc.escalate_exception(
        exception_id,
        reason=request.reason,
        escalated_by=request.escalated_by,
    )
    if "error" in result:
        raise ValidationException(result["error"])
    return ApiResponse(success=True, data=result)
