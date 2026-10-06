"""
Exception Management Routes for Razorpay CloseLoop.

Thin API layer over existing exception services.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- GET  /exceptions                        — List exceptions with filtering
- GET  /exceptions/{id}                   — Get exception details
- GET  /exceptions/{id}/audit            — Get audit trail for exception (Phase 13)
- POST /exceptions/{id}/resolve           — Submit a resolution
- POST /exceptions/{id}/approve           — Approve a resolution
- POST /exceptions/{id}/reject            — Reject a resolution
- POST /exceptions/{id}/escalate          — Escalate for human review
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query

from app.api.auth_dependencies import require_permission
from app.auth.principal import Permission, Principal

from app.api.dependencies import get_exception_service, get_audit_service
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
from app.services.audit_log import AuditLogService

router = APIRouter(prefix="/exceptions", tags=["Exceptions"])

# Shared error responses
_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Exception not found"}}
_ERRORS_409 = {409: {"model": ErrorResponse, "description": "Conflict — invalid state transition or duplicate operation"}}
_ERRORS_422 = {422: {"model": ErrorResponse, "description": "Validation error"}}
_ERRORS_401 = {401: {"model": ErrorResponse, "description": "Authentication required"}}
_ERRORS_403 = {403: {"model": ErrorResponse, "description": "Permission denied"}}


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


@router.get(
    "/{exception_id}/audit",
    summary="Get audit trail",
    description=(
        "Retrieve the audit trail for a specific exception. "
        "Returns immutable audit events in chronological order. "
        "Actor identity comes from the validated audit record, not the current session."
    ),
    response_model=ApiResponse,
    responses={**_ERRORS_404, **_ERRORS_401, **_ERRORS_403},
)
async def get_exception_audit(
    exception_id: str,
    principal: Principal = Depends(require_permission(Permission.VIEW_AUDIT)),
) -> ApiResponse:
    """Get the audit trail for an exception.

    Returns all audit events for the exception, ordered by timestamp.
    The events are immutable and actor identity comes from the backend.
    Uses the in-memory AuditLogService for Phase 13.
    """
    # Verify exception exists
    svc = get_exception_service()
    exc = svc.get_exception(exception_id)
    if exc is None:
        raise NotFoundException("Exception", exception_id)

    # Get audit events from the shared service
    audit_svc = get_audit_service()

    # Get all events and filter by exception_id
    all_events = audit_svc.get_all_events()
    events = [e for e in all_events if e.exception_id == exception_id]

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    # Convert to serializable format using correct schema fields
    audit_data = [
        {
            "event_id": e.event_id,
            "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
            "exception_id": e.exception_id,
            "workflow_id": e.workflow_id,
            "actor": e.actor,
            "actor_type": e.actor_type.value if hasattr(e.actor_type, "value") else str(e.actor_type),
            "decision": e.decision,
            "confidence": e.confidence,
            "risk": e.risk,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "final_outcome": e.final_outcome.value if e.final_outcome else None,
            "error": e.error,
            "correction_of": e.correction_of,
            "correction_reason": e.correction_reason,
        }
        for e in events
    ]

    return ApiResponse(success=True, data=audit_data, count=len(audit_data))


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
async def resolve_exception(
    exception_id: str,
    request: ResolveRequest,
    principal: Principal = Depends(
        require_permission(Permission.INITIATE_RESOLUTION)
    ),
):
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
async def approve_exception(
    exception_id: str,
    request: ApproveRequest,
    principal: Principal = Depends(
        require_permission(Permission.APPROVE_RESOLUTION)
    ),
):
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

    # SECURITY: the approver identity comes from the authenticated principal,
    # never from the request payload. A contradictory payload identity is a
    # forgery attempt and is rejected.
    authenticated_actor = principal.actor_id()
    if request.approved_by and request.approved_by != authenticated_actor:
        raise ValidationException(
            "approved_by does not match the authenticated identity"
        )

    approval_data = request.model_dump()
    approval_data["approved_by"] = authenticated_actor
    result = svc.approve_exception(exception_id, approval_data)
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
async def reject_exception(
    exception_id: str,
    request: RejectRequest,
    principal: Principal = Depends(
        require_permission(Permission.REJECT_RESOLUTION)
    ),
):
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

    # SECURITY: the rejecting identity comes from the authenticated principal.
    authenticated_actor = principal.actor_id()
    if request.rejected_by and request.rejected_by != authenticated_actor:
        raise ValidationException(
            "rejected_by does not match the authenticated identity"
        )

    rejection_data = request.model_dump()
    rejection_data["rejected_by"] = authenticated_actor
    result = svc.reject_exception(exception_id, rejection_data)
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
        **_ERRORS_409,
        **_ERRORS_422,
    },
)
async def escalate_exception(
    exception_id: str,
    request: EscalateRequest,
    principal: Principal = Depends(
        require_permission(Permission.INITIATE_RESOLUTION)
    ),
):
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

    if exc.get("status") not in ("RESOLVED", "PENDING"):
        raise ConflictException(
            f"Cannot escalate exception in '{exc.get('status')}' status"
        )

    result = svc.escalate_exception(
        exception_id,
        reason=request.reason,
        escalated_by=principal.actor_id(),
    )
    if "error" in result:
        raise ValidationException(result["error"])
    return ApiResponse(success=True, data=result)
