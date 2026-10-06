"""
Authentication Routes for Razorpay CloseLoop Phase 13.

Minimal authenticated endpoint exposing the existing Phase 12 Principal.

Endpoints:
- GET /auth/me — Current authenticated principal information
"""

from typing import List

from fastapi import APIRouter, Depends

from app.api.auth_dependencies import get_current_principal
from app.api.schemas import ApiResponse
from app.api.errors import ErrorResponse
from app.auth.principal import Permission, Principal, Role

router = APIRouter(prefix="/auth", tags=["Authentication"])

_ERROR_401 = {401: {"model": ErrorResponse, "description": "Authentication required"}}


@router.get(
    "/me",
    summary="Current principal",
    description=(
        "Return information about the currently authenticated principal. "
        "Identity is derived from the validated JWT token. "
        "Caller-supplied identity or role information is ignored."
    ),
    response_model=ApiResponse,
    responses=_ERROR_401,
)
async def get_current_principal_info(
    principal: Principal = Depends(get_current_principal),
) -> ApiResponse:
    """Return the current authenticated principal information.

    The principal is constructed from the validated JWT token by the
    authentication layer. No identity information is accepted from the
    request payload.
    """
    return ApiResponse(
        success=True,
        data={
            "subject": principal.subject,
            "actor_id": principal.actor_id(),
            "roles": [role.value for role in principal.roles],
            "permissions": [perm.value for perm in principal.permissions],
        },
    )
