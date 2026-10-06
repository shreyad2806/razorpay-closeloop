"""
FastAPI authentication + RBAC dependencies for Razorpay CloseLoop Phase 12.

Typed dependencies for the existing route architecture:

    get_current_principal()              — authenticate the bearer token
    require_permission(Permission.X)     — authenticate + authorize an operation
    require_role(Role.Y)                 — authenticate + authorize by role

Authorization is fail-closed: a missing, malformed or unverifiable token raises
``401``; a valid token without the required permission raises ``403``. Identity
always originates from the validated token — never from a request payload.

Tests install a deterministic verifier with :func:`set_token_verifier`; no test
bypass exists in the verification path itself.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

from fastapi import Depends, Header, HTTPException, status

from app.auth.authentication import (
    AuthConfig,
    AuthenticationError,
    JwtTokenVerifier,
)
from app.auth.principal import Permission, Principal, Role
from app.auth.rbac import AccessDeniedError, require_any_permission, require_role

_token_verifier: Optional[JwtTokenVerifier] = None


def get_token_verifier() -> JwtTokenVerifier:
    """Return the process verifier, creating it from configuration on first use."""
    global _token_verifier
    if _token_verifier is None:
        _token_verifier = JwtTokenVerifier(AuthConfig.from_env())
    return _token_verifier


def set_token_verifier(verifier: Optional[JwtTokenVerifier]) -> None:
    """Replace the process verifier (used by deterministic test fixtures)."""
    global _token_verifier
    _token_verifier = verifier


def reset_token_verifier() -> None:
    """Reset to configuration-driven construction (used by test teardown)."""
    global _token_verifier
    _token_verifier = None


def _bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1]


def get_current_principal(
    authorization: Optional[str] = Header(default=None),
) -> Principal:
    """Authenticate the request and return the validated principal."""
    token = _bearer_token(authorization)
    verifier = get_token_verifier()
    try:
        return verifier.authenticate(token)
    except AuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"authentication failed: {error.reason}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def require_permission(
    permission: Union[Permission, Iterable[Permission]],
) -> Callable[..., Principal]:
    """Build a dependency that authenticates and authorizes an operation."""
    required: Iterable[Permission]
    if isinstance(permission, Permission):
        required = [permission]
    else:
        required = list(permission)

    def _dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        try:
            return require_any_permission(principal, required)
        except AccessDeniedError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"forbidden: {error.reason}",
            ) from error

    _dependency.__name__ = f"require_permission_{getattr(required[0], 'value', 'any')}"
    return _dependency


def require_rbac_role(*roles: Role) -> Callable[..., Principal]:
    """Build a dependency that authenticates and requires one of ``roles``."""

    def _dependency(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        try:
            return require_role(principal, *roles)
        except AccessDeniedError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"forbidden: {error.reason}",
            ) from error

    _dependency.__name__ = f"require_role_{ '_or_'.join(r.value for r in roles) }"
    return _dependency


__all__ = [
    "get_current_principal",
    "get_token_verifier",
    "set_token_verifier",
    "reset_token_verifier",
    "require_permission",
    "require_rbac_role",
]
