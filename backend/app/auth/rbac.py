"""
Role-based access control enforcement for Razorpay CloseLoop Phase 12.

Every check is fail-closed: ALLOW is returned only when the required identity,
role and permission are *positively* established. Unknown roles, unknown
permissions, missing permissions and ambiguous identities all DENY.

RBAC is the user-facing authorization layer. It is separate from — and never a
substitute for — Phase 8 financial policy: holding a permission permits an
identity to *request* an operation, and the financial policy engine still
decides whether that specific financial action is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Union

from app.auth.principal import Permission, Principal, Role


class AccessDeniedError(Exception):
    """Raised when RBAC refuses an operation. Always fail-closed."""

    def __init__(self, reason: str, *, permission: Optional[Permission] = None) -> None:
        self.reason = reason
        self.permission = permission
        super().__init__(reason)


@dataclass(frozen=True)
class AccessDecision:
    """Explicit, auditable RBAC outcome."""

    allowed: bool
    principal_subject: Optional[str]
    permission: Optional[Permission]
    reason: str

    def is_allowed(self) -> bool:
        return self.allowed


def _principal_arg(principal: Optional[Principal]) -> Principal:
    if principal is None:
        raise AccessDeniedError("no authenticated principal supplied")
    if not isinstance(principal, Principal):
        raise AccessDeniedError("principal is not a validated identity object")
    if not principal.subject:
        raise AccessDeniedError("principal has no subject")
    return principal


def check_permission(
    principal: Optional[Principal],
    permission: Permission,
) -> AccessDecision:
    """Return an explicit ALLOW/DENY decision for ``permission``."""
    if not isinstance(permission, Permission):
        raise AccessDeniedError(f"unknown permission: {permission!r}", permission=None)

    resolved = _principal_arg(principal)
    if resolved.has_permission(permission):
        return AccessDecision(
            allowed=True,
            principal_subject=resolved.actor_id(),
            permission=permission,
            reason="permission granted by role membership",
        )
    return AccessDecision(
        allowed=False,
        principal_subject=resolved.actor_id(),
        permission=permission,
        reason=f"role(s) {[r.value for r in resolved.roles]} do not grant {permission.value}",
    )


def require_permission(principal: Optional[Principal], permission: Permission) -> Principal:
    """Return the principal when permitted; raise :class:`AccessDeniedError` otherwise."""
    decision = check_permission(principal, permission)
    if not decision.allowed:
        raise AccessDeniedError(decision.reason, permission=permission)
    return principal  # type: ignore[return-value]


def require_role(principal: Optional[Principal], *roles: Role) -> Principal:
    """Return the principal when it holds at least one of ``roles``; else deny."""
    for role in roles:
        if not isinstance(role, Role):
            raise AccessDeniedError(f"unknown role: {role!r}")
    resolved = _principal_arg(principal)
    if any(resolved.has_role(role) for role in roles):
        return resolved
    held = [r.value for r in resolved.roles]
    wanted = [r.value for r in roles]
    raise AccessDeniedError(f"requires role {wanted}, principal holds {held}")


def require_any_permission(
    principal: Optional[Principal], permissions: Iterable[Permission]
) -> Principal:
    """Allow when at least one permission is granted; otherwise deny with detail."""
    permissions = list(permissions)
    if not permissions:
        raise AccessDeniedError("no permissions specified")
    resolved = _principal_arg(principal)
    for permission in permissions:
        if not isinstance(permission, Permission):
            raise AccessDeniedError(f"unknown permission: {permission!r}")
        if resolved.has_permission(permission):
            return resolved
    held = sorted(p.value for p in resolved.permissions)
    wanted = sorted(p.value for p in permissions)
    raise AccessDeniedError(f"requires one of {wanted}, principal holds {held}")


def check_approval_authorization(
    principal: Optional[Principal],
    *,
    required_role: Optional[str] = None,
    initiator_subject: Optional[str] = None,
    now: Optional[datetime] = None,
) -> AccessDecision:
    """Authorize a human approval decision.

    Rules (all fail-closed):

    * a validated principal with the ``APPROVE_RESOLUTION`` permission is required;
    * when the Approval record specifies ``required_role``, the principal must
      also hold that role;
    * where an initiating identity is known, separation of duties applies: the
      initiator may not approve their own action.
    """
    try:
        resolved = _principal_arg(principal)
    except AccessDeniedError as error:
        return AccessDecision(False, None, Permission.APPROVE_RESOLUTION, error.reason)

    decision = check_permission(resolved, Permission.APPROVE_RESOLUTION)
    if not decision.allowed:
        return decision

    if required_role:
        try:
            role = Role(str(required_role).upper())
        except ValueError:
            return AccessDecision(
                False,
                resolved.actor_id(),
                Permission.APPROVE_RESOLUTION,
                f"approval requires unknown role {required_role!r}",
            )
        if not resolved.has_role(role):
            return AccessDecision(
                False,
                resolved.actor_id(),
                Permission.APPROVE_RESOLUTION,
                f"approval requires role {role.value}",
            )

    if initiator_subject and initiator_subject == resolved.subject:
        return AccessDecision(
            False,
            resolved.actor_id(),
            Permission.APPROVE_RESOLUTION,
            "separation of duties: the initiator cannot approve their own action",
        )

    if now is not None and resolved.expires_at is not None:
        if resolved.expires_at < now:
            return AccessDecision(
                False,
                resolved.actor_id(),
                Permission.APPROVE_RESOLUTION,
                "authenticated session has expired",
            )

    return AccessDecision(
        True,
        resolved.actor_id(),
        Permission.APPROVE_RESOLUTION,
        "approval authorized",
    )


__all__ = [
    "AccessDeniedError",
    "AccessDecision",
    "check_permission",
    "require_permission",
    "require_role",
    "require_any_permission",
    "check_approval_authorization",
]
