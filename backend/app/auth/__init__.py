"""
Authentication and role-based access control for Razorpay CloseLoop.

Phase 12 boundary:

    authenticated principal -> RBAC -> application operation
      -> Phase 8 financial policy -> Phase 9 controlled execution
      -> Phase 10 reconciliation

Authentication and RBAC strengthen this boundary; they never replace the
financial policy, execution or reconciliation layers.
"""

from app.auth.authentication import (
    AuthConfig,
    AuthenticationError,
    JwtTokenVerifier,
)
from app.auth.principal import (
    ROLE_PERMISSIONS,
    Permission,
    Principal,
    Role,
    permissions_for_role,
)
from app.auth.rbac import (
    AccessDecision,
    AccessDeniedError,
    check_approval_authorization,
    check_permission,
    require_any_permission,
    require_permission,
    require_role,
)

__all__ = [
    "AuthConfig",
    "AuthenticationError",
    "JwtTokenVerifier",
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "permissions_for_role",
    "Principal",
    "AccessDeniedError",
    "AccessDecision",
    "check_permission",
    "require_permission",
    "require_role",
    "require_any_permission",
    "check_approval_authorization",
]
