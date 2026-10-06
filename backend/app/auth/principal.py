"""
Typed identity model for Razorpay CloseLoop Phase 12.

A :class:`Principal` is created **only** by the authentication layer from a
validated token. It is never constructed from request payloads, tool arguments
or caller-supplied identity strings.

USER RBAC answers: "is this authenticated identity allowed to invoke this
application operation?" It is deliberately separate from Phase 8 financial
policy, which answers "is THIS resolution authorized under financial risk
policy?". The two layers are never merged:

    authenticated user
      -> RBAC (this module)
      -> application operation
      -> Phase 8 financial policy
      -> Phase 9 controlled execution
      -> Phase 10 reconciliation

An ADMIN role grants administrative permissions; it never bypasses Phase 8
policy, reconciliation, or audit requirements.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Application roles. Least privilege by default.

    Adding a role here does not grant anything by itself: every role must be
    explicitly listed in ``ROLE_PERMISSIONS`` or it has no permissions at all.
    """

    VIEWER = "VIEWER"
    ANALYST = "ANALYST"
    OPERATOR = "OPERATOR"
    APPROVER = "APPROVER"
    ADMIN = "ADMIN"


class Permission(str, Enum):
    """Explicit application operations an identity may perform."""

    # read / inspection
    VIEW_EXCEPTIONS = "view:exceptions"
    VIEW_EVIDENCE = "view:evidence"
    VIEW_RECONCILIATION = "view:reconciliation"
    VIEW_AUDIT = "view:audit"

    # investigation
    INVESTIGATE_EXCEPTIONS = "investigate:exceptions"
    GENERATE_PROPOSALS = "generate:proposals"
    RETRIEVE_HISTORICAL_CASES = "retrieve:historical-cases"

    # resolution initiation (Phase 7 boundary)
    INITIATE_RESOLUTION = "initiate:resolution"
    REQUEST_EXECUTION = "request:execution"

    # human approval boundary
    APPROVE_RESOLUTION = "approve:resolution"
    REJECT_RESOLUTION = "reject:resolution"

    # administration
    ADMINISTRATIVE_INSPECTION = "admin:inspection"
    MANAGE_CONFIGURATION = "admin:configuration"


#: Least-privilege role -> permission map. The single source of RBAC truth.
ROLE_PERMISSIONS: "dict[Role, FrozenSet[Permission]]" = {
    Role.VIEWER: frozenset(
        {
            Permission.VIEW_EXCEPTIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_RECONCILIATION,
        }
    ),
    Role.ANALYST: frozenset(
        {
            # viewer capabilities
            Permission.VIEW_EXCEPTIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_RECONCILIATION,
            Permission.VIEW_AUDIT,
            # investigation
            Permission.INVESTIGATE_EXCEPTIONS,
            Permission.GENERATE_PROPOSALS,
            Permission.RETRIEVE_HISTORICAL_CASES,
        }
    ),
    Role.OPERATOR: frozenset(
        {
            # analyst capabilities
            Permission.VIEW_EXCEPTIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_RECONCILIATION,
            Permission.VIEW_AUDIT,
            Permission.INVESTIGATE_EXCEPTIONS,
            Permission.GENERATE_PROPOSALS,
            Permission.RETRIEVE_HISTORICAL_CASES,
            # initiation / controlled execution request
            Permission.INITIATE_RESOLUTION,
            Permission.REQUEST_EXECUTION,
        }
    ),
    Role.APPROVER: frozenset(
        {
            # viewer capabilities + the human approval boundary
            Permission.VIEW_EXCEPTIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_RECONCILIATION,
            Permission.VIEW_AUDIT,
            Permission.APPROVE_RESOLUTION,
            Permission.REJECT_RESOLUTION,
        }
    ),
    Role.ADMIN: frozenset(
        {
            # administrative inspection and configuration management.
            # Deliberately does NOT include APPROVE_RESOLUTION: financial
            # approval remains a distinct human accountability, and granting
            # ADMIN the approval permission would erode separation of duties.
            Permission.VIEW_EXCEPTIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_RECONCILIATION,
            Permission.VIEW_AUDIT,
            Permission.INVESTIGATE_EXCEPTIONS,
            Permission.GENERATE_PROPOSALS,
            Permission.RETRIEVE_HISTORICAL_CASES,
            Permission.INITIATE_RESOLUTION,
            Permission.REQUEST_EXECUTION,
            Permission.ADMINISTRATIVE_INSPECTION,
            Permission.MANAGE_CONFIGURATION,
        }
    ),
}


def permissions_for_role(role: Role) -> FrozenSet[Permission]:
    """Return the explicit permission set for ``role`` (empty if unlisted)."""
    return ROLE_PERMISSIONS.get(role, frozenset())


class Principal(BaseModel):
    """A validated authenticated identity.

    Constructed exclusively by :mod:`app.auth.authentication` after token
    verification. Roles always originate from the validated token's claim set —
    never from request payloads or tool arguments.
    """

    model_config = {"frozen": True}

    subject: str = Field(..., min_length=1, description="Validated token subject")
    roles: Tuple[Role, ...] = Field(default_factory=tuple)
    issuer: Optional[str] = Field(default=None, description="Validated token issuer")
    token_use: Optional[str] = Field(
        default=None, description="Validated token use (id / access)"
    )
    auth_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None

    # ── permission helpers ────────────────────────────────────────────────────

    @property
    def permissions(self) -> FrozenSet[Permission]:
        """Union of permissions across every role held by this principal."""
        granted: set = set()
        for role in self.roles:
            granted |= permissions_for_role(role)
        return frozenset(granted)

    def has_role(self, role: Role) -> bool:
        return role in self.roles

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def actor_id(self) -> str:
        """The immutable audit attribution value.

        Audit ``actor`` fields must be set from this value, never from request
        payloads or tool arguments.
        """
        return self.subject

    def actor_type(self) -> str:
        """Map to the existing AuditEvent actor_type vocabulary."""
        return "HUMAN"

    def summary(self) -> str:
        roles = ",".join(r.value for r in self.roles) or "none"
        return f"{self.subject} [{roles}]"


__all__ = [
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "permissions_for_role",
    "Principal",
]
