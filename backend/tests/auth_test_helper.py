"""
Deterministic test principals for Razorpay CloseLoop Phase 12.

Test-only helpers. Production code never imports this module.

Two mechanisms are provided:

* ``make_test_verifier()`` — a *real* HS256 verifier with a test signing key,
  used by the authentication tests to exercise actual signature verification.
* ``install_principal_override(app, principal)`` — installs a deterministic
  authenticated principal for legacy API tests that predate authentication.
  The override lives entirely in test code and replaces only the
  ``get_current_principal`` dependency; it is not a production bypass.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple

import jwt as pyjwt

from app.api.auth_dependencies import (
    get_current_principal,
    set_token_verifier,
)
from app.auth.authentication import AuthConfig, JwtTokenVerifier
from app.auth.principal import Principal, Role

#: Test-only signing secret. Never used by production configuration.
TEST_SECRET = "phase12-unit-test-signing-secret-do-not-use-in-prod"

TEST_ISSUER = "https://test-issuer.example.com"
TEST_AUDIENCE = "closeloop-test-client"

#: Deterministic reviewer identity used by legacy API test fixtures.
LEGACY_TEST_SUBJECT = "reviewer@example.com"


def make_test_config(**overrides) -> AuthConfig:
    config = AuthConfig(
        algorithms=("HS256",),
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
        token_use=None,
        secret_env_var="AUTH_SECRET_TEST_ONLY",
        roles_claim="roles",
    )
    return config.model_copy(update=overrides) if overrides else config


def make_test_verifier(**overrides) -> JwtTokenVerifier:
    """A real verifier backed by the deterministic test key."""
    return JwtTokenVerifier(make_test_config(**overrides), secret=TEST_SECRET)


def make_test_token(
    subject: str = "user-001",
    roles: Sequence[str] = ("ANALYST",),
    *,
    expired: bool = False,
    issuer: Optional[str] = TEST_ISSUER,
    audience: Optional[str] = TEST_AUDIENCE,
    token_use: Optional[str] = None,
    include_subject: bool = True,
    include_exp: bool = True,
    override_roles: Optional[Sequence[str]] = None,
    signing_key: Optional[str] = None,
    algorithm: str = "HS256",
) -> str:
    """Build a deterministic test JWT.

    ``expired``, ``issuer``, ``audience`` and ``override_roles`` let tests
    produce invalid tokens to prove the verifier rejects them.
    """
    now = datetime.now(timezone.utc)
    payload: dict = {"iat": now}
    if include_exp:
        payload["exp"] = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    if include_subject:
        payload["sub"] = subject
    if issuer is not None:
        payload["iss"] = issuer
    if audience is not None:
        payload["aud"] = audience
    if token_use is not None:
        payload["token_use"] = token_use
    if override_roles is not None:
        payload["roles"] = list(override_roles)
    else:
        payload["roles"] = list(roles)
    return pyjwt.encode(
        payload, signing_key if signing_key is not None else TEST_SECRET, algorithm=algorithm
    )


def deterministic_principal(
    subject: str = LEGACY_TEST_SUBJECT,
    roles: Sequence[Role] = (Role.OPERATOR, Role.APPROVER),
) -> Principal:
    """A typed principal built directly for legacy API test fixtures."""
    return Principal(
        subject=subject,
        roles=tuple(roles),
        issuer=TEST_ISSUER,
        token_use="id",
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def authenticated_test_client(app, subject: str = LEGACY_TEST_SUBJECT, roles: Sequence[Role] = (Role.OPERATOR, Role.APPROVER)):
    """A TestClient that authenticates every request with a REAL token.

    The token is signed with the test key and verified by a real HS256
    verifier, so legacy tests exercise the actual authentication + RBAC path
    rather than bypassing it.
    """
    from fastapi.testclient import TestClient

    set_token_verifier(make_test_verifier())
    token = make_test_token(subject=subject, roles=[r.value for r in roles])
    headers = {"Authorization": f"Bearer {token}"}

    class _AuthenticatedTestClient(TestClient):
        def request(self, method, url, **kwargs):  # type: ignore[override]
            merged = dict(headers)
            merged.update(kwargs.pop("headers", None) or {})
            return super().request(method, url, headers=merged, **kwargs)

    return _AuthenticatedTestClient(app, raise_server_exceptions=False)


__all__ = [
    "TEST_SECRET",
    "TEST_ISSUER",
    "TEST_AUDIENCE",
    "LEGACY_TEST_SUBJECT",
    "make_test_config",
    "make_test_verifier",
    "make_test_token",
    "deterministic_principal",
    "install_principal_override",
    "authenticated_test_client",
]
