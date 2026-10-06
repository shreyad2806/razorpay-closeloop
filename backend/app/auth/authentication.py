"""
Authentication for Razorpay CloseLoop Phase 12.

Validates a bearer token and produces a :class:`~app.auth.principal.Principal`.
Signature verification is mandatory: a token whose signature cannot be verified
is rejected, never decoded-and-trusted.

Supported configurations:

* ``HS256`` with a symmetric secret (``AUTH_SECRET``) — local/development and
  deterministic tests.
* ``RS256`` against a Cognito-style JWKS endpoint (``AUTH_JWKS_URI``) — the
  production path.

Fail-closed rules:

* missing token, malformed token, bad signature, expired token, wrong issuer,
  wrong audience, wrong token use or missing subject are all **rejected**;
* a configured verifier that cannot obtain a signing key **rejects**;
* there is no anonymous, development or environment bypass in this path.

Secrets are never hardcoded: they come from environment configuration only.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
from jwt import PyJWKClient
from pydantic import BaseModel, Field

from app.auth.principal import Principal, Role


class AuthenticationError(Exception):
    """Raised when a token cannot be authenticated. Always fail-closed."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class AuthConfig(BaseModel):
    """Authentication configuration. Loaded from the environment only."""

    algorithms: Tuple[str, ...] = ("HS256",)
    issuer: Optional[str] = None
    audience: Optional[str] = None
    token_use: Optional[str] = None  # Cognito: "id" | "access"
    secret_env_var: str = "AUTH_SECRET"
    jwks_uri: Optional[str] = None
    leeway_seconds: int = 0

    # claim that carries the role/group membership
    roles_claim: str = "cognito:groups"

    @classmethod
    def from_env(cls) -> "AuthConfig":
        algorithms = [
            a.strip().upper()
            for a in os.environ.get("AUTH_ALGORITHMS", "HS256").split(",")
            if a.strip()
        ]
        return cls(
            algorithms=tuple(algorithms) or ("HS256",),
            issuer=os.environ.get("AUTH_ISSUER") or None,
            audience=os.environ.get("AUTH_AUDIENCE") or None,
            token_use=os.environ.get("AUTH_TOKEN_USE") or None,
            secret_env_var=os.environ.get("AUTH_SECRET_ENV", "AUTH_SECRET"),
            jwks_uri=os.environ.get("AUTH_JWKS_URI") or None,
            leeway_seconds=int(os.environ.get("AUTH_LEEWAY_SECONDS", "0")),
            roles_claim=os.environ.get("AUTH_ROLES_CLAIM", "cognito:groups"),
        )


class JwtTokenVerifier:
    """Verifies a JWT and returns its validated claims.

    Construct with a secret (``HS256``) or a ``jwks_uri`` (``RS256``). At least
    one verification material must be available: a verifier with neither
    rejects every token instead of trusting it.
    """

    def __init__(
        self,
        config: Optional[AuthConfig] = None,
        *,
        secret: Optional[str] = None,
        jwks_client: Optional[PyJWKClient] = None,
    ) -> None:
        self.config = config or AuthConfig.from_env()
        resolved_secret = (
            secret
            if secret is not None
            else os.environ.get(self.config.secret_env_var)
        )
        self._secret = resolved_secret
        self._jwks_client = jwks_client
        if self.config.jwks_uri and jwks_client is None:
            self._jwks_client = PyJWKClient(self.config.jwks_uri)

    # ── verification ─────────────────────────────────────────────────────────

    def verify(self, token: str) -> Dict[str, Any]:
        if not token or not isinstance(token, str):
            raise AuthenticationError("missing bearer token")

        try:
            key = self._signing_key(token)
        except AuthenticationError:
            raise
        except Exception as error:  # noqa: BLE001 - fail closed on any key error
            raise AuthenticationError(f"signing key unavailable: {error}") from error

        options = {"require": ["exp", "iat"]}
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(self.config.algorithms),
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.leeway_seconds,
                options=options,
            )
        except jwt.ExpiredSignatureError as error:
            raise AuthenticationError("token expired") from error
        except jwt.InvalidIssuerError as error:
            raise AuthenticationError("token issuer mismatch") from error
        except jwt.InvalidAudienceError as error:
            raise AuthenticationError("token audience mismatch") from error
        except jwt.InvalidSignatureError as error:
            raise AuthenticationError("token signature verification failed") from error
        except jwt.MissingRequiredClaimError as error:
            raise AuthenticationError("token missing required claim") from error
        except jwt.PyJWTError as error:
            raise AuthenticationError(f"token verification failed: {error}") from error

        self._validate_token_use(claims)
        self._validate_subject(claims)
        return claims

    def _signing_key(self, token: str) -> Any:
        """Resolve the verification key, or fail closed."""
        has_rs = any(a.startswith("RS") or a.startswith("ES") for a in self.config.algorithms)
        if has_rs:
            if self._jwks_client is None:
                raise AuthenticationError(
                    "asymmetric verification configured without a JWKS source"
                )
            try:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token)
                return signing_key.key
            except Exception as error:  # noqa: BLE001 - never trust unverifiable tokens
                raise AuthenticationError(f"jwks lookup failed: {error}") from error

        if self._secret is None:
            raise AuthenticationError("no verification secret configured")
        return self._secret

    def _validate_token_use(self, claims: Dict[str, Any]) -> None:
        expected = self.config.token_use
        if not expected:
            return
        actual = claims.get("token_use")
        if actual != expected:
            raise AuthenticationError(
                f"token_use mismatch: expected {expected!r}, got {actual!r}"
            )

    def _validate_subject(self, claims: Dict[str, Any]) -> None:
        subject = claims.get("sub")
        if not subject or not isinstance(subject, str):
            raise AuthenticationError("token missing subject claim")

    # ── principal mapping ────────────────────────────────────────────────────

    def principal_from_claims(self, claims: Dict[str, Any]) -> Principal:
        """Map validated claims to a Principal (fail-closed on unknown roles)."""
        subject = claims["sub"]
        roles = self._roles_from_claims(claims)
        return Principal(
            subject=subject,
            roles=tuple(roles),
            issuer=claims.get("iss"),
            token_use=claims.get("token_use"),
            auth_time=_to_datetime(claims.get("auth_time")),
            expires_at=_to_datetime(claims.get("exp")),
            issued_at=_to_datetime(claims.get("iat")),
        )

    def authenticate(self, token: str) -> Principal:
        return self.principal_from_claims(self.verify(token))

    def _roles_from_claims(self, claims: Dict[str, Any]) -> List[Role]:
        raw = claims.get(self.config.roles_claim) or claims.get("roles") or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise AuthenticationError("invalid role claim shape")

        roles: List[Role] = []
        for item in raw:
            try:
                role = Role(str(item).upper())
            except ValueError as error:
                # Unknown role -> reject, never silently downgrade.
                raise AuthenticationError(f"unknown role claim: {item!r}") from error
            if role not in roles:
                roles.append(role)
        return roles


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError):
        return None


__all__ = ["AuthConfig", "AuthenticationError", "JwtTokenVerifier"]
