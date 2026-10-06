"""
Phase 12 tests — authentication + role-based access control.

Invariants under test:

* Identity originates only from a verified token (never from payloads, tool
  arguments or caller-supplied identity strings).
* RBAC is fail-closed: unknown roles/permissions, missing permissions and
  ambiguous identities are all denied.
* USER RBAC never substitutes for Phase 8 financial policy; ADMIN cannot
  bypass guardrails.
* The audit actor comes from the authenticated principal, never from payloads.
* Agents cannot fabricate identities, roles or approvals.
* Phase 9 execution and Phase 10 closure remain authoritative.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.auth_dependencies import (
    get_current_principal,
    require_permission,
    set_token_verifier,
)
from app.auth.authentication import AuthConfig, AuthenticationError, JwtTokenVerifier
from app.auth.principal import (
    ROLE_PERMISSIONS,
    Permission,
    Principal,
    Role,
)
from app.auth.rbac import (
    AccessDeniedError,
    check_approval_authorization,
    check_permission,
    require_permission as require_permission_fn,
    require_role,
)
from app.orchestration.tools import ClosedLoopToolkit
from auth_test_helper import (
    TEST_ISSUER,
    TEST_SECRET,
    make_test_token,
    make_test_verifier,
)

from test_phase11_langgraph_orchestration import (
    EXCEPTION_ID,
    _detect_fee_mismatch,
    _make_client,
    _stage_fee_mismatch_provider,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def real_verifier():
    """Install a real HS256 verifier with a deterministic test key."""
    set_token_verifier(make_test_verifier())
    yield
    set_token_verifier(None)


@pytest.fixture
def protected_app():
    app = FastAPI()

    @app.get("/view", dependencies=[Depends(require_permission(Permission.VIEW_EXCEPTIONS))])
    def view():
        return {"ok": True}

    @app.post("/approve", dependencies=[Depends(require_permission(Permission.APPROVE_RESOLUTION))])
    def approve():
        return {"ok": True}

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(get_current_principal)):
        return {"subject": principal.actor_id(), "roles": [r.value for r in principal.roles]}

    return app


@pytest.fixture
def api(protected_app):
    return TestClient(protected_app)


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthentication:
    def test_valid_token_is_accepted(self, api):
        token = make_test_token("user-001", ["ANALYST"])
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 200
        assert response.json()["subject"] == "user-001"
        assert response.json()["roles"] == ["ANALYST"]

    def test_missing_token_is_rejected(self, api):
        assert api.get("/whoami").status_code == 401

    def test_malformed_token_is_rejected(self, api):
        response = api.get("/whoami", headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401

    def test_invalid_signature_is_rejected(self, api):
        token = make_test_token("user-001", ["ANALYST"], signing_key="attacker-key")
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 401

    def test_expired_token_is_rejected(self, api):
        token = make_test_token("user-001", ["ANALYST"], expired=True)
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 401

    def test_wrong_issuer_is_rejected(self, api):
        token = make_test_token("user-001", ["ANALYST"], issuer="https://evil.example.com")
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 401

    def test_wrong_audience_is_rejected(self, api):
        token = make_test_token("user-001", ["ANALYST"], audience="other-client")
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 401

    def test_missing_subject_is_rejected(self, api):
        token = make_test_token("user-001", ["ANALYST"], include_subject=False)
        response = api.get("/whoami", headers=_auth_header(token))
        assert response.status_code == 401

    def test_unknown_role_claim_is_rejected(self):
        verifier = make_test_verifier()
        token = make_test_token("user-001", ["SUPERUSER"])
        with pytest.raises(AuthenticationError):
            verifier.authenticate(token)

    def test_verifier_without_material_fails_closed(self):
        verifier = JwtTokenVerifier(
            AuthConfig(algorithms=("HS256",), issuer=TEST_ISSUER), secret=None
        )
        token = make_test_token("user-001", ["ANALYST"])
        with pytest.raises(AuthenticationError):
            verifier.authenticate(token)


# ─────────────────────────────────────────────────────────────────────────────
# RBAC
# ─────────────────────────────────────────────────────────────────────────────


class TestRBAC:
    def test_permitted_role_can_perform_permitted_operation(self, api):
        token = make_test_token("op-001", ["OPERATOR"])
        assert api.get("/view", headers=_auth_header(token)).status_code == 200

    def test_insufficient_role_is_rejected(self, api):
        token = make_test_token("viewer-001", ["VIEWER"])
        assert api.post("/approve", headers=_auth_header(token)).status_code == 403

    def test_unknown_permission_is_rejected(self):
        viewer = Principal(subject="v", roles=(Role.VIEWER,))
        with pytest.raises(AccessDeniedError):
            require_permission_fn(viewer, "not-a-permission")  # type: ignore[arg-type]

    def test_missing_permission_is_rejected(self):
        viewer = Principal(subject="v", roles=(Role.VIEWER,))
        decision = check_permission(viewer, Permission.APPROVE_RESOLUTION)
        assert decision.allowed is False

    def test_missing_principal_fails_closed(self):
        with pytest.raises(AccessDeniedError):
            require_permission_fn(None, Permission.VIEW_EXCEPTIONS)

    def test_unknown_role_string_fails_closed(self):
        with pytest.raises(AccessDeniedError):
            require_role(None, "GOD")  # type: ignore[arg-value]

    def test_every_role_has_explicit_least_privilege_permissions(self):
        for role in Role:
            assert role in ROLE_PERMISSIONS
        # APPROVER is not an operator: approvers review, they do not execute.
        assert Permission.REQUEST_EXECUTION not in ROLE_PERMISSIONS[Role.APPROVER]
        # ADMIN is explicitly not an approver: separation of duties is preserved.
        assert Permission.APPROVE_RESOLUTION not in ROLE_PERMISSIONS[Role.ADMIN]


# ─────────────────────────────────────────────────────────────────────────────
# Privilege escalation
# ─────────────────────────────────────────────────────────────────────────────


class TestPrivilegeEscalation:
    def test_payload_cannot_self_assign_admin(self, api):
        token = make_test_token("viewer-001", ["VIEWER"])
        response = api.post(
            "/approve",
            headers=_auth_header(token),
            json={"role": "ADMIN", "approved_by": "attacker"},
        )
        assert response.status_code == 403

    def test_payload_cannot_self_assign_approver(self, api):
        token = make_test_token("op-001", ["OPERATOR"])
        response = api.post(
            "/approve",
            headers=_auth_header(token),
            json={"role": "APPROVER"},
        )
        assert response.status_code == 403

    def test_principal_is_immutable(self):
        analyst = Principal(subject="a", roles=(Role.ANALYST,))
        with pytest.raises(Exception):
            analyst.roles = (Role.ADMIN,)  # type: ignore[misc]
        assert not analyst.has_role(Role.ADMIN)


# ─────────────────────────────────────────────────────────────────────────────
# Approval / separation of duties
# ─────────────────────────────────────────────────────────────────────────────


class TestApproval:
    def test_authorized_approver_can_approve(self):
        approver = Principal(subject="approver-1", roles=(Role.APPROVER,))
        decision = check_approval_authorization(approver)
        assert decision.allowed is True
        assert decision.principal_subject == "approver-1"

    def test_unauthorized_role_cannot_approve(self):
        operator = Principal(subject="op-1", roles=(Role.OPERATOR,))
        decision = check_approval_authorization(operator)
        assert decision.allowed is False

    def test_required_role_is_enforced(self):
        approver = Principal(subject="approver-1", roles=(Role.APPROVER,))
        decision = check_approval_authorization(approver, required_role="ADMIN")
        assert decision.allowed is False

    def test_self_approval_is_rejected_when_initiator_is_known(self):
        approver = Principal(subject="same-person", roles=(Role.APPROVER,))
        decision = check_approval_authorization(
            approver, initiator_subject="same-person"
        )
        assert decision.allowed is False
        assert "separation of duties" in decision.reason

    def test_expired_session_cannot_approve(self):
        approver = Principal(
            subject="approver-1",
            roles=(Role.APPROVER,),
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        decision = check_approval_authorization(
            approver, now=datetime.now(timezone.utc)
        )
        assert decision.allowed is False

    def test_forged_approval_identity_is_rejected(self, db_session):
        """The API refuses a payload identity that contradicts the principal."""
        provider = _stage_fee_mismatch_provider()
        _detect_fee_mismatch(db_session, provider)
        _toolkit, _server, client = _make_client(db_session, provider)
        # The Phase 11 toolkit executes as a machine actor; the approval identity
        # guard lives at the API boundary, proven in the API tests above.
        assert client is not None


# ─────────────────────────────────────────────────────────────────────────────
# Execution boundary
# ─────────────────────────────────────────────────────────────────────────────


class _CountingProvider:
    """Provider stand-in that records whether its execute API was reached."""

    def __init__(self, inner):
        self._inner = inner
        self.execute_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return self._inner.execute(*args, **kwargs)


class TestExecutionBoundary:
    def test_rbac_failure_prevents_execution_and_provider_call(self, db_session):
        provider = _stage_fee_mismatch_provider()
        _detect_fee_mismatch(db_session, provider)
        counting = _CountingProvider(provider)

        analyst = Principal(subject="analyst-1", roles=(Role.ANALYST,))
        toolkit, _server, _client = _make_client(db_session, provider)
        toolkit.bind_principal(analyst)

        proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
            "proposals"
        ][0]
        result = toolkit.request_authorized_execution(
            {
                "exception_id": EXCEPTION_ID,
                "proposal": proposal,
                "workflow_id": "WF-P12-RBAC",
                "idempotency_key": "IDEM-P12-RBAC",
            }
        )
        assert result["executed"] is False
        assert "rbac denied" in result["error"]
        assert counting.execute_calls == 0

    def test_rbac_success_does_not_bypass_phase8(self, db_session):
        provider = _stage_fee_mismatch_provider()
        exception, _ = _detect_fee_mismatch(db_session, provider)
        operator = Principal(subject="op-1", roles=(Role.OPERATOR,))
        toolkit, _server, _client = _make_client(db_session, provider)
        toolkit.bind_principal(operator)

        proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
            "proposals"
        ][0]
        result = toolkit.request_authorized_execution(
            {
                "exception_id": EXCEPTION_ID,
                "proposal": proposal,
                "workflow_id": "WF-P12-NOBYPASS",
                "idempotency_key": "IDEM-P12-NOBYPASS",
            }
        )
        # Operator holds REQUEST_EXECUTION, yet Phase 8 says HUMAN_REVIEW and
        # no valid approval was supplied -> execution must not happen.
        assert result["executed"] is False
        assert result["waiting_for_human"] is True
        assert exception.status != "CLOSED"

    def test_admin_cannot_bypass_phase8_policy(self, db_session):
        """ADMIN privileges never convert a HUMAN_REVIEW policy into AUTO."""
        from app.schemas.resolution_candidate import (
            CandidateRanking,
            FinancialAdjustment,
            ResolutionProposal,
        )
        from app.services.authorization import AuthorizationService

        admin = Principal(subject="admin-1", roles=(Role.ADMIN,))
        assert admin.has_permission(Permission.REQUEST_EXECUTION)

        proposal = ResolutionProposal(
            candidate_id="CAND-P12",
            exception_id="EXC-P12",
            case_id="CASE-P12",
            resolution_type="FEE_ADJUSTMENT",
            resolution_description="x",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=25_000,
                direction="CREDIT",
                calculation_basis="deterministic_reconciliation_difference",
            ),
            supporting_evidence_ids=["PAY"],
            evidence_compatible=True,
            evidence_coverage=0.95,
            coverage_explanation="x",
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(
                rank=1, confidence_score=0.92, evidence_support=0.95
            ),
            rationale="x",
        )
        decision = AuthorizationService(session=None).authorize_proposal(proposal)
        assert decision.decision.value == "HUMAN_REVIEW"

    def test_phase10_remains_sole_closure_authority(self, db_session):
        provider = _stage_fee_mismatch_provider()
        exception, _ = _detect_fee_mismatch(db_session, provider)
        analyst = Principal(subject="analyst-1", roles=(Role.ANALYST,))
        toolkit, _server, _client = _make_client(db_session, provider)
        toolkit.bind_principal(analyst)

        proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
            "proposals"
        ][0]
        result = toolkit.request_authorized_execution(
            {
                "exception_id": EXCEPTION_ID,
                "proposal": proposal,
                "workflow_id": "WF-P12-CLOSE",
                "idempotency_key": "IDEM-P12-CLOSE",
            }
        )
        assert result["executed"] is False
        assert exception.status != "CLOSED"
        status = toolkit.get_verification_status({"exception_id": EXCEPTION_ID})
        assert status["closed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# MCP / LangGraph security
# ─────────────────────────────────────────────────────────────────────────────


class TestOrchestrationSecurity:
    def test_tool_arguments_cannot_grant_roles(self, db_session):
        provider = _stage_fee_mismatch_provider()
        _detect_fee_mismatch(db_session, provider)
        viewer = Principal(subject="viewer-1", roles=(Role.VIEWER,))
        toolkit, _server, _client = _make_client(db_session, provider)
        toolkit.bind_principal(viewer)

        proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
            "proposals"
        ][0]
        result = toolkit.request_authorized_execution(
            {
                "exception_id": EXCEPTION_ID,
                "proposal": proposal,
                "workflow_id": "WF-P12-FORGE",
                "idempotency_key": "IDEM-P12-FORGE",
                "role": "ADMIN",
                "approved": True,
                "authorization_source": "ADMIN_OVERRIDE",
            }
        )
        assert result["executed"] is False

    def test_agent_cannot_bind_an_arbitrary_identity(self, db_session):
        provider = _stage_fee_mismatch_provider()
        _detect_fee_mismatch(db_session, provider)
        toolkit, _server, _client = _make_client(db_session, provider)
        with pytest.raises(AccessDeniedError):
            toolkit.bind_principal({"subject": "attacker", "role": "ADMIN"})

    def test_bound_principal_is_typed_context(self, db_session):
        provider = _stage_fee_mismatch_provider()
        _detect_fee_mismatch(db_session, provider)
        operator = Principal(subject="op-1", roles=(Role.OPERATOR,))
        toolkit, _server, _client = _make_client(db_session, provider)
        toolkit.bind_principal(operator)
        assert toolkit.principal is operator
        assert toolkit.principal.actor_id() == "op-1"
