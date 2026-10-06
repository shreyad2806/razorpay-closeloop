"""
Phase 13 Audit API Tests

Tests for the new GET /exceptions/{id}/audit endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth.authentication import JwtTokenVerifier, AuthConfig
from app.auth.principal import Role, Permission
from app.api.auth_dependencies import set_token_verifier
from app.schemas.audit import AuditEventType, ActorType


class TestAuditAPI:
    """Tests for GET /exceptions/{id}/audit endpoint."""

    @pytest.fixture
    def audit_svc(self):
        """Return a shared audit service instance."""
        from app.api.dependencies import get_audit_service
        return get_audit_service()

    @pytest.fixture
    def authenticated_client(self):
        """Return a test client with authenticated token."""
        test_verifier = JwtTokenVerifier(
            config=AuthConfig(algorithms=("HS256",), secret="test-secret"),
            secret="test-secret",
        )
        set_token_verifier(test_verifier)

        import jwt
        token = jwt.encode(
            {
                "sub": "user-audit-test",
                "cognito:groups": ["ANALYST"],
                "exp": 9999999999,
                "iat": 1234567890,
            },
            "test-secret",
            algorithm="HS256",
        )

        client = TestClient(app)
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client
        set_token_verifier(None)

    def test_authorized_user_can_read_audit_events(self, authenticated_client, audit_svc):
        """Authorized user should be able to read audit events."""
        # Seed audit data using the shared service
        audit_svc.create_event(
            event_type=AuditEventType.WORKFLOW_STARTED,
            workflow_id="WF-001",
            exception_id="CASE-DEMO-001",
            actor="system:policy-engine",
            actor_type=ActorType.SYSTEM,
        )

        response = authenticated_client.get("/exceptions/CASE-DEMO-001/audit")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

    def test_unauthorized_user_receives_403(self):
        """Unauthorized user should receive 403."""
        # Create a user without VIEW_AUDIT permission
        test_verifier = JwtTokenVerifier(
            config=AuthConfig(algorithms=("HS256",), secret="test-secret"),
            secret="test-secret",
        )
        set_token_verifier(test_verifier)

        import jwt
        token = jwt.encode(
            {
                "sub": "user-viewer",
                "cognito:groups": ["VIEWER"],  # VIEWER doesn't have VIEW_AUDIT
                "exp": 9999999999,
                "iat": 1234567890,
            },
            "test-secret",
            algorithm="HS256",
        )

        client = TestClient(app)
        response = client.get(
            "/exceptions/CASE-DEMO-001/audit",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        set_token_verifier(None)

    def test_unauthenticated_user_receives_401(self):
        """Unauthenticated user should receive 401."""
        client = TestClient(app)
        response = client.get("/exceptions/CASE-DEMO-001/audit")

        assert response.status_code == 401

    def test_nonexistent_exception_handled_correctly(self, authenticated_client):
        """Nonexistent exception should return 404."""
        response = authenticated_client.get("/exceptions/NONEXISTENT/audit")

        assert response.status_code == 404

    def test_returned_actor_comes_from_audit_event(self, authenticated_client, audit_svc):
        """Actor should come from AuditEvent, not session."""
        audit_svc.create_event(
            event_type=AuditEventType.WORKFLOW_STARTED,
            workflow_id="WF-002",
            exception_id="CASE-DEMO-002",
            actor="agent:resolution-analyst@v3",
            actor_type=ActorType.AGENT,
        )

        response = authenticated_client.get("/exceptions/CASE-DEMO-002/audit")

        assert response.status_code == 200
        data = response.json()
        events = data["data"]
        assert len(events) >= 1
        # Actor should be from the audit event, not the authenticated user
        assert events[0]["actor"] == "agent:resolution-analyst@v3"
        assert events[0]["actor_type"] == "AGENT"

    def test_endpoint_cannot_create_modify_audit_records(self, authenticated_client):
        """Endpoint should be read-only."""
        # Try POST (should fail)
        response = authenticated_client.post(
            "/exceptions/CASE-DEMO-001/audit",
            json={"action": "test"},
        )

        # 405 Method Not Allowed
        assert response.status_code == 405

    def test_audit_events_ordered_by_timestamp(self, authenticated_client, audit_svc):
        """Audit events should be ordered by timestamp."""
        from datetime import datetime, timedelta

        # Create events with different timestamps
        event1 = audit_svc.create_event(
            event_type=AuditEventType.WORKFLOW_STARTED,
            workflow_id="WF-003",
            exception_id="CASE-DEMO-003",
            actor="system:test",
            actor_type=ActorType.SYSTEM,
        )

        event2 = audit_svc.create_event(
            event_type=AuditEventType.CLASSIFICATION_COMPLETE,
            workflow_id="WF-003",
            exception_id="CASE-DEMO-003",
            actor="system:test",
            actor_type=ActorType.SYSTEM,
        )

        response = authenticated_client.get("/exceptions/CASE-DEMO-003/audit")

        assert response.status_code == 200
        data = response.json()
        events = data["data"]
        assert len(events) >= 2
        # Events should be in chronological order
        assert events[0]["event_type"] == "WORKFLOW_STARTED"
        assert events[1]["event_type"] == "CLASSIFICATION_COMPLETE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
