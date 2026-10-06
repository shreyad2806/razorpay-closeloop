"""
Phase 13 Authentication API Tests

Tests for the new /auth/me endpoint that exposes the current principal.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth.authentication import JwtTokenVerifier, AuthConfig
from app.auth.principal import Principal, Role, Permission
from app.api.auth_dependencies import set_token_verifier


class TestCurrentPrincipalEndpoint:
    """Tests for GET /auth/me endpoint."""

    def test_valid_token_returns_principal(self):
        """Valid token should return current principal information."""
        # Create a test verifier that returns a known principal
        test_verifier = JwtTokenVerifier(
            config=AuthConfig(
                algorithms=("HS256",),
                secret="test-secret",
            ),
            secret="test-secret",
        )

        # Register the test verifier
        set_token_verifier(test_verifier)

        # Generate a valid test token
        import jwt
        token = jwt.encode(
            {
                "sub": "user-123",
                "cognito:groups": ["ANALYST"],
                "exp": 9999999999,
                "iat": 1234567890,
            },
            "test-secret",
            algorithm="HS256",
        )

        client = TestClient(app)
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["subject"] == "user-123"
        assert "ANALYST" in data["data"]["roles"]

        # Reset verifier
        set_token_verifier(None)

    def test_missing_token_returns_401(self):
        """Missing token should return 401."""
        client = TestClient(app)
        response = client.get("/auth/me")

        assert response.status_code == 401
        assert "bearer" in response.json()["detail"].lower()

    def test_invalid_token_returns_401(self):
        """Invalid token should return 401."""
        client = TestClient(app)
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401

    def test_token_without_sub_returns_401(self):
        """Token without subject should return 401."""
        test_verifier = JwtTokenVerifier(
            config=AuthConfig(algorithms=("HS256",), secret="test-secret"),
            secret="test-secret",
        )
        set_token_verifier(test_verifier)

        import jwt
        token = jwt.encode(
            {"exp": 9999999999, "iat": 1234567890},
            "test-secret",
            algorithm="HS256",
        )

        client = TestClient(app)
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401
        set_token_verifier(None)

    def test_principal_reflects_validated_token_not_payload(self):
        """Principal should reflect validated token, not request payload."""
        test_verifier = JwtTokenVerifier(
            config=AuthConfig(algorithms=("HS256",), secret="test-secret"),
            secret="test-secret",
        )
        set_token_verifier(test_verifier)

        import jwt
        # Token says ANALYST
        token = jwt.encode(
            {
                "sub": "user-456",
                "cognito:groups": ["ANALYST"],
                "exp": 9999999999,
                "iat": 1234567890,
            },
            "test-secret",
            algorithm="HS256",
        )

        client = TestClient(app)
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        # Response should have ANALYST from token, not ADMIN from any forged payload
        assert "ANALYST" in data["data"]["roles"]
        assert "ADMIN" not in data["data"]["roles"]

        set_token_verifier(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
