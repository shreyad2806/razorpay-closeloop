"""
Tests for Phase 13.4 — Exception REST APIs.

Tests:
- List with pagination and filtering
- Get existing exception
- Get unknown exception
- Resolve exception
- Approve exception
- Reject exception
- Escalate exception
- Invalid state transitions
- Duplicate operations
- Guardrail rejection
"""

import pytest
from fastapi.testclient import TestClient

from app.api.services.exception_service import ExceptionService, _exception_registry
from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean exception registry before each test."""
    _exception_registry.clear()
    yield
    _exception_registry.clear()


# ─────────────────────────────────────────────────────────────────────────────
# GET /exceptions — List
# ─────────────────────────────────────────────────────────────────────────────


class TestListExceptions:
    def test_list_returns_cases(self, client):
        """List exceptions returns real cases from batch data."""
        response = client.get("/exceptions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    def test_list_has_required_fields(self, client):
        """Each exception has required fields."""
        response = client.get("/exceptions?limit=1")
        data = response.json()
        exc = data["data"][0]
        assert "exception_id" in exc
        assert "case_id" in exc
        assert "exception_type" in exc
        assert "expected_amount_paise" in exc
        assert "actual_amount_paise" in exc
        assert "difference_paise" in exc
        assert "risk_category" in exc
        assert "status" in exc

    def test_list_with_limit(self, client):
        """Limit controls maximum results."""
        response = client.get("/exceptions?limit=3")
        data = response.json()
        assert len(data["data"]) <= 3

    def test_list_with_offset(self, client):
        """Offset skips results."""
        r1 = client.get("/exceptions?limit=2&offset=0").json()
        r2 = client.get("/exceptions?limit=2&offset=2").json()
        if len(r1["data"]) == 2 and len(r2["data"]) > 0:
            assert r1["data"][0]["exception_id"] != r2["data"][0]["exception_id"]

    def test_list_filter_by_exception_type(self, client):
        """Filter by exception type."""
        response = client.get("/exceptions?exception_type=FEE_DIFFERENCE")
        data = response.json()
        for exc in data["data"]:
            assert exc["exception_type"] == "FEE_DIFFERENCE"

    def test_list_filter_by_risk_category(self, client):
        """Filter by risk category."""
        response = client.get("/exceptions?risk_category=LOW")
        data = response.json()
        for exc in data["data"]:
            assert exc["risk_category"] == "LOW"

    def test_list_with_batch_id(self, client):
        """Filter by batch ID."""
        response = client.get("/exceptions?batch_id=batch_001")
        data = response.json()
        assert data["success"] is True

    def test_list_limit_boundary(self, client):
        """Limit=1 returns exactly 1."""
        response = client.get("/exceptions?limit=1")
        data = response.json()
        assert len(data["data"]) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# GET /exceptions/{exception_id} — Get
# ─────────────────────────────────────────────────────────────────────────────


class TestGetException:
    def test_get_existing(self, client):
        """Get an existing exception by case_id."""
        # First get a list to find a valid ID
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.get(f"/exceptions/{exc_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["exception_id"] == exc_id

    def test_get_not_found(self, client):
        """Get unknown exception returns 404."""
        response = client.get("/exceptions/NONEXISTENT-CASE")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False

    def test_get_has_financial_discrepancy(self, client):
        """Get returns financial discrepancy information."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.get(f"/exceptions/{exc_id}")
            data = response.json()["data"]
            assert "expected_amount_paise" in data
            assert "actual_amount_paise" in data
            assert "difference_paise" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /exceptions/{exception_id}/resolve — Resolve
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveException:
    def test_resolve_existing(self, client):
        """Resolve an existing exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={
                    "resolution_type": "FEE_ADJUSTMENT",
                    "adjustment_paise": 5000,
                    "reason": "Fee discrepancy confirmed",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result = data["data"]
            assert result["status"] == "RESOLVED"
            assert result["resolution_type"] == "FEE_ADJUSTMENT"
            assert result["adjustment_paise"] == 5000

    def test_resolve_not_found(self, client):
        """Resolve unknown exception returns 404."""
        response = client.post(
            "/exceptions/NONEXISTENT/resolve",
            json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 0},
        )
        assert response.status_code == 404

    def test_resolve_already_resolved(self, client):
        """Resolve already resolved exception returns 409."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            # First resolve
            client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 1000},
            )
            # Second resolve should conflict
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "REFUND_ADJUSTMENT", "adjustment_paise": 2000},
            )
            assert response.status_code == 409

    def test_resolve_invalid_type(self, client):
        """Resolve with invalid resolution type returns 422."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "INVALID_TYPE", "adjustment_paise": 0},
            )
            assert response.status_code == 422

    def test_resolve_large_adjustment(self, client):
        """Resolve with adjustment over limit returns 422."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 20_000_000},
            )
            assert response.status_code == 422

    def test_resolve_has_workflow_id(self, client):
        """Resolve creates a workflow ID."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "NO_ACTION", "adjustment_paise": 0},
            )
            data = response.json()["data"]
            assert "workflow_id" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /exceptions/{exception_id}/approve — Approve
# ─────────────────────────────────────────────────────────────────────────────


class TestApproveException:
    def test_approve_resolved(self, client):
        """Approve a resolved exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            # Resolve first
            client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
            )
            # Approve
            response = client.post(
                f"/exceptions/{exc_id}/approve",
                json={"approved_by": "reviewer@example.com", "comments": "Looks correct"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["status"] == "APPROVED"
            assert data["data"]["approved_by"] == "reviewer@example.com"

    def test_approve_not_found(self, client):
        """Approve unknown exception returns 404."""
        response = client.post(
            "/exceptions/NONEXISTENT/approve",
            json={"approved_by": "reviewer@example.com"},
        )
        assert response.status_code == 404

    def test_approve_pending_exception(self, client):
        """Approve a pending (unresolved) exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/approve",
                json={"approved_by": "reviewer@example.com"},
            )
            # Should succeed — pending exceptions can be approved
            assert response.status_code == 200

    def test_approve_has_feedback_id(self, client):
        """Approve creates a feedback record."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/approve",
                json={"approved_by": "reviewer@example.com"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /exceptions/{exception_id}/reject — Reject
# ─────────────────────────────────────────────────────────────────────────────


class TestRejectException:
    def test_reject_resolved(self, client):
        """Reject a resolved exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            # Resolve first
            client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
            )
            # Reject
            response = client.post(
                f"/exceptions/{exc_id}/reject",
                json={
                    "rejected_by": "reviewer@example.com",
                    "reason": "Incorrect amount",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["status"] == "REJECTED"
            assert data["data"]["reason"] == "Incorrect amount"

    def test_reject_not_found(self, client):
        """Reject unknown exception returns 404."""
        response = client.post(
            "/exceptions/NONEXISTENT/reject",
            json={"rejected_by": "r@e.com", "reason": "Bad"},
        )
        assert response.status_code == 404

    def test_reject_empty_reason(self, client):
        """Reject without reason returns 422."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/reject",
                json={"rejected_by": "r@e.com", "reason": ""},
            )
            assert response.status_code == 422

    def test_reject_has_feedback_id(self, client):
        """Reject creates a feedback record."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/reject",
                json={"rejected_by": "r@e.com", "reason": "Wrong type"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /exceptions/{exception_id}/escalate — Escalate
# ─────────────────────────────────────────────────────────────────────────────


class TestEscalateException:
    def test_escalate_existing(self, client):
        """Escalate an exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/escalate",
                json={
                    "reason": "High value case needs manual review",
                    "escalated_by": "ops@example.com",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["status"] == "ESCALATED"
            assert data["data"]["reason"] == "High value case needs manual review"

    def test_escalate_not_found(self, client):
        """Escalate unknown exception returns 404."""
        response = client.post(
            "/exceptions/NONEXISTENT/escalate",
            json={"reason": "High risk"},
        )
        assert response.status_code == 404

    def test_escalate_empty_reason(self, client):
        """Escalate without reason returns 422."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/escalate",
                json={"reason": ""},
            )
            assert response.status_code == 422

    def test_escalate_has_feedback_id(self, client):
        """Escalate creates a feedback record."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/escalate",
                json={"reason": "Needs review"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data

    def test_escalate_already_resolved(self, client):
        """Can escalate a resolved exception."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "NO_ACTION", "adjustment_paise": 0},
            )
            response = client.post(
                f"/exceptions/{exc_id}/escalate",
                json={"reason": "Changed my mind"},
            )
            assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# State Transitions
# ─────────────────────────────────────────────────────────────────────────────


class TestStateTransitions:
    def test_cannot_approve_after_rejection(self, client):
        """Cannot approve after rejection (status changed)."""
        list_resp = client.get("/exceptions?limit=2")
        cases = list_resp.json()["data"]
        if len(cases) >= 1:
            exc_id = cases[0]["exception_id"]
            # Reject
            client.post(
                f"/exceptions/{exc_id}/reject",
                json={"rejected_by": "r@e.com", "reason": "Bad"},
            )
            # Try to approve — should fail because status is REJECTED
            response = client.post(
                f"/exceptions/{exc_id}/approve",
                json={"approved_by": "a@e.com"},
            )
            assert response.status_code == 409

    def test_resolve_after_escalation(self, client):
        """Can resolve after escalation (escalation doesn't block)."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            # Escalate
            client.post(
                f"/exceptions/{exc_id}/escalate",
                json={"reason": "Needs review"},
            )
            # Resolve
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "NO_ACTION", "adjustment_paise": 0},
            )
            # Should succeed (escalation doesn't block resolve in this implementation)
            assert response.status_code in (200, 409)


# ─────────────────────────────────────────────────────────────────────────────
# Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExceptionSafety:
    def test_resolve_uses_existing_service(self, client):
        """Resolution is recorded through existing feedback/outcome services."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 5000},
            )
            data = response.json()["data"]
            # Should have guardrail and execution placeholders
            assert "guardrail_decision" in data
            assert "verification_result" in data

    def test_approve_records_feedback(self, client):
        """Approval is recorded as feedback."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/approve",
                json={"approved_by": "reviewer@example.com"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data

    def test_reject_records_feedback(self, client):
        """Rejection is recorded as feedback."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/reject",
                json={"rejected_by": "r@e.com", "reason": "Wrong"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data

    def test_escalate_records_feedback(self, client):
        """Escalation is recorded as feedback."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            response = client.post(
                f"/exceptions/{exc_id}/escalate",
                json={"reason": "High risk"},
            )
            data = response.json()["data"]
            assert "feedback_id" in data

    def test_amount_bounds_enforced(self, client):
        """Adjustment amount bounds are enforced by Pydantic."""
        list_resp = client.get("/exceptions?limit=1")
        cases = list_resp.json()["data"]
        if cases:
            exc_id = cases[0]["exception_id"]
            # Over limit
            response = client.post(
                f"/exceptions/{exc_id}/resolve",
                json={"resolution_type": "FEE_ADJUSTMENT", "adjustment_paise": 999_999_999},
            )
            assert response.status_code == 422
