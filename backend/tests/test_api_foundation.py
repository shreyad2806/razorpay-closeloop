"""
Tests for Phase 13.1 — REST API Foundation.

Tests application startup, router registration, OpenAPI, health,
and all route stubs.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Application Startup
# ─────────────────────────────────────────────────────────────────────────────


class TestApplicationStartup:
    """Test that the application starts correctly."""

    def test_app_is_fastapi_instance(self):
        assert isinstance(app, FastAPI)

    def test_app_title(self):
        assert app.title == "Razorpay CloseLoop"

    def test_app_version(self):
        assert app.version == "1.0.0"

    def test_app_has_description(self):
        assert "financial" in app.description.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Router Registration
# ─────────────────────────────────────────────────────────────────────────────


class TestRouterRegistration:
    """Test that all routers are registered."""

    def _get_routes(self, client):
        """Get all registered routes."""
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
        return routes

    def test_health_route(self, client):
        routes = self._get_routes(client)
        assert "/health" in routes

    def test_explain_route(self, client):
        routes = self._get_routes(client)
        assert "/explain" in routes

    def test_analyze_route(self, client):
        routes = self._get_routes(client)
        assert "/analyze" in routes

    def test_batches_routes(self, client):
        routes = self._get_routes(client)
        assert "/batches" in routes

    def test_batch_detail_route(self, client):
        routes = self._get_routes(client)
        assert "/batches/{batch_id}" in routes

    def test_batch_run_route(self, client):
        routes = self._get_routes(client)
        assert "/batches/{batch_id}/run" in routes

    def test_batch_summary_route(self, client):
        routes = self._get_routes(client)
        assert "/batches/{batch_id}/summary" in routes

    def test_exceptions_routes(self, client):
        routes = self._get_routes(client)
        assert "/exceptions" in routes

    def test_exception_detail_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}" in routes

    def test_exception_resolve_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/resolve" in routes

    def test_exception_approve_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/approve" in routes

    def test_exception_reject_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/reject" in routes

    def test_exception_escalate_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/escalate" in routes

    def test_exception_analyze_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/analyze" in routes

    def test_exception_explain_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/explain" in routes

    def test_exception_similar_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/similar" in routes

    def test_exception_evidence_route(self, client):
        routes = self._get_routes(client)
        assert "/exceptions/{exception_id}/evidence" in routes

    def test_feedback_route(self, client):
        routes = self._get_routes(client)
        assert "/feedback" in routes

    def test_learning_metrics_route(self, client):
        routes = self._get_routes(client)
        assert "/learning/metrics" in routes

    def test_learning_datasets_route(self, client):
        routes = self._get_routes(client)
        assert "/learning/datasets" in routes

    def test_metrics_route(self, client):
        routes = self._get_routes(client)
        assert "/metrics" in routes

    def test_metrics_safety_route(self, client):
        routes = self._get_routes(client)
        assert "/metrics/safety" in routes

    def test_metrics_throughput_route(self, client):
        routes = self._get_routes(client)
        assert "/metrics/throughput" in routes

    def test_metrics_batch_route(self, client):
        routes = self._get_routes(client)
        assert "/metrics/batches/{batch_id}" in routes

    def test_models_route(self, client):
        routes = self._get_routes(client)
        assert "/models" in routes

    def test_model_detail_route(self, client):
        routes = self._get_routes(client)
        assert "/models/{model_id}" in routes

    def test_model_lineage_route(self, client):
        routes = self._get_routes(client)
        assert "/models/{model_id}/lineage" in routes


# ─────────────────────────────────────────────────────────────────────────────
# OpenAPI
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAPI:
    """Test OpenAPI documentation generation."""

    def test_openapi_json(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec

    def test_docs_endpoint(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_has_route_tags(self, client):
        spec = client.get("/openapi.json").json()
        # Collect all tags used across routes
        all_route_tags = set()
        for path_info in spec.get("paths", {}).values():
            for method_info in path_info.values():
                if isinstance(method_info, dict):
                    for t in method_info.get("tags", []):
                        all_route_tags.add(t)
        assert "Batches" in all_route_tags
        assert "Exceptions" in all_route_tags
        assert "Intelligence" in all_route_tags
        assert "Learning" in all_route_tags
        assert "Metrics" in all_route_tags
        assert "Models" in all_route_tags
        assert "System" in all_route_tags

    def test_openapi_path_count(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec.get("paths", {})
        # Should have at least 25 distinct path entries
        assert len(paths) >= 25


# ─────────────────────────────────────────────────────────────────────────────
# Health Endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_has_phases(self, client):
        response = client.get("/health")
        data = response.json()
        assert "phases" in data
        assert isinstance(data["phases"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchRoutes:
    """Test batch management routes."""

    def test_list_batches(self, client):
        response = client.get("/batches")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data

    def test_create_batch(self, client):
        response = client.post("/batches", json={"name": "test-batch"})
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True

    def test_get_batch_not_found(self, client):
        response = client.get("/batches/NONEXISTENT")
        assert response.status_code == 404

    def test_run_batch_not_found(self, client):
        response = client.post("/batches/NONEXISTENT/run")
        assert response.status_code == 404

    def test_get_batch_summary_not_found(self, client):
        response = client.get("/batches/NONEXISTENT/summary")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Exception Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestExceptionRoutes:
    """Test exception management routes."""

    def test_list_exceptions(self, client):
        response = client.get("/exceptions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_list_exceptions_with_params(self, client):
        response = client.get("/exceptions?limit=10&offset=0")
        assert response.status_code == 200

    def test_get_exception_not_found(self, client):
        response = client.get("/exceptions/NONEXISTENT")
        assert response.status_code == 404

    def test_resolve_exception_not_found(self, client):
        """Resolve with nonexistent exception returns 404."""
        response = client.post(
            "/exceptions/EXC-001/resolve",
            json={"resolution_type": "REFUND_ADJUSTMENT", "adjustment_paise": 50000},
        )
        assert response.status_code == 404

    def test_approve_exception_not_found(self, client):
        """Approve with nonexistent exception returns 404."""
        response = client.post(
            "/exceptions/EXC-001/approve",
            json={"approved_by": "reviewer@example.com"},
        )
        assert response.status_code == 404

    def test_reject_exception_not_found(self, client):
        """Reject with nonexistent exception returns 404."""
        response = client.post(
            "/exceptions/EXC-001/reject",
            json={"rejected_by": "reviewer@example.com", "reason": "Incorrect resolution"},
        )
        assert response.status_code == 404

    def test_escalate_exception_without_reason(self, client):
        response = client.post("/exceptions/EXC-001/escalate", json={"reason": ""})
        assert response.status_code == 422

    def test_escalate_exception_not_found(self, client):
        """Escalate with nonexistent exception returns 404."""
        response = client.post(
            "/exceptions/EXC-001/escalate",
            json={"reason": "High value, needs manual review", "escalated_by": "ops@example.com"},
        )
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestIntelligenceRoutes:
    """Test intelligence routes (analyze, explain, similar, evidence)."""

    def test_similar_cases_not_found(self, client):
        response = client.get("/exceptions/EXC-001/similar")
        assert response.status_code == 404

    def test_evidence_not_found(self, client):
        response = client.get("/exceptions/EXC-001/evidence")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Learning Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestLearningRoutes:
    """Test learning and feedback routes."""

    def test_record_feedback_valid(self, client):
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "APPROVE",
                "workflow_id": "WF-001",
                "exception_id": "EXC-001",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True

    def test_record_feedback_reject(self, client):
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "REJECT",
                "workflow_id": "WF-001",
                "rejection_reason": "Incorrect resolution",
            },
        )
        assert response.status_code == 201

    def test_record_feedback_correct(self, client):
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "CORRECT",
                "workflow_id": "WF-001",
                "original_resolution": "REFUND",
                "corrected_resolution": "REFUND_ADJUSTMENT",
                "correction_reason": "Wrong amount calculated",
            },
        )
        assert response.status_code == 201

    def test_record_feedback_escalate(self, client):
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "ESCALATE",
                "workflow_id": "WF-001",
                "escalation_reason": "High risk",
            },
        )
        assert response.status_code == 201

    def test_record_feedback_invalid_type(self, client):
        response = client.post(
            "/feedback",
            json={"feedback_type": "INVALID_TYPE", "workflow_id": "WF-001"},
        )
        assert response.status_code == 422

    def test_record_feedback_missing_workflow(self, client):
        response = client.post(
            "/feedback",
            json={"feedback_type": "APPROVE"},
        )
        assert response.status_code == 422

    def test_get_feedback_not_found(self, client):
        response = client.get("/feedback/NONEXISTENT")
        assert response.status_code == 404

    def test_learning_metrics(self, client):
        response = client.get("/learning/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_learning_datasets(self, client):
        response = client.get("/learning/datasets")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsRoutes:
    """Test metrics routes."""

    def test_get_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_safety_metrics(self, client):
        response = client.get("/metrics/safety")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_throughput_metrics(self, client):
        response = client.get("/metrics/throughput")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_batch_metrics_not_found(self, client):
        response = client.get("/metrics/batches/NONEXISTENT")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Model Routes
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRoutes:
    """Test model management routes."""

    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_model_not_found(self, client):
        response = client.get("/models/NONEXISTENT")
        assert response.status_code == 404

    def test_get_model_lineage(self, client):
        response = client.get("/models/MODEL-001/lineage")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    """Test error response format."""

    def test_not_found_format(self, client):
        response = client.get("/batches/NONEXISTENT")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "error_code" in data

    def test_validation_error_format(self, client):
        response = client.post("/feedback", json={"feedback_type": "INVALID"})
        assert response.status_code == 422

    def test_404_has_error_code(self, client):
        response = client.get("/models/NONEXISTENT")
        data = response.json()
        assert data["error_code"] == "NOT_FOUND"


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────


class TestCORS:
    """Test CORS middleware is configured."""

    def test_cors_headers(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
