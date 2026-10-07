"""
Phase 14 — Observability tests.

These are real, executable tests. They assert on actually-emitted telemetry
(spans/metrics recorded in-process) and on real HTTP behaviour, not on source
text.

Coverage:
  * request correlation (generate / propagate / echo)
  * structured logging correlation + secret safety
  * span recording, error spans, sensitive attribute masking
  * metric recording and high-cardinality label rejection
  * health + readiness endpoints
  * instrumentation of reconciliation, evidence, ML, historical retrieval,
    policy, provider execution, post-execution verification, LangGraph, MCP
  * observability failure isolation (telemetry must never change business)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import observability as obs
from app.core.structured_logging import api_logger, set_correlation_ids
from auth_test_helper import (
    authenticated_test_client,
    make_test_token,
    make_test_verifier,
    set_token_verifier,
)


BATCH_DIR = Path(__file__).resolve().parent.parent / "data" / "batch_001"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return authenticated_test_client(app)


@pytest.fixture(autouse=True)
def clean_telemetry():
    """Every test starts and ends with an empty telemetry registry."""
    obs.reset_observability()
    yield
    obs.reset_observability()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Request correlation
# ─────────────────────────────────────────────────────────────────────────────


class TestRequestCorrelation:
    def test_request_id_generated_when_absent(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        request_id = response.headers.get("X-Request-ID")
        assert request_id
        assert request_id.startswith("req-")

    def test_request_id_propagated_when_supplied(self, client):
        custom = "req-client-supplied-001"
        response = client.get("/health", headers={"X-Request-ID": custom})
        assert response.headers.get("X-Request-ID") == custom

    def test_request_id_returned_in_response_header(self, client):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_request_id_is_not_accepted_as_identity(self):
        """A caller-supplied request ID must not influence authorization."""
        # Supply a "privileged-looking" request id; an unauthenticated request
        # to an authenticated endpoint must still be rejected.
        bare = TestClient(app, raise_server_exceptions=False)
        response = bare.get("/auth/me", headers={"X-Request-ID": "req-admin-override"})
        assert response.status_code in (401, 403)
        # Request ID echoing must not grant identity.
        assert response.headers.get("X-Request-ID") in (None, "req-admin-override")

    def test_http_request_span_recorded(self, client):
        client.get("/health")
        assert obs.SpanName.HTTP_REQUEST in obs.recorded_span_names()

    def test_http_span_has_safe_attributes(self, client):
        client.get("/health")
        spans = obs.REGISTRY.spans_named(obs.SpanName.HTTP_REQUEST)
        assert spans, "expected an HTTP span"
        span = spans[-1]
        assert span.attributes.get("http.request.method") == "GET"
        assert span.attributes.get("http.response.status_code") == 200
        assert span.duration_ms >= 0

    def test_http_metrics_recorded(self, client):
        client.get("/health")
        assert obs.REGISTRY.metric_total(obs.MetricName.HTTP_REQUESTS) >= 1
        assert obs.REGISTRY.metric_total(obs.MetricName.HTTP_REQUEST_DURATION) >= 0

    def test_http_error_metric_recorded(self, client):
        response = client.get("/exceptions/DOES-NOT-EXIST-P14")
        assert response.status_code >= 400
        assert obs.REGISTRY.metric_total(obs.MetricName.HTTP_ERRORS) >= 1

    def test_route_label_is_template_not_raw_id(self, client):
        """Raw entity IDs must never become metric dimensions."""
        client.get("/exceptions/EXC-SECRET-ID-P14")
        routes = [
            labels.get("route", "")
            for labels in obs.metric_labels(obs.MetricName.HTTP_REQUESTS)
        ]
        assert routes, "expected route labels"
        assert all("EXC-SECRET-ID-P14" not in r for r in routes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Structured logging
# ─────────────────────────────────────────────────────────────────────────────


class TestStructuredLogging:
    def test_structured_log_contains_correlation_information(self, caplog):
        set_correlation_ids(request_id="req-log-correlation-001")
        with caplog.at_level(logging.INFO, logger="closeloop.api"):
            api_logger.info("TEST_EVENT", "hello", outcome="ok")
        assert "req-log-correlation-001" in caplog.text
        assert "TEST_EVENT" in caplog.text

    def test_sensitive_headers_are_not_logged(self, caplog):
        set_token_verifier(make_test_verifier())
        token = make_test_token(subject="token-secret-subject-p14")
        client = TestClient(app, raise_server_exceptions=False)

        with caplog.at_level(logging.DEBUG, logger="api.errors"):
            client.get("/health", headers={"Authorization": f"Bearer {token}"})
            client.get("/health", headers={"Authorization": f"Bearer {token}"})

        assert token not in caplog.text
        assert "Bearer " + token not in caplog.text

    def test_sanitizer_masks_tokens_in_error_messages(self):
        from app.core.structured_logging import sanitize_error

        message = "failed with Authorization: Bearer abcdef123456"
        assert "abcdef123456" not in sanitize_error(message)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spans
# ─────────────────────────────────────────────────────────────────────────────


class TestSpans:
    def test_span_records_duration_and_attributes(self):
        with obs.span("closeloop.test.span", {"provider": "razorpay"}) as sp:
            sp.set_attribute("outcome", "matched")
        spans = obs.REGISTRY.spans_named("closeloop.test.span")
        assert len(spans) == 1
        assert spans[0].attributes["provider"] == "razorpay"
        assert spans[0].attributes["outcome"] == "matched"
        assert spans[0].status == "OK"

    def test_span_records_error_and_reraises(self):
        with pytest.raises(ValueError):
            with obs.span("closeloop.test.error") as sp:
                sp.set_attribute("alpha", 1)
                raise ValueError("boom")
        spans = obs.REGISTRY.spans_named("closeloop.test.error")
        assert spans and spans[0].status == "ERROR"
        assert spans[0].error_type == "ValueError"

    def test_nested_spans_carry_parent(self):
        with obs.span("closeloop.test.parent"):
            with obs.span("closeloop.test.child"):
                pass
        parent = obs.REGISTRY.spans_named("closeloop.test.parent")[0]
        child = obs.REGISTRY.spans_named("closeloop.test.child")[0]
        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_sensitive_span_attributes_are_masked(self):
        with obs.span("closeloop.test.secrets", {"authorization": "Bearer abc", "provider": "razorpay"}):
            pass
        span = obs.REGISTRY.spans_named("closeloop.test.secrets")[0]
        assert span.attributes["authorization"] == "***MASKED***"
        assert span.attributes["provider"] == "razorpay"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics + cardinality guard
# ─────────────────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_counter_accumulates(self):
        obs.record_counter("closeloop.test.counter", 1, {"outcome": "ok"})
        obs.record_counter("closeloop.test.counter", 2, {"outcome": "ok"})
        assert obs.REGISTRY.metric_total("closeloop.test.counter") == 3

    def test_histogram_tracks_count_and_range(self):
        for value in (5.0, 15.0, 25.0):
            obs.record_histogram("closeloop.test.duration", value)
        series = obs.REGISTRY.metrics_named("closeloop.test.duration")[0]
        assert series.count == 3
        assert series.minimum == 5.0
        assert series.maximum == 25.0

    def test_high_cardinality_labels_are_dropped(self):
        safe, dropped = obs.sanitize_labels(
            {"route": "/health", "exception_id": "EXC-1", "request_id": "req-1"}
        )
        assert safe == {"route": "/health"}
        assert set(dropped) == {"exception_id", "request_id"}

    def test_recording_with_id_labels_drops_them(self):
        obs.record_counter("closeloop.test.guard", 1, {"payment_id": "PAY-1", "outcome": "ok"})
        labels = obs.metric_labels("closeloop.test.guard")
        assert labels == [{"outcome": "ok"}]
        assert obs.REGISTRY.metric_total(obs.MetricName.DROPPED_LABELS) >= 1

    def test_no_high_cardinality_labels_in_any_recorded_metric(self, client):
        client.get("/health")
        client.get("/exceptions/EXC-P14")
        snapshot = obs.observability_snapshot()
        for metric in snapshot["metrics"]:
            for key in metric["labels"]:
                assert not obs.is_high_cardinality_label(key), f"bad label {key} on {metric['name']}"

    def test_sensitive_label_values_are_masked(self):
        obs.record_counter("closeloop.test.safe", 1, {"authorization": "Bearer xyz"})
        labels = obs.metric_labels("closeloop.test.safe")
        assert labels == [{"authorization": "***MASKED***"}]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Health / readiness
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthAndReadiness:
    def test_health_endpoint_works_without_auth(self):
        bare = TestClient(app, raise_server_exceptions=False)
        response = bare.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "14" in body["phases"]

    def test_ready_endpoint_reports_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["database"] is True
        assert body["required"] == ["database"]

    def test_ready_endpoint_reports_optional_dependencies(self, client):
        body = client.get("/ready").json()
        assert "ml_classifier" in body["checks"]
        assert "llm" in body["checks"]
        assert "observability" in body["checks"]

    def test_ready_returns_503_when_required_dependency_down(self, client, monkeypatch):
        class _BoomEngine:
            def connect(self):
                raise RuntimeError("database down")

        monkeypatch.setattr("app.database.database.engine", _BoomEngine(), raising=True)
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["checks"]["database"] is False

    def test_ready_does_not_leak_secrets(self, client):
        body = client.get("/ready").text.lower()
        for needle in ("password", "secret", "token", "postgresql://", "api_key"):
            assert needle not in body


# ─────────────────────────────────────────────────────────────────────────────
# 6. Domain instrumentation — reconciliation
# ─────────────────────────────────────────────────────────────────────────────


class TestReconciliationObservability:
    def test_reconciliation_emits_telemetry(self):
        from app.reconciliation.batch import BatchReconciler

        summary = BatchReconciler().reconcile_batch("BATCH-P14-TEST", str(BATCH_DIR))
        assert summary is not None
        assert obs.SpanName.RECONCILIATION in obs.recorded_span_names()
        assert obs.REGISTRY.metric_total(obs.MetricName.RECONCILIATION_RUNS) >= 1
        assert obs.REGISTRY.metric_total(obs.MetricName.RECONCILIATION_DURATION) >= 0

    def test_reconciliation_failure_is_observable(self):
        from app.reconciliation.batch import BatchReconciler

        with pytest.raises(Exception):
            BatchReconciler().reconcile_batch("BATCH-P14-BAD", str(BATCH_DIR / "does-not-exist"))

        spans = obs.REGISTRY.spans_named(obs.SpanName.RECONCILIATION)
        assert spans and spans[-1].status == "ERROR"
        assert obs.REGISTRY.metric_total(obs.MetricName.RECONCILIATION_FAILURES) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Domain instrumentation — execution + closed loop
# ─────────────────────────────────────────────────────────────────────────────


def _action_request(**overrides):
    base = {
        "action_id": "ACT-P14",
        "idempotency_key": "key-p14-001",
        "workflow_id": "WF-P14",
        "exception_id": "EXC-P14",
        "case_id": "CASE-P14",
        "candidate_id": "CAND-P14",
        "resolution_type": "APPLY_FEE_CORRECTION",
        "financial_adjustment_paise": 3000,
        "authorization_source": "AUTO_GUARDRAIL",
        "verification_passed": True,
        "guardrail_decision": "AUTO",
        "guardrail_confidence": 0.85,
        "evidence_summary": {"coverage": 0.9},
    }
    base.update(overrides)
    return base


class TestExecutionObservability:
    def test_provider_execution_emits_telemetry(self):
        from app.services.execution import ResolutionExecutionService

        service = ResolutionExecutionService()
        result = service.execute(_action_request(), None)
        assert result is not None
        assert obs.SpanName.PROVIDER_EXECUTE in obs.recorded_span_names()
        assert obs.REGISTRY.metric_total(obs.MetricName.EXECUTION_ATTEMPTS) >= 1
        assert obs.REGISTRY.metric_total(obs.MetricName.EXECUTION_DURATION) >= 0

    def test_provider_failure_emits_telemetry(self):
        from app.services.execution import ResolutionExecutionService

        service = ResolutionExecutionService()
        # A malformed request makes the decorated boundary raise.
        with pytest.raises(Exception):
            service.execute(None, None)
        spans = obs.REGISTRY.spans_named(obs.SpanName.PROVIDER_EXECUTE)
        assert spans and spans[-1].status == "ERROR"
        assert obs.REGISTRY.metric_total(obs.MetricName.EXECUTION_FAILED) >= 1

    def test_post_execution_verification_boundary_is_instrumented(self):
        from app.services.post_execution_reconciliation import (
            PostExecutionReconciliationService,
        )

        service = PostExecutionReconciliationService()
        # Wiring check: the boundary is decorated, so a call that fails
        # validation still produces a span on the verification channel.
        try:
            service.verify_and_close(
                None,
                exception=SimpleNamespace(status="EXECUTED", exception_id="EXC-P14"),
                provider=None,
                case_id="CASE-P14",
                batch_id="BATCH-P14",
                payment_id="PAY-P14",
            )
        except Exception:
            pass
        assert obs.SpanName.POST_EXECUTION_RECONCILE in obs.recorded_span_names()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Domain instrumentation — ML / retrieval / evidence / policy
# ─────────────────────────────────────────────────────────────────────────────


class TestIntelligenceObservability:
    def test_ml_inference_emits_telemetry(self):
        from app.ml.classifier import ExceptionClassifier

        try:
            ExceptionClassifier().predict(np.zeros((1, 4)))
        except Exception:
            pass
        assert obs.SpanName.ML_PREDICT in obs.recorded_span_names()
        recorded = (
            obs.REGISTRY.metric_total(obs.MetricName.ML_PREDICTIONS)
            + obs.REGISTRY.metric_total(obs.MetricName.ML_FAILURES)
        )
        assert recorded >= 1

    def test_ml_telemetry_does_not_record_feature_values(self):
        from app.ml.classifier import ExceptionClassifier

        try:
            ExceptionClassifier().predict(np.full((1, 4), 123456.789))
        except Exception:
            pass
        spans = obs.REGISTRY.spans_named(obs.SpanName.ML_PREDICT)
        assert spans
        flattened = str(spans[-1].attributes)
        assert "123456" not in flattened

    def test_historical_retrieval_emits_telemetry(self):
        from app.services.similarity_service import SimilarityService

        try:
            SimilarityService(session=None).search({})
        except Exception:
            pass
        assert obs.SpanName.HISTORICAL_RETRIEVE in obs.recorded_span_names()

    def test_evidence_retrieval_emits_telemetry(self):
        from app.services.evidence_retrieval import EvidenceRetrievalService

        try:
            EvidenceRetrievalService(session=None).retrieve_by_exception_id("EXC-P14")
        except Exception:
            pass
        assert obs.SpanName.EVIDENCE in obs.recorded_span_names()

    def test_policy_evaluation_emits_telemetry(self):
        from app.services.guardrail_engine import GuardrailEngine

        stub = SimpleNamespace(
            exception_id="EXC-P14",
            confidence=0.9,
            risk_category="LOW",
            candidates=[],
            selected_candidate=None,
        )
        try:
            GuardrailEngine().evaluate(stub, {})
        except Exception:
            pass
        assert obs.SpanName.POLICY_EVALUATE in obs.recorded_span_names()
        recorded = (
            obs.REGISTRY.metric_total(obs.MetricName.POLICY_EVALUATIONS)
            + obs.REGISTRY.metric_total(obs.MetricName.POLICY_FAILURES)
            + obs.REGISTRY.metric_total(obs.MetricName.POLICY_DECISIONS)
        )
        assert recorded >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Orchestration instrumentation — LangGraph + MCP
# ─────────────────────────────────────────────────────────────────────────────


class TestOrchestrationObservability:
    def test_langgraph_workflow_emits_telemetry(self):
        from app.agent.workflow import run_workflow

        try:
            run_workflow("EXC-P14-DOES-NOT-EXIST")
        except Exception:
            pass
        assert obs.SpanName.LANGGRAPH_WORKFLOW in obs.recorded_span_names()
        assert obs.REGISTRY.metric_total(obs.MetricName.LANGGRAPH_RUNS) >= 1

    def test_mcp_invocation_emits_telemetry(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from mcp.schemas import MCPToolRequest
        from mcp.server import MCPServer

        server = MCPServer()
        try:
            server.invoke(MCPToolRequest(tool_name="closeloop.nonexistent.tool", parameters={}))
        except Exception:
            pass
        assert obs.SpanName.MCP_TOOL in obs.recorded_span_names()
        assert obs.REGISTRY.metric_total(obs.MetricName.MCP_INVOCATIONS) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. Failure isolation — telemetry must never change business behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestObservabilityFailureIsolation:
    def test_registry_failure_does_not_break_decorated_call(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("telemetry backend down")

        monkeypatch.setattr(obs.REGISTRY, "record_span", _boom)

        @obs.observed("closeloop.test.isolated", metric="closeloop.test.isolated.count")
        def work(x):
            return x * 2

        assert work(21) == 42

    def test_registry_failure_does_not_break_real_execution(self, monkeypatch):
        from app.services.execution import ResolutionExecutionService

        def _boom(*args, **kwargs):
            raise RuntimeError("telemetry backend down")

        monkeypatch.setattr(obs.REGISTRY, "record_span", _boom, raising=True)
        result = ResolutionExecutionService().execute(_action_request(idempotency_key="key-p14-iso"), None)
        assert result is not None
        assert getattr(result, "status", None) is not None

    def test_disabled_observability_records_nothing_but_keeps_working(self, client):
        obs.set_observability_enabled(False)
        try:
            response = client.get("/health")
            assert response.status_code == 200
            assert obs.REGISTRY.spans() == []
            assert obs.REGISTRY.metrics() == []
        finally:
            obs.set_observability_enabled(True)

    def test_metric_recording_failure_returns_false_and_continues(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("metrics exporter down")

        monkeypatch.setattr(obs.REGISTRY, "record_metric", _boom)
        assert obs.record_counter("closeloop.test.broken", 1) is False

    def test_span_context_does_not_swallow_business_exceptions(self):
        with pytest.raises(KeyError):
            with obs.span("closeloop.test.business"):
                raise KeyError("must propagate")

    def test_observability_never_opens_or_closes_exceptions(self, monkeypatch):
        """Telemetry absence cannot change the Phase 10 decision itself."""
        from app.core import observability as o

        monkeypatch.setattr(o.REGISTRY, "record_span", lambda *a, **k: None)
        monkeypatch.setattr(o.REGISTRY, "record_metric", lambda *a, **k: (False, []))

        stub_result = SimpleNamespace(
            financially_verified=False,
            exception_closed=False,
            decision=SimpleNamespace(value="NOT_VERIFIED"),
        )
        # The outcome resolver runs with telemetry disabled and must not modify
        # the authoritative result object.
        from app.services.post_execution_reconciliation import _closed_loop_outcome

        _closed_loop_outcome(stub_result)
        assert stub_result.financially_verified is False
        assert stub_result.exception_closed is False
        assert stub_result.decision.value == "NOT_VERIFIED"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Snapshot / diagnostics surface
# ─────────────────────────────────────────────────────────────────────────────


class TestDiagnostics:
    def test_snapshot_contains_spans_and_metrics(self, client):
        client.get("/health")
        snapshot = obs.observability_snapshot()
        assert snapshot["enabled"] is True
        assert any(span["name"] == obs.SpanName.HTTP_REQUEST for span in snapshot["spans"])
        assert any(m["name"] == obs.MetricName.HTTP_REQUESTS for m in snapshot["metrics"])

    def test_otel_availability_is_reported(self):
        info = obs.init_observability()
        assert "otel_available" in info
        assert isinstance(info["otel_available"], bool)

    def test_span_names_are_stable_and_namespaced(self):
        names = [
            obs.SpanName.HTTP_REQUEST,
            obs.SpanName.RECONCILIATION,
            obs.SpanName.EVIDENCE,
            obs.SpanName.ML_PREDICT,
            obs.SpanName.HISTORICAL_RETRIEVE,
            obs.SpanName.RESOLUTION_PROPOSE,
            obs.SpanName.POLICY_EVALUATE,
            obs.SpanName.PROVIDER_EXECUTE,
            obs.SpanName.POST_EXECUTION_RECONCILE,
            obs.SpanName.LANGGRAPH_WORKFLOW,
            obs.SpanName.MCP_TOOL,
        ]
        for name in names:
            assert name.startswith("closeloop.")
            assert re.match(r"^closeloop\.[a-z_.]+$", name)
            assert " " not in name and name == name.lower()
