"""
HTTP observability middleware (Phase 14).

Adds request-scoped telemetry on top of the existing ``RequestIDMiddleware``
correlation, without changing request handling:

* binds a trace ID for the request
* propagates the request ID into structured-logging correlation context
* emits one ``closeloop.http.request`` span per request
* records request count / error count / duration metrics with safe labels

The middleware never alters response bodies, status codes or headers, and any
telemetry failure is swallowed so the request proceeds normally.
"""

from __future__ import annotations

import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware

from app.api.errors import generate_request_id, get_request_id
from app.core import observability as obs
from app.core.structured_logging import set_correlation_ids


def _route_label(request: Any) -> str:
    """A bounded-cardinality route label.

    Prefers the matched route template (``/exceptions/{exception_id}``) so raw
    IDs never become metric dimensions. Falls back to the first path segment.
    """
    try:
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        if path:
            return str(path)
    except Exception:
        pass
    try:
        parts = [p for p in request.url.path.split("/") if p]
        return f"/{parts[0]}" if parts else "/"
    except Exception:
        return "unknown"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Emit request telemetry. Read-only: never changes the response."""

    async def dispatch(self, request, call_next):
        request_id = get_request_id() or request.headers.get("X-Request-ID") or generate_request_id()

        # Propagate correlation into structured logging (telemetry context only).
        try:
            set_correlation_ids(request_id=request_id)
        except Exception:
            pass

        method = request.method
        route = _route_label(request)
        started = time.perf_counter()

        with obs.trace_context(), obs.span(
            obs.SpanName.HTTP_REQUEST,
            {"http.request.method": method, "http.route": route},
            kind="server",
        ) as sp:
            try:
                response = await call_next(request)
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                sp.set_status("ERROR", type(exc).__name__)
                sp.set_attribute("error.type", type(exc).__name__)
                _record_http(method, route, 500, duration_ms)
                raise

            status_code = getattr(response, "status_code", 0)
            duration_ms = (time.perf_counter() - started) * 1000.0

            sp.set_attribute("http.response.status_code", int(status_code))
            sp.set_attribute("duration_ms", round(duration_ms, 3))
            if int(status_code) >= 500:
                sp.set_status("ERROR")

            _record_http(method, route, status_code, duration_ms)
            return response


def _record_http(method: str, route: str, status_code: int, duration_ms: float) -> None:
    """Record HTTP metrics. Labels are low-cardinality only. Never raises."""
    try:
        status_class = f"{int(status_code) // 100}xx" if status_code else "unknown"
        labels = {"method": method, "route": route, "status": status_class}
        obs.record_counter(obs.MetricName.HTTP_REQUESTS, 1, labels)
        obs.record_histogram(obs.MetricName.HTTP_REQUEST_DURATION, round(duration_ms, 3), labels)
        if int(status_code) >= 400:
            obs.record_counter(obs.MetricName.HTTP_ERRORS, 1, labels)
    except Exception:  # pragma: no cover - telemetry must never break a request
        pass
