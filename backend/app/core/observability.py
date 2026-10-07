"""
Observability (Phase 14) — tracing + metrics for Razorpay CloseLoop.

This module is *read-only telemetry*. It never decides anything financial.

Design rules
------------
* **Fail independently.** Every telemetry call is best-effort. If a tracer,
  metric exporter or logger backend is unavailable or raises, the exception is
  swallowed and the business operation proceeds with its existing semantics.
* **Local-first.** OpenTelemetry is used when the ``opentelemetry`` API/SDK is
  importable. No exporter, AWS credential, CloudWatch/X-Ray or collector is
  required. With no exporter configured, spans are recorded in-process only.
* **No high-cardinality metric labels.** ``request_id``, ``trace_id``,
  ``span_id``, ``exception_id``, ``payment_id``, ``batch_id``, ``user_id`` and
  friends are rejected as metric label keys. They belong in logs/traces.
* **No secrets.** Span attributes and metric labels are scrubbed for sensitive
  field names (tokens, authorization headers, keys, passwords) before storage.

The in-process registry exists so telemetry is inspectable without a backend
and so tests can assert on emitted spans/metrics deterministically.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from app.core.structured_logging import mask_sensitive

logger = logging.getLogger("closeloop.observability")


# ============================================================================
# Dependency detection — optional OpenTelemetry
# ============================================================================


def _load_otel():
    """Best-effort import of the OpenTelemetry API. Returns None if absent."""
    try:
        from opentelemetry import trace as otel_trace  # type: ignore

        return otel_trace
    except Exception:  # pragma: no cover - exercised only when OTel is absent
        return None


_OTEL_TRACE = _load_otel()


def is_otel_available() -> bool:
    """Whether the OpenTelemetry API is importable in this environment."""
    return _OTEL_TRACE is not None


# ============================================================================
# Stable span / metric names
# ============================================================================


class SpanName:
    """Stable span names for the CloseLoop workflow."""

    HTTP_REQUEST = "closeloop.http.request"
    INGESTION = "closeloop.ingestion"
    RECONCILIATION = "closeloop.reconciliation"
    EVIDENCE = "closeloop.evidence"
    ML_PREDICT = "closeloop.ml.predict"
    HISTORICAL_RETRIEVE = "closeloop.historical.retrieve"
    RESOLUTION_PROPOSE = "closeloop.resolution.propose"
    POLICY_EVALUATE = "closeloop.policy.evaluate"
    HUMAN_APPROVAL = "closeloop.human.approval"
    PROVIDER_EXECUTE = "closeloop.provider.execute"
    POST_EXECUTION_RECONCILE = "closeloop.post_execution.reconcile"
    LANGGRAPH_WORKFLOW = "closeloop.langgraph.workflow"
    MCP_TOOL = "closeloop.mcp.tool"


class MetricName:
    """Stable metric names for the CloseLoop workflow."""

    # HTTP
    HTTP_REQUESTS = "closeloop.http.requests"
    HTTP_REQUEST_DURATION = "closeloop.http.request.duration"
    HTTP_ERRORS = "closeloop.http.errors"

    # Reconciliation
    RECONCILIATION_RUNS = "closeloop.reconciliation.runs"
    RECONCILIATION_FAILURES = "closeloop.reconciliation.failures"
    RECONCILIATION_MATCHED = "closeloop.reconciliation.matched"
    RECONCILIATION_MISMATCHED = "closeloop.reconciliation.mismatched"
    RECONCILIATION_DURATION = "closeloop.reconciliation.duration"

    # Exceptions
    EXCEPTIONS_CREATED = "closeloop.exceptions.created"
    EXCEPTIONS_BY_TYPE = "closeloop.exceptions.by_type"
    EXCEPTIONS_BY_STATE = "closeloop.exceptions.by_state"
    EXCEPTIONS_CLOSED = "closeloop.exceptions.closed"
    EXCEPTIONS_ESCALATED = "closeloop.exceptions.escalated"
    EXCEPTIONS_UNRESOLVED = "closeloop.exceptions.unresolved"

    # Resolution
    RESOLUTION_PROPOSALS = "closeloop.resolution.proposals"
    RESOLUTION_FAILURES = "closeloop.resolution.failures"
    RESOLUTION_HUMAN_REVIEW = "closeloop.resolution.human_review"
    RESOLUTION_APPROVED = "closeloop.resolution.approved"
    RESOLUTION_REJECTED = "closeloop.resolution.rejected"
    RESOLUTION_VERIFIED = "closeloop.resolution.verified"
    RESOLUTION_VERIFICATION_FAILED = "closeloop.resolution.verification_failed"

    # Provider execution
    EXECUTION_ATTEMPTS = "closeloop.execution.attempts"
    EXECUTION_SUCCESS = "closeloop.execution.success"
    EXECUTION_FAILED = "closeloop.execution.failed"
    EXECUTION_DURATION = "closeloop.execution.duration"

    # Closed loop
    POST_EXECUTION_RUNS = "closeloop.post_execution.runs"
    POST_EXECUTION_VERIFIED = "closeloop.post_execution.verified"
    POST_EXECUTION_FAILED = "closeloop.post_execution.failed"
    POST_EXECUTION_DURATION = "closeloop.post_execution.duration"

    # ML / retrieval / policy
    ML_PREDICTIONS = "closeloop.ml.predictions"
    ML_FAILURES = "closeloop.ml.failures"
    ML_DURATION = "closeloop.ml.duration"
    HISTORICAL_RETRIEVALS = "closeloop.historical.retrievals"
    HISTORICAL_FAILURES = "closeloop.historical.failures"
    HISTORICAL_DURATION = "closeloop.historical.duration"
    POLICY_EVALUATIONS = "closeloop.policy.evaluations"
    POLICY_DECISIONS = "closeloop.policy.decisions"
    POLICY_FAILURES = "closeloop.policy.failures"

    # Orchestration
    LANGGRAPH_RUNS = "closeloop.langgraph.runs"
    LANGGRAPH_FAILURES = "closeloop.langgraph.failures"
    LANGGRAPH_DURATION = "closeloop.langgraph.duration"
    MCP_INVOCATIONS = "closeloop.mcp.invocations"
    MCP_FAILURES = "closeloop.mcp.failures"
    MCP_DURATION = "closeloop.mcp.duration"

    # Self-observation
    SPANS_RECORDED = "closeloop.observability.spans"
    DROPPED_LABELS = "closeloop.observability.dropped_labels"
    TELEMETRY_ERRORS = "closeloop.observability.errors"


# ============================================================================
# Label safety
# ============================================================================

#: Label keys that would blow up metric cardinality. They are dropped, never
#: silently kept. Correlation belongs in logs and traces.
HIGH_CARDINALITY_LABEL_KEYS = frozenset({
    "request_id", "trace_id", "span_id", "parent_span_id", "correlation_id",
    "exception_id", "payment_id", "batch_id", "user_id", "actor_id",
    "subject", "resolution_id", "resolution_action_id", "reconciliation_run_id",
    "workflow_id", "run_id", "idempotency_key", "provider_ref",
})

#: Attribute/label keys whose values must never be stored.
SENSITIVE_ATTRIBUTE_KEYS = frozenset({
    "authorization", "authorization_header", "auth_header", "cookie",
    "set_cookie", "token", "access_token", "refresh_token", "id_token",
    "jwt", "api_key", "apikey", "secret", "password", "credential",
    "private_key", "client_secret", "secret_key", "session_token",
    "database_url",
})

MAX_METRIC_SERIES = 2000
MAX_SPANS = 2000


def is_high_cardinality_label(key: str) -> bool:
    """Whether a label key is a per-entity ID that must not be a metric label."""
    return str(key).strip().lower() in HIGH_CARDINALITY_LABEL_KEYS


def is_sensitive_key(key: str) -> bool:
    """Whether a key names a credential/secret that must never be recorded."""
    return str(key).strip().lower().replace("-", "_") in SENSITIVE_ATTRIBUTE_KEYS


def sanitize_labels(labels: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, str], List[str]]:
    """Return (safe_labels, dropped_keys).

    Drops high-cardinality ID labels and masks anything that looks sensitive.
    Never raises.
    """
    safe: Dict[str, str] = {}
    dropped: List[str] = []
    if not labels:
        return safe, dropped
    try:
        items = labels.items()
    except Exception:  # pragma: no cover - defensive
        return safe, dropped
    for key, value in items:
        if is_high_cardinality_label(key):
            dropped.append(str(key))
            continue
        if is_sensitive_key(key):
            safe[str(key)] = "***MASKED***"
            continue
        # Values are coerced to short strings: labels are dimensions, not data.
        text = str(value)
        if len(text) > 64:
            text = text[:64]
        safe[str(key)] = text
    return safe, dropped


def sanitize_attributes(attributes: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Scrub span attributes: mask sensitive keys, stringify safely.

    Never raises; returns a plain dict.
    """
    if not attributes:
        return {}
    clean: Dict[str, Any] = {}
    for key, value in attributes.items():
        if is_sensitive_key(key):
            clean[str(key)] = "***MASKED***"
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)
    try:
        return mask_sensitive(clean)
    except Exception:  # pragma: no cover - defensive
        return clean


# ============================================================================
# In-process records
# ============================================================================


@dataclass
class SpanRecord:
    """A recorded span, inspectable without a tracing backend."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str
    start_time: str
    duration_ms: float
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    error_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_type": self.error_type,
            "attributes": dict(self.attributes),
        }


@dataclass
class MetricSeries:
    """A recorded metric series (counter / histogram / gauge)."""

    name: str
    kind: str
    labels: Dict[str, str]
    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    last: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.last = value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def value(self) -> float:
        """Counter/gauge value; for counters this is the running total."""
        return self.total

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "labels": dict(self.labels),
            "count": self.count,
            "total": round(self.total, 6),
            "min": self.minimum,
            "max": self.maximum,
            "last": self.last,
        }


# ============================================================================
# Registry
# ============================================================================


class ObservabilityRegistry:
    """Thread-safe in-process store of spans and metrics.

    This is telemetry storage only — it is never consulted by domain logic.
    """

    def __init__(self, max_spans: int = MAX_SPANS, max_series: int = MAX_METRIC_SERIES):
        self._lock = threading.RLock()
        self._spans: deque = deque(maxlen=max_spans)
        self._metrics: "OrderedDict[Tuple[str, str, Tuple[Tuple[str, str], ...]], MetricSeries]" = OrderedDict()
        self._max_series = max_series
        self._enabled = True

    # ── enable / disable ──

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    # ── spans ──

    def record_span(self, span: SpanRecord) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._spans.append(span)

    def spans(self) -> List[SpanRecord]:
        with self._lock:
            return list(self._spans)

    def spans_named(self, name: str) -> List[SpanRecord]:
        return [s for s in self.spans() if s.name == name]

    # ── metrics ──

    def record_metric(
        self,
        name: str,
        value: float,
        kind: str = "counter",
        labels: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """Record one metric observation. Returns (recorded, dropped_label_keys)."""
        safe_labels, dropped = sanitize_labels(labels)
        with self._lock:
            if not self._enabled:
                return False, dropped
            key = (str(name), str(kind), tuple(sorted(safe_labels.items())))
            series = self._metrics.get(key)
            if series is None:
                if len(self._metrics) >= self._max_series:
                    return False, dropped
                series = MetricSeries(name=str(name), kind=str(kind), labels=dict(safe_labels))
                self._metrics[key] = series
            series.observe(float(value))
        return True, dropped

    def metrics(self) -> List[MetricSeries]:
        with self._lock:
            return list(self._metrics.values())

    def metrics_named(self, name: str) -> List[MetricSeries]:
        return [m for m in self.metrics() if m.name == name]

    def metric_total(self, name: str) -> float:
        """Sum of all series values for a metric name."""
        return sum(m.total for m in self.metrics_named(name))

    # ── snapshot / reset ──

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            enabled = self._enabled
            spans = [s.to_dict() for s in self._spans]
            metrics = [m.to_dict() for m in self._metrics.values()]
        return {"enabled": enabled, "spans": spans, "metrics": metrics}

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()
            self._metrics.clear()
            self._enabled = True


#: Process-wide registry. Tests may reset it.
REGISTRY = ObservabilityRegistry()


# ============================================================================
# Trace context
# ============================================================================

_trace_id_var: ContextVar[str] = ContextVar("observability_trace_id", default="")
_span_stack_var: ContextVar[Tuple[str, ...]] = ContextVar("observability_span_stack", default=())


def current_trace_id() -> str:
    """Trace ID for the current context (empty when none is active)."""
    return _trace_id_var.get("")


def set_trace_id(trace_id: str) -> None:
    """Bind a trace ID for the current context (telemetry only)."""
    _trace_id_var.set(trace_id or "")


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:16]}"


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def current_span_id() -> str:
    stack = _span_stack_var.get(())
    return stack[-1] if stack else ""


@contextmanager
def trace_context(trace_id: str = "") -> Iterator[str]:
    """Bind a trace ID for the duration of the block."""
    token = _trace_id_var.set(trace_id or new_trace_id())
    try:
        yield _trace_id_var.get("")
    finally:
        _trace_id_var.reset(token)


# ============================================================================
# Spans
# ============================================================================


class SpanHandle:
    """Mutable handle returned by :func:`span`."""

    def __init__(self, record: SpanRecord):
        self._record = record

    def set_attribute(self, key: str, value: Any) -> None:
        if is_sensitive_key(key):
            self._record.attributes[str(key)] = "***MASKED***"
            return
        self._record.attributes[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_status(self, status: str, error_type: str = "") -> None:
        self._record.status = str(status)
        if error_type:
            self._record.error_type = str(error_type)


@contextmanager
def span(name: str, attributes: Optional[Mapping[str, Any]] = None, kind: str = "internal") -> Iterator[SpanHandle]:
    """Open a telemetry span. Never raises; never affects business behaviour.

    Usage::

        with span(SpanName.RECONCILIATION, {"provider": "razorpay"}) as sp:
            ...
            sp.set_attribute("outcome", "matched")
    """
    trace_id = current_trace_id()
    trace_token = None
    if not trace_id:
        # Bind a trace ID so nested spans correlate to the same trace.
        trace_id = new_trace_id()
        trace_token = _trace_id_var.set(trace_id)
    parent_span_id = current_span_id()
    record = SpanRecord(
        name=str(name),
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=parent_span_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=0.0,
        attributes=sanitize_attributes(attributes),
    )
    handle = SpanHandle(record)
    stack_token = _span_stack_var.set(_span_stack_var.get(()) + (record.span_id,))
    started = time.perf_counter()
    otel_span = None
    otel_cm = None

    try:
        otel_span, otel_cm = _start_otel_span(record.name, kind)
    except Exception:
        otel_span, otel_cm = None, None

    try:
        yield handle
    except Exception as exc:
        record.status = "ERROR"
        record.error_type = type(exc).__name__
        _finish(record, started, sp=otel_span, cm=otel_cm)
        _safe_record(record)
        _span_stack_var.reset(stack_token)
        if trace_token is not None:
            _trace_id_var.reset(trace_token)
        raise
    else:
        _finish(record, started, sp=otel_span, cm=otel_cm)
        _safe_record(record)
        _span_stack_var.reset(stack_token)
        if trace_token is not None:
            _trace_id_var.reset(trace_token)


def _finish(record: SpanRecord, started: float, sp: Any = None, cm: Any = None) -> None:
    record.duration_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        if cm is not None:
            cm.__exit__(None, None, None)
        if sp is not None:
            try:
                from opentelemetry.trace import Status, StatusCode  # type: ignore

                if record.status == "ERROR":
                    sp.set_status(Status(StatusCode.ERROR, record.error_type or "error"))
                else:
                    sp.set_status(Status(StatusCode.OK))
            except Exception:
                pass
            try:
                sp.end()
            except Exception:
                pass
    except Exception:
        pass


def _start_otel_span(name: str, kind: str):
    """Start an OTel span if the API is available. Returns (span, cm) or (None, None)."""
    if _OTEL_TRACE is None:
        return None, None
    try:
        from opentelemetry.trace import SpanKind  # type: ignore

        kinds = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        tracer = _OTEL_TRACE.get_tracer("closeloop")
        cm = tracer.start_as_current_span(name, kind=kinds.get(kind, SpanKind.INTERNAL))
        sp = cm.__enter__()
        return sp, cm
    except Exception:
        return None, None


def _safe_record(record: SpanRecord) -> None:
    """Record into the in-process registry, swallowing any telemetry failure."""
    try:
        REGISTRY.record_span(record)
    except Exception:  # pragma: no cover - telemetry must never break business
        logger.debug("observability: failed to record span", exc_info=False)
    try:
        REGISTRY.record_metric(MetricName.SPANS_RECORDED, 1, "counter", {"span": record.name, "status": record.status})
    except Exception:  # pragma: no cover
        pass


# ============================================================================
# Metrics
# ============================================================================


def record_counter(name: str, value: float = 1, labels: Optional[Mapping[str, Any]] = None) -> bool:
    """Increment a counter. Never raises. Returns whether it was recorded."""
    recorded, dropped = _record(name, value, "counter", labels)
    _note_dropped(dropped)
    return recorded


def record_histogram(name: str, value: float, labels: Optional[Mapping[str, Any]] = None) -> bool:
    """Observe a histogram value (e.g. a duration). Never raises."""
    recorded, dropped = _record(name, value, "histogram", labels)
    _note_dropped(dropped)
    return recorded


def record_gauge(name: str, value: float, labels: Optional[Mapping[str, Any]] = None) -> bool:
    """Set a gauge value. Never raises."""
    recorded, dropped = _record(name, value, "gauge", labels)
    _note_dropped(dropped)
    return recorded


def _record(name: str, value: float, kind: str, labels: Optional[Mapping[str, Any]]) -> Tuple[bool, List[str]]:
    try:
        return REGISTRY.record_metric(name, value, kind, labels)
    except Exception:  # pragma: no cover - telemetry must never break business
        return False, []


def _note_dropped(dropped: List[str]) -> None:
    if not dropped:
        return
    try:
        for key in dropped:
            REGISTRY.record_metric(MetricName.DROPPED_LABELS, 1, "counter", {"label": key})
    except Exception:  # pragma: no cover
        pass


# ============================================================================
# Decorator — wrap an existing operation with a span + metric
# ============================================================================


def observed(
    span_name: str,
    metric: str = "",
    error_metric: str = "",
    duration_metric: str = "",
    attributes: Optional[Mapping[str, Any]] = None,
    outcome_resolver: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    kind: str = "internal",
):
    """Wrap a callable so its execution is observable.

    Pure additive: the wrapped function's return value, exceptions and
    side-effects are untouched. Telemetry failures are swallowed.

    Works on sync and async callables.
    """

    base_attributes = dict(attributes or {})

    def _resolve_outcome(result: Any) -> Dict[str, Any]:
        if outcome_resolver is None:
            return {}
        try:
            resolved = outcome_resolver(result) or {}
            return dict(resolved)
        except Exception:
            return {}

    def decorator(fn):
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                started = time.perf_counter()
                with span(span_name, base_attributes, kind=kind) as sp:
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        _record_outcome(span_name, duration_metric, error_metric, started, sp, exc)
                        raise
                    sp.set_attributes(_resolve_outcome(result))
                    _record_success(metric, duration_metric, started)
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            with span(span_name, base_attributes, kind=kind) as sp:
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    _record_outcome(span_name, duration_metric, error_metric, started, sp, exc)
                    raise
                sp.set_attributes(_resolve_outcome(result))
                _record_success(metric, duration_metric, started)
                return result

        return wrapper

    return decorator


def _record_success(metric: str, duration_metric: str, started: float) -> None:
    try:
        if metric:
            record_counter(metric, 1)
        if duration_metric:
            record_histogram(duration_metric, round((time.perf_counter() - started) * 1000, 3))
    except Exception:  # pragma: no cover
        pass


def _record_outcome(span_name: str, duration_metric: str, error_metric: str, started: float, sp: SpanHandle, exc: Exception) -> None:
    try:
        sp.set_status("ERROR", type(exc).__name__)
        sp.set_attribute("error.type", type(exc).__name__)
        if error_metric:
            record_counter(error_metric, 1, {"error_type": type(exc).__name__})
        if duration_metric:
            record_histogram(duration_metric, round((time.perf_counter() - started) * 1000, 3))
    except Exception:  # pragma: no cover
        pass


# ============================================================================
# Initialisation
# ============================================================================

_initialised = False


def init_observability(enable_otel: bool = True) -> Dict[str, Any]:
    """Initialise telemetry providers. Idempotent and failure-tolerant.

    Local development works with no exporter: spans/metres are no-ops at the
    OTel layer while the in-process registry stays fully functional.
    """
    global _initialised
    info: Dict[str, Any] = {
        "otel_available": is_otel_available(),
        "otel_configured": False,
        "registry_enabled": REGISTRY.is_enabled(),
    }
    if _initialised:
        info["already_initialised"] = True
        return info

    if enable_otel and _OTEL_TRACE is not None:
        try:
            from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider  # type: ignore
            from opentelemetry.trace import get_tracer_provider, set_tracer_provider  # type: ignore

            current = get_tracer_provider()
            # Only install an SDK provider when nothing meaningful is set.
            if current.__class__.__name__ in ("ProxyTracerProvider", "_DefaultTracerProvider"):
                set_tracer_provider(SdkTracerProvider())
                info["otel_configured"] = True
            else:
                info["otel_configured"] = True
                info["otel_provider"] = current.__class__.__name__
        except Exception as exc:
            info["otel_error"] = type(exc).__name__

    _initialised = True
    try:
        logger.debug("[OBSERVABILITY] initialised %s", info)
    except Exception:
        pass
    return info


def is_initialised() -> bool:
    return _initialised


# ============================================================================
# Test / diagnostic helpers
# ============================================================================


def set_observability_enabled(enabled: bool) -> None:
    """Enable/disable telemetry capture process-wide (used by tests)."""
    REGISTRY.set_enabled(enabled)


def observability_enabled() -> bool:
    return REGISTRY.is_enabled()


def reset_observability() -> None:
    """Clear all recorded telemetry and re-enable capture."""
    REGISTRY.reset()


def observability_snapshot() -> Dict[str, Any]:
    """Full snapshot of recorded spans and metrics."""
    return REGISTRY.snapshot()


def recorded_span_names() -> List[str]:
    return [s.name for s in REGISTRY.spans()]


def metric_labels(name: str) -> List[Dict[str, str]]:
    """All label sets recorded for a metric name."""
    return [dict(m.labels) for m in REGISTRY.metrics_named(name)]
