from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.structured_logging import (
    WorkflowEvent, system_logger, api_logger,
)
from app.core import observability as obs
from app.api.observability_middleware import ObservabilityMiddleware
from app.api.analyze import AnalyzeRequest, AnalyzeResponse, AnalyzeService
from app.api.errors import (
    BusinessRuleException,
    ConflictException,
    DatabaseFailureException,
    GuardrailRejectionException,
    InternalServerException,
    InvalidStateException,
    NotFoundException,
    RequestIDMiddleware,
    ServiceUnavailableException,
    ValidationException,
    business_rule_handler,
    conflict_handler,
    database_failure_handler,
    generic_error_handler,
    guardrail_rejection_handler,
    internal_error_handler,
    invalid_state_handler,
    not_found_handler,
    request_validation_handler,
    service_unavailable_handler,
    validation_handler,
)
from fastapi.exceptions import RequestValidationError
from app.api.explain import ExplainRequest, ExplainResponse, ExplainService
from app.api.routes import all_routers

app = FastAPI(
    title="Razorpay CloseLoop",
    description=(
        "Automated financial exception resolution with safety guardrails.\n\n"
        "## Architecture\n\n"
        "This API exposes a deterministic financial reconciliation system "
        "with ML-assisted classification and LLM-powered explanations.\n\n"
        "### Safety Guarantees\n"
        "- **Phase 6 Hard Guardrails**: Financial exposure limits, conflict detection, "
        "novelty checks, and evidence safety are mandatory and cannot be bypassed.\n"
        "- **Phase 8 Verification**: Every resolution is verified before execution.\n"
        "- **Phase 9 Learning**: Feedback improves future predictions offline — "
        "it never modifies production thresholds directly.\n"
        "- **LLM is Explanation-Only**: The LLM explains financial context but "
        "never makes financial decisions, sets amounts, or bypasses guardrails.\n\n"
        "### Request Correlation\n"
        "Every response includes an `X-Request-ID` header for tracing. "
        "Clients can provide their own via the `X-Request-ID` request header.\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ──
    # Phase 14: initialise local-first telemetry (no exporter required).
    try:
        obs_info = obs.init_observability()
        system_logger.info(
            WorkflowEvent.STARTUP.value,
            "Observability initialised",
            otel_available=obs_info.get("otel_available", False),
        )
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"Observability init skipped: {type(e).__name__}")

    system_logger.info(WorkflowEvent.STARTUP.value, "CloseLoop starting up")

    # Check database connectivity
    try:
        from app.database.database import engine
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        system_logger.info(WorkflowEvent.HEALTH_CHECK.value, "Database connected")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"Database unavailable: {type(e).__name__}")

    # Check ML components
    try:
        from app.ml.classifier import ExceptionClassifier
        system_logger.info(WorkflowEvent.HEALTH_CHECK.value, "ML classifier available")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"ML classifier unavailable: {type(e).__name__}")

    # Check MLflow
    try:
        from app.services.mlflow_model_registry import MLflowModelRegistry
        registry = MLflowModelRegistry()
        models = registry.list_models()
        system_logger.info(WorkflowEvent.HEALTH_CHECK.value,
                         f"MLflow registry available: {len(models)} models")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"MLflow unavailable: {type(e).__name__}")

    # Check LLM
    try:
        from app.llm.config import LLMConfig
        config = LLMConfig.from_env()
        if config.enabled:
            system_logger.info(WorkflowEvent.HEALTH_CHECK.value,
                             f"LLM enabled: provider={config.provider}")
        else:
            system_logger.info(WorkflowEvent.HEALTH_CHECK.value,
                             "LLM disabled (deterministic fallback active)")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"LLM config unavailable: {type(e).__name__}")

    # Check LangGraph
    try:
        from app.agent.workflow import create_workflow
        wf = create_workflow()
        system_logger.info(WorkflowEvent.HEALTH_CHECK.value, "LangGraph workflow compiled")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"LangGraph unavailable: {type(e).__name__}")

    # Seed demo feedback records for Learning page
    try:
        from scripts.seed_feedback import seed_demo_feedback
        seed_demo_feedback()
        system_logger.info(WorkflowEvent.STARTUP.value, "Demo feedback records seeded")
    except Exception as e:
        system_logger.warning(WorkflowEvent.DEPENDENCY_UNAVAILABLE.value,
                            f"Feedback seeding skipped: {type(e).__name__}: {e}")

    system_logger.info(WorkflowEvent.STARTUP.value, "CloseLoop startup complete",
                     version="1.0.0")

    yield

    # ── Shutdown ──
    system_logger.info(WorkflowEvent.SHUTDOWN.value, "CloseLoop shutting down")

    # Flush telemetry (best-effort; never blocks shutdown).
    try:
        from opentelemetry.trace import get_tracer_provider  # type: ignore
        provider = get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


app.router.lifespan_context = lifespan

# ─────────────────────────────────────────────────────────────────────────────
# Middleware (order matters: outermost first)
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(RequestIDMiddleware)

# ─────────────────────────────────────────────────────────────────────────────
# Exception Handlers (most specific first)
# ─────────────────────────────────────────────────────────────────────────────
app.add_exception_handler(NotFoundException, not_found_handler)
app.add_exception_handler(ValidationException, validation_handler)
app.add_exception_handler(ConflictException, conflict_handler)
app.add_exception_handler(InvalidStateException, invalid_state_handler)
app.add_exception_handler(BusinessRuleException, business_rule_handler)
app.add_exception_handler(GuardrailRejectionException, guardrail_rejection_handler)
app.add_exception_handler(ServiceUnavailableException, service_unavailable_handler)
app.add_exception_handler(DatabaseFailureException, database_failure_handler)
app.add_exception_handler(InternalServerException, internal_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_handler)
app.add_exception_handler(Exception, generic_error_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton services
# ─────────────────────────────────────────────────────────────────────────────
_explain_service = ExplainService()
_analyze_service = AnalyzeService()

# ─────────────────────────────────────────────────────────────────────────────
# Register API Routers
# ─────────────────────────────────────────────────────────────────────────────
for router in all_routers:
    # Register concrete routes so the application exposes the same route table
    # across FastAPI versions with and without lazy included-router support.
    app.router.routes.extend(router.routes)


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description=(
        "Check system health and availability. "
        "Returns system status, version, and which phases are implemented."
    ),
)
def health():
    """Health check endpoint.

    Returns:
    - `status`: "ok" if system is healthy
    - `version`: API version
    - `phases`: List of implemented phase numbers

    This endpoint does NOT require authentication and is always available.
    It does NOT check database or service connectivity.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "phases": ["1-12", "13.9", "14"],
    }


@app.get(
    "/ready",
    tags=["System"],
    summary="Readiness check",
    description=(
        "Report whether the service can currently operate its required "
        "dependencies. Returns 200 when ready, 503 when a required dependency "
        "is unavailable. Optional dependencies are reported but never mark the "
        "service unready. No secrets or credentials are exposed."
    ),
)
def ready():
    """Readiness probe (Phase 14).

    Required: database connectivity.
    Optional (reported only): ML classifier, LLM config, observability.
    """
    from fastapi.responses import JSONResponse

    checks = {
        "database": False,
        "ml_classifier": False,
        "llm": False,
        "observability": False,
    }

    # ── Required: database ──
    try:
        from sqlalchemy import text as _text
        from app.database.database import engine
        with engine.connect() as conn:
            conn.execute(_text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    # ── Optional dependencies (never fail readiness) ──
    try:
        from app.ml.classifier import ExceptionClassifier  # noqa: F401
        checks["ml_classifier"] = True
    except Exception:
        checks["ml_classifier"] = False
    try:
        from app.llm.config import LLMConfig
        checks["llm"] = bool(LLMConfig.from_env().enabled)
    except Exception:
        checks["llm"] = False
    try:
        checks["observability"] = obs.is_initialised() or obs.is_otel_available()
    except Exception:
        checks["observability"] = False

    required_ok = checks["database"]
    body = {
        "status": "ready" if required_ok else "not_ready",
        "version": "1.0.0",
        "checks": checks,
        "required": ["database"],
    }
    return JSONResponse(status_code=200 if required_ok else 503, content=body)


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Intelligence"],
    summary="Explain a financial exception",
    description=(
        "Provide a human-readable explanation of an existing financial exception. "
        "Accepts an exception ID and returns a structured explanation including "
        "summary, evidence summary, uncertainty, and LLM provider status."
    ),
)
async def explain_exception(request: ExplainRequest):
    """Provide a human-readable explanation of a financial exception.

    **Flow**:
    1. Load exception by ID
    2. Retrieve evidence from Phase 3
    3. Build structured explanation context
    4. Send to LLM provider (or use deterministic fallback)
    5. Validate LLM output
    6. Return explanation

    **LLM behavior**:
    - When available: AI-generated natural language
    - When unavailable: Deterministic template from structured data
    - The LLM never invents financial facts or makes decisions

    **Input validation**: Only the exception ID is accepted as input.
    Users cannot submit arbitrary financial truth values.
    """
    return await _explain_service.explain(request)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Intelligence"],
    summary="Full AI-assisted analysis",
    description=(
        "Provide a complete AI-assisted investigation summary. "
        "Combines reconciliation, evidence, classification, candidates, "
        "guardrails, and LLM explanation into a single structured response."
    ),
)
async def analyze_exception(request: AnalyzeRequest):
    """Provide a complete AI-assisted investigation summary.

    **Data sources combined**:
    - Phase 2: Deterministic reconciliation
    - Phase 3: Financial evidence
    - Phase 4: ML classification + similarity
    - Phase 5: Resolution candidates
    - Phase 6: Guardrail decisions
    - Phase 12: LLM explanation

    **Safety**: The LLM is used for natural language explanation only.
    Financial calculations, risk assessments, and guardrail decisions
    are always deterministic and independent of the LLM.
    """
    return await _analyze_service.analyze(request)
