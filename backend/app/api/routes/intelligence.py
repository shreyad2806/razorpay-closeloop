"""
Intelligence Routes for Razorpay CloseLoop.

Thin API layer for analysis, explanation, similarity, and evidence.
Uses typed Pydantic schemas for all requests and responses.
Delegates to IntelligenceService which connects to real Phase 3-6, 12 services.

Endpoints:
- POST /exceptions/{id}/analyze  — Full AI-assisted analysis
- GET  /exceptions/{id}/explain  — Human-readable explanation
- GET  /exceptions/{id}/similar  — Similar historical cases
- GET  /exceptions/{id}/evidence — Financial evidence graph
"""

from fastapi import APIRouter, Query

from app.api.dependencies import get_intelligence_service
from app.api.errors import NotFoundException
from app.api.schemas import (
    AnalysisDepth,
    ApiResponse,
    ExplanationDepth,
)
from app.api.errors import ErrorResponse

router = APIRouter(prefix="/exceptions", tags=["Intelligence"])

_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Exception not found"}}


@router.post(
    "/{exception_id}/analyze",
    summary="Analyze an exception",
    description=(
        "Provide a complete AI-assisted investigation summary. "
        "Combines Phase 2 reconciliation, Phase 3 evidence, Phase 4 classification "
        "and similarity, Phase 5 resolution candidates, Phase 6 guardrails, "
        "and Phase 12 LLM explanation into a single structured response."
    ),
    responses=_ERRORS_404,
)
async def analyze_exception(exception_id: str):
    """Full analysis of a financial exception.

    Returns a comprehensive analysis including:
    - Financial discrepancy (expected vs actual)
    - Evidence summary with coverage and conflicts
    - Classification type and confidence
    - Similar historical cases
    - Resolution candidates with confidence scores
    - Guardrail decision and risk assessment
    - AI-generated explanation (or deterministic fallback)

    The LLM is used for natural language explanation only.
    Financial calculations are always deterministic.
    """
    svc = get_intelligence_service()
    result = await svc.analyze(exception_id)
    if isinstance(result, dict) and result.get("error") and "not found" in result.get("error", "").lower():
        raise NotFoundException("Exception", exception_id)
    return result


@router.get(
    "/{exception_id}/explain",
    summary="Explain an exception",
    description=(
        "Provide a human-readable explanation of a financial exception "
        "using the Phase 12 LLM service with deterministic fallback. "
        "The LLM explains evidence and context — it does NOT make financial decisions."
    ),
    responses=_ERRORS_404,
)
async def explain_exception(
    exception_id: str,
    depth: ExplanationDepth = Query(
        ExplanationDepth.STANDARD,
        description="Explanation depth: brief, standard, or detailed",
    ),
):
    """Get a human-readable explanation of a financial exception.

    **Explanation depths**:
    - `brief`: One-paragraph summary
    - `standard`: Summary with key evidence and reasoning
    - `detailed`: Full analysis with all evidence, conflicts, and uncertainty

    **LLM behavior**:
    - When LLM is available: AI-generated natural language explanation
    - When LLM is unavailable: Deterministic template from structured data
    - The LLM never invents financial facts or makes decisions
    """
    svc = get_intelligence_service()
    result = await svc.explain(exception_id, depth=depth.value)
    if isinstance(result, dict) and result.get("error") and "not found" in result.get("error", "").lower():
        raise NotFoundException("Exception", exception_id)
    return result


@router.get(
    "/{exception_id}/similar",
    summary="Find similar historical cases",
    description=(
        "Retrieve similar historical cases using Phase 4 similarity patterns. "
        "Returns cases with the highest similarity scores, along with how they "
        "were previously resolved. Top-k is bounded (1-20)."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_similar_cases(
    exception_id: str,
    limit: int = Query(
        5,
        ge=1,
        le=20,
        description="Maximum number of similar cases to return",
    ),
):
    """Find similar historical cases.

    Returns similar cases with:
    - Case ID and exception type
    - Similarity score (0.0 to 1.0)
    - How the case was previously resolved
    - Risk category of the historical case

    **Similarity calculation**:
    - Same exception type: base similarity 0.7
    - Similar financial difference: +0.3 × normalized ratio
    - Confidence: HIGH (top score > 0.8), MEDIUM (> 0.5), LOW (≤ 0.5)
    """
    svc = get_intelligence_service()
    result = svc.get_similar(exception_id, top_k=limit)
    if "error" in result and result.get("error"):
        if "not found" in result.get("error", "").lower():
            raise NotFoundException("Exception", exception_id)
    return ApiResponse(success=True, data=result)


@router.get(
    "/{exception_id}/evidence",
    summary="Get financial evidence",
    description=(
        "Retrieve structured financial evidence from the Phase 3 evidence layer. "
        "Returns all related financial records (payments, settlements, refunds, fees, "
        "adjustments) with coverage analysis and conflict detection."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_evidence(exception_id: str):
    """Get the evidence graph for a financial exception.

    Returns structured evidence including:
    - **Payments**: Original payment records
    - **Settlements**: Settlement records linked to payments
    - **Refunds**: Refund records
    - **Fees**: Fee/charge records
    - **Adjustments**: Adjustment records
    - **Coverage**: How well evidence explains the discrepancy
    - **Conflicts**: Evidence records that contradict each other
    - **Missing evidence**: Expected records not found

    Evidence is retrieved from the FinancialDataAdapter (Phase 3)
    — the LLM does NOT access the database directly.
    """
    svc = get_intelligence_service()
    result = svc.get_evidence(exception_id)
    if "error" in result and result.get("error"):
        if "not found" in result.get("error", "").lower():
            raise NotFoundException("Exception", exception_id)
    return ApiResponse(success=True, data=result)
