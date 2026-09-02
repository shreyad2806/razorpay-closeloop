"""
Batch Management Routes for Razorpay CloseLoop.

Thin API layer over existing batch services.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- POST /batches          — Create a new batch (synthetic or uploaded)
- GET  /batches          — List all batches with pagination
- GET  /batches/{id}     — Get batch details
- POST /batches/{id}/run — Run batch reconciliation
- GET  /batches/{id}/summary — Get batch processing summary
"""

from fastapi import APIRouter, Query

from app.api.dependencies import get_batch_service
from app.api.errors import NotFoundException, ValidationException
from app.api.schemas import (
    ApiResponse,
    BatchCreateRequest,
    BatchResponse,
    BatchStatus,
    BatchSummaryResponse,
)
from app.api.errors import ErrorResponse

router = APIRouter(prefix="/batches", tags=["Batches"])

# Shared error responses for this router
_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Batch not found"}}
_ERRORS_422 = {422: {"model": ErrorResponse, "description": "Validation error"}}


@router.get(
    "",
    summary="List all batches",
    description=(
        "Retrieve a paginated list of all batch processing runs. "
        "Returns batch metadata including status, exception count, and timestamps."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_422,
)
async def list_batches(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of batches to return"),
    offset: int = Query(0, ge=0, description="Pagination offset for large result sets"),
):
    """List all batch processing runs.

    Supports pagination via `limit` and `offset` query parameters.
    Returns a list of batch summaries ordered by creation time (newest first).
    """
    svc = get_batch_service()
    batches = svc.list_batches(limit=limit, offset=offset)
    return ApiResponse(success=True, data=batches, count=len(batches))


@router.post(
    "",
    summary="Create a new batch",
    description=(
        "Create a new batch for processing. Accepts either a name for synthetic data generation "
        "or a pre-built payload with financial records (payments, settlements, cases). "
        "The batch is created in CREATED status and can be run via POST /batches/{id}/run."
    ),
    status_code=201,
    response_model=ApiResponse,
    responses=_ERRORS_422,
)
async def create_batch(request: BatchCreateRequest):
    """Upload and create a new batch.

    **Synthetic generation**: Provide `name` and optionally `num_merchants`, `num_cases`.
    The system generates financial data automatically.

    **Pre-built payload**: Provide `payload` with `payments` and `cases` lists.
    The system validates the structure before creation.

    Returns the created batch with its assigned `batch_id`.
    """
    svc = get_batch_service()
    result = svc.create_batch(request.model_dump())
    if result.get("status") == "VALIDATION_FAILED":
        raise ValidationException(
            "Batch validation failed",
            details={"errors": result.get("errors", [])},
        )
    return ApiResponse(success=True, data=result)


@router.get(
    "/{batch_id}",
    summary="Get batch details",
    description=(
        "Retrieve detailed information about a specific batch, "
        "including its current status, metadata, and processing state."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_batch(batch_id: str):
    """Get details of a specific batch.

    Returns the batch metadata including status (CREATED, RUNNING, COMPLETED, FAILED),
    creation timestamp, and processing information.
    """
    svc = get_batch_service()
    batch = svc.get_batch(batch_id)
    if batch is None:
        raise NotFoundException("Batch", batch_id)
    return ApiResponse(success=True, data=batch)


@router.post(
    "/{batch_id}/run",
    summary="Run batch reconciliation",
    description=(
        "Start deterministic reconciliation processing for a batch. "
        "The batch must be in CREATED or COMPLETED status (re-runnable). "
        "Processing uses the existing Phase 2 reconciliation engine synchronously."
    ),
    response_model=ApiResponse,
    responses={
        **_ERRORS_404,
        409: {"model": ErrorResponse, "description": "Batch already running"},
    },
)
async def run_batch(batch_id: str):
    """Start processing a batch.

    Loads the batch data, runs it through the Phase 2 reconciliation engine,
    and returns a summary of processing results including match rate and exception count.

    Processing is synchronous — the response includes the complete summary
    once processing finishes.
    """
    svc = get_batch_service()
    result = svc.run_batch(batch_id)
    if result is None:
        raise NotFoundException("Batch", batch_id)
    return ApiResponse(success=True, data=result)


@router.get(
    "/{batch_id}/summary",
    summary="Get batch processing summary",
    description=(
        "Get a summary of batch processing results including total records, "
        "matched records, exceptions, match rate, exception rate, and throughput."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_batch_summary(batch_id: str):
    """Get summary of batch results.

    Returns a structured summary with:
    - Total records processed
    - Matched vs exception counts
    - Match rate and exception rate percentages
    - Processing time and throughput
    - Status of the batch processing run
    """
    svc = get_batch_service()
    summary = svc.get_summary(batch_id)
    if summary is None:
        raise NotFoundException("Batch", batch_id)
    return ApiResponse(success=True, data=summary)
