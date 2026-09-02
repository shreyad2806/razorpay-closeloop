"""
Metrics Routes for Razorpay CloseLoop.

Thin API layer for system-wide and safety metrics.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- GET /metrics                — Overall system metrics
- GET /metrics/safety         — Safety-critical metrics
- GET /metrics/throughput     — Processing throughput metrics
- GET /metrics/batches/{id}   — Metrics for a specific batch
"""

from fastapi import APIRouter, HTTPException

from app.api.dependencies import get_metrics_service
from app.api.errors import NotFoundException
from app.api.schemas import ApiResponse
from app.api.errors import ErrorResponse

router = APIRouter(prefix="/metrics", tags=["Metrics"])

_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Batch not found"}}


@router.get(
    "",
    summary="Get overall system metrics",
    description=(
        "Retrieve system-wide metrics aggregated from all batch processing runs. "
        "Includes total records, matched records, exceptions, match rate, exception rate, "
        "automation rate, human review rate, unresolved rate, and financial impact."
    ),
    response_model=ApiResponse,
)
async def get_metrics():
    """Get overall system metrics.

    Returns comprehensive metrics aggregated across all processed batches:

    | Metric | Description |
    |--------|-------------|
    | total_records | Total reconciliation results processed |
    | matched_records | Successfully matched records |
    | exceptions | Records with financial discrepancies |
    | match_rate | matched / total (0.0 to 1.0) |
    | exception_rate | exceptions / total (0.0 to 1.0) |
    | automation_rate | auto-resolved / total (0.0 to 1.0) |
    | human_review | Cases sent to human review |
    | unresolved | Cases that could not be resolved |
    | verification_passed | Cases passing verification |
    | verification_failed | Cases failing verification |
    | financial_impact_paise | Total financial impact in paise |
    """
    try:
        svc = get_metrics_service()
        metrics = svc.get_metrics()
        return ApiResponse(success=True, data=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.get(
    "/batches/{batch_id}",
    summary="Get batch-specific metrics",
    description=(
        "Retrieve metrics for a specific batch processing run including "
        "processing time, match rate, exception rate, and throughput."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_batch_metrics(batch_id: str):
    """Get metrics for a specific batch.

    Returns batch-level metrics including:
    - Total records processed in this batch
    - Match rate and exception rate for the batch
    - Processing time and throughput
    - Status of the batch processing run

    Returns 404 if the batch does not exist.
    """
    try:
        svc = get_metrics_service()
        metrics = svc.get_batch_metrics(batch_id)
        if metrics is None:
            raise NotFoundException("Batch", batch_id)
        return ApiResponse(success=True, data=metrics)
    except NotFoundException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve batch metrics: {str(e)}")


@router.get(
    "/safety",
    summary="Get safety metrics",
    description=(
        "Retrieve safety-critical metrics including guardrail effectiveness, "
        "high-value error counts, conflict detection, and verification failures. "
        "These metrics are used to ensure the system remains safe as it automates."
    ),
    response_model=ApiResponse,
)
async def get_safety_metrics():
    """Get safety-critical metrics.

    Returns safety metrics for monitoring system guardrails:

    | Metric | Description |
    |--------|-------------|
    | guardrail_blocks | Times guardrails prevented an action |
    | high_value_blocks | High-value transactions blocked |
    | conflict_blocks | Conflicts detected and blocked |
    | novelty_blocks | Novel patterns escalated |
    | verification_failures | Cases failing verification |
    | guardrail_pass_rate | auto / total decisions (0.0 to 1.0) |

    **Safety rule**: An increase in automation rate WITHOUT a corresponding
    improvement in these metrics is NOT success.
    """
    try:
        svc = get_metrics_service()
        metrics = svc.get_safety_metrics()
        return ApiResponse(success=True, data=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve safety metrics: {str(e)}")


@router.get(
    "/throughput",
    summary="Get throughput metrics",
    description=(
        "Retrieve processing throughput metrics including total records processed, "
        "average processing time, records per second, and number of batches processed."
    ),
    response_model=ApiResponse,
)
async def get_throughput_metrics():
    """Get processing throughput metrics.

    Returns throughput metrics for monitoring system performance:

    | Metric | Description |
    |--------|-------------|
    | total_records_processed | Total records across all batches |
    | total_processing_time_ms | Total processing time in milliseconds |
    | avg_processing_time_ms | Average processing time per batch |
    | records_per_second | Processing speed (records/sec) |
    | batches_processed | Number of batches processed |

    These metrics help identify performance bottlenecks
    and capacity planning needs.
    """
    try:
        svc = get_metrics_service()
        metrics = svc.get_throughput_metrics()
        return ApiResponse(success=True, data=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve throughput metrics: {str(e)}")
