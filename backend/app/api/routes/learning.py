"""
Learning Routes for Razorpay CloseLoop.

Thin API layer for feedback collection and learning metrics.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- POST /feedback            — Record human feedback on a workflow
- GET  /feedback/{id}       — Get a specific feedback record
- GET  /learning/metrics    — Get Phase 9 learning metrics
- GET  /learning/datasets   — Get learning dataset information
"""

from fastapi import APIRouter, HTTPException

from app.api.dependencies import get_feedback_service, get_learning_service
from app.api.errors import NotFoundException, ValidationException
from app.api.schemas import (
    ApiResponse,
    FeedbackRequest,
    FeedbackType,
)
from app.api.errors import ErrorResponse
from app.schemas.feedback import (
    CorrectionDetail,
    EscalationDetail,
    RejectionDetail,
)

router = APIRouter(tags=["Learning"])

_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Resource not found"}}
_ERRORS_422 = {422: {"model": ErrorResponse, "description": "Validation error"}}


@router.post(
    "/feedback",
    summary="Record human feedback",
    description=(
        "Record feedback on a workflow outcome. Supports four feedback types: "
        "APPROVE, REJECT, CORRECT, ESCALATE. Each type requires specific fields. "
        "Feedback is recorded through the Phase 9 feedback service for learning."
    ),
    status_code=201,
    response_model=ApiResponse,
    responses=_ERRORS_422,
)
async def record_feedback(request: FeedbackRequest):
    """Record human feedback on a workflow outcome.

    **Feedback types and required fields**:

    | Type | Required Fields | Purpose |
    |------|----------------|---------|
    | APPROVE | workflow_id | Confirm the resolution was correct |
    | REJECT | workflow_id, rejection_reason | Mark the resolution as incorrect |
    | CORRECT | workflow_id, original_resolution, corrected_resolution, correction_reason | Fix an incorrect resolution |
    | ESCALATE | workflow_id, escalation_reason | Route to higher-level human review |

    **Phase 9 integration**: Feedback is linked to the workflow for:
    - Reward calculation (CORRECT_AUTO_RESOLUTION, HUMAN_CONFIRMED, etc.)
    - Learning dataset generation
    - Model improvement
    - Policy evaluation

    **Safety**: Feedback does NOT directly modify production policies or thresholds.
    It feeds the learning pipeline for offline evaluation.
    """
    svc = get_feedback_service()

    # Build feedback-specific details based on type
    correction = None
    rejection = None
    escalation = None

    if request.feedback_type == FeedbackType.CORRECT:
        if not request.original_resolution or not request.corrected_resolution or not request.correction_reason:
            raise ValidationException(
                "CORRECT feedback requires original_resolution, corrected_resolution, and correction_reason"
            )
        correction = CorrectionDetail(
            original_resolution=request.original_resolution,
            corrected_resolution=request.corrected_resolution.value if request.corrected_resolution else "",
            correction_reason=request.correction_reason,
        )

    elif request.feedback_type == FeedbackType.REJECT:
        if not request.rejection_reason:
            raise ValidationException("REJECT feedback requires rejection_reason")
        rejection = RejectionDetail(
            rejection_reason=request.rejection_reason,
            suggested_alternative=None,
        )

    elif request.feedback_type == FeedbackType.ESCALATE:
        if not request.escalation_reason:
            raise ValidationException("ESCALATE feedback requires escalation_reason")
        escalation = EscalationDetail(
            escalation_reason=request.escalation_reason,
            escalation_target=None,
        )

    # For APPROVE, no additional details needed

    # Record feedback using the real service
    try:
        result = svc.record_feedback(
            workflow_id=request.workflow_id,
            exception_id=request.exception_id or "",
            feedback_type=request.feedback_type,
            reviewer=request.reviewer or "system",
            system_prediction="UNKNOWN",  # Would come from workflow context
            case_id=None,
            candidate_id=request.candidate_id,
            correction=correction,
            rejection=rejection,
            escalation=escalation,
            reason=None,
        )
        return ApiResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get(
    "/feedback/{feedback_id}",
    summary="Get feedback details",
    description=(
        "Retrieve details of a specific feedback record including "
        "feedback type, reviewer, outcome, and timestamp."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_feedback(feedback_id: str):
    """Get a specific feedback record.

    Returns the complete feedback record with:
    - Feedback type (APPROVE, REJECT, CORRECT, ESCALATE)
    - Reviewer identity
    - Workflow and exception IDs
    - Correction/rejection/escalation details
    - Timestamp
    """
    svc = get_feedback_service()
    fb = svc.get_feedback(feedback_id)
    if fb is None:
        raise NotFoundException("Feedback", feedback_id)
    return ApiResponse(success=True, data=fb)


@router.get(
    "/learning/metrics",
    summary="Get learning metrics",
    description=(
        "Retrieve Phase 9 learning metrics including automation rate, precision, "
        "false automation rate, human review rate, reward, verification failure rate, "
        "high-value error rate, and financial impact."
    ),
    response_model=ApiResponse,
)
async def get_learning_metrics():
    """Get learning system metrics.

    Returns comprehensive Phase 9 metrics:
    - **Automation Rate**: Automated successful resolutions / eligible exceptions
    - **Precision**: Correctness of automated decisions
    - **False Automation Rate**: Cases automated incorrectly (safety-critical)
    - **Human Review Rate**: Cases sent to human review
    - **Reward**: Average reward across all outcomes
    - **Verification Failure Rate**: Cases that failed verification
    - **Financial Impact**: Total financial impact of decisions

    These metrics measure whether the system is improving without
    sacrificing safety thresholds.
    """
    svc = get_learning_service()
    metrics = svc.get_metrics()
    return ApiResponse(success=True, data=metrics)


@router.get(
    "/learning/datasets",
    summary="Get dataset information",
    description=(
        "Retrieve information about learning datasets including "
        "total examples, training/validation/test split sizes, "
        "dataset version, and feature version."
    ),
    response_model=ApiResponse,
)
async def get_learning_datasets():
    """Get learning dataset information.

    Returns metadata about the learning dataset:
    - Total examples available
    - Training/validation/test split sizes
    - Dataset version for reproducibility
    - Feature schema version
    - Last update timestamp

    This does NOT return raw financial data — only metadata
    and statistics for monitoring the learning pipeline.
    """
    svc = get_learning_service()
    info = svc.get_dataset_info()
    return ApiResponse(success=True, data=info)
