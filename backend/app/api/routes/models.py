"""
Model Management Routes for Razorpay CloseLoop.

Thin API layer for MLflow model registry and lineage.
Uses typed Pydantic schemas for all requests and responses.

Endpoints:
- GET /models             — List all registered models
- GET /models/{id}        — Get model details
- GET /models/{id}/lineage — Get full model lineage
"""

from fastapi import APIRouter, HTTPException

from app.api.dependencies import get_model_service
from app.api.errors import NotFoundException, ServiceUnavailableException
from app.api.schemas import ApiResponse
from app.api.errors import ErrorResponse

router = APIRouter(prefix="/models", tags=["Models"])

_ERRORS_404 = {404: {"model": ErrorResponse, "description": "Model not found"}}


@router.get(
    "",
    summary="List registered models",
    description=(
        "Retrieve all registered models from the MLflow model registry. "
        "Returns model name, version, status, training dataset version, "
        "metrics, and production/candidate status."
    ),
    response_model=ApiResponse,
)
async def list_models():
    """List all registered models.

    Returns models from the MLflow registry with:

    | Field | Description |
    |-------|-------------|
    | model_id | Unique model identifier |
    | model_name | Model name (e.g., exception_classifier) |
    | model_version | Semantic version |
    | status | CANDIDATE, VALIDATION, PRODUCTION, or ARCHIVED |
    | mlflow_run_id | Link to MLflow training run |
    | dataset_version | Training dataset version |
    | feature_version | Feature schema version |
    | precision | Model precision metric |
    | f1 | Model F1 score |
    | promoted_at | When the model was promoted to production |

    **Model lifecycle**: Models progress through CANDIDATE → VALIDATION → PRODUCTION.
    Old models are ARCHIVED for rollback capability.
    """
    try:
        svc = get_model_service()
        models = svc.list_models()
        return ApiResponse(success=True, data=models, count=len(models))
    except Exception as e:
        raise ServiceUnavailableException(service="MLflow registry", reason=str(e))


@router.get(
    "/{model_id}",
    summary="Get model details",
    description=(
        "Retrieve detailed information about a specific model version "
        "from the MLflow model registry."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_model(model_id: str):
    """Get details of a specific model.

    Returns complete model information including:
    - Model name and version
    - Current lifecycle status
    - MLflow run ID for traceability
    - Dataset and feature versions
    - Evaluation metrics (precision, recall, F1)
    - Promotion timestamp

    Returns 404 if the model does not exist.
    """
    try:
        svc = get_model_service()
        model = svc.get_model(model_id)
        if model is None:
            raise NotFoundException("Model", model_id)
        return ApiResponse(success=True, data=model)
    except NotFoundException:
        raise
    except Exception as e:
        raise ServiceUnavailableException(service="MLflow registry", reason=str(e))


@router.get(
    "/{model_id}/lineage",
    summary="Get model lineage",
    description=(
        "Retrieve the full lineage chain of a model: training run, dataset, "
        "features, configuration, and evaluation metrics. This enables "
        "end-to-end traceability from prediction back to training."
    ),
    response_model=ApiResponse,
    responses=_ERRORS_404,
)
async def get_model_lineage(model_id: str):
    """Get the full lineage of a model.

    Returns the complete traceability chain:

    ```
    Model Version
      → MLflow Run ID
        → Training Configuration
          → Dataset Version
            → Feature Schema Version
              → Evaluation Metrics
                → Artifacts
    ```

    **Audit capability**: Given any model prediction, this endpoint
    answers: "Which exact model produced this result, and how was it trained?"

    **Historical immutability**: Old results always reference their
    original model version, even after a new model is promoted.
    """
    try:
        svc = get_model_service()
        lineage = svc.get_model_lineage(model_id)
        if not lineage or lineage.get("model_id") != model_id:
            raise NotFoundException("Model", model_id)
        return ApiResponse(success=True, data=lineage)
    except NotFoundException:
        raise
    except Exception as e:
        raise ServiceUnavailableException(service="MLflow registry", reason=str(e))
