"""
API Schemas for Razorpay CloseLoop Phase 13.2.

Pydantic request/response schemas for all REST API endpoints.

Rules:
- Never expose raw database models as API contracts
- Validate IDs, enums, amounts, pagination, limits
- Do not allow invalid financial amounts
- Do not allow arbitrary fields
- Pydantic validation is API validation, NOT a replacement for
  Phase 6 guardrails, business validation, or authorization
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.enums import (
    ExceptionType,
    ResolutionType,
    RiskCategory,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Common / Shared
# ═══════════════════════════════════════════════════════════════════════════════


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(..., description="Whether the request succeeded")
    data: Any = Field(default=None, description="Response payload")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    count: Optional[int] = Field(default=None, description="Total count for list endpoints")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    phases: List[str] = Field(default_factory=lambda: ["1-12"])


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class BatchStatus(str, Enum):
    """Status of a batch processing run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class BatchCreateRequest(BaseModel):
    """Request body for POST /batches.

    Use ``payload=None`` (or omit it) to generate synthetic data.
    Provide a ``payload`` dict with ``payments`` and ``cases`` lists
    to upload pre-built financial records.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "demo-batch",
                    "description": "Synthetic reconciliation test batch",
                    "source": "synthetic",
                    "num_merchants": 5,
                    "num_cases": 20,
                }
            ]
        }
    }

    name: str = Field(..., min_length=1, max_length=200, description="Batch name")
    description: Optional[str] = Field(default=None, max_length=1000)
    source: Optional[str] = Field(default=None, description="Data source identifier")
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional pre-built payload with financial records",
    )
    num_merchants: int = Field(default=5, ge=1, le=100, description="Merchants for synthetic generation")
    num_cases: int = Field(default=20, ge=1, le=500, description="Cases for synthetic generation")


class BatchResponse(BaseModel):
    """Response for a batch operation."""

    batch_id: str = Field(..., description="Unique batch identifier")
    name: str = Field(default="", description="Batch name")
    status: BatchStatus = Field(default=BatchStatus.CREATED)
    created_at: Optional[datetime] = None
    exception_count: int = Field(default=0, description="Number of exceptions in batch")
    success_count: int = Field(default=0, description="Successfully processed count")
    failure_count: int = Field(default=0, description="Failed processing count")


class BatchSummaryResponse(BaseModel):
    """Summary of batch processing results."""

    batch_id: str
    status: BatchStatus
    total_exceptions: int = 0
    resolved: int = 0
    unresolved: int = 0
    escalated: int = 0
    auto_resolved: int = 0
    human_review: int = 0
    verification_passed: int = 0
    verification_failed: int = 0
    financial_impact_paise: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Exception Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class ExceptionStatus(str, Enum):
    """Status of an exception."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    UNRESOLVED = "UNRESOLVED"


class ExceptionListItem(BaseModel):
    """Summary view of an exception for list endpoints."""

    exception_id: str = Field(..., description="Unique exception identifier")
    case_id: str = Field(..., description="Reference to the case")
    merchant_id: str = Field(..., description="Merchant identifier")
    payment_id: str = Field(..., description="Payment identifier")
    exception_type: ExceptionType
    expected_amount_paise: int = Field(..., ge=0)
    actual_amount_paise: int = Field(..., ge=0)
    difference_paise: int = Field(description="actual - expected")
    risk_category: RiskCategory
    status: ExceptionStatus = Field(default=ExceptionStatus.PENDING)
    created_at: Optional[datetime] = None


class ExceptionDetail(BaseModel):
    """Full exception detail view."""

    exception_id: str
    case_id: str
    merchant_id: str
    payment_id: str
    exception_type: ExceptionType
    expected_amount_paise: int = Field(..., ge=0)
    actual_amount_paise: int = Field(..., ge=0)
    difference_paise: int
    risk_category: RiskCategory
    status: ExceptionStatus = Field(default=ExceptionStatus.PENDING)
    classification_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="ML classification confidence"
    )
    guardrail_decision: Optional[str] = None
    similar_case_count: int = Field(default=0)
    evidence_record_count: int = Field(default=0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ResolveRequest(BaseModel):
    """Request body for POST /exceptions/{id}/resolve."""

    resolution_type: ResolutionType = Field(..., description="Type of resolution")
    adjustment_paise: int = Field(
        default=0, description="Financial adjustment in paise (negative=debit, positive=credit)"
    )
    reason: Optional[str] = Field(default=None, max_length=2000, description="Resolution reason")
    candidate_id: Optional[str] = Field(default=None, description="Reference to resolution candidate")

    @field_validator("adjustment_paise")
    @classmethod
    def validate_adjustment(cls, v: int) -> int:
        if abs(v) > 10_000_000:  # 10 lakh paise = ₹100,000
            raise ValueError("Adjustment exceeds maximum allowed (₹100,000)")
        return v


class ResolveResponse(BaseModel):
    """Response for resolve operation."""

    exception_id: str
    resolution_type: ResolutionType
    status: str = Field(description="Resolution status")
    adjustment_paise: int = 0
    guardrail_decision: Optional[str] = None
    execution_result: Optional[str] = None
    verification_result: Optional[str] = None


class ApproveRequest(BaseModel):
    """Request body for POST /exceptions/{id}/approve."""

    approved_by: str = Field(..., min_length=1, max_length=200, description="Reviewer identifier")
    comments: Optional[str] = Field(default=None, max_length=2000)


class RejectRequest(BaseModel):
    """Request body for POST /exceptions/{id}/reject."""

    rejected_by: str = Field(..., min_length=1, max_length=200, description="Reviewer identifier")
    reason: str = Field(..., min_length=1, max_length=2000, description="Rejection reason")


class EscalateRequest(BaseModel):
    """Request body for POST /exceptions/{id}/escalate."""

    reason: str = Field(..., min_length=1, max_length=2000, description="Escalation reason")
    escalated_by: Optional[str] = Field(default=None, max_length=200)
    priority: Optional[str] = Field(default="NORMAL", description="Escalation priority")


# ═══════════════════════════════════════════════════════════════════════════════
# Intelligence Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class AnalysisDepth(str, Enum):
    """Depth of analysis."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class EvidenceCoverage(str, Enum):
    """Evidence coverage status."""

    FULLY_EXPLAINED = "FULLY_EXPLAINED"
    PARTIALLY_EXPLAINED = "PARTIALLY_EXPLAINED"
    UNEXPLAINED = "UNEXPLAINED"
    CONFLICTING = "CONFLICTING"


class AnalyzeRequest(BaseModel):
    """Request body for POST /exceptions/{id}/analyze."""

    include_evidence: bool = Field(default=True)
    include_candidates: bool = Field(default=True)
    include_similar_cases: bool = Field(default=True)
    analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.STANDARD)


class FinancialDiscrepancy(BaseModel):
    """Financial discrepancy information."""

    expected_amount_paise: int = Field(..., ge=0)
    actual_amount_paise: int = Field(..., ge=0)
    difference_paise: int
    exception_type: ExceptionType


class EvidenceSummary(BaseModel):
    """Evidence summary for analysis."""

    record_count: int = Field(default=0, ge=0)
    coverage: EvidenceCoverage = Field(default=EvidenceCoverage.UNEXPLAINED)
    explained_amount_paise: int = Field(default=0, ge=0)
    remaining_amount_paise: int = Field(default=0, ge=0)
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class ResolutionCandidateSummary(BaseModel):
    """Summary of a resolution candidate."""

    resolution_type: ResolutionType
    adjustment_paise: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(description="Where the candidate came from")
    description: Optional[str] = None


class GuardrailSummary(BaseModel):
    """Guardrail decision summary."""

    decision: str = Field(description="AUTO, HUMAN_REVIEW, or UNRESOLVED")
    confidence: float = Field(ge=0.0, le=1.0)
    risk_category: RiskCategory
    reasons: List[str] = Field(default_factory=list)
    exposure_paise: int = Field(default=0, ge=0)


class AnalysisResult(BaseModel):
    """Complete analysis result."""

    exception_id: str
    case_id: Optional[str] = None
    financial_discrepancy: FinancialDiscrepancy
    evidence: EvidenceSummary
    classification_type: Optional[ExceptionType] = None
    classification_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similar_case_count: int = Field(default=0, ge=0)
    similar_cases_summary: Optional[str] = None
    candidates: List[ResolutionCandidateSummary] = Field(default_factory=list)
    guardrail: GuardrailSummary
    ai_explanation: Optional[str] = None
    ai_uncertainty: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    fallback_used: bool = Field(default=True)


class ExplanationDepth(str, Enum):
    """Explanation depth levels."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class ExplainRequest(BaseModel):
    """Request body for GET /exceptions/{id}/explain (via query params)."""

    exception_id: str = Field(..., min_length=1, max_length=100)
    case_id: Optional[str] = None
    include_evidence: bool = Field(default=True)
    include_candidates: bool = Field(default=True)
    explanation_depth: ExplanationDepth = Field(default=ExplanationDepth.STANDARD)


class ExplanationResult(BaseModel):
    """Structured explanation result."""

    exception_id: str
    case_id: Optional[str] = None
    summary: str = Field(default="")
    reason: str = Field(default="")
    evidence_summary: str = Field(default="")
    uncertainty: str = Field(default="")
    limitations: str = Field(default="")
    expected_amount_paise: Optional[int] = None
    actual_amount_paise: Optional[int] = None
    difference_paise: Optional[int] = None
    exception_type: Optional[ExceptionType] = None
    evidence_record_count: int = Field(default=0, ge=0)
    evidence_coverage: Optional[EvidenceCoverage] = None
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    fallback_used: bool = Field(default=True)


class SimilarCaseItem(BaseModel):
    """A single similar case."""

    case_id: str
    exception_type: ExceptionType
    similarity_score: float = Field(ge=0.0, le=1.0)
    resolution_type: Optional[ResolutionType] = None
    adjustment_paise: Optional[int] = None
    risk_category: Optional[RiskCategory] = None


class SimilarCasesResponse(BaseModel):
    """Response for similar cases lookup."""

    exception_id: str
    similar_cases: List[SimilarCaseItem] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    confidence: Optional[str] = Field(default=None, description="HIGH, MEDIUM, LOW")


class EvidenceRecord(BaseModel):
    """A single evidence record."""

    record_type: str = Field(description="PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT")
    record_id: str
    amount_paise: int = Field(default=0, ge=0)
    status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceResponse(BaseModel):
    """Response for evidence lookup."""

    exception_id: str
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    total_amount_paise: int = Field(default=0, ge=0)
    coverage: EvidenceCoverage = Field(default=EvidenceCoverage.UNEXPLAINED)
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Learning / Feedback Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class FeedbackType(str, Enum):
    """Types of human feedback."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    CORRECT = "CORRECT"
    ESCALATE = "ESCALATE"


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    workflow_id: str = Field(..., min_length=1, max_length=100, description="Workflow ID")
    exception_id: Optional[str] = Field(default=None, max_length=100)
    candidate_id: Optional[str] = Field(default=None, max_length=100)
    reviewer: Optional[str] = Field(default=None, max_length=200, description="Reviewer identity")

    # CORRECT-specific fields
    original_resolution: Optional[str] = Field(default=None, max_length=200)
    corrected_resolution: Optional[ResolutionType] = Field(default=None)
    correction_reason: Optional[str] = Field(default=None, max_length=2000)

    # REJECT-specific fields
    rejection_reason: Optional[str] = Field(default=None, max_length=2000)

    # ESCALATE-specific fields
    escalation_reason: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Response for feedback operation."""

    feedback_id: str
    workflow_id: str
    exception_id: Optional[str] = None
    feedback_type: FeedbackType
    reviewer: str
    system_prediction: str
    status: str = Field(default="recorded")
    recorded_at: Optional[datetime] = None


class LearningMetrics(BaseModel):
    """Phase 9 learning metrics."""

    automation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    false_automation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    human_review_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolved_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_reward: float = Field(default=0.0)
    verification_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    high_value_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_financial_impact_paise: int = Field(default=0)
    total_exceptions: int = Field(default=0, ge=0)
    total_resolved: int = Field(default=0, ge=0)


class DatasetInfo(BaseModel):
    """Learning dataset information."""

    total_examples: int = Field(default=0, ge=0)
    training_size: int = Field(default=0, ge=0)
    validation_size: int = Field(default=0, ge=0)
    test_size: int = Field(default=0, ge=0)
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    last_updated: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class OverallMetrics(BaseModel):
    """System-wide metrics."""

    total_exceptions: int = Field(default=0, ge=0)
    resolved: int = Field(default=0, ge=0)
    unresolved: int = Field(default=0, ge=0)
    escalated: int = Field(default=0, ge=0)
    auto_resolved: int = Field(default=0, ge=0)
    human_review: int = Field(default=0, ge=0)
    automation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_reward: float = Field(default=0.0)
    total_financial_impact_paise: int = Field(default=0)


class SafetyMetrics(BaseModel):
    """Safety-critical metrics."""

    guardrail_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    false_automation_count: int = Field(default=0, ge=0)
    false_automation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    high_value_error_count: int = Field(default=0, ge=0)
    verification_failure_count: int = Field(default=0, ge=0)
    verification_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unsafe_exposure_paise: int = Field(default=0, ge=0)
    max_single_exposure_paise: int = Field(default=0, ge=0)


class ThroughputMetrics(BaseModel):
    """Processing throughput metrics."""

    exceptions_per_hour: float = Field(default=0.0, ge=0.0)
    avg_processing_time_ms: float = Field(default=0.0, ge=0.0)
    p95_processing_time_ms: float = Field(default=0.0, ge=0.0)
    active_workflows: int = Field(default=0, ge=0)
    queued_workflows: int = Field(default=0, ge=0)
    uptime_hours: float = Field(default=0.0, ge=0.0)


class BatchMetrics(BaseModel):
    """Metrics for a specific batch."""

    batch_id: str
    total_exceptions: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    automation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    false_automation_count: int = Field(default=0, ge=0)
    verification_failures: int = Field(default=0, ge=0)
    financial_impact_paise: int = Field(default=0)
    avg_reward: float = Field(default=0.0)
    model_version: Optional[str] = None
    policy_version: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Model Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    CANDIDATE = "CANDIDATE"
    VALIDATION = "VALIDATION"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"


class ModelVersion(BaseModel):
    """Model version summary."""

    model_id: str
    model_name: str
    model_version: str
    status: ModelStatus = Field(default=ModelStatus.CANDIDATE)
    mlflow_run_id: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    policy_version: Optional[str] = None
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None


class ModelLineage(BaseModel):
    """Full lineage chain for a model."""

    model_id: str
    model_version: str
    mlflow_run_id: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    training_config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    promotion_history: List[Dict[str, Any]] = Field(default_factory=list)
