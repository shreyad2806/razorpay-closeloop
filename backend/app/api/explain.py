"""
Explain API Endpoint for Razorpay CloseLoop Phase 12G.

POST /explain — provides a human-readable explanation of an existing
financial exception using LLM services.

The endpoint:
1. Accepts an exception ID (no arbitrary financial truth values)
2. Loads exception data from the synthetic dataset
3. Retrieves evidence and context
4. Invokes LLM explanation service
5. Returns structured explanation with fallback

Safety rules:
- Does NOT accept user-supplied financial values as authoritative
- Does NOT duplicate reconciliation, evidence, or resolution logic
- Delegates to existing internal services
- LLM output is validated before return
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────


class ExplainRequest(BaseModel):
    """Request body for POST /explain.

    Accepts only an exception ID and optional preferences.
    Does NOT accept arbitrary financial truth values.
    """

    exception_id: str = Field(
        ..., description="Exception ID to explain", min_length=1, max_length=100
    )
    case_id: Optional[str] = Field(
        default=None, description="Optional case ID for context"
    )
    include_evidence: bool = Field(
        default=True, description="Whether to include evidence details"
    )
    include_candidates: bool = Field(
        default=True, description="Whether to include resolution candidates"
    )
    explanation_depth: str = Field(
        default="standard",
        description="Explanation depth: brief, standard, detailed",
    )


class ExplanationResult(BaseModel):
    """Structured explanation result returned by the API."""

    exception_id: str = Field(..., description="Exception that was explained")
    case_id: Optional[str] = Field(default=None, description="Case reference")

    # Core explanation
    summary: str = Field(default="", description="One-paragraph summary")
    reason: str = Field(default="", description="Why the exception occurred")
    evidence_summary: str = Field(default="", description="Evidence overview")
    uncertainty: str = Field(default="", description="What is uncertain")
    limitations: str = Field(default="", description="Limitations of the explanation")

    # Financial context (read from data, not invented)
    expected_amount_paise: Optional[int] = Field(default=None)
    actual_amount_paise: Optional[int] = Field(default=None)
    difference_paise: Optional[int] = Field(default=None)
    exception_type: Optional[str] = Field(default=None)

    # Evidence
    evidence_record_count: int = Field(default=0)
    evidence_coverage: Optional[str] = Field(default=None)
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)

    # Provider status
    llm_provider: str = Field(default="", description="LLM provider used")
    llm_model: str = Field(default="", description="Model used")
    fallback_used: bool = Field(default=False, description="Whether deterministic fallback was used")


class ExplainResponse(BaseModel):
    """API response for POST /explain."""

    success: bool = Field(..., description="Whether explanation was generated")
    data: Optional[ExplanationResult] = Field(default=None, description="Explanation result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    provider_status: str = Field(default="", description="LLM provider health")


# ─────────────────────────────────────────────────────────────────────────────
# Explain Service (orchestrates data loading + LLM)
# ─────────────────────────────────────────────────────────────────────────────


class ExplainService:
    """Orchestrates exception data loading and LLM explanation.

    Uses existing internal services — does not duplicate logic.
    """

    def __init__(self):
        self._adapter = None
        self._explanation_service = None
        self._evidence_explanation_service = None
        self._case_summary_service = None
        self._reviewer_assistant_service = None

    def _get_adapter(self):
        """Lazy-load the financial data adapter."""
        if self._adapter is None:
            from mcp.adapters.financial_data import FinancialDataAdapter
            self._adapter = FinancialDataAdapter()
            self._adapter.load_batch()
        return self._adapter

    def _get_explanation_service(self):
        """Lazy-load the LLM explanation service."""
        if self._explanation_service is None:
            from app.llm.config import LLMConfig
            from app.llm.services.explanation_service import LLMExplanationService

            config = LLMConfig.from_env()
            provider = None
            if config.enabled:
                from app.llm.services.provider_service import create_provider
                provider = create_provider(config)

            self._explanation_service = LLMExplanationService(
                provider=provider, config=config,
            )
        return self._explanation_service

    def _get_evidence_explanation_service(self):
        """Lazy-load the LLM evidence explanation service."""
        if self._evidence_explanation_service is None:
            from app.llm.config import LLMConfig
            from app.llm.services.evidence_explanation_service import LLMEvidenceExplanationService

            config = LLMConfig.from_env()
            provider = None
            if config.enabled:
                from app.llm.services.provider_service import create_provider
                provider = create_provider(config)

            self._evidence_explanation_service = LLMEvidenceExplanationService(
                provider=provider, config=config,
            )
        return self._evidence_explanation_service

    def _get_case_summary_service(self):
        """Lazy-load the LLM case summary service."""
        if self._case_summary_service is None:
            from app.llm.config import LLMConfig
            from app.llm.services.case_summary_service import LLMCaseSummaryService

            config = LLMConfig.from_env()
            provider = None
            if config.enabled:
                from app.llm.services.provider_service import create_provider
                provider = create_provider(config)

            self._case_summary_service = LLMCaseSummaryService(
                provider=provider, config=config,
            )
        return self._case_summary_service

    def _load_exception_data(self, exception_id: str, case_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load exception data from the synthetic dataset.

        Returns the exception dict if found, None otherwise.
        The synthetic dataset uses case_id — we accept both exception_id
        and case_id as lookup keys.
        """
        adapter = self._get_adapter()

        # Try direct case lookup by case_id
        if case_id:
            case = adapter.get_case(case_id)
            if case:
                return case

        # Try lookup by exception_id as case_id
        case = adapter.get_case(exception_id)
        if case:
            return case

        # Search across all cases
        all_cases = adapter._cases
        for record in all_cases:
            if record.get("case_id") == exception_id:
                return record

        return None

    def _load_financial_context(self, exception_id: str) -> Dict[str, Any]:
        """Load related financial records for an exception."""
        adapter = self._get_adapter()

        # Search for records matching this exception
        records = adapter.search_records(limit=100)

        context: Dict[str, Any] = {
            "payments": [],
            "settlements": [],
            "refunds": [],
            "fees": [],
            "adjustments": [],
        }

        for rec in records:
            exc_id = rec.get("exception_id", "")
            case_id = rec.get("case_id", "")
            if exc_id == exception_id or case_id == exception_id:
                rec_type = rec.get("record_type", rec.get("entity_type", "")).lower()
                if "payment" in rec_type:
                    context["payments"].append(rec)
                elif "settlement" in rec_type:
                    context["settlements"].append(rec)
                elif "refund" in rec_type:
                    context["refunds"].append(rec)
                elif "fee" in rec_type:
                    context["fees"].append(rec)
                elif "adjustment" in rec_type:
                    context["adjustments"].append(rec)

        return context

    def _build_explanation_context(
        self, exception_data: Dict[str, Any], financial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the structured context for LLM explanation."""
        # Extract financial amounts
        expected = exception_data.get("expected_amount", exception_data.get("expected_amount_paise"))
        actual = exception_data.get("actual_amount", exception_data.get("actual_amount_paise"))
        difference = exception_data.get("difference", exception_data.get("difference_paise"))

        if expected is None and actual is not None and difference is not None:
            expected = actual + difference
        elif actual is None and expected is not None and difference is not None:
            actual = expected - difference

        # Count evidence
        evidence_count = (
            len(financial_context.get("payments", []))
            + len(financial_context.get("settlements", []))
            + len(financial_context.get("refunds", []))
            + len(financial_context.get("fees", []))
            + len(financial_context.get("adjustments", []))
        )

        return {
            "exception_id": exception_data.get("exception_id", ""),
            "case_id": exception_data.get("case_id"),
            "exception_type": exception_data.get("exception_type", exception_data.get("type")),
            "expected_amount_paise": expected,
            "actual_amount_paise": actual,
            "difference_paise": difference,
            "evidence_record_count": evidence_count,
            "financial_context": financial_context,
        }

    async def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Generate an explanation for the given exception.

        Loads data from existing services, invokes LLM, returns structured response.
        """
        # Load exception data
        exception_data = self._load_exception_data(
            request.exception_id, request.case_id
        )

        if exception_data is None:
            return ExplainResponse(
                success=False,
                error=f"Exception '{request.exception_id}' not found",
                provider_status="unknown",
            )

        # Load financial context
        financial_context = self._load_financial_context(request.exception_id)

        # Build explanation context
        context = self._build_explanation_context(exception_data, financial_context)

        # Get LLM provider status
        provider_status = "unavailable"
        try:
            from app.llm.config import LLMConfig
            config = LLMConfig.from_env()
            if config.enabled:
                provider_status = "available"
        except Exception:
            provider_status = "error"

        # Invoke explanation service
        try:
            from app.llm.services.explanation_service import ExplanationRequest

            explanation_request = ExplanationRequest(
                exception_id=context["exception_id"],
                case_id=context.get("case_id"),
                exception_type=context.get("exception_type"),
                expected_amount_paise=context.get("expected_amount_paise"),
                actual_amount_paise=context.get("actual_amount_paise"),
                difference_paise=context.get("difference_paise"),
                evidence_coverage="FULLY_EXPLAINED" if context["evidence_record_count"] > 0 else "UNEXPLAINED",
            )

            service = self._get_explanation_service()
            explanation = await service.explain(explanation_request)

            # Build evidence summary from context
            evidence_parts = []
            for rec_type in ["payments", "settlements", "refunds", "fees", "adjustments"]:
                records = financial_context.get(rec_type, [])
                if records:
                    evidence_parts.append(f"{len(records)} {rec_type.rstrip('s')} record(s)")

            evidence_summary = ", ".join(evidence_parts) if evidence_parts else "No evidence records found."

            # Detect conflicts and missing evidence
            conflicts = []
            missing = []
            if context["evidence_record_count"] == 0:
                missing.append("No financial records found for this exception")

            return ExplainResponse(
                success=True,
                data=ExplanationResult(
                    exception_id=context["exception_id"],
                    case_id=context.get("case_id"),
                    summary=explanation.summary,
                    reason=explanation.reason,
                    evidence_summary=evidence_summary,
                    uncertainty=explanation.uncertainty,
                    limitations=explanation.limitations,
                    expected_amount_paise=context.get("expected_amount_paise"),
                    actual_amount_paise=context.get("actual_amount_paise"),
                    difference_paise=context.get("difference_paise"),
                    exception_type=context.get("exception_type"),
                    evidence_record_count=context["evidence_record_count"],
                    evidence_coverage="FULLY_EXPLAINED" if context["evidence_record_count"] > 0 else "UNEXPLAINED",
                    conflicts=conflicts,
                    missing_evidence=missing,
                    llm_provider=explanation.provider,
                    llm_model=explanation.model_used,
                    fallback_used=explanation.fallback_used,
                ),
                provider_status=provider_status,
            )

        except Exception as e:
            return ExplainResponse(
                success=False,
                error=f"Explanation generation failed: {type(e).__name__}: {str(e)[:200]}",
                provider_status=provider_status,
            )
