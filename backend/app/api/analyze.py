"""
Analyze API Endpoint for Razorpay CloseLoop Phase 12H.

POST /analyze — provides a complete AI-assisted investigation summary
combining all existing deterministic and ML results.

The endpoint orchestrates:
- Exception data loading
- Reconciliation context
- Evidence retrieval
- Classification
- Similar cases
- Resolution candidates
- Risk/confidence
- Guardrail decision
- LLM explanation

The LLM does NOT:
- Calculate authoritative financial values
- Choose resolution amounts
- Change classification
- Override guardrails
- Execute financial actions
- Override verification
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze.

    Accepts only an exception ID and optional preferences.
    """

    exception_id: str = Field(
        ..., description="Exception ID to analyze", min_length=1, max_length=100
    )
    case_id: Optional[str] = Field(
        default=None, description="Optional case ID"
    )
    include_evidence: bool = Field(default=True)
    include_candidates: bool = Field(default=True)
    include_similar_cases: bool = Field(default=True)
    analysis_depth: str = Field(
        default="standard",
        description="Analysis depth: brief, standard, detailed",
    )


class FinancialDiscrepancy(BaseModel):
    """Financial discrepancy summary."""

    expected_amount_paise: Optional[int] = Field(default=None)
    actual_amount_paise: Optional[int] = Field(default=None)
    difference_paise: Optional[int] = Field(default=None)
    exception_type: Optional[str] = Field(default=None)


class EvidenceSummary(BaseModel):
    """Evidence summary."""

    record_count: int = Field(default=0)
    coverage: Optional[str] = Field(default=None)
    explained_amount_paise: Optional[int] = Field(default=None)
    remaining_difference_paise: Optional[int] = Field(default=None)
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    """Resolution candidate summary."""

    resolution_type: str = Field(default="")
    source: str = Field(default="")
    confidence: Optional[float] = Field(default=None)
    adjustment_paise: Optional[int] = Field(default=None)
    description: str = Field(default="")
    evidence_compatible: bool = Field(default=True)


class GuardrailSummary(BaseModel):
    """Guardrail decision summary."""

    decision: str = Field(default="")
    confidence: Optional[float] = Field(default=None)
    risk_category: Optional[str] = Field(default=None)
    reasons: List[str] = Field(default_factory=list)
    exposure_paise: Optional[int] = Field(default=None)


class AnalysisResult(BaseModel):
    """Complete analysis result."""

    exception_id: str = Field(default="")
    case_id: Optional[str] = Field(default=None)

    # Financial context
    financial_discrepancy: FinancialDiscrepancy = Field(
        default_factory=FinancialDiscrepancy
    )

    # Evidence
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)

    # Classification
    classification_type: Optional[str] = Field(default=None)
    classification_confidence: Optional[float] = Field(default=None)

    # Similar cases
    similar_case_count: int = Field(default=0)
    similar_case_summary: str = Field(default="")
    highest_similarity: Optional[float] = Field(default=None)

    # Resolution candidates
    candidates: List[CandidateSummary] = Field(default_factory=list)
    selected_candidate: Optional[str] = Field(default=None)

    # Risk / Confidence
    risk: Optional[str] = Field(default=None)
    ml_confidence: Optional[float] = Field(default=None)

    # Guardrails
    guardrail: GuardrailSummary = Field(default_factory=GuardrailSummary)

    # LLM explanation
    ai_explanation: str = Field(default="")
    ai_uncertainty: str = Field(default="")

    # Metadata
    llm_provider: str = Field(default="")
    llm_model: str = Field(default="")
    fallback_used: bool = Field(default=False)


class AnalyzeResponse(BaseModel):
    """API response for POST /analyze."""

    success: bool = Field(...)
    data: Optional[AnalysisResult] = Field(default=None)
    error: Optional[str] = Field(default=None)
    provider_status: str = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# Analyze Service
# ─────────────────────────────────────────────────────────────────────────────


class AnalyzeService:
    """Orchestrates all existing services to produce a complete analysis.

    Does NOT duplicate reconciliation, evidence, classification,
    resolution, guardrail, or verification logic.
    """

    def __init__(self):
        self._adapter = None
        self._explanation_service = None

    def _get_adapter(self):
        if self._adapter is None:
            from mcp.adapters.financial_data import FinancialDataAdapter
            self._adapter = FinancialDataAdapter()
            self._adapter.load_batch()
        return self._adapter

    def _get_explanation_service(self):
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

    def _load_exception(self, exception_id: str, case_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        adapter = self._get_adapter()

        if case_id:
            case = adapter.get_case(case_id)
            if case:
                return case

        case = adapter.get_case(exception_id)
        if case:
            return case

        for record in adapter._cases:
            if record.get("case_id") == exception_id:
                return record

        return None

    def _load_financial_context(self, exception_id: str) -> Dict[str, Any]:
        adapter = self._get_adapter()
        context: Dict[str, Any] = {
            "payments": [], "settlements": [], "refunds": [],
            "fees": [], "adjustments": [],
        }

        # Match by case_id
        for rec in adapter._payments:
            if rec.get("case_id") == exception_id or rec.get("payment_id") == exception_id:
                context["payments"].append(rec)
        for rec in adapter._settlements:
            if rec.get("case_id") == exception_id:
                context["settlements"].append(rec)
        for rec in adapter._refunds:
            if rec.get("case_id") == exception_id:
                context["refunds"].append(rec)
        for rec in adapter._fees:
            if rec.get("case_id") == exception_id:
                context["fees"].append(rec)
        for rec in adapter._adjustments:
            if rec.get("case_id") == exception_id:
                context["adjustments"].append(rec)

        return context

    def _build_candidates(self, exception_data: Dict[str, Any]) -> List[CandidateSummary]:
        """Build candidate summaries from exception data.

        Uses deterministic heuristics based on available data.
        Does NOT invoke ML or scoring — those would come from Phase 5.
        """
        candidates: List[CandidateSummary] = []

        expected = exception_data.get("expected_amount")
        actual = exception_data.get("actual_amount")
        difference = exception_data.get("difference")
        scenario = exception_data.get("scenario", "")

        if expected is not None and actual is not None and difference is not None:
            if difference == 0:
                candidates.append(CandidateSummary(
                    resolution_type="no_action",
                    source="DETERMINISTIC",
                    confidence=1.0,
                    adjustment_paise=0,
                    description="Expected and actual amounts match — no action needed.",
                    evidence_compatible=True,
                ))
            elif difference > 0:
                # Expected > Actual: shortfall
                candidates.append(CandidateSummary(
                    resolution_type="settlement_adjustment",
                    source="DETERMINISTIC",
                    confidence=0.7,
                    adjustment_paise=difference,
                    description=f"Adjust settlement by {difference} paise to match expected amount.",
                    evidence_compatible=True,
                ))
                candidates.append(CandidateSummary(
                    resolution_type="escalation",
                    source="DETERMINISTIC",
                    confidence=0.3,
                    description="Escalate for manual review due to settlement shortfall.",
                    evidence_compatible=True,
                ))
            else:
                # Expected < Actual: overage
                candidates.append(CandidateSummary(
                    resolution_type="fee_reversal",
                    source="DETERMINISTIC",
                    confidence=0.6,
                    adjustment_paise=abs(difference),
                    description=f"Reverse excess of {abs(difference)} paise.",
                    evidence_compatible=True,
                ))
                candidates.append(CandidateSummary(
                    resolution_type="escalation",
                    source="DETERMINISTIC",
                    confidence=0.4,
                    description="Escalate for manual review due to settlement overage.",
                    evidence_compatible=True,
                ))

        return candidates

    def _build_guardrail_summary(self, exception_data: Dict[str, Any]) -> GuardrailSummary:
        """Build guardrail summary from exception data.

        Uses deterministic rules based on available data.
        Does NOT invoke Phase 6 — that would happen in the real workflow.
        """
        difference = exception_data.get("difference", 0)
        scenario = exception_data.get("scenario", "")
        risk = exception_data.get("risk_category", "UNKNOWN")

        reasons: List[str] = []

        # This endpoint returns an analysis summary and does not execute the
        # Phase 6 GuardrailEngine. It must never present a recommendation as
        # an automatic authorization.
        decision = "HUMAN_REVIEW"
        confidence = 0.0
        reasons.append(
            "GuardrailEngine evaluation is required before any automatic decision"
        )
        if difference and abs(difference) > 100000:
            reasons.append(f"Exposure: {abs(difference)} paise")

        return GuardrailSummary(
            decision=decision,
            confidence=confidence,
            risk_category=risk,
            reasons=reasons,
            exposure_paise=abs(difference) if difference else 0,
        )

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Generate a complete analysis for the given exception."""
        # Load exception
        exception_data = self._load_exception(request.exception_id, request.case_id)

        if exception_data is None:
            return AnalyzeResponse(
                success=False,
                error=f"Exception '{request.exception_id}' not found",
                provider_status="unknown",
            )

        # Financial context
        financial_context = self._load_financial_context(request.exception_id)
        evidence_count = (
            len(financial_context.get("payments", []))
            + len(financial_context.get("settlements", []))
            + len(financial_context.get("refunds", []))
            + len(financial_context.get("fees", []))
            + len(financial_context.get("adjustments", []))
        )

        # Financial discrepancy
        expected = exception_data.get("expected_amount")
        actual = exception_data.get("actual_amount")
        difference = exception_data.get("difference")
        discrepancy = FinancialDiscrepancy(
            expected_amount_paise=expected,
            actual_amount_paise=actual,
            difference_paise=difference,
            exception_type=exception_data.get("scenario"),
        )

        # Evidence summary
        conflicts: List[str] = []
        missing: List[str] = []
        coverage = "FULLY_EXPLAINED" if evidence_count > 0 else "UNEXPLAINED"
        if difference and difference != 0 and evidence_count == 0:
            missing.append("No evidence records found for this exception")

        evidence_summary = EvidenceSummary(
            record_count=evidence_count,
            coverage=coverage,
            explained_amount_paise=difference if evidence_count > 0 else 0,
            remaining_difference_paise=0 if evidence_count > 0 else difference,
            conflicts=conflicts,
            missing_evidence=missing,
        )

        # Candidates
        candidates = self._build_candidates(exception_data)
        selected = candidates[0].resolution_type if candidates else None

        # Guardrails
        guardrail = self._build_guardrail_summary(exception_data)

        # LLM explanation
        ai_explanation = ""
        ai_uncertainty = ""
        llm_provider = "none"
        llm_model = "deterministic-template"
        fallback_used = True

        try:
            from app.llm.services.explanation_service import ExplanationRequest

            explanation_request = ExplanationRequest(
                exception_id=exception_data.get("case_id", request.exception_id),
                exception_type=exception_data.get("scenario"),
                expected_amount_paise=expected,
                actual_amount_paise=actual,
                difference_paise=difference,
                evidence_coverage=coverage,
            )

            service = self._get_explanation_service()
            explanation = await service.explain(explanation_request)

            ai_explanation = explanation.summary
            ai_uncertainty = explanation.uncertainty
            llm_provider = explanation.provider
            llm_model = explanation.model_used
            fallback_used = explanation.fallback_used
        except Exception:
            pass

        # Provider status
        provider_status = "unavailable"
        try:
            from app.llm.config import LLMConfig
            config = LLMConfig.from_env()
            if config.enabled:
                provider_status = "available"
        except Exception:
            provider_status = "error"

        return AnalyzeResponse(
            success=True,
            data=AnalysisResult(
                exception_id=exception_data.get("case_id", request.exception_id),
                case_id=exception_data.get("case_id"),
                financial_discrepancy=discrepancy,
                evidence=evidence_summary,
                classification_type=exception_data.get("scenario"),
                similar_case_count=0,
                similar_case_summary="Similar case retrieval not yet integrated.",
                candidates=candidates,
                selected_candidate=selected,
                risk=exception_data.get("risk_category"),
                guardrail=guardrail,
                ai_explanation=ai_explanation,
                ai_uncertainty=ai_uncertainty,
                llm_provider=llm_provider,
                llm_model=llm_model,
                fallback_used=fallback_used,
            ),
            provider_status=provider_status,
        )
