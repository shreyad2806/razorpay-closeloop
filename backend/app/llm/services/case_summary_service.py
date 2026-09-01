"""
LLM Case Summary Service for Razorpay CloseLoop Phase 12E.

Uses the LLM to summarize historical similar cases retrieved by Phase 4.

The LLM may:
- Summarize similar cases
- Identify recurring patterns
- Describe how previous cases were resolved
- Highlight differences between current and historical cases

The LLM must NOT:
- Decide that current case equals historical case
- Automatically copy historical resolution
- Change current financial values
- Bypass Phase 5 scoring
- Bypass Phase 6 guardrails
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.llm.config import LLMConfig
from app.llm.logging import LLMLogger
from app.llm.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)
from app.llm.retry import LLMRetryExecutor, RetryConfig


# ─────────────────────────────────────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────────────────────────────────────


class SimilarCaseInfo(BaseModel):
    """Simplified similar case for summary input."""

    case_id: str = Field(..., description="Historical case identifier")
    similarity_score: float = Field(..., description="Similarity score 0.0-1.0")
    exception_type: str = Field(default="", description="Historical exception type")
    resolution_type: str = Field(default="", description="Resolution applied")
    resolution_outcome: str = Field(default="", description="Outcome of resolution")
    payment_amount_paise: int = Field(default=0, description="Payment amount in paise")
    difference_paise: int = Field(default=0, description="Discrepancy in paise")
    evidence_count: int = Field(default=0, description="Number of evidence records")
    tags: List[str] = Field(default_factory=list, description="Case tags")


class CaseSummaryRequest(BaseModel):
    """Input to the case summary service.

    Contains the current exception context and historical similar cases
    from Phase 4 similarity search.
    """

    # Current exception context
    exception_id: str = Field(..., description="Current exception ID")
    case_id: Optional[str] = Field(default=None, description="Current case ID")
    exception_type: Optional[str] = Field(default=None, description="Current exception type")
    expected_amount_paise: Optional[int] = Field(default=None, description="Expected amount")
    actual_amount_paise: Optional[int] = Field(default=None, description="Actual amount")
    difference_paise: Optional[int] = Field(default=None, description="Discrepancy")

    # Historical cases from Phase 4
    similar_cases: List[SimilarCaseInfo] = Field(
        default_factory=list, description="Ranked similar historical cases"
    )
    total_indexed: int = Field(default=0, description="Total cases in historical index")
    similarity_metric: str = Field(default="cosine", description="Similarity metric used")
    top_k: int = Field(default=5, description="Number of results requested")

    # Metadata
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @classmethod
    def from_similarity_result(cls, result: Any, exception_id: str = "", **kwargs) -> "CaseSummaryRequest":
        """Build from a SimilaritySearchResult object or dict."""
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        similar_raw = _get(result, "similar_cases", []) or []
        cases = []
        for sc in similar_raw:
            cases.append(SimilarCaseInfo(
                case_id=_get(sc, "case_id", ""),
                similarity_score=_get(sc, "similarity_score", 0.0),
                exception_type=_get(sc, "exception_type", ""),
                resolution_type=_get(sc, "resolution_type", ""),
                resolution_outcome=_get(sc, "resolution_outcome", ""),
                payment_amount_paise=_get(sc, "payment_amount", 0),
                difference_paise=_get(sc, "difference", 0),
                evidence_count=_get(sc, "evidence_count", 0),
                tags=_get(sc, "tags", []) or [],
            ))

        return cls(
            exception_id=exception_id or _get(result, "query_case_id", ""),
            similar_cases=cases,
            total_indexed=_get(result, "total_indexed", 0),
            similarity_metric=_get(result, "similarity_metric", "cosine"),
            top_k=_get(result, "top_k", 5),
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────────────────────────────────────


class CaseSummaryOutput(BaseModel):
    """Structured output from the case summary service."""

    similar_cases_summary: str = Field(default="", description="Overall summary of similar cases")
    common_pattern: str = Field(default="", description="Recurring pattern across cases")
    important_differences: str = Field(default="", description="How current case differs from historical")
    historical_resolution_summary: str = Field(default="", description="How similar cases were resolved")
    confidence: str = Field(default="", description="Confidence in the pattern match")
    uncertainty: str = Field(default="", description="What is uncertain")
    recommendation_note: str = Field(default="", description="Important note about using historical data")
    model_used: str = Field(default="", description="Model that produced the summary")
    provider: str = Field(default="", description="Provider used")
    fallback_used: bool = Field(default=False, description="Whether deterministic fallback was used")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

CASE_SUMMARY_SYSTEM_PROMPT = """You are a historical case analysis assistant.

Your role:
- Summarize what historical similar cases tell us about the current exception
- Identify recurring patterns across cases
- Highlight important differences between current and historical cases
- Describe how similar cases were resolved

You MUST:
- Use ONLY the historical cases provided
- Be precise about similarity scores — note when similarity is low
- Clearly state that historical cases are REFERENCE ONLY, not directives
- Highlight when cases differ significantly from the current exception

You MUST NOT:
- Decide that the current case equals a historical case
- Automatically recommend copying a historical resolution
- Change any financial amounts for the current case
- Bypass scoring or guardrail systems
- Treat low-similarity cases as strong precedent

Format your response as JSON with these fields:
- similar_cases_summary: Overall summary of what historical cases show
- common_pattern: Recurring pattern across cases (or "No clear pattern")
- important_differences: How current case differs from historical cases
- historical_resolution_summary: How similar cases were resolved
- confidence: Confidence in the pattern match (HIGH/MEDIUM/LOW)
- uncertainty: What remains uncertain
- recommendation_note: Important caveat about using historical data"""


# ─────────────────────────────────────────────────────────────────────────────
# Build User Prompt
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(amount: Optional[int]) -> str:
    if amount is None:
        return "Not provided"
    return f"₹{amount / 100:,.2f} ({amount} paise)"


def build_case_summary_prompt(request: CaseSummaryRequest) -> str:
    """Build the user prompt from structured case data."""
    parts = [
        "# Historical Case Summary Request",
        "",
        f"Current Exception ID: {request.exception_id}",
    ]

    if request.case_id:
        parts.append(f"Current Case ID: {request.case_id}")
    if request.exception_type:
        parts.append(f"Exception Type: {request.exception_type}")

    parts.append("")
    parts.append("## Current Exception Context")
    parts.append(f"Expected Amount: {_fmt(request.expected_amount_paise)}")
    parts.append(f"Actual Amount: {_fmt(request.actual_amount_paise)}")
    parts.append(f"Discrepancy: {_fmt(request.difference_paise)}")

    # Similar cases
    parts.append("")
    parts.append(f"## Historical Similar Cases ({len(request.similar_cases)} found, {request.total_indexed} indexed)")
    parts.append(f"Similarity Metric: {request.similarity_metric}")

    if request.similar_cases:
        for i, sc in enumerate(request.similar_cases, 1):
            parts.append(f"\n### Case {i}: {sc.case_id}")
            parts.append(f"Similarity: {sc.similarity_score:.1%}")
            parts.append(f"Exception Type: {sc.exception_type}")
            parts.append(f"Resolution: {sc.resolution_type} → {sc.resolution_outcome}")
            parts.append(f"Payment: {_fmt(sc.payment_amount_paise)}")
            parts.append(f"Discrepancy: {_fmt(sc.difference_paise)}")
            parts.append(f"Evidence Records: {sc.evidence_count}")
            if sc.tags:
                parts.append(f"Tags: {', '.join(sc.tags)}")
    else:
        parts.append("\nNo similar historical cases were found.")

    parts.append("")
    parts.append("Please analyze these historical cases and summarize what they tell us.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Response
# ─────────────────────────────────────────────────────────────────────────────


def _parse_case_summary_response(
    content: str,
    provider: str = "",
    model: str = "",
    fallback: bool = False,
) -> CaseSummaryOutput:
    """Parse LLM response into structured case summary."""
    import json

    try:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
        return CaseSummaryOutput(
            similar_cases_summary=data.get("similar_cases_summary", ""),
            common_pattern=data.get("common_pattern", ""),
            important_differences=data.get("important_differences", ""),
            historical_resolution_summary=data.get("historical_resolution_summary", ""),
            confidence=data.get("confidence", ""),
            uncertainty=data.get("uncertainty", ""),
            recommendation_note=data.get("recommendation_note", ""),
            model_used=model,
            provider=provider,
            fallback_used=fallback,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return CaseSummaryOutput(
        similar_cases_summary=content.strip() if content.strip() else "No case summary available.",
        model_used=model,
        provider=provider,
        fallback_used=fallback,
        uncertainty="Could not parse structured LLM response.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback
# ─────────────────────────────────────────────────────────────────────────────


def _case_summary_deterministic_fallback(request: CaseSummaryRequest) -> CaseSummaryOutput:
    """Produce deterministic case summary when LLM is unavailable."""
    summary_parts: List[str] = []
    pattern_parts: List[str] = []
    differences_parts: List[str] = []
    resolution_parts: List[str] = []
    uncertainty_parts: List[str] = []
    limitations_parts: List[str] = [
        "This summary was generated without LLM assistance.",
        "A template-based summary was produced instead.",
    ]

    cases = request.similar_cases
    count = len(cases)

    # Summary
    if count == 0:
        summary_parts.append(
            f"No similar historical cases found for exception {request.exception_id} "
            f"out of {request.total_indexed} indexed cases."
        )
        pattern_parts.append("No pattern can be identified — no similar cases available.")
        uncertainty_parts.append("Without historical precedent, resolution must rely solely on current evidence.")
    else:
        summary_parts.append(
            f"{count} similar historical case(s) found for exception {request.exception_id} "
            f"out of {request.total_indexed} indexed cases."
        )

        # Similarity distribution
        high = [c for c in cases if c.similarity_score >= 0.8]
        medium = [c for c in cases if 0.5 <= c.similarity_score < 0.8]
        low = [c for c in cases if c.similarity_score < 0.5]

        if high:
            summary_parts.append(f"{len(high)} case(s) with high similarity (≥80%).")
        if medium:
            summary_parts.append(f"{len(medium)} case(s) with medium similarity (50-80%).")
        if low:
            summary_parts.append(f"{len(low)} case(s) with low similarity (<50%).")

        # Pattern detection
        resolution_types = [c.resolution_type for c in cases if c.resolution_type]
        if resolution_types:
            from collections import Counter
            type_counts = Counter(resolution_types)
            most_common = type_counts.most_common(1)
            if most_common:
                pattern_parts.append(
                    f"Most common resolution type: {most_common[0][0]} "
                    f"({most_common[0][1]}/{count} cases)."
                )

        outcome_types = [c.resolution_outcome for c in cases if c.resolution_outcome]
        if outcome_types:
            from collections import Counter
            outcome_counts = Counter(outcome_types)
            successes = sum(1 for o in outcome_types if "success" in o.lower() or "verified" in o.lower())
            if successes > 0:
                pattern_parts.append(
                    f"{successes}/{len(outcome_types)} historical cases had successful outcomes."
                )

        # Resolution summary
        for sc in cases[:3]:  # Top 3
            resolution_parts.append(
                f"[{sc.case_id}] {sc.resolution_type} → {sc.resolution_outcome} "
                f"(similarity: {sc.similarity_score:.0%})"
            )

        # Differences
        if request.difference_paise is not None:
            current_diff = request.difference_paise
            hist_diffs = [c.difference_paise for c in cases if c.difference_paise != 0]
            if hist_diffs:
                avg_hist = sum(hist_diffs) // len(hist_diffs)
                if abs(current_diff - avg_hist) > abs(avg_hist) * 0.5:
                    differences_parts.append(
                        f"Current discrepancy ({_fmt(current_diff)}) differs significantly "
                        f"from historical average ({_fmt(avg_hist)})."
                    )
                else:
                    differences_parts.append(
                        f"Current discrepancy ({_fmt(current_diff)}) is within range "
                        f"of historical cases (avg: {_fmt(avg_hist)})."
                    )

        # Uncertainty
        if low:
            uncertainty_parts.append(
                f"{len(low)} case(s) have low similarity — may not be relevant."
            )
        if count < 3:
            uncertainty_parts.append("Limited historical data available.")
        if not uncertainty_parts:
            uncertainty_parts.append("Pattern confidence is moderate based on available cases.")

    # Confidence
    if count == 0:
        confidence = "LOW — No historical precedent available."
    elif any(c.similarity_score >= 0.8 for c in cases):
        confidence = "HIGH — At least one high-similarity match found."
    elif any(c.similarity_score >= 0.5 for c in cases):
        confidence = "MEDIUM — Moderate similarity matches found."
    else:
        confidence = "LOW — Only low-similarity matches found."

    return CaseSummaryOutput(
        similar_cases_summary=" ".join(summary_parts),
        common_pattern=" ".join(pattern_parts) if pattern_parts else "No clear pattern identified.",
        important_differences=" ".join(differences_parts) if differences_parts else "No significant differences noted.",
        historical_resolution_summary=" ".join(resolution_parts) if resolution_parts else "No resolution history available.",
        confidence=confidence,
        uncertainty=" ".join(uncertainty_parts),
        recommendation_note=(
            "Historical cases are REFERENCE ONLY. "
            "They do not determine the current resolution. "
            "Phase 5 scoring and Phase 6 guardrails are independent."
        ),
        model_used="deterministic-template",
        provider="none",
        fallback_used=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case Summary Service
# ─────────────────────────────────────────────────────────────────────────────


class LLMCaseSummaryService:
    """Service that produces human-readable summaries of historical similar cases.

    Uses the LLM to summarize cases from Phase 4 similarity search.
    Falls back to deterministic templates when LLM is unavailable.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[LLMConfig] = None,
        logger: Optional[LLMLogger] = None,
    ):
        self._provider = provider
        self._config = config or LLMConfig.from_env()
        self._logger = logger or LLMLogger("llm.case_summary")

        if provider is not None:
            self._executor = LLMRetryExecutor(
                provider,
                RetryConfig(
                    max_retries=min(self._config.get_provider_config().max_retries, 2),
                    base_delay=0.5,
                    jitter=True,
                ),
                on_retry=self._on_retry,
            )
        else:
            self._executor = None

    def _on_retry(self, attempt: int, max_retries: int, reason: str) -> None:
        model = ""
        if self._provider:
            cfg = self._config.get_provider_config()
            model = getattr(cfg, "model", "")
        self._logger.log_retry(
            provider=self._provider.provider_name if self._provider else "none",
            model=model,
            attempt=attempt,
            max_retries=max_retries,
            reason=reason,
        )

    async def summarize(self, request: CaseSummaryRequest) -> CaseSummaryOutput:
        """Produce a summary of historical similar cases.

        If LLM is available, uses it. Otherwise, uses deterministic templates.
        """
        if self._executor is None:
            self._logger.log_provider_unavailable(
                provider="none", model="",
                reason="No LLM provider configured — using deterministic fallback",
            )
            return _case_summary_deterministic_fallback(request)

        provider_name = self._provider.provider_name if self._provider else ""
        model = ""
        if self._provider:
            cfg = self._config.get_provider_config()
            model = getattr(cfg, "model", "")

        user_prompt = build_case_summary_prompt(request)

        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=CASE_SUMMARY_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            metadata={
                "workflow_id": request.workflow_id or "",
                "exception_id": request.exception_id,
                "service": "case_summary",
                "similar_case_count": len(request.similar_cases),
            },
        )

        self._logger.log_request_start(
            provider=provider_name, model=model,
            workflow_id=request.workflow_id,
            exception_id=request.exception_id,
            metadata={"service": "case_summary"},
        )

        try:
            response: LLMResponse = await self._executor.generate(llm_request)

            self._logger.log_request_success(
                provider=provider_name, model=model,
                duration_ms=response.metadata.get("elapsed_ms", 0.0),
                tokens_used=response.usage.get("total_tokens"),
                finish_reason=response.finish_reason,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
            )

            return _parse_case_summary_response(
                response.content,
                provider=provider_name,
                model=response.model or model,
                fallback=False,
            )

        except LLMProviderError as e:
            self._logger.log_request_error(
                provider=provider_name, model=model, duration_ms=0.0,
                error_type=type(e).__name__, error_message=str(e),
                workflow_id=request.workflow_id, exception_id=request.exception_id,
            )
            result = _case_summary_deterministic_fallback(request)
            result.uncertainty = f"LLM unavailable ({type(e).__name__}). {result.uncertainty}"
            return result

        except Exception as e:
            self._logger.log_request_error(
                provider=provider_name, model=model, duration_ms=0.0,
                error_type=type(e).__name__, error_message=str(e),
                workflow_id=request.workflow_id, exception_id=request.exception_id,
            )
            result = _case_summary_deterministic_fallback(request)
            result.uncertainty = f"Unexpected error ({type(e).__name__}). {result.uncertainty}"
            return result

    async def health_check(self):
        if self._executor is None:
            from app.llm.providers.base import LLMHealthStatus
            return LLMHealthStatus(
                provider="none", healthy=True, model="deterministic-template",
                details={"message": "LLM unavailable — deterministic fallback active"},
            )
        return await self._executor.health_check()

    async def close(self):
        if self._executor:
            await self._executor.close()
