"""
LLM Explanation Service for Razorpay CloseLoop Phase 12C.

Uses the LLM to explain already-computed financial information in
natural language. The LLM does NOT calculate or decide anything.

The explanation service receives structured data produced by:
- Phase 2: Deterministic reconciliation
- Phase 3: Financial evidence
- Phase 4: ML classification + similar cases
- Phase 5: Resolution candidates
- Phase 6: Guardrails

And produces a human-readable explanation of what happened and why.

IMPORTANT:
- The LLM is an explanation assistant, not a financial authority
- The LLM does NOT independently calculate financial state
- The LLM does NOT make financial decisions
- The LLM does NOT change amounts
- The LLM does NOT create resolution instructions
- The LLM does NOT override guardrails
- If evidence is insufficient, the LLM must say so
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
# Explanation Input
# ─────────────────────────────────────────────────────────────────────────────


class ExplanationEvidence(BaseModel):
    """A single piece of evidence for explanation."""

    evidence_id: str = Field(..., description="Unique evidence identifier")
    record_type: str = Field(..., description="Type: payment, settlement, refund, fee, tax, adjustment")
    description: str = Field(default="", description="What this evidence shows")
    amount_paise: Optional[int] = Field(default=None, description="Financial amount in paise")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ExplanationRequest(BaseModel):
    """Input to the LLM explanation service.

    Contains all structured data already computed by deterministic systems.
    The LLM will rephrase this into natural language — not recalculate it.
    """

    exception_id: str = Field(..., description="Exception being explained")
    case_id: Optional[str] = Field(default=None, description="Case reference")
    exception_type: Optional[str] = Field(default=None, description="Classified exception type")

    # Financial context from Phase 2
    expected_amount_paise: Optional[int] = Field(default=None, description="Expected settlement in paise")
    actual_amount_paise: Optional[int] = Field(default=None, description="Actual settlement in paise")
    difference_paise: Optional[int] = Field(default=None, description="Discrepancy in paise")

    # Evidence from Phase 3
    evidence_items: List[ExplanationEvidence] = Field(
        default_factory=list, description="Supporting evidence records"
    )
    evidence_coverage: Optional[str] = Field(
        default=None, description="FULLY_EXPLAINED, PARTIALLY_EXPLAINED, UNEXPLAINED, CONFLICTING"
    )
    explained_amount_paise: Optional[int] = Field(default=None, description="Portion explained by evidence")
    remaining_difference_paise: Optional[int] = Field(default=None, description="Unexplained portion")

    # Classification from Phase 4
    classification_confidence: Optional[float] = Field(default=None, description="Classification confidence")
    similar_case_count: Optional[int] = Field(default=None, description="Number of similar historical cases")

    # Candidate from Phase 5
    candidate_resolution_type: Optional[str] = Field(default=None, description="Proposed resolution type")
    candidate_adjustment_paise: Optional[int] = Field(default=None, description="Proposed adjustment in paise")
    candidate_description: Optional[str] = Field(default=None, description="What the resolution proposes")

    # Guardrails from Phase 6
    guardrail_decision: Optional[str] = Field(default=None, description="AUTO, HUMAN_REVIEW, UNRESOLVED")
    guardrail_confidence: Optional[float] = Field(default=None, description="Final confidence score")
    risk_category: Optional[str] = Field(default=None, description="Risk level")
    guardrail_reasons: List[str] = Field(default_factory=list, description="Why guardrails decided this way")

    # Metadata
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Output
# ─────────────────────────────────────────────────────────────────────────────


class LLMExplanationOutput(BaseModel):
    """Structured output from the LLM explanation service."""

    # Core explanation
    summary: str = Field(default="", description="One-paragraph summary of the exception and resolution")
    reason: str = Field(default="", description="Why this exception occurred")
    supporting_evidence: str = Field(default="", description="How the evidence supports the explanation")

    # Uncertainty and limitations
    uncertainty: str = Field(default="", description="What is uncertain or incomplete")
    limitations: str = Field(default="", description="Limitations of the explanation")

    # Metadata
    model_used: str = Field(default="", description="Which model produced the explanation")
    provider: str = Field(default="", description="Which provider was used")
    fallback_used: bool = Field(
        default=False, description="Whether a deterministic fallback was used instead of LLM"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM_PROMPT = """You are a financial exception explanation assistant for Razorpay.

Your role:
- Explain what happened with a financial exception in clear, professional language
- Use ONLY the structured evidence and data provided to you
- Make technical financial information easier to understand

You MUST:
- Use only the supplied evidence and data
- Stay factual — do not invent financial information
- Acknowledge uncertainty when evidence is incomplete
- Note any limitations in the available information

You MUST NOT:
- Independently calculate or verify financial amounts
- Make financial decisions or recommendations
- Change or override any amounts
- Create new resolution instructions
- Override guardrail decisions
- Speculate about information not provided
- Provide legal or compliance advice

Format your response as JSON with these fields:
- summary: One paragraph explaining what happened
- reason: Why this exception occurred
- supporting_evidence: How the evidence supports the conclusion
- uncertainty: What is uncertain or incomplete
- limitations: Any limitations of this explanation"""


# ─────────────────────────────────────────────────────────────────────────────
# Build User Prompt
# ─────────────────────────────────────────────────────────────────────────────


def _format_paise(amount: Optional[int]) -> str:
    """Format paise amount as readable string."""
    if amount is None:
        return "Not provided"
    rupees = amount / 100
    return f"₹{rupees:,.2f} ({amount} paise)"


def build_explanation_prompt(request: ExplanationRequest) -> str:
    """Build the user prompt from structured explanation data.

    This converts structured data into a text prompt for the LLM.
    All values come from deterministic systems — the LLM just rephrases them.
    """
    parts = [
        "# Financial Exception Explanation Request",
        "",
        f"Exception ID: {request.exception_id}",
    ]

    if request.case_id:
        parts.append(f"Case ID: {request.case_id}")
    if request.exception_type:
        parts.append(f"Exception Type: {request.exception_type}")

    # Financial context
    parts.append("")
    parts.append("## Financial Context")
    parts.append(f"Expected Amount: {_format_paise(request.expected_amount_paise)}")
    parts.append(f"Actual Amount: {_format_paise(request.actual_amount_paise)}")
    parts.append(f"Discrepancy: {_format_paise(request.difference_paise)}")

    # Evidence
    if request.evidence_items:
        parts.append("")
        parts.append("## Evidence")
        for ev in request.evidence_items:
            amount_str = _format_paise(ev.amount_paise) if ev.amount_paise else ""
            parts.append(f"- [{ev.record_type}] {ev.evidence_id}: {ev.description} {amount_str}")

    if request.evidence_coverage:
        parts.append(f"\nEvidence Coverage: {request.evidence_coverage}")
    if request.explained_amount_paise is not None:
        parts.append(f"Explained Amount: {_format_paise(request.explained_amount_paise)}")
    if request.remaining_difference_paise is not None:
        parts.append(f"Remaining Unexplained: {_format_paise(request.remaining_difference_paise)}")

    # Classification
    if request.classification_confidence is not None or request.similar_case_count is not None:
        parts.append("")
        parts.append("## Classification")
        if request.classification_confidence is not None:
            parts.append(f"Classification Confidence: {request.classification_confidence:.1%}")
        if request.similar_case_count is not None:
            parts.append(f"Similar Historical Cases: {request.similar_case_count}")

    # Candidate resolution
    if request.candidate_resolution_type:
        parts.append("")
        parts.append("## Proposed Resolution")
        parts.append(f"Resolution Type: {request.candidate_resolution_type}")
        if request.candidate_adjustment_paise is not None:
            parts.append(f"Adjustment: {_format_paise(request.candidate_adjustment_paise)}")
        if request.candidate_description:
            parts.append(f"Description: {request.candidate_description}")

    # Guardrails
    if request.guardrail_decision:
        parts.append("")
        parts.append("## Guardrail Decision")
        parts.append(f"Decision: {request.guardrail_decision}")
        if request.guardrail_confidence is not None:
            parts.append(f"Confidence: {request.guardrail_confidence:.1%}")
        if request.risk_category:
            parts.append(f"Risk: {request.risk_category}")
        if request.guardrail_reasons:
            parts.append("Reasons:")
            for reason in request.guardrail_reasons:
                parts.append(f"  - {reason}")

    parts.append("")
    parts.append("Please explain this exception in clear, professional language.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM Response
# ─────────────────────────────────────────────────────────────────────────────


def _parse_explanation_response(
    content: str,
    provider: str = "",
    model: str = "",
    fallback: bool = False,
) -> LLMExplanationOutput:
    """Parse LLM response into structured explanation output.

    Handles both JSON-formatted and plain text responses.
    """
    import json

    # Try to parse as JSON
    try:
        # Handle potential markdown code blocks
        text = content.strip()
        if text.startswith("```"):
            # Remove markdown code fences
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
        return LLMExplanationOutput(
            summary=data.get("summary", ""),
            reason=data.get("reason", ""),
            supporting_evidence=data.get("supporting_evidence", ""),
            uncertainty=data.get("uncertainty", ""),
            limitations=data.get("limitations", ""),
            model_used=model,
            provider=provider,
            fallback_used=fallback,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fall back to treating entire response as summary
    return LLMExplanationOutput(
        summary=content.strip() if content.strip() else "No explanation available.",
        reason="",
        supporting_evidence="",
        uncertainty="",
        limitations="Could not parse structured response from LLM.",
        model_used=model,
        provider=provider,
        fallback_used=fallback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback
# ─────────────────────────────────────────────────────────────────────────────


def _deterministic_fallback(request: ExplanationRequest) -> LLMExplanationOutput:
    """Produce a deterministic explanation when LLM is unavailable.

    Uses templates based on available data — no LLM required.
    """
    summary_parts = []
    reason_parts = []
    evidence_parts = []
    uncertainty_parts = []
    limitations_parts = [
        "This explanation was generated without LLM assistance.",
        "A language model was unavailable, so a template-based explanation was used.",
    ]

    # Build summary
    summary_parts.append(
        f"Exception {request.exception_id} involves a discrepancy "
        f"of {_format_paise(request.difference_paise)} "
        f"between expected ({_format_paise(request.expected_amount_paise)}) "
        f"and actual ({_format_paise(request.actual_amount_paise)}) settlement amounts."
    )

    if request.exception_type:
        summary_parts.append(f"The exception has been classified as '{request.exception_type}'.")

    # Build reason from evidence coverage
    if request.evidence_coverage == "FULLY_EXPLAINED":
        reason_parts.append(
            f"All {_format_paise(request.difference_paise)} of the discrepancy "
            f"is explained by {len(request.evidence_items)} evidence record(s)."
        )
    elif request.evidence_coverage == "PARTIALLY_EXPLAINED":
        reason_parts.append(
            f"Part of the discrepancy ({_format_paise(request.explained_amount_paise)}) "
            f"is explained. {_format_paise(request.remaining_difference_paise)} remains unexplained."
        )
    elif request.evidence_coverage == "UNEXPLAINED":
        reason_parts.append("The discrepancy could not be explained by available evidence.")
    elif request.evidence_coverage == "CONFLICTING":
        reason_parts.append("The available evidence contains conflicting information.")
    else:
        reason_parts.append("Evidence analysis status is not available.")

    # Evidence summary
    if request.evidence_items:
        type_counts: Dict[str, int] = {}
        for ev in request.evidence_items:
            type_counts[ev.record_type] = type_counts.get(ev.record_type, 0) + 1
        evidence_parts.append(
            f"Available evidence includes: "
            + ", ".join(f"{count} {rtype} record(s)" for rtype, count in type_counts.items())
            + "."
        )
    else:
        uncertainty_parts.append("No evidence records are available.")

    # Uncertainty
    if request.remaining_difference_paise and request.remaining_difference_paise > 0:
        uncertainty_parts.append(
            f"{_format_paise(request.remaining_difference_paise)} of the discrepancy remains unexplained."
        )
    if request.classification_confidence is not None and request.classification_confidence < 0.5:
        uncertainty_parts.append(
            f"Classification confidence is low ({request.classification_confidence:.0%})."
        )

    return LLMExplanationOutput(
        summary=" ".join(summary_parts),
        reason=" ".join(reason_parts),
        supporting_evidence=" ".join(evidence_parts),
        uncertainty=" ".join(uncertainty_parts) if uncertainty_parts else "No significant uncertainties.",
        limitations=" ".join(limitations_parts),
        model_used="deterministic-template",
        provider="none",
        fallback_used=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Service
# ─────────────────────────────────────────────────────────────────────────────


class LLMExplanationService:
    """Service that produces natural-language explanations of financial exceptions.

    Uses the LLM to rephrase already-computed structured data into
    human-readable explanations. Falls back to deterministic templates
    when the LLM is unavailable.

    IMPORTANT: This service does NOT calculate financial state.
    All financial values are pre-computed by Phases 2-6.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[LLMConfig] = None,
        logger: Optional[LLMLogger] = None,
    ):
        """Initialize the explanation service.

        Args:
            provider: LLM provider. If None, uses deterministic fallback.
            config: LLM config for retry settings.
            logger: Observability logger.
        """
        self._provider = provider
        self._config = config or LLMConfig.from_env()
        self._logger = logger or LLMLogger("llm.explanation")

        # Wrap provider with retry if available
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
        """Log retry attempts."""
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

    async def explain(self, request: ExplanationRequest) -> LLMExplanationOutput:
        """Produce an explanation of the given financial exception.

        If the LLM is available, uses it to generate a natural language explanation.
        If the LLM is unavailable, uses deterministic templates.

        Args:
            request: Structured explanation input from Phases 2-6.

        Returns:
            Structured explanation output.
        """
        if self._executor is None:
            self._logger.log_provider_unavailable(
                provider="none",
                model="",
                reason="No LLM provider configured — using deterministic fallback",
            )
            return _deterministic_fallback(request)

        model = ""
        provider_name = ""
        if self._provider:
            provider_name = self._provider.provider_name
            cfg = self._config.get_provider_config()
            model = getattr(cfg, "model", "")

        # Build prompt
        user_prompt = build_explanation_prompt(request)

        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=EXPLANATION_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            metadata={
                "workflow_id": request.workflow_id or "",
                "exception_id": request.exception_id,
                "service": "explanation",
            },
        )

        self._logger.log_request_start(
            provider=provider_name,
            model=model,
            workflow_id=request.workflow_id,
            exception_id=request.exception_id,
            metadata={"service": "explanation"},
        )

        try:
            response: LLMResponse = await self._executor.generate(llm_request)

            self._logger.log_request_success(
                provider=provider_name,
                model=model,
                duration_ms=response.metadata.get("elapsed_ms", 0.0),
                tokens_used=response.usage.get("total_tokens"),
                finish_reason=response.finish_reason,
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
            )

            return _parse_explanation_response(
                response.content,
                provider=provider_name,
                model=response.model or model,
                fallback=False,
            )

        except LLMProviderError as e:
            self._logger.log_request_error(
                provider=provider_name,
                model=model,
                duration_ms=0.0,
                error_type=type(e).__name__,
                error_message=str(e),
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
            )
            # Fall back to deterministic explanation
            result = _deterministic_fallback(request)
            result.limitations = (
                f"LLM unavailable ({type(e).__name__}). "
                f"Using deterministic template. {result.limitations}"
            )
            return result

        except Exception as e:
            self._logger.log_request_error(
                provider=provider_name,
                model=model,
                duration_ms=0.0,
                error_type=type(e).__name__,
                error_message=str(e),
                workflow_id=request.workflow_id,
                exception_id=request.exception_id,
            )
            result = _deterministic_fallback(request)
            result.limitations = (
                f"Unexpected error ({type(e).__name__}). "
                f"Using deterministic template. {result.limitations}"
            )
            return result

    async def health_check(self):
        """Check explanation service health."""
        if self._executor is None:
            from app.llm.providers.base import LLMHealthStatus
            return LLMHealthStatus(
                provider="none",
                healthy=True,
                model="deterministic-template",
                details={"message": "LLM unavailable — deterministic fallback active"},
            )
        return await self._executor.health_check()

    async def close(self):
        """Close the underlying provider."""
        if self._executor:
            await self._executor.close()
