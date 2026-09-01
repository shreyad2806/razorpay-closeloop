"""
LLM Reviewer Assistant Service for Razorpay CloseLoop Phase 12F.

Uses the LLM to generate a structured reviewer briefing that reduces
cognitive load for human reviewers without replacing their judgment.

The LLM may:
- Summarize the exception and its context
- Explain evidence and its implications
- Describe candidate resolutions and why they were proposed
- Highlight what prevents automation
- Suggest what the reviewer should verify

The LLM must NOT:
- Make the review decision
- Override guardrail outcomes
- Execute any financial action
- Replace the human reviewer
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


class CandidateInfo(BaseModel):
    """Simplified resolution candidate for the briefing."""

    resolution_type: str = Field(default="", description="Proposed resolution type")
    source: str = Field(default="", description="Where candidate came from")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    adjustment_paise: Optional[int] = Field(default=None, description="Proposed adjustment")
    description: str = Field(default="", description="What the resolution proposes")
    evidence_compatible: bool = Field(default=True, description="Whether evidence supports it")


class GuardrailInfo(BaseModel):
    """Simplified guardrail decision for the briefing."""

    decision: str = Field(default="", description="AUTO, HUMAN_REVIEW, UNRESOLVED")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    risk_category: Optional[str] = Field(default=None, description="Risk level")
    reasons: List[str] = Field(default_factory=list, description="Decision reasons")
    exposure_paise: Optional[int] = Field(default=None, description="Financial exposure")


class ReviewerBriefingRequest(BaseModel):
    """Input to the reviewer assistant.

    Aggregates all workflow context into a single request for the LLM
    to produce a structured reviewer briefing.
    """

    # Exception context
    exception_id: str = Field(..., description="Exception being reviewed")
    case_id: Optional[str] = Field(default=None, description="Case ID")
    exception_type: Optional[str] = Field(default=None, description="Classified exception type")

    # Financial discrepancy
    expected_amount_paise: Optional[int] = Field(default=None, description="Expected settlement")
    actual_amount_paise: Optional[int] = Field(default=None, description="Actual settlement")
    difference_paise: Optional[int] = Field(default=None, description="Discrepancy")

    # Evidence
    evidence_summary: str = Field(default="", description="Summary of available evidence")
    evidence_record_count: int = Field(default=0, description="Number of evidence records")
    evidence_coverage: Optional[str] = Field(
        default=None, description="FULLY_EXPLAINED, PARTIALLY_EXPLAINED, UNEXPLAINED, CONFLICTING"
    )
    explained_amount_paise: Optional[int] = Field(default=None, description="Amount explained")
    remaining_difference_paise: Optional[int] = Field(default=None, description="Unexplained amount")
    conflicts: List[str] = Field(default_factory=list, description="Evidence conflicts")
    missing_evidence: List[str] = Field(default_factory=list, description="Missing evidence types")

    # Classification
    classification_type: Optional[str] = Field(default=None, description="Exception type")
    classification_confidence: Optional[float] = Field(default=None, description="Classification confidence")
    classification_agreement: Optional[bool] = Field(default=None, description="Deterministic vs ML agree")
    classification_note: Optional[str] = Field(default=None, description="Disagreement note")

    # Similar cases
    similar_case_count: int = Field(default=0, description="Number of similar cases found")
    similar_case_summary: str = Field(default="", description="Summary from Phase 12E")
    highest_similarity: Optional[float] = Field(default=None, description="Highest similarity score")

    # Candidates
    candidates: List[CandidateInfo] = Field(default_factory=list, description="Resolution candidates")
    selected_candidate_type: Optional[str] = Field(default=None, description="Selected candidate type")
    selected_candidate_adjustment: Optional[int] = Field(default=None, description="Selected adjustment")

    # Guardrails
    guardrail: Optional[GuardrailInfo] = Field(default=None, description="Guardrail decision")
    ml_confidence: Optional[float] = Field(default=None, description="ML confidence")
    risk: Optional[str] = Field(default=None, description="Risk category")

    # Verification
    verification_status: Optional[str] = Field(default=None, description="NOT_REQUIRED, PENDING, VERIFIED, FAILED")

    # Metadata
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


# ─────────────────────────────────────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────────────────────────────────────


class ReviewerBriefingOutput(BaseModel):
    """Structured reviewer briefing — 10-point format."""

    what_happened: str = Field(default="", description="1. What happened?")
    why_it_happened: str = Field(default="", description="2. Why does it appear to have happened?")
    supporting_evidence: str = Field(default="", description="3. What evidence supports this?")
    missing_evidence: str = Field(default="", description="4. What evidence is missing?")
    conflicts: str = Field(default="", description="5. Are there conflicts?")
    candidate_resolutions: str = Field(default="", description="6. What candidate resolutions exist?")
    why_candidates: str = Field(default="", description="7. Why were candidates proposed?")
    system_recommendation: str = Field(default="", description="8. What does the system recommend?")
    automation_barriers: str = Field(default="", description="9. What prevents automation?")
    reviewer_checklist: str = Field(default="", description="10. What should the reviewer verify?")
    model_used: str = Field(default="", description="Model that produced the briefing")
    provider: str = Field(default="", description="Provider used")
    fallback_used: bool = Field(default=False, description="Whether deterministic fallback was used")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

REVIEWER_ASSISTANT_PROMPT = """You are a financial review briefing assistant.

Your role:
- Prepare a structured briefing for a human reviewer evaluating a financial exception
- Make the reviewer's job easier by organizing information clearly
- Highlight what needs human attention

You MUST:
- Use ONLY the information provided in the briefing request
- Clearly distinguish facts from system recommendations
- Note uncertainty honestly
- Respect guardrail decisions — do not override them
- Present candidates without endorsing any single one

You MUST NOT:
- Make the final review decision
- Execute any financial action
- Override guardrail decisions
- Replace the human reviewer's judgment
- Invent information not provided

Format your response as JSON with these fields:
- what_happened: Clear description of the exception
- why_it_happened: Probable cause based on evidence
- supporting_evidence: Key evidence supporting the assessment
- missing_evidence: What evidence is absent and why it matters
- conflicts: Any conflicts in the evidence or analysis
- candidate_resolutions: What resolutions are being considered
- why_candidates: Why each candidate was proposed
- system_recommendation: What the automated system recommends and why
- automation_barriers: Why this case requires human review
- reviewer_checklist: Specific items the reviewer should verify"""


# ─────────────────────────────────────────────────────────────────────────────
# Build User Prompt
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(amount: Optional[int]) -> str:
    if amount is None:
        return "Not provided"
    return f"₹{amount / 100:,.2f} ({amount} paise)"


def build_reviewer_briefing_prompt(request: ReviewerBriefingRequest) -> str:
    """Build the user prompt from structured workflow data."""
    parts = [
        "# Human Review Briefing Request",
        "",
        f"Exception ID: {request.exception_id}",
    ]

    if request.case_id:
        parts.append(f"Case ID: {request.case_id}")
    if request.exception_type:
        parts.append(f"Exception Type: {request.exception_type}")

    # Financial context
    parts.append("")
    parts.append("## Financial Discrepancy")
    parts.append(f"Expected Amount: {_fmt(request.expected_amount_paise)}")
    parts.append(f"Actual Amount: {_fmt(request.actual_amount_paise)}")
    parts.append(f"Discrepancy: {_fmt(request.difference_paise)}")

    # Evidence
    parts.append("")
    parts.append("## Evidence")
    parts.append(f"Evidence Records: {request.evidence_record_count}")
    if request.evidence_coverage:
        parts.append(f"Coverage: {request.evidence_coverage}")
    if request.explained_amount_paise is not None:
        parts.append(f"Explained: {_fmt(request.explained_amount_paise)}")
    if request.remaining_difference_paise is not None:
        parts.append(f"Remaining Unexplained: {_fmt(request.remaining_difference_paise)}")
    if request.evidence_summary:
        parts.append(f"Summary: {request.evidence_summary}")
    if request.conflicts:
        parts.append("Conflicts:")
        for c in request.conflicts:
            parts.append(f"  - {c}")
    if request.missing_evidence:
        parts.append("Missing Evidence:")
        for m in request.missing_evidence:
            parts.append(f"  - {m}")

    # Classification
    parts.append("")
    parts.append("## Classification")
    if request.classification_type:
        parts.append(f"Type: {request.classification_type}")
    if request.classification_confidence is not None:
        parts.append(f"Confidence: {request.classification_confidence:.1%}")
    if request.classification_agreement is not None:
        parts.append(f"Deterministic/ML Agreement: {'Yes' if request.classification_agreement else 'No'}")
    if request.classification_note:
        parts.append(f"Note: {request.classification_note}")

    # Similar cases
    if request.similar_case_count > 0:
        parts.append("")
        parts.append("## Similar Historical Cases")
        parts.append(f"Cases Found: {request.similar_case_count}")
        if request.highest_similarity is not None:
            parts.append(f"Highest Similarity: {request.highest_similarity:.1%}")
        if request.similar_case_summary:
            parts.append(f"Summary: {request.similar_case_summary}")

    # Candidates
    if request.candidates:
        parts.append("")
        parts.append("## Resolution Candidates")
        for i, c in enumerate(request.candidates, 1):
            parts.append(f"\n### Candidate {i}: {c.resolution_type}")
            parts.append(f"Source: {c.source}")
            if c.confidence is not None:
                parts.append(f"Confidence: {c.confidence:.1%}")
            if c.adjustment_paise is not None:
                parts.append(f"Adjustment: {_fmt(c.adjustment_paise)}")
            if c.description:
                parts.append(f"Description: {c.description}")
            parts.append(f"Evidence Compatible: {'Yes' if c.evidence_compatible else 'No'}")

    if request.selected_candidate_type:
        parts.append(f"\nSelected Candidate: {request.selected_candidate_type}")
        if request.selected_candidate_adjustment is not None:
            parts.append(f"Selected Adjustment: {_fmt(request.selected_candidate_adjustment)}")

    # Guardrails
    if request.guardrail:
        parts.append("")
        parts.append("## Guardrail Decision")
        parts.append(f"Decision: {request.guardrail.decision}")
        if request.guardrail.confidence is not None:
            parts.append(f"Confidence: {request.guardrail.confidence:.1%}")
        if request.guardrail.risk_category:
            parts.append(f"Risk: {request.guardrail.risk_category}")
        if request.guardrail.exposure_paise is not None:
            parts.append(f"Exposure: {_fmt(request.guardrail.exposure_paise)}")
        if request.guardrail.reasons:
            parts.append("Reasons:")
            for r in request.guardrail.reasons:
                parts.append(f"  - {r}")

    if request.ml_confidence is not None:
        parts.append(f"ML Confidence: {request.ml_confidence:.1%}")
    if request.risk:
        parts.append(f"Risk Level: {request.risk}")

    # Verification
    if request.verification_status:
        parts.append("")
        parts.append(f"Verification Status: {request.verification_status}")

    parts.append("")
    parts.append("Please prepare a structured briefing for the human reviewer.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Response
# ─────────────────────────────────────────────────────────────────────────────


def _parse_briefing_response(
    content: str,
    provider: str = "",
    model: str = "",
    fallback: bool = False,
) -> ReviewerBriefingOutput:
    """Parse LLM response into structured reviewer briefing."""
    import json

    try:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
        return ReviewerBriefingOutput(
            what_happened=data.get("what_happened", ""),
            why_it_happened=data.get("why_it_happened", ""),
            supporting_evidence=data.get("supporting_evidence", ""),
            missing_evidence=data.get("missing_evidence", ""),
            conflicts=data.get("conflicts", ""),
            candidate_resolutions=data.get("candidate_resolutions", ""),
            why_candidates=data.get("why_candidates", ""),
            system_recommendation=data.get("system_recommendation", ""),
            automation_barriers=data.get("automation_barriers", ""),
            reviewer_checklist=data.get("reviewer_checklist", ""),
            model_used=model,
            provider=provider,
            fallback_used=fallback,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ReviewerBriefingOutput(
        what_happened=content.strip() if content.strip() else "No briefing available.",
        model_used=model,
        provider=provider,
        fallback_used=fallback,
        reviewer_checklist="Could not parse structured briefing. Please review all available data manually.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback
# ─────────────────────────────────────────────────────────────────────────────


def _reviewer_deterministic_fallback(request: ReviewerBriefingRequest) -> ReviewerBriefingOutput:
    """Produce deterministic reviewer briefing when LLM is unavailable."""
    what_parts: List[str] = []
    why_parts: List[str] = []
    evidence_parts: List[str] = []
    missing_parts: List[str] = []
    conflict_parts: List[str] = []
    candidate_parts: List[str] = []
    why_candidate_parts: List[str] = []
    recommendation_parts: List[str] = []
    barrier_parts: List[str] = []
    checklist_parts: List[str] = []

    # 1. What happened
    what_parts.append(
        f"Exception {request.exception_id} represents a discrepancy of "
        f"{_fmt(request.difference_paise)} between expected ({_fmt(request.expected_amount_paise)}) "
        f"and actual ({_fmt(request.actual_amount_paise)}) settlement amounts."
    )
    if request.exception_type:
        what_parts.append(f"Classified as: {request.exception_type}.")

    # 2. Why
    if request.evidence_coverage == "FULLY_EXPLAINED":
        why_parts.append("All of the discrepancy is accounted for by available evidence.")
    elif request.evidence_coverage == "PARTIALLY_EXPLAINED":
        why_parts.append(
            f"Part of the discrepancy ({_fmt(request.explained_amount_paise)}) is explained, "
            f"but {_fmt(request.remaining_difference_paise)} remains unexplained."
        )
    elif request.evidence_coverage == "CONFLICTING":
        why_parts.append("The available evidence contains conflicting information.")
    elif request.evidence_coverage == "UNEXPLAINED":
        why_parts.append("The discrepancy could not be explained by available evidence.")
    else:
        why_parts.append("Evidence analysis status is not available.")

    # 3. Supporting evidence
    if request.evidence_record_count > 0:
        evidence_parts.append(f"{request.evidence_record_count} evidence record(s) available.")
    else:
        evidence_parts.append("No evidence records available.")

    # 4. Missing evidence
    if request.missing_evidence:
        for m in request.missing_evidence:
            missing_parts.append(f"- {m}")
    else:
        missing_parts.append("No missing evidence indicators.")

    # 5. Conflicts
    if request.conflicts:
        for c in request.conflicts:
            conflict_parts.append(f"- {c}")
    else:
        conflict_parts.append("No conflicts detected.")

    # 6 & 7. Candidates
    if request.candidates:
        for i, c in enumerate(request.candidates, 1):
            candidate_parts.append(
                f"{i}. {c.resolution_type} (source: {c.source})"
                + (f", confidence: {c.confidence:.0%}" if c.confidence else "")
                + (f", adjustment: {_fmt(c.adjustment_paise)}" if c.adjustment_paise else "")
            )
            why_candidate_parts.append(
                f"- {c.resolution_type}: {c.description or 'No description provided.'}"
            )
    else:
        candidate_parts.append("No resolution candidates generated.")
        why_candidate_parts.append("No candidates to explain.")

    # 8. System recommendation
    if request.guardrail:
        recommendation_parts.append(f"Guardrail decision: {request.guardrail.decision}.")
        if request.guardrail.confidence is not None:
            recommendation_parts.append(f"Confidence: {request.guardrail.confidence:.0%}.")
        if request.guardrail.reasons:
            recommendation_parts.append("Reasons: " + "; ".join(request.guardrail.reasons) + ".")
    if request.selected_candidate_type:
        recommendation_parts.append(f"Selected candidate: {request.selected_candidate_type}.")

    # 9. Automation barriers
    if request.guardrail and request.guardrail.decision != "AUTO":
        barrier_parts.append(
            f"Guardrail decision is '{request.guardrail.decision}' — requires human review."
        )
    if request.conflicts:
        barrier_parts.append("Evidence conflicts require human judgment.")
    if request.remaining_difference_paise and request.remaining_difference_paise > 0:
        barrier_parts.append("Unexplained discrepancy amount.")
    if request.classification_agreement is False:
        barrier_parts.append("Deterministic and ML classifications disagree.")
    if not barrier_parts:
        barrier_parts.append("No specific automation barriers identified — review may proceed.")

    # 10. Reviewer checklist
    checklist_parts.append("Verify the financial discrepancy amount matches records.")
    if request.evidence_coverage in ("PARTIALLY_EXPLAINED", "UNEXPLAINED"):
        checklist_parts.append("Investigate unexplained portion of the discrepancy.")
    if request.conflicts:
        checklist_parts.append("Resolve evidence conflicts.")
    if request.missing_evidence:
        checklist_parts.append("Determine if missing evidence affects the assessment.")
    if request.guardrail and request.guardrail.decision == "HUMAN_REVIEW":
        checklist_parts.append("Review guardrail concerns before approving.")
    if request.classification_agreement is False:
        checklist_parts.append("Verify the correct exception classification.")
    checklist_parts.append("Confirm the proposed resolution matches the actual issue.")
    checklist_parts.append("Ensure the financial adjustment is appropriate.")

    return ReviewerBriefingOutput(
        what_happened=" ".join(what_parts),
        why_it_happened=" ".join(why_parts),
        supporting_evidence=" ".join(evidence_parts),
        missing_evidence="\n".join(missing_parts),
        conflicts="\n".join(conflict_parts),
        candidate_resolutions="\n".join(candidate_parts),
        why_candidates="\n".join(why_candidate_parts),
        system_recommendation=" ".join(recommendation_parts) if recommendation_parts else "No system recommendation.",
        automation_barriers=" ".join(barrier_parts),
        reviewer_checklist="\n".join(checklist_parts),
        model_used="deterministic-template",
        provider="none",
        fallback_used=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer Assistant Service
# ─────────────────────────────────────────────────────────────────────────────


class LLMReviewerAssistantService:
    """Service that produces structured reviewer briefings.

    Uses the LLM to organize workflow context into a 10-point briefing
    for human reviewers. Falls back to deterministic templates when LLM
    is unavailable.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[LLMConfig] = None,
        logger: Optional[LLMLogger] = None,
    ):
        self._provider = provider
        self._config = config or LLMConfig.from_env()
        self._logger = logger or LLMLogger("llm.reviewer_assistant")

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

    async def generate_briefing(self, request: ReviewerBriefingRequest) -> ReviewerBriefingOutput:
        """Generate a reviewer briefing.

        If LLM is available, uses it. Otherwise, uses deterministic templates.
        """
        if self._executor is None:
            self._logger.log_provider_unavailable(
                provider="none", model="",
                reason="No LLM provider configured — using deterministic fallback",
            )
            return _reviewer_deterministic_fallback(request)

        provider_name = self._provider.provider_name if self._provider else ""
        model = ""
        if self._provider:
            cfg = self._config.get_provider_config()
            model = getattr(cfg, "model", "")

        user_prompt = build_reviewer_briefing_prompt(request)

        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=REVIEWER_ASSISTANT_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            metadata={
                "workflow_id": request.workflow_id or "",
                "exception_id": request.exception_id,
                "service": "reviewer_assistant",
            },
        )

        self._logger.log_request_start(
            provider=provider_name, model=model,
            workflow_id=request.workflow_id,
            exception_id=request.exception_id,
            metadata={"service": "reviewer_assistant"},
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

            return _parse_briefing_response(
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
            result = _reviewer_deterministic_fallback(request)
            result.reviewer_checklist = (
                f"LLM unavailable ({type(e).__name__}). Using template briefing.\n"
                + result.reviewer_checklist
            )
            return result

        except Exception as e:
            self._logger.log_request_error(
                provider=provider_name, model=model, duration_ms=0.0,
                error_type=type(e).__name__, error_message=str(e),
                workflow_id=request.workflow_id, exception_id=request.exception_id,
            )
            result = _reviewer_deterministic_fallback(request)
            result.reviewer_checklist = (
                f"Unexpected error ({type(e).__name__}). Using template briefing.\n"
                + result.reviewer_checklist
            )
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
