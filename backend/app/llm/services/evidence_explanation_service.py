"""
LLM Evidence Explanation Service for Razorpay CloseLoop Phase 12D.

Uses the LLM to explain the financial evidence layer in human-readable language.

The LLM receives structured evidence produced by Phase 3 and produces
a natural-language explanation of:
- What happened
- Which financial records are involved
- How those records explain the discrepancy
- Whether evidence is complete
- Whether there is conflicting evidence
- What remains uncertain

IMPORTANT:
- The LLM receives evidence — it does NOT retrieve arbitrary records
- Evidence retrieval remains owned by Phase 3 / MCP / internal services
- The LLM does NOT change amounts
- The LLM does NOT invent evidence
- The LLM does NOT declare exceptions resolved
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
# Evidence Explanation Input
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceRecordInfo(BaseModel):
    """Simplified evidence record for explanation input."""

    record_id: str = Field(..., description="Record identifier")
    entity_type: str = Field(..., description="PAYMENT, SETTLEMENT, REFUND, FEE, TAX, ADJUSTMENT")
    relationship: str = Field(..., description="PRIMARY_RECORD, CALCULATION_COMPONENT, etc.")
    amount_paise: int = Field(..., description="Amount in paise")
    status: Optional[str] = Field(default=None, description="Record status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ConflictInfo(BaseModel):
    """Simplified conflict for explanation input."""

    conflict_type: str = Field(..., description="Type of conflict")
    description: str = Field(..., description="What the conflict is")
    affected_records: List[str] = Field(default_factory=list, description="Records involved")


class MissingEvidenceInfo(BaseModel):
    """Simplified missing evidence for explanation input."""

    entity_type: str = Field(..., description="What type of record is missing")
    expected: bool = Field(..., description="Whether this record type is expected")
    reason: str = Field(..., description="Why it's considered missing")


class EvidenceExplanationRequest(BaseModel):
    """Input to the evidence explanation service.

    Contains structured evidence from Phase 3.
    The LLM will rephrase into natural language.
    """

    exception_id: str = Field(..., description="Exception being explained")
    case_id: Optional[str] = Field(default=None, description="Case reference")
    payment_id: Optional[str] = Field(default=None, description="Payment reference")
    exception_type: Optional[str] = Field(default=None, description="Classified exception type")

    # Financial context
    expected_amount_paise: Optional[int] = Field(default=None, description="Expected settlement")
    actual_amount_paise: Optional[int] = Field(default=None, description="Actual settlement")
    difference_paise: Optional[int] = Field(default=None, description="Discrepancy")

    # Evidence records
    payment: Optional[EvidenceRecordInfo] = Field(default=None, description="Payment record")
    settlements: List[EvidenceRecordInfo] = Field(default_factory=list, description="Settlement records")
    refunds: List[EvidenceRecordInfo] = Field(default_factory=list, description="Refund records")
    fees: List[EvidenceRecordInfo] = Field(default_factory=list, description="Fee records")
    taxes: List[EvidenceRecordInfo] = Field(default_factory=list, description="Tax records")
    adjustments: List[EvidenceRecordInfo] = Field(default_factory=list, description="Adjustment records")

    # Financial summary
    total_settlement_paise: int = Field(default=0, description="Sum of settlements")
    total_refund_paise: int = Field(default=0, description="Sum of refunds")
    total_fee_paise: int = Field(default=0, description="Sum of fees")
    total_tax_paise: int = Field(default=0, description="Sum of taxes")
    total_adjustment_paise: int = Field(default=0, description="Net adjustments")

    # Missing evidence
    missing_evidence: List[MissingEvidenceInfo] = Field(default_factory=list, description="Missing records")

    # Conflicts
    conflicts: List[ConflictInfo] = Field(default_factory=list, description="Detected conflicts")

    # Evidence quality
    evidence_record_count: int = Field(default=0, description="Total evidence records")
    evidence_link_count: int = Field(default=0, description="Number of evidence links")

    # Metadata
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @classmethod
    def from_evidence_package(cls, pkg: Any) -> "EvidenceExplanationRequest":
        """Build from an EvidencePackage object.

        Accepts any dict-like or EvidencePackage object.
        """
        # Handle dict or EvidencePackage
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _record(rec) -> Optional[EvidenceRecordInfo]:
            if rec is None:
                return None
            return EvidenceRecordInfo(
                record_id=_get(rec, "record_id", ""),
                entity_type=_get(rec, "entity_type", ""),
                relationship=_get(rec, "relationship", ""),
                amount_paise=_get(rec, "amount", 0),
                status=_get(rec, "status"),
                metadata=_get(rec, "metadata") or {},
            )

        def _records(recs) -> List[EvidenceRecordInfo]:
            return [r for r in (_record(x) for x in (recs or [])) if r is not None]

        settlements_raw = _get(pkg, "settlements", [])
        refunds_raw = _get(pkg, "refunds", [])
        fees_raw = _get(pkg, "fees", [])
        taxes_raw = _get(pkg, "taxes", [])
        adjustments_raw = _get(pkg, "adjustments", [])

        missing_raw = _get(pkg, "missing_evidence", []) or []
        conflicts_raw = _get(pkg, "conflicts", []) or []

        return cls(
            exception_id=_get(pkg, "exception_id", ""),
            case_id=_get(pkg, "case_id"),
            payment_id=_get(pkg, "payment_id"),
            exception_type=_get(pkg, "exception_type"),
            expected_amount_paise=_get(pkg, "expected_amount"),
            actual_amount_paise=_get(pkg, "actual_amount"),
            difference_paise=_get(pkg, "difference"),
            payment=_record(_get(pkg, "payment")),
            settlements=_records(settlements_raw),
            refunds=_records(refunds_raw),
            fees=_records(fees_raw),
            taxes=_records(taxes_raw),
            adjustments=_records(adjustments_raw),
            total_settlement_paise=_get(pkg, "total_settlement_amount", 0),
            total_refund_paise=_get(pkg, "total_refund_amount", 0),
            total_fee_paise=_get(pkg, "total_fee_amount", 0),
            total_tax_paise=_get(pkg, "total_tax_amount", 0),
            total_adjustment_paise=_get(pkg, "total_adjustment_amount", 0),
            missing_evidence=[
                MissingEvidenceInfo(
                    entity_type=_get(m, "entity_type", ""),
                    expected=_get(m, "expected", False),
                    reason=_get(m, "reason", ""),
                )
                for m in missing_raw
            ],
            conflicts=[
                ConflictInfo(
                    conflict_type=_get(c, "conflict_type", ""),
                    description=_get(c, "description", ""),
                    affected_records=_get(c, "affected_records", []) or [],
                )
                for c in conflicts_raw
            ],
            evidence_record_count=(
                (1 if _get(pkg, "payment") else 0)
                + len(settlements_raw)
                + len(refunds_raw)
                + len(fees_raw)
                + len(taxes_raw)
                + len(adjustments_raw)
            ),
            evidence_link_count=_get(pkg, "evidence_link_count", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Explanation Output
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceExplanationOutput(BaseModel):
    """Structured output from the evidence explanation service."""

    summary: str = Field(default="", description="One-paragraph overview")
    financial_events: str = Field(default="", description="What financial events occurred")
    evidence_chain: str = Field(default="", description="How evidence connects to the discrepancy")
    explained_amount_paise: int = Field(default=0, description="Amount explained by evidence")
    unexplained_amount_paise: int = Field(default=0, description="Amount not explained")
    conflicts: str = Field(default="", description="Description of any conflicts found")
    missing_evidence: str = Field(default="", description="What evidence is missing and why it matters")
    uncertainty: str = Field(default="", description="What remains uncertain")
    completeness: str = Field(default="", description="Assessment of evidence completeness")
    model_used: str = Field(default="", description="Model that produced the explanation")
    provider: str = Field(default="", description="Provider used")
    fallback_used: bool = Field(default=False, description="Whether deterministic fallback was used")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_EXPLANATION_SYSTEM_PROMPT = """You are a financial evidence explanation assistant.

Your role:
- Explain what financial evidence shows about a payment exception
- Make evidence chains understandable for human reviewers
- Be precise about amounts — use the exact paise values provided

You MUST:
- Use ONLY the evidence records provided
- Reference specific record IDs when discussing them
- Accurately represent the amounts from the evidence
- Note when evidence is missing or conflicting
- Assess completeness honestly

You MUST NOT:
- Invent evidence or financial records that are not provided
- Change any financial amounts
- Calculate new totals — use the provided totals
- Declare the exception resolved or unresolved
- Speculate about records not in the evidence
- Override conflict detection results

Format your response as JSON with these fields:
- summary: One paragraph explaining what the evidence shows
- financial_events: What financial events occurred
- evidence_chain: How the evidence connects to the discrepancy
- explained_amount_paise: Amount explained by evidence (integer)
- unexplained_amount_paise: Amount not explained (integer)
- conflicts: Description of conflicts found (or "None")
- missing_evidence: What's missing and why it matters (or "None")
- uncertainty: What remains uncertain
- completeness: Assessment of evidence completeness"""


# ─────────────────────────────────────────────────────────────────────────────
# Build User Prompt
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(amount: Optional[int]) -> str:
    """Format paise as readable string."""
    if amount is None:
        return "Not provided"
    return f"₹{amount / 100:,.2f} ({amount} paise)"


def _fmt_record(rec: EvidenceRecordInfo) -> str:
    """Format a single evidence record for the prompt."""
    parts = [
        f"  - [{rec.entity_type}] {rec.record_id}",
        f"    Relationship: {rec.relationship}",
        f"    Amount: {_fmt(rec.amount_paise)}",
    ]
    if rec.status:
        parts.append(f"    Status: {rec.status}")
    return "\n".join(parts)


def build_evidence_explanation_prompt(request: EvidenceExplanationRequest) -> str:
    """Build the user prompt from structured evidence data."""
    parts = [
        "# Financial Evidence Explanation Request",
        "",
        f"Exception ID: {request.exception_id}",
    ]

    if request.case_id:
        parts.append(f"Case ID: {request.case_id}")
    if request.payment_id:
        parts.append(f"Payment ID: {request.payment_id}")
    if request.exception_type:
        parts.append(f"Exception Type: {request.exception_type}")

    # Financial context
    parts.append("")
    parts.append("## Financial Context")
    parts.append(f"Expected Amount: {_fmt(request.expected_amount_paise)}")
    parts.append(f"Actual Amount: {_fmt(request.actual_amount_paise)}")
    parts.append(f"Discrepancy: {_fmt(request.difference_paise)}")

    # Totals
    parts.append("")
    parts.append("## Evidence Totals (deterministic)")
    parts.append(f"Total Settlement: {_fmt(request.total_settlement_paise)}")
    parts.append(f"Total Refunds: {_fmt(request.total_refund_paise)}")
    parts.append(f"Total Fees: {_fmt(request.total_fee_paise)}")
    parts.append(f"Total Taxes: {_fmt(request.total_tax_paise)}")
    parts.append(f"Total Adjustments: {_fmt(request.total_adjustment_paise)}")

    # Evidence records
    all_records: List[str] = []
    if request.payment:
        all_records.append(f"\n### Payment Record\n{_fmt_record(request.payment)}")

    for label, records in [
        ("Settlements", request.settlements),
        ("Refunds", request.refunds),
        ("Fees", request.fees),
        ("Taxes", request.taxes),
        ("Adjustments", request.adjustments),
    ]:
        if records:
            all_records.append(f"\n### {label} ({len(records)} record(s))")
            for rec in records:
                all_records.append(_fmt_record(rec))

    if all_records:
        parts.append("")
        parts.append("## Evidence Records")
        parts.extend(all_records)

    parts.append(f"\nTotal Evidence Records: {request.evidence_record_count}")
    parts.append(f"Evidence Links: {request.evidence_link_count}")

    # Missing evidence
    if request.missing_evidence:
        parts.append("")
        parts.append("## Missing Evidence")
        for m in request.missing_evidence:
            expected_str = "expected" if m.expected else "optional"
            parts.append(f"  - [{m.entity_type}] ({expected_str}): {m.reason}")

    # Conflicts
    if request.conflicts:
        parts.append("")
        parts.append("## Structural Conflicts")
        for c in request.conflicts:
            parts.append(f"  - [{c.conflict_type}] {c.description}")
            if c.affected_records:
                parts.append(f"    Affected: {', '.join(c.affected_records)}")

    parts.append("")
    parts.append("Please explain what this evidence shows about the exception.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Response
# ─────────────────────────────────────────────────────────────────────────────


def _parse_evidence_response(
    content: str,
    provider: str = "",
    model: str = "",
    fallback: bool = False,
) -> EvidenceExplanationOutput:
    """Parse LLM response into structured evidence explanation."""
    import json

    try:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
        return EvidenceExplanationOutput(
            summary=data.get("summary", ""),
            financial_events=data.get("financial_events", ""),
            evidence_chain=data.get("evidence_chain", ""),
            explained_amount_paise=data.get("explained_amount_paise", 0),
            unexplained_amount_paise=data.get("unexplained_amount_paise", 0),
            conflicts=data.get("conflicts", ""),
            missing_evidence=data.get("missing_evidence", ""),
            uncertainty=data.get("uncertainty", ""),
            completeness=data.get("completeness", ""),
            model_used=model,
            provider=provider,
            fallback_used=fallback,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return EvidenceExplanationOutput(
        summary=content.strip() if content.strip() else "No evidence explanation available.",
        model_used=model,
        provider=provider,
        fallback_used=fallback,
        completeness="Could not parse structured LLM response.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback
# ─────────────────────────────────────────────────────────────────────────────


def _evidence_deterministic_fallback(request: EvidenceExplanationRequest) -> EvidenceExplanationOutput:
    """Produce deterministic evidence explanation when LLM is unavailable."""
    summary_parts: List[str] = []
    events_parts: List[str] = []
    chain_parts: List[str] = []
    conflicts_parts: List[str] = []
    missing_parts: List[str] = []
    uncertainty_parts: List[str] = []
    limitations_parts: List[str] = [
        "This explanation was generated without LLM assistance.",
        "A template-based explanation was produced instead.",
    ]

    # Summary
    total_records = request.evidence_record_count
    summary_parts.append(
        f"Exception {request.exception_id} has {total_records} evidence record(s) "
        f"relating to a {_fmt(request.difference_paise)} discrepancy "
        f"between expected ({_fmt(request.expected_amount_paise)}) "
        f"and actual ({_fmt(request.actual_amount_paise)}) amounts."
    )

    # Financial events
    event_counts: Dict[str, int] = {}
    for rec_list_name, rec_list in [
        ("settlement", request.settlements),
        ("refund", request.refunds),
        ("fee", request.fees),
        ("tax", request.taxes),
        ("adjustment", request.adjustments),
    ]:
        if rec_list:
            event_counts[rec_list_name] = len(rec_list)
    if request.payment:
        event_counts["payment"] = 1

    if event_counts:
        events_parts.append(
            "Financial events include: "
            + ", ".join(f"{count} {rtype} record(s)" for rtype, count in event_counts.items())
            + "."
        )
    else:
        events_parts.append("No financial evidence records are available.")

    # Evidence chain
    explained = 0
    unexplained = 0
    for rec in request.settlements + request.refunds + request.fees + request.taxes + request.adjustments:
        if rec.relationship in ("PRIMARY_RECORD", "CALCULATION_COMPONENT", "SUPPORTING_EVIDENCE"):
            explained += rec.amount_paise
        elif rec.relationship == "CONFLICTING_EVIDENCE":
            pass  # Conflicting evidence doesn't explain

    if request.difference_paise is not None and request.difference_paise > 0:
        unexplained = max(0, request.difference_paise - explained)

    if explained > 0:
        chain_parts.append(
            f"Evidence explains {_fmt(explained)} of the {_fmt(request.difference_paise)} discrepancy."
        )
    if unexplained > 0:
        chain_parts.append(f"{_fmt(unexplained)} of the discrepancy remains unexplained.")

    # Conflicts
    if request.conflicts:
        for c in request.conflicts:
            conflicts_parts.append(f"[{c.conflict_type}] {c.description}")
    else:
        conflicts_parts.append("No structural conflicts detected.")

    # Missing evidence
    if request.missing_evidence:
        for m in request.missing_evidence:
            missing_parts.append(f"{m.entity_type}: {m.reason}")
    else:
        missing_parts.append("No missing evidence indicators.")

    # Uncertainty
    if unexplained > 0:
        uncertainty_parts.append(f"{_fmt(unexplained)} remains unexplained.")
    if request.conflicts:
        uncertainty_parts.append(f"{len(request.conflicts)} structural conflict(s) detected.")
    if not uncertainty_parts:
        uncertainty_parts.append("No significant uncertainties identified.")

    # Completeness
    if not request.conflicts and not request.missing_evidence and total_records > 0:
        completeness = "Evidence appears complete with no conflicts or missing records."
    elif request.conflicts and request.missing_evidence:
        completeness = "Evidence is incomplete with both conflicts and missing records."
    elif request.conflicts:
        completeness = "Evidence contains structural conflicts that require attention."
    elif request.missing_evidence:
        completeness = "Some expected evidence records are missing."
    else:
        completeness = "Evidence completeness cannot be assessed — no records available."

    return EvidenceExplanationOutput(
        summary=" ".join(summary_parts),
        financial_events=" ".join(events_parts),
        evidence_chain=" ".join(chain_parts),
        explained_amount_paise=explained,
        unexplained_amount_paise=unexplained,
        conflicts=" ".join(conflicts_parts),
        missing_evidence=" ".join(missing_parts),
        uncertainty=" ".join(uncertainty_parts),
        completeness=completeness,
        model_used="deterministic-template",
        provider="none",
        fallback_used=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Explanation Service
# ─────────────────────────────────────────────────────────────────────────────


class LLMEvidenceExplanationService:
    """Service that produces human-readable explanations of financial evidence.

    Uses the LLM to explain evidence produced by Phase 3.
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
        self._logger = logger or LLMLogger("llm.evidence_explanation")

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

    async def explain(self, request: EvidenceExplanationRequest) -> EvidenceExplanationOutput:
        """Produce an evidence explanation.

        If LLM is available, uses it. Otherwise, uses deterministic templates.
        """
        if self._executor is None:
            self._logger.log_provider_unavailable(
                provider="none",
                model="",
                reason="No LLM provider configured — using deterministic fallback",
            )
            return _evidence_deterministic_fallback(request)

        provider_name = self._provider.provider_name if self._provider else ""
        model = ""
        if self._provider:
            cfg = self._config.get_provider_config()
            model = getattr(cfg, "model", "")

        user_prompt = build_evidence_explanation_prompt(request)

        llm_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=EVIDENCE_EXPLANATION_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            metadata={
                "workflow_id": request.workflow_id or "",
                "exception_id": request.exception_id,
                "service": "evidence_explanation",
            },
        )

        self._logger.log_request_start(
            provider=provider_name,
            model=model,
            workflow_id=request.workflow_id,
            exception_id=request.exception_id,
            metadata={"service": "evidence_explanation"},
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

            return _parse_evidence_response(
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
            result = _evidence_deterministic_fallback(request)
            result.completeness = (
                f"LLM unavailable ({type(e).__name__}). {result.completeness}"
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
            result = _evidence_deterministic_fallback(request)
            result.completeness = (
                f"Unexpected error ({type(e).__name__}). {result.completeness}"
            )
            return result

    async def health_check(self):
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
        if self._executor:
            await self._executor.close()
