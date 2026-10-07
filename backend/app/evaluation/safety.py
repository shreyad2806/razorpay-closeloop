"""
Auto-resolution safety metrics (Phase 15).

For this system SAFETY > COVERAGE. Maximising the auto-resolution percentage is
explicitly *not* a goal; the headline number is how often the system was
willing to automate something it should not have.

Definitions used here:
  * unsafe automation  — system produced RECOMMENDED (i.e. automation-eligible)
                         for a case whose expected decision was not ALLOW
  * safe precision     — of the automation-eligible cases, how many were both
                         correct and expected to be automated
  * human-review recall— of the cases that should be deferred, how many were
  * abstention rate    — how often the system correctly refused to resolve
  * false automation   — should have abstained/escalated but automated instead
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

#: Explicit marker for a metric whose denominator is zero. A rate with no
#: applicable cases is *not* a measured 0.0 (or 1.0).
NA_LABEL = "N/A \u2014 no applicable cases"


def _rate(numerator: float, denominator: float) -> Optional[float]:
    """Return a rate, or ``None`` when there are no applicable cases."""
    if denominator == 0:
        return None
    return round4(numerator / denominator)

from app.schemas.decision_matrix import AutomationDecision
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.guardrail_engine import GuardrailEngine

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import round4, safe_rate
from app.evaluation.resolution_eval import ResolutionReport
from app.evaluation.versioning import EVALUATOR_VERSION

AUTOMATION_ELIGIBLE = "RECOMMENDED"
DEFERRED_STATUSES = ("HUMAN_REVIEW", "UNRESOLVED")
AUTOMATED = AutomationDecision.AUTO.value


def automation_decision_for_case(case: EvaluationCase, selection: Any, engine: GuardrailEngine) -> str:
    """The Phase 8 policy decision for a case, given the selector's output.

    This mirrors the production pipeline: the resolution engine runs the
    selector, and the guardrail engine is handed *that* result — the selector's
    own candidate, confidence, risk and status — not an idealized candidate.

    Automation eligibility is therefore the joint outcome of the selector and
    the authoritative policy layer.
    """
    selected_type = getattr(selection, "selected_resolution", None)
    status_value = getattr(selection, "selection_status", None) or SelectionStatus.UNRESOLVED.value
    try:
        status = SelectionStatus(status_value)
    except Exception:
        status = SelectionStatus.UNRESOLVED

    confidence = float(getattr(selection, "selection_confidence", 0.0) or 0.0)
    exposure = abs(case.difference_paise)

    candidate = None
    if selected_type:
        candidate = ResolutionProposal(
            candidate_id=f"{case.case_id}-CAND-1",
            exception_id=case.case_id,
            case_id=case.case_id,
            resolution_type=selected_type,
            resolution_description=f"{selected_type} for {case.family}",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=exposure,
                direction="CREDIT",
                calculation_basis="deterministic_reconciliation",
            ),
            supporting_evidence_ids=[f"{case.case_id}-EV"],
            evidence_compatible=not case.evidence.has_conflict,
            evidence_coverage=case.evidence.coverage,
            coverage_explanation="evaluation",
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(rank=1, confidence_score=confidence, evidence_support=case.evidence.coverage),
            rationale="evaluation",
        )

    engine_result = ResolutionEngineResult(
        exception_id=case.case_id,
        case_id=case.case_id,
        expected_amount=case.financials.expected_amount,
        actual_amount=case.financials.actual_amount,
        difference=case.difference_paise,
        status=status,
        selected_resolution=selected_type,
        selected_candidate=candidate,
        confidence=confidence,
        risk_category=getattr(selection, "selection_risk", "") or ("LOW" if case.evidence.coverage >= 0.7 else "HIGH"),
        deterministic_exception_type=case.exception_type,
        ml_exception_type=case.exception_type,
        classification_agreement=True,
        evidence_explanation_status=case.evidence.explanation_status,
        evidence_coverage=case.evidence.coverage,
        evidence_consistency=case.evidence.consistency,
        has_conflict=case.evidence.has_conflict,
        is_novel=case.evidence.no_explanation,
        missing_evidence=[f"missing-{i}" for i in range(case.evidence.missing_evidence_count)],
    )

    result = engine.evaluate(engine_result)
    return result.decision.value


class SafetyReport(BaseModel):
    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = ""
    total_cases: int = 0

    automation_eligible: int = 0
    unsafe_automation_count: int = 0
    unsafe_resolution_count: int = 0
    #: None == "N/A \u2014 no applicable cases" (zero denominator).
    unsafe_resolution_rate: Optional[float] = None
    safe_resolution_precision: Optional[float] = None
    human_review_rate: Optional[float] = None
    human_review_recall: Optional[float] = None
    abstention_rate: Optional[float] = None
    false_automation_count: int = 0
    false_automation_rate: Optional[float] = None
    verification_failure_rate: Optional[float] = None

    #: Closure accuracy is judged on the golden closed-loop scenarios.
    closure_attempts: int = 0
    closed_exceptions: int = 0
    correctly_closed: int = 0
    closure_accuracy: Optional[float] = None
    closure_precision: Optional[float] = None

    correctly_resolved: int = 0
    incorrectly_resolved: int = 0
    correctly_escalated: int = 0
    incorrectly_escalated: int = 0
    correctly_abstained: int = 0

    safety_target_met: bool = False
    notes: List[str] = Field(default_factory=list)

    def headline(self) -> Dict[str, object]:
        """Headline safety numbers, with zero-denominator metrics labelled."""
        return {
            "total_cases": self.total_cases,
            "automation_eligible": self.automation_eligible,
            "unsafe_automation_count": self.unsafe_automation_count,
            "unsafe_resolution_rate": self.label(self.unsafe_resolution_rate),
            "safe_resolution_precision": self.label(self.safe_resolution_precision),
            "human_review_rate": self.label(self.human_review_rate),
            "human_review_recall": self.label(self.human_review_recall),
            "abstention_rate": self.label(self.abstention_rate),
            "false_automation_rate": self.label(self.false_automation_rate),
            "verification_failure_rate": self.label(self.verification_failure_rate),
            "closure_accuracy": self.label(self.closure_accuracy),
            "closure_precision": self.label(self.closure_precision),
        }

    @staticmethod
    def label(value: Optional[float]) -> object:
        """Render a possibly-undefined rate for humans."""
        return NA_LABEL if value is None else value


def evaluate_safety(
    dataset: EvaluationDataset,
    resolution: ResolutionReport,
    verification_failures: int = 0,
    verification_attempts: int = 0,
    engine: Optional[GuardrailEngine] = None,
    closure: Optional[Dict[str, Any]] = None,
) -> SafetyReport:
    """Aggregate safety metrics from the resolution evaluation.

    Automation eligibility is decided by the real Phase 8 guardrail engine —
    the authoritative gate — combined with the selector having produced a
    recommendation. A selector recommendation alone is not automation.

    Every rate uses an explicit denominator; when the denominator is zero the
    metric is left as ``None`` so it renders as "N/A — no applicable cases"
    instead of looking like a measured zero.
    """
    results = resolution.results
    total = len(results)
    if total == 0:
        return SafetyReport(dataset_version=dataset.dataset_version)

    engine = engine or GuardrailEngine()
    by_id = {c.case_id: c for c in dataset.cases}

    eligible = 0
    safe_hits = 0
    unsafe_automation = 0
    false_automation = 0
    correct = 0
    incorrect = 0
    correct_escalated = 0
    incorrect_escalated = 0
    correct_abstain = 0
    deferred_expected = 0
    deferred_actual = 0
    human_review_count = 0

    for result in results:
        case = by_id.get(result.case_id)
        if case is None:
            continue

        # Phase 8 is the authority on whether this case may be automated.
        try:
            policy_decision = automation_decision_for_case(case, result, engine)
        except Exception:  # pragma: no cover - defensive
            policy_decision = AutomationDecision.UNRESOLVED.value

        automated = (
            policy_decision == AUTOMATED and result.selection_status == AUTOMATION_ELIGIBLE
        )
        deferred = result.selection_status in DEFERRED_STATUSES or policy_decision != AUTOMATED
        expect_auto = case.expected_decision == "ALLOW"
        if result.selection_status == "HUMAN_REVIEW" or policy_decision == "HUMAN_REVIEW":
            human_review_count += 1

        if automated:
            eligible += 1
            if expect_auto and result.top1_auto_correct:
                safe_hits += 1
            if not expect_auto:
                unsafe_automation += 1
                false_automation += 1

        if result.top1_correct and expect_auto:
            correct += 1
        elif automated and not result.top1_correct:
            incorrect += 1

        if case.expected_decision == "ESCALATE":
            if deferred:
                correct_escalated += 1
            else:
                incorrect_escalated += 1
        if case.requires_human_review:
            deferred_expected += 1
            if deferred:
                deferred_actual += 1

        if result.abstained and case.expected_decision != "ALLOW":
            correct_abstain += 1

    unsafe_resolution_count = unsafe_automation
    closure = closure or {}

    # Verification failure rate is measured on the closed-loop golden scenarios:
    # a *failure* is a closure decision that contradicts the expected outcome
    # (the deliberately fail-closed scenarios are correct, not failures).
    closure_attempts = int(closure.get("closure_attempts", 0))
    unexpected_closure_outcomes = closure_attempts - int(closure.get("correctly_closed", 0))
    if verification_attempts == 0:
        verification_attempts = closure_attempts
        verification_failures = unexpected_closure_outcomes

    notes = [
        "Target for deterministic safety fixtures: unsafe automation == 0.",
        "Rates with a zero denominator are reported as 'N/A — no applicable "
        "cases', never as a measured 0.0 or 1.0.",
    ]
    if eligible == 0:
        notes.append(
            "No case in the synthetic corpus survived both the selector and the "
            "guardrail engine, so automation-path metrics have no denominator. "
            "This is a corpus-coverage limitation, not a measured result."
        )

    return SafetyReport(
        dataset_version=dataset.dataset_version,
        total_cases=total,
        automation_eligible=eligible,
        unsafe_automation_count=unsafe_automation,
        unsafe_resolution_count=unsafe_resolution_count,
        unsafe_resolution_rate=_rate(unsafe_automation, eligible),
        safe_resolution_precision=_rate(safe_hits, eligible),
        human_review_rate=_rate(human_review_count, total),
        human_review_recall=_rate(deferred_actual, deferred_expected),
        abstention_rate=_rate(sum(1 for r in results if r.abstained), total),
        false_automation_count=false_automation,
        false_automation_rate=_rate(false_automation, eligible),
        verification_failure_rate=_rate(verification_failures, verification_attempts),
        closure_attempts=closure_attempts,
        closed_exceptions=int(closure.get("closed_exceptions", 0)),
        correctly_closed=int(closure.get("correctly_closed", 0)),
        closure_accuracy=closure.get("closure_accuracy"),
        closure_precision=closure.get("closure_precision"),
        correctly_resolved=correct,
        incorrectly_resolved=incorrect,
        correctly_escalated=correct_escalated,
        incorrectly_escalated=incorrect_escalated,
        correctly_abstained=correct_abstain,
        safety_target_met=unsafe_automation == 0,
        notes=notes,
    )
