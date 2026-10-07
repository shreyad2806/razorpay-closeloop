"""
Evidence quality evaluation (Phase 15).

Evaluates the evidence signals the production pipeline exposes at decision time
(coverage, consistency, missing evidence, conflicts, supporting evidence count,
explanation status) and measures whether weak evidence correlates with
incorrect or unsafe recommendations.

The evaluator does not produce, mutate or re-rank evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import mean, round4, safe_rate
from app.evaluation.resolution_eval import ResolutionReport
from app.evaluation.safety import AUTOMATED, automation_decision_for_case
from app.evaluation.versioning import EVALUATOR_VERSION

UNEXPLAINED_STATUSES = {"UNEXPLAINED", "CONFLICTING"}


class EvidenceReport(BaseModel):
    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = ""
    cases: int = 0

    mean_coverage: float = 0.0
    min_coverage: float = 0.0
    max_coverage: float = 0.0
    mean_consistency: float = 0.0

    fully_explained: int = 0
    partially_explained: int = 0
    unsupported: int = 0
    conflicting: int = 0
    missing_evidence_cases: int = 0
    mean_missing_evidence: float = 0.0
    mean_supporting_evidence: float = 0.0
    candidate_explanation_coverage: float = 0.0

    weak_evidence_cases: int = 0
    weak_evidence_correct_rate: float = 0.0
    strong_evidence_correct_rate: float = 0.0
    weak_evidence_unsafe_rate: float = 0.0

    notes: List[str] = Field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return self.model_dump()


def evaluate_evidence(
    dataset: EvaluationDataset,
    cases: Optional[Sequence[EvaluationCase]] = None,
    resolution: Optional[ResolutionReport] = None,
) -> EvidenceReport:
    """Compute evidence-quality metrics over the dataset."""
    cases = list(cases if cases is not None else dataset.cases)
    if not cases:
        return EvidenceReport(dataset_version=dataset.dataset_version)

    coverages = [c.evidence.coverage for c in cases]
    consistencies = [c.evidence.consistency for c in cases]

    fully = sum(1 for c in cases if c.evidence.explanation_status == "FULLY_EXPLAINED")
    partially = sum(1 for c in cases if c.evidence.explanation_status == "PARTIALLY_EXPLAINED")
    unsupported = sum(1 for c in cases if c.evidence.explanation_status in UNEXPLAINED_STATUSES)
    conflicting = sum(1 for c in cases if c.evidence.has_conflict)
    missing = sum(1 for c in cases if c.evidence.missing_evidence_count > 0)

    # Correlation between evidence strength and recommendation correctness.
    weak_correct: List[bool] = []
    strong_correct: List[bool] = []
    weak_unsafe: List[bool] = []

    if resolution is not None:
        from app.services.guardrail_engine import GuardrailEngine

        engine = GuardrailEngine()
        by_id = {r.case_id: r for r in resolution.results}
        for case in cases:
            result = by_id.get(case.case_id)
            if result is None:
                continue
            weak = case.evidence.coverage < 0.5 or case.evidence.has_conflict
            correct = result.top1_auto_correct
            try:
                policy_decision = automation_decision_for_case(case, result, engine)
            except Exception:  # pragma: no cover - defensive
                policy_decision = "ERROR"
            if weak:
                weak_correct.append(correct)
                # Unsafe: weak evidence but the authoritative policy layer was
                # willing to automate something that should not be automated.
                weak_unsafe.append(
                    policy_decision == AUTOMATED
                    and result.selection_status == "RECOMMENDED"
                    and case.expected_decision != "ALLOW"
                )
            else:
                strong_correct.append(correct)

    return EvidenceReport(
        dataset_version=dataset.dataset_version,
        cases=len(cases),
        mean_coverage=mean(coverages),
        min_coverage=round4(min(coverages)) if coverages else 0.0,
        max_coverage=round4(max(coverages)) if coverages else 0.0,
        mean_consistency=mean(consistencies),
        fully_explained=fully,
        partially_explained=partially,
        unsupported=unsupported,
        conflicting=conflicting,
        missing_evidence_cases=missing,
        mean_missing_evidence=mean([float(c.evidence.missing_evidence_count) for c in cases]),
        mean_supporting_evidence=mean([float(c.evidence.supporting_evidence_count) for c in cases]),
        candidate_explanation_coverage=mean([float(c.evidence.candidate_count) for c in cases]),
        weak_evidence_cases=len(weak_correct) + len(weak_unsafe),
        weak_evidence_correct_rate=round4(safe_rate(sum(1 for c in weak_correct if c), len(weak_correct))),
        strong_evidence_correct_rate=round4(safe_rate(sum(1 for c in strong_correct if c), len(strong_correct))),
        weak_evidence_unsafe_rate=round4(safe_rate(sum(1 for c in weak_unsafe if c), len(weak_unsafe))),
        notes=[
            "Evidence signals are the deterministic values frozen at decision time.",
            "The evaluator never creates, edits or re-ranks evidence.",
            "Unsafe here means: weak evidence, a selector recommendation, the Phase 8 "
            "policy layer returned AUTO, and the case was not expected to be automated.",
        ],
    )
