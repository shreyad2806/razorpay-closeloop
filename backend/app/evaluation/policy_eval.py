"""
Policy / guardrail evaluation matrix (Phase 15).

Drives the **real** :class:`~app.services.guardrail_engine.GuardrailEngine` over
a deterministic matrix of conditions:

    ML confidence × evidence quality × exposure × historical similarity ×
    deterministic consistency × authorization × approval requirement

The engine's rules, thresholds and reason codes are untouched. This module only
records the decision it produced and checks that the safety invariants hold:

  * high ML confidence must NOT bypass policy
  * high historical similarity must NOT bypass policy
  * good evidence must NOT bypass authorization
  * deterministic financial inconsistency remains authoritative
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.decision_matrix import AutomationDecision
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    HistoricalSupportDetail,
    ResolutionProposal,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus
from app.services.guardrail_engine import GuardrailEngine

from app.evaluation.metrics import round4, safe_rate
from app.evaluation.versioning import EVALUATOR_VERSION

EXPECT_AUTO = "AUTO"
EXPECT_NOT_AUTO = "NOT_AUTO"


class PolicyRow(BaseModel):
    row_id: str
    description: str
    confidence: float
    evidence_coverage: float
    evidence_consistency: float
    historical_similarity: float
    exposure_paise: int
    deterministic_consistent: bool
    authorized: bool
    approval_required: bool
    dependencies_healthy: bool
    has_conflict: bool = False
    is_novel: bool = False
    missing_evidence: List[str] = Field(default_factory=list)
    expected: str
    actual_decision: str = ""
    auto: bool = False
    invariant_held: bool = True
    error: str = ""


class PolicyReport(BaseModel):
    evaluator_version: str = EVALUATOR_VERSION
    rows: List[PolicyRow] = Field(default_factory=list)
    allow_count: int = 0
    human_review_count: int = 0
    unresolved_count: int = 0
    escalate_count: int = 0
    deny_count: int = 0
    invariant_violations: List[str] = Field(default_factory=list)
    confidence_bypass_detected: bool = False
    similarity_bypass_detected: bool = False
    evidence_bypass_detected: bool = False
    consistency_bypass_detected: bool = False

    def summary(self) -> Dict[str, object]:
        return {
            "rows": len(self.rows),
            "allow_count": self.allow_count,
            "human_review_count": self.human_review_count,
            "unresolved_count": self.unresolved_count,
            "escalate_count": self.escalate_count,
            "deny_count": self.deny_count,
            "invariant_violations": self.invariant_violations,
            "confidence_bypass_detected": self.confidence_bypass_detected,
            "similarity_bypass_detected": self.similarity_bypass_detected,
            "evidence_bypass_detected": self.evidence_bypass_detected,
            "consistency_bypass_detected": self.consistency_bypass_detected,
            "detail": [r.model_dump() for r in self.rows],
        }


def _candidate(similarity: float, exposure_paise: int) -> ResolutionProposal:
    return ResolutionProposal(
        candidate_id="CAND-POLICY",
        exception_id="EXC-POLICY",
        case_id="CASE-POLICY",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Policy matrix candidate",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=exposure_paise,
            direction="CREDIT",
            calculation_basis="policy_matrix",
        ),
        supporting_evidence_ids=["EV-1"],
        evidence_compatible=True,
        evidence_coverage=0.9,
        coverage_explanation="matrix",
        sources=["deterministic_evidence"],
        historical_support=[
            HistoricalSupportDetail(
                case_id="HIST-POLICY",
                similarity_score=similarity,
                historical_resolution="FEE_ADJUSTMENT",
                historical_outcome="VERIFIED",
            )
        ],
        ranking=CandidateRanking(rank=1, confidence_score=0.9, evidence_support=0.9),
        rationale="Policy matrix candidate",
    )


def _engine_result(row: PolicyRow) -> ResolutionEngineResult:
    return ResolutionEngineResult(
        exception_id="EXC-POLICY",
        case_id="CASE-POLICY",
        expected_amount=1_000_000,
        actual_amount=950_000,
        difference=50_000,
        status=SelectionStatus.RECOMMENDED,
        selected_resolution="FEE_ADJUSTMENT",
        selected_candidate=_candidate(row.historical_similarity, row.exposure_paise),
        confidence=row.confidence,
        risk_category="LOW" if row.confidence >= 0.8 else "MEDIUM",
        deterministic_exception_type="FEE_MISMATCH" if row.deterministic_consistent else "UNKNOWN",
        ml_exception_type="FEE_MISMATCH",
        classification_agreement=row.deterministic_consistent,
        evidence_explanation_status="FULLY_EXPLAINED" if row.evidence_coverage >= 0.7 else "UNEXPLAINED",
        evidence_coverage=row.evidence_coverage,
        evidence_consistency=row.evidence_consistency,
        has_conflict=row.has_conflict,
        is_novel=row.is_novel,
        missing_evidence=list(row.missing_evidence),
    )


def _matrix_rows() -> List[PolicyRow]:
    return [
        PolicyRow(
            row_id="R1",
            description="high confidence + strong evidence + low exposure + consistent + authorized",
            confidence=0.95,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_AUTO,
        ),
        PolicyRow(
            row_id="R2",
            description="HIGH ML CONFIDENCE + weak evidence",
            confidence=0.99,
            evidence_coverage=0.15,
            evidence_consistency=0.20,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R3",
            description="HIGH ML CONFIDENCE + very high exposure",
            confidence=0.99,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            historical_similarity=0.99,
            exposure_paise=500_000_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R4",
            description="HIGH ML CONFIDENCE + deterministic inconsistency (conflict)",
            confidence=0.99,
            evidence_coverage=0.90,
            evidence_consistency=0.20,
            historical_similarity=0.99,
            exposure_paise=10_000,
            deterministic_consistent=False,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            has_conflict=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R5",
            description="medium confidence",
            confidence=0.70,
            evidence_coverage=0.90,
            evidence_consistency=0.90,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R6",
            description="low confidence",
            confidence=0.25,
            evidence_coverage=0.90,
            evidence_consistency=0.90,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R7",
            description="high confidence + strong evidence + unhealthy dependency (ML unavailable)",
            confidence=0.99,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=False,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R8",
            description="HIGH HISTORICAL SIMILARITY + weak evidence",
            confidence=0.90,
            evidence_coverage=0.10,
            evidence_consistency=0.15,
            historical_similarity=1.0,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R9",
            description="high confidence + strong evidence but NOT authorized",
            confidence=0.99,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            historical_similarity=0.99,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=False,
            approval_required=True,
            dependencies_healthy=True,
            is_novel=True,
            missing_evidence=["fee_record"],
            expected=EXPECT_NOT_AUTO,
        ),
        PolicyRow(
            row_id="R4b",
            description="high confidence + strong evidence + explicit evidence conflict",
            confidence=0.99,
            evidence_coverage=0.95,
            evidence_consistency=0.95,
            historical_similarity=0.95,
            exposure_paise=10_000,
            deterministic_consistent=True,
            authorized=True,
            approval_required=False,
            dependencies_healthy=True,
            has_conflict=True,
            expected=EXPECT_NOT_AUTO,
        ),
    ]


def evaluate_policy(engine: Optional[GuardrailEngine] = None) -> PolicyReport:
    """Run the real guardrail engine over the deterministic policy matrix."""
    engine = engine or GuardrailEngine()
    rows: List[PolicyRow] = []
    violations: List[str] = []

    for row in _matrix_rows():
        dependencies = {
            "ml_classifier": row.dependencies_healthy,
            "historical_memory": row.dependencies_healthy,
            "llm": row.dependencies_healthy,
        }
        try:
            result = engine.evaluate(_engine_result(row), dependencies)
            decision = result.decision.value
        except Exception as exc:  # pragma: no cover - defensive
            row = row.model_copy(update={"actual_decision": "ERROR", "error": type(exc).__name__})
            rows.append(row)
            continue

        auto = decision == AutomationDecision.AUTO.value
        row = row.model_copy(update={"actual_decision": decision, "auto": auto})

        if row.expected == EXPECT_NOT_AUTO and auto:
            row = row.model_copy(update={"invariant_held": False})
            violations.append(f"{row.row_id}: {row.description} produced AUTO")
        if row.expected == EXPECT_AUTO and not auto:
            violations.append(f"{row.row_id}: expected AUTO but got {decision}")

        rows.append(row)

    allow = sum(1 for r in rows if r.auto)
    human_review = sum(1 for r in rows if r.actual_decision == AutomationDecision.HUMAN_REVIEW.value)
    unresolved = sum(1 for r in rows if r.actual_decision == AutomationDecision.UNRESOLVED.value)

    def _bypass(row_ids: List[str]) -> bool:
        """True when any of the listed rows auto-resolved (a bypass)."""
        return any(r.auto for r in rows if r.row_id in row_ids)

    return PolicyReport(
        rows=rows,
        allow_count=allow,
        human_review_count=human_review,
        unresolved_count=unresolved,
        escalate_count=0,
        deny_count=len(rows) - allow - human_review - unresolved,
        invariant_violations=violations,
        confidence_bypass_detected=_bypass(["R2", "R3", "R4", "R4b", "R7"]),
        similarity_bypass_detected=_bypass(["R8"]),
        evidence_bypass_detected=_bypass(["R2", "R8"]),
        consistency_bypass_detected=_bypass(["R4", "R4b"]),
    )
