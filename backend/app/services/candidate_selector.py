"""
Resolution Candidate Selector for Razorpay CloseLoop Phase 5D.

Selects resolution candidates based on scoring, thresholds, and safety rules.

Rules:
- Conservative: if evidence insufficient → UNRESOLVED
- If candidates conflict → HUMAN_REVIEW
- Never force a resolution

This is a RECOMMENDATION ONLY.
It must NOT execute financial actions.
"""

from typing import Dict, List, Optional, Tuple

from app.schemas.candidate_scoring import CandidateScore, ScoringConfig
from app.schemas.intelligence import ExceptionIntelligence
from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    ResolutionProposal,
)
from app.schemas.resolution_selection import (
    ExplainabilityDetail,
    ExplainabilityLevel,
    SelectionConfig,
    SelectionResult,
    SelectionStatus,
)
from app.services.candidate_scorer import CandidateScoringService


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SELECTION_CONFIG = SelectionConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Selector
# ─────────────────────────────────────────────────────────────────────────────


class CandidateSelector:
    """
    Selects resolution candidates based on scoring and safety thresholds.

    Conservative approach:
    - UNRESOLVED when evidence insufficient
    - HUMAN_REVIEW when candidates conflict
    - Never force a resolution
    """

    def __init__(
        self,
        scoring_service: Optional[CandidateScoringService] = None,
        selection_config: Optional[SelectionConfig] = None,
    ):
        self.scorer = scoring_service or CandidateScoringService()
        self.config = selection_config or DEFAULT_SELECTION_CONFIG

    def select(
        self,
        generation_result: CandidateGenerationResult,
        intel: ExceptionIntelligence,
    ) -> SelectionResult:
        """Select the best resolution candidate.

        Args:
            generation_result: Candidate generation result with candidates
            intel: Exception intelligence for context

        Returns:
            SelectionResult with selected candidate or UNRESOLVED/HUMAN_REVIEW
        """
        # No candidates at all
        if not generation_result.candidates or generation_result.status == "UNRESOLVED":
            return self._unresolved(
                generation_result.exception_id,
                generation_result.case_id,
                ["No resolution candidates generated"],
            )

        # Score all candidates
        scored_result = self.scorer.score_and_rank(generation_result, intel)

        # Extract scores
        candidates = scored_result.candidates
        scores = []
        for c in candidates:
            score = self.scorer.score_candidate(c, intel)
            scores.append(score)

        # Check if any candidate passes minimum thresholds
        passing = []
        for c, s in zip(candidates, scores):
            if self._passes_thresholds(c, s, intel):
                passing.append((c, s))

        # No candidate passes thresholds
        if not passing:
            reasons = self._build_rejection_reasons(candidates, scores, intel)
            return self._unresolved(
                generation_result.exception_id,
                generation_result.case_id,
                reasons,
            )

        # Check for conflicts between top candidates
        if len(passing) >= 2:
            top_score = passing[0][1].final_score
            second_score = passing[1][1].final_score
            margin = top_score - second_score

            if margin < self.config.min_margin_over_second:
                # Close candidates — human review
                return self._human_review(
                    generation_result.exception_id,
                    generation_result.case_id,
                    passing,
                    intel,
                    [f"Top two candidates are close: margin {margin:.3f} < {self.config.min_margin_over_second}"],
                )

            # Check for evidence conflicts
            if self._has_material_conflict(passing, intel):
                return self._human_review(
                    generation_result.exception_id,
                    generation_result.case_id,
                    passing,
                    intel,
                    ["Material conflict between candidate evidence"],
                )

        # Select the best candidate
        selected_candidate, selected_score = passing[0]
        alternatives = [(c, s) for c, s in passing[1:]]

        # Calculate confidence
        confidence, factors = self._calculate_confidence(
            selected_candidate, selected_score, passing, intel
        )

        # Assess risk
        risk, risk_factors = self._assess_risk(selected_candidate, selected_score, intel)

        # Assess explainability
        explainability = self._assess_explainability(selected_candidate, intel)

        return SelectionResult(
            status=SelectionStatus.RECOMMENDED,
            exception_id=generation_result.exception_id,
            case_id=generation_result.case_id,
            selected_candidate=selected_candidate,
            selected_score=selected_score,
            alternatives=[c for c, _ in alternatives],
            alternative_scores=[s for _, s in alternatives],
            confidence=confidence,
            confidence_factors=factors,
            risk_category=risk,
            risk_factors=risk_factors,
            explainability=explainability,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold Checking
    # ─────────────────────────────────────────────────────────────────────────

    def _passes_thresholds(
        self,
        candidate: ResolutionProposal,
        score: CandidateScore,
        intel: ExceptionIntelligence,
    ) -> bool:
        """Check if a candidate passes minimum selection thresholds."""
        # Minimum final score
        if score.final_score < self.config.min_final_score:
            return False

        # Minimum evidence coverage
        if score.evidence_score < self.config.min_evidence_coverage:
            return False

        # Minimum financial consistency
        if score.financial_consistency_score < self.config.min_financial_consistency:
            return False

        # Maximum conflict penalty
        if score.conflict_penalty > self.config.max_conflict_penalty:
            return False

        # Maximum novelty penalty
        if score.novelty_penalty > self.config.max_novelty_penalty:
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Conflict Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _has_material_conflict(
        self,
        passing: List[Tuple[ResolutionProposal, CandidateScore]],
        intel: ExceptionIntelligence,
    ) -> bool:
        """Check for material conflicts between top candidates."""
        if len(passing) < 2:
            return False

        top1, score1 = passing[0]
        top2, score2 = passing[1]

        # Different resolution types
        if top1.resolution_type != top2.resolution_type:
            # Check if evidence supports both
            if not top1.evidence_compatible or not top2.evidence_compatible:
                return True

        # Check for evidence conflicts
        if intel.evidence and intel.evidence.has_conflict:
            return True

        # Check for ML/evidence disagreement
        if top1.ml_support and top2.ml_support:
            if top1.ml_support.supported != top2.ml_support.supported:
                return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        candidate: ResolutionProposal,
        score: CandidateScore,
        passing: List[Tuple[ResolutionProposal, CandidateScore]],
        intel: ExceptionIntelligence,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall recommendation confidence."""
        factors = {}

        # Base from final score
        factors["final_score"] = score.final_score

        # Margin bonus
        if len(passing) >= 2:
            margin = score.final_score - passing[1][1].final_score
            factors["margin"] = min(1.0, margin * 2)
        else:
            factors["margin"] = 1.0  # Only candidate

        # Evidence support
        factors["evidence_support"] = score.evidence_score

        # ML support
        factors["ml_support"] = score.ml_score if score.has_ml_support else 0.0

        # Historical support
        factors["historical_support"] = score.historical_score if score.has_historical_support else 0.0

        # Financial consistency
        factors["financial_consistency"] = score.financial_consistency_score

        # Conflict penalty (negative factor)
        factors["conflict_penalty"] = -score.conflict_penalty

        # Novelty penalty (negative factor)
        factors["novelty_penalty"] = -score.novelty_penalty

        # Weighted combination
        confidence = (
            factors["final_score"] * 0.4
            + factors["margin"] * 0.15
            + factors["evidence_support"] * 0.2
            + factors["financial_consistency"] * 0.15
            + factors["ml_support"] * 0.05
            + factors["historical_support"] * 0.05
        )

        # Apply penalties
        confidence += factors["conflict_penalty"] * 0.5
        confidence += factors["novelty_penalty"] * 0.5

        confidence = max(0.0, min(1.0, confidence))

        return confidence, factors

    # ─────────────────────────────────────────────────────────────────────────
    # Risk Assessment
    # ─────────────────────────────────────────────────────────────────────────

    def _assess_risk(
        self,
        candidate: ResolutionProposal,
        score: CandidateScore,
        intel: ExceptionIntelligence,
    ) -> Tuple[str, List[str]]:
        """Assess risk category for the recommendation."""
        risk_factors = []
        risk_score = 0

        # Financial adjustment size
        adj_amount = candidate.financial_adjustment.amount_paise
        if adj_amount >= self.config.high_risk_adjustment_paise:
            risk_score += 3
            risk_factors.append(f"High adjustment amount: {adj_amount} paise")
        elif adj_amount >= self.config.medium_risk_adjustment_paise:
            risk_score += 2
            risk_factors.append(f"Medium adjustment amount: {adj_amount} paise")

        # Evidence quality
        if score.evidence_score < 0.3:
            risk_score += 2
            risk_factors.append(f"Weak evidence support: {score.evidence_score:.2f}")

        # Conflict
        if score.conflict_penalty > 0:
            risk_score += 1
            risk_factors.append(f"Conflict detected: penalty {score.conflict_penalty:.2f}")

        # Novelty
        if score.is_novel:
            risk_score += 1
            risk_factors.append("Novel pattern — no strong historical match")

        # No ML support
        if not score.has_ml_support:
            risk_score += 0
            # Not a risk factor by itself

        # Multiple settlements
        if candidate.financial_adjustment.adjustment_type == "SETTLEMENT_ADJUSTMENT":
            if "DUPLICATE" in candidate.resolution_type:
                risk_score += 1
                risk_factors.append("Duplicate settlement resolution")

        # Determine category
        if risk_score >= 4:
            risk = "HIGH"
        elif risk_score >= 2:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return risk, risk_factors

    # ─────────────────────────────────────────────────────────────────────────
    # Explainability
    # ─────────────────────────────────────────────────────────────────────────

    def _assess_explainability(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> ExplainabilityDetail:
        """Assess how explainable the recommendation is."""
        has_evidence_trace = len(candidate.evidence_records) > 0
        has_financial_trace = candidate.financial_adjustment.evidence_record_id is not None
        has_historical = len(candidate.historical_support) > 0
        has_ml = candidate.ml_support is not None and candidate.ml_support.supported
        source_count = len(candidate.sources)

        # Determine level
        if (
            has_evidence_trace
            and has_financial_trace
            and source_count >= self.config.min_sources_for_fully_explainable
        ):
            level = ExplainabilityLevel.FULLY_EXPLAINABLE
        elif source_count >= self.config.min_sources_for_explainable:
            level = ExplainabilityLevel.PARTIALLY_EXPLAINABLE
        else:
            level = ExplainabilityLevel.NOT_EXPLAINABLE

        # Build explanation
        parts = []
        if has_evidence_trace:
            parts.append("Evidence records traced")
        if has_financial_trace:
            parts.append("Financial adjustment traced")
        if has_historical:
            parts.append(f"{len(candidate.historical_support)} historical case(s) support")
        if has_ml:
            parts.append("ML prediction supports")
        if not parts:
            parts.append("Limited traceability")

        return ExplainabilityDetail(
            level=level,
            has_evidence_trace=has_evidence_trace,
            has_financial_trace=has_financial_trace,
            has_historical_basis=has_historical,
            has_ml_basis=has_ml,
            source_count=source_count,
            explanation=". ".join(parts),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rejection Reasons
    # ─────────────────────────────────────────────────────────────────────────

    def _build_rejection_reasons(
        self,
        candidates: List[ResolutionProposal],
        scores: List[CandidateScore],
        intel: ExceptionIntelligence,
    ) -> List[str]:
        """Build reasons why no candidate was selected."""
        reasons = []

        for c, s in zip(candidates, scores):
            if s.final_score < self.config.min_final_score:
                reasons.append(
                    f"Candidate {c.resolution_type}: score {s.final_score:.3f} "
                    f"< minimum {self.config.min_final_score}"
                )
            if s.evidence_score < self.config.min_evidence_coverage:
                reasons.append(
                    f"Candidate {c.resolution_type}: evidence {s.evidence_score:.3f} "
                    f"< minimum {self.config.min_evidence_coverage}"
                )
            if s.financial_consistency_score < self.config.min_financial_consistency:
                reasons.append(
                    f"Candidate {c.resolution_type}: financial consistency "
                    f"{s.financial_consistency_score:.3f} < minimum "
                    f"{self.config.min_financial_consistency}"
                )
            if s.conflict_penalty > self.config.max_conflict_penalty:
                reasons.append(
                    f"Candidate {c.resolution_type}: conflict penalty "
                    f"{s.conflict_penalty:.3f} > maximum {self.config.max_conflict_penalty}"
                )
            if s.novelty_penalty > self.config.max_novelty_penalty:
                reasons.append(
                    f"Candidate {c.resolution_type}: novelty penalty "
                    f"{s.novelty_penalty:.3f} > maximum {self.config.max_novelty_penalty}"
                )

        if not reasons:
            reasons.append("No candidate passed all selection thresholds")

        return reasons

    # ─────────────────────────────────────────────────────────────────────────
    # Result Builders
    # ─────────────────────────────────────────────────────────────────────────

    def _unresolved(
        self,
        exception_id: str,
        case_id: str,
        reasons: List[str],
    ) -> SelectionResult:
        """Build UNRESOLVED result."""
        return SelectionResult(
            status=SelectionStatus.UNRESOLVED,
            exception_id=exception_id,
            case_id=case_id,
            confidence=0.0,
            confidence_factors={},
            risk_category="HIGH",
            risk_factors=["No resolution determined"],
            explainability=ExplainabilityDetail(
                level=ExplainabilityLevel.NOT_EXPLAINABLE,
                explanation="No resolution could be determined",
            ),
            rejection_reasons=reasons,
        )

    def _human_review(
        self,
        exception_id: str,
        case_id: str,
        passing: List[Tuple[ResolutionProposal, CandidateScore]],
        intel: ExceptionIntelligence,
        conflict_reasons: List[str],
    ) -> SelectionResult:
        """Build HUMAN_REVIEW result."""
        # Sort by score
        passing.sort(key=lambda x: x[1].final_score, reverse=True)

        selected = passing[0][0]
        selected_score = passing[0][1]
        alternatives = [c for c, _ in passing[1:]]
        alt_scores = [s for _, s in passing[1:]]

        # Calculate confidence (lower due to conflict)
        confidence = selected_score.final_score * 0.7

        # Assess risk
        risk, risk_factors = self._assess_risk(selected, selected_score, intel)
        risk_factors.append("Human review required due to candidate conflict")

        # Assess explainability
        explainability = self._assess_explainability(selected, intel)

        return SelectionResult(
            status=SelectionStatus.HUMAN_REVIEW,
            exception_id=exception_id,
            case_id=case_id,
            selected_candidate=selected,
            selected_score=selected_score,
            alternatives=alternatives,
            alternative_scores=alt_scores,
            confidence=confidence,
            confidence_factors={"base_score": selected_score.final_score, "conflict_reduction": 0.7},
            risk_category=risk,
            risk_factors=risk_factors,
            explainability=explainability,
            rejection_reasons=conflict_reasons,
        )
