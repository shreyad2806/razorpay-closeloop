"""
Resolution Candidate Scoring Service for Razorpay CloseLoop Phase 5C.

Scores and ranks resolution candidates using independent signals:
- Evidence score
- ML score
- Historical similarity score
- Financial consistency score
- Novelty penalty
- Conflict penalty

This is a RECOMMENDATION SCORE.
It is NOT financial truth.
It does NOT authorize execution.

DOES NOT:
- execute financial actions
- modify records
- auto-resolve exceptions
"""

from typing import Dict, List, Optional

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult
from app.schemas.intelligence import (
    ExceptionIntelligence,
    SimilarCasesIntelligence,
)
from app.schemas.resolution_candidate import (
    ResolutionProposal,
    CandidateGenerationResult,
)
from app.schemas.candidate_scoring import CandidateScore, ScoringConfig


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = ScoringConfig(
    evidence_weight=0.35,
    ml_weight=0.20,
    historical_weight=0.15,
    financial_weight=0.30,
    novelty_penalty_factor=0.15,
    conflict_penalty_factor=0.25,
    min_similarity_for_bonus=0.7,
    max_historical_cases_for_bonus=3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Service
# ─────────────────────────────────────────────────────────────────────────────


class CandidateScoringService:
    """
    Scores resolution candidates using independent signals.

    Each component score is calculated independently.
    Penalties are applied for novelty and conflicts.
    Final score is a weighted combination.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def score_candidate(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> CandidateScore:
        """Score a single resolution candidate.

        Args:
            candidate: The resolution proposal to score
            intel: Exception intelligence for context

        Returns:
            CandidateScore with detailed breakdown
        """
        # 1. Evidence score
        evidence_score = self._score_evidence(candidate, intel)

        # 2. ML score
        ml_score = self._score_ml(candidate, intel)

        # 3. Historical score
        historical_score = self._score_historical(candidate, intel)

        # 4. Financial consistency
        financial_score = self._score_financial(candidate, intel)

        # 5. Novelty penalty
        novelty_penalty = self._calculate_novelty_penalty(candidate, intel)

        # 6. Conflict penalty
        conflict_penalty = self._calculate_conflict_penalty(candidate, intel)

        # Weighted components
        weighted_evidence = evidence_score * self.config.evidence_weight
        weighted_ml = ml_score * self.config.ml_weight
        weighted_historical = historical_score * self.config.historical_weight
        weighted_financial = financial_score * self.config.financial_weight

        # Composite score
        raw_score = (
            weighted_evidence
            + weighted_ml
            + weighted_historical
            + weighted_financial
        )
        final_score = max(0.0, min(1.0, raw_score - novelty_penalty - conflict_penalty))

        return CandidateScore(
            evidence_score=evidence_score,
            ml_score=ml_score,
            historical_score=historical_score,
            financial_consistency_score=financial_score,
            novelty_penalty=novelty_penalty,
            conflict_penalty=conflict_penalty,
            final_score=final_score,
            weighted_evidence=weighted_evidence,
            weighted_ml=weighted_ml,
            weighted_historical=weighted_historical,
            weighted_financial=weighted_financial,
            has_evidence_support=len(candidate.supporting_evidence_ids) > 0,
            has_ml_support=candidate.ml_support is not None and candidate.ml_support.supported,
            has_historical_support=len(candidate.historical_support) > 0,
            is_novel=novelty_penalty > 0,
            has_conflicts=conflict_penalty > 0,
        )

    def score_and_rank(
        self,
        result: CandidateGenerationResult,
        intel: ExceptionIntelligence,
    ) -> CandidateGenerationResult:
        """Score all candidates and re-rank by final score.

        Args:
            result: Candidate generation result
            intel: Exception intelligence

        Returns:
            Updated result with scores and re-ranked candidates
        """
        if not result.candidates:
            return result

        # Score each candidate
        scored = []
        for candidate in result.candidates:
            score = self.score_candidate(candidate, intel)
            scored.append((candidate, score))

        # Sort by final score descending
        scored.sort(key=lambda x: x[1].final_score, reverse=True)

        # Re-rank
        for i, (candidate, score) in enumerate(scored):
            candidate.ranking.rank = i + 1
            candidate.ranking.confidence_score = score.final_score

        # Return updated result
        return CandidateGenerationResult(
            exception_id=result.exception_id,
            case_id=result.case_id,
            status=result.status,
            candidates=[c for c, _ in scored],
            total_candidates=len(scored),
            pipeline_version=result.pipeline_version,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Component Scorers
    # ─────────────────────────────────────────────────────────────────────────

    def _score_evidence(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Score evidence support.

        Considers:
        - Evidence coverage (how much of discrepancy is explained)
        - Evidence consistency
        - Number of supporting records
        - Evidence compatibility
        """
        score = 0.0

        # Base from evidence coverage
        coverage = intel.evidence.evidence_coverage if intel.evidence else 0.0
        score += coverage * 0.4

        # Consistency bonus
        consistency = intel.evidence.consistency_score if intel.evidence else 0.0
        score += consistency * 0.3

        # Supporting evidence count bonus (diminishing returns)
        evidence_count = len(candidate.supporting_evidence_ids)
        if evidence_count > 0:
            score += min(0.2, evidence_count * 0.05)

        # Compatibility bonus
        if candidate.evidence_compatible:
            score += 0.1

        return min(1.0, score)

    def _score_ml(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Score ML support.

        Uses ML confidence where available.
        Returns 0.0 if ML did not produce this candidate.
        """
        if not candidate.ml_support or not candidate.ml_support.supported:
            return 0.0

        # Use ML probability for this specific resolution
        if candidate.ml_support.probability is not None:
            return candidate.ml_support.probability

        # Fall back to overall confidence
        if candidate.ml_support.confidence is not None:
            return candidate.ml_support.confidence

        return 0.0

    def _score_historical(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Score historical similarity support.

        Considers:
        - Top similarity score
        - Number of supporting cases (diminishing returns)
        - Historical resolution agreement
        """
        if not candidate.historical_support:
            return 0.0

        # Top similarity score
        similarities = [h.similarity_score for h in candidate.historical_support]
        top_similarity = max(similarities) if similarities else 0.0

        # Number of supporting cases (diminishing returns)
        case_count = len(candidate.historical_support)
        count_bonus = min(0.3, case_count * 0.1)

        # Historical resolution agreement
        resolutions = [h.historical_resolution for h in candidate.historical_support]
        if resolutions and all(r == resolutions[0] for r in resolutions):
            agreement_bonus = 0.2  # All agree
        elif resolutions:
            agreement_bonus = 0.1  # Partial agreement
        else:
            agreement_bonus = 0.0

        score = top_similarity * 0.5 + count_bonus + agreement_bonus
        return min(1.0, score)

    def _score_financial(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Score financial consistency.

        Checks:
        - Adjustment amount vs discrepancy
        - Adjustment direction correctness
        - Evidence traceability
        """
        adjustment = candidate.financial_adjustment
        difference = intel.difference

        # No adjustment needed
        if adjustment.amount_paise == 0 and difference == 0:
            return 1.0

        # Adjustment exists but no discrepancy
        if adjustment.amount_paise > 0 and difference == 0:
            return 0.0

        # No adjustment but discrepancy exists
        if adjustment.amount_paise == 0 and difference != 0:
            return 0.0

        # Calculate consistency ratio
        abs_diff = abs(difference)
        abs_adj = adjustment.amount_paise

        if abs_diff == 0:
            return 0.0

        # Perfect match
        if abs_adj == abs_diff:
            ratio_score = 1.0
        elif abs_adj < abs_diff:
            # Partial coverage
            ratio_score = abs_adj / abs_diff
        else:
            # Over-adjustment — penalize
            ratio_score = max(0.0, 1.0 - (abs_adj - abs_diff) / abs_diff)

        # Direction correctness
        if difference > 0 and adjustment.direction == "CREDIT":
            direction_score = 1.0
        elif difference < 0 and adjustment.direction == "DEBIT":
            direction_score = 1.0
        elif adjustment.direction == "NONE":
            direction_score = 1.0
        else:
            direction_score = 0.0

        # Evidence traceability
        trace_score = 0.5 if adjustment.evidence_record_id else 0.0
        if adjustment.calculation_basis in ("fee_record_sum", "refund_record_sum", "tax_record_sum"):
            trace_score = 1.0

        return ratio_score * 0.5 + direction_score * 0.3 + trace_score * 0.2

    # ─────────────────────────────────────────────────────────────────────────
    # Penalties
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_novelty_penalty(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Calculate novelty penalty.

        Low historical similarity increases novelty.
        Novel cases receive a penalty.
        """
        # Check if novel based on historical support
        if not candidate.historical_support:
            # No historical support at all — higher novelty
            return self.config.novelty_penalty_factor * 0.8

        # Check similarity levels
        similarities = [h.similarity_score for h in candidate.historical_support]
        max_similarity = max(similarities) if similarities else 0.0

        if max_similarity < self.config.min_similarity_for_bonus:
            # Low similarity — moderate novelty
            return self.config.novelty_penalty_factor * (1.0 - max_similarity)

        return 0.0

    def _calculate_conflict_penalty(
        self,
        candidate: ResolutionProposal,
        intel: ExceptionIntelligence,
    ) -> float:
        """Calculate conflict penalty.

        Applies penalty when signals disagree.
        """
        penalty = 0.0

        # ML disagrees with candidate
        if candidate.ml_support and candidate.ml_support.supported:
            if intel.classification and not intel.classification.agreement:
                penalty += self.config.conflict_penalty_factor * 0.3

        # Evidence conflicts
        if intel.evidence and intel.evidence.has_conflict:
            penalty += self.config.conflict_penalty_factor * 0.3

        # Evidence not compatible
        if not candidate.evidence_compatible:
            penalty += self.config.conflict_penalty_factor * 0.2

        # Historical cases disagree
        if candidate.historical_support:
            resolutions = [h.historical_resolution for h in candidate.historical_support]
            if resolutions and len(set(resolutions)) > 1:
                penalty += self.config.conflict_penalty_factor * 0.2

        return min(self.config.conflict_penalty_factor, penalty)
