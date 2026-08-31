"""
Deterministic evidence quality scoring for Razorpay CloseLoop.

Measures how well available financial evidence supports the explanation
for an exception. These are evidence quality scores, NOT:
- ML prediction confidence
- Resolution confidence
- Authorization to auto-resolve

All scoring is deterministic. No ML, no LLM, no probabilistic reasoning.

Scoring formulas are documented and traceable to observable evidence.
"""

from typing import List

from app.schemas.evidence import EvidencePackage
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
from app.schemas.explanation import ExplanationResult, ExplanationStatus

# ─────────────────────────────────────────────────────────────────────────────
# Consistency Score Deductions
# ─────────────────────────────────────────────────────────────────────────────
#
# The consistency score starts at 1.0 and is reduced by observed inconsistencies.
#
# Deductions:
#   - Missing expected evidence:    -0.15 per missing expected record
#   - Structural conflicts:         -0.20 per conflict
#   - No supporting evidence:       -0.30 when discrepancy exists but no evidence
#   - Unexplained remainder:        proportional to remaining / difference
#
# Additions:
#   - None (score starts at maximum and only decreases)
#
# The score is clamped to [0.0, 1.0].

MISSING_EVIDENCE_PENALTY = 0.15
CONFLICT_PENALTY = 0.20
NO_EVIDENCE_PENALTY = 0.30


class EvidenceQualityScorer:
    """
    Deterministic evidence quality scorer.

    Given an EvidencePackage and ExplanationResult, produces evidence
    quality scores that measure how well the evidence supports the explanation.
    """

    def score(
        self, package: EvidencePackage, explanation: ExplanationResult
    ) -> EvidenceQualityResult:
        """
        Produce deterministic evidence quality scores.

        Args:
            package: The EvidencePackage from evidence retrieval
            explanation: The ExplanationResult from the explanation engine

        Returns:
            EvidenceQualityResult with scores and indicators
        """
        # 1. Calculate consistency score
        consistency, breakdown = self._calculate_consistency(package, explanation)

        # 2. Calculate coverage score
        coverage = self._calculate_coverage(package, explanation)

        # 3. Determine conflict indicator
        conflict = explanation.conflict or package.has_conflicts()

        # 4. Determine novelty (deterministic at this stage)
        novelty = self._determine_novelty(package, explanation)

        # 5. Collect missing evidence
        missing = [m.entity_type for m in package.missing_evidence]

        # 6. Determine explanation flags
        fully_explained = explanation.explanation_status == ExplanationStatus.FULLY_EXPLAINED
        partially_explained = explanation.explanation_status == ExplanationStatus.PARTIALLY_EXPLAINED

        # 7. Count supporting evidence
        evidence_count = len(explanation.supporting_evidence_ids)

        return EvidenceQualityResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            consistency_score=consistency,
            coverage_score=coverage,
            conflict=conflict,
            novelty=novelty,
            missing_evidence=missing,
            fully_explained=fully_explained,
            partially_explained=partially_explained,
            supporting_evidence_count=evidence_count,
            consistency_breakdown=breakdown,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Consistency Score Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_consistency(
        self, package: EvidencePackage, explanation: ExplanationResult
    ) -> tuple:
        """
        Calculate consistency score from observable evidence.

        Returns (score, breakdown_dict).
        """
        score = 1.0
        breakdown = {
            "base": 1.0,
            "missing_evidence_penalty": 0.0,
            "conflict_penalty": 0.0,
            "no_evidence_penalty": 0.0,
            "remainder_penalty": 0.0,
        }

        # Penalty for missing expected evidence
        missing_expected = [
            m for m in package.missing_evidence if m.expected
        ]
        if missing_expected:
            penalty = len(missing_expected) * MISSING_EVIDENCE_PENALTY
            score -= penalty
            breakdown["missing_evidence_penalty"] = -penalty

        # Penalty for structural conflicts in evidence
        if package.has_conflicts():
            penalty = len(package.conflicts) * CONFLICT_PENALTY
            score -= penalty
            breakdown["conflict_penalty"] = -penalty

        # Penalty for explanation conflict
        if explanation.conflict:
            score -= CONFLICT_PENALTY
            breakdown["conflict_penalty"] -= CONFLICT_PENALTY

        # Penalty when discrepancy exists but no supporting evidence
        has_discrepancy = abs(package.difference) >= 1
        has_evidence = len(explanation.supporting_evidence_ids) > 0
        if has_discrepancy and not has_evidence:
            score -= NO_EVIDENCE_PENALTY
            breakdown["no_evidence_penalty"] = -NO_EVIDENCE_PENALTY

        # Penalty for unexplained remainder (proportional)
        if (
            has_discrepancy
            and explanation.remaining_difference != 0
            and explanation.explanation_status != ExplanationStatus.FULLY_EXPLAINED
        ):
            remainder_ratio = abs(explanation.remaining_difference) / abs(package.difference)
            penalty = remainder_ratio * 0.3  # Up to 0.3 for fully unexplained
            score -= penalty
            breakdown["remainder_penalty"] = -penalty

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        return round(score, 4), breakdown

    # ─────────────────────────────────────────────────────────────────────────
    # Coverage Score Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_coverage(
        self, package: EvidencePackage, explanation: ExplanationResult
    ) -> float:
        """
        Calculate coverage score: explained_amount / |discrepancy|.

        Handles zero discrepancy safely.
        """
        difference = package.difference

        # Zero discrepancy = perfect coverage (nothing to explain)
        if abs(difference) < 1:
            return 1.0

        # Coverage = |explained_amount| / |difference|
        explained = abs(explanation.explained_amount)
        discrepancy = abs(difference)

        coverage = explained / discrepancy

        # Clamp to [0.0, 1.0]
        return round(max(0.0, min(1.0, coverage)), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Novelty Determination
    # ─────────────────────────────────────────────────────────────────────────

    def _determine_novelty(
        self, package: EvidencePackage, explanation: ExplanationResult
    ) -> NoveltyLevel:
        """
        Determine novelty using deterministic indicators only.

        At this stage (Phase 3), novelty is determined by:
        - Whether the exception type is KNOWN (EXACT_MATCH, FEE_DIFFERENCE, etc.)
        - Whether the explanation is UNEXPLAINED (suggesting novel pattern)

        True semantic similarity/novelty via embeddings will be added in Phase 4.
        """
        # Known exception types with available evidence are not novel
        known_types = {
            "EXACT_MATCH",
            "FEE_DIFFERENCE",
            "REFUND_ADJUSTMENT",
            "TAX_ADJUSTMENT",
            "TIMING_DIFFERENCE",
            "PARTIAL_SETTLEMENT",
            "DUPLICATE",
            "MISSING_RECORD",
        }

        if package.exception_type in known_types:
            # Known pattern — but check if it's actually explained
            if explanation.explanation_status in [
                ExplanationStatus.FULLY_EXPLAINED,
                ExplanationStatus.PARTIALLY_EXPLAINED,
            ]:
                return NoveltyLevel.KNOWN_PATTERN

        # UNKNOWN or unexplained = novel (no historical match available)
        return NoveltyLevel.NOVEL_NO_HISTORICAL
