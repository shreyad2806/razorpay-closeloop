"""
Comprehensive unit tests for resolution candidate generation, scoring, and selection.

Tests:
- CandidateGenerator.generate() -- candidate creation from intelligence
- CandidateScoringService.score_candidate() -- scoring components
- CandidateScoringService.score_and_rank() -- ranking
- CandidateSelector.select() -- selection with safety thresholds
- No-forcing rule -- UNRESOLVED when no candidate is safe
- Deterministic ordering -- same input produces same ranking
- Safety -- conflicting evidence cannot produce high-confidence candidate
"""

from datetime import datetime, timezone

import pytest

from app.schemas.intelligence import (
    ClassificationResult,
    EvidenceIntelligence,
    ExceptionIntelligence,
    RecommendationStatus,
    SimilarCasesIntelligence,
)
from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    CandidateRanking,
    HistoricalSupportDetail,
    ResolutionProposal,
)
from app.schemas.candidate_scoring import CandidateScore, ScoringConfig
from app.schemas.resolution_selection import SelectionConfig, SelectionStatus
from app.services.candidate_generator import CandidateGenerator
from app.services.candidate_scorer import CandidateScoringService
from app.services.candidate_selector import CandidateSelector

NOW = datetime.now(timezone.utc)


# ============================================================================
# HELPERS
# ============================================================================

def _intelligence(
    exception_id="EXC-001",
    deterministic_type="FEE_DIFFERENCE",
    ml_predicted_type=None,
    ml_probabilities=None,
    agreement=True,
    difference=5000,
    evidence_coverage=0.8,
    consistency_score=0.9,
    has_conflict=False,
    similar_cases=None,
    ml_model_version="v1.0",
    recommendation_status=None,
):
    """Build an ExceptionIntelligence fixture."""
    if recommendation_status is None:
        if has_conflict:
            recommendation_status = RecommendationStatus.CONFLICTING
        else:
            recommendation_status = RecommendationStatus.SUPPORTED

    return ExceptionIntelligence(
        exception_id=exception_id,
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=100000 - difference,
        difference=difference,
        classification=ClassificationResult(
            deterministic_type=deterministic_type,
            ml_predicted_type=ml_predicted_type,
            ml_probabilities=ml_probabilities,
            ml_model_version=ml_model_version,
            agreement=agreement,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=abs(difference),
            remaining_difference=0,
            supporting_evidence_ids=["EVD-001", "EVD-002"],
            evidence_coverage=evidence_coverage,
            consistency_score=consistency_score,
            has_conflict=has_conflict,
        ),
        similar_cases=SimilarCasesIntelligence(
            similar_cases=similar_cases or [],
        ),
        recommendation_status=recommendation_status,
    )


def _gen_result(exception_id="EXC-001", status="CANDIDATES_GENERATED", candidates=None):
    """Build a CandidateGenerationResult fixture."""
    return CandidateGenerationResult(
        exception_id=exception_id,
        case_id="CASE-001",
        status=status,
        candidates=candidates or [],
        total_candidates=len(candidates) if candidates else 0,
    )


def _proposal(
    resolution_type="FEE_ADJUSTMENT",
    confidence=0.7,
    amount=5000,
    evidence_compatible=True,
    evidence_ids=None,
    sources=None,
    has_ml=False,
    has_historical=False,
    exception_id="EXC-001",
):
    """Build a ResolutionProposal fixture."""
    from app.schemas.resolution_candidate import (
        FinancialAdjustment,
        MLSupportDetail,
    )

    ml_support = None
    if has_ml:
        ml_support = MLSupportDetail(
            supported=True, predicted_resolution=resolution_type,
            confidence=confidence, model_version="v1.0",
        )

    historical_support = []
    if has_historical:
        historical_support = [
            HistoricalSupportDetail(
                case_id="HIST-001",
                similarity_score=0.85,
                historical_resolution=resolution_type,
                historical_outcome="APPROVED",
            )
        ]

    return ResolutionProposal(
        candidate_id=f"CAND-{exception_id}-{resolution_type}",
        exception_id=exception_id,
        case_id="CASE-001",
        resolution_type=resolution_type,
        resolution_description=f"Apply {resolution_type}",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION" if "FEE" in resolution_type else "SETTLEMENT_ADJUSTMENT",
            amount_paise=amount,
            direction="CREDIT",
            calculation_basis="evidence_trace",
        ),
        supporting_evidence_ids=evidence_ids or ["EVD-001"],
        evidence_compatible=evidence_compatible,
        evidence_coverage=0.8,
        ml_support=ml_support,
        historical_support=historical_support,
        sources=sources or ["deterministic_evidence"],
        ranking=CandidateRanking(rank=0, confidence_score=confidence, evidence_support=0.8),
        rationale=f"Resolution: {resolution_type}",
    )


# ============================================================================
# 1. CANDIDATE GENERATION
# ============================================================================

class TestCandidateGeneration:
    """Test CandidateGenerator.generate() for various scenarios."""

    def test_clear_single_candidate(self):
        """Fee difference -> FEE_ADJUSTMENT candidate generated."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="FEE_DIFFERENCE", difference=5000)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert len(result.candidates) >= 1
        types = [c.resolution_type for c in result.candidates]
        assert "FEE_ADJUSTMENT" in types

    def test_exact_match_no_action(self):
        """Exact match -> NO_ACTION candidate."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="EXACT_MATCH", difference=0)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(c.resolution_type == "NO_ACTION" for c in result.candidates)

    def test_missing_record_escalation(self):
        """Missing record -> MISSING_RECORD_ESCALATION candidate."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="MISSING_RECORD", difference=100000)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        types = [c.resolution_type for c in result.candidates]
        assert "MISSING_RECORD_ESCALATION" in types

    def test_duplicate_settlement(self):
        """Duplicate -> DUPLICATE_SETTLEMENT candidate."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="DUPLICATE", difference=0)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        types = [c.resolution_type for c in result.candidates]
        assert "DUPLICATE_SETTLEMENT" in types

    def test_unknown_unresolved(self):
        """Unknown type -> UNKNOWN_UNRESOLVED candidate."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="UNKNOWN", difference=50)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        types = [c.resolution_type for c in result.candidates]
        assert "UNKNOWN_UNRESOLVED" in types

    def test_multiple_valid_candidates(self):
        """ML predicts different type -> multiple candidates."""
        gen = CandidateGenerator()
        intel = _intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="REFUND_ADJUSTMENT",
            ml_probabilities={"REFUND_ADJUSTMENT": 0.7, "FEE_DIFFERENCE": 0.3},
            agreement=False,
            difference=5000,
        )
        result = gen.generate(intel)
        types = [c.resolution_type for c in result.candidates]
        assert len(set(types)) >= 2  # At least 2 different types

    def test_ml_same_as_deterministic_not_duplicated(self):
        """ML predicts same as deterministic -> not a separate candidate."""
        gen = CandidateGenerator()
        intel = _intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="FEE_DIFFERENCE",
            ml_probabilities={"FEE_DIFFERENCE": 0.9},
            agreement=True,
        )
        result = gen.generate(intel)
        fee_candidates = [c for c in result.candidates if c.resolution_type == "FEE_ADJUSTMENT"]
        # Should be merged into one, not two separate
        assert len(fee_candidates) <= 1

    def test_no_valid_candidate_unresolved(self):
        """When no candidate can be generated -> UNRESOLVED."""
        gen = CandidateGenerator()
        # Use a type that doesn't map to any resolution
        intel = _intelligence(deterministic_type="NONEXISTENT_TYPE", difference=5000)
        result = gen.generate(intel)
        assert result.status == "UNRESOLVED"
        assert len(result.candidates) == 0

    def test_candidates_are_recommendation_only(self):
        """All candidates must have is_recommendation_only=True."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="FEE_DIFFERENCE", difference=5000)
        result = gen.generate(intel)
        for c in result.candidates:
            assert c.is_recommendation_only is True

    def test_financial_adjustment_traced_to_evidence(self):
        """Financial adjustment must have a calculation_basis (traceability)."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="FEE_DIFFERENCE", difference=5000)
        result = gen.generate(intel)
        for c in result.candidates:
            assert c.financial_adjustment.calculation_basis is not None
            assert len(c.financial_adjustment.calculation_basis) > 0


# ============================================================================
# 2. CANDIDATE SCORING COMPONENTS
# ============================================================================

class TestCandidateScoring:
    """Test individual scoring components."""

    def test_evidence_score_with_coverage(self):
        """High evidence coverage -> higher evidence score."""
        scorer = CandidateScoringService()
        c = _proposal(evidence_ids=["EVD-001", "EVD-002", "EVD-003"])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9)
        score = scorer.score_candidate(c, intel)
        assert score.evidence_score > 0.5

    def test_evidence_score_without_evidence(self):
        """No evidence -> lower evidence score."""
        scorer = CandidateScoringService()
        c = _proposal(evidence_ids=[])
        intel = _intelligence(evidence_coverage=0.0, consistency_score=0.0)
        score = scorer.score_candidate(c, intel)
        assert score.evidence_score < 0.3

    def test_ml_score_with_ml_support(self):
        """Candidate with ML support -> ml_score > 0."""
        scorer = CandidateScoringService()
        c = _proposal(has_ml=True, confidence=0.8)
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.ml_score > 0.0
        assert score.has_ml_support is True

    def test_ml_score_without_ml_support(self):
        """Candidate without ML support -> ml_score = 0."""
        scorer = CandidateScoringService()
        c = _proposal(has_ml=False)
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.ml_score == 0.0
        assert score.has_ml_support is False

    def test_historical_score_with_cases(self):
        """Candidate with historical support -> historical_score > 0."""
        scorer = CandidateScoringService()
        c = _proposal(has_historical=True)
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.historical_score > 0.0
        assert score.has_historical_support is True

    def test_historical_score_without_cases(self):
        """No historical support -> historical_score = 0."""
        scorer = CandidateScoringService()
        c = _proposal()
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.historical_score == 0.0

    def test_financial_score_perfect_match(self):
        """Adjustment matches discrepancy perfectly -> high financial score."""
        scorer = CandidateScoringService()
        c = _proposal(amount=5000)
        intel = _intelligence(difference=5000)
        score = scorer.score_candidate(c, intel)
        assert score.financial_consistency_score > 0.7

    def test_financial_score_no_adjustment_with_discrepancy(self):
        """No adjustment but discrepancy exists -> low financial score."""
        scorer = CandidateScoringService()
        c = _proposal(amount=0)
        intel = _intelligence(difference=5000)
        score = scorer.score_candidate(c, intel)
        assert score.financial_consistency_score == 0.0

    def test_novelty_penalty_no_historical(self):
        """No historical support -> novelty penalty > 0."""
        scorer = CandidateScoringService()
        c = _proposal()
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.novelty_penalty > 0.0
        assert score.is_novel is True

    def test_novelty_penalty_with_strong_historical(self):
        """Strong historical similarity -> no novelty penalty."""
        scorer = CandidateScoringService()
        c = _proposal(has_historical=True)
        c.historical_support[0].similarity_score = 0.9
        intel = _intelligence()
        score = scorer.score_candidate(c, intel)
        assert score.novelty_penalty == 0.0
        assert score.is_novel is False

    def test_conflict_penalty_with_evidence_conflict(self):
        """Evidence conflict -> conflict penalty > 0."""
        scorer = CandidateScoringService()
        c = _proposal()
        intel = _intelligence(has_conflict=True)
        score = scorer.score_candidate(c, intel)
        assert score.conflict_penalty > 0.0
        assert score.has_conflicts is True

    def test_conflict_penalty_no_conflicts(self):
        """No conflicts -> conflict penalty = 0."""
        scorer = CandidateScoringService()
        c = _proposal()
        intel = _intelligence(has_conflict=False)
        score = scorer.score_candidate(c, intel)
        assert score.conflict_penalty == 0.0

    def test_final_score_clamped_0_1(self):
        """Final score is always between 0 and 1."""
        scorer = CandidateScoringService()
        c = _proposal(confidence=1.0, amount=5000, evidence_ids=["E1", "E2", "E3"])
        intel = _intelligence(evidence_coverage=1.0, consistency_score=1.0, difference=5000)
        score = scorer.score_candidate(c, intel)
        assert 0.0 <= score.final_score <= 1.0

    def test_scoring_deterministic(self):
        """Same candidate + same intel -> same score."""
        scorer = CandidateScoringService()
        c = _proposal(amount=5000)
        intel = _intelligence(difference=5000)
        s1 = scorer.score_candidate(c, intel)
        s2 = scorer.score_candidate(c, intel)
        assert s1.final_score == s2.final_score
        assert s1.evidence_score == s2.evidence_score


# ============================================================================
# 3. RANKING
# ============================================================================

class TestRanking:
    """Test score_and_rank produces correct ordering."""

    def test_higher_evidence_ranks_higher(self):
        """Candidate with better evidence should rank higher."""
        scorer = CandidateScoringService()
        c1 = _proposal(resolution_type="FEE_ADJUSTMENT", evidence_ids=["E1", "E2", "E3"])
        c2 = _proposal(resolution_type="REFUND_ADJUSTMENT", evidence_ids=[])
        gen = _gen_result(candidates=[c1, c2])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9)
        result = scorer.score_and_rank(gen, intel)
        # c1 has more evidence -> should rank higher
        assert result.candidates[0].resolution_type == "FEE_ADJUSTMENT"

    def test_ranking_assigns_sequential_ranks(self):
        """Ranks are 1, 2, 3..."""
        scorer = CandidateScoringService()
        c1 = _proposal(resolution_type="FEE_ADJUSTMENT", confidence=0.9)
        c2 = _proposal(resolution_type="REFUND_ADJUSTMENT", confidence=0.5)
        gen = _gen_result(candidates=[c1, c2])
        intel = _intelligence()
        result = scorer.score_and_rank(gen, intel)
        ranks = [c.ranking.rank for c in result.candidates]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1


# ============================================================================
# 4. CANDIDATE SELECTION
# ============================================================================

class TestCandidateSelection:
    """Test CandidateSelector.select() with various scenarios."""

    def test_select_clear_winner(self):
        """One strong candidate with historical support -> RECOMMENDED."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.9, amount=5000, evidence_ids=["E1", "E2"], has_historical=True)
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9, difference=5000)
        result = selector.select(gen, intel)
        assert result.status == SelectionStatus.RECOMMENDED
        assert result.selected_candidate is not None

    def test_select_no_candidates_unresolved(self):
        """No candidates -> UNRESOLVED."""
        selector = CandidateSelector()
        gen = _gen_result(candidates=[], status="UNRESOLVED")
        intel = _intelligence()
        result = selector.select(gen, intel)
        assert result.status == SelectionStatus.UNRESOLVED

    def test_select_low_score_unresolved(self):
        """All candidates below threshold -> UNRESOLVED (no-forcing rule)."""
        selector = CandidateSelector(
            selection_config=SelectionConfig(min_final_score=0.9)  # Very high threshold
        )
        c = _proposal(confidence=0.3, amount=5000)
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.1, consistency_score=0.1, difference=5000)
        result = selector.select(gen, intel)
        # Score will be low -> UNRESOLVED
        assert result.status == SelectionStatus.UNRESOLVED

    def test_select_close_candidates_human_review(self):
        """Two close candidates with historical support -> HUMAN_REVIEW."""
        selector = CandidateSelector(
            selection_config=SelectionConfig(min_margin_over_second=0.3)  # High margin required
        )
        c1 = _proposal(resolution_type="FEE_ADJUSTMENT", confidence=0.7, amount=5000, has_historical=True)
        c2 = _proposal(resolution_type="REFUND_ADJUSTMENT", confidence=0.65, amount=5000, has_historical=True)
        gen = _gen_result(candidates=[c1, c2])
        intel = _intelligence(evidence_coverage=0.8, consistency_score=0.8, difference=5000)
        result = selector.select(gen, intel)
        # Close margin -> HUMAN_REVIEW
        assert result.status == SelectionStatus.HUMAN_REVIEW

    def test_select_weak_evidence_unresolved(self):
        """Weak evidence -> UNRESOLVED (no-forcing rule)."""
        selector = CandidateSelector(
            selection_config=SelectionConfig(min_evidence_coverage=0.5)
        )
        c = _proposal(evidence_ids=[], confidence=0.5, amount=5000)
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.1, consistency_score=0.1, difference=5000)
        result = selector.select(gen, intel)
        assert result.status == SelectionStatus.UNRESOLVED

    def test_selection_deterministic(self):
        """Same input -> same selection."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.8, amount=5000, evidence_ids=["E1", "E2"])
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9, difference=5000)
        r1 = selector.select(gen, intel)
        r2 = selector.select(gen, intel)
        assert r1.status == r2.status
        assert r1.confidence == r2.confidence

    def test_selection_has_risk_assessment(self):
        """Selection includes risk category."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.8, amount=5000, evidence_ids=["E1"], has_historical=True)
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.8, consistency_score=0.8, difference=5000)
        result = selector.select(gen, intel)
        assert result.risk_category in ("LOW", "MEDIUM", "HIGH")

    def test_selection_has_explainability(self):
        """Selection includes explainability detail."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.8, amount=5000, evidence_ids=["E1"], has_historical=True)
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.8, consistency_score=0.8, difference=5000)
        result = selector.select(gen, intel)
        assert result.explainability is not None
        assert result.explainability.level is not None


# ============================================================================
# 5. NO-FORCING RULE
# ============================================================================

class TestNoForcingRule:
    """When no candidate satisfies safety requirements, return UNRESOLVED."""

    def test_no_forcing_weak_evidence(self):
        """Weak evidence + low confidence -> UNRESOLVED, not forced resolution."""
        selector = CandidateSelector()
        c = _proposal(
            confidence=0.2, amount=5000, evidence_ids=[],
            evidence_compatible=False,
        )
        gen = _gen_result(candidates=[c])
        intel = _intelligence(
            evidence_coverage=0.1, consistency_score=0.1,
            has_conflict=True, difference=5000,
        )
        result = selector.select(gen, intel)
        # Must NOT recommend a resolution with such weak signals
        assert result.status in (SelectionStatus.UNRESOLVED, SelectionStatus.HUMAN_REVIEW)

    def test_no_forcing_conflicting_evidence(self):
        """Conflicting evidence -> HUMAN_REVIEW or UNRESOLVED."""
        selector = CandidateSelector()
        c1 = _proposal(resolution_type="FEE_ADJUSTMENT", confidence=0.6, amount=5000)
        c2 = _proposal(resolution_type="REFUND_ADJUSTMENT", confidence=0.55, amount=5000)
        gen = _gen_result(candidates=[c1, c2])
        intel = _intelligence(
            evidence_coverage=0.5, consistency_score=0.5,
            has_conflict=True, difference=5000,
        )
        result = selector.select(gen, intel)
        # Close candidates with conflict -> HUMAN_REVIEW
        assert result.status in (SelectionStatus.UNRESOLVED, SelectionStatus.HUMAN_REVIEW)

    def test_no_forcing_high_novelty(self):
        """Very novel case with no historical support -> lower confidence."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.6, amount=5000, evidence_ids=["E1"])
        gen = _gen_result(candidates=[c])
        intel = _intelligence(
            evidence_coverage=0.6, consistency_score=0.6,
            difference=5000, similar_cases=[],
        )
        result = selector.select(gen, intel)
        # Novel case should have reduced confidence
        if result.status == SelectionStatus.RECOMMENDED:
            assert result.confidence < 0.8  # Reduced by novelty penalty


# ============================================================================
# 6. SAFETY: CONFLICTING EVIDENCE
# ============================================================================

class TestSafetyConflictingEvidence:
    """Conflicting evidence cannot produce a high-confidence candidate."""

    def test_conflict_reduces_score(self):
        """Conflict penalty reduces final score."""
        scorer = CandidateScoringService()
        c = _proposal(amount=5000)
        intel_conflict = _intelligence(has_conflict=True, difference=5000)
        intel_no_conflict = _intelligence(has_conflict=False, difference=5000)
        s_conflict = scorer.score_candidate(c, intel_conflict)
        s_no_conflict = scorer.score_candidate(c, intel_no_conflict)
        assert s_conflict.final_score < s_no_conflict.final_score

    def test_incompatible_evidence_reduces_score(self):
        """Incompatible evidence reduces financial consistency score."""
        scorer = CandidateScoringService()
        c_compat = _proposal(evidence_compatible=True, amount=5000)
        c_incompat = _proposal(evidence_compatible=False, amount=5000)
        intel = _intelligence(difference=5000)
        s_compat = scorer.score_candidate(c_compat, intel)
        s_incompat = scorer.score_candidate(c_incompat, intel)
        assert s_incompat.conflict_penalty >= s_compat.conflict_penalty

    def test_conflict_prevents_high_confidence_selection(self):
        """Selection with conflict should not produce high confidence."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.8, amount=5000, evidence_ids=["E1"])
        gen = _gen_result(candidates=[c])
        intel = _intelligence(
            evidence_coverage=0.8, consistency_score=0.8,
            has_conflict=True, difference=5000,
        )
        result = selector.select(gen, intel)
        if result.status == SelectionStatus.RECOMMENDED:
            # Confidence should be reduced by conflict
            assert result.confidence < 0.9


# ============================================================================
# 7. SCORING WEIGHTS
# ============================================================================

class TestScoringWeights:
    """Verify scoring configuration affects output."""

    def test_default_weights_sum_to_one(self):
        """Default scoring weights sum to ~1.0."""
        config = ScoringConfig()
        assert config.validate_weights() is True

    def test_evidence_weight_affects_score(self):
        """Higher evidence weight increases evidence contribution."""
        config_high = ScoringConfig(evidence_weight=0.5)
        config_low = ScoringConfig(evidence_weight=0.1)
        scorer_high = CandidateScoringService(config=config_high)
        scorer_low = CandidateScoringService(config=config_low)
        c = _proposal(evidence_ids=["E1", "E2", "E3"])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9)
        s_high = scorer_high.score_candidate(c, intel)
        s_low = scorer_low.score_candidate(c, intel)
        # Higher evidence weight should produce higher weighted_evidence
        assert s_high.weighted_evidence > s_low.weighted_evidence


# ============================================================================
# 8. EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge cases for candidate generation and scoring."""

    def test_zero_difference_no_action(self):
        """Zero difference -> NO_ACTION candidate with zero adjustment."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="EXACT_MATCH", difference=0)
        result = gen.generate(intel)
        no_action = [c for c in result.candidates if c.resolution_type == "NO_ACTION"]
        assert len(no_action) >= 1
        assert no_action[0].financial_adjustment.amount_paise == 0

    def test_large_difference(self):
        """Large difference (100000 paise) -> candidate generated."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="TIMING_DIFFERENCE", difference=100000)
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert len(result.candidates) >= 1

    def test_high_risk_adjustment(self):
        """Large adjustment -> HIGH risk category."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.9, amount=100000, evidence_ids=["E1", "E2"])
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9, difference=100000)
        result = selector.select(gen, intel)
        if result.status == SelectionStatus.RECOMMENDED:
            assert result.risk_category == "HIGH"

    def test_medium_risk_adjustment(self):
        """Medium adjustment -> MEDIUM risk category."""
        selector = CandidateSelector()
        c = _proposal(confidence=0.9, amount=20000, evidence_ids=["E1", "E2"])
        gen = _gen_result(candidates=[c])
        intel = _intelligence(evidence_coverage=0.9, consistency_score=0.9, difference=20000)
        result = selector.select(gen, intel)
        if result.status == SelectionStatus.RECOMMENDED:
            assert result.risk_category in ("MEDIUM", "HIGH")

    def test_generator_pipeline_version(self):
        """Result includes pipeline version."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="FEE_DIFFERENCE", difference=5000)
        result = gen.generate(intel)
        assert result.pipeline_version == "1.0.0"

    def test_candidate_has_rationale(self):
        """Every candidate has a non-empty rationale."""
        gen = CandidateGenerator()
        intel = _intelligence(deterministic_type="FEE_DIFFERENCE", difference=5000)
        result = gen.generate(intel)
        for c in result.candidates:
            assert len(c.rationale) > 0
            assert len(c.rationale_components) > 0
