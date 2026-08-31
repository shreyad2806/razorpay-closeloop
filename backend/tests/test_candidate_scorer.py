"""
Tests for Razorpay CloseLoop Phase 5C — Resolution Candidate Scoring.

Tests cover:
- ScoringConfig validation
- CandidateScore structure
- Evidence scoring
- ML scoring
- Historical scoring
- Financial consistency scoring
- Novelty penalty
- Conflict penalty
- Composite scoring
- Ranking behavior
- Strong vs weak evidence
- Strong vs weak ML
- Strong vs weak historical
- Perfect vs inconsistent financial
- Conflicting evidence
- Novel cases
- Ground truth separation
"""

import pytest

from app.schemas.evidence import EvidencePackage, EvidenceRecord
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.intelligence import (
    ClassificationResult,
    ExceptionIntelligence,
    EvidenceIntelligence,
    RecommendationStatus,
    SimilarCasesIntelligence,
)
from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    CandidateRanking,
    CandidateSource,
    EvidenceRecordRef,
    FinancialAdjustment,
    HistoricalSupportDetail,
    MLSupportDetail,
    ResolutionProposal,
)
from app.schemas.candidate_scoring import CandidateScore, ScoringConfig
from app.services.candidate_scorer import CandidateScoringService, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_intelligence(
    difference=3000,
    expected_amount=100000,
    actual_amount=97000,
    evidence_coverage=1.0,
    consistency=0.85,
    has_conflict=False,
    ml_agreement=True,
    similar_cases=None,
):
    """Create an ExceptionIntelligence for testing."""
    return ExceptionIntelligence(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="FEE_DIFFERENCE" if ml_agreement else "REFUND_ADJUSTMENT",
            agreement=ml_agreement,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=-abs(difference),
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"],
            evidence_coverage=evidence_coverage,
            consistency_score=consistency,
            has_conflict=has_conflict,
        ),
        similar_cases=similar_cases or SimilarCasesIntelligence(),
        recommendation_status=RecommendationStatus.SUPPORTED,
    )


def _make_candidate(
    resolution_type="FEE_ADJUSTMENT",
    adjustment_amount=3000,
    adjustment_direction="CREDIT",
    evidence_ids=None,
    evidence_compatible=True,
    evidence_coverage=1.0,
    ml_support=None,
    historical_support=None,
    sources=None,
):
    """Create a ResolutionProposal for testing."""
    return ResolutionProposal(
        candidate_id="CAND-001-DET",
        exception_id="EXC-001",
        case_id="CASE-001",
        resolution_type=resolution_type,
        resolution_description="Apply correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION" if adjustment_amount > 0 else "NO_ADJUSTMENT",
            amount_paise=adjustment_amount,
            direction=adjustment_direction,
            evidence_record_id="FEE-001" if adjustment_amount > 0 else None,
            calculation_basis="fee_record_sum" if adjustment_amount > 0 else "zero_discrepancy",
        ),
        supporting_evidence_ids=evidence_ids or [],
        evidence_records=[],
        evidence_compatible=evidence_compatible,
        evidence_coverage=evidence_coverage,
        ml_support=ml_support,
        historical_support=historical_support or [],
        sources=sources or ["deterministic_evidence"],
        ranking=CandidateRanking(rank=1, confidence_score=0.5, evidence_support=0.5),
        rationale="Test rationale",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScoringConfig:
    def test_default_weights_sum_to_one(self):
        config = ScoringConfig()
        assert config.validate_weights()

    def test_custom_weights(self):
        config = ScoringConfig(
            evidence_weight=0.4,
            ml_weight=0.1,
            historical_weight=0.1,
            financial_weight=0.4,
        )
        assert config.validate_weights()

    def test_invalid_weights(self):
        config = ScoringConfig(
            evidence_weight=0.5,
            ml_weight=0.5,
            historical_weight=0.5,
            financial_weight=0.5,
        )
        assert not config.validate_weights()


class TestCandidateScore:
    def test_basic_creation(self):
        score = CandidateScore(
            evidence_score=0.8,
            ml_score=0.7,
            historical_score=0.6,
            financial_consistency_score=0.9,
            final_score=0.75,
        )
        assert score.final_score == 0.75
        assert score.novelty_penalty == 0.0
        assert score.conflict_penalty == 0.0

    def test_explanation(self):
        score = CandidateScore(
            evidence_score=0.8,
            ml_score=0.7,
            historical_score=0.6,
            financial_consistency_score=0.9,
            novelty_penalty=0.05,
            conflict_penalty=0.1,
            final_score=0.65,
            weighted_evidence=0.28,
            weighted_ml=0.14,
            weighted_historical=0.09,
            weighted_financial=0.27,
        )
        explanation = score.explanation()
        assert "Evidence:" in explanation
        assert "ML:" in explanation
        assert "Final:" in explanation


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceScoring:
    def test_strong_evidence(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=["FEE-001"],
            evidence_compatible=True,
            evidence_coverage=1.0,
        )
        intel = _make_intelligence(evidence_coverage=1.0, consistency=0.85)
        score = scorer.score_candidate(candidate, intel)
        assert score.evidence_score > 0.7
        assert score.has_evidence_support is True

    def test_weak_evidence(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=[],
            evidence_compatible=False,
            evidence_coverage=0.0,
        )
        intel = _make_intelligence(evidence_coverage=0.0, consistency=0.2)
        score = scorer.score_candidate(candidate, intel)
        assert score.evidence_score < 0.3
        assert score.has_evidence_support is False

    def test_multiple_evidence_records(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=["FEE-001", "REF-001", "TAX-001"],
            evidence_compatible=True,
            evidence_coverage=0.9,
        )
        intel = _make_intelligence(evidence_coverage=0.9, consistency=0.8)
        score = scorer.score_candidate(candidate, intel)
        assert score.evidence_score > 0.6


class TestMLScoring:
    def test_strong_ml_support(self):
        scorer = CandidateScoringService()
        ml_detail = MLSupportDetail(
            supported=True,
            predicted_resolution="FEE_ADJUSTMENT",
            confidence=0.9,
            probability=0.85,
        )
        candidate = _make_candidate(ml_support=ml_detail)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.ml_score > 0.7
        assert score.has_ml_support is True

    def test_weak_ml_support(self):
        scorer = CandidateScoringService()
        ml_detail = MLSupportDetail(
            supported=True,
            predicted_resolution="FEE_ADJUSTMENT",
            confidence=0.3,
            probability=0.25,
        )
        candidate = _make_candidate(ml_support=ml_detail)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.ml_score < 0.5

    def test_no_ml_support(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(ml_support=None)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.ml_score == 0.0
        assert score.has_ml_support is False


class TestHistoricalScoring:
    def test_strong_historical_support(self):
        scorer = CandidateScoringService()
        hist = [
            HistoricalSupportDetail(
                case_id="CASE-H01",
                similarity_score=0.9,
                historical_resolution="FEE_ADJUSTMENT",
                historical_outcome="SUCCESSFUL",
            ),
            HistoricalSupportDetail(
                case_id="CASE-H02",
                similarity_score=0.85,
                historical_resolution="FEE_ADJUSTMENT",
                historical_outcome="SUCCESSFUL",
            ),
        ]
        candidate = _make_candidate(historical_support=hist)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.historical_score > 0.6
        assert score.has_historical_support is True

    def test_weak_historical_support(self):
        scorer = CandidateScoringService()
        hist = [
            HistoricalSupportDetail(
                case_id="CASE-H03",
                similarity_score=0.4,
                historical_resolution="FEE_ADJUSTMENT",
            ),
        ]
        candidate = _make_candidate(historical_support=hist)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.historical_score <= 0.5  # Weak but not zero

    def test_no_historical_support(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(historical_support=[])
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.historical_score == 0.0


class TestFinancialScoring:
    def test_perfect_financial_consistency(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            adjustment_amount=3000,
            adjustment_direction="CREDIT",
        )
        intel = _make_intelligence(difference=3000)
        score = scorer.score_candidate(candidate, intel)
        assert score.financial_consistency_score > 0.8

    def test_inconsistent_adjustment(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            adjustment_amount=5000,  # More than discrepancy
            adjustment_direction="CREDIT",
        )
        intel = _make_intelligence(difference=3000)
        score = scorer.score_candidate(candidate, intel)
        assert score.financial_consistency_score < 0.7

    def test_wrong_direction(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            adjustment_amount=3000,
            adjustment_direction="DEBIT",  # Wrong for positive difference
        )
        intel = _make_intelligence(difference=3000)
        score = scorer.score_candidate(candidate, intel)
        # Wrong direction reduces score but amount match gives partial credit
        assert score.financial_consistency_score < 0.8

    def test_no_adjustment_with_discrepancy(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            adjustment_amount=0,
            adjustment_direction="NONE",
        )
        intel = _make_intelligence(difference=3000)
        score = scorer.score_candidate(candidate, intel)
        assert score.financial_consistency_score == 0.0

    def test_no_discrepancy_no_adjustment(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            adjustment_amount=0,
            adjustment_direction="NONE",
        )
        intel = _make_intelligence(difference=0, expected_amount=100000, actual_amount=100000)
        score = scorer.score_candidate(candidate, intel)
        assert score.financial_consistency_score == 1.0


class TestNoveltyPenalty:
    def test_novel_case_penalty(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(historical_support=[])
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.novelty_penalty > 0.0
        assert score.is_novel is True

    def test_familiar_case_no_penalty(self):
        scorer = CandidateScoringService()
        hist = [
            HistoricalSupportDetail(
                case_id="CASE-H01",
                similarity_score=0.9,
                historical_resolution="FEE_ADJUSTMENT",
            ),
        ]
        candidate = _make_candidate(historical_support=hist)
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.novelty_penalty == 0.0
        assert score.is_novel is False


class TestConflictPenalty:
    def test_conflict_penalty(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(evidence_compatible=False)
        intel = _make_intelligence(has_conflict=True)
        score = scorer.score_candidate(candidate, intel)
        assert score.conflict_penalty > 0.0
        assert score.has_conflicts is True

    def test_no_conflict_no_penalty(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(evidence_compatible=True)
        intel = _make_intelligence(has_conflict=False)
        score = scorer.score_candidate(candidate, intel)
        assert score.conflict_penalty == 0.0


class TestCompositeScoring:
    def test_final_score_range(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=["FEE-001"],
            evidence_compatible=True,
            evidence_coverage=1.0,
        )
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert 0.0 <= score.final_score <= 1.0

    def test_strong_candidate_scores_high(self):
        scorer = CandidateScoringService()
        hist = [
            HistoricalSupportDetail(
                case_id="CASE-H01",
                similarity_score=0.9,
                historical_resolution="FEE_ADJUSTMENT",
            ),
        ]
        ml_detail = MLSupportDetail(
            supported=True,
            confidence=0.9,
            probability=0.85,
        )
        candidate = _make_candidate(
            evidence_ids=["FEE-001"],
            evidence_compatible=True,
            evidence_coverage=1.0,
            ml_support=ml_detail,
            historical_support=hist,
        )
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.final_score > 0.6

    def test_weak_candidate_scores_low(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=[],
            evidence_compatible=False,
            evidence_coverage=0.0,
            ml_support=None,
            historical_support=[],
        )
        intel = _make_intelligence(evidence_coverage=0.0, consistency=0.2)
        score = scorer.score_candidate(candidate, intel)
        assert score.final_score < 0.4


class TestRankingBehavior:
    def test_strong_candidate_ranked_first(self):
        scorer = CandidateScoringService()
        hist = [
            HistoricalSupportDetail(
                case_id="CASE-H01",
                similarity_score=0.9,
                historical_resolution="FEE_ADJUSTMENT",
            ),
        ]
        ml_detail = MLSupportDetail(
            supported=True,
            confidence=0.9,
            probability=0.85,
        )
        strong = _make_candidate(
            evidence_ids=["FEE-001"],
            evidence_compatible=True,
            evidence_coverage=1.0,
            ml_support=ml_detail,
            historical_support=hist,
        )
        weak = _make_candidate(
            resolution_type="REFUND_ADJUSTMENT",
            adjustment_amount=3000,
            evidence_ids=[],
            evidence_compatible=False,
            evidence_coverage=0.0,
            sources=["ml_prediction"],
        )

        result = CandidateGenerationResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            status="CANDIDATES_GENERATED",
            candidates=[weak, strong],  # Weak first
        )
        intel = _make_intelligence()

        ranked = scorer.score_and_rank(result, intel)
        assert ranked.candidates[0].resolution_type == "FEE_ADJUSTMENT"
        assert ranked.candidates[0].ranking.rank == 1

    def test_ranking_updates_confidence(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate(
            evidence_ids=["FEE-001"],
            evidence_compatible=True,
        )
        result = CandidateGenerationResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            status="CANDIDATES_GENERATED",
            candidates=[candidate],
        )
        intel = _make_intelligence()

        ranked = scorer.score_and_rank(result, intel)
        assert ranked.candidates[0].ranking.confidence_score > 0.0


class TestScorerSafety:
    def test_no_ground_truth_in_scorer(self):
        """CandidateScoringService must not use ground truth."""
        import inspect

        source = inspect.getsource(CandidateScoringService)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source

    def test_scores_are_floats(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate()
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert isinstance(score.evidence_score, float)
        assert isinstance(score.ml_score, float)
        assert isinstance(score.historical_score, float)
        assert isinstance(score.financial_consistency_score, float)
        assert isinstance(score.final_score, float)

    def test_penalties_non_negative(self):
        scorer = CandidateScoringService()
        candidate = _make_candidate()
        intel = _make_intelligence()
        score = scorer.score_candidate(candidate, intel)
        assert score.novelty_penalty >= 0.0
        assert score.conflict_penalty >= 0.0
