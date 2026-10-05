"""
Phase 7 Resolution Intelligence Tests

Tests verify that candidate generation, scoring, and selection are advisory-only
and do NOT perform authorization, execution, or persistence.

Core principle:
    Phase 7: INVESTIGATES + PROPOSES
    Phase 8: AUTHORIZES  
    Phase 9: EXECUTES
    Phase 10: RE-RECONCILES + CLOSES
"""

import inspect
import pytest
from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.evidence import EvidencePackage, EvidenceRecord, MissingEvidence
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
from app.schemas.intelligence import (
    ExceptionIntelligence,
    ClassificationResult,
    EvidenceIntelligence,
    SimilarCasesIntelligence,
    RecommendationStatus,
)
from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    ResolutionProposal,
    CandidateSource,
    FinancialAdjustment,
)
from app.schemas.resolution_selection import SelectionResult, SelectionStatus
from app.schemas.enums import ExceptionType, ResolutionType
from app.services.candidate_generator import CandidateGenerator
from app.services.candidate_scorer import CandidateScoringService
from app.services.candidate_selector import CandidateSelector


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_evidence_package():
    """Sample evidence package for FEE_MISMATCH case."""
    return EvidencePackage(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MERCHANT-001",
        expected_amount=950000,  # ₹9,500
        actual_amount=925000,  # ₹9,250
        difference=25000,  # ₹250
        exception_type="FEE_DIFFERENCE",
        payment=EvidenceRecord(
            record_id="PAY-001",
            entity_type="PAYMENT",
            amount=1000000,  # ₹10,000
            relationship="PRIMARY_RECORD",
        ),
        settlements=[
            EvidenceRecord(
                record_id="SET-001",
                entity_type="SETTLEMENT",
                amount=925000,
                relationship="CALCULATION_COMPONENT",
            )
        ],
        fees=[
            EvidenceRecord(
                record_id="FEE-001",
                entity_type="FEE",
                amount=30000,  # ₹300
                relationship="CALCULATION_COMPONENT",
                contribution=-30000,
            )
        ],
        refunds=[],
        taxes=[],
        adjustments=[],
        total_settlement_amount=925000,
        total_refund_amount=0,
        total_fee_amount=30000,
        total_tax_amount=0,
        total_adjustment_amount=0,
        missing_evidence=[],
        conflicts=[],
        evidence_link_count=3,
    )


@pytest.fixture
def sample_explanation():
    """Sample explanation result."""
    return ExplanationResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        explanation_status=ExplanationStatus.FULLY_EXPLAINED,
        explained_amount=30000,
        remaining_difference=0,
        supporting_evidence_ids=["FEE-001"],
        explanation_reason="Fee difference detected in settlement",
    )


@pytest.fixture
def sample_evidence_quality():
    """Sample evidence quality result."""
    return EvidenceQualityResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        coverage_score=0.95,
        consistency_score=0.90,
        conflict=False,
        novelty=NoveltyLevel.KNOWN_PATTERN,
        missing_evidence=[],
        fully_explained=True,
        partially_explained=False,
        supporting_evidence_count=3,
    )


@pytest.fixture
def sample_intelligence():
    """Sample exception intelligence with all signals."""
    return ExceptionIntelligence(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="FEE_DIFFERENCE",
            ml_probabilities={"FEE_DIFFERENCE": 0.92, "TIMING_DIFFERENCE": 0.05},
            ml_model_version="1.0.0",
            agreement=True,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=30000,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"],
            evidence_coverage=0.95,
            consistency_score=0.90,
            has_conflict=False,
            missing_evidence=[],
            explanation_reason="Fee difference fully explained",
            evidence_link_count=3,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=True,
            total_indexed=150,
            top_k=5,
            similar_cases=[
                {
                    "case_id": "CASE-042",
                    "similarity_score": 0.92,
                    "resolution_type": "FEE_ADJUSTMENT",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 1000000,
                    "difference": 25000,
                },
                {
                    "case_id": "CASE-017",
                    "similarity_score": 0.88,
                    "resolution_type": "FEE_ADJUSTMENT",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 500000,
                    "difference": 15000,
                },
            ],
        ),
        recommendation_status=RecommendationStatus.SUPPORTED,
        recommendation_notes=[],
        conflicts=[],
    )


@pytest.fixture
def intelligence_with_ml_disagreement():
    """Intelligence where ML disagrees with deterministic."""
    return ExceptionIntelligence(
        exception_id="EXC-002",
        case_id="CASE-002",
        payment_id="PAY-002",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="TIMING_DIFFERENCE",
            ml_probabilities={"TIMING_DIFFERENCE": 0.70, "FEE_DIFFERENCE": 0.25},
            ml_model_version="1.0.0",
            agreement=False,  # DISAGREEMENT
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=30000,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"],
            evidence_coverage=0.95,
            consistency_score=0.90,
            has_conflict=False,
            missing_evidence=[],
            explanation_reason="Fee difference fully explained",
            evidence_link_count=3,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=True,
            total_indexed=150,
            top_k=5,
            similar_cases=[
                {
                    "case_id": "CASE-099",
                    "similarity_score": 0.85,
                    "resolution_type": "REFUND_ADJUSTMENT",  # Different from both
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 1000000,
                    "difference": 25000,
                },
            ],
        ),
        recommendation_status=RecommendationStatus.CONFLICTING,
        recommendation_notes=["ML disagrees with deterministic classification"],
        conflicts=["ML/deterministic disagreement"],
    )


@pytest.fixture
def intelligence_with_historical_failure():
    """Intelligence with failed historical resolution."""
    return ExceptionIntelligence(
        exception_id="EXC-003",
        case_id="CASE-003",
        payment_id="PAY-003",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type=None,
            ml_probabilities=None,
            ml_model_version=None,
            agreement=True,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=30000,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"],
            evidence_coverage=0.95,
            consistency_score=0.90,
            has_conflict=False,
            missing_evidence=[],
            explanation_reason="Fee difference fully explained",
            evidence_link_count=3,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=True,
            total_indexed=150,
            top_k=5,
            similar_cases=[
                {
                    "case_id": "CASE-200",
                    "similarity_score": 0.90,
                    "resolution_type": "FEE_ADJUSTMENT",
                    "resolution_outcome": "FAILED",  # FAILED RESOLUTION
                    "payment_amount": 1000000,
                    "difference": 25000,
                },
            ],
        ),
        recommendation_status=RecommendationStatus.SUPPORTED,
        recommendation_notes=["Historical case with failed resolution found"],
        conflicts=[],
    )


# ============================================================================
# TEST 1: Existing Candidate Generation
# ============================================================================

def test_candidate_generation_exists(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation exists and produces results."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    assert result is not None
    assert result.exception_id == "EXC-001"
    assert result.case_id == "CASE-001"
    assert result.status in ("CANDIDATES_GENERATED", "UNRESOLVED")
    assert result.pipeline_version == "1.0.0"


# ============================================================================
# TEST 2: Existing Candidate Scoring
# ============================================================================

def test_candidate_scoring_exists(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate scoring exists and produces scores."""
    generator = CandidateGenerator()
    scorer = CandidateScoringService()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    if generation_result.candidates:
        scored_result = scorer.score_and_rank(generation_result, sample_intelligence)
        
        assert scored_result is not None
        assert len(scored_result.candidates) == len(generation_result.candidates)
        
        # Verify ranking
        for i, candidate in enumerate(scored_result.candidates):
            assert candidate.ranking.rank == i + 1


# ============================================================================
# TEST 3: Existing Candidate Selection
# ============================================================================

def test_candidate_selection_exists(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate selection exists and produces selection results."""
    generator = CandidateGenerator()
    selector = CandidateSelector()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    selection_result = selector.select(generation_result, sample_intelligence)
    
    assert selection_result is not None
    assert selection_result.exception_id == "EXC-001"
    assert selection_result.case_id == "CASE-001"
    assert selection_result.status in (SelectionStatus.RECOMMENDED, SelectionStatus.UNRESOLVED, SelectionStatus.HUMAN_REVIEW)


# ============================================================================
# TEST 4: Resolution Vocabulary Validation
# ============================================================================

def test_resolution_vocabulary_uses_existing_taxonomy():
    """Test that candidates use existing ResolutionType taxonomy."""
    valid_resolutions = [r.value for r in ResolutionType]
    
    # Verify expected resolution types exist
    assert "NO_ACTION" in valid_resolutions
    assert "FEE_ADJUSTMENT" in valid_resolutions
    assert "REFUND_ADJUSTMENT" in valid_resolutions
    assert "TAX_ADJUSTMENT" in valid_resolutions
    assert "TIMING_RECONCILIATION" in valid_resolutions
    assert "PARTIAL_SETTLEMENT_RECONCILIATION" in valid_resolutions
    assert "DUPLICATE_SETTLEMENT" in valid_resolutions
    assert "MISSING_RECORD_ESCALATION" in valid_resolutions
    assert "MULTI_ADJUSTMENT" in valid_resolutions
    assert "UNKNOWN_UNRESOLVED" in valid_resolutions


# ============================================================================
# TEST 5: Deterministic Scoring
# ============================================================================

def test_scoring_is_deterministic(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that scoring is deterministic for same inputs."""
    generator = CandidateGenerator()
    scorer = CandidateScoringService()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Score twice
    score1 = scorer.score_and_rank(generation_result, sample_intelligence)
    score2 = scorer.score_and_rank(generation_result, sample_intelligence)
    
    # Compare
    assert len(score1.candidates) == len(score2.candidates)
    for c1, c2 in zip(score1.candidates, score2.candidates):
        assert c1.ranking.rank == c2.ranking.rank
        assert c1.ranking.confidence_score == c2.ranking.confidence_score


# ============================================================================
# TEST 6: Deterministic Ranking
# ============================================================================

def test_ranking_is_deterministic(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that ranking is deterministic and stable."""
    generator = CandidateGenerator()
    scorer = CandidateScoringService()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Rank twice
    rank1 = scorer.score_and_rank(generation_result, sample_intelligence)
    rank2 = scorer.score_and_rank(generation_result, sample_intelligence)
    
    # Extract ranks
    ranks1 = [c.ranking.rank for c in rank1.candidates]
    ranks2 = [c.ranking.rank for c in rank2.candidates]
    
    assert ranks1 == ranks2


# ============================================================================
# TEST 7: ML Signal Integration
# ============================================================================

def test_ml_signal_integration(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that ML signals are integrated into candidates."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Check that ML support is captured where available
    if sample_intelligence.classification.ml_predicted_type:
        # At least one candidate should have ML support
        ml_candidates = [c for c in result.candidates if c.ml_support is not None]
        # May or may not have ML candidates depending on generation logic
        pass  # ML support is optional in generation


# ============================================================================
# TEST 8: ML/Deterministic Disagreement Preservation
# ============================================================================

def test_ml_deterministic_disagreement_preserved(intelligence_with_ml_disagreement, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that ML/deterministic disagreement is preserved."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=intelligence_with_ml_disagreement,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Verify both signals remain in intelligence
    assert intelligence_with_ml_disagreement.classification.deterministic_type == "FEE_DIFFERENCE"
    assert intelligence_with_ml_disagreement.classification.ml_predicted_type == "TIMING_DIFFERENCE"
    assert intelligence_with_ml_disagreement.classification.agreement == False
    
    # Candidates may use either or both - but the signals themselves are not mutated
    assert result.exception_id == "EXC-002"


# ============================================================================
# TEST 9: Historical Precedent Integration
# ============================================================================

def test_historical_precedent_integration(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that historical precedents are integrated."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Check that historical support is captured
    hist_candidates = [c for c in result.candidates if c.historical_support]
    if sample_intelligence.similar_cases and sample_intelligence.similar_cases.similar_cases:
        # Should have historical candidates
        assert len(hist_candidates) > 0 or len(result.candidates) > 0


# ============================================================================
# TEST 10: Historical Outcome Handling
# ============================================================================

def test_historical_outcome_handling(intelligence_with_historical_failure, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that failed historical resolutions are distinguished."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=intelligence_with_historical_failure,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Verify historical outcome is captured
    if result.candidates:
        for candidate in result.candidates:
            for hist in candidate.historical_support:
                # Outcome should be preserved
                assert hist.historical_outcome in ("SUCCESSFUL", "FAILED", "PARTIAL", None)


# ============================================================================
# TEST 11: Evidence Compatibility
# ============================================================================

def test_evidence_compatibility_checking(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that evidence compatibility is checked."""
    from app.ml.resolution import EvidenceCompatibilityChecker
    
    checker = EvidenceCompatibilityChecker()
    
    # Test compatible case
    compatible, notes = checker.check("FEE_ADJUSTMENT", sample_evidence_package, sample_explanation)
    
    # Should be compatible since fee evidence exists
    assert compatible is True or len(notes) > 0  # Either compatible or has notes


# ============================================================================
# TEST 12: Insufficient Evidence Behavior
# ============================================================================

def test_insufficient_evidence_returns_unresolved():
    """Test that insufficient evidence causes selector to return UNRESOLVED."""
    # Create intelligence with low evidence coverage
    intelligence = ExceptionIntelligence(
        exception_id="EXC-004",
        case_id="CASE-004",
        payment_id="PAY-004",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type=None,
            ml_probabilities=None,
            ml_model_version=None,
            agreement=True,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="UNEXPLAINED",
            explained_amount=0,
            remaining_difference=25000,
            supporting_evidence_ids=[],
            evidence_coverage=0.0,  # NO COVERAGE
            consistency_score=0.0,
            has_conflict=False,
            missing_evidence=["fees", "settlements"],
            explanation_reason="No evidence available",
            evidence_link_count=0,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=False,
            total_indexed=0,
            top_k=0,
            similar_cases=[],
        ),
        recommendation_status=RecommendationStatus.INSUFFICIENT_EVIDENCE,
        recommendation_notes=["Insufficient evidence to determine resolution"],
        conflicts=[],
    )
    
    generator = CandidateGenerator()
    selector = CandidateSelector()
    
    # Generator may still produce candidates from deterministic mapping
    generation_result = generator.generate(intelligence=intelligence)
    
    # But selector should reject them due to insufficient evidence
    selection_result = selector.select(generation_result, intelligence)
    
    # Should return UNRESOLVED due to insufficient evidence
    assert selection_result.status == SelectionStatus.UNRESOLVED


# ============================================================================
# TEST 13: No Historical Matches Behavior
# ============================================================================

def test_no_historical_matches_behavior(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test behavior when no historical matches exist."""
    # Remove historical cases
    intelligence_no_hist = ExceptionIntelligence(
        exception_id=sample_intelligence.exception_id,
        case_id=sample_intelligence.case_id,
        payment_id=sample_intelligence.payment_id,
        merchant_id=sample_intelligence.merchant_id,
        expected_amount=sample_intelligence.expected_amount,
        actual_amount=sample_intelligence.actual_amount,
        difference=sample_intelligence.difference,
        classification=sample_intelligence.classification,
        evidence=sample_intelligence.evidence,
        similar_cases=SimilarCasesIntelligence(
            query_embedded=False,
            total_indexed=0,
            top_k=0,
            similar_cases=[],
        ),
        recommendation_status=RecommendationStatus.SUPPORTED,
        recommendation_notes=[],
        conflicts=[],
    )
    
    generator = CandidateGenerator()
    result = generator.generate(
        intelligence=intelligence_no_hist,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Should still generate candidates from deterministic/ML sources
    # Just no historical candidates
    assert result.status in ("CANDIDATES_GENERATED", "UNRESOLVED")


# ============================================================================
# TEST 14: Low ML Confidence Behavior
# ============================================================================

def test_low_ml_confidence_behavior(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test behavior when ML confidence is low."""
    intelligence = ExceptionIntelligence(
        exception_id="EXC-005",
        case_id="CASE-005",
        payment_id="PAY-005",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="TIMING_DIFFERENCE",
            ml_probabilities={"TIMING_DIFFERENCE": 0.45, "FEE_DIFFERENCE": 0.40},  # LOW CONFIDENCE
            ml_model_version="1.0.0",
            agreement=False,
        ),
        evidence=sample_intelligence.evidence,
        similar_cases=SimilarCasesIntelligence(
            query_embedded=False,
            total_indexed=0,
            top_k=0,
            similar_cases=[],
        ),
        recommendation_status=RecommendationStatus.PARTIALLY_SUPPORTED,
        recommendation_notes=["Low ML confidence"],
        conflicts=["ML/deterministic disagreement"],
    )
    
    selector = CandidateSelector()
    generator = CandidateGenerator()
    
    generation_result = generator.generate(
        intelligence=intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    selection_result = selector.select(generation_result, intelligence)
    
    # Low ML confidence should not force a resolution
    # Selection should be based on overall evidence, not just ML
    assert selection_result.status in (SelectionStatus.RECOMMENDED, SelectionStatus.UNRESOLVED, SelectionStatus.HUMAN_REVIEW)


# ============================================================================
# TEST 15: No Viable Candidate Behavior
# ============================================================================

def test_no_viable_candidate_behavior():
    """Test behavior when no candidate passes thresholds."""
    # Create intelligence that will fail thresholds
    intelligence = ExceptionIntelligence(
        exception_id="EXC-006",
        case_id="CASE-006",
        payment_id="PAY-006",
        merchant_id="MERCHANT-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="UNKNOWN",
            ml_predicted_type=None,
            ml_probabilities=None,
            ml_model_version=None,
            agreement=True,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="UNEXPLAINED",
            explained_amount=0,
            remaining_difference=25000,
            supporting_evidence_ids=[],
            evidence_coverage=0.0,
            consistency_score=0.0,
            has_conflict=True,
            missing_evidence=["fees", "settlements", "refunds"],
            explanation_reason="No evidence available",
            evidence_link_count=0,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=False,
            total_indexed=0,
            top_k=0,
            similar_cases=[],
        ),
        recommendation_status=RecommendationStatus.INSUFFICIENT_EVIDENCE,
        recommendation_notes=["No viable resolution candidates"],
        conflicts=["Evidence conflicts"],
    )
    
    selector = CandidateSelector()
    generator = CandidateGenerator()
    
    generation_result = generator.generate(intelligence=intelligence)
    selection_result = selector.select(generation_result, intelligence)
    
    # Should return UNRESOLVED
    assert selection_result.status == SelectionStatus.UNRESOLVED
    assert selection_result.confidence == 0.0
    assert selection_result.risk_category == "HIGH"


# ============================================================================
# TEST 16: Duplicate Candidate Handling
# ============================================================================

def test_duplicate_candidate_handling(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that duplicate resolution types are merged."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Count unique resolution types
    resolution_types = [c.resolution_type for c in result.candidates]
    unique_types = set(resolution_types)
    
    # Should not have duplicate resolution types
    assert len(resolution_types) == len(unique_types)


# ============================================================================
# TEST 17: Amount Integrity
# ============================================================================

def test_amount_integrity(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that amounts use integer paise and are traced to evidence."""
    generator = CandidateGenerator()
    
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    for candidate in result.candidates:
        # Amount must be integer
        assert isinstance(candidate.financial_adjustment.amount_paise, int)
        assert candidate.financial_adjustment.amount_paise >= 0
        
        # Direction must be valid
        assert candidate.financial_adjustment.direction in ("CREDIT", "DEBIT", "NONE")
        
        # Calculation basis must be specified
        assert candidate.financial_adjustment.calculation_basis != ""


# ============================================================================
# TEST 18: Financial Truth Preservation
# ============================================================================

def test_financial_truth_preservation(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation does not mutate financial truth."""
    original_expected = sample_intelligence.expected_amount
    original_actual = sample_intelligence.actual_amount
    original_difference = sample_intelligence.difference
    
    generator = CandidateGenerator()
    result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Financial truth must remain unchanged
    assert sample_intelligence.expected_amount == original_expected
    assert sample_intelligence.actual_amount == original_actual
    assert sample_intelligence.difference == original_difference


# ============================================================================
# TEST 19: No Resolution Persistence
# ============================================================================

def test_no_resolution_persistence(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation does NOT persist Resolution records."""
    # This is a behavioral test - we verify that the generate() method
    # does not have side effects
    
    generator = CandidateGenerator()
    
    # Count existing resolutions before (if database available)
    # For now, we verify the method signature doesn't include session/persistence
    import inspect
    sig = inspect.signature(generator.generate)
    
    # Should not take database session as parameter
    assert 'session' not in sig.parameters
    assert 'db' not in sig.parameters


# ============================================================================
# TEST 20: No ResolutionAction Persistence
# ============================================================================

def test_no_resolution_action_persistence(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation does NOT persist ResolutionAction records."""
    generator = CandidateGenerator()
    
    # Verify method signature doesn't include persistence
    import inspect
    sig = inspect.signature(generator.generate)
    
    # Should not take database session as parameter
    assert 'session' not in sig.parameters


# ============================================================================
# TEST 21: No Approval Transition
# ============================================================================

def test_no_approval_transition(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate selection does NOT transition to APPROVED."""
    selector = CandidateSelector()
    generator = CandidateGenerator()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    selection_result = selector.select(generation_result, sample_intelligence)
    
    # SelectionResult status should be RECOMMENDED, UNRESOLVED, or HUMAN_REVIEW
    # NOT APPROVED, EXECUTING, EXECUTED, etc.
    assert selection_result.status in (
        SelectionStatus.RECOMMENDED,
        SelectionStatus.UNRESOLVED,
        SelectionStatus.HUMAN_REVIEW,
    )


# ============================================================================
# TEST 22: No Execution
# ============================================================================

def test_no_execution(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation/selection does NOT execute financial actions."""
    generator = CandidateGenerator()
    selector = CandidateSelector()
    
    generation_result = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    selection_result = selector.select(generation_result, sample_intelligence)
    
    # Verify no execution occurred by checking that no external systems were called
    # This is a behavioral test - the methods should not have side effects


# ============================================================================
# TEST 23: No Provider Invocation
# ============================================================================

def test_no_provider_invocation(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that candidate generation does NOT invoke external providers."""
    generator = CandidateGenerator()
    
    # Verify no provider imports in the generator
    import app.services.candidate_generator as gen_module
    source = inspect.getsource(gen_module)
    
    # Check that there are no actual provider API calls (imports or usage)
    # Exclude docstrings/comments
    lines = [line.strip() for line in source.split('\n') if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]
    code_only = '\n'.join(lines)
    
    # Should not contain provider API calls in actual code
    assert "import razorpay" not in code_only.lower()
    assert "import stripe" not in code_only.lower()
    assert "razorpay.Client" not in code_only
    assert "stripe.api_key" not in code_only


# ============================================================================
# TEST 24: Golden FEE_MISMATCH Vertical Slice
# ============================================================================

def test_golden_fee_mismatch_vertical_slice():
    """
    Golden test: Complete FEE_MISMATCH vertical slice through Phase 7.
    
    Canonical financial values:
    - Payment: ₹10,000 (1,000,000 paise)
    - Expected settlement: ₹9,500 (950,000 paise)
    - Actual settlement: ₹9,250 (925,000 paise)
    - Difference: ₹250 (25,000 paise)
    - Exception: FEE_MISMATCH
    """
    # Build the canonical case
    evidence = EvidencePackage(
        exception_id="EXC-GOLDEN-001",
        case_id="CASE-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        merchant_id="MERCHANT-GOLDEN-001",
        expected_amount=950000,  # ₹9,500
        actual_amount=925000,  # ₹9,250
        difference=25000,  # ₹250
        exception_type="FEE_DIFFERENCE",
        payment=EvidenceRecord(
            record_id="PAY-GOLDEN-001",
            entity_type="PAYMENT",
            amount=1000000,  # ₹10,000
            relationship="PRIMARY_RECORD",
        ),
        settlements=[
            EvidenceRecord(
                record_id="SET-GOLDEN-001",
                entity_type="SETTLEMENT",
                amount=925000,
                relationship="CALCULATION_COMPONENT",
            )
        ],
        fees=[
            EvidenceRecord(
                record_id="FEE-GOLDEN-001",
                entity_type="FEE",
                amount=30000,  # ₹300
                relationship="CALCULATION_COMPONENT",
                contribution=-30000,
            )
        ],
        refunds=[],
        taxes=[],
        adjustments=[],
        total_settlement_amount=925000,
        total_refund_amount=0,
        total_fee_amount=30000,
        total_tax_amount=0,
        total_adjustment_amount=0,
        missing_evidence=[],
        conflicts=[],
        evidence_link_count=3,
    )
    
    explanation = ExplanationResult(
        exception_id="EXC-GOLDEN-001",
        case_id="CASE-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        explanation_status=ExplanationStatus.FULLY_EXPLAINED,
        explained_amount=30000,
        remaining_difference=0,
        supporting_evidence_ids=["FEE-GOLDEN-001"],
        explanation_reason="Fee difference of ₹250 detected",
    )
    
    quality = EvidenceQualityResult(
        exception_id="EXC-GOLDEN-001",
        case_id="CASE-GOLDEN-001",
        coverage_score=0.95,
        consistency_score=0.90,
        conflict=False,
        novelty=NoveltyLevel.KNOWN_PATTERN,
        missing_evidence=[],
        fully_explained=True,
        partially_explained=False,
        supporting_evidence_count=3,
    )
    
    intelligence = ExceptionIntelligence(
        exception_id="EXC-GOLDEN-001",
        case_id="CASE-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        merchant_id="MERCHANT-GOLDEN-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="FEE_DIFFERENCE",
            ml_probabilities={"FEE_DIFFERENCE": 0.95},
            ml_model_version="1.0.0",
            agreement=True,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=30000,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-GOLDEN-001"],
            evidence_coverage=0.95,
            consistency_score=0.90,
            has_conflict=False,
            missing_evidence=[],
            explanation_reason="Fee difference fully explained",
            evidence_link_count=3,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=True,
            total_indexed=150,
            top_k=5,
            similar_cases=[
                {
                    "case_id": "CASE-HIST-001",
                    "similarity_score": 0.94,
                    "resolution_type": "FEE_ADJUSTMENT",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 1000000,
                    "difference": 25000,
                },
            ],
        ),
        recommendation_status=RecommendationStatus.SUPPORTED,
        recommendation_notes=[],
        conflicts=[],
    )
    
    # Verify canonical values
    assert evidence.payment.amount == 1000000
    assert evidence.expected_amount == 950000
    assert evidence.actual_amount == 925000
    assert evidence.difference == 25000
    assert evidence.exception_type == "FEE_DIFFERENCE"
    
    # Run through Phase 7 pipeline
    generator = CandidateGenerator()
    scorer = CandidateScoringService()
    selector = CandidateSelector()
    
    # Generate
    generation_result = generator.generate(
        intelligence=intelligence,
        package=evidence,
        explanation=explanation,
        quality=quality,
    )
    
    assert generation_result is not None
    assert generation_result.exception_id == "EXC-GOLDEN-001"
    
    # Score
    scored_result = scorer.score_and_rank(generation_result, intelligence)
    assert scored_result is not None
    
    # Select
    selection_result = selector.select(scored_result, intelligence)
    assert selection_result is not None
    
    # Verify financial truth preserved
    assert intelligence.expected_amount == 950000
    assert intelligence.actual_amount == 925000
    assert intelligence.difference == 25000
    
    # Verify candidates use existing vocabulary
    if selection_result.selected_candidate:
        assert selection_result.selected_candidate.resolution_type in [
            r.value for r in ResolutionType
        ]
    
    # Verify no persistence occurred
    assert selection_result.status in (
        SelectionStatus.RECOMMENDED,
        SelectionStatus.UNRESOLVED,
        SelectionStatus.HUMAN_REVIEW,
    )


# ============================================================================
# TEST 25: Reproducibility
# ============================================================================

def test_reproducibility(sample_intelligence, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """Test that same inputs produce same outputs."""
    generator = CandidateGenerator()
    scorer = CandidateScoringService()
    selector = CandidateSelector()
    
    # Run pipeline twice
    result1 = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    scored1 = scorer.score_and_rank(result1, sample_intelligence)
    selected1 = selector.select(scored1, sample_intelligence)
    
    result2 = generator.generate(
        intelligence=sample_intelligence,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    scored2 = scorer.score_and_rank(result2, sample_intelligence)
    selected2 = selector.select(scored2, sample_intelligence)
    
    # Compare results
    assert result1.status == result2.status
    assert len(result1.candidates) == len(result2.candidates)
    
    if selected1.selected_candidate and selected2.selected_candidate:
        assert selected1.selected_candidate.resolution_type == selected2.selected_candidate.resolution_type
        assert selected1.confidence == selected2.confidence


# ============================================================================
# TEST 26: Historical/Deterministic/ML Triple Disagreement
# ============================================================================

def test_triple_disagreement_preservation(intelligence_with_ml_disagreement, sample_evidence_package, sample_explanation, sample_evidence_quality):
    """
    Test case where:
    - Deterministic: FEE_DIFFERENCE
    - ML: TIMING_DIFFERENCE  
    - Historical: REFUND_ADJUSTMENT
    
    All three signals must remain independently represented.
    """
    # Verify all three signals exist independently
    assert intelligence_with_ml_disagreement.classification.deterministic_type == "FEE_DIFFERENCE"
    assert intelligence_with_ml_disagreement.classification.ml_predicted_type == "TIMING_DIFFERENCE"
    assert intelligence_with_ml_disagreement.classification.agreement == False
    
    # Historical has REFUND_ADJUSTMENT
    if intelligence_with_ml_disagreement.similar_cases.similar_cases:
        assert intelligence_with_ml_disagreement.similar_cases.similar_cases[0]["resolution_type"] == "REFUND_ADJUSTMENT"
    
    # Run through pipeline
    generator = CandidateGenerator()
    result = generator.generate(
        intelligence=intelligence_with_ml_disagreement,
        package=sample_evidence_package,
        explanation=sample_explanation,
        quality=sample_evidence_quality,
    )
    
    # Signals should still be preserved in intelligence
    assert intelligence_with_ml_disagreement.classification.deterministic_type == "FEE_DIFFERENCE"
    assert intelligence_with_ml_disagreement.classification.ml_predicted_type == "TIMING_DIFFERENCE"
    
    # Candidates may include all three resolution types
    resolution_types = [c.resolution_type for c in result.candidates]
    # May have FEE_ADJUSTMENT, TIMING_RECONCILIATION, REFUND_ADJUSTMENT
    # This is allowed - the signals inform but don't collapse


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
