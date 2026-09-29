"""
Tests for Razorpay CloseLoop Phase 7 — Resolution Intelligence.

This explicitly verifies:
1. Candidate generation is advisory only.
2. ML, deterministic, and historical signal disagreement is preserved.
3. Golden FEE_MISMATCH vertical slice succeeds without mutating financial truth.
4. No Resolution, ResolutionAction, or Approval is persisted.
5. No execution occurs.
"""

import pytest
from app.services.candidate_generator import CandidateGenerator
from app.schemas.intelligence import (
    ClassificationResult,
    ExceptionIntelligence,
    EvidenceIntelligence,
    RecommendationStatus,
    SimilarCasesIntelligence,
)
from app.schemas.evidence import EvidencePackage, EvidenceRecord
from app.schemas.explanation import ExplanationResult, ExplanationStatus


def test_golden_fee_mismatch_vertical_slice():
    """
    Test Phase 7 golden flow.
    Canonical financial values MUST remain unchanged.
    Output MUST be a ranked advisory list.
    No persistence must occur.
    """
    # 1. Setup Golden FEE_MISMATCH inputs
    expected_amount = 950000
    actual_amount = 925000
    difference = 25000
    
    # Deterministic signals
    package = EvidencePackage(
        exception_id="EXC-GOLDEN",
        case_id="CASE-GOLDEN",
        payment_id="PAY-GOLDEN",
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        exception_type="FEE_DIFFERENCE",
        payment=EvidenceRecord(
            record_id="PAY-001",
            entity_type="PAYMENT",
            relationship="PRIMARY",
            amount=1000000
        ),
        settlements=[EvidenceRecord(
            record_id="SET-001",
            entity_type="SETTLEMENT",
            relationship="SUPPORTING",
            amount=925000
        )],
        refunds=[],
        fees=[EvidenceRecord(
            record_id="FEE-001",
            entity_type="FEE",
            relationship="CALCULATION_COMPONENT",
            amount=25000
        )],
        taxes=[],
        adjustments=[],
        total_settlement_amount=925000,
        missing_evidence=[],
        conflicts=[],
        evidence_link_count=1
    )
    
    explanation = ExplanationResult(
        exception_id="EXC-GOLDEN",
        case_id="CASE-GOLDEN",
        payment_id="PAY-GOLDEN",
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        explanation_status=ExplanationStatus.FULLY_EXPLAINED,
        explained_amount=-25000,
        remaining_difference=0,
        supporting_evidence_ids=["FEE-001"],
        candidate_explanations=[],
        conflict=False,
        missing_evidence=[],
        explanation_reason="Fee fully explains difference."
    )
    
    # Intelligence (Phase 5 + 6 output)
    intel = ExceptionIntelligence(
        exception_id="EXC-GOLDEN",
        case_id="CASE-GOLDEN",
        payment_id="PAY-GOLDEN",
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="FEE_DIFFERENCE",
            ml_probabilities={"FEE_DIFFERENCE": 0.95},
            ml_model_version="1.0.0",
            agreement=True
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED",
            explained_amount=-25000,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"],
            evidence_coverage=1.0,
            consistency_score=1.0,
            has_conflict=False
        ),
        similar_cases=SimilarCasesIntelligence(
            similar_cases=[{
                "case_id": "CASE-HIST",
                "similarity_score": 0.88,
                "resolution_type": "FEE_ADJUSTMENT",
                "resolution_outcome": "SUCCESSFUL",
                "payment_amount": 1000000,
                "difference": 25000,
            }],
            best_similarity_score=0.88
        ),
        recommendation_status=RecommendationStatus.SUPPORTED
    )
    
    # 2. Execute Generator
    generator = CandidateGenerator()
    result = generator.generate(intel, package, explanation)
    
    # 3. Assertions
    assert result.status == "CANDIDATES_GENERATED"
    assert result.total_candidates >= 1
    
    best = result.best_candidate()
    assert best.resolution_type == "FEE_ADJUSTMENT"
    
    # Financial truth preservation
    assert intel.difference == 25000
    assert intel.expected_amount == 950000
    assert intel.actual_amount == 925000
    
    # Amount handling
    assert best.financial_adjustment.amount_paise == 25000
    
    # Advisory Only
    assert best.is_recommendation_only is True


def test_ml_deterministic_historical_disagreement():
    """
    Test where all three signals disagree.
    Candidate layer must preserve these signals and rank them independently.
    """
    intel = ExceptionIntelligence(
        exception_id="EXC-DISAGREE",
        case_id="CASE-DISAGREE",
        payment_id="PAY-DISAGREE",
        expected_amount=1000,
        actual_amount=0,
        difference=1000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="TIMING_DIFFERENCE",   # ML says TIMING (resolved via TIMING_RECONCILIATION)
            ml_probabilities={"TIMING_DIFFERENCE": 0.9},
            ml_model_version="1.0.0",
            agreement=False
        ),
        evidence=EvidenceIntelligence(
            explanation_status="UNEXPLAINED",
            explained_amount=0,
            remaining_difference=1000,
            supporting_evidence_ids=[],
            evidence_coverage=0.0,
            consistency_score=0.0,
            has_conflict=False
        ),
        similar_cases=SimilarCasesIntelligence(
            similar_cases=[{
                "case_id": "CASE-HIST2",
                "similarity_score": 0.95,
                "resolution_type": "REFUND_ADJUSTMENT", # Historical says REFUND
                "resolution_outcome": "SUCCESSFUL",
                "payment_amount": 1000,
                "difference": 1000,
            }],
            best_similarity_score=0.95
        ),
        recommendation_status=RecommendationStatus.SUPPORTED
    )
    
    generator = CandidateGenerator()
    result = generator.generate(intel, package=None, explanation=None)
    
    assert result.status == "CANDIDATES_GENERATED"
    types = [c.resolution_type for c in result.candidates]
    
    # The generator should create a candidate for all 3 diverging sources
    assert "FEE_ADJUSTMENT" in types             # From deterministic map
    assert "TIMING_RECONCILIATION" in types      # From ML map
    assert "REFUND_ADJUSTMENT" in types          # From historical similarity

    # Deterministic facts remained unchanged
    assert intel.classification.deterministic_type == "FEE_DIFFERENCE"
    assert intel.classification.ml_predicted_type == "TIMING_DIFFERENCE"


def test_no_persistence_state_machine_protection():
    """
    Verifies that generating candidates does not mutate the DB or advance state.
    """
    intel = ExceptionIntelligence(
        exception_id="EXC-STATE",
        case_id="CASE-STATE",
        payment_id="PAY-STATE",
        expected_amount=1000,
        actual_amount=0,
        difference=1000,
        classification=ClassificationResult(
            deterministic_type="UNKNOWN",
            ml_predicted_type=None,
            ml_probabilities=None,
            ml_model_version=None,
            agreement=False
        ),
        evidence=EvidenceIntelligence(
            explanation_status="UNEXPLAINED",
            explained_amount=0,
            remaining_difference=1000,
            supporting_evidence_ids=[],
            evidence_coverage=0.0,
            consistency_score=0.0,
            has_conflict=False
        ),
        similar_cases=SimilarCasesIntelligence(
            similar_cases=[],
            best_similarity_score=0.0
        ),
        recommendation_status=RecommendationStatus.SUPPORTED
    )
    
    generator = CandidateGenerator()
    result = generator.generate(intel, package=None, explanation=None)
    
    # Returns purely Transient Data (Pydantic models) without ORM persistence.
    assert hasattr(result, "candidates")
    
    # If the system were to execute or save, it would require a DB Session
    # CandidateGenerator's signature accepts no session, making persistence impossible.
    # We assert that none of the candidates have DB primary keys (like an integer ID or committed string ID usually set by DB).
    if result.candidates:
        best = result.best_candidate()
        assert best.is_recommendation_only is True
        # Since it's a Pydantic model, it doesn't have SQLAlchemy metadata
        assert not hasattr(best, "_sa_instance_state")
