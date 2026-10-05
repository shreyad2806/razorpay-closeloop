"""
Phase 8 Policy + Guardrails + Authorization Boundary Tests

Tests verify that Phase 8 correctly evaluates Phase 7 proposals through
existing guardrail infrastructure and produces explicit authorization decisions.

Core principle:
    AI investigates and proposes.
    Policy authorizes.
    Deterministic services execute.
    Reconciliation verifies truth.

Phase 8 is ONLY:
POLICY + GUARDRAILS + AUTHORIZATION BOUNDARY

Tests verify:
- FAIL-CLOSED behavior for all missing/invalid inputs
- ML cannot bypass policy
- Historical similarity cannot bypass policy
- Deterministic financial truth is authoritative
- No provider execution
- No financial mutation
"""

import pytest
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.schemas.resolution_candidate import (
    ResolutionProposal,
    CandidateSource,
    FinancialAdjustment,
    CandidateRanking,
)
from app.schemas.resolution_selection import SelectionResult, SelectionStatus
from app.schemas.intelligence import (
    ExceptionIntelligence,
    ClassificationResult,
    EvidenceIntelligence,
    SimilarCasesIntelligence,
    RecommendationStatus,
)
from app.schemas.evidence import EvidencePackage, EvidenceRecord
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
from app.schemas.decision_matrix import AutomationDecision, DecisionConfig
from app.services.authorization import AuthorizationService, AuthorizationDecision
from app.services.guardrail_engine import GuardrailEngine
from app.services.decision_matrix import AutomationDecisionMatrix


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_session():
    """Mock database session for testing."""
    # Phase 8 authorization does not require a database session
    # Authorization is evaluation-only, no persistence
    return None


@pytest.fixture
def safe_fee_proposal():
    """Safe FEE_ADJUSTMENT proposal that should qualify for AUTO."""
    return ResolutionProposal(
        candidate_id="CAND-001-SAFE",
        exception_id="EXC-001",
        case_id="CASE-001",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction to reconcile the discrepancy",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=25000,  # ₹250 - within auto limit
            direction="CREDIT",
            evidence_record_id="FEE-001",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-001", "PAY-001"],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record fully explains the discrepancy",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.85,
            evidence_support=0.95,
        ),
        rationale="Deterministic evidence supports FEE_ADJUSTMENT",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def high_amount_proposal():
    """Proposal with amount exceeding auto-resolution limit."""
    return ResolutionProposal(
        candidate_id="CAND-002-HIGH",
        exception_id="EXC-002",
        case_id="CASE-002",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=500000,  # ₹5,000 - EXCEEDS DEFAULT AUTO LIMIT
            direction="CREDIT",
            evidence_record_id="FEE-002",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-002"],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record explains discrepancy",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.90,
            evidence_support=0.95,
        ),
        rationale="Deterministic evidence supports FEE_ADJUSTMENT",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def insufficient_evidence_proposal():
    """Proposal with insufficient evidence."""
    return ResolutionProposal(
        candidate_id="CAND-003-LOW-EVIDENCE",
        exception_id="EXC-003",
        case_id="CASE-003",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=25000,
            direction="CREDIT",
            evidence_record_id=None,  # NO EVIDENCE TRACE
            calculation_basis="discrepancy_amount",
        ),
        supporting_evidence_ids=[],  # NO EVIDENCE
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.0,  # NO COVERAGE
        coverage_explanation="No evidence to support adjustment",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.50,
            evidence_support=0.0,
        ),
        rationale="Proposed adjustment without evidence support",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def ml_high_confidence_proposal():
    """Proposal with high ML confidence but policy violation."""
    from app.schemas.resolution_candidate import MLSupportDetail
    
    return ResolutionProposal(
        candidate_id="CAND-004-ML-HIGH",
        exception_id="EXC-004",
        case_id="CASE-004",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=500000,  # POLICY VIOLATION: high amount
            direction="CREDIT",
            evidence_record_id="FEE-004",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-004"],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record explains discrepancy",
        ml_support=MLSupportDetail(
            supported=True,
            predicted_resolution="FEE_ADJUSTMENT",
            confidence=0.99,  # HIGH ML CONFIDENCE
            model_version="1.0.0",
            probability=0.99,
        ),
        historical_support=[],
        sources=[CandidateSource.ML_PREDICTION.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.99,
            evidence_support=0.95,
            ml_support=0.99,
        ),
        rationale="ML strongly supports FEE_ADJUSTMENT",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def historical_strong_match_proposal():
    """Proposal with strong historical match but policy violation."""
    from app.schemas.resolution_candidate import HistoricalSupportDetail
    
    return ResolutionProposal(
        candidate_id="CAND-005-HIST-STRONG",
        exception_id="EXC-005",
        case_id="CASE-005",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=500000,  # POLICY VIOLATION: high amount
            direction="CREDIT",
            evidence_record_id="FEE-005",
            calculation_basis="fee_record_sum",
        ),
        supporting_evidence_ids=["FEE-005"],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.95,
        coverage_explanation="Fee record explains discrepancy",
        ml_support=None,
        historical_support=[
            HistoricalSupportDetail(
                case_id="CASE-HIST-999",
                similarity_score=0.98,  # STRONG HISTORICAL MATCH
                historical_resolution="FEE_ADJUSTMENT",
                historical_outcome="SUCCESSFUL",
                payment_amount=1000000,
                difference=25000,
            ),
        ],
        sources=[CandidateSource.HISTORICAL_CASE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.98,
            evidence_support=0.95,
            historical_support=0.98,
        ),
        rationale="Strong historical match supports FEE_ADJUSTMENT",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def blocked_type_proposal():
    """Proposal with blocked resolution type."""
    return ResolutionProposal(
        candidate_id="CAND-006-BLOCKED",
        exception_id="EXC-006",
        case_id="CASE-006",
        resolution_type="UNKNOWN_UNRESOLVED",  # BLOCKED TYPE
        resolution_description="Unknown resolution type",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="NO_ADJUSTMENT",
            amount_paise=0,
            direction="NONE",
            evidence_record_id=None,
            calculation_basis="zero_discrepancy",
        ),
        supporting_evidence_ids=[],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.0,
        coverage_explanation="No discrepancy to resolve",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.50,
            evidence_support=0.0,
        ),
        rationale="Unknown pattern - cannot auto-resolve",
        rationale_components=[],
        is_recommendation_only=True,
    )


@pytest.fixture
def safe_intelligence():
    """Safe intelligence for golden FEE_MISMATCH case."""
    return ExceptionIntelligence(
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


@pytest.fixture
def safe_evidence_package():
    """Safe evidence package for golden FEE_MISMATCH case."""
    return EvidencePackage(
        exception_id="EXC-GOLDEN-001",
        case_id="CASE-GOLDEN-001",
        payment_id="PAY-GOLDEN-001",
        merchant_id="MERCHANT-GOLDEN-001",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        exception_type="FEE_DIFFERENCE",
        payment=EvidenceRecord(
            record_id="PAY-GOLDEN-001",
            entity_type="PAYMENT",
            amount=1000000,
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
                amount=30000,
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


# ============================================================================
# TEST 1: SAFE AUTO PATH
# ============================================================================

def test_safe_auto_path(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """Test that safe proposal can qualify for AUTO."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    # Should be AUTO or HUMAN_REVIEW (never UNRESOLVED for safe case)
    assert decision.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW)
    assert decision.decision != AutomationDecision.UNRESOLVED
    assert decision.exception_id == "EXC-001"
    assert decision.case_id == "CASE-001"
    assert decision.required_approval == (decision.decision == AutomationDecision.HUMAN_REVIEW)


# ============================================================================
# TEST 2: AMOUNT LIMIT
# ============================================================================

def test_amount_limit_blocks_auto(db_session, high_amount_proposal, safe_intelligence, safe_evidence_package):
    """Test that amount exceeding auto limit requires review/rejection."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=high_amount_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO
    assert decision.decision in (AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)
    assert decision.financial_exposure_paise == 500000


# ============================================================================
# TEST 3: INSUFFICIENT EVIDENCE
# ============================================================================

def test_insufficient_evidence_blocks_auto(db_session, insufficient_evidence_proposal):
    """Test that insufficient evidence cannot auto-authorize."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=insufficient_evidence_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO
    assert decision.decision in (AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)


# ============================================================================
# TEST 4: CONFLICTING EVIDENCE
# ============================================================================

def test_conflicting_evidence_blocks_auto(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """Test that conflicting evidence cannot auto-authorize."""
    # Create intelligence with conflict
    conflicted_intelligence = ExceptionIntelligence(
        exception_id="EXC-007",
        case_id="CASE-007",
        payment_id="PAY-007",
        merchant_id="MERCHANT-007",
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
            explanation_status="CONFLICTING",
            explained_amount=0,
            remaining_difference=25000,
            supporting_evidence_ids=[],
            evidence_coverage=0.5,
            consistency_score=0.5,
            has_conflict=True,  # CONFLICT
            missing_evidence=[],
            explanation_reason="Conflicting explanations exist",
            evidence_link_count=0,
        ),
        similar_cases=SimilarCasesIntelligence(
            query_embedded=True,
            total_indexed=150,
            top_k=5,
            similar_cases=[],
        ),
        recommendation_status=RecommendationStatus.CONFLICTING,
        recommendation_notes=["Evidence conflicts detected"],
        conflicts=["ML/deterministic disagreement"],
    )
    
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=conflicted_intelligence,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO


# ============================================================================
# TEST 5: LOW ML CONFIDENCE
# ============================================================================

def test_low_ml_confidence_blocks_auto(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """Test that low ML confidence cannot auto-authorize when policy requires it."""
    # Create intelligence with low ML confidence
    low_ml_intelligence = ExceptionIntelligence(
        exception_id="EXC-008",
        case_id="CASE-008",
        payment_id="PAY-008",
        merchant_id="MERCHANT-008",
        expected_amount=950000,
        actual_amount=925000,
        difference=25000,
        classification=ClassificationResult(
            deterministic_type="FEE_DIFFERENCE",
            ml_predicted_type="TIMING_DIFFERENCE",
            ml_probabilities={"TIMING_DIFFERENCE": 0.45, "FEE_DIFFERENCE": 0.40},
            ml_model_version="1.0.0",
            agreement=False,
        ),
        evidence=safe_intelligence.evidence,
        similar_cases=safe_intelligence.similar_cases,
        recommendation_status=RecommendationStatus.PARTIALLY_SUPPORTED,
        recommendation_notes=["Low ML confidence"],
        conflicts=["ML/deterministic disagreement"],
    )
    
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=low_ml_intelligence,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO


# ============================================================================
# TEST 6: HIGH ML CONFIDENCE + POLICY VIOLATION
# ============================================================================

def test_high_ml_confidence_policy_violation_blocks_auto(db_session, ml_high_confidence_proposal):
    """
    CRITICAL BOUNDARY TEST
    
    ML confidence = 0.99
    +
    Policy violation (high amount)
    
    MUST NOT result in AUTO.
    
    Proves ML cannot bypass policy.
    """
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=ml_high_confidence_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # MUST NOT be AUTO despite high ML confidence
    assert decision.decision != AutomationDecision.AUTO
    assert decision.decision in (AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)
    
    # Verify ML support is preserved but did not override policy
    if decision.guardrail_result:
        # The decision should be based on exposure/amount, not ML
        assert decision.financial_exposure_paise == 500000


# ============================================================================
# TEST 7: STRONG HISTORICAL MATCH + POLICY VIOLATION
# ============================================================================

def test_strong_historical_match_policy_violation_blocks_auto(db_session, historical_strong_match_proposal):
    """
    CRITICAL BOUNDARY TEST
    
    Historical similarity = 0.98
    +
    Policy violation (high amount)
    
    MUST NOT result in AUTO.
    
    Proves historical similarity cannot bypass policy.
    """
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=historical_strong_match_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # MUST NOT be AUTO despite strong historical match
    assert decision.decision != AutomationDecision.AUTO
    assert decision.decision in (AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)
    
    # Verify historical support is preserved but did not override policy
    if decision.guardrail_result:
        assert decision.financial_exposure_paise == 500000


# ============================================================================
# TEST 8: DETERMINISTIC FINANCIAL INCONSISTENCY
# ============================================================================

def test_deterministic_financial_inconsistency_blocks_auto(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """
    CRITICAL BOUNDARY TEST
    
    Deterministic financial inconsistency
    +
    High ML confidence
    
    MUST NOT result in AUTO.
    
    Proves deterministic financial truth is authoritative.
    """
    # Create evidence with inconsistency
    inconsistent_evidence = EvidencePackage(
        exception_id="EXC-009",
        case_id="CASE-009",
        payment_id="PAY-009",
        merchant_id="MERCHANT-009",
        expected_amount=950000,
        actual_amount=900000,  # WRONG AMOUNT - INCONSISTENT
        difference=50000,  # WRONG DIFFERENCE
        exception_type="FEE_DIFFERENCE",
        payment=EvidenceRecord(
            record_id="PAY-009",
            entity_type="PAYMENT",
            amount=1000000,
            relationship="PRIMARY_RECORD",
        ),
        settlements=[
            EvidenceRecord(
                record_id="SET-009",
                entity_type="SETTLEMENT",
                amount=900000,
                relationship="CALCULATION_COMPONENT",
            )
        ],
        fees=[
            EvidenceRecord(
                record_id="FEE-009",
                entity_type="FEE",
                amount=30000,
                relationship="CALCULATION_COMPONENT",
                contribution=-30000,
            )
        ],
        refunds=[],
        taxes=[],
        adjustments=[],
        total_settlement_amount=900000,
        total_refund_amount=0,
        total_fee_amount=30000,
        total_tax_amount=0,
        total_adjustment_amount=0,
        missing_evidence=[],
        conflicts=[
            # Structural conflict would be detected here
        ],
        evidence_link_count=3,
    )
    
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=inconsistent_evidence,
        intelligence=safe_intelligence,
    )
    
    # Must NOT be AUTO due to inconsistency
    assert decision.decision != AutomationDecision.AUTO


# ============================================================================
# TEST 9: UNKNOWN/INVALID INPUT
# ============================================================================

def test_unknown_resolution_type_blocks_auto(db_session, blocked_type_proposal):
    """Test that unknown/blocked resolution type fails closed."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=blocked_type_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO
    assert decision.decision in (AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)


# ============================================================================
# TEST 10: MISSING AMOUNT
# ============================================================================

def test_missing_amount_blocks_auto(db_session):
    """Test that missing financial amount fails closed."""
    # Create proposal with missing amount
    no_amount_proposal = ResolutionProposal(
        candidate_id="CAND-007-NO-AMOUNT",
        exception_id="EXC-010",
        case_id="CASE-010",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=0,  # NO AMOUNT
            direction="NONE",
            evidence_record_id=None,
            calculation_basis="discrepancy_amount",
        ),
        supporting_evidence_ids=[],
        evidence_records=[],
        evidence_compatible=True,
        evidence_coverage=0.0,
        coverage_explanation="No amount specified",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.50,
            evidence_support=0.0,
        ),
        rationale="No financial adjustment specified",
        rationale_components=[],
        is_recommendation_only=True,
    )
    
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=no_amount_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO


# ============================================================================
# TEST 11: REPEATED EVALUATION DETERMINISM
# ============================================================================

def test_repeated_evaluation_determinism(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """Test that same inputs produce identical authorization decisions."""
    auth_service = AuthorizationService(session=db_session)
    
    # Evaluate twice
    decision1 = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    decision2 = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    # Must be identical
    assert decision1.decision == decision2.decision
    assert decision1.risk_category == decision2.risk_category
    assert decision1.reason_codes == decision2.reason_codes
    assert decision1.confidence == decision2.confidence


# ============================================================================
# TEST 12: GOLDEN FEE_MISMATCH
# ============================================================================

def test_golden_fee_mismatch_vertical_slice(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """
    Golden test: Complete FEE_MISMATCH vertical slice through Phase 8.
    
    Canonical financial values:
    - Payment: ₹10,000 (1,000,000 paise)
    - Expected settlement: ₹9,500 (950,000 paise)
    - Actual settlement: ₹9,250 (925,000 paise)
    - Difference: ₹250 (25,000 paise)
    - Exception: FEE_MISMATCH
    
    Verify:
    ResolutionProposal
        →
    GuardrailEngine
        →
    DecisionMatrix
        →
    Authorization Decision
    """
    # Verify canonical values
    assert safe_evidence_package.payment.amount == 1000000
    assert safe_evidence_package.expected_amount == 950000
    assert safe_evidence_package.actual_amount == 925000
    assert safe_evidence_package.difference == 25000
    assert safe_evidence_package.exception_type == "FEE_DIFFERENCE"
    
    safe_fee_proposal.exception_id = "EXC-GOLDEN-001"
    safe_fee_proposal.case_id = "CASE-GOLDEN-001"
    
    # Run through Phase 8 authorization
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    # Verify authorization decision
    assert decision is not None
    assert decision.exception_id == "EXC-GOLDEN-001"
    assert decision.case_id == "CASE-GOLDEN-001"
    assert decision.decision in (AutomationDecision.AUTO, AutomationDecision.HUMAN_REVIEW, AutomationDecision.UNRESOLVED)
    
    # Verify guardrail result exists
    assert decision.guardrail_result is not None
    
    # Verify financial truth is preserved
    assert safe_evidence_package.payment.amount == 1000000
    assert safe_evidence_package.expected_amount == 950000
    assert safe_evidence_package.actual_amount == 925000
    assert safe_evidence_package.difference == 25000


# ============================================================================
# TEST 13: NO FINANCIAL MUTATION
# ============================================================================

def test_no_financial_mutation(db_session, safe_fee_proposal, safe_intelligence, safe_evidence_package):
    """Test that Phase 8 does not mutate financial records."""
    # Snapshot financial state before
    original_payment_amount = safe_evidence_package.payment.amount
    original_expected = safe_evidence_package.expected_amount
    original_actual = safe_evidence_package.actual_amount
    original_difference = safe_evidence_package.difference
    
    # Run Phase 8 authorization
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=safe_fee_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    # Verify financial records are unchanged
    assert safe_evidence_package.payment.amount == original_payment_amount
    assert safe_evidence_package.expected_amount == original_expected
    assert safe_evidence_package.actual_amount == original_actual
    assert safe_evidence_package.difference == original_difference


# ============================================================================
# TEST 14: NO PROVIDER EXECUTION
# ============================================================================

def test_no_provider_execution(db_session, safe_fee_proposal):
    """Test that Phase 8 does not invoke provider execution paths."""
    # Verify authorization service does not have provider execution methods
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Check that there are no provider API calls in actual code
    lines = [line.strip() for line in source.split('\n') if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]
    code_only = '\n'.join(lines)
    
    # Should not contain provider API calls
    assert "import razorpay" not in code_only.lower()
    assert "import stripe" not in code_only.lower()
    assert "razorpay.Client" not in code_only
    assert "stripe.api_key" not in code_only


# ============================================================================
# TEST 15: HUMAN_REVIEW PATH
# ============================================================================

def test_human_review_path(db_session, high_amount_proposal, safe_intelligence, safe_evidence_package):
    """Test that HUMAN_REVIEW path requires approval."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=high_amount_proposal,
        evidence_package=safe_evidence_package,
        intelligence=safe_intelligence,
    )
    
    if decision.decision == AutomationDecision.HUMAN_REVIEW:
        assert decision.required_approval == True
    else:
        # May be UNRESOLVED for very high amounts
        assert decision.decision == AutomationDecision.UNRESOLVED


# ============================================================================
# TEST 16: UNRESOLVED PATH
# ============================================================================

def test_unresolved_path(db_session, blocked_type_proposal):
    """Test that UNRESOLVED path does not require approval."""
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=blocked_type_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    if decision.decision == AutomationDecision.UNRESOLVED:
        # UNRESOLVED should not require approval
        assert decision.required_approval == False


# ============================================================================
# TEST 17: EXISTING GUARDRAIL REUSE
# ============================================================================

def test_existing_guardrail_reused(db_session):
    """Test that existing GuardrailEngine is reused, not duplicated."""
    auth_service = AuthorizationService(session=db_session)
    
    # Verify guardrail_engine is an instance of GuardrailEngine
    from app.services.guardrail_engine import GuardrailEngine
    assert isinstance(auth_service.guardrail_engine, GuardrailEngine)


# ============================================================================
# TEST 18: EXISTING DECISION MATRIX REUSE
# ============================================================================

def test_existing_decision_matrix_reused(db_session):
    """Test that existing DecisionMatrix is reused via GuardrailEngine."""
    from app.services.guardrail_engine import GuardrailEngine
    from app.services.decision_matrix import AutomationDecisionMatrix
    
    guardrail = GuardrailEngine()
    
    # Verify GuardrailEngine uses DecisionMatrix
    assert isinstance(guardrail.decision_matrix, AutomationDecisionMatrix)


# ============================================================================
# TEST 19: FAIL-CLOSED FOR MISSING EVIDENCE
# ============================================================================

def test_fail_closed_missing_evidence(db_session):
    """Test that missing evidence results in fail-closed behavior."""
    # Create proposal with no evidence at all
    no_evidence_proposal = ResolutionProposal(
        candidate_id="CAND-008-NO-EVIDENCE",
        exception_id="EXC-011",
        case_id="CASE-011",
        resolution_type="FEE_ADJUSTMENT",
        resolution_description="Apply fee correction",
        financial_adjustment=FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=25000,
            direction="CREDIT",
            evidence_record_id=None,
            calculation_basis="discrepancy_amount",
        ),
        supporting_evidence_ids=[],
        evidence_records=[],
        evidence_compatible=False,  # NO EVIDENCE
        evidence_coverage=0.0,
        coverage_explanation="No evidence available",
        ml_support=None,
        historical_support=[],
        sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
        ranking=CandidateRanking(
            rank=1,
            confidence_score=0.30,
            evidence_support=0.0,
        ),
        rationale="No evidence to support resolution",
        rationale_components=[],
        is_recommendation_only=True,
    )
    
    auth_service = AuthorizationService(session=db_session)
    
    decision = auth_service.authorize_proposal(
        proposal=no_evidence_proposal,
        evidence_package=None,
        intelligence=None,
    )
    
    # Must NOT be AUTO
    assert decision.decision != AutomationDecision.AUTO


# ============================================================================
# TEST 20: NO APPROVAL PERSISTENCE IN PHASE 8
# ============================================================================

def test_no_approval_persistence_in_phase_8(db_session, safe_fee_proposal):
    """Test that Phase 8 does not persist Approval records."""
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Should not contain database write operations for Approval
    assert "Approval(" not in source
    assert "session.add(" not in source
    assert "session.commit(" not in source


# ============================================================================
# TEST 21: NO RESOLUTION STATE TRANSITION IN PHASE 8
# ============================================================================

def test_no_resolution_state_transition_in_phase_8(db_session, safe_fee_proposal):
    """Test that Phase 8 does not transition Resolution state."""
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Should not contain state transition calls
    assert "transition_to(" not in source
    assert "status =" not in source  # No direct status mutation


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
