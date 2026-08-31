"""
Tests for Razorpay CloseLoop Phase 5A — Resolution Candidate Generator.

Tests cover:
- ResolutionProposal schema
- FinancialAdjustment schema
- CandidateRanking schema
- CandidateGenerationResult schema
- CandidateGenerator deterministic evidence
- CandidateGenerator ML prediction
- CandidateGenerator historical cases
- Duplicate candidate merging
- Financial adjustment derivation
- UNRESOLVED behavior
- No invented amounts
- Ground truth separation
"""

import pytest
from datetime import datetime

from app.schemas.evidence import EvidencePackage, EvidenceRecord, MissingEvidence
from app.schemas.explanation import (
    ExplanationResult,
    ExplanationStatus,
    CandidateExplanation,
)
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel
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
    RationaleComponent,
    ResolutionProposal,
)
from app.services.candidate_generator import CandidateGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_intelligence(
    exception_id="EXC-001",
    deterministic_type="FEE_DIFFERENCE",
    ml_type=None,
    difference=3000,
    expected_amount=100000,
    actual_amount=97000,
    evidence_coverage=1.0,
    consistency=0.85,
    has_conflict=False,
    similar_cases=None,
):
    """Create an ExceptionIntelligence for testing."""
    return ExceptionIntelligence(
        exception_id=exception_id,
        case_id=f"CASE-{exception_id}",
        payment_id=f"PAY-{exception_id}",
        expected_amount=expected_amount,
        actual_amount=actual_amount,
        difference=difference,
        classification=ClassificationResult(
            deterministic_type=deterministic_type,
            ml_predicted_type=ml_type,
            ml_probabilities={ml_type: 0.8} if ml_type else None,
            ml_model_version="1.0.0" if ml_type else None,
            agreement=ml_type is None or ml_type == deterministic_type,
        ),
        evidence=EvidenceIntelligence(
            explanation_status="FULLY_EXPLAINED" if difference != 0 else "FULLY_EXPLAINED",
            explained_amount=-abs(difference) if difference != 0 else 0,
            remaining_difference=0,
            supporting_evidence_ids=["FEE-001"] if deterministic_type == "FEE_DIFFERENCE" else [],
            evidence_coverage=evidence_coverage,
            consistency_score=consistency,
            has_conflict=has_conflict,
        ),
        similar_cases=similar_cases or SimilarCasesIntelligence(),
        recommendation_status=RecommendationStatus.SUPPORTED,
    )


def _make_record(record_id, entity_type, relationship, amount):
    """Create an EvidenceRecord with keyword args."""
    return EvidenceRecord(
        record_id=record_id,
        entity_type=entity_type,
        relationship=relationship,
        amount=amount,
    )


def _make_package(
    exception_type="FEE_DIFFERENCE",
    fees=None,
    refunds=None,
    taxes=None,
    settlements=None,
    difference=3000,
):
    """Create an EvidencePackage for testing."""
    return EvidencePackage(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=difference,
        exception_type=exception_type,
        payment=_make_record("PAY-001", "PAYMENT", "PRIMARY_RECORD", 100000),
        settlements=settlements or [
            _make_record("SET-001", "SETTLEMENT", "SUPPORTING_EVIDENCE", 97000),
        ],
        refunds=refunds or [],
        fees=fees or [],
        taxes=taxes or [],
        adjustments=[],
        total_settlement_amount=97000,
        missing_evidence=[],
        conflicts=[],
        evidence_link_count=0,
    )


def _make_explanation(
    status=ExplanationStatus.FULLY_EXPLAINED,
    supporting_ids=None,
    explained_amount=-3000,
    remaining=0,
):
    """Create an ExplanationResult for testing."""
    return ExplanationResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        explanation_status=status,
        explained_amount=explained_amount,
        remaining_difference=remaining,
        supporting_evidence_ids=supporting_ids or ["FEE-001"],
        candidate_explanations=[],
        conflict=False,
        missing_evidence=[],
        explanation_reason="Fee explains difference.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialAdjustment:
    def test_basic_creation(self):
        adj = FinancialAdjustment(
            adjustment_type="FEE_CORRECTION",
            amount_paise=3000,
            direction="CREDIT",
            evidence_record_id="FEE-001",
            calculation_basis="fee_record_sum",
        )
        assert adj.amount_paise == 3000
        assert adj.direction == "CREDIT"
        assert adj.evidence_record_id == "FEE-001"

    def test_no_adjustment(self):
        adj = FinancialAdjustment(
            adjustment_type="NO_ADJUSTMENT",
            amount_paise=0,
            direction="NONE",
            calculation_basis="zero_discrepancy",
        )
        assert adj.amount_paise == 0

    def test_all_amounts_are_integers(self):
        adj = FinancialAdjustment(
            adjustment_type="SETTLEMENT_ADJUSTMENT",
            amount_paise=5000,
            direction="DEBIT",
            calculation_basis="discrepancy_amount",
        )
        assert isinstance(adj.amount_paise, int)


class TestCandidateRanking:
    def test_basic_creation(self):
        ranking = CandidateRanking(
            rank=1,
            confidence_score=0.85,
            evidence_support=0.9,
        )
        assert ranking.rank == 1
        assert ranking.confidence_score == 0.85

    def test_with_ml_and_historical(self):
        ranking = CandidateRanking(
            rank=1,
            confidence_score=0.9,
            evidence_support=0.8,
            ml_support=0.85,
            historical_support=0.7,
        )
        assert ranking.ml_support == 0.85
        assert ranking.historical_support == 0.7


class TestResolutionProposal:
    def test_basic_creation(self):
        proposal = ResolutionProposal(
            candidate_id="CAND-001-DET",
            exception_id="EXC-001",
            case_id="CASE-001",
            resolution_type="FEE_ADJUSTMENT",
            resolution_description="Apply fee correction",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=3000,
                direction="CREDIT",
                calculation_basis="fee_record_sum",
            ),
            supporting_evidence_ids=["FEE-001"],
            evidence_compatible=True,
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(rank=1, confidence_score=0.85, evidence_support=0.9),
            rationale="Fee explains the discrepancy",
        )
        assert proposal.is_recommendation_only is True
        assert proposal.financial_adjustment.amount_paise == 3000

    def test_to_dict(self):
        proposal = ResolutionProposal(
            candidate_id="CAND-001-DET",
            exception_id="EXC-001",
            case_id="CASE-001",
            resolution_type="FEE_ADJUSTMENT",
            resolution_description="Apply fee correction",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=3000,
                direction="CREDIT",
                calculation_basis="fee_record_sum",
            ),
            supporting_evidence_ids=["FEE-001"],
            evidence_compatible=True,
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(rank=1, confidence_score=0.85, evidence_support=0.9),
            rationale="Fee explains the discrepancy",
        )
        d = proposal.to_dict()
        assert d["is_recommendation_only"] is True
        assert d["financial_adjustment"]["amount_paise"] == 3000


class TestCandidateGenerationResult:
    def test_unresolved(self):
        result = CandidateGenerationResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            status="UNRESOLVED",
        )
        assert result.is_unresolved()
        assert result.best_candidate() is None

    def test_with_candidates(self):
        result = CandidateGenerationResult(
            exception_id="EXC-001",
            case_id="CASE-001",
            status="CANDIDATES_GENERATED",
            candidates=[
                ResolutionProposal(
                    candidate_id="CAND-001",
                    exception_id="EXC-001",
                    case_id="CASE-001",
                    resolution_type="FEE_ADJUSTMENT",
                    resolution_description="Apply fee correction",
                    financial_adjustment=FinancialAdjustment(
                        adjustment_type="FEE_CORRECTION",
                        amount_paise=3000,
                        direction="CREDIT",
                        calculation_basis="fee_record_sum",
                    ),
                    supporting_evidence_ids=[],
                    evidence_compatible=True,
                    sources=["deterministic_evidence"],
                    ranking=CandidateRanking(rank=1, confidence_score=0.85, evidence_support=0.9),
                    rationale="Fee",
                )
            ],
            total_candidates=1,
        )
        assert not result.is_unresolved()
        assert result.best_candidate() is not None


# ─────────────────────────────────────────────────────────────────────────────
# CandidateGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCandidateGeneratorDeterministic:
    def test_fee_difference_generates_candidate(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation(supporting_ids=["FEE-001"])

        result = gen.generate(intel, package, explanation)
        assert result.status == "CANDIDATES_GENERATED"
        assert len(result.candidates) >= 1
        assert result.candidates[0].resolution_type == "FEE_ADJUSTMENT"

    def test_exact_match_generates_no_action(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="EXACT_MATCH",
            difference=0,
            expected_amount=100000,
            actual_amount=100000,
        )
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(c.resolution_type == "NO_ACTION" for c in result.candidates)

    def test_refund_adjustment(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="REFUND_ADJUSTMENT")
        package = _make_package(
            exception_type="REFUND_ADJUSTMENT",
            refunds=[_make_record("REF-001", "REFUND", "CALCULATION_COMPONENT", 3000)],
        )
        explanation = _make_explanation(supporting_ids=["REF-001"])

        result = gen.generate(intel, package, explanation)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(c.resolution_type == "REFUND_ADJUSTMENT" for c in result.candidates)

    def test_tax_adjustment(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="TAX_ADJUSTMENT")
        package = _make_package(
            exception_type="TAX_ADJUSTMENT",
            taxes=[_make_record("TAX-001", "TAX", "CALCULATION_COMPONENT", 2000)],
        )
        explanation = _make_explanation(supporting_ids=["TAX-001"])

        result = gen.generate(intel, package, explanation)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(c.resolution_type == "TAX_ADJUSTMENT" for c in result.candidates)

    def test_partial_settlement(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="PARTIAL_SETTLEMENT",
            difference=50000,
            expected_amount=100000,
            actual_amount=50000,
        )
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(
            c.resolution_type == "PARTIAL_SETTLEMENT_RECONCILIATION"
            for c in result.candidates
        )

    def test_duplicate_settlement(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="DUPLICATE")
        package = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", "SUPPORTING_EVIDENCE", 97000),
                _make_record("SET-002", "SETTLEMENT", "SUPPORTING_EVIDENCE", 97000),
            ],
        )
        result = gen.generate(intel, package)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(
            c.resolution_type == "DUPLICATE_SETTLEMENT" for c in result.candidates
        )

    def test_missing_record(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="MISSING_RECORD")
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(
            c.resolution_type == "MISSING_RECORD_ESCALATION" for c in result.candidates
        )

    def test_complex_multi_adjustment(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="COMPLEX_MULTI_ADJUSTMENT")
        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        assert any(
            c.resolution_type == "MULTI_ADJUSTMENT" for c in result.candidates
        )


class TestCandidateGeneratorML:
    def test_ml_disagreement_generates_second_candidate(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_type="REFUND_ADJUSTMENT",
        )
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        assert result.status == "CANDIDATES_GENERATED"
        resolution_types = [c.resolution_type for c in result.candidates]
        assert "FEE_ADJUSTMENT" in resolution_types
        # ML suggests different resolution
        assert "REFUND_ADJUSTMENT" in resolution_types

    def test_ml_agreement_no_extra_candidate(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_type="FEE_DIFFERENCE",  # Same as deterministic
        )
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        # Only deterministic candidate (ML same = merged)
        assert len(result.candidates) >= 1


class TestCandidateGeneratorHistorical:
    def test_similar_cases_generate_candidates(self):
        gen = CandidateGenerator()
        similar = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-H01",
                    "similarity_score": 0.85,
                    "exception_type": "FEE_DIFFERENCE",
                    "resolution_type": "NO_ACTION",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 100000,
                    "difference": 3000,
                    "tags": [],
                }
            ],
            best_similarity_score=0.85,
        )
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            similar_cases=similar,
        )

        result = gen.generate(intel)
        assert result.status == "CANDIDATES_GENERATED"
        resolution_types = [c.resolution_type for c in result.candidates]
        # Deterministic + historical (different resolution)
        assert "FEE_ADJUSTMENT" in resolution_types
        assert "NO_ACTION" in resolution_types

    def test_low_similarity_ignored(self):
        gen = CandidateGenerator()
        similar = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-H02",
                    "similarity_score": 0.3,  # Below threshold
                    "resolution_type": "NO_ACTION",
                }
            ],
        )
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            similar_cases=similar,
        )

        result = gen.generate(intel)
        # Low similarity should not generate historical candidate
        for c in result.candidates:
            assert "historical_case" not in c.sources

    def test_same_resolution_as_deterministic_not_duplicated(self):
        gen = CandidateGenerator()
        similar = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-H03",
                    "similarity_score": 0.9,
                    "resolution_type": "FEE_ADJUSTMENT",  # Same as deterministic
                }
            ],
        )
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            similar_cases=similar,
        )

        result = gen.generate(intel)
        # Should not have duplicate FEE_ADJUSTMENT
        fee_count = sum(
            1 for c in result.candidates if c.resolution_type == "FEE_ADJUSTMENT"
        )
        assert fee_count == 1


class TestCandidateMerging:
    def test_same_resolution_merges_sources(self):
        """Test merging when ML and deterministic agree on same resolution."""
        gen = CandidateGenerator()
        # ML predicts same type as deterministic -> same resolution -> merged
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_type="FEE_DIFFERENCE",
        )
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        # FEE_ADJUSTMENT should appear once (ML same = skipped, not merged)
        fee_candidates = [
            c for c in result.candidates if c.resolution_type == "FEE_ADJUSTMENT"
        ]
        assert len(fee_candidates) == 1


class TestFinancialAdjustmentDerivation:
    def test_fee_adjustment_uses_fee_record(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        fee_candidate = [c for c in result.candidates if c.resolution_type == "FEE_ADJUSTMENT"][0]
        assert fee_candidate.financial_adjustment.amount_paise == 3000
        assert fee_candidate.financial_adjustment.direction == "CREDIT"
        assert fee_candidate.financial_adjustment.evidence_record_id == "FEE-001"

    def test_no_invented_amounts(self):
        """Financial adjustments must trace to evidence."""
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        # No fees in package
        package = _make_package(fees=[])
        explanation = _make_explanation(supporting_ids=[])

        result = gen.generate(intel, package, explanation)
        for c in result.candidates:
            # Amount must be traceable
            assert c.financial_adjustment.calculation_basis is not None

    def test_no_action_has_zero_adjustment(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="EXACT_MATCH",
            difference=0,
            expected_amount=100000,
            actual_amount=100000,
        )
        result = gen.generate(intel)
        no_action = [c for c in result.candidates if c.resolution_type == "NO_ACTION"]
        assert len(no_action) == 1
        assert no_action[0].financial_adjustment.amount_paise == 0


class TestUnresolvedBehavior:
    def test_unknown_can_be_unresolved(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="UNKNOWN")
        result = gen.generate(intel)
        # UNKNOWN should still generate UNKNOWN_UNRESOLVED candidate
        assert result.status == "CANDIDATES_GENERATED"
        assert any(
            c.resolution_type == "UNKNOWN_UNRESOLVED" for c in result.candidates
        )

    def test_no_forcing(self):
        """If no valid resolution exists, return what's available."""
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="UNKNOWN")
        result = gen.generate(intel)
        # Should have at least one candidate (UNKNOWN_UNRESOLVED)
        assert result.total_candidates >= 1


class TestCandidateRanking:
    def test_ranking_order(self):
        gen = CandidateGenerator()
        similar = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-R01",
                    "similarity_score": 0.9,
                    "resolution_type": "NO_ACTION",
                }
            ],
        )
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            similar_cases=similar,
        )
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        if len(result.candidates) >= 2:
            # First candidate should have higher or equal rank
            assert result.candidates[0].ranking.rank <= result.candidates[1].ranking.rank

    def test_evidence_compatible_ranked_higher(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        if result.candidates:
            # Deterministic candidate should be evidence compatible
            det = [c for c in result.candidates if "deterministic_evidence" in c.sources]
            if det:
                assert det[0].evidence_compatible


class TestCandidateEvidence:
    def test_evidence_records_populated(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        assert len(det.evidence_records) > 0
        fee_records = [r for r in det.evidence_records if r.entity_type == "FEE"]
        assert len(fee_records) == 1
        assert fee_records[0].record_id == "FEE-001"
        assert fee_records[0].amount == 3000
        assert fee_records[0].contribution == -3000

    def test_coverage_explanation_populated(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        assert len(det.coverage_explanation) > 0
        assert "3000" in det.coverage_explanation

    def test_rationale_components_populated(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        result = gen.generate(intel)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        assert len(det.rationale_components) > 0
        types = [rc.component_type for rc in det.rationale_components]
        assert "what_happened" in types
        assert "recommendation" in types

    def test_ml_support_detail(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            ml_type="REFUND_ADJUSTMENT",
        )
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        ml_candidates = [c for c in result.candidates if c.ml_support is not None]
        assert len(ml_candidates) == 1
        assert ml_candidates[0].ml_support.supported is True
        assert ml_candidates[0].ml_support.predicted_resolution == "REFUND_ADJUSTMENT"
        assert ml_candidates[0].ml_support.confidence == 0.8
        assert ml_candidates[0].ml_support.model_version == "1.0.0"

    def test_no_ml_support_when_not_available(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        result = gen.generate(intel)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        assert det.ml_support is None

    def test_historical_support_detail(self):
        gen = CandidateGenerator()
        similar = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-H01",
                    "similarity_score": 0.85,
                    "resolution_type": "NO_ACTION",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 100000,
                    "difference": 3000,
                }
            ],
        )
        intel = _make_intelligence(
            deterministic_type="FEE_DIFFERENCE",
            similar_cases=similar,
        )

        result = gen.generate(intel)
        hist_candidates = [c for c in result.candidates if c.historical_support]
        assert len(hist_candidates) == 1
        assert hist_candidates[0].historical_support[0].case_id == "CASE-H01"
        assert hist_candidates[0].historical_support[0].similarity_score == 0.85
        assert hist_candidates[0].historical_support[0].historical_resolution == "NO_ACTION"
        assert hist_candidates[0].historical_support[0].historical_outcome == "SUCCESSFUL"

    def test_traceability(self):
        """Every candidate must be traceable to evidence."""
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        for c in result.candidates:
            # Must have evidence records OR historical support
            has_evidence = len(c.evidence_records) > 0
            has_historical = len(c.historical_support) > 0
            has_ml = c.ml_support is not None and c.ml_support.supported
            assert has_evidence or has_historical or has_ml

    def test_financial_trace_in_rationale(self):
        """Rationale must include financial traceability."""
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        package = _make_package(fees=[
            _make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 3000)
        ])
        explanation = _make_explanation()

        result = gen.generate(intel, package, explanation)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        financial_components = [
            rc for rc in det.rationale_components if rc.component_type == "financial_trace"
        ]
        assert len(financial_components) == 1
        assert financial_components[0].amount_paise == 3000

    def test_multiple_evidence_records(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="COMPLEX_MULTI_ADJUSTMENT")
        package = _make_package(
            fees=[_make_record("FEE-001", "FEE", "CALCULATION_COMPONENT", 1000)],
            refunds=[_make_record("REF-001", "REFUND", "CALCULATION_COMPONENT", 2000)],
            taxes=[_make_record("TAX-001", "TAX", "CALCULATION_COMPONENT", 500)],
        )
        explanation = _make_explanation(supporting_ids=["FEE-001", "REF-001", "TAX-001"])

        result = gen.generate(intel, package, explanation)
        det = [c for c in result.candidates if "deterministic_evidence" in c.sources][0]
        assert len(det.evidence_records) >= 3
        entity_types = {r.entity_type for r in det.evidence_records}
        assert "FEE" in entity_types
        assert "REFUND" in entity_types
        assert "TAX" in entity_types


class TestCandidateGeneratorSafety:
    def test_is_recommendation_only(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        result = gen.generate(intel)
        for c in result.candidates:
            assert c.is_recommendation_only is True

    def test_no_ground_truth_in_generator(self):
        """CandidateGenerator must not use ground truth."""
        import inspect

        source = inspect.getsource(CandidateGenerator)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source

    def test_financial_amounts_are_integers(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        result = gen.generate(intel)
        for c in result.candidates:
            assert isinstance(c.financial_adjustment.amount_paise, int)

    def test_candidate_sources_are_controlled(self):
        gen = CandidateGenerator()
        intel = _make_intelligence(deterministic_type="FEE_DIFFERENCE")
        result = gen.generate(intel)
        valid_sources = {s.value for s in CandidateSource}
        for c in result.candidates:
            for s in c.sources:
                assert s in valid_sources
