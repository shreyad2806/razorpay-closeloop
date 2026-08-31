"""
Tests for Razorpay CloseLoop Phase 4G — Exception Intelligence Aggregator.

Tests cover:
- Intelligence schema structure
- ClassificationResult
- EvidenceIntelligence
- SimilarCasesIntelligence
- ResolutionCandidate
- RecommendationStatus
- Conflict detection logic
- No execution guarantee
- Ground truth separation
- Summary generation
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
    ResolutionCandidate,
    SimilarCasesIntelligence,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_classification(
    deterministic_type="FEE_DIFFERENCE",
    ml_type=None,
    agreement=True,
):
    """Create a ClassificationResult for testing."""
    return ClassificationResult(
        deterministic_type=deterministic_type,
        ml_predicted_type=ml_type,
        ml_probabilities=None,
        ml_model_version=None,
        agreement=agreement,
        disagreement_note=None if agreement else f"Det: {deterministic_type}, ML: {ml_type}",
    )


def _make_evidence_intel(
    explanation_status="FULLY_EXPLAINED",
    coverage=1.0,
    consistency=0.85,
    has_conflict=False,
    missing=None,
):
    """Create EvidenceIntelligence for testing."""
    return EvidenceIntelligence(
        explanation_status=explanation_status,
        explained_amount=-3000,
        remaining_difference=0,
        supporting_evidence_ids=["FEE-001"],
        evidence_coverage=coverage,
        consistency_score=consistency,
        has_conflict=has_conflict,
        missing_evidence=missing or [],
        explanation_reason="Fee explains difference.",
    )


def _make_intelligence(
    exception_id="EXC-001",
    recommendation_status=RecommendationStatus.SUPPORTED,
    conflicts=None,
    candidates=None,
    agreement=True,
    recommendation_notes=None,
):
    """Create an ExceptionIntelligence for testing."""
    return ExceptionIntelligence(
        exception_id=exception_id,
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        classification=_make_classification(agreement=agreement),
        evidence=_make_evidence_intel(),
        similar_cases=SimilarCasesIntelligence(),
        resolution_candidates=candidates or [],
        conflicts=conflicts or [],
        recommendation_status=recommendation_status,
        recommendation_notes=recommendation_notes or ["Test note"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClassificationResult:
    def test_basic_creation(self):
        cr = _make_classification()
        assert cr.deterministic_type == "FEE_DIFFERENCE"
        assert cr.agreement is True

    def test_disagreement(self):
        cr = _make_classification(ml_type="REFUND_ADJUSTMENT", agreement=False)
        assert cr.agreement is False
        assert cr.disagreement_note is not None
        assert "FEE_DIFFERENCE" in cr.disagreement_note

    def test_no_ml(self):
        cr = _make_classification()
        assert cr.ml_predicted_type is None
        assert cr.ml_probabilities is None
        assert cr.agreement is True  # No ML = agreement


class TestEvidenceIntelligence:
    def test_basic_creation(self):
        ei = _make_evidence_intel()
        assert ei.explanation_status == "FULLY_EXPLAINED"
        assert ei.evidence_coverage == 1.0

    def test_partial_explanation(self):
        ei = _make_evidence_intel(
            explanation_status="PARTIALLY_EXPLAINED",
            coverage=0.5,
        )
        assert ei.explanation_status == "PARTIALLY_EXPLAINED"
        assert ei.evidence_coverage == 0.5

    def test_missing_evidence(self):
        ei = _make_evidence_intel(missing=["SETTLEMENT", "FEE"])
        assert len(ei.missing_evidence) == 2
        assert "SETTLEMENT" in ei.missing_evidence


class TestSimilarCasesIntelligence:
    def test_empty(self):
        sci = SimilarCasesIntelligence()
        assert sci.query_embedded is True
        assert len(sci.similar_cases) == 0

    def test_with_cases(self):
        sci = SimilarCasesIntelligence(
            similar_cases=[
                {
                    "case_id": "CASE-H01",
                    "similarity_score": 0.92,
                    "exception_type": "FEE_DIFFERENCE",
                    "resolution_type": "FEE_ADJUSTMENT",
                    "resolution_outcome": "SUCCESSFUL",
                    "payment_amount": 100000,
                    "difference": 3000,
                    "tags": ["fee"],
                }
            ],
            best_similarity_score=0.92,
        )
        assert len(sci.similar_cases) == 1
        assert sci.best_similarity_score == 0.92


class TestResolutionCandidate:
    def test_deterministic_candidate(self):
        rc = ResolutionCandidate(
            resolution_type="FEE_ADJUSTMENT",
            source="DETERMINISTIC_EVIDENCE",
            supporting_evidence_ids=["FEE-001"],
            evidence_compatible=True,
        )
        assert rc.source == "DETERMINISTIC_EVIDENCE"
        assert rc.evidence_compatible is True

    def test_ml_candidate(self):
        rc = ResolutionCandidate(
            resolution_type="FEE_ADJUSTMENT",
            source="ML_PREDICTION",
            evidence_compatible=True,
            confidence=0.95,
        )
        assert rc.confidence == 0.95

    def test_historical_candidate(self):
        rc = ResolutionCandidate(
            resolution_type="FEE_ADJUSTMENT",
            source="HISTORICAL_SIMILARITY",
            evidence_compatible=True,
            similarity_score=0.88,
            historical_case_id="CASE-H01",
        )
        assert rc.similarity_score == 0.88
        assert rc.historical_case_id == "CASE-H01"


class TestRecommendationStatus:
    def test_status_values(self):
        assert RecommendationStatus.SUPPORTED.value == "SUPPORTED"
        assert RecommendationStatus.PARTIALLY_SUPPORTED.value == "PARTIALLY_SUPPORTED"
        assert RecommendationStatus.CONFLICTING.value == "CONFLICTING"
        assert RecommendationStatus.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"


class TestExceptionIntelligence:
    def test_basic_creation(self):
        intel = _make_intelligence()
        assert intel.exception_id == "EXC-001"
        assert intel.is_intelligence_only is True
        assert intel.pipeline_version == "1.0.0"

    def test_financial_context(self):
        intel = _make_intelligence()
        assert intel.expected_amount == 100000
        assert intel.actual_amount == 97000
        assert intel.difference == 3000

    def test_has_conflicts_true(self):
        intel = _make_intelligence(
            conflicts=["Classification disagreement", "Resolution conflict"]
        )
        assert intel.has_conflicts() is True

    def test_has_conflicts_false(self):
        intel = _make_intelligence()
        assert intel.has_conflicts() is False

    def test_is_supported(self):
        intel = _make_intelligence(recommendation_status=RecommendationStatus.SUPPORTED)
        assert intel.is_supported() is True

    def test_is_not_supported(self):
        intel = _make_intelligence(recommendation_status=RecommendationStatus.CONFLICTING)
        assert intel.is_supported() is False

    def test_summary(self):
        intel = _make_intelligence(
            candidates=[
                ResolutionCandidate(
                    resolution_type="FEE_ADJUSTMENT",
                    source="DETERMINISTIC_EVIDENCE",
                    evidence_compatible=True,
                )
            ]
        )
        summary = intel.summary()
        assert "EXC-001" in summary
        assert "FEE_DIFFERENCE" in summary
        assert "SUPPORTED" in summary

    def test_timestamp(self):
        intel = _make_intelligence()
        assert isinstance(intel.processing_timestamp, datetime)


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictHandling:
    def test_classification_conflict(self):
        intel = _make_intelligence(
            conflicts=["Classification disagreement: Det: FEE, ML: REFUND"],
            recommendation_status=RecommendationStatus.CONFLICTING,
        )
        assert intel.has_conflicts()
        assert intel.recommendation_status == RecommendationStatus.CONFLICTING

    def test_resolution_conflict(self):
        candidates = [
            ResolutionCandidate(
                resolution_type="FEE_ADJUSTMENT",
                source="ML_PREDICTION",
                evidence_compatible=True,
            ),
            ResolutionCandidate(
                resolution_type="TIMING_RECONCILIATION",
                source="HISTORICAL_SIMILARITY",
                evidence_compatible=True,
            ),
        ]
        intel = _make_intelligence(
            candidates=candidates,
            conflicts=["Resolution candidates disagree: FEE_ADJUSTMENT, TIMING_RECONCILIATION"],
            recommendation_status=RecommendationStatus.CONFLICTING,
        )
        assert intel.has_conflicts()

    def test_no_conflict_when_agree(self):
        candidates = [
            ResolutionCandidate(
                resolution_type="FEE_ADJUSTMENT",
                source="DETERMINISTIC_EVIDENCE",
                evidence_compatible=True,
            ),
            ResolutionCandidate(
                resolution_type="FEE_ADJUSTMENT",
                source="ML_PREDICTION",
                evidence_compatible=True,
            ),
        ]
        intel = _make_intelligence(
            candidates=candidates,
            recommendation_status=RecommendationStatus.SUPPORTED,
        )
        assert not intel.has_conflicts()


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecommendationHandling:
    def test_supported_status(self):
        intel = _make_intelligence(
            recommendation_status=RecommendationStatus.SUPPORTED,
            recommendation_notes=["All sources agree on FEE_ADJUSTMENT"],
        )
        assert intel.is_supported()
        assert len(intel.recommendation_notes) > 0

    def test_insufficient_evidence(self):
        intel = _make_intelligence(
            recommendation_status=RecommendationStatus.INSUFFICIENT_EVIDENCE,
            recommendation_notes=["No resolution candidates generated"],
        )
        assert not intel.is_supported()

    def test_partially_supported(self):
        intel = _make_intelligence(
            recommendation_status=RecommendationStatus.PARTIALLY_SUPPORTED,
            recommendation_notes=["Sources suggest different resolutions"],
        )
        assert not intel.is_supported()
        assert intel.recommendation_status == RecommendationStatus.PARTIALLY_SUPPORTED


# ─────────────────────────────────────────────────────────────────────────────
# Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_is_always_intelligence_only(self):
        intel = _make_intelligence()
        assert intel.is_intelligence_only is True

    def test_no_execution_fields(self):
        """Intelligence must not have fields that could execute financial actions."""
        intel = _make_intelligence()
        assert not hasattr(intel, "execute_resolution")
        assert not hasattr(intel, "settle_payment")
        assert not hasattr(intel, "issue_refund")
        assert not hasattr(intel, "close_exception")

    def test_no_ground_truth_in_intelligence(self):
        """Intelligence must not contain ground truth labels."""
        import inspect

        source = inspect.getsource(ExceptionIntelligence)
        assert "true_exception_type" not in source
        assert "true_resolution" not in source
        assert "resolvable" not in source
        assert "risk_category" not in source

    def test_no_ground_truth_in_classification(self):
        """Classification must not use ground truth."""
        import inspect

        source = inspect.getsource(ClassificationResult)
        assert "true_exception_type" not in source

    def test_no_ground_truth_in_evidence_intel(self):
        """Evidence intelligence must not use ground truth."""
        import inspect

        source = inspect.getsource(EvidenceIntelligence)
        assert "true_" not in source


# ─────────────────────────────────────────────────────────────────────────────
# Data Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataIntegrity:
    def test_amounts_are_integers(self):
        intel = _make_intelligence()
        assert isinstance(intel.expected_amount, int)
        assert isinstance(intel.actual_amount, int)
        assert isinstance(intel.difference, int)

    def test_difference_consistency(self):
        intel = _make_intelligence()
        assert intel.expected_amount - intel.actual_amount == intel.difference

    def test_evidence_ids_are_strings(self):
        intel = _make_intelligence()
        for eid in intel.evidence.supporting_evidence_ids:
            assert isinstance(eid, str)

    def test_empty_candidates(self):
        intel = _make_intelligence(candidates=[])
        assert len(intel.resolution_candidates) == 0

    def test_multiple_candidates(self):
        candidates = [
            ResolutionCandidate(
                resolution_type="FEE_ADJUSTMENT",
                source="DETERMINISTIC_EVIDENCE",
                evidence_compatible=True,
            ),
            ResolutionCandidate(
                resolution_type="FEE_ADJUSTMENT",
                source="ML_PREDICTION",
                evidence_compatible=True,
                confidence=0.9,
            ),
        ]
        intel = _make_intelligence(candidates=candidates)
        assert len(intel.resolution_candidates) == 2
