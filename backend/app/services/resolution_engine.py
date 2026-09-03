"""
Resolution Engine for Razorpay CloseLoop Phase 5E.

Integrates all pipeline stages into one unified resolution engine:

1. Deterministic reconciliation
2. Evidence retrieval + graph
3. Explanation
4. Intelligence aggregation
5. Candidate generation
6. Candidate evidence
7. Candidate scoring
8. Candidate selection

This is a RECOMMENDATION ONLY.
It must NOT execute financial actions.

DOES NOT:
- execute refunds
- change settlement records
- modify payment records
- modify merchant balances
- close exceptions
- call external financial APIs
"""

import time
from typing import Optional

from sqlalchemy.orm import Session

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult
from app.schemas.evidence_quality import EvidenceQualityResult
from app.schemas.intelligence import ExceptionIntelligence
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionConfig, SelectionStatus
from app.services.candidate_generator import CandidateGenerator
from app.services.candidate_scorer import CandidateScoringService
from app.services.candidate_selector import CandidateSelector
from app.services.evidence_retrieval import EvidenceRetrievalService
from app.services.explanation_engine import DeterministicExplanationEngine
from app.services.evidence_quality import EvidenceQualityScorer
from app.services.evidence_graph import EvidenceGraphBuilder
from app.ml.resolution import EXCEPTION_TO_RESOLUTION_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Engine
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionEngine:
    """
    Unified resolution engine that integrates all pipeline stages.

    Given an exception_id, produces a complete resolution recommendation
    with full audit trail.

    This is a RECOMMENDATION ONLY.
    It must NOT execute financial actions.
    """

    def __init__(
        self,
        session: Session,
        classifier=None,
        resolution_predictor=None,
        similarity_service=None,
        selection_config: Optional[SelectionConfig] = None,
    ):
        """Initialize the resolution engine.

        Args:
            session: SQLAlchemy session
            classifier: ExceptionClassifierService (optional, for ML)
            resolution_predictor: ResolutionPredictor (optional, for ML resolution)
            similarity_service: SimilarityService (optional, for historical similarity)
            selection_config: SelectionConfig (optional, for selection thresholds)
        """
        self.session = session
        self.evidence_service = EvidenceRetrievalService(session)
        self.explanation_engine = DeterministicExplanationEngine()
        self.quality_scorer = EvidenceQualityScorer()
        self.graph_builder = EvidenceGraphBuilder()
        self.candidate_generator = CandidateGenerator()
        self.candidate_scorer = CandidateScoringService()
        self.candidate_selector = CandidateSelector(
            scoring_service=self.candidate_scorer,
            selection_config=selection_config,
        )
        self.classifier = classifier
        self.resolution_predictor = resolution_predictor
        self.similarity_service = similarity_service

    def resolve(self, exception_id: str) -> Optional[ResolutionEngineResult]:
        """Run the full resolution pipeline for an exception.

        Args:
            exception_id: The exception to resolve

        Returns:
            ResolutionEngineResult or None if exception not found
        """
        start_time = time.time()

        # 1. Evidence retrieval
        package = self.evidence_service.retrieve_by_exception_id(
            exception_id, persist_links=True
        )
        if package is None:
            return None

        # 2. Evidence graph (built for completeness, not persisted)
        graph = self.graph_builder.build(package)

        # 3. Explanation
        explanation = self.explanation_engine.explain(package, graph)

        # 4. Evidence quality
        quality = self.quality_scorer.score(package, explanation)

        # 5. Build intelligence (without full ML pipeline — use available signals)
        intelligence = self._build_intelligence(
            package, explanation, quality
        )

        # 6. Generate candidates
        generation_result = self.candidate_generator.generate(
            intelligence, package, explanation, quality
        )

        # 7. Score and select
        selection_result = self.candidate_selector.select(
            generation_result, intelligence
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # 8. Build engine result
        return self._build_engine_result(
            package, intelligence, selection_result, processing_time_ms
        )

    def _build_intelligence(
        self,
        package: EvidencePackage,
        explanation: ExplanationResult,
        quality: EvidenceQualityResult,
    ) -> ExceptionIntelligence:
        """Build ExceptionIntelligence from available signals.

        Uses deterministic signals. ML is optional.
        """
        from app.schemas.intelligence import (
            ClassificationResult,
            EvidenceIntelligence,
            SimilarCasesIntelligence,
        )
        from app.schemas.intelligence import RecommendationStatus

        # Classification
        deterministic_type = package.exception_type
        ml_type = None
        ml_probs = None
        ml_version = None

        if self.classifier:
            try:
                from app.ml.features import extract_features
                feature_dict = self._extract_features(package)
                ml_pred = self.classifier.predict(feature_dict)
                ml_type = ml_pred.predicted_type
                ml_probs = ml_pred.probabilities
                ml_version = ml_pred.model_version
            except Exception:
                pass

        classification = ClassificationResult(
            deterministic_type=deterministic_type,
            ml_predicted_type=ml_type,
            ml_probabilities=ml_probs,
            ml_model_version=ml_version,
            agreement=ml_type is None or ml_type == deterministic_type,
        )

        # Evidence intelligence
        missing = [m.entity_type for m in package.missing_evidence]
        evidence_intel = EvidenceIntelligence(
            explanation_status=explanation.explanation_status.value,
            explained_amount=explanation.explained_amount,
            remaining_difference=explanation.remaining_difference,
            supporting_evidence_ids=explanation.supporting_evidence_ids,
            evidence_coverage=quality.coverage_score,
            consistency_score=quality.consistency_score,
            has_conflict=quality.conflict,
            missing_evidence=missing,
            explanation_reason=explanation.explanation_reason,
            evidence_link_count=package.evidence_link_count,
        )

        # Historical similarity (simplified — no full search here)
        similar_intel = SimilarCasesIntelligence(
            query_embedded=False,
            total_indexed=0,
            top_k=5,
        )

        return ExceptionIntelligence(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            merchant_id=package.merchant_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            classification=classification,
            evidence=evidence_intel,
            similar_cases=similar_intel,
            recommendation_status=RecommendationStatus.SUPPORTED,
        )

    def _extract_features(self, package: EvidencePackage) -> dict:
        """Extract features from a package for ML classification."""
        from app.ml.features import extract_features

        payment_amount = package.payment.amount if package.payment else 0

        return extract_features(
            difference=package.difference,
            payment_amount=payment_amount,
            settlement_amount=package.total_settlement_amount,
            refund_amount=package.total_refund_amount,
            fee_amount=package.total_fee_amount,
            tax_amount=package.total_tax_amount,
            adjustment_amount=package.total_adjustment_amount,
            num_settlements=len(package.settlements),
            num_refunds=len(package.refunds),
            num_fees=len(package.fees),
            num_taxes=len(package.taxes),
            num_adjustments=len(package.adjustments),
            has_missing_evidence=len(package.missing_evidence) > 0,
            num_missing_evidence=len(package.missing_evidence),
            evidence_coverage=0.0,
            consistency_score=1.0,
            fully_explained=False,
            partially_explained=False,
            has_conflict=package.has_conflicts(),
            supporting_evidence_count=0,
            num_candidate_explanations=0,
        )

    def _build_engine_result(
        self,
        package: EvidencePackage,
        intelligence: ExceptionIntelligence,
        selection_result,
        processing_time_ms: float,
    ) -> ResolutionEngineResult:
        """Build the final engine result from all pipeline outputs."""
        return ResolutionEngineResult(
            exception_id=package.exception_id,
            case_id=package.case_id,
            payment_id=package.payment_id,
            merchant_id=package.merchant_id,
            expected_amount=package.expected_amount,
            actual_amount=package.actual_amount,
            difference=package.difference,
            status=selection_result.status,
            selected_resolution=(
                selection_result.selected_candidate.resolution_type
                if selection_result.selected_candidate
                else None
            ),
            selected_candidate=selection_result.selected_candidate,
            selected_score=selection_result.selected_score,
            ranked_candidates=[
                selection_result.selected_candidate
            ] + selection_result.alternatives
            if selection_result.selected_candidate
            else selection_result.alternatives,
            candidate_scores=(
                [selection_result.selected_score] + selection_result.alternative_scores
                if selection_result.selected_score
                else selection_result.alternative_scores
            ),
            confidence=selection_result.confidence,
            confidence_factors=selection_result.confidence_factors,
            risk_category=selection_result.risk_category,
            risk_factors=selection_result.risk_factors,
            explainability=selection_result.explainability,
            rejection_reasons=selection_result.rejection_reasons,
            deterministic_exception_type=intelligence.classification.deterministic_type,
            ml_exception_type=intelligence.classification.ml_predicted_type,
            classification_agreement=intelligence.classification.agreement,
            evidence_explanation_status=intelligence.evidence.explanation_status,
            evidence_coverage=intelligence.evidence.evidence_coverage,
            evidence_consistency=intelligence.evidence.consistency_score,
            has_conflict=intelligence.evidence.has_conflict,
            is_novel=selection_result.selected_score.novelty_penalty > 0 if selection_result.selected_score else False,
            missing_evidence=intelligence.evidence.missing_evidence,
            pipeline_version=ENGINE_VERSION,
            processing_time_ms=processing_time_ms,
        )
