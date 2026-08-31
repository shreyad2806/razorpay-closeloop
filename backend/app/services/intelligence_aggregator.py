"""
Exception Intelligence Aggregator for Razorpay CloseLoop Phase 4G.

Combines all intelligence sources into one unified result:
- Deterministic reconciliation
- Evidence retrieval + explanation
- ML classification + resolution prediction
- Historical similarity
- Conflict detection

This is an INTELLIGENCE OUTPUT only.
It must NOT modify financial records.

DOES NOT execute financial actions.
Those capabilities belong to future guarded agent workflows.
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult
from app.schemas.evidence_quality import EvidenceQualityResult
from app.schemas.similarity import SimilaritySearchResult
from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION
from app.schemas.intelligence import (
    ClassificationResult,
    ExceptionIntelligence,
    EvidenceIntelligence,
    RecommendationStatus,
    ResolutionCandidate,
    SimilarCasesIntelligence,
)
from app.schemas.enums import ExceptionType
from app.services.evidence_retrieval import EvidenceRetrievalService
from app.services.explanation_engine import DeterministicExplanationEngine
from app.services.evidence_quality import EvidenceQualityScorer
from app.services.evidence_graph import EvidenceGraphBuilder
from app.ml.features import extract_features, validate_features
from app.ml.resolution import (
    ResolutionPredictor,
    EvidenceCompatibilityChecker,
    EXCEPTION_TO_RESOLUTION_MAP,
)
from app.services.similarity_service import SimilarityService


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

INTELLIGENCE_PIPELINE_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator Service
# ─────────────────────────────────────────────────────────────────────────────


class ExceptionIntelligenceAggregator:
    """
    Combines all Phase 4 intelligence sources into one unified result.

    Pipeline:
    1. Evidence retrieval
    2. Evidence graph
    3. Deterministic explanation
    4. Evidence quality scoring
    5. Feature engineering
    6. ML classification (optional)
    7. Resolution prediction (optional)
    8. Historical similarity (optional)
    9. Resolution candidate generation
    10. Conflict detection
    11. Recommendation status

    This component produces intelligence only.
    It must NOT modify financial records.
    """

    def __init__(
        self,
        session: Session,
        classifier=None,
        resolution_predictor=None,
        similarity_service: Optional[SimilarityService] = None,
        top_k_similar: int = 5,
    ):
        """Initialize the aggregator.

        Args:
            session: SQLAlchemy session
            classifier: ExceptionClassifierService (optional, for ML classification)
            resolution_predictor: ResolutionPredictor (optional, for ML resolution)
            similarity_service: SimilarityService (optional, for historical similarity)
            top_k_similar: Number of similar cases to retrieve
        """
        self.session = session
        self.evidence_service = EvidenceRetrievalService(session)
        self.explanation_engine = DeterministicExplanationEngine()
        self.quality_scorer = EvidenceQualityScorer()
        self.graph_builder = EvidenceGraphBuilder()
        self.classifier = classifier
        self.resolution_predictor = resolution_predictor
        self.similarity_service = similarity_service
        self.top_k_similar = top_k_similar

    def aggregate(
        self,
        exception_id: Optional[str] = None,
        case_id: Optional[str] = None,
        reconciliation_result=None,
    ) -> Optional[ExceptionIntelligence]:
        """Run the full intelligence pipeline for an exception.

        Args:
            exception_id: Exception to analyze (provide one of exception_id/case_id)
            case_id: Case to analyze
            reconciliation_result: Optional pre-computed reconciliation result

        Returns:
            ExceptionIntelligence or None if exception not found
        """
        # 1. Evidence retrieval
        if exception_id:
            package = self.evidence_service.retrieve_by_exception_id(
                exception_id, persist_links=True
            )
        elif case_id:
            package = self.evidence_service.retrieve_by_case_id(
                case_id, persist_links=True
            )
        else:
            return None

        if package is None:
            return None

        # 2. Evidence graph (built but not persisted to DB in this pipeline)
        graph = self.graph_builder.build(package)

        # 3. Deterministic explanation
        explanation = self.explanation_engine.explain(package, graph)

        # 4. Evidence quality scoring
        quality = self.quality_scorer.score(package, explanation)

        # 5. Classification (deterministic + optional ML)
        classification = self._classify(package, reconciliation_result)

        # 6. Evidence intelligence
        evidence_intel = self._build_evidence_intelligence(
            package, explanation, quality
        )

        # 7. Historical similarity (optional)
        similar_intel = self._search_similar(package)

        # 8. Resolution candidates
        candidates = self._generate_candidates(
            package, explanation, classification, similar_intel
        )

        # 9. Conflict detection
        conflicts = self._detect_conflicts(
            classification, explanation, candidates
        )

        # 10. Recommendation status
        status, notes = self._determine_status(
            classification, evidence_intel, candidates, conflicts
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
            resolution_candidates=candidates,
            conflicts=conflicts,
            recommendation_status=status,
            recommendation_notes=notes,
            pipeline_version=INTELLIGENCE_PIPELINE_VERSION,
        )

    def _classify(self, package, reconciliation_result) -> ClassificationResult:
        """Build classification from deterministic + ML sources."""
        # Deterministic type from evidence package
        deterministic_type = package.exception_type

        # ML classification (optional)
        ml_type = None
        ml_probs = None
        ml_version = None

        if self.classifier:
            try:
                feature_dict = self._extract_features_for_classification(package)
                ml_pred = self.classifier.predict(feature_dict)
                ml_type = ml_pred.predicted_type
                ml_probs = ml_pred.probabilities
                ml_version = ml_pred.model_version
            except Exception:
                # ML failure should not block the pipeline
                pass

        # Agreement check
        agreement = ml_type is None or ml_type == deterministic_type
        disagreement_note = None
        if not agreement:
            disagreement_note = (
                f"Deterministic: {deterministic_type}, ML: {ml_type}"
            )

        return ClassificationResult(
            deterministic_type=deterministic_type,
            ml_predicted_type=ml_type,
            ml_probabilities=ml_probs,
            ml_model_version=ml_version,
            agreement=agreement,
            disagreement_note=disagreement_note,
        )

    def _build_evidence_intelligence(
        self,
        package: EvidencePackage,
        explanation: ExplanationResult,
        quality: EvidenceQualityResult,
    ) -> EvidenceIntelligence:
        """Build evidence intelligence from explanation + quality."""
        missing = [
            m.entity_type for m in package.missing_evidence
        ]
        return EvidenceIntelligence(
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

    def _search_similar(self, package: EvidencePackage) -> SimilarCasesIntelligence:
        """Search for similar historical cases."""
        if self.similarity_service is None:
            return SimilarCasesIntelligence(
                query_embedded=False,
                total_indexed=0,
                top_k=self.top_k_similar,
            )

        try:
            query_dict = {
                "case_id": package.case_id,
                "exception_type": package.exception_type,
                "financial_context": {
                    "payment_amount": (
                        package.payment.amount if package.payment else 0
                    ),
                    "expected_amount": package.expected_amount,
                    "actual_amount": package.actual_amount,
                    "difference": package.difference,
                    "total_refunds": package.total_refund_amount,
                    "total_fees": package.total_fee_amount,
                    "total_taxes": package.total_tax_amount,
                    "total_adjustments": package.total_adjustment_amount,
                },
                "supporting_evidence_count": len(
                    package.settlements
                    + package.refunds
                    + package.fees
                    + package.taxes
                    + package.adjustments
                ),
            }

            result = self.similarity_service.search(
                query_dict, top_k=self.top_k_similar
            )

            similar_list = []
            for sc in result.similar_cases:
                similar_list.append(
                    {
                        "case_id": sc.case_id,
                        "similarity_score": sc.similarity_score,
                        "exception_type": sc.exception_type,
                        "resolution_type": sc.resolution_type,
                        "resolution_outcome": sc.resolution_outcome,
                        "payment_amount": sc.payment_amount,
                        "difference": sc.difference,
                        "tags": sc.tags,
                    }
                )

            best_score = (
                result.similar_cases[0].similarity_score
                if result.similar_cases
                else None
            )

            return SimilarCasesIntelligence(
                query_embedded=True,
                total_indexed=result.total_indexed,
                top_k=self.top_k_similar,
                similar_cases=similar_list,
                best_similarity_score=best_score,
                embedding_model=result.embedding_model,
            )
        except Exception:
            return SimilarCasesIntelligence(
                query_embedded=False,
                total_indexed=0,
                top_k=self.top_k_similar,
            )

    def _generate_candidates(
        self,
        package: EvidencePackage,
        explanation: ExplanationResult,
        classification: ClassificationResult,
        similar_intel: SimilarCasesIntelligence,
    ) -> List[ResolutionCandidate]:
        """Generate resolution candidates from multiple sources."""
        candidates = []

        # 1. Deterministic evidence-based candidate
        det_resolution = EXCEPTION_TO_RESOLUTION_MAP.get(
            classification.deterministic_type
        )
        if det_resolution:
            compatible, notes = EvidenceCompatibilityChecker.check(
                det_resolution, package, explanation
            )
            candidates.append(
                ResolutionCandidate(
                    resolution_type=det_resolution,
                    source="DETERMINISTIC_EVIDENCE",
                    supporting_evidence_ids=explanation.supporting_evidence_ids,
                    evidence_compatible=compatible,
                    notes=notes[0] if notes else None,
                )
            )

        # 2. ML resolution prediction (optional)
        if self.resolution_predictor:
            try:
                feature_dict = self._extract_features_for_classification(package)
                ml_resolution = self.resolution_predictor.predict(
                    feature_dict, package, explanation
                )
                candidates.append(
                    ResolutionCandidate(
                        resolution_type=ml_resolution.predicted_resolution,
                        source="ML_PREDICTION",
                        supporting_evidence_ids=ml_resolution.supporting_evidence_ids,
                        evidence_compatible=ml_resolution.evidence_compatible,
                        confidence=max(ml_resolution.probabilities.values())
                        if ml_resolution.probabilities
                        else None,
                        notes=ml_resolution.compatibility_notes[0]
                        if ml_resolution.compatibility_notes
                        else None,
                    )
                )
            except Exception:
                pass

        # 3. Historical similarity candidates
        for sc in similar_intel.similar_cases[:3]:
            candidates.append(
                ResolutionCandidate(
                    resolution_type=sc["resolution_type"],
                    source="HISTORICAL_SIMILARITY",
                    supporting_evidence_ids=[],
                    evidence_compatible=True,  # Historical cases were resolved
                    similarity_score=sc["similarity_score"],
                    historical_case_id=sc["case_id"],
                    notes=f"Historical case {sc['case_id']} with similarity {sc['similarity_score']:.3f}",
                )
            )

        return candidates

    def _detect_conflicts(
        self,
        classification: ClassificationResult,
        explanation: ExplanationResult,
        candidates: List[ResolutionCandidate],
    ) -> List[str]:
        """Detect conflicts between intelligence sources."""
        conflicts = []

        # Classification disagreement
        if not classification.agreement:
            conflicts.append(
                f"Classification disagreement: {classification.disagreement_note}"
            )

        # Explanation conflict
        if explanation.conflict:
            conflicts.append("Explanation engine found conflicting evidence")

        # Resolution candidate disagreement
        if len(candidates) >= 2:
            resolutions = [c.resolution_type for c in candidates]
            unique = set(resolutions)
            if len(unique) > 1:
                conflicts.append(
                    f"Resolution candidates disagree: {', '.join(sorted(unique))}"
                )

        return conflicts

    def _determine_status(
        self,
        classification: ClassificationResult,
        evidence_intel: EvidenceIntelligence,
        candidates: List[ResolutionCandidate],
        conflicts: List[str],
    ) -> tuple:
        """Determine recommendation status and notes."""
        notes = []

        # Check for conflicts
        if conflicts:
            notes.append(f"{len(conflicts)} conflict(s) detected")
            return RecommendationStatus.CONFLICTING, notes

        # Check evidence sufficiency
        if evidence_intel.explanation_status == "UNEXPLAINED":
            notes.append("Evidence does not explain the discrepancy")
            return RecommendationStatus.INSUFFICIENT_EVIDENCE, notes

        if evidence_intel.explanation_status == "CONFLICTING":
            notes.append("Multiple conflicting explanations found")
            return RecommendationStatus.CONFLICTING, notes

        # Check if candidates agree
        if candidates:
            unique_resolutions = set(c.resolution_type for c in candidates)
            compatible_count = sum(
                1 for c in candidates if c.evidence_compatible
            )

            if len(unique_resolutions) == 1:
                if compatible_count == len(candidates):
                    notes.append(
                        f"All {len(candidates)} sources agree on {unique_resolutions.pop()}"
                    )
                    return RecommendationStatus.SUPPORTED, notes
                else:
                    notes.append(
                        "Sources agree but some lack evidence compatibility"
                    )
                    return RecommendationStatus.PARTIALLY_SUPPORTED, notes
            else:
                notes.append(
                    f"Candidates suggest different resolutions: {', '.join(sorted(unique_resolutions))}"
                )
                return RecommendationStatus.PARTIALLY_SUPPORTED, notes

        # No candidates
        notes.append("No resolution candidates generated")
        return RecommendationStatus.INSUFFICIENT_EVIDENCE, notes

    def _extract_features_for_classification(
        self, package: EvidencePackage
    ) -> Dict[str, float]:
        """Extract features from a package for ML classification."""
        # Build a minimal feature dict from the package
        payment_amount = package.payment.amount if package.payment else 0
        refund_amount = package.total_refund_amount
        fee_amount = package.total_fee_amount
        tax_amount = package.total_tax_amount
        adjustment_amount = package.total_adjustment_amount
        settlement_amount = package.total_settlement_amount

        return extract_features(
            difference=package.difference,
            payment_amount=payment_amount,
            settlement_amount=settlement_amount,
            refund_amount=refund_amount,
            fee_amount=fee_amount,
            tax_amount=tax_amount,
            adjustment_amount=adjustment_amount,
            num_settlements=len(package.settlements),
            num_refunds=len(package.refunds),
            num_fees=len(package.fees),
            num_taxes=len(package.taxes),
            num_adjustments=len(package.adjustments),
            has_missing_evidence=len(package.missing_evidence) > 0,
            num_missing_evidence=len(package.missing_evidence),
            evidence_coverage=0.0,  # Will be filled by quality scorer
            consistency_score=1.0,  # Default before scoring
            fully_explained=False,
            partially_explained=False,
            has_conflict=package.has_conflicts(),
            supporting_evidence_count=0,
            num_candidate_explanations=0,
        )
