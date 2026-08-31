"""
Resolution Candidate Generator for Razorpay CloseLoop Phase 5A.

Turns intelligence into concrete, ranked financial resolution proposals.

Each candidate must:
- Come from actual intelligence signals
- Have explicit financial adjustments traced to evidence
- Never invent amounts
- Be mergeable if multiple sources agree
- Support an UNRESOLVED status when no valid candidate exists

This is a RECOMMENDATION ONLY.
It must NOT execute financial actions.
"""

from typing import Dict, List, Optional, Tuple

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.evidence_quality import EvidenceQualityResult
from app.schemas.intelligence import (
    ClassificationResult,
    ExceptionIntelligence,
    EvidenceIntelligence,
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
from app.schemas.enums import ResolutionType
from app.ml.resolution import (
    EXCEPTION_TO_RESOLUTION_MAP,
    EvidenceCompatibilityChecker,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_VERSION = "1.0.0"

# Resolution descriptions
RESOLUTION_DESCRIPTIONS = {
    "NO_ACTION": "No action required — settlement matches expected amount",
    "FEE_ADJUSTMENT": "Apply fee correction to reconcile the discrepancy",
    "REFUND_ADJUSTMENT": "Apply refund adjustment to reconcile the discrepancy",
    "TAX_ADJUSTMENT": "Apply tax correction to reconcile the discrepancy",
    "TIMING_RECONCILIATION": "Reconcile timing difference — funds expected but not yet settled",
    "PARTIAL_SETTLEMENT_RECONCILIATION": "Accept partial settlement and reconcile remainder",
    "DUPLICATE_SETTLEMENT": "Remove duplicate settlement and correct the account",
    "MISSING_RECORD_ESCALATION": "Escalate missing record for manual investigation",
    "MULTI_ADJUSTMENT": "Apply multiple corrections to reconcile the discrepancy",
    "UNKNOWN_UNRESOLVED": "Unable to determine resolution — escalate for investigation",
}


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Generator
# ─────────────────────────────────────────────────────────────────────────────


class CandidateGenerator:
    """
    Generates ranked resolution candidates from intelligence signals.

    Sources:
    1. Deterministic evidence-based resolution
    2. ML resolution prediction
    3. Historical similar case resolutions
    4. Exception type mapping
    5. Financial discrepancy analysis

    Financial adjustments are ALWAYS derived from actual evidence.
    Never invented amounts.
    """

    def __init__(self):
        self.compat_checker = EvidenceCompatibilityChecker()

    def generate(
        self,
        intelligence: ExceptionIntelligence,
        package: Optional[EvidencePackage] = None,
        explanation: Optional[ExplanationResult] = None,
        quality: Optional[EvidenceQualityResult] = None,
    ) -> CandidateGenerationResult:
        """Generate resolution candidates from intelligence.

        Args:
            intelligence: ExceptionIntelligence from Phase 4 aggregator
            package: Optional EvidencePackage for financial detail
            explanation: Optional ExplanationResult for evidence detail
            quality: Optional EvidenceQualityResult for quality scores

        Returns:
            CandidateGenerationResult with ranked candidates or UNRESOLVED
        """
        # Collect candidates from all sources
        raw_candidates = []

        # 1. Deterministic evidence-based candidate
        det_candidate = self._from_deterministic_evidence(
            intelligence, package, explanation, quality
        )
        if det_candidate:
            raw_candidates.append(det_candidate)

        # 2. ML prediction candidate
        ml_candidate = self._from_ml_prediction(
            intelligence, package, explanation
        )
        if ml_candidate:
            raw_candidates.append(ml_candidate)

        # 3. Historical similarity candidates
        hist_candidates = self._from_historical_cases(
            intelligence, package
        )
        raw_candidates.extend(hist_candidates)

        # Merge duplicates
        merged = self._merge_candidates(raw_candidates)

        # Rank
        ranked = self._rank_candidates(merged, intelligence)

        # If no valid candidates, return UNRESOLVED
        if not ranked:
            return CandidateGenerationResult(
                exception_id=intelligence.exception_id,
                case_id=intelligence.case_id,
                status="UNRESOLVED",
                candidates=[],
                total_candidates=0,
                pipeline_version=PIPELINE_VERSION,
            )

        return CandidateGenerationResult(
            exception_id=intelligence.exception_id,
            case_id=intelligence.case_id,
            status="CANDIDATES_GENERATED",
            candidates=ranked,
            total_candidates=len(ranked),
            pipeline_version=PIPELINE_VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Source 1: Deterministic Evidence
    # ─────────────────────────────────────────────────────────────────────────

    def _from_deterministic_evidence(
        self,
        intel: ExceptionIntelligence,
        package: Optional[EvidencePackage],
        explanation: Optional[ExplanationResult],
        quality: Optional[EvidenceQualityResult],
    ) -> Optional[ResolutionProposal]:
        """Generate candidate from deterministic evidence."""
        det_type = intel.classification.deterministic_type
        resolution = EXCEPTION_TO_RESOLUTION_MAP.get(det_type)
        if not resolution:
            return None

        # Calculate financial adjustment from evidence
        adjustment = self._calculate_adjustment_from_evidence(
            resolution, intel, package, explanation
        )

        # Check evidence compatibility
        compatible = True
        if package and explanation:
            compatible, _ = self.compat_checker.check(resolution, package, explanation)

        # Build evidence list
        evidence_ids = []
        if explanation:
            evidence_ids = explanation.supporting_evidence_ids[:]

        # Calculate confidence from quality
        confidence = 0.5  # Default
        if quality:
            confidence = (quality.consistency_score + quality.coverage_score) / 2
        if intel.evidence:
            confidence = (intel.evidence.consistency_score + intel.evidence.evidence_coverage) / 2

        # Build evidence records
        evidence_records = self._build_evidence_records(package, explanation)

        # Build coverage explanation
        coverage_explanation = self._build_coverage_explanation(
            resolution, adjustment, intel
        )

        # Build structured rationale
        rationale_text, rationale_components = self._build_structured_rationale(
            det_type, resolution, adjustment, evidence_ids, intel, package
        )

        return ResolutionProposal(
            candidate_id=f"CAND-{intel.exception_id}-DET",
            exception_id=intel.exception_id,
            case_id=intel.case_id,
            resolution_type=resolution,
            resolution_description=RESOLUTION_DESCRIPTIONS.get(resolution, resolution),
            financial_adjustment=adjustment,
            supporting_evidence_ids=evidence_ids,
            evidence_records=evidence_records,
            evidence_compatible=compatible,
            evidence_coverage=intel.evidence.evidence_coverage if intel.evidence else 0.0,
            coverage_explanation=coverage_explanation,
            sources=[CandidateSource.DETERMINISTIC_EVIDENCE.value],
            ranking=CandidateRanking(
                rank=0,  # Will be set during ranking
                confidence_score=confidence,
                evidence_support=intel.evidence.evidence_coverage if intel.evidence else 0.0,
            ),
            rationale=rationale_text,
            rationale_components=rationale_components,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Source 2: ML Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def _from_ml_prediction(
        self,
        intel: ExceptionIntelligence,
        package: Optional[EvidencePackage],
        explanation: Optional[ExplanationResult],
    ) -> Optional[ResolutionProposal]:
        """Generate candidate from ML resolution prediction."""
        if not intel.classification.ml_predicted_type:
            return None

        # ML predicts exception type, map to resolution
        ml_exc_type = intel.classification.ml_predicted_type
        ml_resolution = EXCEPTION_TO_RESOLUTION_MAP.get(ml_exc_type)
        if not ml_resolution:
            return None

        # Skip if same as deterministic (will be merged)
        det_resolution = EXCEPTION_TO_RESOLUTION_MAP.get(
            intel.classification.deterministic_type
        )
        if ml_resolution == det_resolution:
            return None

        # Get ML confidence
        ml_confidence = 0.5
        if intel.classification.ml_probabilities:
            ml_confidence = max(intel.classification.ml_probabilities.values())

        # Calculate financial adjustment
        adjustment = self._calculate_adjustment_from_evidence(
            ml_resolution, intel, package, explanation
        )

        # Check evidence compatibility
        compatible = True
        if package and explanation:
            compatible, _ = self.compat_checker.check(ml_resolution, package, explanation)

        evidence_ids = []
        if explanation:
            evidence_ids = explanation.supporting_evidence_ids[:]

        # Build ML support detail
        ml_detail = MLSupportDetail(
            supported=True,
            predicted_resolution=ml_resolution,
            confidence=ml_confidence,
            model_version=intel.classification.ml_model_version,
            probability=intel.classification.ml_probabilities.get(ml_resolution)
            if intel.classification.ml_probabilities
            else None,
        )

        # Build evidence records
        evidence_records = self._build_evidence_records(package, explanation)

        # Build coverage explanation
        coverage_explanation = self._build_coverage_explanation(
            ml_resolution, adjustment, intel
        )

        rationale_text = (
            f"ML model (v{intel.classification.ml_model_version or '?'}) predicts "
            f"{ml_exc_type} with {ml_confidence:.1%} confidence, "
            f"suggesting {ml_resolution}"
        )

        rationale_components = [
            RationaleComponent(
                component_type="ml_support",
                description=f"ML predicts {ml_exc_type} with {ml_confidence:.1%} confidence",
                amount_paise=adjustment.amount_paise if adjustment.amount_paise > 0 else None,
            ),
        ]

        return ResolutionProposal(
            candidate_id=f"CAND-{intel.exception_id}-ML",
            exception_id=intel.exception_id,
            case_id=intel.case_id,
            resolution_type=ml_resolution,
            resolution_description=RESOLUTION_DESCRIPTIONS.get(ml_resolution, ml_resolution),
            financial_adjustment=adjustment,
            supporting_evidence_ids=evidence_ids,
            evidence_records=evidence_records,
            evidence_compatible=compatible,
            evidence_coverage=intel.evidence.evidence_coverage if intel.evidence else 0.0,
            coverage_explanation=coverage_explanation,
            ml_support=ml_detail,
            sources=[CandidateSource.ML_PREDICTION.value],
            ranking=CandidateRanking(
                rank=0,
                confidence_score=ml_confidence,
                evidence_support=intel.evidence.evidence_coverage if intel.evidence else 0.0,
                ml_support=ml_confidence,
            ),
            rationale=rationale_text,
            rationale_components=rationale_components,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Source 3: Historical Similar Cases
    # ─────────────────────────────────────────────────────────────────────────

    def _from_historical_cases(
        self,
        intel: ExceptionIntelligence,
        package: Optional[EvidencePackage],
    ) -> List[ResolutionProposal]:
        """Generate candidates from historical similar cases."""
        candidates = []

        if not intel.similar_cases or not intel.similar_cases.similar_cases:
            return candidates

        # Track seen resolutions to avoid duplicates
        seen = set()
        det_resolution = EXCEPTION_TO_RESOLUTION_MAP.get(
            intel.classification.deterministic_type
        )

        for sc in intel.similar_cases.similar_cases[:3]:
            resolution = sc.get("resolution_type", "")
            similarity = sc.get("similarity_score", 0.0)
            case_id = sc.get("case_id", "")

            # Skip if same as deterministic (will be merged)
            if resolution == det_resolution:
                continue

            # Skip if already seen
            if resolution in seen:
                continue
            seen.add(resolution)

            # Only include if similarity is meaningful
            if similarity < 0.5:
                continue

            # Calculate financial adjustment
            adjustment = self._calculate_adjustment_from_historical(
                resolution, intel, sc
            )

            # Build historical support detail
            hist_detail = HistoricalSupportDetail(
                case_id=case_id,
                similarity_score=similarity,
                historical_resolution=sc.get("resolution_type", ""),
                historical_outcome=sc.get("resolution_outcome"),
                payment_amount=sc.get("payment_amount"),
                difference=sc.get("difference"),
            )

            rationale_text = (
                f"Similar historical case {case_id} (similarity: {similarity:.3f}) "
                f"was resolved with {resolution}"
            )

            rationale_components = [
                RationaleComponent(
                    component_type="historical_support",
                    description=f"Historical case {case_id} with similarity {similarity:.3f}",
                ),
            ]

            candidates.append(
                ResolutionProposal(
                    candidate_id=f"CAND-{intel.exception_id}-HIST-{case_id}",
                    exception_id=intel.exception_id,
                    case_id=intel.case_id,
                    resolution_type=resolution,
                    resolution_description=RESOLUTION_DESCRIPTIONS.get(
                        resolution, resolution
                    ),
                    financial_adjustment=adjustment,
                    supporting_evidence_ids=[],
                    evidence_records=[],
                    evidence_compatible=True,
                    evidence_coverage=0.0,
                    coverage_explanation=f"Based on historical pattern from case {case_id}",
                    historical_support=[hist_detail],
                    sources=[CandidateSource.HISTORICAL_CASE.value],
                    ranking=CandidateRanking(
                        rank=0,
                        confidence_score=similarity,
                        evidence_support=0.0,
                        historical_support=similarity,
                    ),
                    rationale=rationale_text,
                    rationale_components=rationale_components,
                )
            )

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Financial Adjustment Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_adjustment_from_evidence(
        self,
        resolution: str,
        intel: ExceptionIntelligence,
        package: Optional[EvidencePackage],
        explanation: Optional[ExplanationResult],
    ) -> FinancialAdjustment:
        """Calculate financial adjustment from actual evidence.

        NEVER invents amounts. Always traces back to financial records.
        """
        difference = intel.difference

        if resolution == "NO_ACTION":
            return FinancialAdjustment(
                adjustment_type="NO_ADJUSTMENT",
                amount_paise=0,
                direction="NONE",
                calculation_basis="zero_discrepancy",
            )

        if resolution == "FEE_ADJUSTMENT" and package:
            # Use actual fee amounts from evidence
            total_fees = sum(f.amount for f in package.fees)
            if total_fees > 0:
                return FinancialAdjustment(
                    adjustment_type="FEE_CORRECTION",
                    amount_paise=total_fees,
                    direction="CREDIT",
                    evidence_record_id=package.fees[0].record_id if package.fees else None,
                    calculation_basis="fee_record_sum",
                )

        if resolution == "REFUND_ADJUSTMENT" and package:
            total_refunds = sum(r.amount for r in package.refunds)
            if total_refunds > 0:
                return FinancialAdjustment(
                    adjustment_type="REFUND_CORRECTION",
                    amount_paise=total_refunds,
                    direction="DEBIT",
                    evidence_record_id=package.refunds[0].record_id if package.refunds else None,
                    calculation_basis="refund_record_sum",
                )

        if resolution == "TAX_ADJUSTMENT" and package:
            total_taxes = sum(t.amount for t in package.taxes)
            if total_taxes > 0:
                return FinancialAdjustment(
                    adjustment_type="TAX_CORRECTION",
                    amount_paise=total_taxes,
                    direction="CREDIT",
                    evidence_record_id=package.taxes[0].record_id if package.taxes else None,
                    calculation_basis="tax_record_sum",
                )

        if resolution == "PARTIAL_SETTLEMENT_RECONCILIATION":
            return FinancialAdjustment(
                adjustment_type="SETTLEMENT_ADJUSTMENT",
                amount_paise=abs(difference),
                direction="CREDIT" if difference > 0 else "DEBIT",
                calculation_basis="discrepancy_amount",
            )

        if resolution == "DUPLICATE_SETTLEMENT" and package:
            if len(package.settlements) >= 2:
                amounts = [s.amount for s in package.settlements]
                return FinancialAdjustment(
                    adjustment_type="SETTLEMENT_ADJUSTMENT",
                    amount_paise=max(amounts),
                    direction="DEBIT",
                    evidence_record_id=package.settlements[0].record_id,
                    calculation_basis="duplicate_settlement_max",
                )

        if resolution == "MULTI_ADJUSTMENT":
            return FinancialAdjustment(
                adjustment_type="SETTLEMENT_ADJUSTMENT",
                amount_paise=abs(difference),
                direction="CREDIT" if difference > 0 else "DEBIT",
                calculation_basis="discrepancy_amount_multi_component",
            )

        # Default: use discrepancy as the adjustment basis
        return FinancialAdjustment(
            adjustment_type="SETTLEMENT_ADJUSTMENT",
            amount_paise=abs(difference),
            direction="CREDIT" if difference > 0 else "DEBIT",
            calculation_basis="discrepancy_amount",
        )

    def _calculate_adjustment_from_historical(
        self,
        resolution: str,
        intel: ExceptionIntelligence,
        similar_case: Dict,
    ) -> FinancialAdjustment:
        """Calculate financial adjustment from historical case pattern."""
        # Use the discrepancy from the current exception
        difference = intel.difference

        return FinancialAdjustment(
            adjustment_type="SETTLEMENT_ADJUSTMENT",
            amount_paise=abs(difference),
            direction="CREDIT" if difference > 0 else "DEBIT",
            calculation_basis="historical_pattern_discrepancy",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate Merging
    # ─────────────────────────────────────────────────────────────────────────

    def _merge_candidates(
        self, candidates: List[ResolutionProposal]
    ) -> List[ResolutionProposal]:
        """Merge candidates with the same resolution type.

        Combines sources and supporting evidence.
        """
        by_resolution: Dict[str, List[ResolutionProposal]] = {}
        for c in candidates:
            key = c.resolution_type
            if key not in by_resolution:
                by_resolution[key] = []
            by_resolution[key].append(c)

        merged = []
        for resolution_type, group in by_resolution.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_group(group))

        return merged

    def _merge_group(self, group: List[ResolutionProposal]) -> ResolutionProposal:
        """Merge a group of candidates with the same resolution type."""
        # Use the best candidate as base
        best = max(group, key=lambda c: c.ranking.confidence_score)

        # Combine all sources
        all_sources = []
        all_evidence_ids = set()
        all_evidence_records = []
        all_historical = []
        ml_detail = None
        for c in group:
            for s in c.sources:
                if s not in all_sources:
                    all_sources.append(s)
            all_evidence_ids.update(c.supporting_evidence_ids)
            all_evidence_records.extend(c.evidence_records)
            all_historical.extend(c.historical_support)
            if c.ml_support and c.ml_support.supported:
                ml_detail = c.ml_support

        # Combine confidence signals
        ml_supports = [c.ranking.ml_support for c in group if c.ranking.ml_support]
        hist_supports = [
            c.ranking.historical_support
            for c in group
            if c.ranking.historical_support
        ]

        combined_confidence = max(c.ranking.confidence_score for c in group)

        rationale_parts = [c.rationale for c in group]
        combined_rationale = " | ".join(rationale_parts)

        # Combine rationale components
        all_components = []
        for c in group:
            all_components.extend(c.rationale_components)

        return ResolutionProposal(
            candidate_id=best.candidate_id,
            exception_id=best.exception_id,
            case_id=best.case_id,
            resolution_type=best.resolution_type,
            resolution_description=best.resolution_description,
            financial_adjustment=best.financial_adjustment,
            supporting_evidence_ids=list(all_evidence_ids),
            evidence_records=all_evidence_records,
            evidence_compatible=any(c.evidence_compatible for c in group),
            evidence_coverage=max(c.evidence_coverage for c in group),
            coverage_explanation=best.coverage_explanation,
            ml_support=ml_detail,
            historical_support=all_historical,
            sources=all_sources,
            ranking=CandidateRanking(
                rank=0,
                confidence_score=combined_confidence,
                evidence_support=max(c.ranking.evidence_support for c in group),
                ml_support=max(ml_supports) if ml_supports else None,
                historical_support=max(hist_supports) if hist_supports else None,
            ),
            rationale=combined_rationale,
            rationale_components=all_components,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ranking
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_candidates(
        self,
        candidates: List[ResolutionProposal],
        intel: ExceptionIntelligence,
    ) -> List[ResolutionProposal]:
        """Rank candidates by confidence and evidence support."""
        # Score each candidate
        for c in candidates:
            # Base score from confidence
            score = c.ranking.confidence_score

            # Bonus for evidence compatibility
            if c.evidence_compatible:
                score += 0.1

            # Bonus for multiple sources
            if len(c.sources) >= 2:
                score += 0.05

            # Penalty for no evidence
            if not c.supporting_evidence_ids:
                score -= 0.1

            # Store score for sorting (not part of the schema)
            c._sort_score = score

        # Sort by score descending
        candidates.sort(key=lambda c: c._sort_score, reverse=True)

        # Assign ranks
        for i, c in enumerate(candidates):
            c.ranking.rank = i + 1
            del c._sort_score  # Clean up temporary field

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Evidence Records
    # ─────────────────────────────────────────────────────────────────────────

    def _build_evidence_records(
        self,
        package: Optional[EvidencePackage],
        explanation: Optional[ExplanationResult],
    ) -> List[EvidenceRecordRef]:
        """Build detailed evidence record references from package."""
        records = []
        if not package:
            return records

        # Payment
        if package.payment:
            records.append(
                EvidenceRecordRef(
                    record_id=package.payment.record_id,
                    entity_type="PAYMENT",
                    amount=package.payment.amount,
                    relationship="PRIMARY_RECORD",
                )
            )

        # Settlements
        for s in package.settlements:
            records.append(
                EvidenceRecordRef(
                    record_id=s.record_id,
                    entity_type="SETTLEMENT",
                    amount=s.amount,
                    relationship=s.relationship,
                )
            )

        # Refunds
        for r in package.refunds:
            records.append(
                EvidenceRecordRef(
                    record_id=r.record_id,
                    entity_type="REFUND",
                    amount=r.amount,
                    relationship=r.relationship,
                    contribution=-r.amount,
                )
            )

        # Fees
        for f in package.fees:
            records.append(
                EvidenceRecordRef(
                    record_id=f.record_id,
                    entity_type="FEE",
                    amount=f.amount,
                    relationship=f.relationship,
                    contribution=-f.amount,
                )
            )

        # Taxes
        for t in package.taxes:
            records.append(
                EvidenceRecordRef(
                    record_id=t.record_id,
                    entity_type="TAX",
                    amount=t.amount,
                    relationship=t.relationship,
                    contribution=-t.amount,
                )
            )

        # Adjustments
        for a in package.adjustments:
            records.append(
                EvidenceRecordRef(
                    record_id=a.record_id,
                    entity_type="ADJUSTMENT",
                    amount=a.amount,
                    relationship=a.relationship,
                    contribution=a.amount,
                )
            )

        return records

    # ─────────────────────────────────────────────────────────────────────────
    # Coverage Explanation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_coverage_explanation(
        self,
        resolution: str,
        adjustment: FinancialAdjustment,
        intel: ExceptionIntelligence,
    ) -> str:
        """Build explanation of how the candidate covers the discrepancy."""
        if adjustment.amount_paise == 0:
            return "No financial adjustment needed — amounts match."

        parts = [
            f"The {resolution} candidate proposes a {adjustment.direction} of "
            f"{adjustment.amount_paise} paise to reconcile the {intel.difference} paise discrepancy.",
            f"Basis: {adjustment.calculation_basis}.",
        ]

        if adjustment.evidence_record_id:
            parts.append(
                f"Amount traced to record {adjustment.evidence_record_id}."
            )

        coverage_pct = (adjustment.amount_paise / abs(intel.difference) * 100) if intel.difference != 0 else 100
        parts.append(f"Coverage: {coverage_pct:.0f}% of the discrepancy.")

        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Structured Rationale
    # ─────────────────────────────────────────────────────────────────────────

    def _build_structured_rationale(
        self,
        exception_type: str,
        resolution: str,
        adjustment: FinancialAdjustment,
        evidence_ids: List[str],
        intel: ExceptionIntelligence,
        package: Optional[EvidencePackage],
    ) -> tuple:
        """Build structured rationale with components.

        Returns:
            (rationale_text, rationale_components)
        """
        components = []

        # 1. What happened
        what_happened = f"Exception type: {exception_type}. Discrepancy: {intel.difference} paise."
        components.append(
            RationaleComponent(
                component_type="what_happened",
                description=what_happened,
            )
        )

        # 2. Evidence support
        if evidence_ids:
            evidence_desc = f"Supporting evidence: {', '.join(evidence_ids[:5])}"
            components.append(
                RationaleComponent(
                    component_type="evidence_support",
                    description=evidence_desc,
                    evidence_ids=evidence_ids,
                )
            )

        # 3. Financial trace
        if adjustment.amount_paise > 0:
            trace_desc = (
                f"{adjustment.amount_paise} paise {adjustment.direction} "
                f"based on {adjustment.calculation_basis}"
            )
            components.append(
                RationaleComponent(
                    component_type="financial_trace",
                    description=trace_desc,
                    evidence_ids=[adjustment.evidence_record_id] if adjustment.evidence_record_id else [],
                    amount_paise=adjustment.amount_paise,
                )
            )

        # 4. Evidence quality
        if intel.evidence:
            quality_desc = (
                f"Evidence coverage: {intel.evidence.evidence_coverage:.1%}, "
                f"consistency: {intel.evidence.consistency_score:.1%}"
            )
            components.append(
                RationaleComponent(
                    component_type="evidence_support",
                    description=quality_desc,
                )
            )

        # 5. Recommendation
        components.append(
            RationaleComponent(
                component_type="recommendation",
                description=f"Proposed resolution: {resolution}",
            )
        )

        # Build text
        parts = [
            f"Exception type: {exception_type}",
            f"Discrepancy: {intel.difference} paise",
        ]

        if adjustment.amount_paise > 0:
            parts.append(
                f"Proposed adjustment: {adjustment.amount_paise} paise "
                f"({adjustment.direction}, basis: {adjustment.calculation_basis})"
            )

        if evidence_ids:
            parts.append(f"Supporting evidence: {', '.join(evidence_ids[:3])}")

        if intel.evidence:
            parts.append(
                f"Evidence coverage: {intel.evidence.evidence_coverage:.1%}, "
                f"consistency: {intel.evidence.consistency_score:.1%}"
            )

        parts.append(f"Resolution: {resolution}")

        return ". ".join(parts), components
