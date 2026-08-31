"""
Deterministic feature engineering for Razorpay CloseLoop Phase 4.

Converts reconciliation results, evidence packages, explanation results,
and quality scores into a numeric feature vector for ML classification.

All features are derived from information available at inference time.
No ground truth is used.

Numeric safety:
- Zero denominators produce 0.0
- No NaN/inf in output
- Extreme values are clipped
- Missing timestamps produce 0.0 delay
"""

from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult
from app.schemas.evidence_quality import EvidenceQualityResult
from app.schemas.ml_dataset import FeatureVector, FEATURE_SCHEMA_VERSION
from app.ml.features import ML_FEATURE_SCHEMA, extract_features, validate_features


class FeatureEngineer:
    """
    Deterministic feature engineering service.

    Takes Phase 3 outputs (evidence, explanation, quality) and produces
    a numeric feature vector suitable for ML classification.

    Features are traceable to observable data at inference time.
    No ground truth labels are used.
    """

    def __init__(self):
        self._schema = ML_FEATURE_SCHEMA
        self._feature_names = self._schema.get_feature_names()

    def engineer(
        self,
        package: EvidencePackage,
        explanation: ExplanationResult,
        quality: EvidenceQualityResult,
        payment_timestamp: Optional[datetime] = None,
        settlement_timestamp: Optional[datetime] = None,
        refund_timestamps: Optional[List[datetime]] = None,
        merchant_exception_rate: float = 0.0,
    ) -> FeatureVector:
        """
        Engineer a feature vector from deterministic outputs.

        Args:
            package: EvidencePackage from evidence retrieval
            explanation: ExplanationResult from explanation engine
            quality: EvidenceQualityResult from quality scorer
            payment_timestamp: When the payment was captured (for temporal features)
            settlement_timestamp: When settlement was processed (for temporal features)
            refund_timestamps: When refunds were processed (for temporal features)
            merchant_exception_rate: Historical exception rate for this merchant

        Returns:
            FeatureVector with all 29 features
        """
        # 1. Financial features
        financial = self._extract_financial(package)

        # 2. Structural features
        structural = self._extract_structural(package)

        # 3. Evidence features
        evidence = self._extract_evidence(explanation, quality)

        # 4. Temporal features
        temporal = self._extract_temporal(
            payment_timestamp, settlement_timestamp, refund_timestamps
        )

        # 5. Merchant features
        merchant = self._extract_merchant(merchant_exception_rate)

        # 6. Historical features (placeholder)
        historical = self._extract_historical()

        # Merge all features
        all_features = {}
        all_features.update(financial)
        all_features.update(structural)
        all_features.update(evidence)
        all_features.update(temporal)
        all_features.update(merchant)
        all_features.update(historical)

        # Validate against schema
        errors = validate_features(all_features)
        if errors:
            raise ValueError(f"Feature validation failed: {errors}")

        return FeatureVector(
            features=all_features,
            schema_version=FEATURE_SCHEMA_VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Financial Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_financial(self, package: EvidencePackage) -> Dict[str, float]:
        """
        Extract financial features from evidence package.

        Features:
        - difference_amount: |expected - actual|
        - relative_difference: difference / payment_amount
        - payment_amount: original payment
        - settlement_amount: total settlement
        - refund_amount: total refunds
        - fee_amount: total fees
        - tax_amount: total taxes
        - adjustment_amount: net adjustments
        - refund_ratio: refund / payment
        - fee_ratio: fee / payment
        - tax_ratio: tax / payment
        """
        payment_amount = self._get_payment_amount(package)
        difference = package.difference

        return {
            "difference_amount": float(abs(difference)),
            "relative_difference": self._safe_ratio(difference, payment_amount),
            "payment_amount": float(payment_amount),
            "settlement_amount": float(package.total_settlement_amount),
            "refund_amount": float(package.total_refund_amount),
            "fee_amount": float(package.total_fee_amount),
            "tax_amount": float(package.total_tax_amount),
            "adjustment_amount": float(package.total_adjustment_amount),
            "refund_ratio": self._safe_ratio(package.total_refund_amount, payment_amount),
            "fee_ratio": self._safe_ratio(package.total_fee_amount, payment_amount),
            "tax_ratio": self._safe_ratio(package.total_tax_amount, payment_amount),
        }

    def _get_payment_amount(self, package: EvidencePackage) -> int:
        """Get payment amount from package, handling missing payment record."""
        if package.payment is not None:
            return package.payment.amount
        # Fallback: reconstruct from expected + fees + taxes + refunds - adjustments
        return (
            package.expected_amount
            + package.total_refund_amount
            + package.total_fee_amount
            + package.total_tax_amount
            - package.total_adjustment_amount
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_structural(self, package: EvidencePackage) -> Dict[str, float]:
        """
        Extract structural features from evidence package.

        Features:
        - num_settlements, num_refunds, num_fees, num_taxes, num_adjustments
        - has_missing_evidence: 1.0 if any evidence is missing
        - num_missing_evidence: count of missing evidence types
        """
        num_missing = len(package.missing_evidence)

        return {
            "num_settlements": float(len(package.settlements)),
            "num_refunds": float(len(package.refunds)),
            "num_fees": float(len(package.fees)),
            "num_taxes": float(len(package.taxes)),
            "num_adjustments": float(len(package.adjustments)),
            "has_missing_evidence": 1.0 if num_missing > 0 else 0.0,
            "num_missing_evidence": float(num_missing),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Evidence Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_evidence(
        self, explanation: ExplanationResult, quality: EvidenceQualityResult
    ) -> Dict[str, float]:
        """
        Extract evidence features from explanation and quality results.

        Features:
        - evidence_coverage: coverage score from quality scorer
        - consistency_score: consistency score from quality scorer
        - fully_explained: 1.0 if fully explained
        - partially_explained: 1.0 if partially explained
        - has_conflict: 1.0 if conflicting explanations exist
        - supporting_evidence_count: number of supporting evidence records
        - num_candidate_explanations: number of candidate combinations
        """
        return {
            "evidence_coverage": quality.coverage_score,
            "consistency_score": quality.consistency_score,
            "fully_explained": 1.0 if explanation.is_fully_explained() else 0.0,
            "partially_explained": 1.0 if (
                explanation.explanation_status.value == "PARTIALLY_EXPLAINED"
            ) else 0.0,
            "has_conflict": 1.0 if explanation.conflict else 0.0,
            "supporting_evidence_count": float(len(explanation.supporting_evidence_ids)),
            "num_candidate_explanations": float(len(explanation.candidate_explanations)),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_temporal(
        self,
        payment_timestamp: Optional[datetime],
        settlement_timestamp: Optional[datetime],
        refund_timestamps: Optional[List[datetime]],
    ) -> Dict[str, float]:
        """
        Extract temporal features from timestamps.

        Features:
        - payment_to_settlement_delay_hours: hours between payment and settlement

        Units: hours (float)
        Default: 0.0 when timestamps unavailable
        """
        delay_hours = 0.0

        if payment_timestamp and settlement_timestamp:
            delta = settlement_timestamp - payment_timestamp
            delay_hours = delta.total_seconds() / 3600.0
            # Clip extreme values (negative delays or > 365 days)
            delay_hours = max(0.0, min(delay_hours, 8760.0))

        return {
            "payment_to_settlement_delay_hours": delay_hours,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Merchant Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_merchant(self, merchant_exception_rate: float) -> Dict[str, float]:
        """
        Extract merchant-level features.

        Features:
        - merchant_historical_exception_rate: historical exception rate for this merchant

        Only uses information available before the current case.
        """
        # Clamp to [0.0, 1.0]
        rate = max(0.0, min(1.0, merchant_exception_rate))
        return {
            "merchant_historical_exception_rate": rate,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Historical Features
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_historical(self) -> Dict[str, float]:
        """
        Extract historical pattern features.

        Features:
        - historical_pattern_match_score: similarity to known patterns

        At this stage, returns default (0.0).
        True semantic similarity will be added when embeddings are implemented.
        """
        return {
            "historical_pattern_match_score": 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """
        Safe division producing 0.0 for zero denominators.

        No NaN, no inf, no exceptions.
        """
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)
