"""
ML feature definitions for Razorpay CloseLoop Phase 4.

Defines the versioned feature schema and feature extraction logic.
All features are deterministic — derived from financial records and
deterministic evidence analysis.

Feature categories:
- Financial: amount-based features
- Structural: record-count features
- Evidence: explanation engine output features
- Temporal: timing features
- Merchant: merchant-level aggregates (placeholder)
- Historical: pattern features (placeholder)
"""

from typing import List

from app.schemas.ml_dataset import (
    FEATURE_SCHEMA_VERSION,
    FeatureCategory,
    FeatureDefinition,
    FeatureSchema,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Definitions
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DEFINITIONS: List[FeatureDefinition] = [
    # ── Financial Features ──────────────────────────────────────────────────
    FeatureDefinition(
        name="difference_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Absolute difference between expected and actual settlement in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="relative_difference",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="difference / payment_amount (0.0 if payment is zero)",
        min_value=-1.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="payment_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Original payment amount in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="settlement_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Total settlement amount in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="refund_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Total refund amount in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="fee_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Total fee amount in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="tax_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Total tax amount in paise",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="adjustment_amount",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="Net adjustment amount in paise (positive=credit, negative=debit)",
    ),
    FeatureDefinition(
        name="refund_ratio",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="refund_amount / payment_amount (0.0 if payment is zero)",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="fee_ratio",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="fee_amount / payment_amount (0.0 if payment is zero)",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="tax_ratio",
        category=FeatureCategory.FINANCIAL,
        dtype="float",
        description="tax_amount / payment_amount (0.0 if payment is zero)",
        min_value=0.0,
        max_value=1.0,
    ),

    # ── Structural Features ─────────────────────────────────────────────────
    FeatureDefinition(
        name="num_settlements",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Number of settlement records",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="num_refunds",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Number of refund records",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="num_fees",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Number of fee records",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="num_taxes",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Number of tax records",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="num_adjustments",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Number of adjustment records",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="has_missing_evidence",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="1.0 if any evidence is missing, 0.0 otherwise",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="num_missing_evidence",
        category=FeatureCategory.STRUCTURAL,
        dtype="float",
        description="Count of missing evidence types",
        min_value=0.0,
    ),

    # ── Evidence Features ───────────────────────────────────────────────────
    FeatureDefinition(
        name="evidence_coverage",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="Proportion of discrepancy explained by evidence (0.0-1.0)",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="consistency_score",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="Evidence consistency score (0.0-1.0)",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="fully_explained",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="1.0 if discrepancy is fully explained, 0.0 otherwise",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="partially_explained",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="1.0 if discrepancy is partially explained, 0.0 otherwise",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="has_conflict",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="1.0 if conflicting explanations exist, 0.0 otherwise",
        min_value=0.0,
        max_value=1.0,
    ),
    FeatureDefinition(
        name="supporting_evidence_count",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="Number of evidence records supporting the explanation",
        min_value=0.0,
    ),
    FeatureDefinition(
        name="num_candidate_explanations",
        category=FeatureCategory.EVIDENCE,
        dtype="float",
        description="Number of candidate explanation combinations found",
        min_value=0.0,
    ),

    # ── Temporal Features (placeholder — timestamps not yet in all records) ─
    FeatureDefinition(
        name="payment_to_settlement_delay_hours",
        category=FeatureCategory.TEMPORAL,
        dtype="float",
        description="Hours between payment and settlement (0.0 if unavailable)",
        min_value=0.0,
        default_value=0.0,
    ),

    # ── Merchant Features (placeholder — merchant aggregates not yet computed) ─
    FeatureDefinition(
        name="merchant_historical_exception_rate",
        category=FeatureCategory.MERCHANT,
        dtype="float",
        description="Historical exception rate for this merchant (0.0 if unavailable)",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
    ),

    # ── Historical Features (placeholder — similarity not yet implemented) ──
    FeatureDefinition(
        name="historical_pattern_match_score",
        category=FeatureCategory.HISTORICAL,
        dtype="float",
        description="Similarity to known historical patterns (0.0 if unavailable)",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Schema
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATURE_SCHEMA = FeatureSchema(
    version=FEATURE_SCHEMA_VERSION,
    features=FEATURE_DEFINITIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_features(
    difference: int,
    payment_amount: int,
    settlement_amount: int,
    refund_amount: int,
    fee_amount: int,
    tax_amount: int,
    adjustment_amount: int,
    num_settlements: int,
    num_refunds: int,
    num_fees: int,
    num_taxes: int,
    num_adjustments: int,
    has_missing_evidence: bool,
    num_missing_evidence: int,
    evidence_coverage: float,
    consistency_score: float,
    fully_explained: bool,
    partially_explained: bool,
    has_conflict: bool,
    supporting_evidence_count: int,
    num_candidate_explanations: int,
    payment_to_settlement_delay_hours: float = 0.0,
    merchant_historical_exception_rate: float = 0.0,
    historical_pattern_match_score: float = 0.0,
) -> dict:
    """
    Extract a deterministic feature vector from financial and evidence data.

    All inputs must be derived from deterministic systems (reconciliation, evidence retrieval,
    explanation engine, quality scorer). No ground truth is used.

    Returns:
        Dict mapping feature names to float values.
    """
    # Safe division helper
    def safe_ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    return {
        # Financial
        "difference_amount": float(abs(difference)),
        "relative_difference": safe_ratio(difference, payment_amount),
        "payment_amount": float(payment_amount),
        "settlement_amount": float(settlement_amount),
        "refund_amount": float(refund_amount),
        "fee_amount": float(fee_amount),
        "tax_amount": float(tax_amount),
        "adjustment_amount": float(adjustment_amount),
        "refund_ratio": safe_ratio(refund_amount, payment_amount),
        "fee_ratio": safe_ratio(fee_amount, payment_amount),
        "tax_ratio": safe_ratio(tax_amount, payment_amount),
        # Structural
        "num_settlements": float(num_settlements),
        "num_refunds": float(num_refunds),
        "num_fees": float(num_fees),
        "num_taxes": float(num_taxes),
        "num_adjustments": float(num_adjustments),
        "has_missing_evidence": 1.0 if has_missing_evidence else 0.0,
        "num_missing_evidence": float(num_missing_evidence),
        # Evidence
        "evidence_coverage": evidence_coverage,
        "consistency_score": consistency_score,
        "fully_explained": 1.0 if fully_explained else 0.0,
        "partially_explained": 1.0 if partially_explained else 0.0,
        "has_conflict": 1.0 if has_conflict else 0.0,
        "supporting_evidence_count": float(supporting_evidence_count),
        "num_candidate_explanations": float(num_candidate_explanations),
        # Temporal
        "payment_to_settlement_delay_hours": payment_to_settlement_delay_hours,
        # Merchant
        "merchant_historical_exception_rate": merchant_historical_exception_rate,
        # Historical
        "historical_pattern_match_score": historical_pattern_match_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_features(features: dict) -> List[str]:
    """
    Validate a feature dict against the schema.

    Returns list of validation errors (empty = valid).
    """
    errors = []
    schema_names = set(ML_FEATURE_SCHEMA.get_feature_names())

    # Check for missing features
    for name in schema_names:
        if name not in features:
            errors.append(f"Missing feature: {name}")

    # Check for unexpected features
    for name in features:
        if name not in schema_names:
            errors.append(f"Unexpected feature: {name}")

    # Check for non-numeric values
    for name, value in features.items():
        if not isinstance(value, (int, float)):
            errors.append(f"Non-numeric value for {name}: {type(value).__name__}")

    return errors
