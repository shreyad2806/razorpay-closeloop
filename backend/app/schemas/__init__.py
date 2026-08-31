"""
Razorpay CloseLoop financial data schemas.

Centralized data contracts for synthetic financial data generation and reconciliation.
"""

from app.schemas.case import Case, GroundTruth
from app.schemas.config import GeneratorConfig, ScenarioDistribution
from app.schemas.enums import (
    AdjustmentType,
    Currency,
    ExceptionType,
    FeeType,
    MatchStatus,
    MissingRecordSubtype,
    PaymentStatus,
    ReconciliationStatus,
    RefundStatus,
    ResolutionType,
    RiskCategory,
    SettlementStatus,
    TaxType,
)
from app.schemas.financial import (
    Adjustment,
    Fee,
    Merchant,
    Payment,
    Refund,
    Settlement,
    Tax,
)
from app.schemas.reconciliation import (
    CalculationBreakdown,
    MatchingRule,
    ReconciliationResult,
)
from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
    StructuralConflict,
)
from app.schemas.explanation import (
    CandidateExplanation,
    ExplainingEvent,
    ExplanationResult,
    ExplanationStatus,
)
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel

__all__ = [
    # Enums
    "ExceptionType",
    "ResolutionType",
    "RiskCategory",
    "MatchStatus",
    "ReconciliationStatus",
    "MissingRecordSubtype",
    "PaymentStatus",
    "SettlementStatus",
    "RefundStatus",
    "FeeType",
    "TaxType",
    "AdjustmentType",
    "Currency",
    # Financial entities
    "Merchant",
    "Payment",
    "Settlement",
    "Refund",
    "Fee",
    "Tax",
    "Adjustment",
    # Case and ground truth
    "Case",
    "GroundTruth",
    # Reconciliation
    "ReconciliationResult",
    "CalculationBreakdown",
    "MatchingRule",
    # Config
    "GeneratorConfig",
    "ScenarioDistribution",
    # Evidence
    "EvidencePackage",
    "EvidenceRecord",
    "MissingEvidence",
    "StructuralConflict",
    # Explanation
    "ExplanationResult",
    "ExplanationStatus",
    "CandidateExplanation",
    "ExplainingEvent",
    # Evidence Quality
    "EvidenceQualityResult",
    "NoveltyLevel",
]
