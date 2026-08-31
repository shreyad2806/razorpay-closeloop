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
from app.schemas.ml_dataset import (
    DatasetManifest,
    DatasetSplit,
    FeatureDefinition,
    FeatureSchema,
    FeatureVector,
    LEAKED_FIELDS,
    MLLabels,
    MLSample,
    SplitType,
)
from app.schemas.historical_case import (
    FinancialContext,
    HistoricalCase,
    HistoricalEvidenceRef,
    ResolutionOrigin,
    ResolutionOutcome,
)
from app.schemas.similarity import (
    SimilarCase,
    SimilaritySearchResult,
)
from app.schemas.intelligence import (
    ClassificationResult,
    ExceptionIntelligence,
    EvidenceIntelligence,
    RecommendationStatus,
    ResolutionCandidate,
    SimilarCasesIntelligence,
)
from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    CandidateRanking,
    CandidateSource,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.schemas.candidate_scoring import CandidateScore, ScoringConfig
from app.schemas.resolution_selection import (
    ExplainabilityDetail,
    ExplainabilityLevel,
    SelectionConfig,
    SelectionResult,
    SelectionStatus,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.confidence_gate import (
    ConfidenceGateConfig,
    ConfidenceGateResult,
    GateAction,
    GateCheck,
    RiskOverrideLevel,
)
from app.schemas.exposure_guard import (
    ExposureAction,
    ExposureBlockReason,
    ExposureCheck,
    ExposureGuardConfig,
    ExposureGuardResult,
)
from app.schemas.evidence_guard import (
    EvidenceAction,
    EvidenceBlockReason,
    EvidenceGuardCheck,
    EvidenceGuardConfig,
    EvidenceGuardResult,
)
from app.schemas.failure_fallback import (
    DependencyFailure,
    DependencyPolicy,
    ErrorCategory,
    FailureFallbackResult,
    FailureSeverity,
    FallbackAction,
)
from app.schemas.decision_matrix import (
    AutomationDecision,
    AutomationDecisionResult,
    DecisionConfig,
    GateResult,
    GateStatus,
    ReasonCode,
)
from app.schemas.guardrail_engine import GuardrailEngineResult
from app.schemas.action_request import (
    ActionRequest,
    ActionRequestResult,
    ActionStatus,
    AuthorizationSource,
)
from app.schemas.outcome import (
    HistoricalLearningRecord,
    RewardSignal,
    RewardType,
    WorkflowOutcome,
    WorkflowOutcomeRecord,
)
from app.schemas.verification import (
    CheckStatus,
    VerificationAction,
    VerificationCheck,
    VerificationConfig,
    VerificationResult,
)
from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    HumanReviewState,
    NodeStatus,
    RewardState,
    RewardStatus,
    VerificationState,
    VerificationStatus,
    WorkflowMetadata,
    WorkflowStatus,
)

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
    # ML Dataset
    "MLSample",
    "MLLabels",
    "FeatureVector",
    "FeatureSchema",
    "FeatureDefinition",
    "DatasetSplit",
    "DatasetManifest",
    "SplitType",
    "LEAKED_FIELDS",
    # Historical Case
    "HistoricalCase",
    "HistoricalEvidenceRef",
    "FinancialContext",
    "ResolutionOutcome",
    "ResolutionOrigin",
    # Similarity
    "SimilarCase",
    "SimilaritySearchResult",
    # Intelligence
    "ExceptionIntelligence",
    "ClassificationResult",
    "EvidenceIntelligence",
    "SimilarCasesIntelligence",
    "ResolutionCandidate",
    "RecommendationStatus",
    # Resolution Candidates
    "ResolutionProposal",
    "FinancialAdjustment",
    "CandidateRanking",
    "CandidateSource",
    "CandidateGenerationResult",
    # Candidate Scoring
    "CandidateScore",
    "ScoringConfig",
    # Resolution Selection
    "SelectionResult",
    "SelectionStatus",
    "SelectionConfig",
    "ExplainabilityDetail",
    "ExplainabilityLevel",
    # Resolution Engine
    "ResolutionEngineResult",
    # Confidence Gate
    "ConfidenceGateConfig",
    "ConfidenceGateResult",
    "GateAction",
    "GateCheck",
    "RiskOverrideLevel",
    # Exposure Guard
    "ExposureGuardConfig",
    "ExposureGuardResult",
    "ExposureAction",
    "ExposureBlockReason",
    "ExposureCheck",
    # Evidence Guard
    "EvidenceGuardConfig",
    "EvidenceGuardResult",
    "EvidenceAction",
    "EvidenceBlockReason",
    "EvidenceGuardCheck",
    # Failure Fallback
    "DependencyPolicy",
    "DependencyFailure",
    "ErrorCategory",
    "FailureSeverity",
    "FallbackAction",
    "FailureFallbackResult",
    # Decision Matrix
    "AutomationDecision",
    "AutomationDecisionResult",
    "DecisionConfig",
    "GateResult",
    "GateStatus",
    "ReasonCode",
    # Guardrail Engine
    "GuardrailEngineResult",
    # Action Request
    "ActionRequest",
    "ActionRequestResult",
    "ActionStatus",
    "AuthorizationSource",
    # Outcome
    "WorkflowOutcome",
    "WorkflowOutcomeRecord",
    "RewardSignal",
    "RewardType",
    "HistoricalLearningRecord",
    # Verification
    "VerificationAction",
    "VerificationCheck",
    "VerificationConfig",
    "VerificationResult",
    "CheckStatus",
    # Agent State
    "AgentState",
    "WorkflowMetadata",
    "WorkflowStatus",
    "NodeStatus",
    "HumanReviewState",
    "HumanApprovalStatus",
    "VerificationState",
    "VerificationStatus",
    "RewardState",
    "RewardStatus",
]
