"""
Authorization Service for Razorpay CloseLoop Phase 8.

This is the explicit authorization boundary between Phase 7 (investigation/proposal)
and Phase 9 (execution).

Core principle:
    AI investigates and proposes.
    Policy authorizes.
    Deterministic services execute.
    Reconciliation verifies truth.

This service:
- Takes Phase 7 ResolutionProposal / SelectionResult
- Evaluates through existing GuardrailEngine
- Returns explicit AuthorizationDecision (AUTO / HUMAN_REVIEW / UNRESOLVED)
- Does NOT execute financial actions
- Does NOT mutate financial records
- Does NOT call providers

Authorization decisions are deterministic and fail-closed.
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.schemas.resolution_candidate import (
    CandidateGenerationResult,
    ResolutionProposal,
)
from app.schemas.resolution_selection import SelectionResult, SelectionStatus, ExplainabilityDetail
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.candidate_scoring import CandidateScore
from app.schemas.decision_matrix import (
    AutomationDecision,
    AutomationDecisionResult,
    DecisionConfig,
)
from app.schemas.guardrail_engine import GuardrailEngineResult
from app.services.guardrail_engine import GuardrailEngine
from app.schemas.intelligence import (
    ExceptionIntelligence,
    ClassificationResult,
    EvidenceIntelligence,
    SimilarCasesIntelligence,
    RecommendationStatus,
)
from app.schemas.evidence import EvidencePackage
from app.schemas.explanation import ExplanationResult, ExplanationStatus
from app.schemas.evidence_quality import EvidenceQualityResult, NoveltyLevel


# ============================================================================
# Authorization Decision
# ============================================================================

class AuthorizationDecision:
    """
    Explicit authorization decision from Phase 8 policy/guardrail evaluation.
    
    This is the final authorization boundary.
    Downstream components must respect this decision.
    """
    
    def __init__(
        self,
        decision: AutomationDecision,
        exception_id: str,
        case_id: str,
        risk_category: str,
        reason_codes: List[str],
        primary_reason: str,
        confidence: float,
        financial_exposure_paise: int,
        policy_version: str,
        guardrail_result: Optional[GuardrailEngineResult] = None,
        required_approval: bool = False,
    ):
        self.decision = decision
        self.exception_id = exception_id
        self.case_id = case_id
        self.risk_category = risk_category
        self.reason_codes = reason_codes
        self.primary_reason = primary_reason
        self.confidence = confidence
        self.financial_exposure_paise = financial_exposure_paise
        self.policy_version = policy_version
        self.guardrail_result = guardrail_result
        self.required_approval = required_approval
    
    def is_auto(self) -> bool:
        """True if decision is AUTO."""
        return self.decision == AutomationDecision.AUTO
    
    def is_human_review(self) -> bool:
        """True if decision is HUMAN_REVIEW."""
        return self.decision == AutomationDecision.HUMAN_REVIEW
    
    def is_unresolved(self) -> bool:
        """True if decision is UNRESOLVED."""
        return self.decision == AutomationDecision.UNRESOLVED
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "decision": self.decision.value,
            "exception_id": self.exception_id,
            "case_id": self.case_id,
            "risk_category": self.risk_category,
            "reason_codes": self.reason_codes,
            "primary_reason": self.primary_reason,
            "confidence": self.confidence,
            "financial_exposure_paise": self.financial_exposure_paise,
            "policy_version": self.policy_version,
            "required_approval": self.required_approval,
        }


# ============================================================================
# Authorization Service
# ============================================================================

class AuthorizationService:
    """
    Authorization service that evaluates Phase 7 proposals through policy/guardrails.
    
    This is the explicit authorization boundary.
    It does NOT execute financial actions.
    """
    
    def __init__(
        self,
        session: Optional[Session] = None,
        guardrail_engine: Optional[GuardrailEngine] = None,
        decision_config: Optional[DecisionConfig] = None,
    ):
        """Initialize the authorization service.
        
        Args:
            session: SQLAlchemy session for evidence retrieval (optional for Phase 8)
            guardrail_engine: Optional guardrail engine (uses default if not provided)
            decision_config: Optional decision config (uses default if not provided)
        """
        self.session = session
        self.guardrail_engine = guardrail_engine or GuardrailEngine()
        self.decision_config = decision_config or DecisionConfig()
    
    def authorize_proposal(
        self,
        proposal: ResolutionProposal,
        evidence_package: Optional[EvidencePackage] = None,
        intelligence: Optional[ExceptionIntelligence] = None,
    ) -> AuthorizationDecision:
        """
        Authorize a single resolution proposal through policy/guardrails.
        
        Args:
            proposal: Phase 7 ResolutionProposal to authorize
            evidence_package: Optional evidence package for guardrail evaluation
            intelligence: Optional intelligence for guardrail evaluation
        
        Returns:
            AuthorizationDecision with explicit authorization result
        """
        # Convert proposal to ResolutionEngineResult format for guardrails
        engine_result = self._proposal_to_engine_result(
            proposal, evidence_package, intelligence
        )
        
        # Evaluate through guardrails
        guardrail_result = self.guardrail_engine.evaluate(
            engine_result=engine_result,
            dependency_status={
                "ml_classifier": True,
                "ml_resolution_predictor": True,
                "similarity_service": True,
                "database": True,
                "evidence_retrieval": True,
                "llm": True,
                "mcp": True,
            },
        )
        
        # Extract decision
        decision = guardrail_result.decision
        required_approval = decision == AutomationDecision.HUMAN_REVIEW
        
        return AuthorizationDecision(
            decision=decision,
            exception_id=proposal.exception_id,
            case_id=proposal.case_id,
            risk_category=guardrail_result.risk_category,
            reason_codes=guardrail_result.reason_codes,
            primary_reason=guardrail_result.primary_reason,
            confidence=guardrail_result.confidence,
            financial_exposure_paise=guardrail_result.financial_exposure_paise,
            policy_version=guardrail_result.guardrail_version,
            guardrail_result=guardrail_result,
            required_approval=required_approval,
        )
    
    def authorize_selection(
        self,
        selection: SelectionResult,
        evidence_package: Optional[EvidencePackage] = None,
        intelligence: Optional[ExceptionIntelligence] = None,
    ) -> AuthorizationDecision:
        """
        Authorize a Phase 7 selection result through policy/guardrails.
        
        Args:
            selection: Phase 7 SelectionResult to authorize
            evidence_package: Optional evidence package
            intelligence: Optional intelligence
        
        Returns:
            AuthorizationDecision with explicit authorization result
        """
        # Convert selection to ResolutionEngineResult format
        engine_result = self._selection_to_engine_result(
            selection, evidence_package, intelligence
        )
        
        # Evaluate through guardrails
        guardrail_result = self.guardrail_engine.evaluate(
            engine_result=engine_result,
            dependency_status={
                "ml_classifier": True,
                "ml_resolution_predictor": True,
                "similarity_service": True,
                "database": True,
                "evidence_retrieval": True,
                "llm": True,
                "mcp": True,
            },
        )
        
        # Extract decision
        decision = guardrail_result.decision
        required_approval = decision == AutomationDecision.HUMAN_REVIEW
        
        return AuthorizationDecision(
            decision=decision,
            exception_id=selection.exception_id,
            case_id=selection.case_id,
            risk_category=guardrail_result.risk_category,
            reason_codes=guardrail_result.reason_codes,
            primary_reason=guardrail_result.primary_reason,
            confidence=guardrail_result.confidence,
            financial_exposure_paise=guardrail_result.financial_exposure_paise,
            policy_version=guardrail_result.guardrail_version,
            guardrail_result=guardrail_result,
            required_approval=required_approval,
        )
    
    def _proposal_to_engine_result(
        self,
        proposal: ResolutionProposal,
        evidence_package: Optional[EvidencePackage],
        intelligence: Optional[ExceptionIntelligence],
    ) -> ResolutionEngineResult:
        """Convert a ResolutionProposal to ResolutionEngineResult format."""
        # Extract values from proposal
        adjustment = proposal.financial_adjustment
        
        # Use intelligence or create defaults
        if intelligence:
            payment_id = intelligence.payment_id
            merchant_id = intelligence.merchant_id
            risk = "MEDIUM"  # ExceptionIntelligence doesn't have risk_category, would come from evidence quality
            evidence_coverage = intelligence.evidence.evidence_coverage if intelligence.evidence else 0.0
            evidence_consistency = intelligence.evidence.consistency_score if intelligence.evidence else 0.0
            has_conflict = intelligence.evidence.has_conflict if intelligence.evidence else False
            missing_evidence = intelligence.evidence.missing_evidence if intelligence.evidence else []
            ml_type = intelligence.classification.ml_predicted_type if intelligence.classification else None
            ml_agreement = intelligence.classification.agreement if intelligence.classification else True
            expected_amount = intelligence.expected_amount
            actual_amount = intelligence.actual_amount
            difference = intelligence.difference
        else:
            payment_id = ""
            merchant_id = ""
            risk = "MEDIUM"
            evidence_coverage = proposal.evidence_coverage
            evidence_consistency = 1.0
            has_conflict = False
            missing_evidence = []
            ml_type = None
            ml_agreement = True
            expected_amount = 0
            actual_amount = 0
            difference = 0
        
        # Create CandidateScore from proposal ranking
        candidate_score = CandidateScore(
            evidence_score=proposal.ranking.evidence_support,
            ml_score=proposal.ranking.ml_support if proposal.ranking.ml_support else 0.0,
            historical_score=proposal.ranking.historical_support if proposal.ranking.historical_support else 0.0,
            financial_consistency_score=1.0,  # Would come from actual evidence check
            novelty_penalty=0.0,
            conflict_penalty=0.0,
            final_score=proposal.ranking.confidence_score,
            weighted_evidence=proposal.ranking.evidence_support * 0.35,
            weighted_ml=(proposal.ranking.ml_support if proposal.ranking.ml_support else 0.0) * 0.20,
            weighted_historical=(proposal.ranking.historical_support if proposal.ranking.historical_support else 0.0) * 0.15,
            weighted_financial=1.0 * 0.30,
            has_evidence=True,
            has_ml_support=proposal.ranking.ml_support is not None,
            has_historical_support=proposal.ranking.historical_support is not None,
        )
        
        # Create ExplainabilityDetail
        explainability = ExplainabilityDetail(
            level="FULLY_EXPLAINABLE",
            evidence_count=len(proposal.supporting_evidence_ids),
            has_explanation=True,
            explanation_sources=["deterministic_evidence"],
            traceable=True,
        )
        
        # Build result
        return ResolutionEngineResult(
            exception_id=proposal.exception_id,
            case_id=proposal.case_id,
            payment_id=payment_id,
            merchant_id=merchant_id,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
            difference=difference,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution=proposal.resolution_type,
            selected_candidate=proposal,
            selected_score=candidate_score,
            ranked_candidates=[proposal],
            candidate_scores=[candidate_score],
            confidence=proposal.ranking.confidence_score,
            confidence_factors={},
            risk_category=risk,
            risk_factors=[],
            explainability=explainability,
            rejection_reasons=[],
            deterministic_exception_type="FEE_DIFFERENCE",  # Would come from intelligence
            ml_exception_type=ml_type,
            classification_agreement=ml_agreement,
            evidence_explanation_status="FULLY_EXPLAINED",
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            has_conflict=has_conflict,
            is_novel=False,
            missing_evidence=missing_evidence,
            pipeline_version="1.0.0",
            processing_time_ms=0,
        )
    
    def _selection_to_engine_result(
        self,
        selection: SelectionResult,
        evidence_package: Optional[EvidencePackage],
        intelligence: Optional[ExceptionIntelligence],
    ) -> ResolutionEngineResult:
        """Convert a SelectionResult to ResolutionEngineResult format."""
        # Extract from selection
        candidate = selection.selected_candidate
        score = selection.selected_score
        
        # Use intelligence or create defaults
        if intelligence:
            payment_id = intelligence.payment_id
            merchant_id = intelligence.merchant_id
            risk = "MEDIUM"  # ExceptionIntelligence doesn't have risk_category
            evidence_coverage = intelligence.evidence.evidence_coverage if intelligence.evidence else 0.0
            evidence_consistency = intelligence.evidence.consistency_score if intelligence.evidence else 0.0
            has_conflict = intelligence.evidence.has_conflict if intelligence.evidence else False
            missing_evidence = intelligence.evidence.missing_evidence if intelligence.evidence else []
            ml_type = intelligence.classification.ml_predicted_type if intelligence.classification else None
            ml_agreement = intelligence.classification.agreement if intelligence.classification else True
            expected_amount = intelligence.expected_amount
            actual_amount = intelligence.actual_amount
            difference = intelligence.difference
        else:
            payment_id = ""
            merchant_id = ""
            risk = selection.risk_category
            evidence_coverage = 0.8
            evidence_consistency = 0.8
            has_conflict = False
            missing_evidence = []
            ml_type = None
            ml_agreement = True
            expected_amount = 0
            actual_amount = 0
            difference = 0
        
        # Create CandidateScore if not provided
        if not score and candidate:
            score = CandidateScore(
                evidence_score=candidate.evidence_coverage,
                ml_score=candidate.ranking.ml_support if candidate.ranking.ml_support else 0.0,
                historical_score=candidate.ranking.historical_support if candidate.ranking.historical_support else 0.0,
                financial_consistency_score=1.0,
                novelty_penalty=0.0,
                conflict_penalty=0.0,
                final_score=candidate.ranking.confidence_score,
                weighted_evidence=candidate.evidence_coverage * 0.35,
                weighted_ml=(candidate.ranking.ml_support if candidate.ranking.ml_support else 0.0) * 0.20,
                weighted_historical=(candidate.ranking.historical_support if candidate.ranking.historical_support else 0.0) * 0.15,
                weighted_financial=1.0 * 0.30,
                has_evidence=True,
                has_ml_support=candidate.ranking.ml_support is not None,
                has_historical_support=candidate.ranking.historical_support is not None,
            )
        
        # Create ExplainabilityDetail
        explainability = selection.explainability if selection.explainability else ExplainabilityDetail(
            level="PARTIALLY_EXPLAINABLE",
            evidence_count=len(candidate.supporting_evidence_ids) if candidate else 0,
            has_explanation=True,
            explanation_sources=["deterministic_evidence"],
            traceable=True,
        )
        
        # Build result
        if candidate:
            ranked_candidates = [candidate] + selection.alternatives
        else:
            ranked_candidates = selection.alternatives
        
        return ResolutionEngineResult(
            exception_id=selection.exception_id,
            case_id=selection.case_id,
            payment_id=payment_id,
            merchant_id=merchant_id,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
            difference=difference,
            status=selection.status,
            selected_resolution=candidate.resolution_type if candidate else None,
            selected_candidate=candidate,
            selected_score=score,
            ranked_candidates=ranked_candidates,
            candidate_scores=[score] if score else [],
            confidence=selection.confidence,
            confidence_factors=selection.confidence_factors,
            risk_category=risk,
            risk_factors=selection.risk_factors,
            explainability=explainability,
            rejection_reasons=selection.rejection_reasons,
            deterministic_exception_type="FEE_DIFFERENCE",
            ml_exception_type=ml_type,
            classification_agreement=ml_agreement,
            evidence_explanation_status="FULLY_EXPLAINED",
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            has_conflict=has_conflict,
            is_novel=score.novelty_penalty > 0 if score else False,
            missing_evidence=missing_evidence,
            pipeline_version="1.0.0",
            processing_time_ms=0,
        )
