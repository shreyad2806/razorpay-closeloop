"""
Simplified Phase 8 Authorization Tests

These tests verify the Phase 8 authorization boundary using direct tests
without complex fixture dependencies.
"""

import pytest
from app.schemas.resolution_candidate import (
    ResolutionProposal,
    CandidateSource,
    FinancialAdjustment,
    CandidateRanking,
    MLSupportDetail,
    HistoricalSupportDetail,
)
from app.schemas.decision_matrix import AutomationDecision
from app.services.authorization import AuthorizationService


def test_authorization_service_exists():
    """Test that AuthorizationService can be instantiated."""
    auth_service = AuthorizationService(session=None)
    assert auth_service is not None
    assert auth_service.guardrail_engine is not None


def test_authorization_service_reuses_guardrail_engine():
    """Test that AuthorizationService reuses existing GuardrailEngine."""
    from app.services.guardrail_engine import GuardrailEngine
    
    auth_service = AuthorizationService(session=None)
    assert isinstance(auth_service.guardrail_engine, GuardrailEngine)


def test_no_provider_execution():
    """Test that AuthorizationService does not invoke provider execution."""
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Check that there are no provider API calls in actual code
    lines = [line.strip() for line in source.split('\n') if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]
    code_only = '\n'.join(lines)
    
    # Should not contain provider API calls
    assert "import razorpay" not in code_only.lower()
    assert "import stripe" not in code_only.lower()
    assert "razorpay.Client" not in code_only
    assert "stripe.api_key" not in code_only


def test_no_approval_persistence():
    """Test that AuthorizationService does not persist Approval records."""
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Should not contain database write operations for Approval
    assert "Approval(" not in source
    assert "session.add(" not in source
    assert "session.commit(" not in source


def test_no_resolution_state_transition():
    """Test that AuthorizationService does not transition Resolution state."""
    import inspect
    from app.services.authorization import AuthorizationService
    
    source = inspect.getsource(AuthorizationService)
    
    # Should not contain state transition calls
    assert "transition_to(" not in source
    assert "status =" not in source  # No direct status mutation


def test_existing_decision_matrix_reused():
    """Test that existing DecisionMatrix is reused via GuardrailEngine."""
    from app.services.guardrail_engine import GuardrailEngine
    from app.services.decision_matrix import AutomationDecisionMatrix
    
    guardrail = GuardrailEngine()
    
    # Verify GuardrailEngine uses DecisionMatrix
    assert isinstance(guardrail.decision_matrix, AutomationDecisionMatrix)


def test_authorization_decision_schema():
    """Test that AuthorizationDecision has correct structure."""
    from app.services.authorization import AuthorizationDecision
    
    decision = AuthorizationDecision(
        decision=AutomationDecision.AUTO,
        exception_id="EXC-001",
        case_id="CASE-001",
        risk_category="LOW",
        reason_codes=["ALL_GATES_PASSED"],
        primary_reason="All gates passed",
        confidence=0.85,
        financial_exposure_paise=25000,
        policy_version="1.0.0",
        required_approval=False,
    )
    
    assert decision.is_auto() == True
    assert decision.is_human_review() == False
    assert decision.is_unresolved() == False
    assert decision.required_approval == False
    
    # Test serialization
    decision_dict = decision.to_dict()
    assert decision_dict["decision"] == "AUTO"
    assert decision_dict["exception_id"] == "EXC-001"
    assert decision_dict["required_approval"] == False


def test_authorization_decision_human_review():
    """Test that HUMAN_REVIEW decision sets required_approval."""
    from app.services.authorization import AuthorizationDecision
    
    decision = AuthorizationDecision(
        decision=AutomationDecision.HUMAN_REVIEW,
        exception_id="EXC-002",
        case_id="CASE-002",
        risk_category="MEDIUM",
        reason_codes=["MODERATE_EXPOSURE"],
        primary_reason="Amount requires review",
        confidence=0.75,
        financial_exposure_paise=50000,
        policy_version="1.0.0",
        required_approval=True,
    )
    
    assert decision.is_auto() == False
    assert decision.is_human_review() == True
    assert decision.is_unresolved() == False
    assert decision.required_approval == True


def test_authorization_decision_unresolved():
    """Test that UNRESOLVED decision does not require approval."""
    from app.services.authorization import AuthorizationDecision
    
    decision = AuthorizationDecision(
        decision=AutomationDecision.UNRESOLVED,
        exception_id="EXC-003",
        case_id="CASE-003",
        risk_category="HIGH",
        reason_codes=["UNKNOWN_PATTERN"],
        primary_reason="Unknown pattern",
        confidence=0.30,
        financial_exposure_paise=0,
        policy_version="1.0.0",
        required_approval=False,
    )
    
    assert decision.is_auto() == False
    assert decision.is_human_review() == False
    assert decision.is_unresolved() == True
    assert decision.required_approval == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
