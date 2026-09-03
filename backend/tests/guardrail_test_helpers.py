from typing import Any, Dict

from app.agent.guardrail_node import _build_engine_result_for_guardrails
from app.schemas.agent_state import AgentState
from app.services.guardrail_engine import GuardrailEngine


def simulate_guardrail_evaluation(
    state: AgentState,
    engine_result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a test state through the real GuardrailEngine."""
    engine_result = _build_engine_result_for_guardrails(state)
    # The fixture represents a known, consistent case unless a test overrides
    # these safety fields explicitly. Production keeps unknown values fail-closed.
    if engine_result.has_conflict is None:
        engine_result.has_conflict = False
    if engine_result.is_novel is None:
        engine_result.is_novel = False
    for key, value in engine_result_dict.items():
        if hasattr(engine_result, key):
            setattr(engine_result, key, value)

    result = GuardrailEngine().evaluate(engine_result)
    return {
        "decision": result.decision.value,
        "confidence": result.confidence,
        "risk_category": result.risk_category,
        "reason_codes": result.reason_codes,
        "primary_reason": result.primary_reason,
        "financial_exposure_paise": result.financial_exposure_paise,
        "evidence_coverage": result.evidence_coverage,
        "evidence_consistency": result.evidence_consistency,
        "is_novel": result.is_novel,
        "has_conflict": result.has_conflict,
        "system_healthy": result.system_healthy,
        "passed_gates": result.passed_gates,
        "failed_gates": result.failed_gates,
    }
