"""
Outcome Recording and Reward Generation Node for Phase 7J.

Records the final outcome of the workflow and generates reward signals.

Ground truth may be used for:
- reward calculation
- offline evaluation
- model training

It must NOT be used to decide the current case.

This node:
1. Records the complete workflow outcome
2. Generates a reward signal
3. Stores a historical learning record
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import (
    AgentState,
    HumanApprovalStatus,
    VerificationStatus,
    WorkflowStatus,
)
from app.schemas.outcome import (
    HistoricalLearningRecord,
    RewardSignal,
    RewardType,
    WorkflowOutcome,
    WorkflowOutcomeRecord,
)


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Node
# ─────────────────────────────────────────────────────────────────────────────


def _record_node(
    state: AgentState,
    node_name: str,
    success: bool,
    error: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    elapsed_ms = None
    if start_time:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    log_entry = {
        "node": node_name,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms else None,
        "error": error,
    }

    metadata = state.metadata.model_dump()
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + [node_name]
    metadata["execution_log"] = list(state.metadata.execution_log) + [log_entry]

    if error:
        metadata["errors"] = list(state.metadata.errors) + [error]

    return {"metadata": metadata}


def record_outcome(state: AgentState) -> Dict[str, Any]:
    """Record workflow outcome and generate reward signal.

    This node:
    1. Determines the final workflow outcome
    2. Records all relevant context
    3. Generates a reward signal for future learning
    4. Stores a historical learning record

    Ground truth is used ONLY for reward calculation.
    It does NOT affect the outcome decision.
    """
    start_time = time.perf_counter()
    node_name = "record_outcome"

    try:
        # ── Determine outcome ──
        outcome = _determine_outcome(state)

        # ── Build outcome record ──
        outcome_record = _build_outcome_record(state, outcome)

        # ── Generate reward signal ──
        reward = _calculate_reward(state, outcome_record)

        # ── Build historical learning record ──
        learning_record = _build_learning_record(state, outcome_record, reward)

        # ── Update reward state ──
        reward_state = state.reward.model_dump()
        reward_state["reward_status"] = "CALCULATED"
        reward_state["reward"] = reward.reward_value
        reward_state["reward_reason"] = reward.reward_reason
        reward_state["reward_calculated_at"] = datetime.utcnow().isoformat()

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["outcome"] = outcome_record.model_dump(mode="json")
        updates["reward"] = reward.model_dump(mode="json")
        updates["learning_record"] = learning_record.model_dump(mode="json")
        updates["reward_state"] = reward_state
        updates["metadata"]["current_node"] = node_name

        return updates

    except Exception as e:
        updates = _record_node(state, node_name, success=False, error=str(e), start_time=start_time)
        updates["metadata"]["current_node"] = node_name
        return updates


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Determination
# ─────────────────────────────────────────────────────────────────────────────


def _determine_outcome(state: AgentState) -> WorkflowOutcome:
    """Determine the final workflow outcome from state.

    Priority order:
    1. System errors (highest priority — always surface)
    2. Human rejection (explicit human action)
    3. Verification failure (explicit verification failure)
    4. Unresolved (guardrails blocked)
    5. Escalation warnings (explicit escalation)
    6. Human review pending (waiting for human)
    7. Completed resolution (action boundary passed)
    8. Default: UNRESOLVED
    """
    decision = state.decision
    verification_status = state.verification.verification_status.value
    approval_status = state.human_review.approval_status.value
    workflow_status = state.metadata.workflow_status.value

    # 1. System errors (highest priority)
    if workflow_status == "FAILED" and state.errors:
        if verification_status == "FAILED":
            return WorkflowOutcome.VERIFICATION_FAILED
        if approval_status == "REJECTED":
            return WorkflowOutcome.REJECTED_BY_HUMAN
        if decision == "UNRESOLVED":
            return WorkflowOutcome.UNRESOLVED
        return WorkflowOutcome.SYSTEM_ERROR

    # 2. Human rejection (explicit human action)
    if approval_status == "REJECTED":
        return WorkflowOutcome.REJECTED_BY_HUMAN

    # 3. Verification failure
    if verification_status == "FAILED":
        return WorkflowOutcome.VERIFICATION_FAILED

    # 4. Unresolved (guardrails blocked — highest decision priority)
    if decision == "UNRESOLVED":
        return WorkflowOutcome.UNRESOLVED

    # 5. Escalation warnings
    for w in state.warnings:
        if "ESCALATED" in w:
            return WorkflowOutcome.ESCALATED

    # 6. Human review pending (not yet approved)
    if decision == "HUMAN_REVIEW":
        if approval_status == "APPROVED":
            return WorkflowOutcome.RESOLVED_HUMAN
        return WorkflowOutcome.ESCALATED

    # 7. Completed resolution (action boundary passed)
    has_action = state.metadata.current_node in ("resolve_action_boundary", "record_outcome")
    if has_action and verification_status == "VERIFIED":
        return WorkflowOutcome.RESOLVED_AUTO

    # 8. Auto with verification
    if decision == "AUTO" and verification_status == "VERIFIED":
        return WorkflowOutcome.RESOLVED_AUTO

    return WorkflowOutcome.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# Record Building
# ─────────────────────────────────────────────────────────────────────────────


def _build_outcome_record(state: AgentState, outcome: WorkflowOutcome) -> WorkflowOutcomeRecord:
    """Build complete outcome record."""
    candidate = state.selected_candidate or {}
    return WorkflowOutcomeRecord(
        workflow_id=state.metadata.workflow_id,
        exception_id=state.metadata.exception_id,
        case_id=state.metadata.case_id,
        candidate_id=candidate.get("candidate_id"),
        decision=state.decision or "NONE",
        resolution_type=candidate.get("resolution_type"),
        authorization_source=(
            "HUMAN_APPROVAL"
            if state.human_review.approval_status == HumanApprovalStatus.APPROVED
            else "AUTO_GUARDRAIL" if state.decision == "AUTO" else None
        ),
        human_approved=state.human_review.approval_status == HumanApprovalStatus.APPROVED,
        verification_passed=state.verification.verification_status == VerificationStatus.VERIFIED,
        verification_action=state.verification.verification_status.value,
        financial_adjustment_paise=candidate.get("amount_paise", 0),
        action_created=state.metadata.current_node in ("resolve_action_boundary", "record_outcome"),
        outcome=outcome,
        outcome_reason=_outcome_reason(state, outcome),
        confidence=state.confidence,
        risk=state.risk,
        exception_type=(state.classification or {}).get("exception_type"),
        nodes_executed=list(state.metadata.nodes_executed),
        completed_at=datetime.utcnow(),
    )


def _outcome_reason(state: AgentState, outcome: WorkflowOutcome) -> str:
    """Generate deterministic reason for outcome."""
    if outcome == WorkflowOutcome.RESOLVED_AUTO:
        return "All guardrails passed, verification succeeded, auto-resolution authorized"
    elif outcome == WorkflowOutcome.RESOLVED_HUMAN:
        return "Human reviewer approved resolution after guardrail review"
    elif outcome == WorkflowOutcome.REJECTED_BY_HUMAN:
        return f"Human reviewer rejected: {state.human_review.reviewer_notes or 'no reason'}"
    elif outcome == WorkflowOutcome.VERIFICATION_FAILED:
        return "Post-resolution verification detected stale or inconsistent state"
    elif outcome == WorkflowOutcome.UNRESOLVED:
        return f"Guardrail decision: {state.decision}. No resolution path available."
    elif outcome == WorkflowOutcome.ESCALATED:
        return "Case escalated for manual investigation"
    elif outcome == WorkflowOutcome.SYSTEM_ERROR:
        errors = state.errors[-1] if state.errors else "unknown error"
        return f"System error: {errors}"
    return "Unknown outcome"


# ─────────────────────────────────────────────────────────────────────────────
# Reward Calculation
# ─────────────────────────────────────────────────────────────────────────────


def _calculate_reward(state: AgentState, outcome: WorkflowOutcomeRecord) -> RewardSignal:
    """Calculate reward signal for future learning.

    May use ground truth for evaluation.
    Does NOT affect current case decision.
    """
    reward_value = 0.0
    reward_type = RewardType.NO_REWARD
    components = []

    # Base reward from outcome
    if outcome.outcome == WorkflowOutcome.RESOLVED_AUTO:
        reward_value += 0.5
        reward_type = RewardType.CORRECT_RESOLUTION
        components.append("auto_resolution_success")
    elif outcome.outcome == WorkflowOutcome.RESOLVED_HUMAN:
        reward_value += 0.3
        reward_type = RewardType.CORRECT_RESOLUTION
        components.append("human_approved_resolution")
    elif outcome.outcome == WorkflowOutcome.REJECTED_BY_HUMAN:
        reward_value += 0.1
        reward_type = RewardType.PARTIAL_CREDIT
        components.append("human_rejection_valid")
    elif outcome.outcome == WorkflowOutcome.VERIFICATION_FAILED:
        reward_value -= 0.3
        reward_type = RewardType.PENALTY
        components.append("verification_failure_penalty")
    elif outcome.outcome == WorkflowOutcome.UNRESOLVED:
        reward_value += 0.0
        reward_type = RewardType.NO_REWARD
        components.append("unresolved_no_reward")
    elif outcome.outcome == WorkflowOutcome.ESCALATED:
        reward_value += 0.05
        reward_type = RewardType.PARTIAL_CREDIT
        components.append("escalated_partial_credit")
    elif outcome.outcome == WorkflowOutcome.SYSTEM_ERROR:
        reward_value -= 0.5
        reward_type = RewardType.PENALTY
        components.append("system_error_penalty")

    # Verification bonus
    verification_bonus = 0.0
    if outcome.verification_passed:
        verification_bonus = 0.2
        reward_value += verification_bonus
        components.append("verification_bonus")

    # Financial accuracy bonus
    financial_accuracy = None
    if outcome.resolution_type and outcome.financial_adjustment_paise > 0:
        # Heuristic: if resolution was attempted and verification passed, good accuracy
        if outcome.verification_passed:
            financial_accuracy = 0.9
            reward_value += 0.1
            components.append("financial_accuracy_bonus")
        else:
            financial_accuracy = 0.5

    # Human approval bonus
    human_bonus = 0.0
    if outcome.human_approved:
        human_bonus = 0.1
        reward_value += human_bonus
        components.append("human_approval_bonus")

    # Clamp reward
    reward_value = max(-1.0, min(1.0, reward_value))

    reason = f"Reward components: {', '.join(components)}" if components else "No reward components"

    return RewardSignal(
        workflow_id=outcome.workflow_id,
        exception_id=outcome.exception_id,
        reward_type=reward_type,
        reward_value=round(reward_value, 3),
        reward_reason=reason,
        resolution_correct=(
            outcome.outcome in (WorkflowOutcome.RESOLVED_AUTO, WorkflowOutcome.RESOLVED_HUMAN)
        ),
        verification_bonus=verification_bonus,
        financial_accuracy=financial_accuracy,
        human_approval_bonus=human_bonus,
        calculated_at=datetime.utcnow(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Learning Record
# ─────────────────────────────────────────────────────────────────────────────


def _build_learning_record(
    state: AgentState,
    outcome: WorkflowOutcomeRecord,
    reward: RewardSignal,
) -> HistoricalLearningRecord:
    """Build historical learning record for future retrieval."""
    evidence = state.evidence_package or {}
    candidate = state.selected_candidate or {}

    return HistoricalLearningRecord(
        workflow_id=outcome.workflow_id,
        exception_id=outcome.exception_id,
        case_id=outcome.case_id,
        exception_type=outcome.exception_type or "UNKNOWN",
        resolution_type=outcome.resolution_type,
        outcome=outcome.outcome,
        financial_adjustment_paise=outcome.financial_adjustment_paise,
        confidence=outcome.confidence,
        risk=outcome.risk,
        evidence_coverage=evidence.get("evidence_coverage"),
        evidence_consistency=evidence.get("evidence_consistency"),
        supporting_evidence_count=len(evidence.get("supporting_evidence_ids", [])),
        verification_passed=outcome.verification_passed,
        human_approved=outcome.human_approved,
        authorization_source=outcome.authorization_source,
        reward=reward,
        nodes_executed=outcome.nodes_executed,
        created_at=datetime.utcnow(),
        resolved_at=outcome.completed_at,
    )
