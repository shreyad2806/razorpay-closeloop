"""
Resolution Execution Service for Razorpay CloseLoop Phase 8A.

Simulates financial execution in a controlled environment.

CRITICAL:
- This service does NOT connect to real financial systems
- It simulates financial actions safely using synthetic data
- Real execution belongs to a future guarded agent/action layer

Execution and verification are separate stages.
This service handles EXECUTION only.
"""

import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.schemas.execution import (
    AdjustmentRecord,
    ExecutionResult,
    ExecutionStatus,
    ExecutionTransitionError,
    FinancialStateSnapshot,
    is_valid_transition,
)
from app.services.idempotency import ConcurrencyGuard, IdempotencyStore


# ─────────────────────────────────────────────────────────────────────────────
# Execution Service
# ─────────────────────────────────────────────────────────────────────────────


class ResolutionExecutionService:
    """Simulated resolution execution service.

    Executes financial adjustments in a controlled environment.
    All actions are simulated — no real financial operations.
    Uses IdempotencyStore and ConcurrencyGuard for duplicate/concurrent protection.
    """

    def __init__(self):
        self._executed_keys: Dict[str, ExecutionResult] = {}
        self._idempotency_store = IdempotencyStore()
        self._concurrency_guard = ConcurrencyGuard(self._idempotency_store)

    @property
    def concurrency_guard(self) -> ConcurrencyGuard:
        """Access the concurrency guard for testing/integration."""
        return self._concurrency_guard

    @property
    def idempotency_store(self) -> IdempotencyStore:
        """Access the idempotency store for testing/integration."""
        return self._idempotency_store

    def execute(
        self,
        action_request: Dict[str, Any],
        current_financial_state: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a resolution request.

        Args:
            action_request: The authorized action request from Phase 7I
            current_financial_state: Current financial state for before/after capture

        Returns:
            ExecutionResult with full audit trail
        """
        start_time = time.perf_counter()
        idempotency_key = action_request.get("idempotency_key", "")

        # ── Concurrency guard + idempotency check ──
        worker_id = action_request.get("worker_id", f"worker-{uuid.uuid4().hex[:8]}")
        dedup = self._concurrency_guard.deduplicate(idempotency_key, worker_id)
        if dedup.is_duplicate and dedup.existing_result:
            # Return cached result from idempotency store
            try:
                return ExecutionResult.model_validate(dedup.existing_result)
            except Exception:
                pass
        if dedup.is_duplicate:
            # Could not claim lock
            return self._create_failed_result(
                action_request, f"Duplicate request: key already claimed by {dedup.worker_id or 'unknown'}"
            )

        # ── Validate preconditions ──
        validation_errors = self._validate_preconditions(action_request)
        if validation_errors:
            return self._create_failed_result(
                action_request, "; ".join(validation_errors)
            )

        # ── Capture before state ──
        before_state = self._capture_before_state(
            action_request, current_financial_state
        )

        # ── Execute simulated adjustment ──
        try:
            adjustment = self._execute_adjustment(action_request, before_state)
            status = ExecutionStatus.EXECUTED
            error = None
        except Exception as e:
            adjustment = None
            status = ExecutionStatus.EXECUTION_FAILED
            error = str(e)

        # ── Capture after state (only if execution succeeded) ──
        after_state = None
        if status == ExecutionStatus.EXECUTED:
            after_state = self._capture_after_state(
                action_request, before_state, adjustment
            )

        # ── Build result ──
        execution_id = f"EXE-{uuid.uuid4().hex[:8].upper()}"
        requested_amount = action_request.get("financial_adjustment_paise", 0)
        actual_amount = adjustment.amount_paise if adjustment else 0

        result = ExecutionResult(
            execution_id=execution_id,
            action_id=action_request.get("action_id", ""),
            idempotency_key=idempotency_key,
            workflow_id=action_request.get("workflow_id", ""),
            exception_id=action_request.get("exception_id", ""),
            case_id=action_request.get("case_id"),
            candidate_id=action_request.get("candidate_id"),
            resolution_type=action_request.get("resolution_type", ""),
            authorization_source=action_request.get("authorization_source", ""),
            before_state=before_state,
            after_state=after_state,
            adjustment=adjustment,
            requested_adjustment_paise=requested_amount,
            actual_adjustment_paise=actual_amount,
            status=status,
            decision=action_request.get("guardrail_decision"),
            confidence=action_request.get("guardrail_confidence"),
            risk=action_request.get("metadata", {}).get("risk"),
            guardrail_reason_codes=action_request.get("metadata", {}).get("reason_codes", []),
            evidence_references=list(
                (action_request.get("evidence_summary") or {}).get("evidence_ids", [])
            ),
            error=error,
            created_at=datetime.utcnow(),
            executed_at=datetime.utcnow() if status == ExecutionStatus.EXECUTED else None,
        )

        # ── Store for idempotency ──
        if status == ExecutionStatus.EXECUTED:
            self._executed_keys[idempotency_key] = result
            self._concurrency_guard.complete(idempotency_key, result)
        else:
            self._concurrency_guard.fail(idempotency_key, error or "Execution failed")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Precondition Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_preconditions(
        self, action_request: Dict[str, Any]
    ) -> List[str]:
        """Validate all execution preconditions."""
        errors = []

        if not action_request.get("workflow_id"):
            errors.append("Missing workflow_id")

        if not action_request.get("exception_id"):
            errors.append("Missing exception_id")

        if not action_request.get("idempotency_key"):
            errors.append("Missing idempotency_key")

        if not action_request.get("resolution_type"):
            errors.append("Missing resolution_type")

        # Guardrail must allow execution
        guardrail_decision = action_request.get("guardrail_decision")
        if guardrail_decision not in ("AUTO", "HUMAN_REVIEW"):
            errors.append(
                f"Guardrail decision '{guardrail_decision}' does not allow execution"
            )

        # Authorization must exist
        auth_source = action_request.get("authorization_source")
        if not auth_source or auth_source == "NONE":
            errors.append("No authorization source")
        normalized_auth_source = str(auth_source).upper()
        auto_sources = {"AUTO_GUARDRAIL", "GUARDRAIL_AUTO"}
        human_sources = {"HUMAN_APPROVAL", "HUMAN_APPROVED"}
        if guardrail_decision == "AUTO" and normalized_auth_source not in auto_sources:
            errors.append("AUTO execution requires AUTO_GUARDRAIL authorization")
        elif guardrail_decision == "HUMAN_REVIEW" and normalized_auth_source not in human_sources:
            errors.append("HUMAN_REVIEW execution requires HUMAN_APPROVAL authorization")

        # Verification must have passed
        if not action_request.get("verification_passed"):
            errors.append("Verification has not passed")

        # Financial adjustment must be specified
        adjustment = action_request.get("financial_adjustment_paise")
        if adjustment is None:
            errors.append("Missing financial_adjustment_paise")

        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # State Capture
    # ─────────────────────────────────────────────────────────────────────────

    def _capture_before_state(
        self,
        action_request: Dict[str, Any],
        financial_state: Optional[Dict[str, Any]],
    ) -> FinancialStateSnapshot:
        """Capture financial state immediately before execution."""
        state = financial_state or {}
        return FinancialStateSnapshot(
            exception_id=action_request.get("exception_id", ""),
            case_id=action_request.get("case_id"),
            payment_amount=state.get("payment_amount", 0),
            expected_amount=state.get("expected_amount", 0),
            actual_amount=state.get("actual_amount", 0),
            difference=state.get("difference", 0),
            total_refunds=state.get("total_refunds", 0),
            total_fees=state.get("total_fees", 0),
            total_taxes=state.get("total_taxes", 0),
            total_adjustments=state.get("total_adjustments", 0),
            settlement_count=state.get("settlement_count", 0),
            refund_count=state.get("refund_count", 0),
            fee_count=state.get("fee_count", 0),
            tax_count=state.get("tax_count", 0),
            adjustment_count=state.get("adjustment_count", 0),
            captured_at=datetime.utcnow(),
            snapshot_reason="pre_execution",
        )

    def _capture_after_state(
        self,
        action_request: Dict[str, Any],
        before_state: FinancialStateSnapshot,
        adjustment: AdjustmentRecord,
    ) -> FinancialStateSnapshot:
        """Capture financial state after simulated execution."""
        # Simulate the adjustment effect on amounts
        adj_amount = adjustment.amount_paise
        resolution_type = action_request.get("resolution_type", "")

        # Calculate new amounts based on resolution type
        new_actual = before_state.actual_amount
        new_adjustments = before_state.total_adjustments

        if "FEE" in resolution_type.upper():
            # Fee adjustment reduces the fee component
            new_actual = before_state.actual_amount + adj_amount
        elif "REFUND" in resolution_type.upper():
            # Refund adjustment adds to the refund component
            new_actual = before_state.actual_amount + adj_amount
        elif "TAX" in resolution_type.upper():
            # Tax adjustment
            new_actual = before_state.actual_amount + adj_amount
        else:
            # Generic adjustment
            new_actual = before_state.actual_amount + adj_amount

        new_adjustments += adj_amount
        new_diff = before_state.expected_amount - new_actual

        return FinancialStateSnapshot(
            exception_id=before_state.exception_id,
            case_id=before_state.case_id,
            payment_amount=before_state.payment_amount,
            expected_amount=before_state.expected_amount,
            actual_amount=new_actual,
            difference=new_diff,
            total_refunds=before_state.total_refunds,
            total_fees=before_state.total_fees,
            total_taxes=before_state.total_taxes,
            total_adjustments=new_adjustments,
            settlement_count=before_state.settlement_count,
            refund_count=before_state.refund_count,
            fee_count=before_state.fee_count,
            tax_count=before_state.tax_count,
            adjustment_count=before_state.adjustment_count + 1,
            captured_at=datetime.utcnow(),
            snapshot_reason="post_execution",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Adjustment Execution
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_adjustment(
        self,
        action_request: Dict[str, Any],
        before_state: FinancialStateSnapshot,
    ) -> AdjustmentRecord:
        """Execute a simulated financial adjustment.

        In production, this would call the actual financial API.
        Here it creates an adjustment record.
        """
        requested = action_request.get("financial_adjustment_paise", 0)
        resolution_type = action_request.get("resolution_type", "")

        # Validate the adjustment is sensible
        if requested < 0:
            raise ValueError(f"Negative adjustment not allowed: {requested}")

        # Determine adjustment type from resolution type
        adj_type = "CORRECTION"
        if "FEE" in resolution_type.upper():
            adj_type = "FEE_REVERSAL"
        elif "REFUND" in resolution_type.upper():
            adj_type = "REFUND"
        elif "TAX" in resolution_type.upper():
            adj_type = "TAX_ADJUSTMENT"

        # Determine affected records
        affected = []
        if "FEE" in resolution_type.upper():
            affected = [f"FEE-{before_state.exception_id}"]
        elif "REFUND" in resolution_type.upper():
            affected = [f"REF-{before_state.exception_id}"]

        adjustment_id = f"ADJ-{uuid.uuid4().hex[:8].upper()}"

        return AdjustmentRecord(
            adjustment_id=adjustment_id,
            adjustment_type=adj_type,
            amount_paise=requested,
            requested_amount_paise=requested,
            affected_records=affected,
            status="applied",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_failed_result(
        self,
        action_request: Dict[str, Any],
        error: str,
    ) -> ExecutionResult:
        """Create a failed execution result."""
        before_state = self._capture_before_state(action_request, None)

        return ExecutionResult(
            execution_id=f"EXE-{uuid.uuid4().hex[:8].upper()}",
            action_id=action_request.get("action_id", ""),
            idempotency_key=action_request.get("idempotency_key", ""),
            workflow_id=action_request.get("workflow_id", ""),
            exception_id=action_request.get("exception_id", ""),
            case_id=action_request.get("case_id"),
            candidate_id=action_request.get("candidate_id"),
            resolution_type=action_request.get("resolution_type", ""),
            authorization_source=action_request.get("authorization_source", ""),
            before_state=before_state,
            requested_adjustment_paise=action_request.get("financial_adjustment_paise") or 0,
            actual_adjustment_paise=0,
            status=ExecutionStatus.EXECUTION_FAILED,
            error=error,
            created_at=datetime.utcnow(),
        )

    def transition_status(
        self,
        result: ExecutionResult,
        new_status: ExecutionStatus,
        reason: Optional[str] = None,
    ) -> ExecutionResult:
        """Transition an execution result to a new status.

        Only allows valid transitions per the centralized policy.
        This is the ONLY way to change execution status.

        Args:
            result: Current execution result
            new_status: Target status
            reason: Optional reason for the transition

        Returns:
            Updated execution result

        Raises:
            ExecutionTransitionError: If transition is not allowed
        """
        if not is_valid_transition(result.status, new_status):
            raise ExecutionTransitionError(
                f"Invalid transition: {result.status.value} → {new_status.value}"
            )

        # Apply transition via the result's own method
        result.transition_to(new_status)

        # Update timestamps
        now = datetime.utcnow()
        if new_status == ExecutionStatus.VERIFIED:
            result.verified_at = now
        elif new_status == ExecutionStatus.ROLLED_BACK:
            result.rolled_back_at = now
            if reason:
                result.rollback_reason = reason

        # Update both in-memory cache and idempotency store
        if result.idempotency_key in self._executed_keys:
            self._executed_keys[result.idempotency_key] = result
            self._idempotency_store.complete_key(result.idempotency_key, result)

        return result

    def has_executed(self, idempotency_key: str) -> bool:
        """Check if a request has already been executed."""
        return idempotency_key in self._executed_keys

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionResult]:
        """Get a previous execution result by idempotency key."""
        return self._executed_keys.get(idempotency_key)
