"""
Unified Guardrail Engine for Razorpay CloseLoop Phase 6F.

Chains all Phase 6 safety guards into one integrated pipeline:

1. Confidence Gate (6A)
2. Financial Exposure Guard (6B)
3. Evidence Safety Guard (6C)
4. System Failure Fallbacks (6D)
5. Decision Matrix (6E)

This is the FINAL safety decision layer.
It must NOT execute financial actions.
It must NOT alter financial amounts.
It must NOT invent evidence.

FAIL-CLOSED: If this engine encounters an unexpected error,
it must NEVER default to AUTO.
"""

import time
import traceback
from typing import Dict, Optional

from app.core.structured_logging import (
    WorkflowEvent, guardrail_logger, set_correlation_ids,
)
from app.schemas.confidence_gate import ConfidenceGateConfig, GateAction
from app.schemas.decision_matrix import AutomationDecision
from app.schemas.evidence_guard import EvidenceAction
from app.schemas.exposure_guard import ExposureAction
from app.schemas.guardrail_engine import GuardrailEngineResult
from app.schemas.resolution_engine import ResolutionEngineResult
from app.services.confidence_gate import ConfidenceGate
from app.services.decision_matrix import AutomationDecisionMatrix
from app.services.evidence_guard import EvidenceGuard
from app.services.exposure_guard import ExposureGuard
from app.services.fallback_guard import FallbackGuard


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail Engine
# ─────────────────────────────────────────────────────────────────────────────


class GuardrailEngine:
    """
    Unified guardrail engine that chains all Phase 6 safety guards.

    Given a ResolutionEngineResult, evaluates all safety conditions
    and produces a final AUTO / HUMAN_REVIEW / UNRESOLVED decision.

    This is the FINAL safety decision layer.
    It must NOT execute financial actions.
    """

    def __init__(
        self,
        confidence_gate: Optional[ConfidenceGate] = None,
        exposure_guard: Optional[ExposureGuard] = None,
        evidence_guard: Optional[EvidenceGuard] = None,
        fallback_guard: Optional[FallbackGuard] = None,
        decision_matrix: Optional[AutomationDecisionMatrix] = None,
    ):
        """Initialize the guardrail engine.

        Args:
            confidence_gate: Phase 6A confidence gate
            exposure_guard: Phase 6B exposure guard
            evidence_guard: Phase 6C evidence guard
            fallback_guard: Phase 6D fallback guard
            decision_matrix: Phase 6E decision matrix
        """
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.exposure_guard = exposure_guard or ExposureGuard()
        self.evidence_guard = evidence_guard or EvidenceGuard()
        self.fallback_guard = fallback_guard or FallbackGuard()
        self.decision_matrix = decision_matrix or AutomationDecisionMatrix()

    def evaluate(
        self,
        engine_result: ResolutionEngineResult,
        dependency_status: Optional[Dict[str, bool]] = None,
    ) -> GuardrailEngineResult:
        """Evaluate all safety guards and produce a final decision.

        Args:
            engine_result: Phase 5 resolution engine output
            dependency_status: Map of dependency_name → is_healthy

        Returns:
            GuardrailEngineResult with final decision and full audit trail
        """
        start_time = time.perf_counter()

        set_correlation_ids(exception_id=engine_result.exception_id)
        guardrail_logger.info(WorkflowEvent.GUARDRAILS_CHECKED.value,
                            f"Evaluating guardrails",
                            exception_id=engine_result.exception_id,
                            confidence=engine_result.confidence,
                            risk=engine_result.risk_category)

        try:
            result = self._evaluate_inner(engine_result, dependency_status)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            guardrail_logger.success(WorkflowEvent.GUARDRAILS_CHECKED.value,
                                   f"Guardrail decision: {result.decision.value}",
                                   duration_ms=elapsed_ms,
                                   exception_id=engine_result.exception_id,
                                   decision=result.decision.value,
                                   confidence=result.confidence,
                                   risk=result.risk_category,
                                   exposure_paise=result.financial_exposure_paise,
                                   passed_gates=len(result.passed_gates),
                                   failed_gates=len(result.failed_gates))
            return result
        except Exception:
            # FAIL-CLOSED: unexpected error → HUMAN_REVIEW
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            guardrail_logger.failure(WorkflowEvent.GUARDRAILS_FAILED.value,
                                   f"Guardrail evaluation failed",
                                   duration_ms=elapsed_ms,
                                   exception_id=engine_result.exception_id,
                                   error_type="unexpected",
                                   error_message=str(Exception))
            return self._fail_closed_on_error(engine_result, start_time)

    def _evaluate_inner(
        self,
        engine_result: ResolutionEngineResult,
        dependency_status: Optional[Dict[str, bool]],
    ) -> GuardrailEngineResult:
        """Inner evaluation with error handling."""

        # Default dependency status
        if dependency_status is None:
            dependency_status = {
                "ml_classifier": True,
                "ml_resolution_predictor": True,
                "similarity_service": True,
                "database": True,
                "evidence_retrieval": True,
                "llm": True,
                "mcp": True,
            }

        # ── Step 1: Confidence Gate (6A) ──
        gate_result = self.confidence_gate.evaluate(engine_result)

        # ── Step 2: Financial Exposure Guard (6B) ──
        exposure_result = self.exposure_guard.evaluate(engine_result, gate_result)

        # ── Step 3: Evidence Safety Guard (6C) ──
        evidence_result = self.evidence_guard.evaluate(engine_result)

        # ── Step 4: System Failure Fallbacks (6D) ──
        fallback_result = self.fallback_guard.evaluate(
            dependency_status, engine_result
        )

        # ── Step 5: Decision Matrix (6E) ──
        decision_result = self.decision_matrix.evaluate(
            engine_result,
            gate_result,
            exposure_result,
            evidence_result,
            fallback_result,
        )

        # ── Extract values ──
        confidence = engine_result.confidence
        risk = engine_result.risk_category
        evidence_coverage = engine_result.evidence_coverage
        evidence_consistency = engine_result.evidence_consistency
        exposure_paise = exposure_result.adjustment_amount_paise
        is_novel = evidence_result.is_novel
        has_conflict = evidence_result.has_conflict
        system_healthy = fallback_result.can_proceed
        critical_failures = [
            f.dependency_name for f in fallback_result.critical_failures
        ]

        # Candidate info
        candidate_id = None
        selected_resolution = None
        if engine_result.selected_candidate:
            candidate_id = engine_result.selected_candidate.candidate_id
            selected_resolution = engine_result.selected_resolution

        # Timing
        elapsed_ms = (time.perf_counter() - (time.perf_counter() - 0)) * 1000

        return GuardrailEngineResult(
            # Identification
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
            payment_id=engine_result.payment_id,
            merchant_id=engine_result.merchant_id,
            # Candidate
            candidate_id=candidate_id,
            selected_resolution=selected_resolution,
            # Decision
            decision=decision_result.decision,
            # Confidence and risk
            confidence=confidence,
            risk_category=risk,
            # Financial exposure
            financial_exposure_paise=exposure_paise,
            # Evidence
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            # Novelty and conflict
            is_novel=is_novel,
            has_conflict=has_conflict,
            # Verification
            verification_possible=True,
            # Gates
            passed_gates=decision_result.passed_gates,
            failed_gates=decision_result.failed_gates,
            # Reasons
            reason_codes=decision_result.reason_codes,
            primary_reason=decision_result.primary_reason,
            # Individual guard results (audit trail)
            confidence_gate_result=gate_result,
            exposure_guard_result=exposure_result,
            evidence_guard_result=evidence_result,
            fallback_result=fallback_result,
            decision_result=decision_result,
            # System health
            system_healthy=system_healthy,
            critical_failures=critical_failures,
            # Metadata
            guardrail_version="1.0.0",
        )

    def _fail_closed_on_error(
        self,
        engine_result: ResolutionEngineResult,
        start_time: float,
    ) -> GuardrailEngineResult:
        """Fail-closed response when the guardrail engine itself errors."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return GuardrailEngineResult(
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
            payment_id=engine_result.payment_id,
            merchant_id=engine_result.merchant_id,
            decision=AutomationDecision.UNRESOLVED,
            confidence=engine_result.confidence,
            risk_category=engine_result.risk_category,
            financial_exposure_paise=0,
            evidence_coverage=engine_result.evidence_coverage,
            evidence_consistency=engine_result.evidence_consistency,
            is_novel=False,
            has_conflict=False,
            verification_possible=False,
            passed_gates=[],
            failed_gates=[],
            reason_codes=[],
            primary_reason="Guardrail engine encountered an unexpected error — failing closed",
            system_healthy=False,
            critical_failures=["guardrail_engine"],
            guardrail_version="1.0.0",
            processing_time_ms=elapsed_ms,
        )
