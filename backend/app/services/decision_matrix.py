"""
Automation Decision Matrix for Razorpay CloseLoop Phase 6E.

Evaluates all safety signals and determines the final automation decision:
AUTO, HUMAN_REVIEW, or UNRESOLVED.

Priority order:
1. CRITICAL BLOCK → UNRESOLVED
2. HARD SAFETY FAILURE → HUMAN_REVIEW
3. ALL AUTO CONDITIONS PASS → AUTO

No hidden overrides allowed.
ML confidence, historical similarity, LLM output, and agent output
must NEVER bypass hard guardrails.
"""

from typing import List, Optional

from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.decision_matrix import (
    AutomationDecision,
    AutomationDecisionResult,
    DecisionConfig,
    GateResult,
    GateStatus,
    ReasonCode,
)
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.failure_fallback import FailureFallbackResult
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DECISION_CONFIG = DecisionConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Decision Matrix
# ─────────────────────────────────────────────────────────────────────────────


class AutomationDecisionMatrix:
    """
    Final safety decision layer that evaluates all guard results
    and produces an automation decision.

    Priority order:
    1. CRITICAL BLOCK → UNRESOLVED
    2. HARD SAFETY FAILURE → HUMAN_REVIEW
    3. ALL AUTO CONDITIONS PASS → AUTO

    No hidden overrides allowed.
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        """Initialize the decision matrix.

        Args:
            config: Decision configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_DECISION_CONFIG

    def evaluate(
        self,
        engine_result: ResolutionEngineResult,
        gate_result: Optional[ConfidenceGateResult] = None,
        exposure_result: Optional[ExposureGuardResult] = None,
        evidence_result: Optional[EvidenceGuardResult] = None,
        fallback_result: Optional[FailureFallbackResult] = None,
    ) -> AutomationDecisionResult:
        """Evaluate all safety signals and produce a final decision.

        Args:
            engine_result: Phase 5 resolution engine output
            gate_result: Confidence gate result
            exposure_result: Exposure guard result
            evidence_result: Evidence guard result
            fallback_result: System failure fallback result

        Returns:
            AutomationDecisionResult with final decision and audit trail
        """
        passed_gates: List[GateResult] = []
        failed_gates: List[GateResult] = []
        reason_codes: List[ReasonCode] = []
        primary_reason = ""

        # Extract values from engine result
        confidence = engine_result.confidence
        risk = engine_result.risk_category
        evidence_coverage = engine_result.evidence_coverage
        evidence_consistency = engine_result.evidence_consistency

        # Extract from guard results where available
        exposure_paise = 0
        if exposure_result:
            exposure_paise = exposure_result.adjustment_amount_paise

        is_novel = False
        has_conflict = False
        if evidence_result:
            is_novel = evidence_result.is_novel
            has_conflict = evidence_result.has_conflict

        # System health
        system_healthy = True
        critical_failures: List[str] = []
        if fallback_result:
            system_healthy = fallback_result.can_proceed
            critical_failures = [
                f.dependency_name for f in fallback_result.critical_failures
            ]

        verification_possible = True

        # ── PRIORITY 1: CRITICAL BLOCK → UNRESOLVED ──

        # Engine already deferred
        if engine_result.status in (
            SelectionStatus.UNRESOLVED,
            SelectionStatus.HUMAN_REVIEW,
        ):
            reason_codes.append(ReasonCode.ENGINE_DEFERRED)
            if not primary_reason:
                primary_reason = (
                    f"Engine already deferred to {engine_result.status.value}"
                )

        # Critical dependency failure
        if not system_healthy:
            reason_codes.append(ReasonCode.CRITICAL_DEP_FAILURE)
            if not primary_reason:
                primary_reason = (
                    f"Critical dependency failure: {', '.join(critical_failures)}"
                )

        # Unknown pattern
        if engine_result.deterministic_exception_type == "UNKNOWN":
            reason_codes.append(ReasonCode.UNKNOWN_PATTERN)
            if not primary_reason:
                primary_reason = "Unknown exception pattern — cannot auto-resolve"

        # Blocked exception type
        blocked_exc = ["UNKNOWN", "COMPLEX_MULTI_ADJUSTMENT", "MISSING_RECORD"]
        if engine_result.deterministic_exception_type in blocked_exc:
            reason_codes.append(ReasonCode.BLOCKED_EXCEPTION_TYPE)
            if not primary_reason:
                primary_reason = (
                    f"Exception type {engine_result.deterministic_exception_type} "
                    f"is blocked from auto-resolution"
                )

        # Blocked resolution type
        blocked_res = [
            "UNKNOWN_UNRESOLVED",
            "MISSING_RECORD_ESCALATION",
            "MULTI_ADJUSTMENT",
        ]
        if engine_result.selected_resolution in blocked_res:
            reason_codes.append(ReasonCode.BLOCKED_RESOLUTION_TYPE)
            if not primary_reason:
                primary_reason = (
                    f"Resolution type {engine_result.selected_resolution} "
                    f"is blocked from auto-resolution"
                )

        # Very low confidence
        if confidence < self.config.min_confidence_for_human:
            reason_codes.append(ReasonCode.VERY_LOW_CONFIDENCE)
            if not primary_reason:
                primary_reason = (
                    f"Confidence {confidence:.1%} below minimum "
                    f"{self.config.min_confidence_for_human:.1%}"
                )

        # High exposure
        if exposure_paise > self.config.max_exposure_for_human:
            reason_codes.append(ReasonCode.HIGH_EXPOSURE)
            if not primary_reason:
                primary_reason = (
                    f"Financial exposure {exposure_paise} paise exceeds "
                    f"maximum {self.config.max_exposure_for_human} paise"
                )

        # Check for UNRESOLVED triggers
        unresolved_triggers = {
            ReasonCode.ENGINE_DEFERRED,
            ReasonCode.CRITICAL_DEP_FAILURE,
            ReasonCode.UNKNOWN_PATTERN,
            ReasonCode.BLOCKED_EXCEPTION_TYPE,
            ReasonCode.BLOCKED_RESOLUTION_TYPE,
            ReasonCode.VERY_LOW_CONFIDENCE,
            ReasonCode.HIGH_EXPOSURE,
        }
        has_unresolved_trigger = bool(unresolved_triggers & set(reason_codes))

        if has_unresolved_trigger:
            return self._build_result(
                decision=AutomationDecision.UNRESOLVED,
                reason_codes=reason_codes,
                primary_reason=primary_reason,
                confidence=confidence,
                risk=risk,
                exposure_paise=exposure_paise,
                evidence_coverage=evidence_coverage,
                evidence_consistency=evidence_consistency,
                is_novel=is_novel,
                has_conflict=has_conflict,
                verification_possible=verification_possible,
                system_healthy=system_healthy,
                critical_failures=critical_failures,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                engine_result=engine_result,
            )

        # ── PRIORITY 2: HARD SAFETY FAILURE → HUMAN_REVIEW ──

        human_reasons: List[ReasonCode] = []

        # Confidence gate failed
        if gate_result and gate_result.action == GateAction.HUMAN_REVIEW:
            human_reasons.append(ReasonCode.MEDIUM_CONFIDENCE)
            failed_gates.append(GateResult(
                gate_name="confidence_gate",
                status=GateStatus.FAILED,
                value=f"{confidence:.1%}",
                threshold=f"{self.config.min_confidence_for_auto:.1%}",
                reason_code=ReasonCode.MEDIUM_CONFIDENCE,
                description="Confidence gate blocked",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="confidence_gate",
                status=GateStatus.PASSED,
                value=f"{confidence:.1%}",
                description="Confidence gate passed",
            ))

        # Exposure guard failed
        if exposure_result and exposure_result.action == ExposureAction.BLOCK:
            if exposure_paise > self.config.max_exposure_for_auto:
                human_reasons.append(ReasonCode.MODERATE_EXPOSURE)
            failed_gates.append(GateResult(
                gate_name="exposure_guard",
                status=GateStatus.FAILED,
                value=f"{exposure_paise}",
                threshold=f"{self.config.max_exposure_for_auto}",
                reason_code=ReasonCode.MODERATE_EXPOSURE,
                description="Exposure guard blocked",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="exposure_guard",
                status=GateStatus.PASSED,
                value=f"{exposure_paise}",
                description="Exposure guard passed",
            ))

        # Evidence guard failed
        evidence_blocked = (
            evidence_result and evidence_result.action == EvidenceAction.BLOCK
        )
        if evidence_blocked:
            failed_gates.append(GateResult(
                gate_name="evidence_guard",
                status=GateStatus.FAILED,
                value=f"coverage={evidence_coverage:.1%}, consistency={evidence_consistency:.1%}",
                description="Evidence guard blocked",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="evidence_guard",
                status=GateStatus.PASSED,
                value=f"coverage={evidence_coverage:.1%}, consistency={evidence_consistency:.1%}",
                description="Evidence guard passed",
            ))

        # Conflict check (independent of evidence guard)
        if has_conflict:
            human_reasons.append(ReasonCode.CONFLICTING_EVIDENCE)
            failed_gates.append(GateResult(
                gate_name="conflict_check",
                status=GateStatus.FAILED,
                reason_code=ReasonCode.CONFLICTING_EVIDENCE,
                description="Conflicting evidence detected",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="conflict_check",
                status=GateStatus.PASSED,
                description="No conflicting evidence",
            ))

        # Novelty check (independent of evidence guard)
        if is_novel:
            human_reasons.append(ReasonCode.NOVEL_PATTERN)
            failed_gates.append(GateResult(
                gate_name="novelty_check",
                status=GateStatus.FAILED,
                reason_code=ReasonCode.NOVEL_PATTERN,
                description="Novel pattern detected",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="novelty_check",
                status=GateStatus.PASSED,
                description="Known pattern",
            ))

        # Coverage and consistency (from evidence guard or direct check)
        if evidence_coverage < self.config.min_evidence_coverage_for_auto:
            if ReasonCode.LOW_COVERAGE not in human_reasons:
                human_reasons.append(ReasonCode.LOW_COVERAGE)
        if evidence_consistency < self.config.min_evidence_consistency_for_auto:
            if ReasonCode.LOW_CONSISTENCY not in human_reasons:
                human_reasons.append(ReasonCode.LOW_CONSISTENCY)

        # Fallback guard (optional dep failures)
        if fallback_result and not fallback_result.can_proceed:
            # Already handled in CRITICAL block above
            pass
        elif fallback_result and not fallback_result.can_use_deterministic_only:
            # All deps healthy
            passed_gates.append(GateResult(
                gate_name="fallback_guard",
                status=GateStatus.PASSED,
                description="All dependencies healthy",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="fallback_guard",
                status=GateStatus.PASSED,
                description="Fallback guard passed",
            ))

        # Risk check
        if risk not in self.config.allowed_risk_for_human:
            human_reasons.append(ReasonCode.ELEVATED_RISK)
            failed_gates.append(GateResult(
                gate_name="risk_check",
                status=GateStatus.FAILED,
                value=risk,
                threshold=str(self.config.allowed_risk_for_human),
                reason_code=ReasonCode.ELEVATED_RISK,
                description="Risk level too high for human review",
            ))
        else:
            passed_gates.append(GateResult(
                gate_name="risk_check",
                status=GateStatus.PASSED,
                value=risk,
                description="Risk check passed",
            ))

        if human_reasons:
            reason_codes.extend(human_reasons)
            if not primary_reason:
                primary_reason = "; ".join(
                    rc.value.replace("_", " ").lower() for rc in human_reasons[:2]
                )

            return self._build_result(
                decision=AutomationDecision.HUMAN_REVIEW,
                reason_codes=reason_codes,
                primary_reason=primary_reason,
                confidence=confidence,
                risk=risk,
                exposure_paise=exposure_paise,
                evidence_coverage=evidence_coverage,
                evidence_consistency=evidence_consistency,
                is_novel=is_novel,
                has_conflict=has_conflict,
                verification_possible=verification_possible,
                system_healthy=system_healthy,
                critical_failures=critical_failures,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                engine_result=engine_result,
            )

        # ── PRIORITY 3: ALL AUTO CONDITIONS PASS → AUTO ──

        # Additional AUTO checks
        auto_checks = [
            (
                "confidence",
                confidence >= self.config.min_confidence_for_auto,
                ReasonCode.MEDIUM_CONFIDENCE,
            ),
            (
                "exposure",
                exposure_paise <= self.config.max_exposure_for_auto,
                ReasonCode.MODERATE_EXPOSURE,
            ),
            (
                "coverage",
                evidence_coverage >= self.config.min_evidence_coverage_for_auto,
                ReasonCode.LOW_COVERAGE,
            ),
            (
                "consistency",
                evidence_consistency >= self.config.min_evidence_consistency_for_auto,
                ReasonCode.LOW_CONSISTENCY,
            ),
            (
                "risk",
                risk in self.config.allowed_risk_for_auto,
                ReasonCode.ELEVATED_RISK,
            ),
        ]

        for check_name, passed, fail_code in auto_checks:
            if passed:
                passed_gates.append(GateResult(
                    gate_name=f"auto_{check_name}",
                    status=GateStatus.PASSED,
                    description=f"Auto {check_name} check passed",
                ))
            else:
                failed_gates.append(GateResult(
                    gate_name=f"auto_{check_name}",
                    status=GateStatus.FAILED,
                    reason_code=fail_code,
                    description=f"Auto {check_name} check failed",
                ))
                human_reasons.append(fail_code)

        if human_reasons:
            reason_codes.extend(human_reasons)
            if not primary_reason:
                primary_reason = "; ".join(
                    rc.value.replace("_", " ").lower() for rc in human_reasons[:2]
                )
            return self._build_result(
                decision=AutomationDecision.HUMAN_REVIEW,
                reason_codes=reason_codes,
                primary_reason=primary_reason,
                confidence=confidence,
                risk=risk,
                exposure_paise=exposure_paise,
                evidence_coverage=evidence_coverage,
                evidence_consistency=evidence_consistency,
                is_novel=is_novel,
                has_conflict=has_conflict,
                verification_possible=verification_possible,
                system_healthy=system_healthy,
                critical_failures=critical_failures,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                engine_result=engine_result,
            )

        # ALL PASSED → AUTO
        reason_codes.append(ReasonCode.ALL_GATES_PASSED)
        primary_reason = (
            "All mandatory confidence, exposure, evidence and "
            "verification gates passed."
        )

        return self._build_result(
            decision=AutomationDecision.AUTO,
            reason_codes=reason_codes,
            primary_reason=primary_reason,
            confidence=confidence,
            risk=risk,
            exposure_paise=exposure_paise,
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            is_novel=is_novel,
            has_conflict=has_conflict,
            verification_possible=verification_possible,
            system_healthy=system_healthy,
            critical_failures=critical_failures,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            engine_result=engine_result,
        )

    def _build_result(
        self,
        decision,
        reason_codes,
        primary_reason,
        confidence,
        risk,
        exposure_paise,
        evidence_coverage,
        evidence_consistency,
        is_novel,
        has_conflict,
        verification_possible,
        system_healthy,
        critical_failures,
        passed_gates,
        failed_gates,
        engine_result,
    ):
        return AutomationDecisionResult(
            decision=decision,
            reason_codes=reason_codes,
            primary_reason=primary_reason,
            confidence=confidence,
            risk_category=risk,
            financial_exposure_paise=exposure_paise,
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            is_novel=is_novel,
            has_conflict=has_conflict,
            verification_possible=verification_possible,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            system_healthy=system_healthy,
            critical_failures=critical_failures,
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
        )
