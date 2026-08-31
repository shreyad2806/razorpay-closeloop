"""
Financial Exposure Guard Service for Razorpay CloseLoop Phase 6B.

Implements a hard safety gate that prevents high-value or financially
risky resolutions from being automatically recommended for execution.

This is a HARD SAFETY GATE.

It does NOT:
- execute financial actions
- modify financial records
- generate resolutions
- override reconciliation
- override confidence gate

Exposure is calculated from the ACTUAL proposed financial adjustment.
ML confidence does NOT determine exposure.
"""

from typing import List, Optional

from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.exposure_guard import (
    ExposureAction,
    ExposureBlockReason,
    ExposureCheck,
    ExposureGuardConfig,
    ExposureGuardResult,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EXPOSURE_CONFIG = ExposureGuardConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Exposure Guard
# ─────────────────────────────────────────────────────────────────────────────


class ExposureGuard:
    """
    Hard safety gate that evaluates whether a proposed financial adjustment
    is safe enough for automated processing based on financial exposure.

    Exposure is calculated from the actual proposed financial adjustment amount.
    ML confidence does NOT determine exposure.

    This is a HARD SAFETY GATE.
    It cannot be overridden by confidence or ML scores.
    """

    def __init__(self, config: Optional[ExposureGuardConfig] = None):
        """Initialize the exposure guard.

        Args:
            config: Guard configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_EXPOSURE_CONFIG

    def evaluate(
        self,
        engine_result: ResolutionEngineResult,
        gate_result: Optional[ConfidenceGateResult] = None,
    ) -> ExposureGuardResult:
        """Evaluate financial exposure for a resolution engine result.

        Args:
            engine_result: The Phase 5 resolution engine output
            gate_result: Optional confidence gate result for context

        Returns:
            ExposureGuardResult with pass/block decision and detailed checks
        """
        checks: List[ExposureCheck] = []
        block_reasons: List[ExposureBlockReason] = []
        blocked = False
        primary_reason = ""

        # ── Determine adjustment amount ──

        adjustment_paise = 0
        if engine_result.selected_candidate:
            adjustment_paise = abs(
                engine_result.selected_candidate.financial_adjustment.amount_paise
            )

        # Calculate cumulative exposure from all candidates
        cumulative_paise = sum(
            abs(c.financial_adjustment.amount_paise)
            for c in engine_result.ranked_candidates
        )

        # ── If engine returned UNRESOLVED/HUMAN_REVIEW, no adjustment to evaluate ──

        if engine_result.status in (
            SelectionStatus.UNRESOLVED,
            SelectionStatus.HUMAN_REVIEW,
        ):
            return ExposureGuardResult(
                passed=True,  # Nothing to block — no adjustment proposed
                action=ExposureAction.PASS,
                adjustment_amount_paise=0,
                max_auto_resolution_paise=self.config.max_auto_resolution_paise,
                cumulative_exposure_paise=cumulative_paise,
                reason=(
                    f"No adjustment to evaluate — engine status: {engine_result.status.value}"
                ),
                checks=[],
                block_reasons=[],
                is_high_value=False,
                exception_id=engine_result.exception_id,
                case_id=engine_result.case_id,
            )

        # ── Hard Block Checks ──

        # 1. Maximum auto-resolution amount (HARD BLOCK)
        max_blocked = adjustment_paise > self.config.max_auto_resolution_paise
        max_check = ExposureCheck(
            check_name="max_auto_resolution",
            passed=not max_blocked,
            value=float(adjustment_paise),
            threshold=float(self.config.max_auto_resolution_paise),
            reason=(
                f"Adjustment {adjustment_paise} paise "
                f"{'exceeds' if max_blocked else 'within'} "
                f"max auto-resolution {self.config.max_auto_resolution_paise} paise"
            ),
            block_reason=ExposureBlockReason.ABOVE_MAX_AMOUNT if max_blocked else None,
        )
        checks.append(max_check)
        if max_blocked:
            blocked = True
            block_reasons.append(ExposureBlockReason.ABOVE_MAX_AMOUNT)
            if not primary_reason:
                primary_reason = max_check.reason

        # 2. High-value threshold (informational + escalation)
        is_high_value = adjustment_paise >= self.config.high_value_threshold_paise
        hv_check = ExposureCheck(
            check_name="high_value_threshold",
            passed=True,  # Informational — does not block by itself
            value=float(adjustment_paise),
            threshold=float(self.config.high_value_threshold_paise),
            reason=(
                f"Adjustment {adjustment_paise} paise "
                f"{'exceeds' if is_high_value else 'within'} "
                f"high-value threshold {self.config.high_value_threshold_paise} paise"
            ),
        )
        checks.append(hv_check)

        # 3. Cumulative exposure limit (HARD BLOCK)
        cumulative_blocked = (
            cumulative_paise > self.config.cumulative_exposure_limit_paise
        )
        cum_check = ExposureCheck(
            check_name="cumulative_exposure",
            passed=not cumulative_blocked,
            value=float(cumulative_paise),
            threshold=float(self.config.cumulative_exposure_limit_paise),
            reason=(
                f"Cumulative exposure {cumulative_paise} paise "
                f"{'exceeds' if cumulative_blocked else 'within'} "
                f"limit {self.config.cumulative_exposure_limit_paise} paise"
            ),
            block_reason=(
                ExposureBlockReason.ABOVE_MAX_AMOUNT
                if cumulative_blocked
                else None
            ),
        )
        checks.append(cum_check)
        if cumulative_blocked:
            blocked = True
            block_reasons.append(ExposureBlockReason.ABOVE_MAX_AMOUNT)
            if not primary_reason:
                primary_reason = cum_check.reason

        # 4. High-risk exception type (HARD BLOCK)
        exc_type_blocked = (
            engine_result.deterministic_exception_type
            in self.config.blocked_exception_types
        )
        exc_check = ExposureCheck(
            check_name="blocked_exception_type",
            passed=not exc_type_blocked,
            value=None,
            threshold=None,
            reason=(
                f"Exception type {engine_result.deterministic_exception_type} "
                f"{'is' if exc_type_blocked else 'is not'} in blocked list "
                f"{self.config.blocked_exception_types}"
            ),
            block_reason=(
                ExposureBlockReason.HIGH_RISK_CATEGORY
                if exc_type_blocked
                else None
            ),
        )
        checks.append(exc_check)
        if exc_type_blocked:
            blocked = True
            block_reasons.append(ExposureBlockReason.HIGH_RISK_CATEGORY)
            if not primary_reason:
                primary_reason = exc_check.reason

        # 5. High-risk resolution type (HARD BLOCK)
        res_type = engine_result.selected_resolution or ""
        res_type_blocked = res_type in self.config.blocked_resolution_types
        res_check = ExposureCheck(
            check_name="blocked_resolution_type",
            passed=not res_type_blocked,
            value=None,
            threshold=None,
            reason=(
                f"Resolution type {res_type} "
                f"{'is' if res_type_blocked else 'is not'} in blocked list "
                f"{self.config.blocked_resolution_types}"
            ),
            block_reason=(
                ExposureBlockReason.HIGH_RISK_CATEGORY
                if res_type_blocked
                else None
            ),
        )
        checks.append(res_check)
        if res_type_blocked:
            blocked = True
            block_reasons.append(ExposureBlockReason.HIGH_RISK_CATEGORY)
            if not primary_reason:
                primary_reason = res_check.reason

        # 6. Conflict penalty check (HARD BLOCK)
        if engine_result.selected_score:
            conflict_blocked = (
                engine_result.selected_score.conflict_penalty
                > self.config.max_conflict_for_auto
            )
            conflict_check = ExposureCheck(
                check_name="conflict_penalty",
                passed=not conflict_blocked,
                value=engine_result.selected_score.conflict_penalty,
                threshold=self.config.max_conflict_for_auto,
                reason=(
                    f"Conflict penalty {engine_result.selected_score.conflict_penalty:.1%} "
                    f"{'exceeds' if conflict_blocked else 'within'} "
                    f"threshold {self.config.max_conflict_for_auto:.1%}"
                ),
                block_reason=(
                    ExposureBlockReason.CONFLICTING_CASE
                    if conflict_blocked
                    else None
                ),
            )
            checks.append(conflict_check)
            if conflict_blocked:
                blocked = True
                block_reasons.append(ExposureBlockReason.CONFLICTING_CASE)
                if not primary_reason:
                    primary_reason = conflict_check.reason

        # 7. Supporting evidence count (HARD BLOCK)
        if engine_result.selected_candidate:
            evidence_count = len(
                engine_result.selected_candidate.supporting_evidence_ids
            )
            evidence_blocked = (
                evidence_count < self.config.min_evidence_for_auto
            )
            ev_check = ExposureCheck(
                check_name="insufficient_evidence",
                passed=not evidence_blocked,
                value=float(evidence_count),
                threshold=float(self.config.min_evidence_for_auto),
                reason=(
                    f"Supporting evidence count {evidence_count} "
                    f"{'<' if evidence_blocked else '>='} "
                    f"minimum {self.config.min_evidence_for_auto}"
                ),
                block_reason=(
                    ExposureBlockReason.NO_EXPOSURE_DATA
                    if evidence_blocked
                    else None
                ),
            )
            checks.append(ev_check)
            if evidence_blocked:
                blocked = True
                block_reasons.append(ExposureBlockReason.NO_EXPOSURE_DATA)
                if not primary_reason:
                    primary_reason = ev_check.reason

        # ── Determine final action ──

        if blocked:
            action = ExposureAction.BLOCK
        else:
            action = ExposureAction.PASS

        if not primary_reason:
            primary_reason = (
                f"Adjustment {adjustment_paise} paise within "
                f"limit {self.config.max_auto_resolution_paise} paise. "
                f"All exposure checks passed."
            )

        return ExposureGuardResult(
            passed=not blocked,
            action=action,
            adjustment_amount_paise=adjustment_paise,
            max_auto_resolution_paise=self.config.max_auto_resolution_paise,
            cumulative_exposure_paise=cumulative_paise,
            reason=primary_reason,
            checks=checks,
            block_reasons=block_reasons,
            is_high_value=is_high_value,
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
        )
