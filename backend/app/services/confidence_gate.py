"""
Confidence Gate Service for Razorpay CloseLoop Phase 6A.

Implements a configurable safety gate that evaluates whether a
Phase 5 resolution recommendation is safe enough for automated processing.

This is a SAFETY GATE.

It does NOT:
- generate financial resolutions
- modify financial records
- execute financial actions
- override Phase 2 reconciliation
- override Phase 5 recommendations

It ONLY decides: CONTINUE or HUMAN_REVIEW.
"""

from typing import Optional

from app.schemas.candidate_scoring import CandidateScore
from app.schemas.confidence_gate import (
    ConfidenceGateConfig,
    ConfidenceGateResult,
    GateAction,
    GateCheck,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GATE_CONFIG = ConfidenceGateConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Gate
# ─────────────────────────────────────────────────────────────────────────────


class ConfidenceGate:
    """
    Safety gate that evaluates whether a resolution recommendation
    is safe enough for automated processing.

    Receives a Phase 5 ResolutionEngineResult and applies
    configurable safety checks.

    This is a SAFETY GATE only.
    It must NOT execute financial actions.
    """

    def __init__(self, config: Optional[ConfidenceGateConfig] = None):
        """Initialize the confidence gate.

        Args:
            config: Gate configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_GATE_CONFIG

    def evaluate(
        self,
        engine_result: ResolutionEngineResult,
    ) -> ConfidenceGateResult:
        """Evaluate a resolution engine result against gate thresholds.

        Args:
            engine_result: The Phase 5 resolution engine output

        Returns:
            ConfidenceGateResult with pass/fail decision and detailed checks
        """
        checks: list[GateCheck] = []
        blocked = False
        primary_reason = ""
        blocked_by_high_value = False
        blocked_by_risk = False
        blocked_by_blocked_type = False

        # If engine already resolved to UNRESOLVED, gate is irrelevant
        if engine_result.status == SelectionStatus.UNRESOLVED:
            return ConfidenceGateResult(
                passed=False,
                action=GateAction.HUMAN_REVIEW,
                confidence=engine_result.confidence,
                threshold=self.config.min_confidence,
                reason="Resolution engine returned UNRESOLVED — no candidate to evaluate",
                checks=[],
                adjustment_amount_paise=None,
                exception_id=engine_result.exception_id,
                case_id=engine_result.case_id,
            )

        # If engine returned HUMAN_REVIEW, gate is irrelevant
        if engine_result.status == SelectionStatus.HUMAN_REVIEW:
            return ConfidenceGateResult(
                passed=False,
                action=GateAction.HUMAN_REVIEW,
                confidence=engine_result.confidence,
                threshold=self.config.min_confidence,
                reason="Resolution engine already returned HUMAN_REVIEW",
                checks=[],
                adjustment_amount_paise=None,
                exception_id=engine_result.exception_id,
                case_id=engine_result.case_id,
            )

        # ── Now evaluate the RECOMMENDED candidate ──

        confidence = engine_result.confidence
        adjustment_paise = None
        if engine_result.selected_candidate:
            adjustment_paise = engine_result.selected_candidate.financial_adjustment.amount_paise

        # 1. Confidence threshold check
        confidence_check = GateCheck(
            check_name="confidence_threshold",
            passed=confidence >= self.config.min_confidence,
            value=confidence,
            threshold=self.config.min_confidence,
            reason=(
                f"Confidence {confidence:.1%} {'>=' if confidence >= self.config.min_confidence else '<'} "
                f"threshold {self.config.min_confidence:.1%}"
            ),
        )
        checks.append(confidence_check)
        if not confidence_check.passed and not blocked:
            blocked = True
            primary_reason = confidence_check.reason

        # 2. Financial consistency check (from score)
        if engine_result.selected_score:
            fin_check = GateCheck(
                check_name="financial_consistency",
                passed=(
                    engine_result.selected_score.financial_consistency_score
                    >= self.config.min_financial_consistency
                ),
                value=engine_result.selected_score.financial_consistency_score,
                threshold=self.config.min_financial_consistency,
                reason=(
                    f"Financial consistency "
                    f"{engine_result.selected_score.financial_consistency_score:.1%} "
                    f"{'>=' if engine_result.selected_score.financial_consistency_score >= self.config.min_financial_consistency else '<'} "
                    f"threshold {self.config.min_financial_consistency:.1%}"
                ),
            )
            checks.append(fin_check)
            if not fin_check.passed and not blocked:
                blocked = True
                primary_reason = fin_check.reason

        # 3. Evidence coverage check
        evidence_coverage = engine_result.evidence_coverage
        evidence_check = GateCheck(
            check_name="evidence_coverage",
            passed=evidence_coverage >= self.config.min_evidence_coverage,
            value=evidence_coverage,
            threshold=self.config.min_evidence_coverage,
            reason=(
                f"Evidence coverage {evidence_coverage:.1%} "
                f"{'>=' if evidence_coverage >= self.config.min_evidence_coverage else '<'} "
                f"threshold {self.config.min_evidence_coverage:.1%}"
            ),
        )
        checks.append(evidence_check)
        if not evidence_check.passed and not blocked:
            blocked = True
            primary_reason = evidence_check.reason

        # 4. Risk level check
        risk_allowed = engine_result.risk_category in self.config.allowed_risk_levels
        risk_check = GateCheck(
            check_name="risk_level",
            passed=risk_allowed,
            value=None,
            threshold=None,
            reason=(
                f"Risk level {engine_result.risk_category} "
                f"{'is' if risk_allowed else 'is not'} in allowed levels {self.config.allowed_risk_levels}"
            ),
        )
        checks.append(risk_check)
        if not risk_check.passed and not blocked:
            blocked = True
            blocked_by_risk = True
            primary_reason = risk_check.reason

        # 5. High-value adjustment check
        if adjustment_paise is not None and adjustment_paise > 0:
            high_value_blocked = adjustment_paise >= self.config.high_value_threshold_paise
            hv_check = GateCheck(
                check_name="high_value_adjustment",
                passed=not high_value_blocked,
                value=float(adjustment_paise),
                threshold=float(self.config.high_value_threshold_paise),
                reason=(
                    f"Adjustment {adjustment_paise} paise "
                    f"{'exceeds' if high_value_blocked else 'within'} "
                    f"high-value threshold {self.config.high_value_threshold_paise} paise"
                ),
            )
            checks.append(hv_check)
            if high_value_blocked and not blocked:
                blocked = True
                blocked_by_high_value = True
                primary_reason = hv_check.reason

        # 6. Conflict penalty check
        if engine_result.selected_score:
            conflict_check = GateCheck(
                check_name="conflict_penalty",
                passed=(
                    engine_result.selected_score.conflict_penalty
                    <= self.config.max_conflict_penalty
                ),
                value=engine_result.selected_score.conflict_penalty,
                threshold=self.config.max_conflict_penalty,
                reason=(
                    f"Conflict penalty {engine_result.selected_score.conflict_penalty:.1%} "
                    f"{'<=' if engine_result.selected_score.conflict_penalty <= self.config.max_conflict_penalty else '>'} "
                    f"threshold {self.config.max_conflict_penalty:.1%}"
                ),
            )
            checks.append(conflict_check)
            if not conflict_check.passed and not blocked:
                blocked = True
                primary_reason = conflict_check.reason

        # 7. Novelty penalty check
        if engine_result.selected_score:
            novelty_check = GateCheck(
                check_name="novelty_penalty",
                passed=(
                    engine_result.selected_score.novelty_penalty
                    <= self.config.max_novelty_penalty
                ),
                value=engine_result.selected_score.novelty_penalty,
                threshold=self.config.max_novelty_penalty,
                reason=(
                    f"Novelty penalty {engine_result.selected_score.novelty_penalty:.1%} "
                    f"{'<=' if engine_result.selected_score.novelty_penalty <= self.config.max_novelty_penalty else '>'} "
                    f"threshold {self.config.max_novelty_penalty:.1%}"
                ),
            )
            checks.append(novelty_check)
            if not novelty_check.passed and not blocked:
                blocked = True
                primary_reason = novelty_check.reason

        # 8. Blocked resolution type check
        if engine_result.selected_resolution:
            type_blocked = engine_result.selected_resolution in self.config.blocked_resolution_types
            type_check = GateCheck(
                check_name="blocked_resolution_type",
                passed=not type_blocked,
                value=None,
                threshold=None,
                reason=(
                    f"Resolution type {engine_result.selected_resolution} "
                    f"{'is' if type_blocked else 'is not'} in blocked list"
                ),
            )
            checks.append(type_check)
            if type_blocked and not blocked:
                blocked = True
                blocked_by_blocked_type = True
                primary_reason = type_check.reason

        # 9. Supporting evidence count check
        if engine_result.selected_candidate:
            evidence_count = len(engine_result.selected_candidate.supporting_evidence_ids)
            ev_count_check = GateCheck(
                check_name="supporting_evidence_count",
                passed=evidence_count >= self.config.min_supporting_evidence,
                value=float(evidence_count),
                threshold=float(self.config.min_supporting_evidence),
                reason=(
                    f"Supporting evidence count {evidence_count} "
                    f"{'>=' if evidence_count >= self.config.min_supporting_evidence else '<'} "
                    f"minimum {self.config.min_supporting_evidence}"
                ),
            )
            checks.append(ev_count_check)
            if not ev_count_check.passed and not blocked:
                blocked = True
                primary_reason = ev_count_check.reason

        # Determine final action
        if blocked:
            action = GateAction.HUMAN_REVIEW
        else:
            action = GateAction.CONTINUE

        # If no primary reason was set (all checks passed), generate one
        if not primary_reason:
            primary_reason = (
                f"All gate checks passed: confidence {confidence:.1%} "
                f">= {self.config.min_confidence:.1%}"
            )

        return ConfidenceGateResult(
            passed=not blocked,
            action=action,
            confidence=confidence,
            threshold=self.config.min_confidence,
            reason=primary_reason,
            checks=checks,
            adjustment_amount_paise=adjustment_paise,
            blocked_by_high_value=blocked_by_high_value,
            blocked_by_risk=blocked_by_risk,
            blocked_by_blocked_type=blocked_by_blocked_type,
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
        )
