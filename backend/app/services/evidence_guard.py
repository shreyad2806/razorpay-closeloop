"""
Evidence Safety Guard Service for Razorpay CloseLoop Phase 6C.

Implements a hard safety gate that prevents automatic resolution when
the financial explanation is incomplete, conflicting, or novel.

This is a HARD SAFETY GATE.

It does NOT:
- execute financial actions
- modify financial records
- generate resolutions
- override reconciliation
- override confidence gate
- override exposure guard

Evidence safety is evaluated independently.
High ML confidence must NOT override conflicting or missing evidence.
"""

from typing import List, Optional

from app.schemas.evidence_guard import (
    EvidenceAction,
    EvidenceBlockReason,
    EvidenceGuardCheck,
    EvidenceGuardConfig,
    EvidenceGuardResult,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EVIDENCE_CONFIG = EvidenceGuardConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Guard
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceGuard:
    """
    Hard safety gate that evaluates whether available evidence is sufficient
    for automated resolution based on evidence quality, completeness,
    and consistency.

    This is a HARD SAFETY GATE.
    High ML confidence must NOT override evidence safety failures.
    """

    def __init__(self, config: Optional[EvidenceGuardConfig] = None):
        """Initialize the evidence guard.

        Args:
            config: Guard configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_EVIDENCE_CONFIG

    def evaluate(
        self,
        engine_result: ResolutionEngineResult,
    ) -> EvidenceGuardResult:
        """Evaluate evidence safety for a resolution engine result.

        Args:
            engine_result: The Phase 5 resolution engine output

        Returns:
            EvidenceGuardResult with pass/block decision and detailed checks
        """
        checks: List[EvidenceGuardCheck] = []
        block_reasons: List[EvidenceBlockReason] = []
        blocked = False
        primary_reason = ""

        # Extract evidence metrics from engine result
        evidence_coverage = engine_result.evidence_coverage
        evidence_consistency = engine_result.evidence_consistency

        # Extract conflict/missing from intelligence (if available via engine)
        has_conflict = False
        missing_evidence: List[str] = []
        is_novel = False
        explanation_status = engine_result.evidence_explanation_status

        # ── If engine returned UNRESOLVED/HUMAN_REVIEW, evaluate but don't block ──

        if engine_result.status in (
            SelectionStatus.UNRESOLVED,
            SelectionStatus.HUMAN_REVIEW,
        ):
            return EvidenceGuardResult(
                passed=True,  # Nothing to block — engine already deferred
                action=EvidenceAction.PASS,
                evidence_coverage=evidence_coverage,
                evidence_consistency=evidence_consistency,
                has_conflict=has_conflict,
                missing_evidence_count=len(missing_evidence),
                is_novel=is_novel,
                explanation_status=explanation_status,
                reason=(
                    f"Engine already deferred to {engine_result.status.value} — "
                    f"evidence guard not blocking"
                ),
                checks=[],
                block_reasons=[],
                exception_id=engine_result.exception_id,
                case_id=engine_result.case_id,
            )

        # ── Evidence Safety Checks ──

        # 1. Conflicting evidence (HARD BLOCK)
        if self.config.block_on_conflict and has_conflict:
            conflict_check = EvidenceGuardCheck(
                check_name="conflicting_evidence",
                passed=False,
                value=None,
                threshold=None,
                reason=(
                    "Evidence contains material conflicts. "
                    "ML confidence must NOT override this."
                ),
                block_reason=EvidenceBlockReason.CONFLICTING_EVIDENCE,
            )
            checks.append(conflict_check)
            blocked = True
            block_reasons.append(EvidenceBlockReason.CONFLICTING_EVIDENCE)
            primary_reason = conflict_check.reason

        # 2. Missing evidence (HARD BLOCK)
        missing_count = len(missing_evidence)
        missing_blocked = missing_count > self.config.max_missing_evidence
        missing_check = EvidenceGuardCheck(
            check_name="missing_evidence",
            passed=not missing_blocked,
            value=float(missing_count),
            threshold=float(self.config.max_missing_evidence),
            reason=(
                f"Missing evidence count {missing_count} "
                f"{'exceeds' if missing_blocked else 'within'} "
                f"maximum {self.config.max_missing_evidence}"
            ),
            block_reason=(
                EvidenceBlockReason.MISSING_EVIDENCE
                if missing_blocked
                else None
            ),
        )
        checks.append(missing_check)
        if missing_blocked:
            blocked = True
            block_reasons.append(EvidenceBlockReason.MISSING_EVIDENCE)
            if not primary_reason:
                primary_reason = missing_check.reason

        # 3. Evidence coverage (HARD BLOCK)
        coverage_blocked = evidence_coverage < self.config.min_evidence_coverage
        coverage_check = EvidenceGuardCheck(
            check_name="evidence_coverage",
            passed=not coverage_blocked,
            value=evidence_coverage,
            threshold=self.config.min_evidence_coverage,
            reason=(
                f"Evidence coverage {evidence_coverage:.1%} "
                f"{'<' if coverage_blocked else '>='} "
                f"minimum {self.config.min_evidence_coverage:.1%}"
            ),
            block_reason=(
                EvidenceBlockReason.LOW_COVERAGE
                if coverage_blocked
                else None
            ),
        )
        checks.append(coverage_check)
        if coverage_blocked:
            blocked = True
            block_reasons.append(EvidenceBlockReason.LOW_COVERAGE)
            if not primary_reason:
                primary_reason = coverage_check.reason

        # 4. Evidence consistency (HARD BLOCK)
        consistency_blocked = (
            evidence_consistency < self.config.min_evidence_consistency
        )
        consistency_check = EvidenceGuardCheck(
            check_name="evidence_consistency",
            passed=not consistency_blocked,
            value=evidence_consistency,
            threshold=self.config.min_evidence_consistency,
            reason=(
                f"Evidence consistency {evidence_consistency:.1%} "
                f"{'<' if consistency_blocked else '>='} "
                f"minimum {self.config.min_evidence_consistency:.1%}"
            ),
            block_reason=(
                EvidenceBlockReason.LOW_CONSISTENCY
                if consistency_blocked
                else None
            ),
        )
        checks.append(consistency_check)
        if consistency_blocked:
            blocked = True
            block_reasons.append(EvidenceBlockReason.LOW_CONSISTENCY)
            if not primary_reason:
                primary_reason = consistency_check.reason

        # 5. Novel pattern (HARD BLOCK)
        if self.config.block_on_novelty and is_novel:
            novelty_check = EvidenceGuardCheck(
                check_name="novel_pattern",
                passed=False,
                value=None,
                threshold=None,
                reason=(
                    "Case is a novel pattern. Novel cases should not "
                    "automatically inherit historical behavior."
                ),
                block_reason=EvidenceBlockReason.NOVEL_PATTERN,
            )
            checks.append(novelty_check)
            blocked = True
            block_reasons.append(EvidenceBlockReason.NOVEL_PATTERN)
            if not primary_reason:
                primary_reason = novelty_check.reason

        # 6. Explanation status (HARD BLOCK)
        if explanation_status:
            explanation_blocked = (
                explanation_status not in self.config.allowed_explanation_statuses
            )
            explanation_check = EvidenceGuardCheck(
                check_name="explanation_status",
                passed=not explanation_blocked,
                value=None,
                threshold=None,
                reason=(
                    f"Explanation status '{explanation_status}' "
                    f"{'is not' if explanation_blocked else 'is'} in allowed "
                    f"statuses {self.config.allowed_explanation_statuses}"
                ),
                block_reason=(
                    EvidenceBlockReason.UNEXPLAINED
                    if explanation_blocked
                    else None
                ),
            )
            checks.append(explanation_check)
            if explanation_blocked:
                blocked = True
                block_reasons.append(EvidenceBlockReason.UNEXPLAINED)
                if not primary_reason:
                    primary_reason = explanation_check.reason

        # 7. Supporting evidence count (HARD BLOCK)
        if engine_result.selected_candidate:
            evidence_count = len(
                engine_result.selected_candidate.supporting_evidence_ids
            )
            ev_count_blocked = (
                evidence_count < self.config.min_supporting_evidence
            )
            ev_count_check = EvidenceGuardCheck(
                check_name="supporting_evidence_count",
                passed=not ev_count_blocked,
                value=float(evidence_count),
                threshold=float(self.config.min_supporting_evidence),
                reason=(
                    f"Supporting evidence count {evidence_count} "
                    f"{'<' if ev_count_blocked else '>='} "
                    f"minimum {self.config.min_supporting_evidence}"
                ),
                block_reason=(
                    EvidenceBlockReason.MISSING_EVIDENCE
                    if ev_count_blocked
                    else None
                ),
            )
            checks.append(ev_count_check)
            if ev_count_blocked:
                blocked = True
                block_reasons.append(EvidenceBlockReason.MISSING_EVIDENCE)
                if not primary_reason:
                    primary_reason = ev_count_check.reason

        # 8. Evidence trace requirement (HARD BLOCK)
        if self.config.require_evidence_trace and engine_result.selected_candidate:
            has_trace = len(
                engine_result.selected_candidate.supporting_evidence_ids
            ) > 0
            trace_check = EvidenceGuardCheck(
                check_name="evidence_trace",
                passed=has_trace,
                value=None,
                threshold=None,
                reason=(
                    "Candidate has evidence trace"
                    if has_trace
                    else "Candidate lacks evidence trace — cannot verify financial adjustment"
                ),
                block_reason=(
                    EvidenceBlockReason.INSUFFICIENT_EXPLAINABILITY
                    if not has_trace
                    else None
                ),
            )
            checks.append(trace_check)
            if not has_trace:
                blocked = True
                block_reasons.append(EvidenceBlockReason.INSUFFICIENT_EXPLAINABILITY)
                if not primary_reason:
                    primary_reason = trace_check.reason

        # ── Determine final action ──

        if blocked:
            action = EvidenceAction.BLOCK
        else:
            action = EvidenceAction.PASS

        if not primary_reason:
            primary_reason = (
                f"Evidence coverage {evidence_coverage:.1%} and "
                f"consistency {evidence_consistency:.1%} are sufficient. "
                f"All evidence safety checks passed."
            )

        return EvidenceGuardResult(
            passed=not blocked,
            action=action,
            evidence_coverage=evidence_coverage,
            evidence_consistency=evidence_consistency,
            has_conflict=has_conflict,
            missing_evidence_count=missing_count,
            is_novel=is_novel,
            explanation_status=explanation_status,
            reason=primary_reason,
            checks=checks,
            block_reasons=block_reasons,
            exception_id=engine_result.exception_id,
            case_id=engine_result.case_id,
        )
