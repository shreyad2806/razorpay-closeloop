"""
Policy Learning services for Razorpay CloseLoop Phase 9D.

Implements policy versioning, decision logging, metrics calculation,
and policy comparison with safety regression detection.

Safety principle:
  A learned policy is a CANDIDATE.
  It is NOT automatically trusted.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from app.schemas.policy_learning import (
    PolicyDecisionLogEntry,
    PolicyDefinition,
    PolicyMetrics,
    PolicyPromotionDecision,
    PolicyStatus,
    PolicyThresholds,
    PolicyComparison,
    SafetyRegression,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Policy Store
# ─────────────────────────────────────────────────────────────────────────────


class PolicyStore:
    """Stores and manages policy versions."""

    def __init__(self) -> None:
        self._policies: Dict[str, PolicyDefinition] = {}
        self._by_status: Dict[str, List[str]] = {
            "CANDIDATE": [],
            "ACTIVE": [],
            "RETIRED": [],
            "REJECTED": [],
        }

    def create_policy(
        self,
        version: str,
        thresholds: Optional[PolicyThresholds] = None,
        applicable_categories: Optional[List[str]] = None,
        risk_limits: Optional[List[str]] = None,
        description: str = "",
        created_by: str = "system",
    ) -> PolicyDefinition:
        """Create a new candidate policy."""
        policy_id = _gen_id("POL")
        policy = PolicyDefinition(
            policy_id=policy_id,
            version=version,
            status=PolicyStatus.CANDIDATE,
            thresholds=thresholds or PolicyThresholds(),
            applicable_categories=applicable_categories or [],
            risk_limits=risk_limits or ["LOW"],
            description=description,
            created_by=created_by,
        )
        self._policies[policy_id] = policy
        self._by_status["CANDIDATE"].append(policy_id)
        return policy

    def get_policy(self, policy_id: str) -> Optional[PolicyDefinition]:
        return self._policies.get(policy_id)

    def get_active_policy(self) -> Optional[PolicyDefinition]:
        """Get the currently active policy."""
        active_ids = self._by_status.get("ACTIVE", [])
        if active_ids:
            return self._policies.get(active_ids[-1])
        return None

    def get_policies_by_status(self, status: PolicyStatus) -> List[PolicyDefinition]:
        ids = self._by_status.get(status.value, [])
        return [self._policies[pid] for pid in ids if pid in self._policies]

    def promote(self, policy_id: str) -> bool:
        """Promote a candidate to active. Retires current active."""
        policy = self._policies.get(policy_id)
        if not policy or policy.status != PolicyStatus.CANDIDATE:
            return False

        # Retire current active
        for active_id in list(self._by_status.get("ACTIVE", [])):
            old = self._policies.get(active_id)
            if old:
                old.status = PolicyStatus.RETIRED
                old.retired_at = datetime.utcnow()
                self._by_status["ACTIVE"].remove(active_id)
                self._by_status["RETIRED"].append(active_id)

        # Promote
        policy.status = PolicyStatus.ACTIVE
        policy.promoted_at = datetime.utcnow()
        self._by_status["CANDIDATE"].remove(policy_id)
        self._by_status["ACTIVE"].append(policy_id)
        return True

    def reject(self, policy_id: str) -> bool:
        policy = self._policies.get(policy_id)
        if not policy or policy.status != PolicyStatus.CANDIDATE:
            return False
        policy.status = PolicyStatus.REJECTED
        self._by_status["CANDIDATE"].remove(policy_id)
        self._by_status["REJECTED"].append(policy_id)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Decision Logger
# ─────────────────────────────────────────────────────────────────────────────


class PolicyDecisionLogger:
    """Logs decisions made under specific policies."""

    def __init__(self) -> None:
        self._log: List[PolicyDecisionLogEntry] = []
        self._by_policy: Dict[str, List[str]] = {}

    def log_decision(
        self,
        policy_id: str,
        policy_version: str,
        exception_id: str,
        confidence: float,
        decision: str,
        case_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        risk: str = "LOW",
        reason_codes: Optional[List[str]] = None,
        resolution_type: Optional[str] = None,
        financial_adjustment_paise: int = 0,
    ) -> PolicyDecisionLogEntry:
        """Record a decision under a policy."""
        entry = PolicyDecisionLogEntry(
            log_id=_gen_id("PDL"),
            policy_id=policy_id,
            policy_version=policy_version,
            exception_id=exception_id,
            case_id=case_id,
            candidate_id=candidate_id,
            confidence=confidence,
            risk=risk,
            decision=decision,
            reason_codes=reason_codes or [],
            resolution_type=resolution_type,
            financial_adjustment_paise=financial_adjustment_paise,
            logged_at=datetime.utcnow(),
        )
        self._log.append(entry)
        self._by_policy.setdefault(policy_id, []).append(entry.log_id)
        return entry

    def record_outcome(
        self,
        log_id: str,
        correct: Optional[bool] = None,
        executed: bool = False,
        verified: bool = False,
        rolled_back: bool = False,
        reward: Optional[float] = None,
        human_feedback: Optional[str] = None,
    ) -> Optional[PolicyDecisionLogEntry]:
        """Record the outcome for a logged decision."""
        for entry in self._log:
            if entry.log_id == log_id:
                entry.outcome_correct = correct
                entry.outcome_executed = executed
                entry.outcome_verified = verified
                entry.outcome_rolled_back = rolled_back
                entry.outcome_reward = reward
                entry.human_feedback = human_feedback
                entry.outcome_recorded_at = datetime.utcnow()
                return entry
        return None

    def get_log_for_policy(self, policy_id: str) -> List[PolicyDecisionLogEntry]:
        ids = self._by_policy.get(policy_id, [])
        return [e for e in self._log if e.log_id in ids]

    def get_all_entries(self) -> List[PolicyDecisionLogEntry]:
        return list(self._log)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────


class PolicyMetricsCalculator:
    """Computes metrics from decision log entries."""

    def calculate(
        self,
        entries: List[PolicyDecisionLogEntry],
        policy_id: str = "",
        policy_version: str = "",
        high_value_threshold: int = 100000,
    ) -> PolicyMetrics:
        """Calculate metrics from log entries."""
        if not entries:
            return PolicyMetrics(
                policy_id=policy_id,
                policy_version=policy_version,
            )

        total = len(entries)
        auto = sum(1 for e in entries if e.decision == "AUTO")
        human = sum(1 for e in entries if e.decision == "HUMAN_REVIEW")
        unresolved = sum(1 for e in entries if e.decision == "UNRESOLVED")

        automation_rate = auto / total if total > 0 else 0.0
        human_review_rate = human / total if total > 0 else 0.0
        unresolved_rate = unresolved / total if total > 0 else 0.0

        # Outcome-dependent metrics
        with_outcomes = [e for e in entries if e.outcome_correct is not None]
        auto_with_outcomes = [
            e for e in with_outcomes if e.decision == "AUTO"
        ]
        correct_auto = sum(1 for e in auto_with_outcomes if e.outcome_correct is True)
        incorrect_auto = sum(1 for e in auto_with_outcomes if e.outcome_correct is False)
        precision = (
            correct_auto / (correct_auto + incorrect_auto)
            if (correct_auto + incorrect_auto) > 0
            else None
        )

        # Execution metrics
        auto_executed = sum(
            1 for e in entries
            if e.decision == "AUTO" and e.outcome_executed
        )
        auto_verified = sum(
            1 for e in entries
            if e.decision == "AUTO" and e.outcome_executed and e.outcome_verified
        )
        auto_rolled_back = sum(
            1 for e in entries
            if e.decision == "AUTO" and e.outcome_rolled_back
        )
        ver_fail_rate = (
            auto_rolled_back / auto_executed
            if auto_executed > 0
            else None
        )

        # Financial
        total_exposure = sum(
            e.financial_adjustment_paise for e in entries if e.decision == "AUTO"
        )
        error_impact = sum(
            e.financial_adjustment_paise
            for e in entries
            if e.decision == "AUTO" and e.outcome_correct is False
        )

        # High-value errors
        high_value_errors = sum(
            1 for e in entries
            if e.decision == "AUTO"
            and e.outcome_correct is False
            and e.financial_adjustment_paise >= high_value_threshold
        )

        # Rewards
        rewards = [e.outcome_reward for e in entries if e.outcome_reward is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else None

        return PolicyMetrics(
            policy_id=policy_id,
            policy_version=policy_version,
            total_decisions=total,
            auto_decisions=auto,
            human_decisions=human,
            unresolved_decisions=unresolved,
            automation_rate=automation_rate,
            human_review_rate=human_review_rate,
            unresolved_rate=unresolved_rate,
            decisions_with_outcomes=len(with_outcomes),
            correct_auto=correct_auto,
            incorrect_auto=incorrect_auto,
            precision=precision,
            false_automation=incorrect_auto,
            auto_executed=auto_executed,
            auto_verified=auto_verified,
            auto_rolled_back=auto_rolled_back,
            verification_failure_rate=ver_fail_rate,
            total_exposure_paise=total_exposure,
            total_error_impact_paise=error_impact,
            avg_reward=avg_reward,
            high_value_errors=high_value_errors,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Comparator
# ─────────────────────────────────────────────────────────────────────────────


class PolicyComparator:
    """Compares current vs candidate policy metrics.

    A candidate is better only if safety and quality remain
    within required thresholds.
    """

    # Safety thresholds — candidates must not regress past these
    SAFETY_RULES = {
        "false_automation": "max_increase",
        "high_value_errors": "no_increase",
        "verification_failure_rate": "max_10pct_increase",
        "precision": "min_threshold",
    }

    PRECISION_MIN_THRESHOLD = 0.70
    FALSE_AUTO_MAX_INCREASE_RATIO = 1.2  # Max 20% increase
    VER_FAIL_MAX_INCREASE_RATIO = 1.10   # Max 10% increase

    def compare(
        self,
        current: PolicyMetrics,
        candidate: PolicyMetrics,
        high_value_threshold: int = 100000,
    ) -> PolicyComparison:
        """Compare current and candidate policy metrics.

        Returns a comparison with safety regression detection.
        """
        improvements: List[str] = []
        regressions: List[str] = []
        safety_regressions: List[SafetyRegression] = []

        # 1. Precision check
        if candidate.precision is not None and current.precision is not None:
            if candidate.precision > current.precision:
                improvements.append(
                    f"precision: {current.precision:.1%} → {candidate.precision:.1%}"
                )
            elif candidate.precision < current.precision:
                regressions.append(
                    f"precision: {current.precision:.1%} → {candidate.precision:.1%}"
                )
                if candidate.precision < self.PRECISION_MIN_THRESHOLD:
                    safety_regressions.append(SafetyRegression(
                        metric_name="precision",
                        current_value=current.precision,
                        candidate_value=candidate.precision,
                        threshold=self.PRECISION_MIN_THRESHOLD,
                        severity="critical",
                        description=(
                            f"Candidate precision {candidate.precision:.1%} "
                            f"below minimum threshold {self.PRECISION_MIN_THRESHOLD:.1%}"
                        ),
                    ))

        # 2. False automation check
        if candidate.false_automation > 0 or current.false_automation > 0:
            if current.false_automation > 0:
                increase_ratio = (
                    candidate.false_automation / current.false_automation
                )
                if increase_ratio > self.FALSE_AUTO_MAX_INCREASE_RATIO:
                    safety_regressions.append(SafetyRegression(
                        metric_name="false_automation",
                        current_value=float(current.false_automation),
                        candidate_value=float(candidate.false_automation),
                        threshold=float(
                            current.false_automation * self.FALSE_AUTO_MAX_INCREASE_RATIO
                        ),
                        severity="critical",
                        description=(
                            f"False automation increased by "
                            f"{(increase_ratio - 1) * 100:.0f}% "
                            f"({current.false_automation} → {candidate.false_automation})"
                        ),
                    ))
                    regressions.append(
                        f"false_automation: {current.false_automation} → {candidate.false_automation}"
                    )
                elif candidate.false_automation < current.false_automation:
                    improvements.append(
                        f"false_automation: {current.false_automation} → {candidate.false_automation}"
                    )
            elif candidate.false_automation > 0:
                # Current has 0, candidate has > 0 → regression
                safety_regressions.append(SafetyRegression(
                    metric_name="false_automation",
                    current_value=0.0,
                    candidate_value=float(candidate.false_automation),
                    threshold=0.0,
                    severity="critical",
                    description=(
                        f"Candidate introduced {candidate.false_automation} "
                        f"false automation(s) where none existed"
                    ),
                ))
                regressions.append(
                    f"false_automation: 0 → {candidate.false_automation}"
                )

        # 3. High-value errors — no increase allowed
        if candidate.high_value_errors > current.high_value_errors:
            safety_regressions.append(SafetyRegression(
                metric_name="high_value_errors",
                current_value=float(current.high_value_errors),
                candidate_value=float(candidate.high_value_errors),
                threshold=float(current.high_value_errors),
                severity="critical",
                description=(
                    f"High-value errors increased: "
                    f"{current.high_value_errors} → {candidate.high_value_errors}"
                ),
            ))
            regressions.append(
                f"high_value_errors: {current.high_value_errors} → {candidate.high_value_errors}"
            )
        elif candidate.high_value_errors < current.high_value_errors:
            improvements.append(
                f"high_value_errors: {current.high_value_errors} → {candidate.high_value_errors}"
            )

        # 4. Verification failure rate
        if (
            candidate.verification_failure_rate is not None
            and current.verification_failure_rate is not None
        ):
            if candidate.verification_failure_rate > current.verification_failure_rate:
                increase_ratio = (
                    candidate.verification_failure_rate
                    / current.verification_failure_rate
                    if current.verification_failure_rate > 0
                    else float("inf")
                )
                if increase_ratio > self.VER_FAIL_MAX_INCREASE_RATIO:
                    safety_regressions.append(SafetyRegression(
                        metric_name="verification_failure_rate",
                        current_value=current.verification_failure_rate,
                        candidate_value=candidate.verification_failure_rate,
                        threshold=(
                            current.verification_failure_rate
                            * self.VER_FAIL_MAX_INCREASE_RATIO
                        ),
                        severity="warning",
                        description=(
                            f"Verification failure rate increased: "
                            f"{current.verification_failure_rate:.1%} → "
                            f"{candidate.verification_failure_rate:.1%}"
                        ),
                    ))
                    regressions.append(
                        f"verification_failure_rate: "
                        f"{current.verification_failure_rate:.1%} → "
                        f"{candidate.verification_failure_rate:.1%}"
                    )
            elif candidate.verification_failure_rate < current.verification_failure_rate:
                improvements.append(
                    f"verification_failure_rate: "
                    f"{current.verification_failure_rate:.1%} → "
                    f"{candidate.verification_failure_rate:.1%}"
                )

        # 5. Automation rate (informational, not safety-critical)
        if candidate.automation_rate > current.automation_rate:
            improvements.append(
                f"automation_rate: {current.automation_rate:.1%} → {candidate.automation_rate:.1%}"
            )
        elif candidate.automation_rate < current.automation_rate:
            regressions.append(
                f"automation_rate: {current.automation_rate:.1%} → {candidate.automation_rate:.1%}"
            )

        # 6. Reward (informational)
        if candidate.avg_reward is not None and current.avg_reward is not None:
            if candidate.avg_reward > current.avg_reward:
                improvements.append(
                    f"avg_reward: {current.avg_reward:.3f} → {candidate.avg_reward:.3f}"
                )
            elif candidate.avg_reward < current.avg_reward:
                regressions.append(
                    f"avg_reward: {current.avg_reward:.3f} → {candidate.avg_reward:.3f}"
                )

        # 7. Financial error impact
        if candidate.total_error_impact_paise < current.total_error_impact_paise:
            improvements.append(
                f"error_impact: {current.total_error_impact_paise} → {candidate.total_error_impact_paise} paise"
            )
        elif candidate.total_error_impact_paise > current.total_error_impact_paise:
            regressions.append(
                f"error_impact: {current.total_error_impact_paise} → {candidate.total_error_impact_paise} paise"
            )

        # Determine recommendation
        has_safety = len(safety_regressions) > 0
        critical_safety = any(
            s.severity == "critical" for s in safety_regressions
        )

        if critical_safety:
            recommendation = PolicyPromotionDecision.REJECT
            reason = (
                f"Rejected: {len([s for s in safety_regressions if s.severity == 'critical'])} "
                f"critical safety regression(s) detected"
            )
        elif has_safety:
            recommendation = PolicyPromotionDecision.DEFER
            reason = (
                f"Deferred: {len(safety_regressions)} safety concern(s) need investigation"
            )
        elif len(improvements) > len(regressions):
            recommendation = PolicyPromotionDecision.PROMOTE
            reason = (
                f"Promoted: {len(improvements)} improvements, "
                f"{len(regressions)} regressions, no safety issues"
            )
        elif len(improvements) == len(regressions):
            recommendation = PolicyPromotionDecision.DEFER
            reason = "Deferred: equal improvements and regressions — needs more data"
        else:
            recommendation = PolicyPromotionDecision.REJECT
            reason = (
                f"Rejected: {len(regressions)} regressions "
                f"outweigh {len(improvements)} improvements"
            )

        return PolicyComparison(
            current_policy_id=current.policy_id,
            current_version=current.policy_version,
            candidate_policy_id=candidate.policy_id,
            candidate_version=candidate.policy_version,
            current_metrics=current,
            candidate_metrics=candidate,
            improvements=improvements,
            regressions=regressions,
            safety_regressions=safety_regressions,
            has_safety_regression=has_safety,
            recommendation=recommendation,
            recommendation_reason=reason,
        )
