"""
Tests for Phase 9D — Policy Learning.

Tests cover:
- Policy versioning
- Policy decision logging
- Policy metrics calculation
- Policy comparison
- Safety regression detection
- Candidate rejection
- Candidate improvement
- Edge cases
"""

import pytest
from datetime import datetime, timedelta

from app.schemas.policy_learning import (
    PolicyComparison,
    PolicyDecisionLogEntry,
    PolicyDefinition,
    PolicyMetrics,
    PolicyPromotionDecision,
    PolicyStatus,
    PolicyThresholds,
    SafetyRegression,
)
from app.services.policy_learning import (
    PolicyComparator,
    PolicyDecisionLogger,
    PolicyMetricsCalculator,
    PolicyStore,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_log_entry(
    log_id: str = "PDL-001",
    policy_id: str = "POL-001",
    policy_version: str = "1.0.0",
    exception_id: str = "EXC-001",
    confidence: float = 0.85,
    decision: str = "AUTO",
    risk: str = "LOW",
    resolution_type: str = "FEE_ADJUSTMENT",
    financial_adjustment_paise: int = 3000,
    outcome_correct: bool = None,
    outcome_executed: bool = False,
    outcome_verified: bool = False,
    outcome_rolled_back: bool = False,
    outcome_reward: float = None,
    human_feedback: str = None,
) -> PolicyDecisionLogEntry:
    entry = PolicyDecisionLogEntry(
        log_id=log_id,
        policy_id=policy_id,
        policy_version=policy_version,
        exception_id=exception_id,
        confidence=confidence,
        decision=decision,
        risk=risk,
        resolution_type=resolution_type,
        financial_adjustment_paise=financial_adjustment_paise,
    )
    if outcome_correct is not None:
        entry.outcome_correct = outcome_correct
    if outcome_executed:
        entry.outcome_executed = outcome_executed
    if outcome_verified:
        entry.outcome_verified = outcome_verified
    if outcome_rolled_back:
        entry.outcome_rolled_back = outcome_rolled_back
    if outcome_reward is not None:
        entry.outcome_reward = outcome_reward
    if human_feedback is not None:
        entry.human_feedback = human_feedback
    return entry


def _make_entries(
    policy_id: str = "POL-001",
    policy_version: str = "1.0.0",
    auto_correct: int = 8,
    auto_incorrect: int = 2,
    human: int = 3,
    unresolved: int = 1,
    high_value_incorrect: int = 0,
) -> list:
    """Build a batch of log entries for testing."""
    entries = []
    idx = 0
    for _ in range(auto_correct):
        entries.append(_make_log_entry(
            log_id=f"PDL-{idx:03d}", policy_id=policy_id,
            policy_version=policy_version,
            decision="AUTO", outcome_correct=True,
            outcome_executed=True, outcome_verified=True,
            outcome_reward=0.8,
        ))
        idx += 1
    for i in range(auto_incorrect):
        adj = 150000 if i < high_value_incorrect else 3000
        entries.append(_make_log_entry(
            log_id=f"PDL-{idx:03d}", policy_id=policy_id,
            policy_version=policy_version,
            decision="AUTO", outcome_correct=False,
            outcome_executed=True,
            outcome_verified=(i % 2 == 0),
            outcome_rolled_back=(i % 2 == 1),
            financial_adjustment_paise=adj,
            outcome_reward=-0.7,
        ))
        idx += 1
    for _ in range(human):
        entries.append(_make_log_entry(
            log_id=f"PDL-{idx:03d}", policy_id=policy_id,
            policy_version=policy_version,
            decision="HUMAN_REVIEW",
        ))
        idx += 1
    for _ in range(unresolved):
        entries.append(_make_log_entry(
            log_id=f"PDL-{idx:03d}", policy_id=policy_id,
            policy_version=policy_version,
            decision="UNRESOLVED",
        ))
        idx += 1
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Policy Store Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyStore:
    """Test policy versioning and lifecycle."""

    def test_create_policy(self):
        """Create a new candidate policy."""
        store = PolicyStore()
        policy = store.create_policy("1.0.0", description="Baseline")
        assert policy.policy_id.startswith("POL-")
        assert policy.version == "1.0.0"
        assert policy.status == PolicyStatus.CANDIDATE

    def test_promote_policy(self):
        """Promote candidate to active."""
        store = PolicyStore()
        policy = store.create_policy("1.0.0")
        assert store.promote(policy.policy_id) is True
        assert policy.status == PolicyStatus.ACTIVE
        assert policy.promoted_at is not None

    def test_promote_replaces_active(self):
        """New promotion retires old active."""
        store = PolicyStore()
        p1 = store.create_policy("1.0.0")
        store.promote(p1.policy_id)
        p2 = store.create_policy("2.0.0")
        store.promote(p2.policy_id)
        assert p1.status == PolicyStatus.RETIRED
        assert p1.retired_at is not None
        assert p2.status == PolicyStatus.ACTIVE
        active = store.get_active_policy()
        assert active.policy_id == p2.policy_id

    def test_reject_policy(self):
        """Reject a candidate."""
        store = PolicyStore()
        policy = store.create_policy("1.0.0")
        assert store.reject(policy.policy_id) is True
        assert policy.status == PolicyStatus.REJECTED

    def test_reject_non_candidate_fails(self):
        """Cannot reject an active policy."""
        store = PolicyStore()
        policy = store.create_policy("1.0.0")
        store.promote(policy.policy_id)
        assert store.reject(policy.policy_id) is False

    def test_get_policies_by_status(self):
        """Filter policies by status."""
        store = PolicyStore()
        p1 = store.create_policy("1.0.0")
        p2 = store.create_policy("2.0.0")
        store.promote(p1.policy_id)
        candidates = store.get_policies_by_status(PolicyStatus.CANDIDATE)
        active = store.get_policies_by_status(PolicyStatus.ACTIVE)
        assert len(candidates) == 1
        assert len(active) == 1
        assert candidates[0].policy_id == p2.policy_id

    def test_get_active_policy_none(self):
        """No active policy returns None."""
        store = PolicyStore()
        assert store.get_active_policy() is None

    def test_policy_thresholds(self):
        """Policy stores configurable thresholds."""
        store = PolicyStore()
        thresholds = PolicyThresholds(
            min_confidence_for_auto=0.85,
            max_exposure_for_auto_paise=50000,
        )
        policy = store.create_policy("1.0.0", thresholds=thresholds)
        assert policy.thresholds.min_confidence_for_auto == 0.85
        assert policy.thresholds.max_exposure_for_auto_paise == 50000


# ─────────────────────────────────────────────────────────────────────────────
# Decision Logger Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionLogger:
    """Test policy decision logging."""

    def test_log_decision(self):
        """Record a decision."""
        logger = PolicyDecisionLogger()
        entry = logger.log_decision(
            policy_id="POL-001",
            policy_version="1.0.0",
            exception_id="EXC-001",
            confidence=0.85,
            decision="AUTO",
        )
        assert entry.log_id.startswith("PDL-")
        assert entry.decision == "AUTO"

    def test_record_outcome(self):
        """Record outcome for a logged decision."""
        logger = PolicyDecisionLogger()
        entry = logger.log_decision(
            policy_id="POL-001",
            policy_version="1.0.0",
            exception_id="EXC-001",
            confidence=0.85,
            decision="AUTO",
        )
        updated = logger.record_outcome(
            entry.log_id,
            correct=True,
            executed=True,
            verified=True,
            reward=0.8,
        )
        assert updated.outcome_correct is True
        assert updated.outcome_executed is True
        assert updated.outcome_verified is True
        assert updated.outcome_reward == 0.8
        assert updated.outcome_recorded_at is not None

    def test_record_outcome_nonexistent(self):
        """Recording outcome for non-existent log returns None."""
        logger = PolicyDecisionLogger()
        result = logger.record_outcome("PDL-NONE", correct=True)
        assert result is None

    def test_get_log_for_policy(self):
        """Retrieve log entries for a specific policy."""
        logger = PolicyDecisionLogger()
        logger.log_decision("POL-001", "1.0.0", "EXC-001", 0.8, "AUTO")
        logger.log_decision("POL-001", "1.0.0", "EXC-002", 0.7, "HUMAN_REVIEW")
        logger.log_decision("POL-002", "1.0.0", "EXC-003", 0.9, "AUTO")
        entries = logger.get_log_for_policy("POL-001")
        assert len(entries) == 2

    def test_log_entry_has_reason_codes(self):
        """Log entries store reason codes."""
        logger = PolicyDecisionLogger()
        entry = logger.log_decision(
            policy_id="POL-001",
            policy_version="1.0.0",
            exception_id="EXC-001",
            confidence=0.5,
            decision="HUMAN_REVIEW",
            reason_codes=["MEDIUM_CONFIDENCE", "EVIDENCE_AMBIGUITY"],
        )
        assert "MEDIUM_CONFIDENCE" in entry.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsCalculator:
    """Test policy metrics calculation."""

    def test_basic_metrics(self):
        """Calculate basic metrics from entries."""
        entries = _make_entries(auto_correct=8, auto_incorrect=2, human=3, unresolved=1)
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries, "POL-001", "1.0.0")
        assert metrics.total_decisions == 14
        assert metrics.auto_decisions == 10
        assert metrics.human_decisions == 3
        assert metrics.unresolved_decisions == 1
        assert metrics.automation_rate == pytest.approx(10 / 14, abs=0.01)

    def test_precision_calculation(self):
        """Precision = correct_auto / (correct_auto + incorrect_auto)."""
        entries = _make_entries(auto_correct=8, auto_incorrect=2)
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries)
        assert metrics.precision == pytest.approx(0.8, abs=0.01)
        assert metrics.false_automation == 2

    def test_precision_no_auto(self):
        """Precision is None when no AUTO decisions."""
        entries = _make_entries(auto_correct=0, auto_incorrect=0, human=5)
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries)
        assert metrics.precision is None

    def test_verification_failure_rate(self):
        """Ver fail rate = rollbacks / auto_executed."""
        entries = []
        for i in range(5):
            entries.append(_make_log_entry(
                log_id=f"PDL-{i}", decision="AUTO",
                outcome_correct=True, outcome_executed=True,
                outcome_verified=(i < 3), outcome_rolled_back=(i >= 3),
            ))
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries)
        assert metrics.auto_executed == 5
        assert metrics.auto_rolled_back == 2
        assert metrics.verification_failure_rate == pytest.approx(0.4, abs=0.01)

    def test_high_value_errors(self):
        """Count high-value incorrect AUTO decisions."""
        entries = _make_entries(
            auto_correct=5, auto_incorrect=3, high_value_incorrect=2,
        )
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries, high_value_threshold=100000)
        assert metrics.high_value_errors == 2

    def test_financial_metrics(self):
        """Financial exposure and error impact."""
        entries = [
            _make_log_entry(
                log_id="PDL-001", decision="AUTO",
                financial_adjustment_paise=5000,
                outcome_correct=True, outcome_executed=True,
            ),
            _make_log_entry(
                log_id="PDL-002", decision="AUTO",
                financial_adjustment_paise=10000,
                outcome_correct=False, outcome_executed=True,
            ),
        ]
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries)
        assert metrics.total_exposure_paise == 15000
        assert metrics.total_error_impact_paise == 10000

    def test_average_reward(self):
        """Average reward across entries with rewards."""
        entries = [
            _make_log_entry(log_id="PDL-001", decision="AUTO", outcome_reward=0.8),
            _make_log_entry(log_id="PDL-002", decision="AUTO", outcome_reward=-0.5),
            _make_log_entry(log_id="PDL-003", decision="HUMAN_REVIEW"),
        ]
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries)
        assert metrics.avg_reward == pytest.approx(0.15, abs=0.01)

    def test_empty_entries(self):
        """Empty entries produce zero metrics."""
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate([])
        assert metrics.total_decisions == 0
        assert metrics.automation_rate == 0.0

    def test_metrics_summary(self):
        """Metrics summary is readable."""
        entries = _make_entries(auto_correct=8, auto_incorrect=2)
        calc = PolicyMetricsCalculator()
        metrics = calc.calculate(entries, "POL-001", "1.0.0")
        summary = metrics.summary()
        assert "POL-001" in summary
        assert "v1.0.0" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Policy Comparator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyComparator:
    """Test policy comparison and safety regression detection."""

    def _make_current(self) -> PolicyMetrics:
        return PolicyMetrics(
            policy_id="POL-CURRENT",
            policy_version="1.0.0",
            total_decisions=100,
            auto_decisions=60,
            human_decisions=30,
            unresolved_decisions=10,
            automation_rate=0.60,
            precision=0.85,
            false_automation=5,
            high_value_errors=1,
            verification_failure_rate=0.05,
            total_exposure_paise=300000,
            total_error_impact_paise=25000,
            avg_reward=0.45,
        )

    def _make_candidate(self, **overrides) -> PolicyMetrics:
        current = self._make_current()
        data = current.model_dump()
        data.update(overrides)
        return PolicyMetrics(**data)

    def test_candidate_improves_precision(self):
        """Better precision + same safety → PROMOTE."""
        current = self._make_current()
        candidate = self._make_candidate(precision=0.92, false_automation=3)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == PolicyPromotionDecision.PROMOTE
        assert any("precision" in i for i in comparison.improvements)

    def test_candidate_reduces_false_automation(self):
        """Fewer false automations → improvement."""
        current = self._make_current()
        candidate = self._make_candidate(false_automation=2, precision=0.90)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == PolicyPromotionDecision.PROMOTE
        assert any("false_automation" in i for i in comparison.improvements)

    def test_candidate_increases_false_automation_critical(self):
        """More false automation → safety regression → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(
            false_automation=15, precision=0.70,
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.has_safety_regression is True
        assert comparison.recommendation == PolicyPromotionDecision.REJECT
        assert any(
            s.metric_name == "false_automation" for s in comparison.safety_regressions
        )

    def test_candidate_high_value_error_increase(self):
        """More high-value errors → critical safety regression → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(high_value_errors=3)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.has_safety_regression is True
        assert comparison.recommendation == PolicyPromotionDecision.REJECT

    def test_candidate_precision_below_threshold(self):
        """Precision below minimum → critical safety regression → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(precision=0.50, false_automation=10)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.has_safety_regression is True
        precision_issues = [
            s for s in comparison.safety_regressions
            if s.metric_name == "precision"
        ]
        assert len(precision_issues) == 1
        assert precision_issues[0].severity == "critical"

    def test_candidate_higher_verification_failure(self):
        """Higher verification failure rate → safety concern → DEFER."""
        current = self._make_current()
        candidate = self._make_candidate(
            verification_failure_rate=0.20,  # was 0.05 → 4× increase
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.has_safety_regression is True
        assert comparison.recommendation == PolicyPromotionDecision.DEFER

    def test_equal_metrics_defer(self):
        """Equal improvements and regressions → DEFER."""
        current = self._make_current()
        # 1 improvement (error_impact down) + 1 regression (precision down)
        candidate = self._make_candidate(
            precision=0.80,  # regression from 0.85
            total_error_impact_paise=15000,  # improvement from 25000
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == PolicyPromotionDecision.DEFER

    def test_no_safety_regression_with_improvement(self):
        """Improvement without safety issues → PROMOTE."""
        current = self._make_current()
        candidate = self._make_candidate(
            precision=0.90,
            false_automation=3,
            avg_reward=0.55,
            total_error_impact_paise=15000,
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == PolicyPromotionDecision.PROMOTE
        assert comparison.has_safety_regression is False
        assert len(comparison.improvements) >= 3

    def test_only_regressions_reject(self):
        """Only regressions, no improvements → REJECT."""
        current = self._make_current()
        candidate = self._make_candidate(
            precision=0.75,  # was 0.85 → regression
            automation_rate=0.50,  # was 0.60 → regression
            avg_reward=0.30,  # was 0.45 → regression
            total_error_impact_paise=30000,  # was 25000 → regression
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert comparison.recommendation == PolicyPromotionDecision.REJECT
        assert len(comparison.regressions) > len(comparison.improvements)

    def test_comparison_summary(self):
        """Comparison summary is readable."""
        current = self._make_current()
        candidate = self._make_candidate(precision=0.90)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        summary = comparison.summary()
        assert "Comparison:" in summary
        assert "PROMOTE" in summary

    def test_candidate_reduces_high_value_errors(self):
        """Fewer high-value errors → improvement."""
        current = self._make_current()
        candidate = self._make_candidate(high_value_errors=0, precision=0.88)
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert any("high_value_errors" in i for i in comparison.improvements)

    def test_candidate_reduces_verification_failure(self):
        """Lower verification failure rate → improvement."""
        current = self._make_current()
        candidate = self._make_candidate(
            verification_failure_rate=0.02, precision=0.87,
        )
        comparator = PolicyComparator()
        comparison = comparator.compare(current, candidate)
        assert any("verification_failure_rate" in i for i in comparison.improvements)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Policy Learning Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyLearningE2E:
    """End-to-end policy learning flow."""

    def test_full_cycle(self):
        """Create policy → log decisions → compute metrics → compare → promote."""
        # 1. Create and promote baseline
        store = PolicyStore()
        baseline = store.create_policy("1.0.0", description="Baseline")
        store.promote(baseline.policy_id)

        # 2. Log decisions under baseline with outcomes
        logger = PolicyDecisionLogger()
        for entry in _make_entries(
            policy_id=baseline.policy_id,
            policy_version="1.0.0",
            auto_correct=8, auto_incorrect=2, human=3, unresolved=1,
        ):
            logged = logger.log_decision(
                policy_id=entry.policy_id,
                policy_version=entry.policy_version,
                exception_id=entry.exception_id,
                confidence=entry.confidence,
                decision=entry.decision,
                resolution_type=entry.resolution_type,
                financial_adjustment_paise=entry.financial_adjustment_paise,
            )
            # Record outcomes for decisions that have them
            if entry.outcome_correct is not None:
                logger.record_outcome(
                    logged.log_id,
                    correct=entry.outcome_correct,
                    executed=entry.outcome_executed,
                    verified=entry.outcome_verified,
                    rolled_back=entry.outcome_rolled_back,
                    reward=entry.outcome_reward,
                )

        # 3. Compute baseline metrics
        calc = PolicyMetricsCalculator()
        baseline_entries = logger.get_log_for_policy(baseline.policy_id)
        baseline_metrics = calc.calculate(
            baseline_entries, baseline.policy_id, "1.0.0"
        )
        assert baseline_metrics.precision is not None

        # 4. Create candidate
        candidate = store.create_policy("2.0.0", description="Improved")

        # 5. Log decisions under candidate with outcomes
        for entry in _make_entries(
            policy_id=candidate.policy_id,
            policy_version="2.0.0",
            auto_correct=9, auto_incorrect=1, human=2, unresolved=1,
        ):
            logged = logger.log_decision(
                policy_id=entry.policy_id,
                policy_version=entry.policy_version,
                exception_id=entry.exception_id,
                confidence=entry.confidence,
                decision=entry.decision,
                resolution_type=entry.resolution_type,
                financial_adjustment_paise=entry.financial_adjustment_paise,
            )
            if entry.outcome_correct is not None:
                logger.record_outcome(
                    logged.log_id,
                    correct=entry.outcome_correct,
                    executed=entry.outcome_executed,
                    verified=entry.outcome_verified,
                    rolled_back=entry.outcome_rolled_back,
                    reward=entry.outcome_reward,
                )

        # 6. Compute candidate metrics
        candidate_entries = logger.get_log_for_policy(candidate.policy_id)
        candidate_metrics = calc.calculate(
            candidate_entries, candidate.policy_id, "2.0.0"
        )

        # 7. Compare
        comparator = PolicyComparator()
        comparison = comparator.compare(baseline_metrics, candidate_metrics)

        # 8. Candidate has better precision → should be promotable
        assert candidate_metrics.precision >= baseline_metrics.precision
        assert comparison.recommendation in (
            PolicyPromotionDecision.PROMOTE,
            PolicyPromotionDecision.DEFER,
        )

        # 9. If promoted
        if comparison.recommendation == PolicyPromotionDecision.PROMOTE:
            assert store.promote(candidate.policy_id) is True
            assert store.get_active_policy().policy_id == candidate.policy_id

    def test_safety_blocks_promotion(self):
        """Candidate with safety regression is rejected even with higher automation."""
        store = PolicyStore()
        baseline = store.create_policy("1.0.0")
        store.promote(baseline.policy_id)

        calc = PolicyMetricsCalculator()
        baseline_metrics = PolicyMetrics(
            policy_id=baseline.policy_id,
            policy_version="1.0.0",
            auto_decisions=50,
            precision=0.90,
            false_automation=2,
            high_value_errors=0,
            verification_failure_rate=0.02,
            automation_rate=0.50,
            total_error_impact_paise=5000,
        )

        candidate_metrics = PolicyMetrics(
            policy_id="POL-CAND",
            policy_version="2.0.0",
            auto_decisions=70,
            precision=0.80,
            false_automation=10,
            high_value_errors=3,
            verification_failure_rate=0.08,
            automation_rate=0.70,
            total_error_impact_paise=25000,
        )

        comparator = PolicyComparator()
        comparison = comparator.compare(baseline_metrics, candidate_metrics)

        assert comparison.recommendation == PolicyPromotionDecision.REJECT
        assert comparison.has_safety_regression is True
        # Even though automation rate improved, safety blocks it
        assert any("automation_rate" in i for i in comparison.improvements)
        assert any(
            s.metric_name == "high_value_errors"
            for s in comparison.safety_regressions
        )
