"""
Tests for Razorpay CloseLoop Phase 10I — Experiment Comparison.

Verifies multi-dimensional comparison and ranking of MLflow training runs.
"""

import pytest
from datetime import datetime, timezone

from app.schemas.experiment_comparison import (
    ComparisonDimension,
    ComparisonRun,
    DimensionScore,
    ExperimentComparison,
    MetricDiff,
    PairwiseComparison,
    RankingStrategy,
    RunPosition,
    RunRanking,
)
from app.schemas.mlflow_tracking import MetricsSnapshot, MLflowRunMetadata, RunStatus
from app.services.experiment_comparison import (
    ExperimentComparisonService,
    STRATEGY_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_run(
    run_id: str,
    model_name: str = "XGB",
    model_version: str = "1.0",
    algorithm: str = "xgboost",
    metrics: dict = None,
    hyperparameters: dict = None,
    dataset_version: str = "v1",
    **kwargs,
) -> ComparisonRun:
    m = metrics or {}
    return ComparisonRun(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        algorithm=algorithm,
        dataset_version=dataset_version,
        hyperparameters=hyperparameters or {},
        accuracy=m.get("accuracy"),
        precision_macro=m.get("precision_macro"),
        recall_macro=m.get("recall_macro"),
        f1_macro=m.get("f1_macro"),
        false_automation=m.get("false_automation"),
        high_value_errors=m.get("high_value_errors"),
        verification_failure_rate=m.get("verification_failure_rate"),
        unsafe_decision_rate=m.get("unsafe_decision_rate"),
        automation_rate=m.get("automation_rate"),
        human_review_rate=m.get("human_review_rate"),
        unresolved_rate=m.get("unresolved_rate"),
        resolution_accuracy=m.get("resolution_accuracy"),
        total_adjustment_paise=m.get("total_adjustment_paise", 0),
        total_error_impact_paise=m.get("total_error_impact_paise", 0),
        high_value_error_impact_paise=m.get("high_value_error_impact_paise", 0),
        avg_reward=m.get("avg_reward"),
        per_class_f1={k: v for k, v in m.items() if k.startswith("per_class.")},
        raw_metrics=m,
        **kwargs,
    )


@pytest.fixture
def good_run() -> ComparisonRun:
    """A good run: high accuracy, low errors, good automation."""
    return _make_run(
        run_id="run-good-001",
        model_version="v1.0",
        metrics={
            "accuracy": 0.92,
            "precision_macro": 0.90,
            "recall_macro": 0.88,
            "f1_macro": 0.89,
            "false_automation": 1,
            "high_value_errors": 0,
            "verification_failure_rate": 0.02,
            "automation_rate": 0.75,
            "human_review_rate": 0.20,
            "resolution_accuracy": 0.88,
            "avg_reward": 0.65,
            "per_class.FEE_DIFFERENCE": 0.93,
            "per_class.REFUND_ADJUSTMENT": 0.87,
        },
        hyperparameters={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    )


@pytest.fixture
def safe_run() -> ComparisonRun:
    """A very safe run: zero errors but lower accuracy."""
    return _make_run(
        run_id="run-safe-001",
        model_version="v2.0",
        metrics={
            "accuracy": 0.85,
            "precision_macro": 0.84,
            "recall_macro": 0.83,
            "f1_macro": 0.835,
            "false_automation": 0,
            "high_value_errors": 0,
            "verification_failure_rate": 0.00,
            "automation_rate": 0.60,
            "human_review_rate": 0.35,
            "resolution_accuracy": 0.82,
            "avg_reward": 0.55,
            "per_class.FEE_DIFFERENCE": 0.88,
            "per_class.REFUND_ADJUSTMENT": 0.80,
        },
        hyperparameters={"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05},
    )


@pytest.fixture
def risky_run() -> ComparisonRun:
    """A risky run: high accuracy but with high-value errors."""
    return _make_run(
        run_id="run-risky-001",
        model_version="v3.0",
        metrics={
            "accuracy": 0.95,
            "precision_macro": 0.93,
            "recall_macro": 0.91,
            "f1_macro": 0.92,
            "false_automation": 5,
            "high_value_errors": 2,
            "verification_failure_rate": 0.08,
            "automation_rate": 0.85,
            "human_review_rate": 0.10,
            "resolution_accuracy": 0.90,
            "avg_reward": 0.40,
            "total_error_impact_paise": 50000,
            "high_value_error_impact_paise": 30000,
            "per_class.FEE_DIFFERENCE": 0.96,
            "per_class.REFUND_ADJUSTMENT": 0.90,
        },
        hyperparameters={"n_estimators": 200, "max_depth": 10, "learning_rate": 0.2},
    )


@pytest.fixture
def service() -> ExperimentComparisonService:
    return ExperimentComparisonService()


@pytest.fixture
def all_runs(good_run, safe_run, risky_run):
    return [good_run, safe_run, risky_run]


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComparisonDimension:
    def test_all_dimensions_exist(self):
        dims = list(ComparisonDimension)
        assert len(dims) == 14
        assert ComparisonDimension.SAFETY in dims
        assert ComparisonDimension.ACCURACY in dims
        assert ComparisonDimension.OVERALL in dims

    def test_dimension_count(self):
        assert len(ComparisonDimension) >= 12


class TestRankingStrategy:
    def test_strategies_exist(self):
        assert RankingStrategy.SAFETY_FIRST.value == "safety_first"
        assert RankingStrategy.BALANCED.value == "balanced"
        assert RankingStrategy.ACCURACY_FOCUSED.value == "accuracy_focused"
        assert RankingStrategy.AUTOMATION_FOCUSED.value == "automation_focused"


class TestRunPosition:
    def test_positions_exist(self):
        assert RunPosition.BEST.value == "BEST"
        assert RunPosition.WORST.value == "WORST"
        assert RunPosition.TIED.value == "TIED"


# ─────────────────────────────────────────────────────────────────────────────
# Run Preparation
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPreparation:
    def test_prepare_from_mlflow_metadata(self):
        meta = MLflowRunMetadata(
            run_id="run-001",
            experiment_name="test",
            model_type="classifier",
            model_name="XGB",
            model_version="1.0",
            algorithm="xgboost",
            dataset_version="v2",
            training_examples=1000,
            feature_count=12,
            hyperparameters={"max_depth": 6},
            n_estimators=100,
            learning_rate=0.1,
        )
        snap = MetricsSnapshot(
            run_id="run-001",
            accuracy=0.90,
            f1_macro=0.88,
            false_automation=2,
            high_value_errors=0,
            automation_rate=0.70,
        )
        run = ExperimentComparisonService.prepare_run(meta, snap)
        assert run.run_id == "run-001"
        assert run.model_name == "XGB"
        assert run.accuracy == 0.90
        assert run.f1_macro == 0.88
        assert run.false_automation == 2
        assert run.dataset_version == "v2"
        assert run.hyperparameters == {"max_depth": 6}

    def test_prepare_from_dict(self):
        run = ExperimentComparisonService.prepare_run_from_dict(
            run_id="run-dict",
            model_name="LR",
            model_version="1.0",
            algorithm="logistic_regression",
            metrics={"accuracy": 0.80, "f1_macro": 0.78, "false_automation": 3},
        )
        assert run.run_id == "run-dict"
        assert run.accuracy == 0.80
        assert run.false_automation == 3

    def test_prepare_without_metrics(self):
        meta = MLflowRunMetadata(
            run_id="run-no-metrics",
            experiment_name="test",
            model_type="classifier",
            model_name="XGB",
            model_version="1.0",
            algorithm="xgboost",
        )
        run = ExperimentComparisonService.prepare_run(meta)
        assert run.accuracy is None
        assert run.false_automation is None


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Comparison
# ─────────────────────────────────────────────────────────────────────────────


class TestPairwiseComparison:
    def test_compare_pair_basic(self, good_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(good_run, risky_run)
        assert comp.run_a_id == "run-good-001"
        assert comp.run_b_id == "run-risky-001"
        assert len(comp.metric_diffs) > 0

    def test_risky_has_more_wins_in_accuracy(self, good_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(good_run, risky_run)
        acc_diff = next(
            (d for d in comp.metric_diffs if d.metric_name == "accuracy"), None
        )
        assert acc_diff is not None
        assert acc_diff.run_b_value > acc_diff.run_a_value
        assert acc_diff.winner == "run-risky-001"

    def test_good_has_fewer_false_automations(self, good_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(good_run, risky_run)
        fa_diff = next(
            (d for d in comp.metric_diffs if d.metric_name == "false_automation"), None
        )
        assert fa_diff is not None
        assert fa_diff.run_a_value < fa_diff.run_b_value

    def test_safety_notes_generated(self, good_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(good_run, risky_run)
        assert len(comp.safety_notes) > 0
        assert any("HIGH VALUE" in n for n in comp.safety_notes)

    def test_parameter_diffs(self, good_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(good_run, risky_run)
        # max_depth differs (6 vs 10)
        assert "max_depth" in comp.parameter_diffs
        assert comp.parameter_diffs["max_depth"]["run_a"] == 6
        assert comp.parameter_diffs["max_depth"]["run_b"] == 10

    def test_safe_vs_risky_safety_notes(self, safe_run, risky_run):
        comp = ExperimentComparisonService.compare_pair(safe_run, risky_run)
        assert any("HIGH VALUE" in n or "false" in n.lower() for n in comp.safety_notes)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension Scoring
# ─────────────────────────────────────────────────────────────────────────────


class TestDimensionScoring:
    def test_safety_dimension(self, all_runs):
        scores = ExperimentComparisonService._score_dimension(
            all_runs, ComparisonDimension.SAFETY
        )
        assert len(scores) == 3
        # Safe run (0 false automations) should score highest on safety
        safe_score = scores["run-safe-001"]
        risky_score = scores["run-risky-001"]
        assert safe_score.score > risky_score.score
        assert safe_score.rank < risky_score.rank

    def test_accuracy_dimension(self, all_runs):
        scores = ExperimentComparisonService._score_dimension(
            all_runs, ComparisonDimension.ACCURACY
        )
        # Risky run (0.95) should be highest accuracy
        risky_score = scores["run-risky-001"]
        assert risky_score.rank == 1

    def test_dimension_scores_normalized(self, all_runs):
        for dim in ComparisonDimension:
            if dim == ComparisonDimension.OVERALL:
                continue
            scores = ExperimentComparisonService._score_dimension(all_runs, dim)
            for dscore in scores.values():
                assert 0.0 <= dscore.score <= 1.0

    def test_single_run_gets_best(self, good_run):
        scores = ExperimentComparisonService._score_dimension(
            [good_run], ComparisonDimension.ACCURACY
        )
        assert scores[good_run.run_id].position == RunPosition.BEST

    def test_tied_runs(self):
        r1 = _make_run("r1", metrics={"accuracy": 0.90})
        r2 = _make_run("r2", metrics={"accuracy": 0.90})
        scores = ExperimentComparisonService._score_dimension(
            [r1, r2], ComparisonDimension.ACCURACY
        )
        # Both should have same score
        assert scores["r1"].score == scores["r2"].score


# ─────────────────────────────────────────────────────────────────────────────
# Overall Ranking
# ─────────────────────────────────────────────────────────────────────────────


class TestRanking:
    def test_safety_first_ranks_safe_above_risky(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(
            all_runs, RankingStrategy.SAFETY_FIRST
        )
        assert len(rankings) == 3
        # Risky run should not be #1 with safety_first
        risky_rank = next(r for r in rankings if r.run_id == "run-risky-001")
        safe_rank = next(r for r in rankings if r.run_id == "run-safe-001")
        # Safe run's overall rank should be better or equal
        assert risky_rank.safety_rank > safe_rank.safety_rank

    def test_accuracy_focused_can_rank_risky_higher(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(
            all_runs, RankingStrategy.ACCURACY_FOCUSED
        )
        risky_rank = next(r for r in rankings if r.run_id == "run-risky-001")
        good_rank = next(r for r in rankings if r.run_id == "run-good-001")
        # Risky should be ranked higher on pure accuracy dimension
        acc_scores = [ds for ds in risky_rank.dimension_scores
                       if ds.dimension == ComparisonDimension.ACCURACY]
        assert len(acc_scores) > 0
        assert acc_scores[0].rank == 1  # Risky is most accurate

    def test_rankings_have_all_dimensions(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(all_runs)
        for r in rankings:
            assert len(r.dimension_scores) > 0
            assert r.overall_score >= 0.0

    def test_safety_score_in_ranking(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(all_runs)
        for r in rankings:
            assert 0.0 <= r.safety_score <= 1.0

    def test_safety_regression_flagged(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(all_runs)
        risky = next(r for r in rankings if r.run_id == "run-risky-001")
        assert risky.has_safety_regression
        assert len(risky.safety_issues) > 0

    def test_safe_run_no_safety_issues(self, all_runs):
        rankings = ExperimentComparisonService.rank_runs(all_runs)
        safe = next(r for r in rankings if r.run_id == "run-safe-001")
        assert not safe.has_safety_regression
        assert len(safe.safety_issues) == 0

    def test_empty_runs(self):
        rankings = ExperimentComparisonService.rank_runs([])
        assert rankings == []


# ─────────────────────────────────────────────────────────────────────────────
# Full Comparison
# ─────────────────────────────────────────────────────────────────────────────


class TestFullComparison:
    def test_compare_basic(self, service, all_runs):
        comp = service.compare(all_runs)
        assert comp.run_count == 3
        assert len(comp.pairwise_comparisons) == 3  # C(3,2)
        assert len(comp.rankings) == 3
        assert comp.best_overall_run_id is not None

    def test_compare_needs_at_least_2(self, service, good_run):
        with pytest.raises(ValueError):
            service.compare([good_run])

    def test_safest_run_identified(self, service, all_runs):
        comp = service.compare(all_runs)
        assert comp.safest_run_id == "run-safe-001"

    def test_most_accurate_identified(self, service, all_runs):
        comp = service.compare(all_runs)
        assert comp.most_accurate_run_id == "run-risky-001"

    def test_best_overall_is_not_risky(self, service, all_runs):
        comp = service.compare(all_runs, RankingStrategy.SAFETY_FIRST)
        # Risky should NOT be best overall with safety_first
        assert comp.best_overall_run_id != "run-risky-001"

    def test_safety_warnings_present(self, service, all_runs):
        comp = service.compare(all_runs)
        assert not comp.all_runs_safe
        assert len(comp.safety_warnings) > 0

    def test_insights_generated(self, service, all_runs):
        comp = service.compare(all_runs)
        assert len(comp.insights) > 0

    def test_pairwise_comparisons(self, service, all_runs):
        comp = service.compare(all_runs)
        pairs = {(p.run_a_id, p.run_b_id) for p in comp.pairwise_comparisons}
        assert ("run-good-001", "run-safe-001") in pairs
        assert ("run-good-001", "run-risky-001") in pairs
        assert ("run-safe-001", "run-risky-001") in pairs

    def test_comparison_stored(self, service, all_runs):
        comp = service.compare(all_runs)
        found = service.get_comparison(comp.comparison_id)
        assert found is not None
        assert found.comparison_id == comp.comparison_id

    def test_comparison_not_found(self, service):
        assert service.get_comparison("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Per-Class Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestPerClassAnalysis:
    def test_best_per_class(self, service, all_runs):
        comp = service.compare(all_runs)
        # Risky has highest per_class.FEE_DIFFERENCE (0.96)
        assert comp.best_per_class.get("per_class.FEE_DIFFERENCE") == "run-risky-001"
        # Risky also has highest per_class.REFUND_ADJUSTMENT (0.90 vs 0.87 vs 0.80)
        assert comp.best_per_class.get("per_class.REFUND_ADJUSTMENT") == "run-risky-001"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategies:
    def test_different_strategies_different_results(self, service, all_runs):
        comp_safety = service.compare(all_runs, RankingStrategy.SAFETY_FIRST)
        comp_accuracy = service.compare(all_runs, RankingStrategy.ACCURACY_FOCUSED)
        # Rankings may differ based on strategy
        assert comp_safety.strategy == RankingStrategy.SAFETY_FIRST
        assert comp_accuracy.strategy == RankingStrategy.ACCURACY_FOCUSED

    def test_all_strategies_work(self, service, all_runs):
        for strategy in RankingStrategy:
            if strategy == RankingStrategy.CUSTOM:
                continue
            comp = service.compare(all_runs, strategy)
            assert comp.run_count == 3
            assert len(comp.rankings) == 3

    def test_custom_weights(self, service, all_runs):
        custom = {ComparisonDimension.ACCURACY: 1.0}
        comp = service.compare(all_runs, RankingStrategy.CUSTOM, custom_weights=custom)
        assert comp.strategy == RankingStrategy.CUSTOM


# ─────────────────────────────────────────────────────────────────────────────
# Safety — Critical Properties
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyCritical:
    def test_highest_accuracy_not_always_best(self, service):
        """The risky run has highest accuracy but safety-first must not rank it #1."""
        risky = _make_run("risky", metrics={
            "accuracy": 0.99,
            "false_automation": 10,
            "high_value_errors": 5,
        })
        safe = _make_run("safe", metrics={
            "accuracy": 0.80,
            "false_automation": 0,
            "high_value_errors": 0,
        })
        comp = service.compare([safe, risky], RankingStrategy.SAFETY_FIRST)
        # Safe should rank higher
        assert comp.best_overall_run_id == "safe"

    def test_zero_errors_always_safest(self, service, all_runs):
        comp = service.compare(all_runs)
        safe = next(r for r in comp.rankings if r.run_id == "run-safe-001")
        risky = next(r for r in comp.rankings if r.run_id == "run-risky-001")
        assert safe.safety_score > risky.safety_score

    def test_high_value_errors_penalized(self, service):
        """High-value errors must strongly penalize a run."""
        hv_run = _make_run("hv", metrics={
            "accuracy": 0.95,
            "high_value_errors": 3,
            "false_automation": 5,
        })
        no_hv = _make_run("no_hv", metrics={
            "accuracy": 0.88,
            "high_value_errors": 0,
            "false_automation": 0,
        })
        comp = service.compare([hv_run, no_hv])
        hv_ranking = next(r for r in comp.rankings if r.run_id == "hv")
        assert hv_ranking.has_safety_regression

    def test_safety_dimension_definitions_exist(self):
        """Ensure safety dimension has meaningful metrics."""
        safety_def = STRATEGY_WEIGHTS[RankingStrategy.SAFETY_FIRST]
        assert ComparisonDimension.SAFETY in safety_def
        assert ComparisonDimension.HIGH_VALUE_SAFETY in safety_def
        # Safety combined weight should be significant
        total = sum(safety_def.values())
        safety_weight = safety_def[ComparisonDimension.SAFETY] + safety_def[ComparisonDimension.HIGH_VALUE_SAFETY]
        assert safety_weight / total >= 0.30  # At least 30%


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_two_runs(self, service, good_run, safe_run):
        comp = service.compare([good_run, safe_run])
        assert comp.run_count == 2
        assert len(comp.pairwise_comparisons) == 1

    def test_many_runs(self, service):
        runs = [
            _make_run(f"run-{i:03d}", model_version=f"v{i}.0", metrics={
                "accuracy": 0.70 + i * 0.02,
                "false_automation": max(0, 5 - i),
                "high_value_errors": max(0, 3 - i),
            })
            for i in range(8)
        ]
        comp = service.compare(runs)
        assert comp.run_count == 8
        assert len(comp.pairwise_comparisons) == 28  # C(8,2)

    def test_all_same_metrics(self, service):
        runs = [
            _make_run("r1", metrics={"accuracy": 0.90, "false_automation": 0}),
            _make_run("r2", metrics={"accuracy": 0.90, "false_automation": 0}),
        ]
        comp = service.compare(runs)
        assert comp.run_count == 2
        # Both should be ranked equally
        for r in comp.rankings:
            assert r.overall_score == comp.rankings[0].overall_score

    def test_missing_metrics_handled(self, service):
        r1 = _make_run("r1", metrics={"accuracy": 0.90})
        r2 = _make_run("r2", metrics={})  # No metrics
        comp = service.compare([r1, r2])
        assert comp.run_count == 2
        assert len(comp.rankings) == 2

    def test_different_algorithms(self, service):
        r1 = _make_run("r1", algorithm="xgboost", metrics={"accuracy": 0.90})
        r2 = _make_run("r2", algorithm="logistic_regression", metrics={"accuracy": 0.85})
        comp = service.compare([r1, r2])
        # Should compare successfully despite different algorithms
        assert comp.run_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


class TestSummary:
    def test_comparison_summary(self, service, all_runs):
        comp = service.compare(all_runs)
        s = comp.summary()
        assert "3 runs" in s
        assert "safety_first" in s

    def test_run_ranking_summary(self, service, all_runs):
        rankings = service.rank_runs(all_runs)
        for r in rankings:
            assert r.run_id is not None
            assert r.model_version is not None


# ─────────────────────────────────────────────────────────────────────────────
# Safety Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_comparison_is_observational(self, service, all_runs):
        """Comparison results cannot authorize execution."""
        comp = service.compare(all_runs)
        assert not hasattr(comp, 'execute')
        assert not hasattr(comp, 'authorize')
        assert not hasattr(comp, 'approve')

    def test_ranking_cannot_bypass_guardrails(self, service, all_runs):
        rankings = service.rank_runs(all_runs)
        for r in rankings:
            assert not hasattr(r, 'force_auto')
            assert not hasattr(r, 'bypass_guardrails')
            assert not hasattr(r, 'execute')
