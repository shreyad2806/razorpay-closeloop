"""
Experiment Comparison service for Razorpay CloseLoop Phase 10I.

Multi-dimensional comparison and ranking of MLflow training runs.

Goal: Answer which model is best across multiple dimensions,
with safety-first ranking that does NOT default to highest accuracy.

Safety principle:
  Comparison is OBSERVATIONAL ONLY.
  It never authorizes execution or bypasses Phase 6 guardrails.
"""

import hashlib
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

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
from app.schemas.mlflow_tracking import MetricsSnapshot, MLflowRunMetadata


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Dimension Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each dimension defines which metrics contribute and how.
# (metric_name, higher_is_better, is_safety_critical, weight)
DIMENSION_DEFINITIONS: Dict[ComparisonDimension, List[Tuple[str, bool, bool, float]]] = {
    ComparisonDimension.SAFETY: [
        ("false_automation", False, True, 0.3),
        ("high_value_errors", False, True, 0.3),
        ("verification_failure_rate", False, True, 0.2),
        ("unsafe_decision_rate", False, True, 0.2),
    ],
    ComparisonDimension.ACCURACY: [
        ("accuracy", True, False, 1.0),
    ],
    ComparisonDimension.PRECISION: [
        ("precision_macro", True, False, 1.0),
    ],
    ComparisonDimension.RECALL: [
        ("recall_macro", True, False, 1.0),
    ],
    ComparisonDimension.F1: [
        ("f1_macro", True, False, 1.0),
    ],
    ComparisonDimension.AUTOMATION_RATE: [
        ("automation_rate", True, False, 0.7),
        ("human_review_rate", False, False, 0.3),
    ],
    ComparisonDimension.FALSE_AUTOMATION: [
        ("false_automation", False, True, 0.6),
        ("incorrect_auto", False, True, 0.4),
    ],
    ComparisonDimension.RESOLUTION_ACCURACY: [
        ("resolution_accuracy", True, False, 1.0),
    ],
    ComparisonDimension.FINANCIAL_IMPACT: [
        ("total_error_impact_paise", False, True, 0.5),
        ("high_value_error_impact_paise", False, True, 0.5),
    ],
    ComparisonDimension.REWARD: [
        ("avg_reward", True, False, 1.0),
    ],
    ComparisonDimension.VERIFICATION: [
        ("verification_failure_rate", False, True, 1.0),
    ],
    ComparisonDimension.HUMAN_EFFICIENCY: [
        ("automation_rate", True, False, 0.5),
        ("human_review_rate", False, False, 0.3),
        ("unresolved_rate", False, False, 0.2),
    ],
    ComparisonDimension.HIGH_VALUE_SAFETY: [
        ("high_value_errors", False, True, 0.5),
        ("high_value_error_impact_paise", False, True, 0.5),
    ],
}

# Strategy weights: which dimensions matter most
STRATEGY_WEIGHTS: Dict[RankingStrategy, Dict[ComparisonDimension, float]] = {
    RankingStrategy.SAFETY_FIRST: {
        ComparisonDimension.SAFETY: 0.30,
        ComparisonDimension.HIGH_VALUE_SAFETY: 0.20,
        ComparisonDimension.ACCURACY: 0.15,
        ComparisonDimension.F1: 0.10,
        ComparisonDimension.RESOLUTION_ACCURACY: 0.10,
        ComparisonDimension.AUTOMATION_RATE: 0.05,
        ComparisonDimension.REWARD: 0.05,
        ComparisonDimension.HUMAN_EFFICIENCY: 0.05,
    },
    RankingStrategy.BALANCED: {
        ComparisonDimension.SAFETY: 0.15,
        ComparisonDimension.HIGH_VALUE_SAFETY: 0.10,
        ComparisonDimension.ACCURACY: 0.15,
        ComparisonDimension.F1: 0.10,
        ComparisonDimension.PRECISION: 0.05,
        ComparisonDimension.RECALL: 0.05,
        ComparisonDimension.RESOLUTION_ACCURACY: 0.10,
        ComparisonDimension.AUTOMATION_RATE: 0.10,
        ComparisonDimension.REWARD: 0.10,
        ComparisonDimension.HUMAN_EFFICIENCY: 0.10,
    },
    RankingStrategy.ACCURACY_FOCUSED: {
        ComparisonDimension.ACCURACY: 0.25,
        ComparisonDimension.F1: 0.20,
        ComparisonDimension.PRECISION: 0.10,
        ComparisonDimension.RECALL: 0.10,
        ComparisonDimension.SAFETY: 0.15,
        ComparisonDimension.HIGH_VALUE_SAFETY: 0.10,
        ComparisonDimension.RESOLUTION_ACCURACY: 0.10,
    },
    RankingStrategy.AUTOMATION_FOCUSED: {
        ComparisonDimension.AUTOMATION_RATE: 0.25,
        ComparisonDimension.HUMAN_EFFICIENCY: 0.20,
        ComparisonDimension.ACCURACY: 0.15,
        ComparisonDimension.SAFETY: 0.15,
        ComparisonDimension.HIGH_VALUE_SAFETY: 0.10,
        ComparisonDimension.RESOLUTION_ACCURACY: 0.10,
        ComparisonDimension.REWARD: 0.05,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentComparisonService:
    """Multi-dimensional experiment comparison and ranking.

    Compares MLflow runs across:
    - Safety (false automation, HV errors, verification failures)
    - Accuracy / Precision / Recall / F1
    - Automation efficiency
    - Financial impact
    - Reward

    Ranking is NEVER based on accuracy alone.
    Safety-first is the default strategy.
    """

    def __init__(self) -> None:
        self._comparisons: Dict[str, ExperimentComparison] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Run Preparation
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_run(
        metadata: MLflowRunMetadata,
        metrics: Optional[MetricsSnapshot] = None,
    ) -> ComparisonRun:
        """Prepare a ComparisonRun from MLflow metadata + metrics snapshot."""
        raw_metrics: Dict[str, float] = {}
        if metrics:
            raw_metrics = metrics.to_mlflow_dict()

        return ComparisonRun(
            run_id=metadata.run_id,
            run_name=metadata.run_name,
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            algorithm=metadata.algorithm,
            dataset_version=metadata.dataset_version,
            feature_schema_version=metadata.feature_schema_version,
            training_examples=metadata.training_examples,
            feature_count=metadata.feature_count,
            hyperparameters=metadata.hyperparameters,
            n_estimators=metadata.n_estimators,
            max_depth=metadata.max_depth,
            learning_rate=metadata.learning_rate,
            # Metrics from snapshot
            accuracy=metrics.accuracy if metrics else None,
            precision_macro=metrics.precision_macro if metrics else None,
            recall_macro=metrics.recall_macro if metrics else None,
            f1_macro=metrics.f1_macro if metrics else None,
            false_automation=metrics.false_automation if metrics else None,
            high_value_errors=metrics.high_value_errors if metrics else None,
            verification_failure_rate=metrics.verification_failure_rate if metrics else None,
            unsafe_decision_rate=metrics.unsafe_decision_rate if metrics else None,
            automation_rate=metrics.automation_rate if metrics else None,
            human_review_rate=metrics.human_review_rate if metrics else None,
            unresolved_rate=metrics.unresolved_rate if metrics else None,
            resolution_accuracy=metrics.resolution_accuracy if metrics else None,
            total_adjustment_paise=metrics.total_adjustment_paise if metrics else 0,
            total_error_impact_paise=metrics.total_error_impact_paise if metrics else 0,
            high_value_error_impact_paise=metrics.high_value_error_impact_paise if metrics else 0,
            avg_reward=metrics.avg_reward if metrics else None,
            per_class_f1=metrics.per_class_f1 if metrics else {},
            started_at=metadata.started_at,
            completed_at=metadata.completed_at,
            raw_metrics=raw_metrics,
        )

    @staticmethod
    def prepare_run_from_dict(
        run_id: str,
        model_name: str,
        model_version: str,
        algorithm: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_version: Optional[str] = None,
        **kwargs: Any,
    ) -> ComparisonRun:
        """Prepare a ComparisonRun from a simple dict (useful for testing)."""
        return ComparisonRun(
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            algorithm=algorithm,
            dataset_version=dataset_version,
            hyperparameters=hyperparameters or {},
            accuracy=metrics.get("accuracy"),
            precision_macro=metrics.get("precision_macro"),
            recall_macro=metrics.get("recall_macro"),
            f1_macro=metrics.get("f1_macro"),
            false_automation=int(metrics.get("false_automation", 0)),
            high_value_errors=int(metrics.get("high_value_errors", 0)),
            verification_failure_rate=metrics.get("verification_failure_rate"),
            unsafe_decision_rate=metrics.get("unsafe_decision_rate"),
            automation_rate=metrics.get("automation_rate"),
            human_review_rate=metrics.get("human_review_rate"),
            unresolved_rate=metrics.get("unresolved_rate"),
            resolution_accuracy=metrics.get("resolution_accuracy"),
            total_adjustment_paise=int(metrics.get("total_adjustment_paise", 0)),
            total_error_impact_paise=int(metrics.get("total_error_impact_paise", 0)),
            high_value_error_impact_paise=int(metrics.get("high_value_error_impact_paise", 0)),
            avg_reward=metrics.get("avg_reward"),
            per_class_f1={k: v for k, v in metrics.items() if k.startswith("per_class.")},
            raw_metrics=metrics,
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Pairwise Comparison
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_metric_value(run: ComparisonRun, metric_name: str) -> Optional[float]:
        """Get a metric value from a run."""
        # Try direct attribute first
        val = getattr(run, metric_name, None)
        if val is not None:
            return float(val)
        # Fall back to raw_metrics
        val = run.raw_metrics.get(metric_name)
        if val is not None:
            return float(val)
        return None

    @staticmethod
    def compare_pair(run_a: ComparisonRun, run_b: ComparisonRun) -> PairwiseComparison:
        """Compare two runs across all defined metrics."""
        metric_diffs: List[MetricDiff] = []
        a_wins = 0
        b_wins = 0
        ties = 0

        # Collect all metrics to compare
        all_metrics: Dict[str, Tuple[bool, bool]] = {}  # metric → (higher_is_better, is_safety)
        for dim_def in DIMENSION_DEFINITIONS.values():
            for metric_name, higher_better, is_safety, _weight in dim_def:
                all_metrics[metric_name] = (higher_better, is_safety)

        for metric_name, (higher_better, is_safety) in all_metrics.items():
            val_a = ExperimentComparisonService._get_metric_value(run_a, metric_name)
            val_b = ExperimentComparisonService._get_metric_value(run_b, metric_name)

            if val_a is None and val_b is None:
                continue

            diff_val = None
            pct_diff = None
            if val_a is not None and val_b is not None:
                diff_val = val_b - val_a
                if val_a != 0:
                    pct_diff = (diff_val / abs(val_a)) * 100

            is_tied = False
            winner = None
            if val_a is not None and val_b is not None:
                if val_a == val_b:
                    is_tied = True
                    ties += 1
                elif higher_better:
                    if val_b > val_a:
                        winner = run_b.run_id
                        b_wins += 1
                    else:
                        winner = run_a.run_id
                        a_wins += 1
                else:
                    # Lower is better
                    if val_b < val_a:
                        winner = run_b.run_id
                        b_wins += 1
                    else:
                        winner = run_a.run_id
                        a_wins += 1
            elif val_a is None:
                winner = run_b.run_id
                b_wins += 1
            elif val_b is None:
                winner = run_a.run_id
                a_wins += 1

            metric_diffs.append(MetricDiff(
                metric_name=metric_name,
                metric_display=metric_name.replace("_", " ").title(),
                higher_is_better=higher_better,
                is_safety_critical=is_safety,
                run_a_value=val_a,
                run_b_value=val_b,
                absolute_diff=diff_val,
                percentage_diff=pct_diff,
                winner=winner,
                is_tied=is_tied,
            ))

        # Safety notes
        safety_notes: List[str] = []
        a_fa = run_a.false_automation
        b_fa = run_b.false_automation
        if a_fa is not None and b_fa is not None:
            if b_fa > a_fa:
                safety_notes.append(
                    f"{run_b.model_version} has more false automations ({b_fa}) than {run_a.model_version} ({a_fa})"
                )
            elif a_fa > b_fa:
                safety_notes.append(
                    f"{run_a.model_version} has more false automations ({a_fa}) than {run_b.model_version} ({b_fa})"
                )

        a_hv = run_a.high_value_errors
        b_hv = run_b.high_value_errors
        if a_hv is not None and b_hv is not None:
            if b_hv > a_hv:
                safety_notes.append(
                    f"{run_b.model_version} has HIGH VALUE ERRORS ({b_hv} vs {a_hv})"
                )

        # Parameter diffs
        param_diffs: Dict[str, Dict[str, Any]] = {}
        all_param_keys = set(run_a.hyperparameters.keys()) | set(run_b.hyperparameters.keys())
        for key in all_param_keys:
            va = run_a.hyperparameters.get(key)
            vb = run_b.hyperparameters.get(key)
            if va != vb:
                param_diffs[key] = {"run_a": va, "run_b": vb}

        return PairwiseComparison(
            run_a_id=run_a.run_id,
            run_b_id=run_b.run_id,
            run_a_name=f"{run_a.model_name} v{run_a.model_version}",
            run_b_name=f"{run_b.model_name} v{run_b.model_version}",
            metric_diffs=metric_diffs,
            run_a_wins=a_wins,
            run_b_wins=b_wins,
            ties=ties,
            parameter_diffs=param_diffs,
            safety_notes=safety_notes,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Dimension Scoring
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_dimension(
        runs: List[ComparisonRun], dimension: ComparisonDimension
    ) -> Dict[str, DimensionScore]:
        """Score all runs on a single dimension."""
        if dimension not in DIMENSION_DEFINITIONS:
            return {}

        dim_def = DIMENSION_DEFINITIONS[dimension]
        scores: Dict[str, Dict[str, float]] = {}  # run_id → {metric: normalized_value}

        for run in runs:
            run_scores: Dict[str, float] = {}
            for metric_name, higher_better, _, weight in dim_def:
                val = ExperimentComparisonService._get_metric_value(run, metric_name)
                if val is not None:
                    run_scores[metric_name] = val
            scores[run.run_id] = run_scores

        # Normalize each metric across runs (min-max)
        normalized: Dict[str, Dict[str, float]] = {}
        for metric_name, higher_better, _, _ in dim_def:
            values = [
                scores[rid].get(metric_name)
                for rid in scores
                if scores[rid].get(metric_name) is not None
            ]
            if not values:
                continue
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            for rid in scores:
                val = scores[rid].get(metric_name)
                if val is None:
                    normalized.setdefault(rid, {})[metric_name] = 0.0
                elif range_val == 0:
                    normalized.setdefault(rid, {})[metric_name] = 1.0
                else:
                    if higher_better:
                        norm = (val - min_val) / range_val
                    else:
                        norm = (max_val - val) / range_val
                    normalized.setdefault(rid, {})[metric_name] = max(0.0, min(1.0, norm))

        # Compute weighted score per run
        result: Dict[str, DimensionScore] = {}
        for run in runs:
            rid = run.run_id
            weighted_sum = 0.0
            total_weight = 0.0
            contributing: Dict[str, float] = {}

            for metric_name, _, _, weight in dim_def:
                norm_val = normalized.get(rid, {}).get(metric_name, 0.0)
                weighted_sum += norm_val * weight
                total_weight += weight
                contributing[metric_name] = norm_val

            score = weighted_sum / total_weight if total_weight > 0 else 0.0

            result[rid] = DimensionScore(
                dimension=dimension,
                score=round(score, 4),
                contributing_metrics=contributing,
            )

        # Assign ranks
        sorted_runs = sorted(result.items(), key=lambda x: x[1].score, reverse=True)
        for rank_idx, (rid, dscore) in enumerate(sorted_runs):
            dscore.rank = rank_idx + 1
            n = len(sorted_runs)
            if n == 1:
                dscore.position = RunPosition.BEST
            elif rank_idx == 0:
                dscore.position = RunPosition.BEST
            elif rank_idx == n - 1:
                dscore.position = RunPosition.WORST
            elif rank_idx == 1 and n == 2:
                dscore.position = RunPosition.SECOND
            elif rank_idx == 1 and n > 2:
                dscore.position = RunPosition.SECOND
            elif rank_idx == 2 and n == 3:
                dscore.position = RunPosition.THIRD
            else:
                dscore.position = RunPosition.MIDDLE

            # Check for ties (same score as neighbor)
            if rank_idx > 0 and abs(dscore.score - sorted_runs[rank_idx - 1][1].score) < 0.001:
                dscore.position = RunPosition.TIED

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Overall Ranking
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def rank_runs(
        runs: List[ComparisonRun],
        strategy: RankingStrategy = RankingStrategy.SAFETY_FIRST,
        custom_weights: Optional[Dict[ComparisonDimension, float]] = None,
    ) -> List[RunRanking]:
        """Rank all runs using the specified strategy."""
        if not runs:
            return []

        strategy_weights = custom_weights or STRATEGY_WEIGHTS.get(
            strategy, STRATEGY_WEIGHTS[RankingStrategy.SAFETY_FIRST]
        )

        # Score all dimensions
        all_dimension_scores: Dict[ComparisonDimension, Dict[str, DimensionScore]] = {}
        for dim in ComparisonDimension:
            if dim == ComparisonDimension.OVERALL:
                continue
            all_dimension_scores[dim] = ExperimentComparisonService._score_dimension(runs, dim)

        # Build run rankings
        rankings: List[RunRanking] = []
        for run in runs:
            rid = run.run_id
            dim_scores: List[DimensionScore] = []
            weighted_sum = 0.0
            total_weight = 0.0

            for dim, weight in strategy_weights.items():
                dscore = all_dimension_scores.get(dim, {}).get(rid)
                if dscore:
                    dim_scores.append(dscore)
                    weighted_sum += dscore.score * weight
                    total_weight += weight

            overall = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Safety score
            safety_dim = all_dimension_scores.get(ComparisonDimension.SAFETY, {}).get(rid)
            hv_safety = all_dimension_scores.get(ComparisonDimension.HIGH_VALUE_SAFETY, {}).get(rid)
            safety_score = 0.0
            if safety_dim and hv_safety:
                safety_score = (safety_dim.score * 0.6 + hv_safety.score * 0.4)
            elif safety_dim:
                safety_score = safety_dim.score

            # Safety issues
            safety_issues: List[str] = []
            if run.false_automation and run.false_automation > 0:
                safety_issues.append(f"False automations: {run.false_automation}")
            if run.high_value_errors and run.high_value_errors > 0:
                safety_issues.append(f"HIGH VALUE ERRORS: {run.high_value_errors}")
            if run.verification_failure_rate and run.verification_failure_rate > 0.05:
                safety_issues.append(f"Verification failure rate: {run.verification_failure_rate:.1%}")

            rankings.append(RunRanking(
                run_id=rid,
                run_name=run.run_name or f"{run.model_name} v{run.model_version}",
                model_name=run.model_name,
                model_version=run.model_version,
                dimension_scores=dim_scores,
                overall_score=round(overall, 4),
                safety_score=round(safety_score, 4),
                has_safety_regression=len(safety_issues) > 0,
                safety_issues=safety_issues,
            ))

        # Sort by overall score (safety-penalized: subtract safety penalty)
        for r in rankings:
            if r.has_safety_regression:
                r.overall_score = max(0.0, r.overall_score - 0.2)

        rankings.sort(key=lambda x: x.overall_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(rankings):
            r.overall_rank = i + 1
            # Safety rank
        safety_sorted = sorted(rankings, key=lambda x: x.safety_score, reverse=True)
        for i, r in enumerate(safety_sorted):
            r.safety_rank = i + 1

        return rankings

    # ─────────────────────────────────────────────────────────────────────
    # Full Comparison
    # ─────────────────────────────────────────────────────────────────────

    def compare(
        self,
        runs: List[ComparisonRun],
        strategy: RankingStrategy = RankingStrategy.SAFETY_FIRST,
        custom_weights: Optional[Dict[ComparisonDimension, float]] = None,
    ) -> ExperimentComparison:
        """Perform a complete multi-dimensional comparison of multiple runs."""
        if len(runs) < 2:
            raise ValueError("Need at least 2 runs to compare")

        comp_id = _gen_id("CMP")

        # Pairwise comparisons
        pairwise: List[PairwiseComparison] = []
        for a, b in combinations(runs, 2):
            pairwise.append(self.compare_pair(a, b))

        # Rankings
        rankings = self.rank_runs(runs, strategy, custom_weights)

        # Answer questions
        safest = max(rankings, key=lambda r: r.safety_score) if rankings else None
        best_overall = rankings[0] if rankings else None

        # Best per accuracy
        accuracy_sorted = sorted(
            [r for r in rankings if any(
                ds.dimension == ComparisonDimension.ACCURACY for ds in r.dimension_scores
            )],
            key=lambda r: next(
                ds.score for ds in r.dimension_scores
                if ds.dimension == ComparisonDimension.ACCURACY
            ),
            reverse=True,
        )
        most_accurate = accuracy_sorted[0] if accuracy_sorted else None

        # Best automation
        auto_sorted = sorted(
            [r for r in rankings if any(
                ds.dimension == ComparisonDimension.AUTOMATION_RATE for ds in r.dimension_scores
            )],
            key=lambda r: next(
                ds.score for ds in r.dimension_scores
                if ds.dimension == ComparisonDimension.AUTOMATION_RATE
            ),
            reverse=True,
        )
        best_auto = auto_sorted[0] if auto_sorted else None

        # Best resolution
        res_sorted = sorted(
            [r for r in rankings if any(
                ds.dimension == ComparisonDimension.RESOLUTION_ACCURACY for ds in r.dimension_scores
            )],
            key=lambda r: next(
                ds.score for ds in r.dimension_scores
                if ds.dimension == ComparisonDimension.RESOLUTION_ACCURACY
            ),
            reverse=True,
        )
        best_res = res_sorted[0] if res_sorted else None

        # Best reward
        rew_sorted = sorted(
            [r for r in rankings if any(
                ds.dimension == ComparisonDimension.REWARD for ds in r.dimension_scores
            )],
            key=lambda r: next(
                ds.score for ds in r.dimension_scores
                if ds.dimension == ComparisonDimension.REWARD
            ),
            reverse=True,
        )
        best_rew = rew_sorted[0] if rew_sorted else None

        # Best per exception class (per_class_f1)
        best_per_class: Dict[str, str] = {}
        all_classes: set = set()
        for run in runs:
            all_classes.update(run.per_class_f1.keys())
        for cls in all_classes:
            best_score = -1.0
            best_rid = ""
            for run in runs:
                score = run.per_class_f1.get(cls, -1.0)
                if score > best_score:
                    best_score = score
                    best_rid = run.run_id
            if best_rid:
                best_per_class[cls] = best_rid

        # Best per dataset
        best_per_dataset: Dict[str, str] = {}
        datasets: set = set()
        for run in runs:
            if run.dataset_version:
                datasets.add(run.dataset_version)
        for ds_ver in datasets:
            ds_runs = [r for r in runs if r.dataset_version == ds_ver]
            if ds_runs and rankings:
                best_rid = max(
                    ds_runs,
                    key=lambda r: next(
                        (rk.overall_score for rk in rankings if rk.run_id == r.run_id),
                        0.0,
                    ),
                ).run_id
                best_per_dataset[ds_ver] = best_rid

        # Safety warnings
        all_safe = True
        safety_warnings: List[str] = []
        for r in rankings:
            if r.has_safety_regression:
                all_safe = False
                for issue in r.safety_issues:
                    safety_warnings.append(f"{r.model_version}: {issue}")

        # Insights
        insights: List[str] = []
        if rankings:
            best = rankings[0]
            worst = rankings[-1]
            insights.append(
                f"Best overall: {best.model_version} v{best.model_version} "
                f"(score: {best.overall_score:.3f})"
            )
            if safest:
                insights.append(f"Safest: {safest.model_version} (safety: {safest.safety_score:.3f})")
            if most_accurate and most_accurate.run_id != best.run_id:
                insights.append(
                    f"Most accurate ({most_accurate.model_version}) is NOT the best overall — "
                    f"safety and other factors matter"
                )
            if safety_warnings:
                insights.append(f"⚠ {len(safety_warnings)} safety warning(s) across runs")

        comparison = ExperimentComparison(
            comparison_id=comp_id,
            strategy=strategy,
            runs=runs,
            run_count=len(runs),
            pairwise_comparisons=pairwise,
            rankings=rankings,
            safest_run_id=safest.run_id if safest else None,
            most_accurate_run_id=most_accurate.run_id if most_accurate else None,
            best_automation_run_id=best_auto.run_id if best_auto else None,
            best_resolution_run_id=best_res.run_id if best_res else None,
            best_reward_run_id=best_rew.run_id if best_rew else None,
            best_overall_run_id=best_overall.run_id if best_overall else None,
            best_per_class=best_per_class,
            best_per_dataset=best_per_dataset,
            all_runs_safe=all_safe,
            safety_warnings=safety_warnings,
            insights=insights,
        )

        self._comparisons[comp_id] = comparison
        return comparison

    def get_comparison(self, comparison_id: str) -> Optional[ExperimentComparison]:
        """Get a stored comparison by ID."""
        return self._comparisons.get(comparison_id)
