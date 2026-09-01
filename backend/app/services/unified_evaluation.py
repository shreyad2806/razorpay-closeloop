"""
Unified Evaluation service for Razorpay CloseLoop Phase 10F.

Runs the complete evaluation pipeline:
  Candidate model → Classification → Resolution → Safety → Automation → Financial → MLflow

Compares CURRENT vs CANDIDATE model with comprehensive safety checks.

Safety principle:
  Evaluation results are OBSERVATIONAL ONLY.
  They never authorize execution or bypass Phase 6 guardrails.
  Phase 6 hard safety constraints remain mandatory.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.mlflow_tracking import ArtifactType
from app.schemas.model_training import EvaluationMetrics
from app.schemas.unified_evaluation import (
    EvaluationVerdict,
    MetricComparison,
    SafetyRegressionCheck,
    SafetyRegressionSeverity,
    UnifiedEvaluationReport,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Thresholds (configurable)
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationThresholds:
    """Configurable thresholds for evaluation safety checks."""

    def __init__(
        self,
        max_false_automation_increase: int = 0,
        max_high_value_error_increase: int = 0,
        max_verification_failure_rate: float = 0.10,
        min_accuracy: float = 0.50,
        min_f1_macro: float = 0.40,
        min_precision_macro: float = 0.40,
        max_false_auto_increase_pct: float = 20.0,
    ) -> None:
        self.max_false_automation_increase = max_false_automation_increase
        self.max_high_value_error_increase = max_high_value_error_increase
        self.max_verification_failure_rate = max_verification_failure_rate
        self.min_accuracy = min_accuracy
        self.min_f1_macro = min_f1_macro
        self.min_precision_macro = min_precision_macro
        self.max_false_auto_increase_pct = max_false_auto_increase_pct


# ─────────────────────────────────────────────────────────────────────────────
# Unified Evaluation Service
# ─────────────────────────────────────────────────────────────────────────────


class UnifiedEvaluationService:
    """Runs the complete evaluation pipeline.

    Pipeline:
      Candidate model
        → Classification evaluation
        → Resolution evaluation
        → Safety evaluation
        → Automation evaluation
        → Financial evaluation
        → MLflow metrics
        → Evaluation report artifact
    """

    def __init__(
        self,
        thresholds: Optional[EvaluationThresholds] = None,
    ) -> None:
        self._thresholds = thresholds or EvaluationThresholds()

    @property
    def thresholds(self) -> EvaluationThresholds:
        return self._thresholds

    # ─────────────────────────────────────────────────────────────────────
    # Core Evaluation: Build Report
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        current_metrics: Optional[EvaluationMetrics],
        candidate_metrics: EvaluationMetrics,
        current_model_id: Optional[str] = None,
        current_model_version: Optional[str] = None,
        candidate_model_id: Optional[str] = None,
        candidate_model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> UnifiedEvaluationReport:
        """Run the complete evaluation pipeline.

        Compares current vs candidate model across all metric categories.
        Returns a comprehensive report with safety checks and verdict.
        """
        report_id = _gen_id("EVAL")

        # ── Classification comparison ───────────────────────────────────
        comparisons = self._compare_classification(current_metrics, candidate_metrics)

        # ── Safety checks ───────────────────────────────────────────────
        safety_checks = self._check_safety(current_metrics, candidate_metrics)
        all_safety_passed = all(s.passed for s in safety_checks)
        max_severity = self._max_safety_severity(safety_checks)

        # ── Improvements / regressions ──────────────────────────────────
        improvements = [c.metric_name for c in comparisons if c.is_improvement]
        regressions = [c.metric_name for c in comparisons if not c.is_improvement and c.change is not None and c.change != 0]

        # ── Verdict ─────────────────────────────────────────────────────
        verdict, verdict_reason, eligible = self._determine_verdict(
            current_metrics, candidate_metrics,
            all_safety_passed, max_severity,
            improvements, regressions,
        )

        # ── Build report ────────────────────────────────────────────────
        report = UnifiedEvaluationReport(
            report_id=report_id,
            current_model_id=current_model_id,
            current_model_version=current_model_version,
            candidate_model_id=candidate_model_id or "",
            candidate_model_version=candidate_model_version or "",
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            mlflow_run_id=mlflow_run_id,
            # Classification
            current_accuracy=current_metrics.accuracy if current_metrics else None,
            candidate_accuracy=candidate_metrics.accuracy,
            current_f1_macro=current_metrics.f1_macro if current_metrics else None,
            candidate_f1_macro=candidate_metrics.f1_macro,
            current_precision_macro=current_metrics.precision_macro if current_metrics else None,
            candidate_precision_macro=candidate_metrics.precision_macro,
            current_recall_macro=current_metrics.recall_macro if current_metrics else None,
            candidate_recall_macro=candidate_metrics.recall_macro,
            # Safety
            current_false_automation=current_metrics.false_automation if current_metrics else 0,
            candidate_false_automation=candidate_metrics.false_automation,
            current_high_value_errors=current_metrics.high_value_errors if current_metrics else 0,
            candidate_high_value_errors=candidate_metrics.high_value_errors,
            current_verification_failure_rate=(
                current_metrics.verification_failure_rate if current_metrics else None
            ),
            candidate_verification_failure_rate=candidate_metrics.verification_failure_rate,
            # Resolution
            current_resolution_accuracy=(
                current_metrics.resolution_accuracy if current_metrics else None
            ),
            candidate_resolution_accuracy=candidate_metrics.resolution_accuracy,
            # Comparisons
            comparisons=comparisons,
            improvements=improvements,
            regressions=regressions,
            # Safety
            safety_checks=safety_checks,
            all_safety_passed=all_safety_passed,
            safety_regression_severity=max_severity,
            # Verdict
            verdict=verdict,
            verdict_reason=verdict_reason,
            promotion_eligible=eligible,
            # Summary
            total_improvements=len(improvements),
            total_regressions=len(regressions),
            safety_checks_passed=sum(1 for s in safety_checks if s.passed),
            safety_checks_failed=sum(1 for s in safety_checks if not s.passed),
        )

        return report

    # ─────────────────────────────────────────────────────────────────────
    # Classification Comparison
    # ─────────────────────────────────────────────────────────────────────

    def _compare_classification(
        self,
        current: Optional[EvaluationMetrics],
        candidate: EvaluationMetrics,
    ) -> List[MetricComparison]:
        """Compare classification metrics between current and candidate."""
        comparisons: List[MetricComparison] = []

        metric_defs = [
            ("accuracy", "accuracy", True, False),
            ("f1_macro", "f1_macro", True, False),
            ("precision_macro", "precision_macro", True, False),
            ("recall_macro", "recall_macro", True, False),
            ("f1_weighted", "f1_weighted", True, False),
            ("precision_weighted", "precision_weighted", True, False),
            ("recall_weighted", "recall_weighted", True, False),
            ("false_automation", "false_automation", False, True),
            ("high_value_errors", "high_value_errors", False, True),
            ("unknown_case_errors", "unknown_case_errors", False, True),
            ("incorrect_auto_resolution", "incorrect_auto_resolution", False, True),
            ("resolution_accuracy", "resolution_accuracy", True, False),
        ]

        for display_name, attr_name, higher_is_better, is_safety in metric_defs:
            cur_val = getattr(current, attr_name, None) if current else None
            cand_val = getattr(candidate, attr_name, None)

            if cur_val is None and cand_val is None:
                continue

            change = None
            change_pct = None
            is_improvement = False

            if cur_val is not None and cand_val is not None:
                change = cand_val - cur_val
                if cur_val != 0:
                    change_pct = (change / abs(cur_val)) * 100
                if higher_is_better:
                    is_improvement = change > 0
                else:
                    is_improvement = change < 0
            elif cand_val is not None:
                # No current value — candidate is new baseline
                is_improvement = True

            comparisons.append(MetricComparison(
                metric_name=display_name,
                current_value=cur_val,
                candidate_value=cand_val,
                change=change,
                change_pct=change_pct,
                is_improvement=is_improvement,
                is_safety_critical=is_safety,
                higher_is_better=higher_is_better,
            ))

        return comparisons

    # ─────────────────────────────────────────────────────────────────────
    # Safety Checks
    # ─────────────────────────────────────────────────────────────────────

    def _check_safety(
        self,
        current: Optional[EvaluationMetrics],
        candidate: EvaluationMetrics,
    ) -> List[SafetyRegressionCheck]:
        """Run safety regression checks."""
        checks: List[SafetyRegressionCheck] = []
        t = self._thresholds

        # When no current model exists, safety checks are informational only
        # (no regression possible against a non-existent baseline)
        has_current = current is not None

        # 1. False automation must not increase beyond threshold
        cur_fa = current.false_automation if current else 0
        cand_fa = candidate.false_automation
        fa_passed = (not has_current) or (cand_fa <= cur_fa + t.max_false_automation_increase)
        fa_severity = (
            SafetyRegressionSeverity.NONE if fa_passed
            else SafetyRegressionSeverity.CRITICAL if cand_fa > cur_fa + 5
            else SafetyRegressionSeverity.MODERATE
        )
        checks.append(SafetyRegressionCheck(
            metric_name="false_automation",
            current_value=float(cur_fa),
            candidate_value=float(cand_fa),
            passed=fa_passed,
            severity=fa_severity,
            description=f"False automation: {cur_fa} → {cand_fa} (max increase: {t.max_false_automation_increase})",
        ))

        # 2. High-value errors must not increase
        cur_hv = current.high_value_errors if current else 0
        cand_hv = candidate.high_value_errors
        hv_passed = (not has_current) or (cand_hv <= cur_hv + t.max_high_value_error_increase)
        hv_severity = (
            SafetyRegressionSeverity.NONE if hv_passed
            else SafetyRegressionSeverity.CRITICAL
        )
        checks.append(SafetyRegressionCheck(
            metric_name="high_value_errors",
            current_value=float(cur_hv),
            candidate_value=float(cand_hv),
            passed=hv_passed,
            severity=hv_severity,
            description=f"HV errors: {cur_hv} → {cand_hv}",
        ))

        # 3. Accuracy must meet minimum
        acc_passed = candidate.accuracy >= t.min_accuracy
        checks.append(SafetyRegressionCheck(
            metric_name="accuracy",
            current_value=current.accuracy if current else None,
            candidate_value=candidate.accuracy,
            passed=acc_passed,
            severity=SafetyRegressionSeverity.NONE if acc_passed else SafetyRegressionSeverity.CRITICAL,
            description=f"Accuracy {candidate.accuracy:.1%} vs min {t.min_accuracy:.1%}",
        ))

        # 4. F1 macro must meet minimum
        f1_passed = candidate.f1_macro >= t.min_f1_macro
        checks.append(SafetyRegressionCheck(
            metric_name="f1_macro",
            current_value=current.f1_macro if current else None,
            candidate_value=candidate.f1_macro,
            passed=f1_passed,
            severity=SafetyRegressionSeverity.NONE if f1_passed else SafetyRegressionSeverity.MODERATE,
            description=f"F1 {candidate.f1_macro:.1%} vs min {t.min_f1_macro:.1%}",
        ))

        # 5. Verification failure rate must not exceed threshold
        if candidate.verification_failure_rate is not None:
            vf_passed = candidate.verification_failure_rate <= t.max_verification_failure_rate
            checks.append(SafetyRegressionCheck(
                metric_name="verification_failure_rate",
                current_value=current.verification_failure_rate if current else None,
                candidate_value=candidate.verification_failure_rate,
                passed=vf_passed,
                severity=SafetyRegressionSeverity.NONE if vf_passed else SafetyRegressionSeverity.MODERATE,
                description=f"Ver fail rate {candidate.verification_failure_rate:.1%} vs max {t.max_verification_failure_rate:.1%}",
            ))

        return checks

    def _max_safety_severity(
        self, checks: List[SafetyRegressionCheck]
    ) -> SafetyRegressionSeverity:
        """Get the maximum severity from all safety checks."""
        severity_order = [
            SafetyRegressionSeverity.NONE,
            SafetyRegressionSeverity.MINOR,
            SafetyRegressionSeverity.MODERATE,
            SafetyRegressionSeverity.CRITICAL,
        ]
        max_idx = 0
        for check in checks:
            if not check.passed:
                idx = severity_order.index(check.severity)
                max_idx = max(max_idx, idx)
        return severity_order[max_idx]

    # ─────────────────────────────────────────────────────────────────────
    # Verdict Determination
    # ─────────────────────────────────────────────────────────────────────

    def _determine_verdict(
        self,
        current: Optional[EvaluationMetrics],
        candidate: EvaluationMetrics,
        all_safety_passed: bool,
        max_severity: SafetyRegressionSeverity,
        improvements: List[str],
        regressions: List[str],
    ) -> tuple[EvaluationVerdict, str, bool]:
        """Determine promotion verdict based on all signals."""
        # Critical safety failure → REJECT
        if max_severity == SafetyRegressionSeverity.CRITICAL:
            return (
                EvaluationVerdict.REJECT,
                "Critical safety regression detected",
                False,
            )

        # Safety checks failed → DEFER
        if not all_safety_passed:
            return (
                EvaluationVerdict.DEFER,
                f"Safety checks failed ({sum(1 for s in [max_severity] if s != SafetyRegressionSeverity.NONE)} regressions)",
                False,
            )

        # No current model → PROMOTE if candidate meets minimum quality
        if current is None:
            if candidate.accuracy >= self._thresholds.min_accuracy and all_safety_passed:
                return (
                    EvaluationVerdict.PROMOTE,
                    "No current model — first model promotion",
                    True,
                )
            return (
                EvaluationVerdict.DEFER,
                "No current model but candidate below quality threshold",
                False,
            )

        # More regressions than improvements → DEFER
        if len(regressions) > len(improvements):
            return (
                EvaluationVerdict.DEFER,
                f"More regressions ({len(regressions)}) than improvements ({len(improvements)})",
                False,
            )

        # Equal → DEFER
        if len(improvements) == len(regressions) and len(improvements) > 0:
            return (
                EvaluationVerdict.DEFER,
                "Equal improvements and regressions",
                False,
            )

        # More improvements → PROMOTE
        if len(improvements) > len(regressions):
            return (
                EvaluationVerdict.PROMOTE,
                f"Net improvement: {len(improvements)} improvements, {len(regressions)} regressions, all safety checks passed",
                True,
            )

        # No changes → DEFER
        return (
            EvaluationVerdict.DEFER,
            "No meaningful changes detected",
            False,
        )

    # ─────────────────────────────────────────────────────────────────────
    # MLflow Integration
    # ─────────────────────────────────────────────────────────────────────

    def log_to_mlflow(
        self,
        report: UnifiedEvaluationReport,
        tracking_service: Any,  # MLflowTrackingService
        run_id: Optional[str] = None,
    ) -> None:
        """Log the evaluation report to MLflow.

        Logs metrics, parameters, and the full report as an artifact.
        """
        target_run_id = run_id or report.mlflow_run_id
        if target_run_id is None:
            return

        # Log key metrics
        metrics_dict: Dict[str, float] = {}
        if report.current_accuracy is not None:
            metrics_dict["eval.current_accuracy"] = report.current_accuracy
        if report.candidate_accuracy is not None:
            metrics_dict["eval.candidate_accuracy"] = report.candidate_accuracy
        if report.current_f1_macro is not None:
            metrics_dict["eval.current_f1_macro"] = report.current_f1_macro
        if report.candidate_f1_macro is not None:
            metrics_dict["eval.candidate_f1_macro"] = report.candidate_f1_macro
        metrics_dict["eval.current_false_automation"] = float(report.current_false_automation)
        metrics_dict["eval.candidate_false_automation"] = float(report.candidate_false_automation)
        metrics_dict["eval.total_improvements"] = float(report.total_improvements)
        metrics_dict["eval.total_regressions"] = float(report.total_regressions)
        metrics_dict["eval.safety_checks_passed"] = float(report.safety_checks_passed)
        metrics_dict["eval.safety_checks_failed"] = float(report.safety_checks_failed)

        # Log via metrics snapshot
        from app.schemas.mlflow_tracking import MetricsSnapshot
        snapshot = MetricsSnapshot(
            run_id=target_run_id,
            custom_metrics=metrics_dict,
        )
        tracking_service.log_metrics(target_run_id, snapshot)

        # Log the full report as an artifact
        tracking_service.log_json_artifact(
            run_id=target_run_id,
            artifact_type=ArtifactType.EVALUATION_REPORT,
            artifact_name=f"evaluation_report_{report.report_id}.json",
            data=report.to_report_dict(),
            description=f"Unified evaluation: {report.candidate_model_version} vs {report.current_model_version or 'baseline'}",
        )
