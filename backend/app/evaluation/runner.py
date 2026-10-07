"""
Evaluation runner (Phase 15).

Orchestrates every evaluator into one reproducible report. Nothing here is
imported by the production request path, and no input is mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.ml_dataset import FEATURE_SCHEMA_VERSION

from app.evaluation.artifacts import new_run_id, write_artifacts
from app.evaluation.dataset import EvaluationDataset, build_synthetic_dataset_v1
from app.evaluation.disagreement import DisagreementReport, measure_disagreement
from app.evaluation.evidence_eval import EvidenceReport, evaluate_evidence
from app.evaluation.golden import (
    DB_SCENARIOS,
    GoldenResult,
    closure_metrics,
    run_stage_scenarios,
)
from app.evaluation.leakage import LeakageReport, audit_dataset
from app.evaluation.ml_eval import MLClassificationReport, evaluate_classifier
from app.evaluation.policy_eval import PolicyReport, evaluate_policy
from app.evaluation.resolution_eval import ResolutionReport, evaluate_resolution
from app.evaluation.retrieval_eval import RetrievalReport, evaluate_retrieval
from app.evaluation.safety import NA_LABEL, SafetyReport, evaluate_safety
from app.evaluation.versioning import (
    DEFAULT_EVAL_SEED,
    EVALUATOR_VERSION,
    SYNTHETIC_DATASET_SCHEMA_VERSION,
)


def _na(value: Any) -> Any:
    """Label an undefined rate instead of printing a misleading zero."""
    return NA_LABEL if value is None else value


class EvaluationRunReport(BaseModel):
    """Complete Phase 15 evaluation report."""

    evaluator_version: str = EVALUATOR_VERSION
    run_id: str = ""
    timestamp: str = ""
    seed: int = DEFAULT_EVAL_SEED

    dataset_version: str = ""
    dataset_schema_version: str = SYNTHETIC_DATASET_SCHEMA_VERSION
    dataset_source: str = ""
    dataset_fingerprint: str = ""
    dataset_cases: int = 0
    embedding_version: str = "not-applicable (no embedding model loaded)"
    model_version: str = "exception_classifier-xgboost"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    ml: Dict[str, Any] = Field(default_factory=dict)
    disagreement: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    resolution: Dict[str, Any] = Field(default_factory=dict)
    safety: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)
    golden: Dict[str, Any] = Field(default_factory=dict)
    leakage: Dict[str, Any] = Field(default_factory=dict)

    notes: List[str] = Field(default_factory=list)

    def headline(self) -> Dict[str, Any]:
        """The numbers that matter most for a safety review.

        Zero-denominator rates are rendered as "N/A — no applicable cases".
        """
        safety = self.safety or {}
        ml = self.ml or {}
        return {
            "dataset_version": self.dataset_version,
            "dataset_source": self.dataset_source,
            "dataset_cases": self.dataset_cases,
            "leakage_free": (self.leakage or {}).get("leakage_free"),
            "accuracy": ml.get("accuracy"),
            "macro_f1": ml.get("f1_macro"),
            "automation_eligible": safety.get("automation_eligible"),
            "unsafe_automation_count": safety.get("unsafe_automation_count"),
            "unsafe_resolution_rate": _na(safety.get("unsafe_resolution_rate")),
            "safe_resolution_precision": _na(safety.get("safe_resolution_precision")),
            "human_review_rate": _na(safety.get("human_review_rate")),
            "human_review_recall": _na(safety.get("human_review_recall")),
            "abstention_rate": _na(safety.get("abstention_rate")),
            "closure_accuracy": _na(safety.get("closure_accuracy")),
            "golden_passed": (self.golden or {}).get("passed"),
            "golden_total": (self.golden or {}).get("total"),
        }


def _core_evaluation(
    dataset: EvaluationDataset,
    closure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every evaluator. Pure with respect to production state."""
    ml: MLClassificationReport = evaluate_classifier(dataset)
    disagreement: DisagreementReport = measure_disagreement(dataset)
    retrieval: RetrievalReport = evaluate_retrieval(dataset)
    resolution: ResolutionReport = evaluate_resolution(dataset)
    safety: SafetyReport = evaluate_safety(dataset, resolution, closure=closure)
    evidence: EvidenceReport = evaluate_evidence(dataset, resolution=resolution)
    policy: PolicyReport = evaluate_policy()
    leakage: LeakageReport = audit_dataset(dataset)

    return {
        "ml": ml,
        "disagreement": disagreement,
        "retrieval": retrieval,
        "resolution": resolution,
        "safety": safety,
        "evidence": evidence,
        "policy": policy,
        "leakage": leakage,
    }


def run_evaluation(
    per_family: int = 20,
    seed: int = DEFAULT_EVAL_SEED,
    session: Any = None,
    write: bool = False,
    output_dir: Any = None,
) -> EvaluationRunReport:
    """Run the full Phase 15 evaluation.

    ``session`` (optional) enables the database-backed golden scenarios. Pass a
    callable to have a fresh session created per scenario — the scenarios share
    well-known record ids, so they are only isolated if they do not share a
    database.
    """
    dataset = build_synthetic_dataset_v1(per_family=per_family, seed=seed)

    golden_results: List[GoldenResult] = run_stage_scenarios()
    if session is not None:
        session_factory = session if callable(session) else (lambda: session)
        for scenario in DB_SCENARIOS:
            try:
                golden_results.append(scenario(session_factory()))
            except Exception as exc:  # pragma: no cover - defensive
                golden_results.append(
                    GoldenResult(
                        scenario_id=scenario.__name__,
                        title=scenario.__name__,
                        passed=False,
                        expected="scenario executes",
                        failures=[f"{type(exc).__name__}: {exc}"],
                    )
                )

    core = _core_evaluation(dataset, closure=closure_metrics(golden_results))

    golden = {
        "evaluator_version": EVALUATOR_VERSION,
        "total": len(golden_results),
        "passed": sum(1 for r in golden_results if r.passed),
        "failed": sum(1 for r in golden_results if not r.passed),
        "closure": closure_metrics(golden_results),
        "results": [r.model_dump() for r in golden_results],
    }

    report = EvaluationRunReport(
        run_id=new_run_id(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        dataset_version=dataset.dataset_version,
        dataset_source=dataset.source,
        dataset_fingerprint=dataset.fingerprint,
        dataset_cases=dataset.case_count,
        ml=core["ml"].model_dump(),
        disagreement=core["disagreement"].summary(),
        retrieval=core["retrieval"].model_dump(),
        resolution=core["resolution"].summary(),
        safety=core["safety"].model_dump(),
        evidence=core["evidence"].summary(),
        policy=core["policy"].summary(),
        golden=golden,
        leakage=core["leakage"].summary(),
        notes=[
            "Synthetic evaluation dataset: these are fixture numbers, not "
            "production performance.",
            "Deterministic financial truth remains authoritative throughout.",
        ],
    )

    if write:
        write_artifacts(report.run_id, report.model_dump(), output_dir=output_dir)

    return report


#: Fields that legitimately differ between two runs and are excluded from the
#: determinism comparison.
NON_DETERMINISTIC_FIELDS = ("run_id", "timestamp")


def _flatten_diff(first: Any, second: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten the differences between two report payloads into dotted paths."""
    out: Dict[str, Any] = {}
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            out.update(_flatten_diff(first.get(key), second.get(key), f"{prefix}{key}."))
    elif isinstance(first, list) and isinstance(second, list):
        if first != second:
            out[prefix.rstrip(".")] = {
                "first": f"<{len(first)} items>",
                "second": f"<{len(second)} items>",
                "note": "same length, different content" if len(first) == len(second) else "length differs",
            }
    elif first != second:
        out[prefix.rstrip(".")] = {"first": first, "second": second}
    return out


def _deterministic_payload(report: EvaluationRunReport) -> Dict[str, Any]:
    """Everything that must match between two identical evaluation runs."""
    payload = report.model_dump()
    for field in NON_DETERMINISTIC_FIELDS:
        payload.pop(field, None)
    return payload


def reproducibility_check(
    per_family: int = 12,
    seed: int = DEFAULT_EVAL_SEED,
    session: Any = None,
) -> Dict[str, Any]:
    """Run the whole evaluation twice and compare every deterministic field.

    Compared exactly: dataset version/fingerprint, split ids, sample counts,
    feature and model versions, ML metrics, confusion matrix, calibration,
    retrieval metrics, resolution metrics, policy matrix, evidence metrics,
    safety metrics and every golden-case outcome. Only ``run_id`` and
    ``timestamp`` are excluded.
    """
    first = run_evaluation(per_family=per_family, seed=seed, session=session)
    second = run_evaluation(per_family=per_family, seed=seed, session=session)

    a = _deterministic_payload(first)
    b = _deterministic_payload(second)
    differences = _flatten_diff(a, b)

    return {
        "seed": seed,
        "sections_compared": sorted(a.keys()),
        "dataset_version_first": first.dataset_version,
        "dataset_version_second": second.dataset_version,
        "dataset_fingerprint_first": first.dataset_fingerprint,
        "dataset_fingerprint_second": second.dataset_fingerprint,
        "fingerprint_match": first.dataset_fingerprint == second.dataset_fingerprint,
        "headline_match": not differences,
        "differences": differences,
        "reproducible": (first.dataset_fingerprint == second.dataset_fingerprint)
        and not differences,
    }
