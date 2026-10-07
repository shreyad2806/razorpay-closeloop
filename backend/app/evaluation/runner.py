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
    run_stage_scenarios,
)
from app.evaluation.leakage import LeakageReport, audit_dataset
from app.evaluation.ml_eval import MLClassificationReport, evaluate_classifier
from app.evaluation.policy_eval import PolicyReport, evaluate_policy
from app.evaluation.resolution_eval import ResolutionReport, evaluate_resolution
from app.evaluation.retrieval_eval import RetrievalReport, evaluate_retrieval
from app.evaluation.safety import SafetyReport, evaluate_safety
from app.evaluation.versioning import (
    DEFAULT_EVAL_SEED,
    EVALUATOR_VERSION,
    SYNTHETIC_DATASET_SCHEMA_VERSION,
)


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
        """The numbers that matter most for a safety review."""
        safety = self.safety or {}
        ml = self.ml or {}
        return {
            "dataset_version": self.dataset_version,
            "dataset_source": self.dataset_source,
            "dataset_cases": self.dataset_cases,
            "leakage_free": (self.leakage or {}).get("leakage_free"),
            "accuracy": ml.get("accuracy"),
            "macro_f1": ml.get("f1_macro"),
            "unsafe_automation_count": safety.get("unsafe_automation_count"),
            "unsafe_resolution_rate": safety.get("unsafe_resolution_rate"),
            "safe_resolution_precision": safety.get("safe_resolution_precision"),
            "human_review_recall": safety.get("human_review_recall"),
            "abstention_rate": safety.get("abstention_rate"),
            "golden_passed": (self.golden or {}).get("passed"),
            "golden_total": (self.golden or {}).get("total"),
        }


def _core_evaluation(dataset: EvaluationDataset) -> Dict[str, Any]:
    """Run every evaluator. Pure with respect to production state."""
    ml: MLClassificationReport = evaluate_classifier(dataset)
    disagreement: DisagreementReport = measure_disagreement(dataset)
    retrieval: RetrievalReport = evaluate_retrieval(dataset)
    resolution: ResolutionReport = evaluate_resolution(dataset)
    safety: SafetyReport = evaluate_safety(dataset, resolution)
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

    ``session`` (optional) enables the database-backed golden scenarios.
    """
    dataset = build_synthetic_dataset_v1(per_family=per_family, seed=seed)
    core = _core_evaluation(dataset)

    golden_results: List[GoldenResult] = run_stage_scenarios()
    if session is not None:
        for scenario in DB_SCENARIOS:
            try:
                golden_results.append(scenario(session))
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

    golden = {
        "evaluator_version": EVALUATOR_VERSION,
        "total": len(golden_results),
        "passed": sum(1 for r in golden_results if r.passed),
        "failed": sum(1 for r in golden_results if not r.passed),
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


def reproducibility_check(
    per_family: int = 12,
    seed: int = DEFAULT_EVAL_SEED,
) -> Dict[str, Any]:
    """Run the evaluation twice and compare fingerprints and headline metrics."""
    first = run_evaluation(per_family=per_family, seed=seed)
    second = run_evaluation(per_family=per_family, seed=seed)

    a = first.headline()
    b = second.headline()
    keys = sorted(set(a) | set(b))
    differences = {k: {"first": a.get(k), "second": b.get(k)} for k in keys if a.get(k) != b.get(k)}

    return {
        "seed": seed,
        "dataset_fingerprint_first": first.dataset_fingerprint,
        "dataset_fingerprint_second": second.dataset_fingerprint,
        "fingerprint_match": first.dataset_fingerprint == second.dataset_fingerprint,
        "headline_match": not differences,
        "differences": differences,
        "reproducible": (first.dataset_fingerprint == second.dataset_fingerprint) and not differences,
    }
