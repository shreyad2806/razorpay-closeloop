"""
Evaluation package for Razorpay CloseLoop (Phase 15).

This package MEASURES the existing intelligence stack. It is deliberately
isolated from the production request path:

  * nothing in ``app/`` imports ``app.evaluation`` for decision making
  * evaluation never reconciles, approves, executes, escalates or closes
  * evaluation never mutates production state

Deterministic financial truth remains authoritative. Where this package
disagrees with the deterministic engine, the deterministic engine wins and the
disagreement is *reported*, not resolved.

Layout:
  dataset.py        versioned evaluation dataset (synthetic + real tags)
  metrics.py        pure metric primitives
  leakage.py        data-leakage audit
  ml_eval.py        classifier evaluation (reuses app.ml.classifier)
  disagreement.py   ML vs deterministic truth
  retrieval_eval.py historical retrieval Recall@K / MRR
  resolution_eval.py candidate generation / scoring / selection
  safety.py         safety metrics (unsafe automation, abstention, routing)
  policy_eval.py    policy / guardrail decision matrix
  golden.py         deterministic end-to-end golden scenarios
  artifacts.py      JSON artifact writer
  runner.py         orchestration + full report
"""

from app.evaluation.versioning import (
    EVALUATOR_VERSION,
    SYNTHETIC_DATASET_V1,
    REAL_DATASET_TAG,
    DATASET_SOURCE_SYNTHETIC,
    DATASET_SOURCE_REAL,
)

__all__ = [
    "EVALUATOR_VERSION",
    "SYNTHETIC_DATASET_V1",
    "REAL_DATASET_TAG",
    "DATASET_SOURCE_SYNTHETIC",
    "DATASET_SOURCE_REAL",
]
