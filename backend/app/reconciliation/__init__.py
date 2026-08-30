"""
Razorpay CloseLoop deterministic reconciliation engine.

This module provides the reconciliation contract and interfaces.
The engine is deterministic — no AI, no ML, no LLM.
"""

from app.reconciliation.batch import (
    BatchReconciler,
    BatchSummary,
    run_batch_reconciliation,
)
from app.reconciliation.engine import (
    calculate_reconciliation,
    reconcile_batch,
)
from app.reconciliation.matching import (
    MatchingEvidence,
    match_and_classify,
)
from app.schemas.reconciliation import (
    CalculationBreakdown,
    MatchingRule,
    ReconciliationResult,
)

__all__ = [
    "ReconciliationResult",
    "CalculationBreakdown",
    "MatchingRule",
    "MatchingEvidence",
    "BatchReconciler",
    "BatchSummary",
    "calculate_reconciliation",
    "reconcile_batch",
    "match_and_classify",
    "run_batch_reconciliation",
]
