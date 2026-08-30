"""
Batch reconciliation runner for Razorpay CloseLoop.

Processes batches of financial records and produces reconciliation results.
Calculates metrics from actual execution.

All logic is deterministic. No ground truth is used.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.reconciliation.engine import calculate_reconciliation, reconcile_batch
from app.schemas.enums import ExceptionType, MatchStatus
from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
    Tax,
)
from app.schemas.reconciliation import ReconciliationResult


# ─────────────────────────────────────────────────────────────────────────────
# Batch Summary
# ─────────────────────────────────────────────────────────────────────────────


class BatchSummary:
    """
    Summary of batch reconciliation execution.

    Contains metrics calculated from actual execution.
    """

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.total_cases = 0
        self.matched_cases = 0
        self.exception_cases = 0
        self.missing_cases = 0
        self.duplicate_cases = 0
        self.match_rate = 0.0
        self.exception_rate = 0.0
        self.processing_duration_seconds = 0.0
        self.throughput_cases_per_second = 0.0
        self.exception_breakdown: Dict[str, int] = {}
        self.created_at = datetime.utcnow()

    def calculate_rates(self):
        """Calculate match and exception rates."""
        if self.total_cases > 0:
            self.match_rate = self.matched_cases / self.total_cases
            self.exception_rate = self.exception_cases / self.total_cases
        else:
            self.match_rate = 0.0
            self.exception_rate = 0.0

    def calculate_throughput(self):
        """Calculate throughput in cases/second."""
        if self.processing_duration_seconds > 0:
            self.throughput_cases_per_second = (
                self.total_cases / self.processing_duration_seconds
            )
        else:
            self.throughput_cases_per_second = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        self.calculate_rates()
        self.calculate_throughput()
        return {
            "batch_id": self.batch_id,
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "exception_cases": self.exception_cases,
            "missing_cases": self.missing_cases,
            "duplicate_cases": self.duplicate_cases,
            "match_rate": round(self.match_rate, 4),
            "exception_rate": round(self.exception_rate, 4),
            "processing_duration_seconds": round(self.processing_duration_seconds, 3),
            "throughput_cases_per_second": round(self.throughput_cases_per_second, 2),
            "exception_breakdown": self.exception_breakdown,
            "created_at": self.created_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Batch Runner
# ─────────────────────────────────────────────────────────────────────────────


class BatchReconciler:
    """
    Batch reconciliation runner.

    Processes batches of financial records and produces reconciliation results.
    Calculates metrics from actual execution.
    """

    def __init__(self):
        self.results: List[ReconciliationResult] = []
        self.summary: Optional[BatchSummary] = None

    def load_batch(self, batch_dir: str) -> Dict[str, Any]:
        """
        Load a batch of financial records from disk.

        Args:
            batch_dir: Path to batch directory (e.g., "data/batch_001")

        Returns:
            Dictionary containing loaded records
        """
        batch_path = Path(batch_dir)
        generated_path = batch_path / "generated"

        # Load financial records
        with open(generated_path / "payments.json") as f:
            payments = [Payment(**p) for p in json.load(f)]
        with open(generated_path / "settlements.json") as f:
            settlements = [Settlement(**s) for s in json.load(f)]
        with open(generated_path / "refunds.json") as f:
            refunds = [Refund(**r) for r in json.load(f)]
        with open(generated_path / "fees.json") as f:
            fees = [Fee(**fe) for fe in json.load(f)]
        with open(generated_path / "taxes.json") as f:
            taxes = [Tax(**t) for t in json.load(f)]
        with open(generated_path / "adjustments.json") as f:
            adjustments = [Adjustment(**a) for a in json.load(f)]
        with open(generated_path / "cases.json") as f:
            cases = json.load(f)

        # Build case mapping (payment_id -> case_id)
        case_mapping = {c["payment_id"]: c["case_id"] for c in cases}

        return {
            "payments": payments,
            "settlements": settlements,
            "refunds": refunds,
            "fees": fees,
            "taxes": taxes,
            "adjustments": adjustments,
            "case_mapping": case_mapping,
            "cases": cases,
        }

    def reconcile_batch(
        self,
        batch_id: str,
        batch_dir: str,
    ) -> BatchSummary:
        """
        Run reconciliation on a batch of financial records.

        Args:
            batch_id: Unique batch identifier
            batch_dir: Path to batch directory

        Returns:
            BatchSummary with metrics
        """
        start_time = time.time()

        # Initialize summary
        self.summary = BatchSummary(batch_id)

        # Load batch
        batch_data = self.load_batch(batch_dir)

        # Run reconciliation
        self.results = reconcile_batch(
            payments=batch_data["payments"],
            settlements=batch_data["settlements"],
            refunds=batch_data["refunds"],
            fees=batch_data["fees"],
            taxes=batch_data["taxes"],
            adjustments=batch_data["adjustments"],
            case_mapping=batch_data["case_mapping"],
        )

        # Calculate metrics from actual results
        self._calculate_metrics()

        # Calculate timing
        self.summary.processing_duration_seconds = time.time() - start_time
        self.summary.calculate_throughput()

        return self.summary

    def _calculate_metrics(self):
        """Calculate metrics from reconciliation results."""
        if not self.summary:
            return

        self.summary.total_cases = len(self.results)

        for result in self.results:
            # Count by match status
            if result.match_status == MatchStatus.MATCHED:
                self.summary.matched_cases += 1
            elif result.match_status == MatchStatus.EXCEPTION:
                self.summary.exception_cases += 1
            elif result.match_status == MatchStatus.MISSING:
                self.summary.missing_cases += 1
            elif result.match_status == MatchStatus.DUPLICATE:
                self.summary.duplicate_cases += 1

            # Count by exception type
            exc_type = result.exception_type.value
            self.summary.exception_breakdown[exc_type] = (
                self.summary.exception_breakdown.get(exc_type, 0) + 1
            )

    def get_results(self) -> List[ReconciliationResult]:
        """Get reconciliation results."""
        return self.results

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """Get batch summary as dictionary."""
        if self.summary:
            return self.summary.to_dict()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────


def run_batch_reconciliation(
    batch_id: str,
    batch_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run batch reconciliation and optionally save results.

    Args:
        batch_id: Unique batch identifier
        batch_dir: Path to batch directory
        output_dir: Optional output directory for results

    Returns:
        Dictionary with results and summary
    """
    reconciler = BatchReconciler()
    summary = reconciler.reconcile_batch(batch_id, batch_dir)
    results = reconciler.get_results()

    # Save results if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save reconciliation results
        results_data = [r.model_dump(mode="json") for r in results]
        with open(output_path / "reconciliation_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save summary
        with open(output_path / "batch_summary.json", "w") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)

    return {
        "summary": summary.to_dict(),
        "results_count": len(results),
    }
