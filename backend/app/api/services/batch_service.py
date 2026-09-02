"""
Batch Service for Razorpay CloseLoop Phase 13.3.

Provides batch management by delegating to existing services:
- BatchGenerator for data generation
- FinancialDataAdapter for data loading
- reconcile_batch() for deterministic processing
- PersistenceService for database writes

Does NOT duplicate any reconciliation or generation logic.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.generator.batch import BatchConfig, BatchGenerator
from app.generator.orchestrator import DatasetGenerator, save_dataset
from app.reconciliation.engine import reconcile_batch
from app.schemas.config import GeneratorConfig
from app.schemas.enums import MatchStatus


# In-memory batch registry (production would use a database)
_batch_registry: Dict[str, Dict[str, Any]] = {}


class BatchService:
    """
    Service for managing batch processing.

    Delegates to existing generator, adapter, and reconciliation services.
    Does NOT duplicate business logic.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent.parent / "data")
        self._data_dir = data_dir

    def list_batches(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List registered batches with pagination."""
        batches = list(_batch_registry.values())
        paginated = batches[offset:offset + limit]
        return [
            {
                "batch_id": b["batch_id"],
                "name": b.get("name", ""),
                "status": b["status"],
                "created_at": b.get("created_at"),
                "exception_count": b.get("exception_count", 0),
                "success_count": b.get("success_count", 0),
                "failure_count": b.get("failure_count", 0),
            }
            for b in paginated
        ]

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch metadata."""
        return _batch_registry.get(batch_id)

    def create_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new batch.

        If data contains a payload with financial records, validate and store them.
        Otherwise, generate a synthetic batch using the existing BatchGenerator.
        """
        batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
        name = data.get("name", f"Batch {batch_id}")
        description = data.get("description", "")
        source = data.get("source", "synthetic")

        start_time = time.time()

        # Check if payload contains pre-defined records
        payload = data.get("payload")
        if payload and isinstance(payload, dict):
            # Validate and store provided records
            validation = self._validate_payload(payload)
            if not validation["valid"]:
                return {
                    "batch_id": batch_id,
                    "status": "VALIDATION_FAILED",
                    "errors": validation["errors"],
                }

            # Store the payload
            batch_dir = os.path.join(self._data_dir, batch_id, "generated")
            os.makedirs(batch_dir, exist_ok=True)
            self._save_payload(payload, batch_dir)

            record_count = sum(
                len(payload.get(k, []))
                for k in ["payments", "settlements", "refunds", "fees", "taxes", "adjustments", "cases"]
            )
        else:
            # Generate synthetic data using existing BatchGenerator
            num_merchants = data.get("num_merchants", 5)
            num_cases = data.get("num_cases", 20)

            generator = BatchGenerator(base_seed=42, output_dir=self._data_dir)
            result = generator.generate_batch(
                batch_id=batch_id,
                num_merchants=num_merchants,
                num_cases=num_cases,
                seed_offset=hash(batch_id) % 1000,
            )
            record_count = result.get("total_records", 0)

        creation_time = time.time() - start_time

        # Register the batch
        batch_meta = {
            "batch_id": batch_id,
            "name": name,
            "description": description,
            "source": source,
            "status": "CREATED",
            "created_at": datetime.utcnow().isoformat(),
            "exception_count": record_count,
            "success_count": 0,
            "failure_count": 0,
            "creation_time_ms": round(creation_time * 1000, 2),
        }
        _batch_registry[batch_id] = batch_meta

        return batch_meta

    def run_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Run deterministic reconciliation on a batch.

        Uses the existing reconcile_batch() function.
        Does NOT duplicate reconciliation logic.
        """
        batch = _batch_registry.get(batch_id)
        if batch is None:
            return None

        if batch["status"] == "COMPLETED":
            return {
                "batch_id": batch_id,
                "status": "ALREADY_COMPLETED",
                "message": "Batch was already processed",
            }

        if batch["status"] == "RUNNING":
            return {
                "batch_id": batch_id,
                "status": "ALREADY_RUNNING",
                "message": "Batch is currently being processed",
            }

        batch["status"] = "RUNNING"
        start_time = time.time()

        try:
            # Load data directly from batch directory
            generated_dir = os.path.join(self._data_dir, batch_id, "generated")
            if not os.path.isdir(generated_dir):
                batch["status"] = "FAILED"
                batch["failure_count"] = 1
                return {
                    "batch_id": batch_id,
                    "status": "FAILED",
                    "error": f"Batch data directory not found: {generated_dir}",
                }

            payments = self._dicts_to_payments(self._load_json(os.path.join(generated_dir, "payments.json")))
            settlements = self._dicts_to_settlements(self._load_json(os.path.join(generated_dir, "settlements.json")))
            refunds = self._dicts_to_refunds(self._load_json(os.path.join(generated_dir, "refunds.json")))
            fees = self._dicts_to_fees(self._load_json(os.path.join(generated_dir, "fees.json")))
            taxes = self._dicts_to_taxes(self._load_json(os.path.join(generated_dir, "taxes.json")))
            adjustments = self._dicts_to_adjustments(self._load_json(os.path.join(generated_dir, "adjustments.json")))

            if not payments:
                batch["status"] = "FAILED"
                batch["failure_count"] = 1
                return {
                    "batch_id": batch_id,
                    "status": "FAILED",
                    "error": "No payment records found in batch data",
                }

            # Build case mapping from loaded cases
            cases = self._load_json(os.path.join(generated_dir, "cases.json"))
            case_mapping = {c.get("payment_id", ""): c.get("case_id", "") for c in cases}

            # Run existing deterministic reconciliation
            results = reconcile_batch(
                payments=payments,
                settlements=settlements,
                refunds=refunds,
                fees=fees,
                taxes=taxes,
                adjustments=adjustments,
                case_mapping=case_mapping,
            )

            # Compute summary statistics
            total = len(results)
            matched = sum(
                1 for r in results
                if r.match_status == MatchStatus.MATCHED
            )
            exceptions = total - matched
            match_rate = matched / total if total > 0 else 0.0
            exception_rate = exceptions / total if total > 0 else 0.0

            processing_time_ms = (time.time() - start_time) * 1000

            # Save reconciliation results to batch directory
            self._save_results(batch_id, results)

            batch["status"] = "COMPLETED"
            batch["success_count"] = total
            batch["exception_count"] = exceptions
            batch["processing_time_ms"] = round(processing_time_ms, 2)
            batch["match_rate"] = round(match_rate, 4)
            batch["exception_rate"] = round(exception_rate, 4)
            batch["total_records"] = total
            batch["matched_records"] = matched

            return {
                "batch_id": batch_id,
                "status": "COMPLETED",
                "total_records": total,
                "matched_records": matched,
                "exceptions": exceptions,
                "match_rate": round(match_rate, 4),
                "exception_rate": round(exception_rate, 4),
                "processing_time_ms": round(processing_time_ms, 2),
                "throughput_records_per_sec": round(
                    total / (processing_time_ms / 1000) if processing_time_ms > 0 else 0, 2
                ),
            }

        except Exception as e:
            batch["status"] = "FAILED"
            batch["failure_count"] = batch.get("failure_count", 0) + 1
            return {
                "batch_id": batch_id,
                "status": "FAILED",
                "error": str(e),
            }

    def get_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing summary."""
        batch = _batch_registry.get(batch_id)
        if batch is None:
            return None

        # If batch has been run, include reconciliation stats
        return {
            "batch_id": batch_id,
            "name": batch.get("name", ""),
            "status": batch["status"],
            "total_exceptions": batch.get("exception_count", 0),
            "resolved": batch.get("success_count", 0),
            "unresolved": batch.get("failure_count", 0),
            "escalated": 0,
            "auto_resolved": 0,
            "human_review": 0,
            "verification_passed": 0,
            "verification_failed": 0,
            "financial_impact_paise": 0,
            "match_rate": batch.get("match_rate", 0.0),
            "exception_rate": batch.get("exception_rate", 0.0),
            "processing_time_ms": batch.get("processing_time_ms", 0),
            "total_records": batch.get("total_records", 0),
            "matched_records": batch.get("matched_records", 0),
            "throughput_records_per_sec": batch.get("throughput_records_per_sec", 0),
            "created_at": batch.get("created_at"),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an uploaded payload structure."""
        errors = []

        required_keys = ["payments", "cases"]
        for key in required_keys:
            if key not in payload:
                errors.append(f"Missing required key: {key}")
            elif not isinstance(payload[key], list):
                errors.append(f"'{key}' must be a list")
            elif len(payload[key]) == 0:
                errors.append(f"'{key}' is empty")

        # Validate payment structure
        for i, payment in enumerate(payload.get("payments", [])[:5]):  # sample first 5
            if "payment_id" not in payment:
                errors.append(f"Payment at index {i} missing 'payment_id'")
            if "amount" not in payment:
                errors.append(f"Payment at index {i} missing 'amount'")
            elif not isinstance(payment["amount"], (int, float)):
                errors.append(f"Payment at index {i} 'amount' must be numeric")
            elif payment["amount"] < 0:
                errors.append(f"Payment at index {i} 'amount' must be non-negative")

        # Validate case structure
        for i, case in enumerate(payload.get("cases", [])[:5]):
            if "case_id" not in case:
                errors.append(f"Case at index {i} missing 'case_id'")
            if "payment_id" not in case:
                errors.append(f"Case at index {i} missing 'payment_id'")

        # Check for duplicate IDs
        payment_ids = [p.get("payment_id") for p in payload.get("payments", []) if p.get("payment_id")]
        if len(payment_ids) != len(set(payment_ids)):
            errors.append("Duplicate payment_id found")

        case_ids = [c.get("case_id") for c in payload.get("cases", []) if c.get("case_id")]
        if len(case_ids) != len(set(case_ids)):
            errors.append("Duplicate case_id found")

        return {"valid": len(errors) == 0, "errors": errors}

    # ─────────────────────────────────────────────────────────────────────
    # Data Conversion Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _dicts_to_payments(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Payment
        return [Payment(**d) for d in dicts]

    def _dicts_to_settlements(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Settlement
        return [Settlement(**d) for d in dicts]

    def _dicts_to_refunds(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Refund
        return [Refund(**d) for d in dicts]

    def _dicts_to_fees(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Fee
        return [Fee(**d) for d in dicts]

    def _dicts_to_taxes(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Tax
        return [Tax(**d) for d in dicts]

    def _dicts_to_adjustments(self, dicts: List[Dict[str, Any]]) -> List:
        from app.schemas.financial import Adjustment
        return [Adjustment(**d) for d in dicts]

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load a JSON file safely."""
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    def _save_payload(self, payload: Dict[str, Any], batch_dir: str):
        """Save uploaded payload to batch directory."""
        for key in ["payments", "settlements", "refunds", "fees", "taxes", "adjustments", "cases", "merchants"]:
            if key in payload:
                path = os.path.join(batch_dir, f"{key}.json")
                with open(path, "w") as f:
                    json.dump(payload[key], f, indent=2)

    def _save_results(self, batch_id: str, results: list):
        """Save reconciliation results to batch directory."""
        batch_dir = os.path.join(self._data_dir, batch_id, "generated")
        os.makedirs(batch_dir, exist_ok=True)
        path = os.path.join(batch_dir, "reconciliation_results.json")
        with open(path, "w") as f:
            json.dump(
                [r.model_dump() if hasattr(r, "model_dump") else r.__dict__ for r in results],
                f,
                indent=2,
                default=str,
            )
