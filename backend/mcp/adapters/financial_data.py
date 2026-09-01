"""
MCP Financial Data Adapter for Razorpay CloseLoop Phase 11B.

Provides controlled READ-ONLY access to financial data loaded from
the synthetic dataset JSON files.

Safety principle:
  This adapter is READ-ONLY.
  It loads data from JSON files and provides filtered access.
  It does NOT:
  - Execute SQL queries
  - Modify financial records
  - Bypass evidence services
  - Authorize financial actions
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.financial import (
    Adjustment,
    Fee,
    Payment,
    Refund,
    Settlement,
)
from app.schemas.case import Case


class FinancialDataAdapter:
    """Read-only adapter for financial data from synthetic dataset.

    Loads data from JSON files and provides controlled search/filter access.
    Does NOT execute arbitrary queries or modify data.
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir is None:
            # Default to the project's data directory
            data_dir = str(Path(__file__).parent.parent.parent.parent / "data")
        self._data_dir = data_dir
        self._loaded = False
        self._payments: List[Dict[str, Any]] = []
        self._settlements: List[Dict[str, Any]] = []
        self._refunds: List[Dict[str, Any]] = []
        self._fees: List[Dict[str, Any]] = []
        self._adjustments: List[Dict[str, Any]] = []
        self._cases: List[Dict[str, Any]] = []
        self._merchants: List[Dict[str, Any]] = []

    def load_batch(self, batch_id: str = "batch_001") -> bool:
        """Load data from a specific batch directory.

        Returns True if data was loaded successfully.
        """
        generated_dir = os.path.join(self._data_dir, batch_id, "generated")
        if not os.path.isdir(generated_dir):
            return False

        try:
            self._payments = self._load_json(os.path.join(generated_dir, "payments.json"))
            self._settlements = self._load_json(os.path.join(generated_dir, "settlements.json"))
            self._refunds = self._load_json(os.path.join(generated_dir, "refunds.json"))
            self._fees = self._load_json(os.path.join(generated_dir, "fees.json"))
            self._adjustments = self._load_json(os.path.join(generated_dir, "adjustments.json"))
            self._cases = self._load_json(os.path.join(generated_dir, "cases.json"))
            self._merchants = self._load_json(os.path.join(generated_dir, "merchants.json"))
            self._loaded = True
            return True
        except Exception:
            self._loaded = False
            return False

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load a JSON file safely."""
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────────────────────────────────────────────────────────────
    # READ-ONLY access methods
    # ─────────────────────────────────────────────────────────────────────

    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """Get a payment by ID. Returns None if not found."""
        for p in self._payments:
            if p.get("payment_id") == payment_id:
                return dict(p)
        return None

    def get_settlement(self, settlement_id: str) -> Optional[Dict[str, Any]]:
        """Get a settlement by ID. Returns None if not found."""
        for s in self._settlements:
            if s.get("settlement_id") == settlement_id:
                return dict(s)
        return None

    def get_settlements_for_payment(self, payment_id: str) -> List[Dict[str, Any]]:
        """Get all settlements for a payment."""
        return [dict(s) for s in self._settlements if s.get("payment_id") == payment_id]

    def get_refund(self, refund_id: str) -> Optional[Dict[str, Any]]:
        """Get a refund by ID. Returns None if not found."""
        for r in self._refunds:
            if r.get("refund_id") == refund_id:
                return dict(r)
        return None

    def get_refunds_for_payment(self, payment_id: str) -> List[Dict[str, Any]]:
        """Get all refunds for a payment."""
        return [dict(r) for r in self._refunds if r.get("payment_id") == payment_id]

    def get_fee(self, fee_id: str) -> Optional[Dict[str, Any]]:
        """Get a fee by ID. Returns None if not found."""
        for f in self._fees:
            if f.get("fee_id") == fee_id:
                return dict(f)
        return None

    def get_fees_for_payment(self, payment_id: str) -> List[Dict[str, Any]]:
        """Get all fees for a payment."""
        return [dict(f) for f in self._fees if f.get("payment_id") == payment_id]

    def get_adjustment(self, adjustment_id: str) -> Optional[Dict[str, Any]]:
        """Get an adjustment by ID. Returns None if not found."""
        for a in self._adjustments:
            if a.get("adjustment_id") == adjustment_id:
                return dict(a)
        return None

    def get_adjustments_for_payment(self, payment_id: str) -> List[Dict[str, Any]]:
        """Get all adjustments for a payment."""
        return [dict(a) for a in self._adjustments if a.get("payment_id") == payment_id]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get a case by ID. Returns None if not found."""
        for c in self._cases:
            if c.get("case_id") == case_id:
                return dict(c)
        return None

    def search_records(
        self,
        merchant_id: Optional[str] = None,
        payment_id: Optional[str] = None,
        settlement_id: Optional[str] = None,
        case_id: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Controlled search across financial records.

        Only supports validated filter parameters.
        Does NOT support arbitrary SQL or query execution.
        """
        results: List[Dict[str, Any]] = []

        record_types = (
            [record_type] if record_type
            else ["payment", "settlement", "refund", "fee", "adjustment", "case"]
        )

        if "payment" in record_types:
            for p in self._payments:
                if self._matches(p, merchant_id=merchant_id, payment_id=payment_id, case_id=case_id):
                    results.append({"type": "payment", "data": dict(p)})
                if len(results) >= limit:
                    break

        if "settlement" in record_types and len(results) < limit:
            for s in self._settlements:
                if self._matches(s, merchant_id=merchant_id, payment_id=payment_id,
                                 settlement_id=settlement_id, case_id=case_id):
                    results.append({"type": "settlement", "data": dict(s)})
                if len(results) >= limit:
                    break

        if "refund" in record_types and len(results) < limit:
            for r in self._refunds:
                if self._matches(r, merchant_id=merchant_id, payment_id=payment_id, case_id=case_id):
                    results.append({"type": "refund", "data": dict(r)})
                if len(results) >= limit:
                    break

        if "fee" in record_types and len(results) < limit:
            for f in self._fees:
                if self._matches(f, merchant_id=merchant_id, payment_id=payment_id, case_id=case_id):
                    results.append({"type": "fee", "data": dict(f)})
                if len(results) >= limit:
                    break

        if "adjustment" in record_types and len(results) < limit:
            for a in self._adjustments:
                if self._matches(a, merchant_id=merchant_id, payment_id=payment_id, case_id=case_id):
                    results.append({"type": "adjustment", "data": dict(a)})
                if len(results) >= limit:
                    break

        if "case" in record_types and len(results) < limit:
            for c in self._cases:
                if self._matches(c, merchant_id=merchant_id, payment_id=payment_id, case_id=case_id):
                    results.append({"type": "case", "data": dict(c)})
                if len(results) >= limit:
                    break

        return results[:limit]

    def _matches(
        self,
        record: Dict[str, Any],
        merchant_id: Optional[str] = None,
        payment_id: Optional[str] = None,
        settlement_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> bool:
        """Check if a record matches the given filters.

        A record matches only if it has ALL specified filter fields
        and they match the filter values. Records missing a filter
        field do NOT match that filter.
        """
        if merchant_id:
            if "merchant_id" not in record or record["merchant_id"] != merchant_id:
                return False
        if payment_id:
            if "payment_id" not in record or record["payment_id"] != payment_id:
                return False
        if settlement_id:
            if "settlement_id" not in record or record["settlement_id"] != settlement_id:
                return False
        if case_id:
            if "case_id" not in record or record["case_id"] != case_id:
                return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Summary stats (read-only)
    # ─────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, int]:
        """Get record counts (read-only)."""
        return {
            "payments": len(self._payments),
            "settlements": len(self._settlements),
            "refunds": len(self._refunds),
            "fees": len(self._fees),
            "adjustments": len(self._adjustments),
            "cases": len(self._cases),
            "merchants": len(self._merchants),
        }
