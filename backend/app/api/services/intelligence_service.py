"""
Intelligence Service for Razorpay CloseLoop Phase 13.5.

Provides analysis, explanation, similarity, and evidence by delegating
to existing services:
- ExplainService (Phase 12G) for LLM explanations
- AnalyzeService (Phase 12H) for full analysis
- FinancialDataAdapter (Phase 11B) for data loading
- LLM services (Phase 12) for explanation generation

Does NOT duplicate:
- Reconciliation logic
- Evidence retrieval logic
- Classification logic
- Guardrail logic
"""

from typing import Any, Dict, List, Optional

from app.api.explain import ExplainService, ExplainRequest
from app.api.analyze import AnalyzeService, AnalyzeRequest
from mcp.adapters.financial_data import FinancialDataAdapter


class IntelligenceService:
    """
    Service for intelligence operations.

    Delegates to existing Phase 12 explain/analyze services
    and Phase 11 data adapter for evidence/similarity.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir
        self._explain_service = ExplainService()
        self._analyze_service = AnalyzeService()
        self._adapter: Optional[FinancialDataAdapter] = None

    def _get_adapter(self) -> FinancialDataAdapter:
        """Lazy-load the financial data adapter."""
        if self._adapter is None:
            self._adapter = FinancialDataAdapter(data_dir=self._data_dir)
            self._adapter.load_batch()
        return self._adapter

    # ─────────────────────────────────────────────────────────────────────
    # ANALYZE
    # ─────────────────────────────────────────────────────────────────────

    async def analyze(self, exception_id: str) -> Dict[str, Any]:
        """
        Full analysis of an exception.

        Delegates to existing AnalyzeService from Phase 12H.
        """
        try:
            req = AnalyzeRequest(exception_id=exception_id)
            result = await self._analyze_service.analyze(req)
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        except Exception as e:
            # Fallback: build analysis from data adapter
            return self._analyze_fallback(exception_id, str(e))

    def _analyze_fallback(self, exception_id: str, error: str) -> Dict[str, Any]:
        """Fallback analysis when AnalyzeService fails."""
        adapter = self._get_adapter()
        case = adapter.get_case(exception_id)

        if case is None:
            return {
                "exception_id": exception_id,
                "error": f"Exception '{exception_id}' not found",
                "success": False,
            }

        # Build basic analysis from case data
        payment = adapter.get_payment(case.get("payment_id", ""))
        settlements = adapter.get_settlements_for_payment(case.get("payment_id", ""))

        return {
            "exception_id": exception_id,
            "case_id": case.get("case_id"),
            "financial_discrepancy": {
                "expected_amount_paise": case.get("expected_amount", 0),
                "actual_amount_paise": case.get("actual_amount", 0),
                "difference_paise": case.get("difference", 0),
                "exception_type": case.get("scenario", "UNKNOWN"),
            },
            "evidence": {
                "record_count": 1 + len(settlements),
                "coverage": "PARTIALLY_EXPLAINED" if case.get("difference", 0) != 0 else "FULLY_EXPLAINED",
                "explained_amount_paise": case.get("actual_amount", 0),
                "remaining_amount_paise": abs(case.get("difference", 0)),
                "conflicts": [],
                "missing_evidence": [],
            },
            "classification_type": case.get("scenario"),
            "classification_confidence": case.get("confidence"),
            "similar_case_count": 0,
            "candidates": [],
            "guardrail": {
                "decision": "HUMAN_REVIEW",
                "confidence": 0.5,
                "risk_category": case.get("risk_category", "LOW"),
                "reasons": ["Fallback analysis — main service unavailable"],
                "exposure_paise": abs(case.get("difference", 0)),
            },
            "ai_explanation": f"Exception {exception_id}: {case.get('scenario', 'UNKNOWN')} with difference of {case.get('difference', 0)} paise.",
            "ai_uncertainty": "Fallback analysis — main service unavailable",
            "llm_provider": "none",
            "fallback_used": True,
            "error": error,
        }

    # ─────────────────────────────────────────────────────────────────────
    # EXPLAIN
    # ─────────────────────────────────────────────────────────────────────

    async def explain(self, exception_id: str, depth: str = "standard") -> Dict[str, Any]:
        """
        Explain an exception.

        Delegates to existing ExplainService from Phase 12G.
        Returns deterministic fallback if LLM unavailable.
        """
        try:
            req = ExplainRequest(exception_id=exception_id, explanation_depth=depth)
            result = await self._explain_service.explain(req)
            if hasattr(result, "model_dump"):
                d = result.model_dump()
                # Ensure exception_id is set at all levels
                if not d.get("exception_id"):
                    d["exception_id"] = exception_id
                data = d.get("data")
                if isinstance(data, dict) and not data.get("exception_id"):
                    data["exception_id"] = exception_id
                return d
            return result
        except Exception as e:
            return self._explain_fallback(exception_id, str(e))

    def _explain_fallback(self, exception_id: str, error: str) -> Dict[str, Any]:
        """Deterministic fallback explanation."""
        adapter = self._get_adapter()
        case = adapter.get_case(exception_id)

        if case is None:
            return {
                "exception_id": exception_id,
                "summary": f"Exception '{exception_id}' not found.",
                "reason": "No case data available.",
                "evidence_summary": "No evidence available.",
                "uncertainty": "Cannot explain — exception not found.",
                "limitations": "Exception not found in dataset.",
                "fallback_used": True,
            }

        exc_type = case.get("scenario", "UNKNOWN")
        expected = case.get("expected_amount", 0)
        actual = case.get("actual_amount", 0)
        diff = case.get("difference", 0)

        return {
            "exception_id": exception_id,
            "case_id": case.get("case_id"),
            "summary": f"Financial exception of type {exc_type}: expected {expected} paise, actual {actual} paise, difference {diff} paise.",
            "reason": f"The {exc_type.lower().replace('_', ' ')} caused a discrepancy of {diff} paise between expected and actual settlement amounts.",
            "evidence_summary": f"Payment {case.get('payment_id', 'N/A')} shows a {exc_type.lower().replace('_', ' ')} scenario.",
            "uncertainty": "Explanation generated from deterministic fallback — no LLM involved.",
            "limitations": "This is a template-based explanation. LLM explanation unavailable.",
            "expected_amount_paise": expected,
            "actual_amount_paise": actual,
            "difference_paise": diff,
            "exception_type": exc_type,
            "fallback_used": True,
        }

    # ─────────────────────────────────────────────────────────────────────
    # SIMILAR
    # ─────────────────────────────────────────────────────────────────────

    def get_similar(self, exception_id: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find similar historical cases.

        Uses the FinancialDataAdapter to find cases with the same
        exception type and similar financial patterns.
        """
        adapter = self._get_adapter()
        case = adapter.get_case(exception_id)

        if case is None:
            return {
                "exception_id": exception_id,
                "similar_cases": [],
                "count": 0,
                "confidence": "LOW",
                "error": f"Exception '{exception_id}' not found",
            }

        # Find cases with same exception type
        target_type = case.get("scenario", "")
        target_diff = abs(case.get("difference", 0))

        all_cases = adapter._cases
        similar = []

        for c in all_cases:
            if c.get("case_id") == exception_id:
                continue

            # Simple similarity: same exception type + similar difference magnitude
            if c.get("scenario") == target_type:
                c_diff = abs(c.get("difference", 0))
                # Compute simple similarity score
                if target_diff > 0:
                    diff_ratio = min(c_diff, target_diff) / max(c_diff, target_diff)
                else:
                    diff_ratio = 1.0 if c_diff == 0 else 0.5

                score = 0.7 + 0.3 * diff_ratio  # Base 0.7 for same type

                similar.append({
                    "case_id": c.get("case_id", ""),
                    "similarity_score": round(score, 4),
                    "exception_type": c.get("scenario", ""),
                    "resolution_type": None,  # Would need resolution data
                    "adjustment_paise": c.get("difference", 0),
                    "risk_category": c.get("risk_category", "LOW"),
                })

        # Sort by similarity score, take top_k
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_similar = similar[:top_k]

        # Determine confidence based on top score
        if top_similar:
            top_score = top_similar[0]["similarity_score"]
            if top_score >= 0.9:
                confidence = "HIGH"
            elif top_score >= 0.7:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "LOW"

        return {
            "exception_id": exception_id,
            "similar_cases": top_similar,
            "count": len(top_similar),
            "confidence": confidence,
            "total_candidates": len(similar),
        }

    # ─────────────────────────────────────────────────────────────────────
    # EVIDENCE
    # ─────────────────────────────────────────────────────────────────────

    def get_evidence(self, exception_id: str) -> Dict[str, Any]:
        """
        Get structured evidence for an exception.

        Loads financial records from the data adapter and builds
        an evidence summary.
        """
        adapter = self._get_adapter()
        case = adapter.get_case(exception_id)

        if case is None:
            return {
                "exception_id": exception_id,
                "evidence": [],
                "total_amount_paise": 0,
                "coverage": "UNKNOWN",
                "conflicts": [],
                "missing_evidence": [],
                "error": f"Exception '{exception_id}' not found",
            }

        payment_id = case.get("payment_id", "")
        payment = adapter.get_payment(payment_id)
        settlements = adapter.get_settlements_for_payment(payment_id)
        refunds = adapter.get_refunds_for_payment(payment_id)
        fees = adapter.get_fees_for_payment(payment_id)
        adjustments = adapter.get_adjustments_for_payment(payment_id)

        evidence = []
        total_amount = 0

        # Payment evidence
        if payment:
            evidence.append({
                "record_type": "PAYMENT",
                "record_id": payment.get("payment_id", ""),
                "amount_paise": payment.get("amount", 0),
                "status": payment.get("status", ""),
            })
            total_amount += payment.get("amount", 0)

        # Settlement evidence
        for s in settlements:
            evidence.append({
                "record_type": "SETTLEMENT",
                "record_id": s.get("settlement_id", ""),
                "amount_paise": s.get("amount", 0),
                "status": s.get("status", ""),
            })
            total_amount += s.get("amount", 0)

        # Refund evidence
        for r in refunds:
            evidence.append({
                "record_type": "REFUND",
                "record_id": r.get("refund_id", ""),
                "amount_paise": r.get("amount", 0),
                "status": r.get("status", ""),
            })

        # Fee evidence
        for f in fees:
            evidence.append({
                "record_type": "FEE",
                "record_id": f.get("fee_id", ""),
                "amount_paise": f.get("amount", 0),
                "status": f.get("fee_type", ""),
            })

        # Adjustment evidence
        for a in adjustments:
            evidence.append({
                "record_type": "ADJUSTMENT",
                "record_id": a.get("adjustment_id", ""),
                "amount_paise": a.get("amount", 0),
                "status": a.get("adjustment_type", ""),
            })

        # Determine coverage
        expected = case.get("expected_amount", 0)
        actual = case.get("actual_amount", 0)
        diff = case.get("difference", 0)

        if diff == 0:
            coverage = "FULLY_EXPLAINED"
        elif abs(diff) < abs(expected) * 0.1:
            coverage = "PARTIALLY_EXPLAINED"
        else:
            coverage = "PARTIALLY_EXPLAINED"

        # Detect conflicts
        conflicts = []
        if settlements:
            settlement_amounts = [s.get("amount", 0) for s in settlements]
            if len(settlement_amounts) > 1 and len(set(settlement_amounts)) > 1:
                conflicts.append("Multiple settlements with different amounts")

        # Detect missing evidence
        missing = []
        if not refunds:
            missing.append("No refund records found")
        if not fees:
            missing.append("No fee records found")

        return {
            "exception_id": exception_id,
            "evidence": evidence,
            "total_amount_paise": total_amount,
            "coverage": coverage,
            "conflicts": conflicts,
            "missing_evidence": missing,
            "payment_id": payment_id,
            "record_count": len(evidence),
        }
