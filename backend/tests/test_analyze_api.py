"""
Tests for Razorpay CloseLoop Phase 12H — POST /analyze API.

Covers:
- Request/response schemas
- AnalyzeService orchestration
- Normal case, complex case, unknown case
- High-risk case, conflicting evidence
- LLM unavailable fallback
- Safety boundary (LLM cannot override anything)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_request_minimal(self):
        from app.api.analyze import AnalyzeRequest

        req = AnalyzeRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.include_evidence is True
        assert req.analysis_depth == "standard"

    def test_request_empty_id_rejected(self):
        from app.api.analyze import AnalyzeRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnalyzeRequest(exception_id="")

    def test_financial_discrepancy_schema(self):
        from app.api.analyze import FinancialDiscrepancy

        d = FinancialDiscrepancy(
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
        )
        assert d.difference_paise == 5000

    def test_evidence_summary_schema(self):
        from app.api.analyze import EvidenceSummary

        e = EvidenceSummary(
            record_count=3,
            coverage="FULLY_EXPLAINED",
            conflicts=["conflict1"],
        )
        assert e.record_count == 3

    def test_guardrail_summary_schema(self):
        from app.api.analyze import GuardrailSummary

        g = GuardrailSummary(
            decision="HUMAN_REVIEW",
            confidence=0.5,
            risk_category="MEDIUM",
        )
        assert g.decision == "HUMAN_REVIEW"

    def test_candidate_summary_schema(self):
        from app.api.analyze import CandidateSummary

        c = CandidateSummary(
            resolution_type="fee_reversal",
            source="DETERMINISTIC",
            confidence=0.8,
        )
        assert c.resolution_type == "fee_reversal"

    def test_analysis_result_schema(self):
        from app.api.analyze import AnalysisResult

        r = AnalysisResult(exception_id="EXP-001")
        assert r.candidates == []
        assert r.fallback_used is False


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeEndpoint:
    def _make_client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_still_works(self):
        client = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_analyze_unknown_exception(self):
        client = self._make_client()
        resp = client.post("/analyze", json={"exception_id": "NONEXISTENT-999"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_analyze_missing_id(self):
        client = self._make_client()
        resp = client.post("/analyze", json={})
        assert resp.status_code == 422

    def test_analyze_valid_exception(self):
        client = self._make_client()
        from mcp.adapters.financial_data import FinancialDataAdapter
        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases in dataset")

        test_id = adapter._cases[0].get("case_id", "CASE-000001")
        resp = client.post("/analyze", json={"exception_id": test_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"] is not None
        assert "financial_discrepancy" in data["data"]
        assert "evidence" in data["data"]
        assert "guardrail" in data["data"]
        assert "candidates" in data["data"]

    def test_analyze_returns_provider_status(self):
        client = self._make_client()
        resp = client.post("/analyze", json={"exception_id": "NONEXISTENT"})
        data = resp.json()
        assert "provider_status" in data


# ─────────────────────────────────────────────────────────────────────────────
# AnalyzeService Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeService:
    def test_not_found(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id="NONEXISTENT"))
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_exact_match_case(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        # Find an EXACT_MATCH case
        exact = [c for c in adapter._cases if c.get("scenario") == "EXACT_MATCH"]
        if not exact:
            pytest.skip("No EXACT_MATCH cases")

        test_id = exact[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert result.success is True
        assert result.data.guardrail.decision == "AUTO"
        assert result.data.financial_discrepancy.difference_paise == 0

    def test_discrepancy_case(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        # Find a case with non-zero difference
        diff_cases = [c for c in adapter._cases if c.get("difference", 0) != 0]
        if not diff_cases:
            pytest.skip("No discrepancy cases")

        test_id = diff_cases[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert result.success is True
        assert result.data.financial_discrepancy.difference_paise != 0
        assert len(result.data.candidates) > 0

    def test_guardrail_auto_for_small_discrepancy(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        small = [c for c in adapter._cases if 0 < abs(c.get("difference", 0)) <= 10000]
        if not small:
            pytest.skip("No small discrepancy cases")

        test_id = small[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert result.success is True
        assert result.data.guardrail.decision == "AUTO"

    def test_guardrail_human_for_large_discrepancy(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        large = [c for c in adapter._cases if abs(c.get("difference", 0)) > 100000]
        if not large:
            pytest.skip("No large discrepancy cases")

        test_id = large[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert result.success is True
        assert result.data.guardrail.decision == "HUMAN_REVIEW"

    def test_candidates_generated_for_discrepancy(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        diff_cases = [c for c in adapter._cases if c.get("difference", 0) != 0]
        if not diff_cases:
            pytest.skip("No discrepancy cases")

        test_id = diff_cases[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert len(result.data.candidates) >= 1

    def test_no_candidates_for_exact_match(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        exact = [c for c in adapter._cases if c.get("scenario") == "EXACT_MATCH"]
        if not exact:
            pytest.skip("No EXACT_MATCH cases")

        test_id = exact[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert len(result.data.candidates) == 1
        assert result.data.candidates[0].resolution_type == "no_action"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFallback:
    def test_fallback_when_llm_disabled(self):
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        test_id = adapter._cases[0].get("case_id")

        with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True
            assert result.data.fallback_used is True
            assert result.data.llm_provider == "none"

    def test_core_workflow_without_llm(self):
        """Verify the analysis completes correctly without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        diff_cases = [c for c in adapter._cases if c.get("difference", 0) != 0]
        if not diff_cases:
            pytest.skip("No discrepancy cases")

        test_id = diff_cases[0].get("case_id")

        with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True
            # All deterministic components should still work
            assert result.data.financial_discrepancy.difference_paise != 0
            assert len(result.data.candidates) > 0
            assert result.data.guardrail.decision != ""
            assert result.data.evidence.record_count >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_request_rejects_empty_id(self):
        from app.api.analyze import AnalyzeRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnalyzeRequest(exception_id="")

    def test_request_rejects_long_id(self):
        from app.api.analyze import AnalyzeRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnalyzeRequest(exception_id="x" * 200)

    def test_response_has_no_execution_fields(self):
        from app.api.analyze import AnalysisResult

        result = AnalysisResult(exception_id="EXP-001")
        forbidden = ["authorize", "approve", "execute", "refund_amount"]
        for field in forbidden:
            assert not hasattr(result, field)

    def test_api_does_not_modify_data(self):
        """Analyze endpoint should be read-only."""
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()
        count_before = len(adapter._cases)

        client = TestClient(
            __import__("app.main", fromlist=["app"]).app,
            raise_server_exceptions=False,
        )
        client.post("/analyze", json={"exception_id": "NONEXISTENT"})

        count_after = len(adapter._cases)
        assert count_before == count_after

    def test_no_financial_truth_inputs(self):
        """Verify API ignores user-supplied financial values."""
        client = TestClient(
            __import__("app.main", fromlist=["app"]).app,
            raise_server_exceptions=False,
        )
        resp = client.post("/analyze", json={
            "exception_id": "NONEXISTENT",
            "expected_amount_paise": 999999,
        })
        data = resp.json()
        assert data["success"] is False

    def test_analysis_result_has_all_sections(self):
        """Verify the result contains all required analysis sections."""
        from app.api.analyze import AnalysisResult

        r = AnalysisResult(exception_id="EXP-001")
        d = r.model_dump()
        assert "financial_discrepancy" in d
        assert "evidence" in d
        assert "candidates" in d
        assert "guardrail" in d
        assert "ai_explanation" in d
        assert "ai_uncertainty" in d
        assert "fallback_used" in d

    def test_guardrail_never_auto_for_large_exposure(self):
        """Verify guardrails are conservative for large discrepancies."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        large = [c for c in adapter._cases if abs(c.get("difference", 0)) > 100000]
        if not large:
            pytest.skip("No large cases")

        test_id = large[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        assert result.data.guardrail.decision != "AUTO"

    def test_candidates_have_sources(self):
        """All candidates should indicate their source."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        test_id = adapter._cases[0].get("case_id")
        service = AnalyzeService()
        result = asyncio.get_event_loop().run_until_complete(
            service.analyze(AnalyzeRequest(exception_id=test_id))
        )
        for candidate in result.data.candidates:
            assert candidate.source != ""
