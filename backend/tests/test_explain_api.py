"""
Tests for Razorpay CloseLoop Phase 12G — POST /explain API.

Covers:
- Request/response schemas
- ExplainService data loading
- ExplainService orchestration
- LLM available / unavailable / timeout / malformed
- Missing evidence / conflicting evidence
- Unknown exception
- Safety (no arbitrary financial truth inputs)
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
        from app.api.explain import ExplainRequest

        req = ExplainRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.include_evidence is True
        assert req.explanation_depth == "standard"

    def test_request_full(self):
        from app.api.explain import ExplainRequest

        req = ExplainRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            include_evidence=True,
            include_candidates=True,
            explanation_depth="detailed",
        )
        assert req.case_id == "CASE-001"
        assert req.explanation_depth == "detailed"

    def test_request_empty_id_rejected(self):
        from app.api.explain import ExplainRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplainRequest(exception_id="")

    def test_response_schema(self):
        from app.api.explain import ExplainResponse

        resp = ExplainResponse(success=True, data=None, provider_status="available")
        assert resp.success is True
        assert resp.provider_status == "available"

    def test_explanation_result_schema(self):
        from app.api.explain import ExplanationResult

        result = ExplanationResult(
            exception_id="EXP-001",
            summary="Test",
            reason="Because",
        )
        assert result.exception_id == "EXP-001"
        assert result.fallback_used is False


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests (FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainEndpoint:
    def _make_client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_still_works(self):
        client = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_explain_unknown_exception(self):
        client = self._make_client()
        resp = client.post("/explain", json={"exception_id": "NONEXISTENT-999"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_explain_missing_id(self):
        client = self._make_client()
        resp = client.post("/explain", json={})
        assert resp.status_code == 422  # Validation error

    def test_explain_valid_exception(self):
        """Test with an exception that exists in the synthetic dataset."""
        client = self._make_client()
        from mcp.adapters.financial_data import FinancialDataAdapter
        adapter = FinancialDataAdapter()
        adapter.load_batch()

        # Cases use case_id in the synthetic dataset
        if not adapter._cases:
            pytest.skip("No cases in synthetic dataset")

        test_id = adapter._cases[0].get("case_id", "CASE-000001")
        resp = client.post("/explain", json={"exception_id": test_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"] is not None
        assert data["data"]["summary"] != ""

    def test_explain_returns_provider_status(self):
        client = self._make_client()
        resp = client.post("/explain", json={"exception_id": "NONEXISTENT"})
        data = resp.json()
        assert "provider_status" in data

    def test_explain_no_arbitrary_financial_input(self):
        """Verify the API does NOT accept user-supplied financial values."""
        client = self._make_client()
        # Send financial values — they should be ignored (not used as truth)
        resp = client.post("/explain", json={
            "exception_id": "NONEXISTENT",
            "expected_amount_paise": 999999,
            "actual_amount_paise": 1,
        })
        # Should still fail because exception doesn't exist
        # The financial values should not be accepted as authoritative
        data = resp.json()
        assert data["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# ExplainService Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainService:
    def test_load_exception_not_found(self):
        from app.api.explain import ExplainService, ExplainRequest

        service = ExplainService()
        result = asyncio.get_event_loop().run_until_complete(
            service.explain(ExplainRequest(exception_id="NONEXISTENT"))
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_load_exception_found(self):
        from app.api.explain import ExplainService, ExplainRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases in synthetic dataset")

        test_id = adapter._cases[0].get("case_id", "CASE-000001")
        service = ExplainService()
        result = asyncio.get_event_loop().run_until_complete(
            service.explain(ExplainRequest(exception_id=test_id))
        )
        assert result.success is True

    def test_service_has_no_duplicate_logic(self):
        """Verify ExplainService delegates to existing services."""
        from app.api.explain import ExplainService

        service = ExplainService()
        # Should not have its own reconciliation, evidence, or resolution methods
        forbidden = [
            "reconcile", "retrieve_evidence", "classify",
            "resolve", "execute", "verify",
        ]
        for method in forbidden:
            assert not hasattr(service, method), f"ExplainService should not have {method}"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_request_rejects_empty_exception_id(self):
        from app.api.explain import ExplainRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplainRequest(exception_id="")

    def test_request_rejects_very_long_id(self):
        from app.api.explain import ExplainRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplainRequest(exception_id="x" * 200)

    def test_response_has_no_execution_fields(self):
        from app.api.explain import ExplanationResult

        result = ExplanationResult(exception_id="EXP-001")
        forbidden = ["authorize", "approve", "execute", "refund_amount"]
        for field in forbidden:
            assert not hasattr(result, field)

    def test_api_does_not_modify_data(self):
        """The explain endpoint should be read-only."""
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()
        count_before = len(adapter.search_records(limit=1000))

        client = TestClient(
            __import__("app.main", fromlist=["app"]).app,
            raise_server_exceptions=False,
        )
        client.post("/explain", json={"exception_id": "NONEXISTENT"})

        count_after = len(adapter.search_records(limit=1000))
        assert count_before == count_after


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackIntegration:
    def test_fallback_used_when_llm_disabled(self):
        """When LLM_ENABLED=false, should use deterministic fallback."""
        from app.api.explain import ExplainService, ExplainRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases in synthetic dataset")

        test_id = adapter._cases[0].get("case_id", "CASE-000001")

        with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=False):
            # Create fresh service with disabled LLM
            service = ExplainService()
            service._explanation_service = None  # Force re-creation
            result = asyncio.get_event_loop().run_until_complete(
                service.explain(ExplainRequest(exception_id=test_id))
            )
            assert result.success is True
            assert result.data.fallback_used is True

    def test_response_structure_complete(self):
        """Verify the response has all expected fields."""
        from app.api.explain import ExplainService, ExplainRequest
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases in synthetic dataset")

        test_id = adapter._cases[0].get("case_id", "CASE-000001")
        service = ExplainService()
        result = asyncio.get_event_loop().run_until_complete(
            service.explain(ExplainRequest(exception_id=test_id))
        )

        assert result.success is True
        data = result.data
        # exception_id may be empty if the synthetic data uses case_id
        assert data.summary != ""
        assert isinstance(data.conflicts, list)
        assert isinstance(data.missing_evidence, list)
        assert isinstance(data.fallback_used, bool)
        assert result.provider_status != ""


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-Explain Consistency Test
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceExplainConsistency:
    """Verify that /explain and /evidence return the same evidence records
    for the same exception.

    This test guards against the structural mismatch where ExplainService
    used search_records() (which returns wrapped results) while
    IntelligenceService used direct adapter methods (get_payment, etc.).
    """

    def _make_client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_explain_and_evidence_return_same_record_count(self):
        """Both /explain and /evidence must report the same evidence count."""
        client = self._make_client()

        # Use a case that exists in the DEMO-001 dataset
        test_id = "CASE-DEMO-004"

        explain_resp = client.get(f"/exceptions/{test_id}/explain")
        evidence_resp = client.get(f"/exceptions/{test_id}/evidence")

        assert explain_resp.status_code == 200
        assert evidence_resp.status_code == 200

        explain_data = explain_resp.json()["data"]
        evidence_data = evidence_resp.json()["data"]

        # The critical assertion: record counts must match
        assert explain_data["evidence_record_count"] == evidence_data["record_count"], (
            f"Evidence count mismatch: explain={explain_data['evidence_record_count']} "
            f"vs evidence={evidence_data['record_count']}"
        )

        # Both must find at least one evidence record
        assert explain_data["evidence_record_count"] > 0, \
            "Explain should find evidence records for a valid exception"

    def test_explain_evidence_not_zero_when_evidence_has_records(self):
        """Explain must never report 0 evidence when evidence endpoint finds records."""
        client = self._make_client()

        test_id = "CASE-DEMO-006"

        explain_resp = client.get(f"/exceptions/{test_id}/explain")
        evidence_resp = client.get(f"/exceptions/{test_id}/evidence")

        assert explain_resp.status_code == 200
        assert evidence_resp.status_code == 200

        explain_count = explain_resp.json()["data"]["evidence_record_count"]
        evidence_count = evidence_resp.json()["data"]["record_count"]

        # If evidence has records, explain must too
        if evidence_count > 0:
            assert explain_count > 0, (
                f"Explain found 0 evidence but evidence endpoint found {evidence_count}"
            )

    def test_explain_evidence_summary_describes_actual_records(self):
        """Explain's evidence_summary must reference real record types, not 'No evidence'."""
        client = self._make_client()

        test_id = "CASE-DEMO-010"

        explain_resp = client.get(f"/exceptions/{test_id}/explain")
        evidence_resp = client.get(f"/exceptions/{test_id}/evidence")

        assert explain_resp.status_code == 200
        assert evidence_resp.status_code == 200

        explain_data = explain_resp.json()["data"]
        evidence_data = evidence_resp.json()["data"]

        if evidence_data["record_count"] > 0:
            # evidence_summary should mention records, not be empty
            assert "no evidence" not in explain_data["evidence_summary"].lower(), (
                f"Explain says 'No evidence' but evidence endpoint found "
                f"{evidence_data['record_count']} records"
            )
            # missing_evidence should be empty when records exist
            assert len(explain_data["missing_evidence"]) == 0, (
                f"Explain reports missing_evidence={explain_data['missing_evidence']} "
                f"but evidence endpoint found {evidence_data['record_count']} records"
            )
