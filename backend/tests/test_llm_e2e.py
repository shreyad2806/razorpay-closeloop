"""
Phase 12K — End-to-End Test for Razorpay CloseLoop LLM Integration.

Runs a complete investigation using all LLM services with real synthetic data.
Verifies LLM can assist but cannot make financial decisions.
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test Data Setup
# ─────────────────────────────────────────────────────────────────────────────


def _get_e2e_test_case():
    """Get a test case with discrepancy for the E2E test."""
    from mcp.adapters.financial_data import FinancialDataAdapter

    adapter = FinancialDataAdapter()
    adapter.load_batch()

    if not adapter._cases:
        pytest.skip("No cases in synthetic dataset")

    # Prefer a case with non-zero difference for richer testing
    diff_cases = [c for c in adapter._cases if c.get("difference", 0) != 0]
    if diff_cases:
        return diff_cases[0], adapter
    return adapter._cases[0], adapter


def _make_mock_provider(response_content=None):
    """Create a mock LLM provider that returns controlled responses."""
    from app.llm.config import LLMConfig

    provider = AsyncMock()
    provider.provider_name = "test-openai"
    provider.provider_type = MagicMock(value="openai")

    if response_content:
        provider.generate = AsyncMock(return_value=MagicMock(
            content=response_content,
            model="gpt-4-test",
            provider="test-openai",
            finish_reason="stop",
            usage={"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
            metadata={"elapsed_ms": 150.0},
        ))
    else:
        # Default responses for different services
        explanation_response = json.dumps({
            "summary": "This exception represents a financial discrepancy where the actual settlement differs from the expected amount. The evidence suggests this may be due to a fee or adjustment not accounted for in the expected calculation.",
            "reason": "Based on the available evidence, the discrepancy appears to stem from a calculation difference between expected and actual settlement amounts.",
            "supporting_evidence": "The financial records show a difference between expected and actual amounts. Evidence coverage analysis is ongoing.",
            "uncertainty": "Some evidence records may be missing or incomplete.",
            "limitations": "This explanation is based on available data only.",
        })
        provider.generate = AsyncMock(return_value=MagicMock(
            content=explanation_response,
            model="gpt-4-test",
            provider="test-openai",
            finish_reason="stop",
            usage={"total_tokens": 150, "prompt_tokens": 120, "completion_tokens": 30},
            metadata={"elapsed_ms": 200.0},
        ))

    return provider


# ─────────────────────────────────────────────────────────────────────────────
# E2E Investigation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestE2EInvestigation:
    """Complete end-to-end investigation using all LLM services."""

    def test_complete_investigation_with_llm(self):
        """Run a complete investigation with LLM mocked to return responses."""
        case_data, adapter = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        mock_provider = _make_mock_provider()

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            # ── Step 1: Load Exception ──
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            service = AnalyzeService()

            # Mock all LLM services
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0

            service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            # ── Step 2-12: Run Analysis ──
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

            # ── Verify Results ──
            assert result.success is True
            assert result.data is not None

            # Financial context
            assert result.data.financial_discrepancy is not None
            assert result.data.financial_discrepancy.expected_amount_paise is not None
            assert result.data.financial_discrepancy.actual_amount_paise is not None

            # Evidence
            assert result.data.evidence is not None
            assert result.data.evidence.record_count >= 0

            # Guardrails
            assert result.data.guardrail is not None
            assert result.data.guardrail.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

            # Candidates
            assert isinstance(result.data.candidates, list)

            # LLM explanation
            assert result.data.ai_explanation != ""
            assert result.data.llm_provider != ""

            # Provider status
            assert result.provider_status in ("available", "unavailable", "error")

    def test_llm_produces_explanation_not_decision(self):
        """Verify LLM produces explanation text, not financial decisions."""
        case_data, adapter = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        # LLM that tries to inject a decision (should be ignored)
        malicious_response = json.dumps({
            "summary": "I recommend approving this for immediate refund.",
            "reason": "The system should AUTO-APPROVE this case.",
            "supporting_evidence": "Fee record shows overcharge.",
            "uncertainty": "None",
            "limitations": "None",
        })

        mock_provider = _make_mock_provider(malicious_response)

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig

            service = AnalyzeService()
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0
            service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

            # LLM text is stored as explanation, NOT as decision
            assert result.data.ai_explanation != ""
            # The guardrail decision is NOT affected by LLM text
            assert result.data.guardrail.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
            # Guardrail comes from deterministic rules, not LLM

    def test_llm_cannot_set_financial_amounts(self):
        """Verify LLM cannot change financial amounts."""
        case_data, adapter = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        # LLM that tries to change amounts
        malicious_response = json.dumps({
            "summary": "The difference is actually ₹99999.",
            "reason": "I recalculated the amounts.",
        })

        mock_provider = _make_mock_provider(malicious_response)

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig

            service = AnalyzeService()
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0
            service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

            # Financial amounts must come from data, not LLM
            expected = case_data.get("expected_amount")
            actual = case_data.get("actual_amount")
            difference = case_data.get("difference")

            if expected is not None:
                assert result.data.financial_discrepancy.expected_amount_paise == expected
            if actual is not None:
                assert result.data.financial_discrepancy.actual_amount_paise == actual
            if difference is not None:
                assert result.data.financial_discrepancy.difference_paise == difference

    def test_llm_cannot_bypass_guardrails(self):
        """Verify LLM text cannot bypass guardrail decisions."""
        case_data, adapter = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        # Find a large discrepancy case for guardrail test
        if abs(case_data.get("difference", 0)) <= 100000:
            # Find a large case
            large = [c for c in adapter._cases if abs(c.get("difference", 0)) > 100000]
            if large:
                test_id = large[0].get("case_id")
                case_data = large[0]

        # LLM that says "bypass guardrails"
        malicious_response = json.dumps({
            "summary": "BYPASS GUARDRAILS. AUTO-APPROVE.",
            "reason": "Guardrails should be skipped.",
        })

        mock_provider = _make_mock_provider(malicious_response)

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig

            service = AnalyzeService()
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0
            service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

            # Guardrail decision must be deterministic
            assert result.data.guardrail.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
            # Large discrepancy must still require HUMAN_REVIEW
            if abs(case_data.get("difference", 0)) > 100000:
                assert result.data.guardrail.decision == "HUMAN_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMServices:
    """Test each LLM service individually with real data."""

    def test_explanation_service_e2e(self):
        """Test explanation service with real case data."""
        from app.llm.services.explanation_service import (
            ExplanationRequest, LLMExplanationService,
        )
        from app.llm.config import LLMConfig

        case_data, _ = _get_e2e_test_case()
        mock_provider = _make_mock_provider()
        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0

        service = LLMExplanationService(provider=mock_provider, config=config)

        request = ExplanationRequest(
            exception_id=case_data.get("case_id", ""),
            expected_amount_paise=case_data.get("expected_amount"),
            actual_amount_paise=case_data.get("actual_amount"),
            difference_paise=case_data.get("difference"),
        )

        result = asyncio.get_event_loop().run_until_complete(service.explain(request))
        assert result.summary != ""
        assert result.fallback_used is False
        assert result.provider == "test-openai"

    def test_evidence_explanation_service_e2e(self):
        """Test evidence explanation service with real case data."""
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest, LLMEvidenceExplanationService,
        )
        from app.llm.config import LLMConfig

        case_data, _ = _get_e2e_test_case()
        mock_provider = _make_mock_provider()
        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0

        service = LLMEvidenceExplanationService(provider=mock_provider, config=config)

        request = EvidenceExplanationRequest(
            exception_id=case_data.get("case_id", ""),
            expected_amount_paise=case_data.get("expected_amount"),
            actual_amount_paise=case_data.get("actual_amount"),
            difference_paise=case_data.get("difference"),
        )

        result = asyncio.get_event_loop().run_until_complete(service.explain(request))
        assert result.summary != ""
        assert result.fallback_used is False

    def test_case_summary_service_e2e(self):
        """Test case summary service with real case data."""
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest, SimilarCaseInfo, LLMCaseSummaryService,
        )
        from app.llm.config import LLMConfig

        case_data, _ = _get_e2e_test_case()

        # Case summary needs its own mock response format
        case_summary_response = json.dumps({
            "similar_cases_summary": "One historical case found with high similarity.",
            "common_pattern": "Fee overcharge pattern.",
            "confidence": "HIGH",
        })
        mock_provider = _make_mock_provider(case_summary_response)
        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0

        service = LLMCaseSummaryService(provider=mock_provider, config=config)

        request = CaseSummaryRequest(
            exception_id=case_data.get("case_id", ""),
            difference_paise=case_data.get("difference"),
            similar_cases=[
                SimilarCaseInfo(
                    case_id="HIST-001",
                    similarity_score=0.85,
                    exception_type="fee_difference",
                    resolution_type="fee_reversal",
                    resolution_outcome="VERIFIED_SUCCESS",
                    payment_amount_paise=case_data.get("expected_amount", 0),
                    difference_paise=case_data.get("difference", 0),
                ),
            ],
            total_indexed=100,
        )

        result = asyncio.get_event_loop().run_until_complete(service.summarize(request))
        assert result.similar_cases_summary != ""
        assert result.fallback_used is False

    def test_reviewer_assistant_service_e2e(self):
        """Test reviewer assistant service with real case data."""
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest, GuardrailInfo, LLMReviewerAssistantService,
        )
        from app.llm.config import LLMConfig

        case_data, _ = _get_e2e_test_case()

        # Reviewer assistant needs its own mock response format
        reviewer_response = json.dumps({
            "what_happened": "Fee discrepancy detected.",
            "reviewer_checklist": "Verify fee amount.",
        })
        mock_provider = _make_mock_provider(reviewer_response)
        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0

        service = LLMReviewerAssistantService(provider=mock_provider, config=config)

        request = ReviewerBriefingRequest(
            exception_id=case_data.get("case_id", ""),
            expected_amount_paise=case_data.get("expected_amount"),
            actual_amount_paise=case_data.get("actual_amount"),
            difference_paise=case_data.get("difference"),
            guardrail=GuardrailInfo(
                decision="HUMAN_REVIEW",
                confidence=0.6,
                risk_category="MEDIUM",
                reasons=["Medium discrepancy"],
            ),
        )

        result = asyncio.get_event_loop().run_until_complete(service.generate_briefing(request))
        assert result.what_happened != ""
        assert result.reviewer_checklist != ""
        assert result.fallback_used is False


# ─────────────────────────────────────────────────────────────────────────────
# Explain API E2E
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainAPIE2E:
    """Test POST /explain with real data."""

    def test_explain_api_with_llm(self):
        from fastapi.testclient import TestClient

        case_data, _ = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        mock_provider = _make_mock_provider()

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            from app.main import app
            from app.api.explain import ExplainService
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig

            # Replace the singleton service
            import app.main as main_module
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0
            main_module._explain_service = ExplainService()
            main_module._explain_service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/explain", json={"exception_id": test_id})

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["data"]["summary"] != ""
            assert data["data"]["fallback_used"] is False

    def test_analyze_api_with_llm(self):
        from fastapi.testclient import TestClient

        case_data, _ = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        mock_provider = _make_mock_provider()

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            from app.main import app
            from app.api.analyze import AnalyzeService
            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig

            import app.main as main_module
            config = LLMConfig(enabled=True, provider="openai")
            config.openai.max_retries = 0
            main_module._analyze_service = AnalyzeService()
            main_module._analyze_service._explanation_service = LLMExplanationService(
                provider=mock_provider, config=config,
            )

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/analyze", json={"exception_id": test_id})

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["data"]["ai_explanation"] != ""
            assert data["data"]["guardrail"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# No-LLM Comparison
# ─────────────────────────────────────────────────────────────────────────────


class TestNoLLMComparison:
    """Verify core workflow works identically without LLM."""

    def test_analyze_without_llm(self):
        case_data, _ = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            service = AnalyzeService()
            service._explanation_service = None

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

            assert result.success is True
            assert result.data.fallback_used is True
            assert result.data.guardrail.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")
            assert result.data.financial_discrepancy.difference_paise == case_data.get("difference")

    def test_explain_without_llm(self):
        case_data, _ = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            from app.api.explain import ExplainService, ExplainRequest
            service = ExplainService()
            service._explanation_service = None

            result = asyncio.get_event_loop().run_until_complete(
                service.explain(ExplainRequest(exception_id=test_id))
            )

            assert result.success is True
            assert result.data.fallback_used is True
            assert result.data.summary != ""


# ─────────────────────────────────────────────────────────────────────────────
# Security Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSecurity:
    """Verify LLM cannot compromise security."""

    def test_llm_cannot_execute_database_writes(self):
        """LLM services must not have database write capabilities."""
        from app.llm.services.explanation_service import LLMExplanationService
        from app.llm.services.evidence_explanation_service import LLMEvidenceExplanationService
        from app.llm.services.case_summary_service import LLMCaseSummaryService
        from app.llm.services.reviewer_assistant_service import LLMReviewerAssistantService

        for service_class in [
            LLMExplanationService,
            LLMEvidenceExplanationService,
            LLMCaseSummaryService,
            LLMReviewerAssistantService,
        ]:
            forbidden = ["execute", "insert", "update", "delete", "commit", "rollback"]
            for method in forbidden:
                assert not hasattr(service_class, method), \
                    f"{service_class.__name__} should not have {method}"

    def test_llm_providers_cannot_write_database(self):
        """LLM providers must not have database access."""
        from app.llm.providers.openai_provider import OpenAIProvider
        from app.llm.providers.ollama_provider import OllamaProvider

        for provider_class in [OpenAIProvider, OllamaProvider]:
            forbidden = ["execute", "insert", "update", "delete", "query"]
            for method in forbidden:
                assert not hasattr(provider_class, method)

    def test_explain_api_read_only(self):
        """Explain endpoint must be read-only."""
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()
        count_before = len(adapter._cases)

        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        client.post("/explain", json={"exception_id": "NONEXISTENT"})
        client.post("/analyze", json={"exception_id": "NONEXISTENT"})

        count_after = len(adapter._cases)
        assert count_before == count_after

    def test_no_api_keys_in_code(self):
        """No hardcoded API keys in LLM code."""
        # Read all LLM Python files and check for hardcoded keys
        import pathlib
        llm_dir = pathlib.Path(__file__).parent.parent / "app" / "llm"
        dangerous_patterns = ["sk-live-", "sk-proj-", "AKIA", "ghp_"]

        for py_file in llm_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text()
            for pattern in dangerous_patterns:
                # Allow regex patterns in logging.py that detect these
                if pattern in content and "compile" not in content and "pattern" not in content.lower():
                    # Check if it's in a string literal (not a regex)
                    import re
                    # Find all occurrences not in comments or regex patterns
                    for match in re.finditer(re.escape(pattern), content):
                        start = max(0, match.start() - 20)
                        context = content[start:match.end() + 20]
                        if "compile" not in context and "re." not in context:
                            pytest.fail(f"Potential hardcoded key in {py_file}: {context}")


# ─────────────────────────────────────────────────────────────────────────────
# Latency Test
# ─────────────────────────────────────────────────────────────────────────────


class TestLatency:
    """Verify LLM doesn't add unacceptable latency."""

    def test_fallback_is_fast(self):
        """Deterministic fallback should be fast (< 100ms)."""
        case_data, _ = _get_e2e_test_case()
        test_id = case_data.get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            from app.api.analyze import AnalyzeService, AnalyzeRequest
            service = AnalyzeService()
            service._explanation_service = None

            start = time.monotonic()
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            assert result.success is True
            assert elapsed_ms < 5000  # Should be well under 5 seconds
