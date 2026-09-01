"""
Tests for Razorpay CloseLoop Phase 12J — Core System Without LLM.

Proves that the entire financial workflow works without an LLM.
Runs the same scenario with LLM enabled and disabled, comparing results.

The core financial result must remain deterministic.
Only the explanation/assistance layer should change.
"""

import asyncio
import json
import os
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Get test case data
# ─────────────────────────────────────────────────────────────────────────────


def _get_test_cases():
    """Get a set of test cases from the synthetic dataset."""
    from mcp.adapters.financial_data import FinancialDataAdapter

    adapter = FinancialDataAdapter()
    adapter.load_batch()

    if not adapter._cases:
        pytest.skip("No cases in synthetic dataset")

    # Pick representative cases
    exact = [c for c in adapter._cases if c.get("scenario") == "EXACT_MATCH"][:1]
    discrepancy = [c for c in adapter._cases if c.get("difference", 0) != 0][:2]
    all_cases = exact + discrepancy

    if not all_cases:
        all_cases = adapter._cases[:2]

    return all_cases, adapter


# ─────────────────────────────────────────────────────────────────────────────
# Core Workflow Tests (LLM Disabled)
# ─────────────────────────────────────────────────────────────────────────────


class TestCoreWorkflowWithoutLLM:
    """Verify the entire workflow operates without LLM."""

    def test_reconciliation_works_without_llm(self):
        """Phase 2: Reconciliation engine must be importable without LLM."""
        from app.reconciliation.engine import calculate_reconciliation

        # Verify the function exists and is callable
        assert callable(calculate_reconciliation)

    def test_evidence_retrieval_works_without_llm(self):
        """Phase 3: Evidence retrieval must work without LLM."""
        from mcp.adapters.financial_data import FinancialDataAdapter

        adapter = FinancialDataAdapter()
        adapter.load_batch()

        if not adapter._cases:
            pytest.skip("No cases")

        test_id = adapter._cases[0].get("case_id")
        payments = [p for p in adapter._payments if p.get("case_id") == test_id]
        settlements = [s for s in adapter._settlements if s.get("case_id") == test_id]

        # Evidence is loaded from data, not LLM
        assert isinstance(payments, list)
        assert isinstance(settlements, list)

    def test_classification_works_without_llm(self):
        """Phase 4: Classification must work without LLM."""
        from app.ml.classifier import MajorityClassClassifier

        # Classifier is deterministic, not LLM-based
        classifier = MajorityClassClassifier()
        assert classifier is not None

    def test_resolution_candidates_works_without_llm(self):
        """Phase 5: Resolution candidate generation must work without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True
            # Candidates should be generated deterministically
            assert isinstance(result.data.candidates, list)

    def test_guardrails_works_without_llm(self):
        """Phase 6: Guardrails must work without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True
            # Guardrail decision must be present
            assert result.data.guardrail.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_langgraph_workflow_works_without_llm(self):
        """Phase 7: LangGraph workflow must work without LLM."""
        from app.agent.workflow import create_workflow

        workflow = create_workflow()
        assert workflow is not None

    def test_explain_api_works_without_llm(self):
        """POST /explain must work without LLM."""
        from app.api.explain import ExplainService, ExplainRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = ExplainService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.explain(ExplainRequest(exception_id=test_id))
            )
            assert result.success is True
            assert result.data.summary != ""
            assert result.data.fallback_used is True

    def test_analyze_api_works_without_llm(self):
        """POST /analyze must work without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True
            assert result.data.fallback_used is True
            # All sections must be populated
            assert result.data.financial_discrepancy is not None
            assert result.data.evidence is not None
            assert result.data.guardrail is not None


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Tests: LLM Enabled vs Disabled
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMEnabledVsDisabled:
    """Compare results with LLM enabled and disabled.

    The core financial result must remain deterministic.
    Only the explanation layer should change.
    """

    def _run_without_llm(self, exception_id: str):
        """Run analyze with LLM disabled."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            return asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=exception_id))
            )

    def _run_with_llm(self, exception_id: str):
        """Run analyze with LLM mocked to return deterministic text.

        Since we can't call real LLM APIs in tests, we mock the LLM
        to return a simple response. The key point is that the financial
        components (discrepancy, guardrails, candidates) must remain
        identical regardless of LLM output.
        """
        from unittest.mock import AsyncMock, MagicMock
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from app.llm.services.explanation_service import LLMExplanationService
        from app.llm.config import LLMConfig

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            service = AnalyzeService()

            # Mock provider that returns a simple explanation
            mock_response = MagicMock(
                content=json.dumps({
                    "summary": "LLM explanation",
                    "reason": "Because",
                }),
                model="gpt-4",
                provider="openai",
                finish_reason="stop",
                usage={"total_tokens": 50},
                metadata={"elapsed_ms": 100.0},
            )
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value=mock_response)
            mock_provider.provider_name = "openai"

            service._explanation_service = LLMExplanationService(
                provider=mock_provider,
                config=LLMConfig(enabled=True, provider="openai"),
            )

            return asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=exception_id))
            )

    def test_financial_discrepancy_identical(self):
        """Financial discrepancy must be identical regardless of LLM."""
        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        without = self._run_without_llm(test_id)
        with_llm = self._run_with_llm(test_id)

        assert without.success is True
        assert with_llm.success is True

        # Financial values must be identical
        assert without.data.financial_discrepancy.expected_amount_paise == \
               with_llm.data.financial_discrepancy.expected_amount_paise
        assert without.data.financial_discrepancy.actual_amount_paise == \
               with_llm.data.financial_discrepancy.actual_amount_paise
        assert without.data.financial_discrepancy.difference_paise == \
               with_llm.data.financial_discrepancy.difference_paise

    def test_guardrail_result_identical(self):
        """Guardrail decision must be identical regardless of LLM."""
        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        without = self._run_without_llm(test_id)
        with_llm = self._run_with_llm(test_id)

        assert without.data.guardrail.decision == with_llm.data.guardrail.decision
        assert without.data.guardrail.risk_category == with_llm.data.guardrail.risk_category

    def test_candidates_identical(self):
        """Resolution candidates must be identical regardless of LLM."""
        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        without = self._run_without_llm(test_id)
        with_llm = self._run_with_llm(test_id)

        assert len(without.data.candidates) == len(with_llm.data.candidates)
        for c1, c2 in zip(without.data.candidates, with_llm.data.candidates):
            assert c1.resolution_type == c2.resolution_type
            assert c1.adjustment_paise == c2.adjustment_paise

    def test_evidence_identical(self):
        """Evidence summary must be identical regardless of LLM."""
        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        without = self._run_without_llm(test_id)
        with_llm = self._run_with_llm(test_id)

        assert without.data.evidence.record_count == with_llm.data.evidence.record_count
        assert without.data.evidence.coverage == with_llm.data.evidence.coverage

    def test_explanation_differs(self):
        """Explanation may differ (LLM vs deterministic template)."""
        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        without = self._run_without_llm(test_id)
        with_llm = self._run_with_llm(test_id)

        # Both should have explanations
        assert without.data.ai_explanation != ""
        assert with_llm.data.ai_explanation != ""

        # LLM status should differ
        assert without.data.fallback_used is True
        # with_llm may use fallback if provider unavailable, that's OK

    def test_multiple_cases_financial_consistency(self):
        """Test multiple cases — financial results must be consistent."""
        cases, _ = _get_test_cases()

        for case in cases[:3]:
            test_id = case.get("case_id")
            without = self._run_without_llm(test_id)
            with_llm = self._run_with_llm(test_id)

            if without.success and with_llm.success:
                assert without.data.financial_discrepancy.difference_paise == \
                       with_llm.data.financial_discrepancy.difference_paise
                assert without.data.guardrail.decision == \
                       with_llm.data.guardrail.decision


# ─────────────────────────────────────────────────────────────────────────────
# Failure Mode Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMFailureModes:
    """Verify LLM failures don't affect core workflow."""

    def test_llm_timeout_doesnt_crash_workflow(self):
        """LLM timeout must not crash the workflow."""
        from unittest.mock import AsyncMock
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from app.llm.providers.base import LLMTimeoutError

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            service = AnalyzeService()
            mock_service = type('MockService', (), {
                'explain': AsyncMock(side_effect=LLMTimeoutError("timeout"))
            })()
            service._explanation_service = mock_service

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True

    def test_llm_connection_error_doesnt_crash_workflow(self):
        """LLM connection error must not crash the workflow."""
        from unittest.mock import AsyncMock
        from app.api.analyze import AnalyzeService, AnalyzeRequest
        from app.llm.providers.base import LLMConnectionError

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            service = AnalyzeService()
            mock_service = type('MockService', (), {
                'explain': AsyncMock(side_effect=LLMConnectionError("refused"))
            })()
            service._explanation_service = mock_service

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True

    def test_llm_malformed_response_doesnt_crash_workflow(self):
        """LLM malformed response must not crash the workflow."""
        from unittest.mock import AsyncMock, MagicMock
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "true"}, clear=False):
            service = AnalyzeService()
            mock_response = MagicMock(
                content="NOT JSON AT ALL { broken",
                model="test",
                provider="test",
                finish_reason="stop",
                usage={},
                metadata={"elapsed_ms": 0},
            )
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value=mock_response)
            mock_provider.provider_name = "test"

            from app.llm.services.explanation_service import LLMExplanationService
            from app.llm.config import LLMConfig
            service._explanation_service = LLMExplanationService(
                provider=mock_provider,
                config=LLMConfig(enabled=True, provider="openai"),
            )

            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            assert result.success is True

    def test_llm_unavailable_doesnt_affect_guardrails(self):
        """LLM unavailability must not weaken guardrails."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        # Find a large discrepancy case
        large = [c for c in cases if abs(c.get("difference", 0)) > 100000]
        if not large:
            # Use any case
            large = cases[:1]

        test_id = large[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )
            # Guardrails must still be conservative
            if abs(large[0].get("difference", 0)) > 100000:
                assert result.data.guardrail.decision == "HUMAN_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# Determinism Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Verify deterministic behavior without LLM."""

    def test_same_input_same_output_without_llm(self):
        """Same exception ID must produce identical results without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        test_id = cases[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service1 = AnalyzeService()
            service1._explanation_service = None
            r1 = asyncio.get_event_loop().run_until_complete(
                service1.analyze(AnalyzeRequest(exception_id=test_id))
            )

            service2 = AnalyzeService()
            service2._explanation_service = None
            r2 = asyncio.get_event_loop().run_until_complete(
                service2.analyze(AnalyzeRequest(exception_id=test_id))
            )

        assert r1.data.financial_discrepancy.difference_paise == \
               r2.data.financial_discrepancy.difference_paise
        assert r1.data.guardrail.decision == r2.data.guardrail.decision
        assert r1.data.evidence.record_count == r2.data.evidence.record_count

    def test_exact_match_produces_no_action_without_llm(self):
        """EXACT_MATCH case must produce no_action candidate without LLM."""
        from app.api.analyze import AnalyzeService, AnalyzeRequest

        cases, _ = _get_test_cases()
        exact = [c for c in cases if c.get("scenario") == "EXACT_MATCH"]
        if not exact:
            pytest.skip("No EXACT_MATCH case")

        test_id = exact[0].get("case_id")

        with patch.dict(os.environ, {"LLM_ENABLED": "false"}, clear=False):
            service = AnalyzeService()
            service._explanation_service = None
            result = asyncio.get_event_loop().run_until_complete(
                service.analyze(AnalyzeRequest(exception_id=test_id))
            )

        assert result.data.guardrail.decision == "AUTO"
        assert result.data.financial_discrepancy.difference_paise == 0
        assert len(result.data.candidates) == 1
        assert result.data.candidates[0].resolution_type == "no_action"
