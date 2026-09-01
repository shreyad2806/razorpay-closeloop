"""
Tests for Razorpay CloseLoop Phase 12E — LLM Case Summary Service.

Covers:
- Schemas (request, output, from_similarity_result)
- Prompt building
- Deterministic fallback (high/low similarity, none, conflicting, misleading)
- LLM response parsing
- LLM-generated summary (mocked)
- Failure handling
- Safety boundary (LLM cannot decide case equality, copy resolution, change values)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    """Tests for input/output schemas."""

    def test_request_minimal(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest

        req = CaseSummaryRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.similar_cases == []

    def test_request_with_cases(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest, SimilarCaseInfo

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            exception_type="fee_difference",
            difference_paise=5000,
            similar_cases=[
                SimilarCaseInfo(
                    case_id="HIST-001", similarity_score=0.92,
                    exception_type="fee_difference", resolution_type="fee_reversal",
                    resolution_outcome="VERIFIED_SUCCESS", payment_amount_paise=100000,
                    difference_paise=5000, evidence_count=3, tags=["fee", "platform"],
                ),
            ],
            total_indexed=150,
        )
        assert len(req.similar_cases) == 1
        assert req.similar_cases[0].similarity_score == 0.92
        assert req.total_indexed == 150

    def test_output_schema(self):
        from app.llm.services.case_summary_service import CaseSummaryOutput

        out = CaseSummaryOutput(
            similar_cases_summary="Test",
            common_pattern="Pattern",
            confidence="HIGH",
        )
        d = out.to_dict()
        assert d["similar_cases_summary"] == "Test"
        assert d["confidence"] == "HIGH"

    def test_from_similarity_result_dict(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest

        result = {
            "query_case_id": "EXP-001",
            "similar_cases": [
                {
                    "case_id": "HIST-001",
                    "similarity_score": 0.85,
                    "exception_type": "fee_difference",
                    "resolution_type": "fee_reversal",
                    "resolution_outcome": "VERIFIED_SUCCESS",
                    "payment_amount": 100000,
                    "difference": 5000,
                    "evidence_count": 3,
                    "tags": ["fee"],
                },
            ],
            "total_indexed": 100,
            "similarity_metric": "cosine",
            "top_k": 5,
        }

        req = CaseSummaryRequest.from_similarity_result(result, exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert len(req.similar_cases) == 1
        assert req.similar_cases[0].case_id == "HIST-001"
        assert req.total_indexed == 100

    def test_from_similarity_result_empty(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest

        req = CaseSummaryRequest.from_similarity_result(
            {"similar_cases": []}, exception_id="EXP-001"
        )
        assert req.similar_cases == []


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Building Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptBuilding:
    """Tests for build_case_summary_prompt."""

    def test_minimal_prompt(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest, build_case_summary_prompt

        req = CaseSummaryRequest(exception_id="EXP-001")
        prompt = build_case_summary_prompt(req)
        assert "EXP-001" in prompt
        assert "No similar historical cases" in prompt

    def test_prompt_with_cases(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            build_case_summary_prompt,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            exception_type="fee_difference",
            difference_paise=5000,
            similar_cases=[
                SimilarCaseInfo(
                    case_id="HIST-001", similarity_score=0.92,
                    exception_type="fee_difference",
                    resolution_type="fee_reversal",
                    resolution_outcome="VERIFIED_SUCCESS",
                    payment_amount_paise=100000,
                    difference_paise=5000,
                    evidence_count=3,
                    tags=["fee", "platform"],
                ),
            ],
            total_indexed=150,
        )
        prompt = build_case_summary_prompt(req)
        assert "EXP-001" in prompt
        assert "HIST-001" in prompt
        assert "92" in prompt  # similarity score 0.92 formatted as 92.0%
        assert "fee_reversal" in prompt
        assert "VERIFIED_SUCCESS" in prompt
        assert "fee, platform" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicFallback:
    """Tests for _case_summary_deterministic_fallback."""

    def test_no_cases(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest, _case_summary_deterministic_fallback

        req = CaseSummaryRequest(exception_id="EXP-001", total_indexed=100)
        result = _case_summary_deterministic_fallback(req)
        assert result.fallback_used is True
        assert "No similar historical cases" in result.similar_cases_summary
        assert "LOW" in result.confidence
        assert "REFERENCE ONLY" in result.recommendation_note

    def test_high_similarity(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            similar_cases=[
                SimilarCaseInfo(
                    case_id="H-001", similarity_score=0.95,
                    exception_type="fee_difference",
                    resolution_type="fee_reversal",
                    resolution_outcome="VERIFIED_SUCCESS",
                    difference_paise=5000,
                ),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "1 similar" in result.similar_cases_summary
        assert "high similarity" in result.similar_cases_summary.lower()
        assert "HIGH" in result.confidence
        assert "fee_reversal" in result.historical_resolution_summary

    def test_low_similarity(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            similar_cases=[
                SimilarCaseInfo(
                    case_id="H-001", similarity_score=0.3,
                    exception_type="unknown",
                    resolution_type="escalation",
                    resolution_outcome="ESCALATED",
                ),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "LOW" in result.confidence
        assert "low similarity" in result.uncertainty.lower()

    def test_multiple_conflicting_cases(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            similar_cases=[
                SimilarCaseInfo(
                    case_id="H-001", similarity_score=0.75,
                    resolution_type="fee_reversal",
                    resolution_outcome="VERIFIED_SUCCESS",
                ),
                SimilarCaseInfo(
                    case_id="H-002", similarity_score=0.70,
                    resolution_type="settlement_adjustment",
                    resolution_outcome="VERIFIED_SUCCESS",
                ),
                SimilarCaseInfo(
                    case_id="H-003", similarity_score=0.65,
                    resolution_type="escalation",
                    resolution_outcome="ESCALATED",
                ),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "3 similar" in result.similar_cases_summary
        assert "fee_reversal" in result.historical_resolution_summary
        assert "settlement_adjustment" in result.historical_resolution_summary
        assert "MEDIUM" in result.confidence

    def test_mixed_similarity(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            similar_cases=[
                SimilarCaseInfo(case_id="H-001", similarity_score=0.92, resolution_type="fee_reversal",
                               resolution_outcome="SUCCESS", difference_paise=5000),
                SimilarCaseInfo(case_id="H-002", similarity_score=0.3, resolution_type="escalation",
                               resolution_outcome="ESCALATED", difference_paise=10000),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "1 case(s) with high similarity" in result.similar_cases_summary
        assert "1 case(s) with low similarity" in result.similar_cases_summary

    def test_discrepancy_difference_highlighted(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            similar_cases=[
                SimilarCaseInfo(
                    case_id="H-001", similarity_score=0.8,
                    resolution_type="fee_reversal",
                    resolution_outcome="SUCCESS",
                    difference_paise=50000,  # Much larger
                ),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "differs significantly" in result.important_differences.lower()

    def test_recommendation_note_always_present(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            similar_cases=[
                SimilarCaseInfo(case_id="H-001", similarity_score=0.9,
                               resolution_type="fee_reversal",
                               resolution_outcome="SUCCESS"),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "REFERENCE ONLY" in result.recommendation_note
        assert "Phase 5" in result.recommendation_note
        assert "Phase 6" in result.recommendation_note


# ─────────────────────────────────────────────────────────────────────────────
# Response Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseParsing:
    """Tests for _parse_case_summary_response."""

    def test_parse_valid_json(self):
        from app.llm.services.case_summary_service import _parse_case_summary_response

        data = {
            "similar_cases_summary": "Cases show fee pattern",
            "common_pattern": "Platform fee overcharge",
            "important_differences": "Current amount is larger",
            "historical_resolution_summary": "Most used fee_reversal",
            "confidence": "HIGH",
            "uncertainty": "None",
            "recommendation_note": "Reference only",
        }
        result = _parse_case_summary_response(json.dumps(data))
        assert result.similar_cases_summary == "Cases show fee pattern"
        assert result.confidence == "HIGH"
        assert result.fallback_used is False

    def test_parse_plain_text(self):
        from app.llm.services.case_summary_service import _parse_case_summary_response

        result = _parse_case_summary_response("Plain text case summary.")
        assert result.similar_cases_summary == "Plain text case summary."

    def test_parse_empty(self):
        from app.llm.services.case_summary_service import _parse_case_summary_response

        result = _parse_case_summary_response("")
        assert "No case summary" in result.similar_cases_summary


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMCaseSummaryService:
    """Tests for LLMCaseSummaryService with mocked provider."""

    def _make_service(self, response_content=None, side_effect=None):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.case_summary_service import LLMCaseSummaryService

        provider = AsyncMock()
        provider.provider_name = "openai"

        if side_effect:
            provider.generate = AsyncMock(side_effect=side_effect)
        else:
            provider.generate = AsyncMock(return_value=MagicMock(
                content=response_content or json.dumps({
                    "similar_cases_summary": "LLM summary",
                    "confidence": "HIGH",
                }),
                model="gpt-4", provider="openai",
                finish_reason="stop", usage={"total_tokens": 40},
                metadata={"elapsed_ms": 180.0},
            ))

        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0
        logger = LLMLogger("test")

        return LLMCaseSummaryService(
            provider=provider, config=config, logger=logger,
        ), provider, logger

    def test_summarize_with_llm(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest

        service, provider, _ = self._make_service()
        req = CaseSummaryRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.summarize(req))
        assert result.similar_cases_summary == "LLM summary"
        assert result.fallback_used is False
        provider.generate.assert_awaited_once()

    def test_summarize_without_llm(self):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.case_summary_service import CaseSummaryRequest, LLMCaseSummaryService

        service = LLMCaseSummaryService(
            provider=None, config=LLMConfig(enabled=False), logger=LLMLogger("test"),
        )
        req = CaseSummaryRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.summarize(req))
        assert result.fallback_used is True

    def test_timeout_fallback(self):
        from app.llm.providers.base import LLMTimeoutError
        from app.llm.services.case_summary_service import CaseSummaryRequest

        service, _, _ = self._make_service(side_effect=LLMTimeoutError("timeout"))
        req = CaseSummaryRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.summarize(req))
        assert result.fallback_used is True

    def test_connection_error_fallback(self):
        from app.llm.providers.base import LLMConnectionError
        from app.llm.services.case_summary_service import CaseSummaryRequest

        service, _, _ = self._make_service(side_effect=LLMConnectionError("refused"))
        req = CaseSummaryRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.summarize(req))
        assert result.fallback_used is True

    def test_logs_start_and_success(self):
        from app.llm.logging import LLMEventType
        from app.llm.services.case_summary_service import CaseSummaryRequest

        service, _, logger = self._make_service()
        req = CaseSummaryRequest(exception_id="EXP-001", workflow_id="WF-001")
        asyncio.get_event_loop().run_until_complete(service.summarize(req))
        starts = logger.get_entries(event_type=LLMEventType.REQUEST_START)
        successes = logger.get_entries(event_type=LLMEventType.REQUEST_SUCCESS)
        assert len(starts) == 1
        assert len(successes) == 1

    def test_health_check_no_provider(self):
        from app.llm.config import LLMConfig
        from app.llm.services.case_summary_service import LLMCaseSummaryService

        service = LLMCaseSummaryService(provider=None, config=LLMConfig(enabled=False))

        async def run():
            return await service.health_check()

        status = asyncio.get_event_loop().run_until_complete(run())
        assert status.healthy is True
        assert status.provider == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    """Verify LLM case summary cannot influence financial decisions."""

    def test_system_prompt_prohibits_case_equality(self):
        from app.llm.services.case_summary_service import CASE_SUMMARY_SYSTEM_PROMPT

        assert "decide" in CASE_SUMMARY_SYSTEM_PROMPT.lower()
        assert "equals" in CASE_SUMMARY_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_copying_resolution(self):
        from app.llm.services.case_summary_service import CASE_SUMMARY_SYSTEM_PROMPT

        assert "copy" in CASE_SUMMARY_SYSTEM_PROMPT.lower()
        assert "resolution" in CASE_SUMMARY_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_changing_amounts(self):
        from app.llm.services.case_summary_service import CASE_SUMMARY_SYSTEM_PROMPT

        assert "change" in CASE_SUMMARY_SYSTEM_PROMPT.lower()
        assert "financial" in CASE_SUMMARY_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_bypassing_scoring(self):
        from app.llm.services.case_summary_service import CASE_SUMMARY_SYSTEM_PROMPT

        assert "bypass" in CASE_SUMMARY_SYSTEM_PROMPT.lower()

    def test_system_prompt_requires_only_provided_cases(self):
        from app.llm.services.case_summary_service import CASE_SUMMARY_SYSTEM_PROMPT

        assert "ONLY" in CASE_SUMMARY_SYSTEM_PROMPT
        assert "provided" in CASE_SUMMARY_SYSTEM_PROMPT.lower()

    def test_service_has_no_financial_methods(self):
        from app.llm.services.case_summary_service import LLMCaseSummaryService

        forbidden = [
            "execute_resolution", "issue_refund", "modify_settlement",
            "copy_resolution", "decide_case_equality", "bypass_scoring",
        ]
        for method in forbidden:
            assert not hasattr(LLMCaseSummaryService, method)

    def test_output_has_no_financial_authorization(self):
        from app.llm.services.case_summary_service import CaseSummaryOutput

        out = CaseSummaryOutput()
        forbidden = ["authorize", "approve", "execute", "force_resolution"]
        for field in forbidden:
            assert not hasattr(out, field)

    def test_request_has_no_modification_fields(self):
        from app.llm.services.case_summary_service import CaseSummaryRequest

        req = CaseSummaryRequest(exception_id="EXP-001")
        # Check for data fields that would indicate financial control
        # (Pydantic models may have a .copy() method — that's not a risk)
        dangerous_fields = ["execute", "approve", "override", "force_resolution"]
        for field in dangerous_fields:
            assert field not in req.model_fields, f"Request should not have field: {field}"

    def test_fallback_always_notes_reference_only(self):
        from app.llm.services.case_summary_service import (
            CaseSummaryRequest,
            SimilarCaseInfo,
            _case_summary_deterministic_fallback,
        )

        req = CaseSummaryRequest(
            exception_id="EXP-001",
            similar_cases=[
                SimilarCaseInfo(case_id="H-001", similarity_score=0.95,
                               resolution_type="fee_reversal",
                               resolution_outcome="SUCCESS"),
            ],
            total_indexed=100,
        )
        result = _case_summary_deterministic_fallback(req)
        assert "REFERENCE ONLY" in result.recommendation_note
        assert "Phase 5" in result.recommendation_note
        assert "Phase 6" in result.recommendation_note
