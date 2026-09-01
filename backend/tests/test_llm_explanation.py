"""
Tests for Razorpay CloseLoop Phase 12C — LLM Explanation Service.

Covers:
- Explanation input/output schemas
- Prompt building
- System prompt safety constraints
- Deterministic fallback
- LLM response parsing
- LLM-generated explanation
- Failure handling (LLM unavailable, timeout, malformed response)
- Edge cases (missing evidence, conflicting evidence, unknown exception)
- Safety boundary (LLM does NOT calculate financial state)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Schemas Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExplanationSchemas:
    """Tests for ExplanationRequest and LLMExplanationOutput schemas."""

    def test_request_minimal(self):
        from app.llm.services.explanation_service import ExplanationRequest

        req = ExplanationRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.evidence_items == []
        assert req.guardrail_reasons == []

    def test_request_full(self):
        from app.llm.services.explanation_service import ExplanationEvidence, ExplanationRequest

        req = ExplanationRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            evidence_items=[
                ExplanationEvidence(
                    evidence_id="EV-001",
                    record_type="fee",
                    description="Platform fee of ₹50 charged",
                    amount_paise=5000,
                )
            ],
            evidence_coverage="FULLY_EXPLAINED",
            explained_amount_paise=5000,
            remaining_difference_paise=0,
            classification_confidence=0.92,
            similar_case_count=12,
            candidate_resolution_type="fee_reversal",
            candidate_adjustment_paise=5000,
            candidate_description="Reverse the incorrect platform fee",
            guardrail_decision="AUTO",
            guardrail_confidence=0.95,
            risk_category="LOW",
            guardrail_reasons=["Low exposure", "Clear evidence"],
        )
        assert req.exception_id == "EXP-001"
        assert len(req.evidence_items) == 1
        assert req.guardrail_decision == "AUTO"

    def test_output_schema(self):
        from app.llm.services.explanation_service import LLMExplanationOutput

        output = LLMExplanationOutput(
            summary="Test summary",
            reason="Test reason",
            supporting_evidence="Test evidence",
            uncertainty="None",
            limitations="None",
            model_used="gpt-4",
            provider="openai",
            fallback_used=False,
        )
        d = output.to_dict()
        assert d["summary"] == "Test summary"
        assert d["fallback_used"] is False

    def test_output_default_values(self):
        from app.llm.services.explanation_service import LLMExplanationOutput

        output = LLMExplanationOutput()
        assert output.summary == ""
        assert output.fallback_used is False
        assert output.model_used == ""

    def test_evidence_schema(self):
        from app.llm.services.explanation_service import ExplanationEvidence

        ev = ExplanationEvidence(
            evidence_id="EV-001",
            record_type="refund",
            description="Refund processed",
            amount_paise=3000,
        )
        assert ev.evidence_id == "EV-001"
        assert ev.amount_paise == 3000


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Building Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptBuilding:
    """Tests for build_explanation_prompt."""

    def test_minimal_prompt(self):
        from app.llm.services.explanation_service import ExplanationRequest, build_explanation_prompt

        req = ExplanationRequest(exception_id="EXP-001")
        prompt = build_explanation_prompt(req)
        assert "EXP-001" in prompt
        assert "Financial Exception Explanation Request" in prompt

    def test_full_prompt(self):
        from app.llm.services.explanation_service import ExplanationEvidence, ExplanationRequest, build_explanation_prompt

        req = ExplanationRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            evidence_items=[
                ExplanationEvidence(
                    evidence_id="EV-001",
                    record_type="fee",
                    description="Platform fee",
                    amount_paise=5000,
                )
            ],
            evidence_coverage="FULLY_EXPLAINED",
            candidate_resolution_type="fee_reversal",
            candidate_adjustment_paise=5000,
            guardrail_decision="AUTO",
            guardrail_confidence=0.95,
            risk_category="LOW",
        )
        prompt = build_explanation_prompt(req)
        assert "EXP-001" in prompt
        assert "CASE-001" in prompt
        assert "fee_difference" in prompt
        assert "₹1,000.00" in prompt  # 100000 paise
        assert "₹950.00" in prompt    # 95000 paise
        assert "₹50.00" in prompt     # 5000 paise
        assert "fee_reversal" in prompt
        assert "AUTO" in prompt
        assert "EV-001" in prompt

    def test_guardrail_reasons_in_prompt(self):
        from app.llm.services.explanation_service import ExplanationRequest, build_explanation_prompt

        req = ExplanationRequest(
            exception_id="EXP-001",
            guardrail_decision="HUMAN_REVIEW",
            guardrail_reasons=["High exposure", "Low confidence"],
        )
        prompt = build_explanation_prompt(req)
        assert "High exposure" in prompt
        assert "Low confidence" in prompt

    def test_missing_amounts(self):
        from app.llm.services.explanation_service import ExplanationRequest, build_explanation_prompt

        req = ExplanationRequest(exception_id="EXP-001")
        prompt = build_explanation_prompt(req)
        assert "Not provided" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicFallback:
    """Tests for _deterministic_fallback."""

    def test_basic_fallback(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
        )
        result = _deterministic_fallback(req)
        assert result.fallback_used is True
        assert result.model_used == "deterministic-template"
        assert result.provider == "none"
        assert "EXP-001" in result.summary
        assert "₹50.00" in result.summary

    def test_fully_explained(self):
        from app.llm.services.explanation_service import ExplanationEvidence, ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_coverage="FULLY_EXPLAINED",
            evidence_items=[ExplanationEvidence(evidence_id="EV-1", record_type="fee", description="fee")],
        )
        result = _deterministic_fallback(req)
        assert "explained" in result.reason.lower()

    def test_partially_explained(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=10000,
            evidence_coverage="PARTIALLY_EXPLAINED",
            explained_amount_paise=3000,
            remaining_difference_paise=7000,
        )
        result = _deterministic_fallback(req)
        assert "Part" in result.reason

    def test_unexplained(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_coverage="UNEXPLAINED",
        )
        result = _deterministic_fallback(req)
        assert "could not be explained" in result.reason.lower()

    def test_conflicting_evidence(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_coverage="CONFLICTING",
        )
        result = _deterministic_fallback(req)
        assert "conflicting" in result.reason.lower()

    def test_no_evidence(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(exception_id="EXP-001", difference_paise=5000)
        result = _deterministic_fallback(req)
        assert "No evidence" in result.uncertainty

    def test_multiple_evidence_types(self):
        from app.llm.services.explanation_service import ExplanationEvidence, ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_items=[
                ExplanationEvidence(evidence_id="E1", record_type="fee", description="fee"),
                ExplanationEvidence(evidence_id="E2", record_type="fee", description="fee2"),
                ExplanationEvidence(evidence_id="E3", record_type="refund", description="refund"),
            ],
        )
        result = _deterministic_fallback(req)
        assert "2 fee" in result.supporting_evidence
        assert "1 refund" in result.supporting_evidence

    def test_low_classification_confidence(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            classification_confidence=0.3,
        )
        result = _deterministic_fallback(req)
        assert "low" in result.uncertainty.lower()

    def test_llm_unavailable_in_limitations(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(exception_id="EXP-001")
        result = _deterministic_fallback(req)
        lim = result.limitations.lower()
        assert "without llm" in lim or "deterministic" in lim or "template" in lim

    def test_exception_type_in_summary(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            exception_type="settlement_mismatch",
            difference_paise=5000,
        )
        result = _deterministic_fallback(req)
        assert "settlement_mismatch" in result.summary


# ─────────────────────────────────────────────────────────────────────────────
# Response Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseParsing:
    """Tests for _parse_explanation_response."""

    def test_parse_valid_json(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        data = {
            "summary": "The exception involves a fee discrepancy.",
            "reason": "An incorrect platform fee was charged.",
            "supporting_evidence": "The fee record confirms the overcharge.",
            "uncertainty": "None",
            "limitations": "None",
        }
        result = _parse_explanation_response(json.dumps(data), provider="openai", model="gpt-4")
        assert result.summary == "The exception involves a fee discrepancy."
        assert result.reason == "An incorrect platform fee was charged."
        assert result.fallback_used is False
        assert result.provider == "openai"

    def test_parse_json_in_code_block(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        data = {"summary": "Test", "reason": "Because"}
        content = f"```json\n{json.dumps(data)}\n```"
        result = _parse_explanation_response(content)
        assert result.summary == "Test"
        assert result.reason == "Because"

    def test_parse_plain_text(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        result = _parse_explanation_response("This is a plain text explanation.")
        assert result.summary == "This is a plain text explanation."
        assert result.fallback_used is False

    def test_parse_empty_response(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        result = _parse_explanation_response("")
        assert "No explanation available" in result.summary

    def test_parse_partial_json(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        data = {"summary": "Partial response"}
        result = _parse_explanation_response(json.dumps(data))
        assert result.summary == "Partial response"
        assert result.reason == ""  # missing fields default to empty

    def test_parse_invalid_json(self):
        from app.llm.services.explanation_service import _parse_explanation_response

        result = _parse_explanation_response("Not JSON at all { broken")
        assert "Not JSON at all" in result.summary


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Tests (Mocked Provider)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMExplanationService:
    """Tests for LLMExplanationService with mocked provider."""

    def _make_service(self, response_content=None, side_effect=None):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.explanation_service import LLMExplanationService

        provider = AsyncMock()
        provider.provider_name = "openai"

        if side_effect:
            provider.generate = AsyncMock(side_effect=side_effect)
        else:
            provider.generate = AsyncMock(return_value=MagicMock(
                content=response_content or json.dumps({
                    "summary": "LLM generated explanation",
                    "reason": "Because the LLM said so",
                }),
                model="gpt-4",
                provider="openai",
                finish_reason="stop",
                usage={"total_tokens": 50},
                metadata={"elapsed_ms": 150.0},
            ))

        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0  # No retry in tests
        logger = LLMLogger("test")

        return LLMExplanationService(
            provider=provider,
            config=config,
            logger=logger,
        ), provider, logger

    def test_explain_with_llm(self):
        service, provider, logger = self._make_service()
        from app.llm.services.explanation_service import ExplanationRequest

        req = ExplanationRequest(
            exception_id="EXP-001",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
        )

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.summary == "LLM generated explanation"
        assert result.fallback_used is False
        assert result.provider == "openai"
        provider.generate.assert_awaited_once()

    def test_explain_without_llm_uses_fallback(self):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.explanation_service import ExplanationRequest, LLMExplanationService

        config = LLMConfig(enabled=False)
        logger = LLMLogger("test")
        service = LLMExplanationService(provider=None, config=config, logger=logger)

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
        )

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True
        assert "deterministic-template" in result.model_used

    def test_explain_timeout_uses_fallback(self):
        from app.llm.providers.base import LLMTimeoutError
        from app.llm.services.explanation_service import ExplanationRequest

        service, provider, _ = self._make_service(side_effect=LLMTimeoutError("timeout"))
        req = ExplanationRequest(exception_id="EXP-001", difference_paise=5000)

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True
        assert "LLM unavailable" in result.limitations or "timeout" in result.limitations.lower()

    def test_explain_connection_error_uses_fallback(self):
        from app.llm.providers.base import LLMConnectionError
        from app.llm.services.explanation_service import ExplanationRequest

        service, provider, _ = self._make_service(side_effect=LLMConnectionError("refused"))
        req = ExplanationRequest(exception_id="EXP-001", difference_paise=5000)

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_explain_provider_error_uses_fallback(self):
        from app.llm.providers.base import LLMProviderError
        from app.llm.services.explanation_service import ExplanationRequest

        service, provider, _ = self._make_service(
            side_effect=LLMProviderError("server error", details={"status_code": 500})
        )
        req = ExplanationRequest(exception_id="EXP-001", difference_paise=5000)

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_explain_unexpected_error_uses_fallback(self):
        from app.llm.services.explanation_service import ExplanationRequest

        service, provider, _ = self._make_service(side_effect=RuntimeError("boom"))
        req = ExplanationRequest(exception_id="EXP-001", difference_paise=5000)

        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_explain_logs_start_and_success(self):
        from app.llm.logging import LLMEventType
        from app.llm.services.explanation_service import ExplanationRequest

        service, _, logger = self._make_service()
        req = ExplanationRequest(exception_id="EXP-001", workflow_id="WF-001")

        asyncio.get_event_loop().run_until_complete(service.explain(req))
        starts = logger.get_entries(event_type=LLMEventType.REQUEST_START)
        successes = logger.get_entries(event_type=LLMEventType.REQUEST_SUCCESS)
        assert len(starts) == 1
        assert len(successes) == 1
        assert starts[0].exception_id == "EXP-001"

    def test_explain_logs_error(self):
        from app.llm.logging import LLMEventType
        from app.llm.providers.base import LLMTimeoutError
        from app.llm.services.explanation_service import ExplanationRequest

        service, _, logger = self._make_service(side_effect=LLMTimeoutError("timeout"))
        req = ExplanationRequest(exception_id="EXP-001")

        asyncio.get_event_loop().run_until_complete(service.explain(req))
        errors = logger.get_entries(event_type=LLMEventType.REQUEST_ERROR)
        assert len(errors) == 1

    def test_health_check_with_provider(self):
        service, _, _ = self._make_service()

        async def run():
            return await service.health_check()

        status = asyncio.get_event_loop().run_until_complete(run())
        # health_check delegates to retry executor → provider.health_check
        # which returns an AsyncMock; assert it was called
        assert status is not None

    def test_health_check_without_provider(self):
        from app.llm.config import LLMConfig
        from app.llm.services.explanation_service import LLMExplanationService

        service = LLMExplanationService(provider=None, config=LLMConfig(enabled=False))

        async def run():
            return await service.health_check()

        status = asyncio.get_event_loop().run_until_complete(run())
        assert status.healthy is True
        assert status.provider == "none"


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemPromptSafety:
    """Verify the system prompt constrains LLM behavior."""

    def test_prompt_prohibits_calculation(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "MUST NOT" in EXPLANATION_SYSTEM_PROMPT
        assert "calculate" in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_prompt_prohibits_decisions(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "financial decisions" in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_prompt_prohibits_amount_changes(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "change" in EXPLANATION_SYSTEM_PROMPT.lower()
        assert "amounts" in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_prompt_prohibits_override(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "override" in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_prompt_requires_evidence_based(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "supplied evidence" in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_prompt_requires_uncertainty_disclosure(self):
        from app.llm.services.explanation_service import EXPLANATION_SYSTEM_PROMPT

        assert "uncertain" in EXPLANATION_SYSTEM_PROMPT.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    """Verify LLM explanation does NOT interfere with financial systems."""

    def test_explanation_service_has_no_execution_methods(self):
        from app.llm.services.explanation_service import LLMExplanationService

        forbidden = [
            "execute_resolution",
            "issue_refund",
            "modify_settlement",
            "update_database",
            "call_razorpay_api",
            "bypass_guardrails",
        ]
        for method in forbidden:
            assert not hasattr(LLMExplanationService, method)

    def test_output_has_no_financial_fields(self):
        from app.llm.services.explanation_service import LLMExplanationOutput

        output = LLMExplanationOutput()
        forbidden = ["authorize", "approve", "execute", "refund_amount", "settlement_amount"]
        for field in forbidden:
            assert not hasattr(output, field)

    def test_request_is_read_only_data(self):
        """ExplanationRequest only contains data to explain, not instructions."""
        from app.llm.services.explanation_service import ExplanationRequest

        req = ExplanationRequest(exception_id="EXP-001")
        forbidden = ["execute", "approve", "override", "bypass", "force_auto"]
        for field in forbidden:
            assert not hasattr(req, field)

    def test_deterministic_fallback_produces_valid_output(self):
        from app.llm.services.explanation_service import ExplanationRequest, _deterministic_fallback

        req = ExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            expected_amount_paise=100000,
            actual_amount_paise=95000,
        )
        result = _deterministic_fallback(req)
        assert result.summary != ""
        assert result.limitations != ""
        assert result.fallback_used is True
