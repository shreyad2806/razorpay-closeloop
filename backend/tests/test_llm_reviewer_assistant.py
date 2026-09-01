"""
Tests for Razorpay CloseLoop Phase 12F — LLM Reviewer Assistant Service.

Covers:
- Schemas (request, output, candidates, guardrails)
- Prompt building
- Deterministic fallback (clear, ambiguous, high-risk, conflicting, missing, guardrail rejection)
- LLM response parsing
- LLM-generated briefing (mocked)
- Failure handling
- Safety boundary (LLM provides explanation, reviewer decides)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_request_minimal(self):
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        req = ReviewerBriefingRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.candidates == []

    def test_request_full(self):
        from app.llm.services.reviewer_assistant_service import (
            CandidateInfo,
            GuardrailInfo,
            ReviewerBriefingRequest,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            evidence_record_count=3,
            evidence_coverage="FULLY_EXPLAINED",
            classification_confidence=0.92,
            candidates=[
                CandidateInfo(
                    resolution_type="fee_reversal",
                    source="ML_PREDICTION",
                    confidence=0.95,
                    adjustment_paise=5000,
                    description="Reverse the fee",
                ),
            ],
            guardrail=GuardrailInfo(
                decision="HUMAN_REVIEW",
                confidence=0.7,
                risk_category="MEDIUM",
                reasons=["Medium confidence"],
                exposure_paise=5000,
            ),
        )
        assert len(req.candidates) == 1
        assert req.guardrail.decision == "HUMAN_REVIEW"

    def test_output_schema(self):
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingOutput

        out = ReviewerBriefingOutput(
            what_happened="Test",
            reviewer_checklist="Check 1\nCheck 2",
        )
        d = out.to_dict()
        assert d["what_happened"] == "Test"

    def test_candidate_info(self):
        from app.llm.services.reviewer_assistant_service import CandidateInfo

        c = CandidateInfo(resolution_type="fee_reversal", source="ML", confidence=0.9)
        assert c.resolution_type == "fee_reversal"
        assert c.evidence_compatible is True

    def test_guardrail_info(self):
        from app.llm.services.reviewer_assistant_service import GuardrailInfo

        g = GuardrailInfo(
            decision="AUTO", confidence=0.95,
            risk_category="LOW", reasons=["Clear evidence"],
        )
        assert g.decision == "AUTO"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Building Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptBuilding:
    def test_minimal_prompt(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            build_reviewer_briefing_prompt,
        )

        req = ReviewerBriefingRequest(exception_id="EXP-001")
        prompt = build_reviewer_briefing_prompt(req)
        assert "EXP-001" in prompt
        assert "Human Review Briefing Request" in prompt

    def test_full_prompt(self):
        from app.llm.services.reviewer_assistant_service import (
            CandidateInfo,
            GuardrailInfo,
            ReviewerBriefingRequest,
            build_reviewer_briefing_prompt,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            evidence_record_count=3,
            evidence_coverage="FULLY_EXPLAINED",
            candidates=[
                CandidateInfo(
                    resolution_type="fee_reversal",
                    source="ML_PREDICTION",
                    confidence=0.95,
                    adjustment_paise=5000,
                    description="Reverse fee",
                ),
            ],
            guardrail=GuardrailInfo(
                decision="HUMAN_REVIEW",
                confidence=0.7,
                risk_category="MEDIUM",
                reasons=["Medium confidence"],
            ),
        )
        prompt = build_reviewer_briefing_prompt(req)
        assert "EXP-001" in prompt
        assert "fee_difference" in prompt
        assert "₹1,000.00" in prompt
        assert "fee_reversal" in prompt
        assert "HUMAN_REVIEW" in prompt
        assert "Medium confidence" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicFallback:
    def test_clear_case(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            evidence_coverage="FULLY_EXPLAINED",
            evidence_record_count=3,
            exception_type="fee_difference",
        )
        result = _reviewer_deterministic_fallback(req)
        assert "EXP-001" in result.what_happened
        # The fallback says 'All of the discrepancy is accounted for'
        assert "all" in result.why_it_happened.lower() or "fully" in result.why_it_happened.lower()
        assert "3 evidence" in result.supporting_evidence
        assert result.fallback_used is True

    def test_ambiguous_case(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_coverage="PARTIALLY_EXPLAINED",
            explained_amount_paise=2000,
            remaining_difference_paise=3000,
        )
        result = _reviewer_deterministic_fallback(req)
        assert "Part" in result.why_it_happened
        assert "₹30.00" in result.why_it_happened

    def test_high_risk_case(self):
        from app.llm.services.reviewer_assistant_service import (
            GuardrailInfo,
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=50000,
            guardrail=GuardrailInfo(
                decision="HUMAN_REVIEW",
                confidence=0.4,
                risk_category="HIGH",
                reasons=["High exposure", "Low confidence"],
            ),
        )
        result = _reviewer_deterministic_fallback(req)
        assert "HUMAN_REVIEW" in result.system_recommendation
        assert "high exposure" in result.system_recommendation.lower()

    def test_conflicting_evidence(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            conflicts=["Settlement amounts disagree", "Fee records conflict"],
        )
        result = _reviewer_deterministic_fallback(req)
        assert "Settlement amounts disagree" in result.conflicts
        assert "Fee records conflict" in result.conflicts
        assert "conflicts" in result.automation_barriers.lower()

    def test_missing_evidence(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            missing_evidence=["SETTLEMENT record missing", "FEE record not found"],
        )
        result = _reviewer_deterministic_fallback(req)
        assert "SETTLEMENT record missing" in result.missing_evidence
        assert "FEE record not found" in result.missing_evidence

    def test_guardrail_rejection(self):
        from app.llm.services.reviewer_assistant_service import (
            GuardrailInfo,
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            guardrail=GuardrailInfo(
                decision="UNRESOLVED",
                confidence=0.3,
                risk_category="HIGH",
                reasons=["Cannot determine resolution"],
            ),
        )
        result = _reviewer_deterministic_fallback(req)
        assert "UNRESOLVED" in result.system_recommendation
        assert "UNRESOLVED" in result.automation_barriers

    def test_no_evidence(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(exception_id="EXP-001", difference_paise=5000)
        result = _reviewer_deterministic_fallback(req)
        assert "No evidence records" in result.supporting_evidence
        assert "No specific automation barriers" in result.automation_barriers

    def test_classification_disagreement(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            classification_agreement=False,
            classification_note="Deterministic: fee_difference, ML: settlement_mismatch",
        )
        result = _reviewer_deterministic_fallback(req)
        assert "disagree" in result.automation_barriers.lower()

    def test_checklist_adapts_to_context(self):
        from app.llm.services.reviewer_assistant_service import (
            GuardrailInfo,
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_coverage="PARTIALLY_EXPLAINED",
            conflicts=["conflict"],
            missing_evidence=["missing"],
            classification_agreement=False,
            guardrail=GuardrailInfo(decision="HUMAN_REVIEW"),
        )
        result = _reviewer_deterministic_fallback(req)
        checklist = result.reviewer_checklist.lower()
        assert "unexplained" in checklist
        assert "conflict" in checklist
        assert "missing" in checklist
        assert "classification" in checklist
        assert "guardrail" in checklist

    def test_candidates_listed(self):
        from app.llm.services.reviewer_assistant_service import (
            CandidateInfo,
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(
            exception_id="EXP-001",
            candidates=[
                CandidateInfo(resolution_type="fee_reversal", source="ML", confidence=0.95, adjustment_paise=5000),
                CandidateInfo(resolution_type="escalation", source="DETERMINISTIC", confidence=0.6),
            ],
        )
        result = _reviewer_deterministic_fallback(req)
        assert "fee_reversal" in result.candidate_resolutions
        assert "escalation" in result.candidate_resolutions


# ─────────────────────────────────────────────────────────────────────────────
# Response Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseParsing:
    def test_parse_valid_json(self):
        from app.llm.services.reviewer_assistant_service import _parse_briefing_response

        data = {
            "what_happened": "Fee discrepancy",
            "why_it_happened": "Incorrect fee charged",
            "supporting_evidence": "Fee record shows overcharge",
            "missing_evidence": "None",
            "conflicts": "None",
            "candidate_resolutions": "fee_reversal",
            "why_candidates": "Evidence supports reversal",
            "system_recommendation": "AUTO — fee_reversal",
            "automation_barriers": "None",
            "reviewer_checklist": "Verify fee amount",
        }
        result = _parse_briefing_response(json.dumps(data))
        assert result.what_happened == "Fee discrepancy"
        assert result.reviewer_checklist == "Verify fee amount"
        assert result.fallback_used is False

    def test_parse_plain_text(self):
        from app.llm.services.reviewer_assistant_service import _parse_briefing_response

        result = _parse_briefing_response("Plain text briefing.")
        assert result.what_happened == "Plain text briefing."

    def test_parse_empty(self):
        from app.llm.services.reviewer_assistant_service import _parse_briefing_response

        result = _parse_briefing_response("")
        assert "No briefing available" in result.what_happened


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMReviewerAssistantService:
    def _make_service(self, response_content=None, side_effect=None):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.reviewer_assistant_service import LLMReviewerAssistantService

        provider = AsyncMock()
        provider.provider_name = "openai"

        if side_effect:
            provider.generate = AsyncMock(side_effect=side_effect)
        else:
            provider.generate = AsyncMock(return_value=MagicMock(
                content=response_content or json.dumps({
                    "what_happened": "LLM briefing",
                    "reviewer_checklist": "LLM checklist",
                }),
                model="gpt-4", provider="openai",
                finish_reason="stop", usage={"total_tokens": 80},
                metadata={"elapsed_ms": 250.0},
            ))

        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0
        logger = LLMLogger("test")

        return LLMReviewerAssistantService(
            provider=provider, config=config, logger=logger,
        ), provider, logger

    def test_generate_with_llm(self):
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        service, provider, _ = self._make_service()
        req = ReviewerBriefingRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.generate_briefing(req))
        assert result.what_happened == "LLM briefing"
        assert result.fallback_used is False
        provider.generate.assert_awaited_once()

    def test_generate_without_llm(self):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            LLMReviewerAssistantService,
        )

        service = LLMReviewerAssistantService(
            provider=None, config=LLMConfig(enabled=False), logger=LLMLogger("test"),
        )
        req = ReviewerBriefingRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.generate_briefing(req))
        assert result.fallback_used is True

    def test_timeout_fallback(self):
        from app.llm.providers.base import LLMTimeoutError
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        service, _, _ = self._make_service(side_effect=LLMTimeoutError("timeout"))
        req = ReviewerBriefingRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.generate_briefing(req))
        assert result.fallback_used is True
        assert "LLM unavailable" in result.reviewer_checklist

    def test_connection_error_fallback(self):
        from app.llm.providers.base import LLMConnectionError
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        service, _, _ = self._make_service(side_effect=LLMConnectionError("refused"))
        req = ReviewerBriefingRequest(exception_id="EXP-001")
        result = asyncio.get_event_loop().run_until_complete(service.generate_briefing(req))
        assert result.fallback_used is True

    def test_logs_start_and_success(self):
        from app.llm.logging import LLMEventType
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        service, _, logger = self._make_service()
        req = ReviewerBriefingRequest(exception_id="EXP-001", workflow_id="WF-001")
        asyncio.get_event_loop().run_until_complete(service.generate_briefing(req))
        starts = logger.get_entries(event_type=LLMEventType.REQUEST_START)
        successes = logger.get_entries(event_type=LLMEventType.REQUEST_SUCCESS)
        assert len(starts) == 1
        assert len(successes) == 1

    def test_health_check_no_provider(self):
        from app.llm.config import LLMConfig
        from app.llm.services.reviewer_assistant_service import LLMReviewerAssistantService

        service = LLMReviewerAssistantService(
            provider=None, config=LLMConfig(enabled=False),
        )

        async def run():
            return await service.health_check()

        status = asyncio.get_event_loop().run_until_complete(run())
        assert status.healthy is True
        assert status.provider == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_system_prompt_prohibits_decision_making(self):
        from app.llm.services.reviewer_assistant_service import REVIEWER_ASSISTANT_PROMPT

        assert "MUST NOT" in REVIEWER_ASSISTANT_PROMPT
        assert "decision" in REVIEWER_ASSISTANT_PROMPT.lower()

    def test_system_prompt_prohibits_execution(self):
        from app.llm.services.reviewer_assistant_service import REVIEWER_ASSISTANT_PROMPT

        assert "execute" in REVIEWER_ASSISTANT_PROMPT.lower()
        assert "financial action" in REVIEWER_ASSISTANT_PROMPT.lower()

    def test_system_prompt_prohibits_overriding_guardrails(self):
        from app.llm.services.reviewer_assistant_service import REVIEWER_ASSISTANT_PROMPT

        assert "override" in REVIEWER_ASSISTANT_PROMPT.lower()
        assert "guardrail" in REVIEWER_ASSISTANT_PROMPT.lower()

    def test_system_prompt_prohibits_replacing_reviewer(self):
        from app.llm.services.reviewer_assistant_service import REVIEWER_ASSISTANT_PROMPT

        assert "replace" in REVIEWER_ASSISTANT_PROMPT.lower()
        assert "reviewer" in REVIEWER_ASSISTANT_PROMPT.lower()

    def test_system_prompt_requires_using_provided_info(self):
        from app.llm.services.reviewer_assistant_service import REVIEWER_ASSISTANT_PROMPT

        assert "ONLY" in REVIEWER_ASSISTANT_PROMPT
        assert "provided" in REVIEWER_ASSISTANT_PROMPT.lower()

    def test_service_has_no_financial_methods(self):
        from app.llm.services.reviewer_assistant_service import LLMReviewerAssistantService

        forbidden = [
            "execute_resolution", "issue_refund", "modify_settlement",
            "make_decision", "approve_resolution", "override_guardrails",
        ]
        for method in forbidden:
            assert not hasattr(LLMReviewerAssistantService, method)

    def test_output_has_no_financial_authorization(self):
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingOutput

        out = ReviewerBriefingOutput()
        forbidden = ["authorize", "approve", "execute", "force_resolution"]
        for field in forbidden:
            assert not hasattr(out, field)

    def test_request_has_no_decision_fields(self):
        from app.llm.services.reviewer_assistant_service import ReviewerBriefingRequest

        dangerous_fields = ["approve", "reject", "override", "force_auto"]
        for field in dangerous_fields:
            assert field not in ReviewerBriefingRequest.model_fields

    def test_fallback_produces_valid_briefing(self):
        from app.llm.services.reviewer_assistant_service import (
            ReviewerBriefingRequest,
            _reviewer_deterministic_fallback,
        )

        req = ReviewerBriefingRequest(exception_id="EXP-001", difference_paise=5000)
        result = _reviewer_deterministic_fallback(req)
        # All 10 points should be non-empty
        assert result.what_happened != ""
        assert result.why_it_happened != ""
        assert result.supporting_evidence != ""
        assert result.missing_evidence != ""
        assert result.conflicts != ""
        assert result.candidate_resolutions != ""
        assert result.why_candidates != ""
        assert result.system_recommendation != ""
        assert result.automation_barriers != ""
        assert result.reviewer_checklist != ""
        assert result.fallback_used is True
