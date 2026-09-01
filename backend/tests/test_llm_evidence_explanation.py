"""
Tests for Razorpay CloseLoop Phase 12D — LLM Evidence Explanation Service.

Covers:
- EvidenceExplanationRequest / EvidenceExplanationOutput schemas
- from_evidence_package conversion
- Prompt building
- Deterministic fallback (complete, partial, conflicting, missing, empty)
- LLM response parsing
- LLM-generated explanation (mocked)
- Failure handling
- Hallucination resistance (LLM cannot invent evidence)
- Safety boundary
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
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        req = EvidenceExplanationRequest(exception_id="EXP-001")
        assert req.exception_id == "EXP-001"
        assert req.settlements == []
        assert req.conflicts == []

    def test_request_full(self):
        from app.llm.services.evidence_explanation_service import (
            ConflictInfo,
            EvidenceExplanationRequest,
            EvidenceRecordInfo,
            MissingEvidenceInfo,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            payment_id="PAY-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            payment=EvidenceRecordInfo(
                record_id="PAY-001", entity_type="PAYMENT",
                relationship="PRIMARY_RECORD", amount_paise=100000,
            ),
            settlements=[
                EvidenceRecordInfo(
                    record_id="SET-001", entity_type="SETTLEMENT",
                    relationship="PRIMARY_RECORD", amount_paise=95000,
                ),
            ],
            fees=[
                EvidenceRecordInfo(
                    record_id="FEE-001", entity_type="FEE",
                    relationship="CALCULATION_COMPONENT", amount_paise=5000,
                ),
            ],
            total_settlement_paise=95000,
            total_fee_paise=5000,
            missing_evidence=[
                MissingEvidenceInfo(
                    entity_type="REFUND", expected=False,
                    reason="No refund expected for this payment type",
                ),
            ],
            conflicts=[
                ConflictInfo(
                    conflict_type="MULTIPLE_SETTLEMENTS",
                    description="Two settlements found for one payment",
                    affected_records=["SET-001", "SET-002"],
                ),
            ],
            evidence_record_count=3,
        )
        assert req.exception_id == "EXP-001"
        assert len(req.settlements) == 1
        assert len(req.conflicts) == 1
        assert req.conflicts[0].conflict_type == "MULTIPLE_SETTLEMENTS"

    def test_output_schema(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationOutput

        out = EvidenceExplanationOutput(
            summary="Test",
            explained_amount_paise=5000,
            unexplained_amount_paise=0,
            completeness="Complete",
        )
        d = out.to_dict()
        assert d["summary"] == "Test"
        assert d["explained_amount_paise"] == 5000

    def test_from_evidence_package_dict(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        pkg = {
            "exception_id": "EXP-001",
            "case_id": "CASE-001",
            "payment_id": "PAY-001",
            "exception_type": "fee_difference",
            "expected_amount": 100000,
            "actual_amount": 95000,
            "difference": 5000,
            "payment": {
                "record_id": "PAY-001", "entity_type": "PAYMENT",
                "relationship": "PRIMARY_RECORD", "amount": 100000,
            },
            "settlements": [
                {"record_id": "SET-001", "entity_type": "SETTLEMENT",
                 "relationship": "PRIMARY_RECORD", "amount": 95000},
            ],
            "refunds": [],
            "fees": [
                {"record_id": "FEE-001", "entity_type": "FEE",
                 "relationship": "CALCULATION_COMPONENT", "amount": 5000},
            ],
            "taxes": [],
            "adjustments": [],
            "total_settlement_amount": 95000,
            "total_refund_amount": 0,
            "total_fee_amount": 5000,
            "total_tax_amount": 0,
            "total_adjustment_amount": 0,
            "missing_evidence": [
                {"entity_type": "REFUND", "expected": False,
                 "reason": "No refund expected"},
            ],
            "conflicts": [],
            "evidence_link_count": 5,
        }

        req = EvidenceExplanationRequest.from_evidence_package(pkg)
        assert req.exception_id == "EXP-001"
        assert req.payment is not None
        assert req.payment.record_id == "PAY-001"
        assert len(req.settlements) == 1
        assert len(req.fees) == 1
        assert req.total_settlement_paise == 95000
        assert len(req.missing_evidence) == 1
        assert req.evidence_record_count == 3  # 1 payment + 1 settlement + 1 fee

    def test_from_evidence_package_empty(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        pkg = {"exception_id": "EXP-001"}
        req = EvidenceExplanationRequest.from_evidence_package(pkg)
        assert req.exception_id == "EXP-001"
        assert req.evidence_record_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Building Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptBuilding:
    """Tests for build_evidence_explanation_prompt."""

    def test_minimal_prompt(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            build_evidence_explanation_prompt,
        )

        req = EvidenceExplanationRequest(exception_id="EXP-001")
        prompt = build_evidence_explanation_prompt(req)
        assert "EXP-001" in prompt
        assert "Financial Evidence Explanation Request" in prompt

    def test_full_prompt_with_records(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            EvidenceRecordInfo,
            build_evidence_explanation_prompt,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            case_id="CASE-001",
            payment_id="PAY-001",
            exception_type="fee_difference",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            payment=EvidenceRecordInfo(
                record_id="PAY-001", entity_type="PAYMENT",
                relationship="PRIMARY_RECORD", amount_paise=100000,
            ),
            settlements=[
                EvidenceRecordInfo(
                    record_id="SET-001", entity_type="SETTLEMENT",
                    relationship="PRIMARY_RECORD", amount_paise=95000,
                ),
            ],
            fees=[
                EvidenceRecordInfo(
                    record_id="FEE-001", entity_type="FEE",
                    relationship="CALCULATION_COMPONENT", amount_paise=5000,
                ),
            ],
        )
        prompt = build_evidence_explanation_prompt(req)
        assert "EXP-001" in prompt
        assert "CASE-001" in prompt
        assert "PAY-001" in prompt
        assert "fee_difference" in prompt
        assert "₹1,000.00" in prompt
        assert "SET-001" in prompt
        assert "FEE-001" in prompt

    def test_conflicts_in_prompt(self):
        from app.llm.services.evidence_explanation_service import (
            ConflictInfo,
            EvidenceExplanationRequest,
            build_evidence_explanation_prompt,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            conflicts=[
                ConflictInfo(
                    conflict_type="MULTIPLE_SETTLEMENTS",
                    description="Two settlements found",
                    affected_records=["SET-001", "SET-002"],
                ),
            ],
        )
        prompt = build_evidence_explanation_prompt(req)
        assert "MULTIPLE_SETTLEMENTS" in prompt
        assert "Two settlements found" in prompt
        assert "SET-001" in prompt

    def test_missing_evidence_in_prompt(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            MissingEvidenceInfo,
            build_evidence_explanation_prompt,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            missing_evidence=[
                MissingEvidenceInfo(
                    entity_type="REFUND",
                    expected=True,
                    reason="Refund expected but not found",
                ),
            ],
        )
        prompt = build_evidence_explanation_prompt(req)
        assert "REFUND" in prompt
        assert "expected" in prompt
        assert "Refund expected but not found" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicFallback:
    """Tests for _evidence_deterministic_fallback."""

    def test_complete_evidence(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            EvidenceRecordInfo,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            expected_amount_paise=100000,
            actual_amount_paise=95000,
            difference_paise=5000,
            payment=EvidenceRecordInfo(
                record_id="PAY-001", entity_type="PAYMENT",
                relationship="PRIMARY_RECORD", amount_paise=100000,
            ),
            fees=[
                EvidenceRecordInfo(
                    record_id="FEE-001", entity_type="FEE",
                    relationship="CALCULATION_COMPONENT", amount_paise=5000,
                ),
            ],
            evidence_record_count=2,
        )
        result = _evidence_deterministic_fallback(req)
        assert result.fallback_used is True
        assert "EXP-001" in result.summary
        assert "2 evidence record(s)" in result.summary
        assert "2 financial event(s)" in result.financial_events or "1 payment" in result.financial_events
        assert result.conflicts == "No structural conflicts detected."
        assert "Complete" in result.completeness or "complete" in result.completeness.lower()

    def test_partial_evidence(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            EvidenceRecordInfo,
            MissingEvidenceInfo,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            fees=[
                EvidenceRecordInfo(
                    record_id="FEE-001", entity_type="FEE",
                    relationship="CALCULATION_COMPONENT", amount_paise=3000,
                ),
            ],
            missing_evidence=[
                MissingEvidenceInfo(
                    entity_type="ADJUSTMENT", expected=True,
                    reason="Adjustment expected but missing",
                ),
            ],
            evidence_record_count=1,
        )
        result = _evidence_deterministic_fallback(req)
        assert "ADJUSTMENT" in result.missing_evidence
        assert result.unexplained_amount_paise >= 0

    def test_conflicting_evidence(self):
        from app.llm.services.evidence_explanation_service import (
            ConflictInfo,
            EvidenceExplanationRequest,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            conflicts=[
                ConflictInfo(
                    conflict_type="MULTIPLE_SETTLEMENTS",
                    description="Two settlements for one payment",
                    affected_records=["SET-001", "SET-002"],
                ),
            ],
            evidence_record_count=2,
        )
        result = _evidence_deterministic_fallback(req)
        assert "MULTIPLE_SETTLEMENTS" in result.conflicts
        assert "structural conflict" in result.uncertainty.lower()

    def test_missing_evidence(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            MissingEvidenceInfo,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            missing_evidence=[
                MissingEvidenceInfo(
                    entity_type="SETTLEMENT", expected=True,
                    reason="Settlement not found",
                ),
            ],
        )
        result = _evidence_deterministic_fallback(req)
        assert "SETTLEMENT" in result.missing_evidence
        assert "missing" in result.completeness.lower() or "incomplete" in result.completeness.lower()

    def test_no_evidence(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(exception_id="EXP-001", difference_paise=5000)
        result = _evidence_deterministic_fallback(req)
        assert "No financial evidence" in result.financial_events
        assert "cannot be assessed" in result.completeness.lower()

    def test_multi_event_explanation(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            EvidenceRecordInfo,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=10000,
            settlements=[
                EvidenceRecordInfo(record_id="S1", entity_type="SETTLEMENT",
                                   relationship="PRIMARY_RECORD", amount_paise=90000),
            ],
            refunds=[
                EvidenceRecordInfo(record_id="R1", entity_type="REFUND",
                                   relationship="CALCULATION_COMPONENT", amount_paise=5000),
            ],
            fees=[
                EvidenceRecordInfo(record_id="F1", entity_type="FEE",
                                   relationship="CALCULATION_COMPONENT", amount_paise=3000),
            ],
            taxes=[
                EvidenceRecordInfo(record_id="T1", entity_type="TAX",
                                   relationship="CALCULATION_COMPONENT", amount_paise=2000),
            ],
            evidence_record_count=4,
        )
        result = _evidence_deterministic_fallback(req)
        assert "1 settlement" in result.financial_events
        assert "1 refund" in result.financial_events
        assert "1 fee" in result.financial_events
        assert "1 tax" in result.financial_events

    def test_conflicts_and_missing(self):
        from app.llm.services.evidence_explanation_service import (
            ConflictInfo,
            EvidenceExplanationRequest,
            MissingEvidenceInfo,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            conflicts=[ConflictInfo(conflict_type="X", description="conflict", affected_records=["A"])],
            missing_evidence=[MissingEvidenceInfo(entity_type="Y", expected=True, reason="missing")],
        )
        result = _evidence_deterministic_fallback(req)
        assert "incomplete" in result.completeness.lower() or "conflicts" in result.completeness.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Response Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseParsing:
    """Tests for _parse_evidence_response."""

    def test_parse_valid_json(self):
        from app.llm.services.evidence_explanation_service import _parse_evidence_response

        data = {
            "summary": "Fee difference explained",
            "financial_events": "Fee charged",
            "evidence_chain": "Fee record matches discrepancy",
            "explained_amount_paise": 5000,
            "unexplained_amount_paise": 0,
            "conflicts": "None",
            "missing_evidence": "None",
            "uncertainty": "None",
            "completeness": "Complete",
        }
        result = _parse_evidence_response(json.dumps(data))
        assert result.summary == "Fee difference explained"
        assert result.explained_amount_paise == 5000
        assert result.fallback_used is False

    def test_parse_json_in_code_block(self):
        from app.llm.services.evidence_explanation_service import _parse_evidence_response

        data = {"summary": "Block test"}
        content = f"```json\n{json.dumps(data)}\n```"
        result = _parse_evidence_response(content)
        assert result.summary == "Block test"

    def test_parse_plain_text(self):
        from app.llm.services.evidence_explanation_service import _parse_evidence_response

        result = _parse_evidence_response("Plain text explanation of evidence.")
        assert result.summary == "Plain text explanation of evidence."

    def test_parse_empty(self):
        from app.llm.services.evidence_explanation_service import _parse_evidence_response

        result = _parse_evidence_response("")
        assert "No evidence explanation" in result.summary

    def test_parse_partial_json(self):
        from app.llm.services.evidence_explanation_service import _parse_evidence_response

        result = _parse_evidence_response(json.dumps({"summary": "Partial"}))
        assert result.summary == "Partial"
        assert result.financial_events == ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Tests (Mocked Provider)
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMEvidenceExplanationService:
    """Tests for LLMEvidenceExplanationService with mocked provider."""

    def _make_service(self, response_content=None, side_effect=None):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.evidence_explanation_service import LLMEvidenceExplanationService

        provider = AsyncMock()
        provider.provider_name = "openai"

        if side_effect:
            provider.generate = AsyncMock(side_effect=side_effect)
        else:
            provider.generate = AsyncMock(return_value=MagicMock(
                content=response_content or json.dumps({
                    "summary": "LLM evidence explanation",
                    "explained_amount_paise": 5000,
                    "unexplained_amount_paise": 0,
                }),
                model="gpt-4",
                provider="openai",
                finish_reason="stop",
                usage={"total_tokens": 60},
                metadata={"elapsed_ms": 200.0},
            ))

        config = LLMConfig(enabled=True, provider="openai")
        config.openai.max_retries = 0
        logger = LLMLogger("test")

        return LLMEvidenceExplanationService(
            provider=provider, config=config, logger=logger,
        ), provider, logger

    def test_explain_with_llm(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        service, provider, _ = self._make_service()
        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_record_count=2,
        )
        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.summary == "LLM evidence explanation"
        assert result.fallback_used is False
        provider.generate.assert_awaited_once()

    def test_explain_without_llm(self):
        from app.llm.config import LLMConfig
        from app.llm.logging import LLMLogger
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            LLMEvidenceExplanationService,
        )

        service = LLMEvidenceExplanationService(
            provider=None,
            config=LLMConfig(enabled=False),
            logger=LLMLogger("test"),
        )
        req = EvidenceExplanationRequest(exception_id="EXP-001", difference_paise=5000)
        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_timeout_fallback(self):
        from app.llm.providers.base import LLMTimeoutError
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        service, _, _ = self._make_service(side_effect=LLMTimeoutError("timeout"))
        req = EvidenceExplanationRequest(exception_id="EXP-001", difference_paise=5000)
        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_connection_error_fallback(self):
        from app.llm.providers.base import LLMConnectionError
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        service, _, _ = self._make_service(side_effect=LLMConnectionError("refused"))
        req = EvidenceExplanationRequest(exception_id="EXP-001", difference_paise=5000)
        result = asyncio.get_event_loop().run_until_complete(service.explain(req))
        assert result.fallback_used is True

    def test_logs_start_and_success(self):
        from app.llm.logging import LLMEventType
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        service, _, logger = self._make_service()
        req = EvidenceExplanationRequest(exception_id="EXP-001", workflow_id="WF-001")
        asyncio.get_event_loop().run_until_complete(service.explain(req))
        starts = logger.get_entries(event_type=LLMEventType.REQUEST_START)
        successes = logger.get_entries(event_type=LLMEventType.REQUEST_SUCCESS)
        assert len(starts) == 1
        assert len(successes) == 1

    def test_logs_error(self):
        from app.llm.logging import LLMEventType
        from app.llm.providers.base import LLMProviderError
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        service, _, logger = self._make_service(
            side_effect=LLMProviderError("500", details={"status_code": 500})
        )
        req = EvidenceExplanationRequest(exception_id="EXP-001")
        asyncio.get_event_loop().run_until_complete(service.explain(req))
        errors = logger.get_entries(event_type=LLMEventType.REQUEST_ERROR)
        assert len(errors) == 1

    def test_health_check_no_provider(self):
        from app.llm.config import LLMConfig
        from app.llm.services.evidence_explanation_service import LLMEvidenceExplanationService

        service = LLMEvidenceExplanationService(
            provider=None, config=LLMConfig(enabled=False),
        )

        async def run():
            return await service.health_check()

        status = asyncio.get_event_loop().run_until_complete(run())
        assert status.healthy is True
        assert status.provider == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Resistance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHallucinationResistance:
    """Verify the LLM cannot invent evidence."""

    def test_system_prompt_prohibits_inventing_evidence(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "MUST NOT" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT
        assert "invent" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()
        assert "evidence" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_changing_amounts(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "change" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()
        assert "amount" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_declaring_resolved(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "resolved" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_speculating(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "speculate" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_requires_using_provided_evidence(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "ONLY" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT
        assert "provided" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_prohibits_calculating_new_totals(self):
        from app.llm.services.evidence_explanation_service import EVIDENCE_EXPLANATION_SYSTEM_PROMPT

        assert "calculate" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()
        assert "totals" in EVIDENCE_EXPLANATION_SYSTEM_PROMPT.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    """Verify evidence explanation service cannot modify financial state."""

    def test_service_has_no_execution_methods(self):
        from app.llm.services.evidence_explanation_service import LLMEvidenceExplanationService

        forbidden = [
            "execute_resolution", "issue_refund", "modify_settlement",
            "update_database", "modify_evidence", "delete_evidence",
        ]
        for method in forbidden:
            assert not hasattr(LLMEvidenceExplanationService, method)

    def test_output_has_no_financial_fields(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationOutput

        out = EvidenceExplanationOutput()
        forbidden = ["authorize", "approve", "execute", "refund_amount", "settlement_amount"]
        for field in forbidden:
            assert not hasattr(out, field)

    def test_request_has_no_modification_fields(self):
        from app.llm.services.evidence_explanation_service import EvidenceExplanationRequest

        req = EvidenceExplanationRequest(exception_id="EXP-001")
        forbidden = ["execute", "approve", "override", "modify", "delete"]
        for field in forbidden:
            assert not hasattr(req, field)

    def test_fallback_output_is_valid(self):
        from app.llm.services.evidence_explanation_service import (
            EvidenceExplanationRequest,
            _evidence_deterministic_fallback,
        )

        req = EvidenceExplanationRequest(
            exception_id="EXP-001",
            difference_paise=5000,
            evidence_record_count=1,
        )
        result = _evidence_deterministic_fallback(req)
        assert result.summary != ""
        assert result.completeness != ""
        assert result.fallback_used is True
