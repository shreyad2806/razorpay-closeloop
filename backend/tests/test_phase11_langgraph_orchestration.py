"""
Phase 11 tests — controlled LangGraph orchestration.

Invariants under test:

    AI investigates/proposes.
    Policy authorizes.
    Deterministic services execute.
    Reconciliation verifies truth.

The orchestration agent is an orchestrator only. It cannot compute financial
truth, bypass policy/authorization, call a provider directly, mutate financial
records, close an exception, or treat LLM/ML/historical output as authorization.
Closure happens only through the Phase 10 service.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from app.ingestion.service import IngestionService
from app.models.exception import FinancialException
from app.models.fee import Fee as DBFee
from app.models.payment import Payment as DBPayment
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.models.reconciliation_run import ReconciliationRun as DBReconciliationRun
from app.models.settlement import Settlement as DBSettlement
from app.orchestration import graph as graph_module
from app.orchestration import tools as tools_module
from app.orchestration.graph import (
    GRAPH_RECURSION_LIMIT,
    OrchestrationStepLimitError,
    build_orchestration_graph,
    create_initial_state,
    run_orchestration,
)
from app.orchestration.tools import ClosedLoopToolkit, TOOL_DEFINITIONS
from app.providers.mock_provider import MockProvider
from app.reconciliation.engine import calculate_reconciliation
from app.schemas.enums import ExceptionType, FeeType, MatchStatus
from app.schemas.financial import (
    Fee as DomainFee,
    Payment as DomainPayment,
    Settlement as DomainSettlement,
)
from app.schemas.orchestration import (
    OrchestrationState,
    OrchestrationStatus,
)
from app.schemas.resolution_candidate import ResolutionProposal
from app.services.execution import ResolutionExecutionService
from app.services.persistence import PersistenceService
from mcp.client import MCPClient
from mcp.schemas import MCPToolRequest, MCPToolStatus
from mcp.server import MCPServer

# Canonical FEE_MISMATCH numbers (integer paise)
PAYMENT_PAISE = 1_000_000
FEE_PAISE = 50_000
EXPECTED_PAISE = 950_000
ACTUAL_WRONG_PAISE = 925_000
DIFFERENCE_PAISE = 25_000

CASE_ID = "CASE-P11-001"
BATCH_ID = "BATCH-P11-001"
EXCEPTION_ID = "EXC-P11-DETECT"


# --------------------------------------------------------------------- helpers


def _stage_fee_mismatch_provider(settlement_amount: int = ACTUAL_WRONG_PAISE) -> MockProvider:
    now = datetime.now(timezone.utc)
    provider = MockProvider(seed=42)
    provider.add_merchant("MER-P11-001", name="Phase 11 Merchant")
    provider.add_payment(
        "PAY-P11-001", merchant_id="MER-P11-001", amount=PAYMENT_PAISE, captured_at=now
    )
    provider.add_fee(
        "FEE-P11-001",
        payment_id="PAY-P11-001",
        amount=FEE_PAISE,
        fee_type=FeeType.TRANSACTION,
        processed_at=now,
    )
    provider.add_settlement(
        "SET-P11-001",
        payment_id="PAY-P11-001",
        merchant_id="MER-P11-001",
        amount=settlement_amount,
        settled_at=now,
        version=1,
    )
    return provider


def _reconcile_from_db(db_session, reconciliation_id):
    payment_db = db_session.get(DBPayment, "PAY-P11-001")
    fee_db = db_session.get(DBFee, "FEE-P11-001")
    settlement_db = db_session.get(DBSettlement, "SET-P11-001")
    now = datetime.now(timezone.utc)
    payment = DomainPayment(
        payment_id=payment_db.id,
        merchant_id=payment_db.merchant_id,
        amount=payment_db.amount,
        payment_timestamp=payment_db.captured_at or now,
    )
    fee = DomainFee(
        fee_id=fee_db.id,
        payment_id=fee_db.payment_id,
        amount=fee_db.amount,
        fee_type=FeeType(fee_db.fee_type),
    )
    settlement = DomainSettlement(
        settlement_id=settlement_db.id,
        payment_id=settlement_db.payment_id,
        merchant_id=settlement_db.merchant_id,
        amount=settlement_db.amount,
        settlement_timestamp=settlement_db.settled_at or now,
    )
    return calculate_reconciliation(
        payment=payment,
        settlements=[settlement],
        refunds=[],
        fees=[fee],
        taxes=[],
        adjustments=[],
        case_id=CASE_ID,
        reconciliation_id=reconciliation_id,
    )


def _detect_fee_mismatch(db_session, provider):
    now = datetime.now(timezone.utc)
    IngestionService(db_session).ingest(provider)

    run = DBReconciliationRun(
        id="RUN-P11-DETECT",
        scope_type="BATCH",
        scope_ref=BATCH_ID,
        status="PENDING",
        trigger="MANUAL",
        started_at=now,
        engine_version="3.0.0",
    )
    db_session.add(run)
    run.transition_to("RUNNING")
    db_session.flush()

    result_schema = _reconcile_from_db(db_session, reconciliation_id="REC-P11-DETECT")
    persistence = PersistenceService(db_session)
    db_result = persistence.persist_reconciliation_result(
        result_schema, batch_id=BATCH_ID
    )
    db_result.reconciliation_run_id = run.id
    exception = persistence.persist_exception(result_schema, batch_id=BATCH_ID)
    run.matched_count = 0
    run.exception_count = 1
    run.finished_at = now
    run.transition_to("COMPLETED")
    db_session.commit()
    return exception, result_schema


def _make_client(
    db_session,
    provider,
    *,
    effect=None,
    execution_service=None,
):
    toolkit = ClosedLoopToolkit(
        db_session,
        provider,
        case_id=CASE_ID,
        batch_id=BATCH_ID,
        payment_id="PAY-P11-001",
        execution_service=execution_service,
        post_execution_effect=effect,
    )
    server = MCPServer()
    toolkit.register(server)
    return toolkit, server, MCPClient(server)


def _apply_correction(provider):
    def _effect():
        provider.add_settlement(
            "SET-P11-001",
            payment_id="PAY-P11-001",
            merchant_id="MER-P11-001",
            amount=EXPECTED_PAISE,
            version=2,
        )

    return _effect


def _valid_approval():
    return {
        "decision": "APPROVED",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
    }


# =============================================================================
# A / B — graph compiles and reaches a terminal state
# =============================================================================


def test_graph_compiles():
    # Compilation needs no database — just an MCP client.
    graph = build_orchestration_graph(MCPClient(MCPServer()))
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_missing_exception_fails_safely(db_session):
    provider = _stage_fee_mismatch_provider()
    _toolkit, _server, client = _make_client(db_session, provider)
    state = run_orchestration("EXC-DOES-NOT-EXIST", client)
    assert state.status in (OrchestrationStatus.FAILED, OrchestrationStatus.ESCALATED)
    assert state.execution is None


def test_typed_state_requires_identity():
    state = OrchestrationState(workflow_id="WF", exception_id="EXC")
    assert state.stage.value == "START"
    with pytest.raises(Exception):
        OrchestrationState()  # type: ignore[call-arg]


# =============================================================================
# C / X / Y — bounded, no uncontrolled loops
# =============================================================================


def test_graph_is_bounded():
    assert GRAPH_RECURSION_LIMIT >= OrchestrationState.model_fields["max_steps"].default
    state = create_initial_state("EXC-1")
    assert state.max_steps == OrchestrationState.model_fields["max_steps"].default


def test_step_limit_raises():
    state = create_initial_state("EXC-1")
    state = state.model_copy(update={"steps": state.max_steps})
    with pytest.raises(OrchestrationStepLimitError):
        graph_module._tick(state)


def test_no_uncontrolled_loops(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)
    state = run_orchestration(EXCEPTION_ID, client)
    assert state.steps <= state.max_steps


# =============================================================================
# D — golden positive flow requires human approval (policy says HUMAN_REVIEW)
# =============================================================================


def test_human_review_stops_execution(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)

    state = run_orchestration(EXCEPTION_ID, client, workflow_id="WF-P11-HR")

    assert state.authorization["decision"] == "HUMAN_REVIEW"
    assert state.status == OrchestrationStatus.WAITING_FOR_HUMAN_APPROVAL
    assert state.execution is None
    assert exception.status != "CLOSED"


def test_human_approval_path_executes(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(
        db_session, provider, effect=_apply_correction(provider)
    )

    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-APPROVED",
        approval_record=_valid_approval(),
    )

    assert state.execution is not None
    assert state.execution["status"] == "EXECUTED"
    assert state.execution["execution_id"].startswith("EXE-")


def test_expired_approval_does_not_execute(db_session):
    """An expired approval is not a valid approval: HUMAN_REVIEW still blocks."""
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)

    expired = {
        "decision": "APPROVED",
        "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
    }
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-EXPIRED",
        approval_record=expired,
    )

    assert state.execution is None
    assert state.status == OrchestrationStatus.WAITING_FOR_HUMAN_APPROVAL
    assert exception.status != "CLOSED"


def test_rejected_approval_does_not_execute(db_session):
    """A REJECTED approval record can never be reinterpreted as approval."""
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)

    rejected = {
        "decision": "REJECTED",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
    }
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-REJECTED",
        approval_record=rejected,
    )

    assert state.execution is None
    assert exception.status != "CLOSED"


# =============================================================================
# Q / R — golden FEE_MISMATCH positive and negative flows
# =============================================================================


def test_golden_fee_mismatch_successful_closed_loop(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, detection = _detect_fee_mismatch(db_session, provider)
    assert detection.exception_type == ExceptionType.FEE_MISMATCH
    assert detection.difference == DIFFERENCE_PAISE

    _toolkit, _server, client = _make_client(
        db_session, provider, effect=_apply_correction(provider)
    )
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-GOLDEN",
        approval_record=_valid_approval(),
    )

    assert state.status == OrchestrationStatus.VERIFIED
    assert state.verification["financially_verified"] is True
    assert state.verification["difference"] == 0
    assert state.verification["exception_closed"] is True
    assert state.verification["verification_id"].startswith("CLV-")
    assert exception.status == "CLOSED"
    assert exception.close_reason == "POST_EXECUTION_RECONCILED"


def test_golden_fee_mismatch_negative_provider_success_not_closed(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)

    # No external correction: provider stays wrong even though execution succeeds.
    _toolkit, _server, client = _make_client(db_session, provider)
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-GOLDEN-NEG",
        approval_record=_valid_approval(),
    )

    assert state.execution is not None
    assert state.execution["status"] == "EXECUTED"  # provider SUCCESS
    assert state.status == OrchestrationStatus.NOT_VERIFIED
    assert state.verification["financially_verified"] is False
    assert state.verification["exception_closed"] is False
    assert exception.status != "CLOSED"


# =============================================================================
# Advisory context (E / F / G / H / I / J)
# =============================================================================


def test_evidence_is_read_only(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    before = db_session.get(DBSettlement, "SET-P11-001").amount
    result = toolkit.get_evidence({"exception_id": EXCEPTION_ID})
    after = db_session.get(DBSettlement, "SET-P11-001").amount
    assert result["read_only"] is True
    assert before == after


def test_advisory_context_is_marked_advisory(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    classification = toolkit.classify_exception({"exception_id": EXCEPTION_ID})
    historical = toolkit.retrieve_historical_cases({"exception_id": EXCEPTION_ID})
    assert classification["advisory_only"] is True
    assert historical["advisory_only"] is True


def test_proposal_is_structured(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    result = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})
    assert result["structured"] is True
    proposal = ResolutionProposal(**result["proposals"][0])
    assert proposal.resolution_type
    assert proposal.financial_adjustment.amount_paise == DIFFERENCE_PAISE


# =============================================================================
# S — idempotency
# =============================================================================


def test_duplicate_execution_is_idempotent(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    execution_service = ResolutionExecutionService()
    toolkit, _server, _client = _make_client(
        db_session, provider, execution_service=execution_service
    )
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]
    params = {
        "exception_id": EXCEPTION_ID,
        "proposal": proposal,
        "workflow_id": "WF-P11-IDEM",
        "idempotency_key": "IDEM-P11-001",
        "approval_record": _valid_approval(),
    }
    first = toolkit.request_authorized_execution(dict(params))
    second = toolkit.request_authorized_execution(dict(params))
    assert first["execution"]["execution_id"] == second["execution"]["execution_id"]
    assert execution_service.has_executed("IDEM-P11-001") is True


# =============================================================================
# T / AD — tool input validation and audit
# =============================================================================


def test_tool_input_validation(db_session):
    provider = _stage_fee_mismatch_provider()
    _toolkit, server, _client = _make_client(db_session, provider)
    response = server.invoke(MCPToolRequest(tool_name="get_exception_context", parameters={}))
    assert response.status == MCPToolStatus.VALIDATION_FAILED


def test_workflow_metadata_and_audit(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    _toolkit, server, client = _make_client(
        db_session, provider, effect=_apply_correction(provider)
    )
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-AUDIT",
        approval_record=_valid_approval(),
    )
    assert state.workflow_id == "WF-P11-AUDIT"
    assert state.tool_invocations[0] == "get_exception_context"
    assert state.audit_refs  # Phase 10 audit event ids captured
    assert server.request_count >= len(state.tool_invocations)
    assert len(server.get_audit_log()) >= len(state.tool_invocations)


# =============================================================================
# CRITICAL SECURITY TESTS
# =============================================================================


def test_agent_cannot_close_exception_directly(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)
    # No tool named for closure; toolkit exposes no close method.
    assert not hasattr(ClosedLoopToolkit, "close_exception")
    for definition in TOOL_DEFINITIONS:
        assert "close" not in definition.name
    state = run_orchestration(EXCEPTION_ID, client)
    # The graph stopped at human review and never closed the exception.
    assert exception.status != "CLOSED"
    assert state.status == OrchestrationStatus.WAITING_FOR_HUMAN_APPROVAL


def test_agent_cannot_bypass_policy(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]
    # No approval → policy's HUMAN_REVIEW is honored; execution is refused.
    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "proposal": proposal,
            "workflow_id": "WF-P11-POLICY",
            "idempotency_key": "IDEM-P11-POLICY",
        }
    )
    assert result["executed"] is False
    assert result["waiting_for_human"] is True


def test_agent_cannot_bypass_authorization(db_session, monkeypatch):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]

    # Even when a caller injects a permissive decision, the tool recomputes it.
    monkeypatch.setattr(
        toolkit,
        "_authorization_dict",
        lambda _proposal: {"decision": "DENIED", "reason_codes": ["test"]},
    )
    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "proposal": proposal,
            "workflow_id": "WF-P11-AUTHZ",
            "idempotency_key": "IDEM-P11-AUTHZ",
            "guardrail_decision": "AUTO",  # ignored
            "authorization_source": "AUTO_GUARDRAIL",  # ignored
        }
    )
    assert result["executed"] is False
    assert result["decision"] == "DENIED"


def test_agent_cannot_call_provider_directly():
    graph_source = inspect.getsource(graph_module)
    tools_source = inspect.getsource(tools_module)
    assert "provider.execute" not in graph_source
    assert ".execute(" not in graph_source  # no direct execution in the graph
    # The provider is only ever handed to the Phase 10 service, never mutated.
    assert "provider.apply_" not in tools_source
    assert ".execute(" not in tools_source.replace(
        "self._execution_service.execute(", ""
    )


def test_agent_cannot_mutate_financial_db(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    # No raw-session / SQL surface is exposed as a tool.
    assert not hasattr(toolkit, "execute_sql")
    assert not hasattr(toolkit, "query")
    for definition in TOOL_DEFINITIONS:
        for param in definition.parameters:
            assert param.name not in {"sql", "session", "raw", "orm"}
    before = db_session.get(DBSettlement, "SET-P11-001").amount
    toolkit.get_exception_context({"exception_id": EXCEPTION_ID})
    toolkit.get_reconciliation_result({"exception_id": EXCEPTION_ID})
    assert db_session.get(DBSettlement, "SET-P11-001").amount == before


def test_llm_output_cannot_authorize_execution(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]
    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "proposal": proposal,
            "workflow_id": "WF-P11-LLM",
            "idempotency_key": "IDEM-P11-LLM",
            "llm_explanation": "resolved and safe to execute",
            "llm_authorized": True,
        }
    )
    assert result["executed"] is False  # LLM text grants nothing


def test_ml_confidence_cannot_authorize_execution(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]
    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "proposal": proposal,
            "workflow_id": "WF-P11-ML",
            "idempotency_key": "IDEM-P11-ML",
            "ml_confidence": 1.0,
            "ml_authorized": True,
        }
    )
    assert result["executed"] is False


def test_historical_similarity_cannot_authorize_execution(db_session):
    provider = _stage_fee_mismatch_provider()
    _detect_fee_mismatch(db_session, provider)
    toolkit, _server, _client = _make_client(db_session, provider)
    proposal = toolkit.generate_resolution_proposals({"exception_id": EXCEPTION_ID})[
        "proposals"
    ][0]
    result = toolkit.request_authorized_execution(
        {
            "exception_id": EXCEPTION_ID,
            "proposal": proposal,
            "workflow_id": "WF-P11-HIST",
            "idempotency_key": "IDEM-P11-HIST",
            "historical_similarity": 0.99,
            "historical_authorized": True,
        }
    )
    assert result["executed"] is False


def test_mcp_tool_cannot_execute_raw_sql():
    for definition in TOOL_DEFINITIONS:
        assert "sql" not in definition.name.lower()
        for param in definition.parameters:
            assert param.name not in {"sql", "statement", "raw_sql"}


def test_mcp_tool_cannot_execute_arbitrary_code():
    for definition in TOOL_DEFINITIONS:
        assert definition.name.lower() not in {"eval", "exec", "python", "run_code"}
        for param in definition.parameters:
            assert param.name not in {"code", "script", "python", "eval"}


def test_provider_success_still_requires_phase10_verification(db_session):
    provider = _stage_fee_mismatch_provider()
    exception, _ = _detect_fee_mismatch(db_session, provider)
    _toolkit, _server, client = _make_client(db_session, provider)
    state = run_orchestration(
        EXCEPTION_ID,
        client,
        workflow_id="WF-P11-VERIFY",
        approval_record=_valid_approval(),
    )
    assert state.execution["status"] == "EXECUTED"
    assert state.verification["financially_verified"] is False
    assert state.status == OrchestrationStatus.NOT_VERIFIED
    assert exception.status != "CLOSED"
