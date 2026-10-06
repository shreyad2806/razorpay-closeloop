"""
Controlled tool boundary for Razorpay CloseLoop Phase 11.

This module is the *only* way the orchestration graph may touch financial
systems. Every tool is a narrow domain capability with a structured input and
output. There is deliberately no tool that can:

* run arbitrary SQL or arbitrary Python,
* open a raw ORM session,
* mutate financial records directly,
* expose secrets/credentials/connection strings,
* call a provider mutation API outside Phase 9,
* close an exception.

The single sensitive tool, ``request_authorized_execution``, does **not** mean
"execute whatever the agent asks". It recomputes authorization itself through
the Phase 8 :class:`~app.services.authorization.AuthorizationService` and only
then delegates to the Phase 9
:class:`~app.services.execution.ResolutionExecutionService`:

    agent → tool → AuthorizationService → ResolutionExecutionService → provider

Closure is never performed here: only the Phase 10
:class:`~app.services.post_execution_reconciliation.PostExecutionReconciliationService`
can establish financial truth and close an exception.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from mcp.schemas import MCPToolDefinition, MCPToolParameter
from app.auth.principal import Permission, Principal
from app.auth.rbac import AccessDeniedError, require_permission
from app.models.exception import FinancialException
from app.models.reconciliation import ReconciliationResult as DBReconciliationResult
from app.schemas.closed_loop import VerificationDecision
from app.schemas.execution import ExecutionResult
from app.schemas.resolution_candidate import (
    CandidateRanking,
    FinancialAdjustment,
    ResolutionProposal,
)
from app.services.authorization import AuthorizationService
from app.services.execution import ResolutionExecutionService
from app.services.post_execution_reconciliation import (
    PostExecutionReconciliationService,
)

#: Hard bounds so a tool argument can never explode work.
MAX_TOP_K = 20


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (structured input/output contract)
# ─────────────────────────────────────────────────────────────────────────────

READ_TOOL_DEFINITIONS: List[MCPToolDefinition] = [
    MCPToolDefinition(
        name="get_exception_context",
        description="Read an exception's deterministic context (read-only).",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_evidence",
        description="Read the current financial evidence snapshot (read-only).",
        category="evidence",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_reconciliation_result",
        description="Read the latest deterministic reconciliation result (read-only).",
        category="reconciliation",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="classify_exception",
        description="Advisory exception classification. Never authorizes.",
        category="classification",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="retrieve_historical_cases",
        description="Advisory historical similar cases. Never authorizes.",
        category="classification",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
            MCPToolParameter(name="top_k", type="number", required=False, default=5),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="generate_resolution_proposals",
        description="Produce structured ResolutionProposal objects from evidence.",
        category="resolution",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="evaluate_resolution_policy",
        description="Evaluate Phase 8 policy/guardrails for a structured proposal.",
        category="guardrails",
        parameters=[
            MCPToolParameter(name="proposal", type="object", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_authorization_decision",
        description="Return the Phase 8 authorization decision for a proposal.",
        category="guardrails",
        parameters=[
            MCPToolParameter(name="proposal", type="object", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="request_authorized_execution",
        description=(
            "Request controlled financial execution. Internally re-authorizes "
            "through Phase 8 and executes through Phase 9 only. Never bypassable."
        ),
        category="execution",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
            MCPToolParameter(name="proposal", type="object", required=True),
            MCPToolParameter(name="workflow_id", type="string", required=True),
            MCPToolParameter(name="idempotency_key", type="string", required=True),
            MCPToolParameter(name="approval_record", type="object", required=False),
        ],
        is_financial=True,
        requires_guardrail=True,
        requires_verification=True,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="verify_post_execution_reconciliation",
        description=(
            "Verify a Phase 9 execution against fresh provider state through the "
            "Phase 10 deterministic closed-loop service (sole closure arbiter)."
        ),
        category="execution",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
            MCPToolParameter(name="execution", type="object", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        requires_verification=True,
        idempotent=True,
    ),
    MCPToolDefinition(
        name="get_verification_status",
        description="Read the current verification/closure status (read-only).",
        category="lineage",
        parameters=[
            MCPToolParameter(name="exception_id", type="string", required=True),
        ],
        is_financial=False,
        requires_guardrail=False,
        idempotent=True,
    ),
]

#: The complete allow-list of orchestration tools.
TOOL_DEFINITIONS: List[MCPToolDefinition] = READ_TOOL_DEFINITIONS


# ─────────────────────────────────────────────────────────────────────────────
# Toolkit
# ─────────────────────────────────────────────────────────────────────────────


class ClosedLoopToolkit:
    """Narrow, auditable domain tools bound to a single workflow context.

    The toolkit is constructed with a database session and the authoritative
    provider for the workflow. Neither is ever handed to the agent directly:
    tools return plain dictionaries only.
    """

    def __init__(
        self,
        session: Session,
        provider: Any,
        *,
        case_id: str,
        batch_id: str,
        payment_id: Optional[str] = None,
        execution_service: Optional[ResolutionExecutionService] = None,
        post_execution_effect: Optional[Callable[[], None]] = None,
        principal: Optional[Principal] = None,
    ) -> None:
        self._session = session
        self._provider = provider
        self._case_id = case_id
        self._batch_id = batch_id
        self._payment_id = payment_id
        self._execution_service = execution_service or ResolutionExecutionService()
        self._post_execution_effect = post_execution_effect
        # Validated identity bound by the caller (never taken from tool args).
        # When bound, RBAC is enforced fail-closed on sensitive operations.
        self._principal = principal

    def bind_principal(self, principal: Optional[Principal]) -> None:
        """Bind the validated application principal for this toolkit.

        The principal always originates from the authentication layer. Tool
        arguments can never set, override or escalate it.
        """
        if principal is not None and not isinstance(principal, Principal):
            raise AccessDeniedError("only a validated Principal may be bound")
        self._principal = principal

    @property
    def principal(self) -> Optional[Principal]:
        return self._principal

    def _require_execution_permission(self) -> Optional[Dict[str, Any]]:
        """Enforce RBAC on the sensitive execution tool when a principal is bound.

        Returns an error dict when the operation is refused, or ``None`` when
        execution may proceed.
        """
        if self._principal is None:
            # Machine scope (accepted Phase 11 behaviour): Phase 8 policy and
            # the Phase 9 approval preconditions remain the authority.
            return None
        try:
            require_permission(self._principal, Permission.REQUEST_EXECUTION)
        except AccessDeniedError as error:
            return {
                "executed": False,
                "error": f"rbac denied: {error.reason}",
            }
        return None

    # ------------------------------------------------------------- registration

    @property
    def handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return {
            "get_exception_context": self.get_exception_context,
            "get_evidence": self.get_evidence,
            "get_reconciliation_result": self.get_reconciliation_result,
            "classify_exception": self.classify_exception,
            "retrieve_historical_cases": self.retrieve_historical_cases,
            "generate_resolution_proposals": self.generate_resolution_proposals,
            "evaluate_resolution_policy": self.evaluate_resolution_policy,
            "get_authorization_decision": self.get_authorization_decision,
            "request_authorized_execution": self.request_authorized_execution,
            "verify_post_execution_reconciliation": self.verify_post_execution_reconciliation,
            "get_verification_status": self.get_verification_status,
        }

    def register(self, server: Any) -> int:
        """Register every allow-listed tool on an MCPServer."""
        count = 0
        handlers = self.handlers
        for definition in TOOL_DEFINITIONS:
            handler = handlers.get(definition.name)
            if handler is None:  # pragma: no cover - programming error
                continue
            server.register_tool(definition, handler)
            count += 1
        return count

    # ------------------------------------------------------------------- reads

    def get_exception_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        exception_id = params.get("exception_id")
        if not exception_id:
            return {"found": False, "error": "exception_id is required"}
        exception = self._session.get(FinancialException, exception_id)
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}
        return {
            "found": True,
            "exception": {
                "exception_id": exception.id,
                "case_id": exception.case_id,
                "payment_id": exception.payment_id,
                "merchant_id": exception.merchant_id,
                "batch_id": exception.batch_id,
                "exception_type": exception.exception_type,
                "status": exception.status,
                "expected_amount": int(exception.expected_amount or 0),
                "actual_amount": int(exception.actual_amount or 0),
                "difference": int(exception.difference or 0),
            },
        }

    def get_evidence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}
        financial_state = self._financial_state()
        return {
            "found": True,
            "exception_id": exception.id,
            "payment_id": exception.payment_id,
            "financial_state": financial_state,
            "read_only": True,
        }

    def get_reconciliation_result(self, params: Dict[str, Any]) -> Dict[str, Any]:
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}
        row = (
            self._session.query(DBReconciliationResult)
            .filter_by(case_id=exception.case_id)
            .order_by(DBReconciliationResult.reconciliation_timestamp.desc())
            .first()
        )
        if row is None:
            return {"found": False, "error": "no reconciliation result"}
        return {
            "found": True,
            "case_id": row.case_id,
            "payment_id": row.payment_id,
            "expected_amount": int(row.expected_amount or 0),
            "actual_amount": int(row.actual_amount or 0),
            "difference": int(getattr(row, "difference", 0) or 0),
            "match_status": getattr(row, "match_status", None),
            "exception_type": getattr(row, "exception_type", None),
        }

    # ---------------------------------------------------- advisory (never auth)

    def classify_exception(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Advisory classification derived from deterministic engine output.

        ``ml_confidence`` is reported for display only. It carries no
        authorization weight and is never read by any policy/execution path.
        """
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}
        return {
            "found": True,
            "exception_id": exception.id,
            "exception_type": exception.exception_type,
            "source": "deterministic_reconciliation",
            "advisory_only": True,
            "ml_confidence": None,
        }

    def retrieve_historical_cases(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Advisory historical lookup. Similarity is never an authorization input."""
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}

        raw_top_k = params.get("top_k", 5)
        try:
            top_k = max(1, min(int(raw_top_k), MAX_TOP_K))
        except (TypeError, ValueError):
            top_k = 5

        rows = (
            self._session.query(FinancialException)
            .filter(FinancialException.exception_type == exception.exception_type)
            .filter(FinancialException.id != exception.id)
            .limit(top_k)
            .all()
        )
        similar = [
            {
                "exception_id": r.id,
                "case_id": r.case_id,
                "exception_type": r.exception_type,
                "status": r.status,
                "difference": int(r.difference or 0),
            }
            for r in rows
        ]
        return {
            "found": True,
            "exception_id": exception.id,
            "advisory_only": True,
            "similar_cases": similar,
            "similarity": float(len(similar) > 0),
        }

    # ------------------------------------------------------------ proposals

    def generate_resolution_proposals(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a structured ``ResolutionProposal`` from deterministic evidence."""
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}

        difference = abs(int(exception.difference or 0))
        proposal = ResolutionProposal(
            candidate_id=f"CAND-{exception.id}",
            exception_id=exception.id,
            case_id=exception.case_id,
            resolution_type="FEE_ADJUSTMENT",
            resolution_description="Correct the deterministic fee settlement difference",
            financial_adjustment=FinancialAdjustment(
                adjustment_type="FEE_CORRECTION",
                amount_paise=difference,
                direction="CREDIT",
                calculation_basis="deterministic_reconciliation_difference",
            ),
            supporting_evidence_ids=[exception.payment_id, exception.reconciliation_id],
            evidence_compatible=True,
            evidence_coverage=0.95,
            coverage_explanation="Deterministic reconciliation identified the difference",
            sources=["deterministic_evidence"],
            ranking=CandidateRanking(
                rank=1,
                confidence_score=0.92,
                evidence_support=0.95,
                ml_support=0.0,
                historical_support=0.0,
            ),
            rationale="Structured proposal derived from the reconciliation engine difference",
        )
        return {
            "found": True,
            "structured": True,
            "proposals": [proposal.model_dump(mode="json")],
        }

    # -------------------------------------------------------------- policy/auth

    def evaluate_resolution_policy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            proposal = _proposal_from_params(params)
        except ValueError as error:
            return {"authorized": False, "error": str(error)}
        return self._authorization_dict(proposal)

    def get_authorization_decision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.evaluate_resolution_policy(params)

    # ---------------------------------------------------------------- execution

    def request_authorized_execution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """The sensitive tool: re-authorize, then execute through Phase 9.

        Authorization is recomputed internally. Any caller-supplied
        ``guardrail_decision`` / ``authorization_source`` / ``ml_confidence`` is
        ignored — there is no field that can grant execution.
        """
        exception_id = params.get("exception_id")
        workflow_id = params.get("workflow_id")
        idempotency_key = params.get("idempotency_key")
        if not exception_id or not workflow_id or not idempotency_key:
            return {"executed": False, "error": "exception_id, workflow_id and idempotency_key are required"}

        # SECURITY: role/authorization arguments supplied by the caller or an
        # agent are ignored — identity and permissions come from the bound
        # principal, and Phase 8 authorization is recomputed below.
        rbac_error = self._require_execution_permission()
        if rbac_error is not None:
            return rbac_error

        exception = self._session.get(FinancialException, exception_id)
        if exception is None:
            return {"executed": False, "error": f"exception {exception_id!r} not found"}

        try:
            proposal = _proposal_from_params(params)
        except ValueError as error:
            return {"executed": False, "error": str(error)}

        # Phase 8: authorization is a decision of the policy engine, never the agent.
        authorization = self._authorization_dict(proposal)
        decision = authorization["decision"]

        if decision == "HUMAN_REVIEW":
            approval = params.get("approval_record")
            if not _is_valid_approval(approval):
                return {
                    "executed": False,
                    "waiting_for_human": True,
                    "decision": decision,
                    "authorization": authorization,
                    "error": "HUMAN_REVIEW requires a valid APPROVED approval record",
                }
            authorization_source = "HUMAN_APPROVAL"
        elif decision == "AUTO":
            approval = None
            authorization_source = "AUTO_GUARDRAIL"
        else:
            # UNRESOLVED / DENIED / BLOCKED / anything unknown → never execute.
            return {
                "executed": False,
                "decision": decision,
                "authorization": authorization,
                "error": f"authorization decision '{decision}' does not permit execution",
            }

        adjustment = proposal.financial_adjustment.amount_paise
        action_request: Dict[str, Any] = {
            "action_id": f"ACT-{workflow_id}-{exception_id}",
            "idempotency_key": idempotency_key,
            "workflow_id": workflow_id,
            "exception_id": exception_id,
            "case_id": exception.case_id,
            "candidate_id": proposal.candidate_id,
            "resolution_type": proposal.resolution_type,
            "financial_adjustment_paise": adjustment,
            "currency": "INR",
            "authorization_source": authorization_source,
            "verification_passed": True,
            "guardrail_decision": decision,
            "guardrail_confidence": authorization.get("confidence"),
            "evidence_summary": {},
            "metadata": {"risk": authorization.get("risk_category")},
            "current_evidence_digest": params.get("current_evidence_digest"),
        }
        if approval is not None:
            action_request["approval_record"] = approval

        # Phase 9: the only execution path.
        result = self._execution_service.execute(action_request, self._financial_state())

        executed = result.status.value == "EXECUTED"
        if executed and self._post_execution_effect is not None:
            # Model the external system applying the correction; never fabricate
            # the after-state ourselves.
            self._post_execution_effect()

        return {
            "executed": executed,
            "decision": decision,
            "authorization": authorization,
            # Full Phase 9 result (needed by the Phase 10 verification tool).
            "execution": result.model_dump(mode="json"),
            "execution_summary": _execution_summary(result),
        }

    # -------------------------------------------------------------- verification

    def verify_post_execution_reconciliation(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        exception_id = params.get("exception_id")
        execution_dict = params.get("execution")
        if not exception_id or not isinstance(execution_dict, dict):
            return {"verified": False, "error": "exception_id and execution are required"}

        exception = self._session.get(FinancialException, exception_id)
        if exception is None:
            return {"verified": False, "error": f"exception {exception_id!r} not found"}

        try:
            execution_result = ExecutionResult(**execution_dict)
        except Exception as error:  # pragma: no cover - defensive
            return {"verified": False, "error": f"malformed execution: {error}"}

        service = PostExecutionReconciliationService(
            session=self._session,
            execution_service=self._execution_service,
        )
        try:
            verification = service.verify_and_close(
                execution_result,
                exception=exception,
                provider=self._provider,
                case_id=exception.case_id,
                batch_id=exception.batch_id or self._batch_id,
                payment_id=exception.payment_id,
                workflow_id=params.get("workflow_id"),
                idempotency_key=execution_result.idempotency_key,
            )
        except Exception as error:
            return {"verified": False, "error": f"verification error: {error}"}

        return {
            "verified": verification.financially_verified,
            "decision": verification.decision.value,
            "financially_verified": verification.financially_verified,
            "exception_closed": verification.exception_closed,
            "exception_closed_directly": False,
            "difference": verification.difference,
            "expected_amount": verification.expected_amount,
            "actual_amount": verification.actual_amount,
            "match_status": verification.match_status,
            "execution_status": verification.execution_status,
            "verification_id": verification.verification_id,
            "audit_event_ids": verification.audit_event_ids,
            "failure_reason": verification.failure_reason,
        }

    def get_verification_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        exception_id = params.get("exception_id")
        exception = self._session.get(FinancialException, exception_id) if exception_id else None
        if exception is None:
            return {"found": False, "error": f"exception {exception_id!r} not found"}
        return {
            "found": True,
            "exception_id": exception.id,
            "status": exception.status,
            "closed": exception.status == "CLOSED",
            "close_reason": exception.close_reason,
        }

    # ---------------------------------------------------------------- internals

    def _authorization_dict(self, proposal: ResolutionProposal) -> Dict[str, Any]:
        decision = AuthorizationService(session=self._session).authorize_proposal(proposal)
        return {
            "decision": decision.decision.value,
            "risk_category": decision.risk_category,
            "reason_codes": list(decision.reason_codes),
            "primary_reason": decision.primary_reason,
            "confidence": decision.confidence,
            "financial_exposure_paise": decision.financial_exposure_paise,
            "policy_version": decision.policy_version,
            "required_approval": decision.required_approval,
        }

    def _financial_state(self) -> Dict[str, Any]:
        """Read the current DB financial state for before/after capture.

        This is a *read* of persisted deterministic state, not a computation of
        financial truth by the agent.
        """
        row = (
            self._session.query(DBReconciliationResult)
            .filter_by(case_id=self._case_id)
            .order_by(DBReconciliationResult.reconciliation_timestamp.desc())
            .first()
        )
        if row is None:
            return {}
        return {
            "payment_amount": int(row.payment_amount or 0),
            "expected_amount": int(row.expected_amount or 0),
            "actual_amount": int(row.actual_amount or 0),
            "difference": int(getattr(row, "difference", 0) or 0),
            "total_refunds": int(row.total_refunds or 0),
            "total_fees": int(row.total_fees or 0),
            "total_taxes": int(row.total_taxes or 0),
            "total_adjustments": int(row.total_adjustments or 0),
            "settlement_count": 0,
            "refund_count": 0,
            "fee_count": 0,
            "tax_count": 0,
            "adjustment_count": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _proposal_from_params(params: Dict[str, Any]) -> ResolutionProposal:
    raw = params.get("proposal")
    if not isinstance(raw, dict):
        raise ValueError("a structured 'proposal' object is required")
    try:
        return ResolutionProposal(**raw)
    except Exception as error:
        raise ValueError(f"malformed proposal: {error}") from error


def _is_valid_approval(approval: Any) -> bool:
    if not isinstance(approval, dict):
        return False
    if approval.get("decision") != "APPROVED":
        return False
    expires_at = approval.get("expires_at")
    if isinstance(expires_at, datetime):
        if datetime.utcnow() > expires_at.replace(tzinfo=None):
            return False
    elif isinstance(expires_at, str):
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.utcnow() > parsed.replace(tzinfo=None):
                return False
        except ValueError:
            return False
    return True


def _execution_summary(result: ExecutionResult) -> Dict[str, Any]:
    return {
        "execution_id": result.execution_id,
        "action_id": result.action_id,
        "exception_id": result.exception_id,
        "case_id": result.case_id,
        "candidate_id": result.candidate_id,
        "resolution_type": result.resolution_type,
        "status": result.status.value,
        "idempotency_key": result.idempotency_key,
        "requested_adjustment_paise": result.requested_adjustment_paise,
        "actual_adjustment_paise": result.actual_adjustment_paise,
        "error": result.error,
    }


__all__ = [
    "ClosedLoopToolkit",
    "TOOL_DEFINITIONS",
    "READ_TOOL_DEFINITIONS",
]


# expose the full ExecutionResult for verification reconstruction
def execution_result_to_dict(result: ExecutionResult) -> Dict[str, Any]:
    """Serialize a full Phase 9 ExecutionResult for state hand-off."""
    return result.model_dump(mode="json")
