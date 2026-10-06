"""
Orchestration state for Razorpay CloseLoop Phase 11.

The Phase 11 agent is an ORCHESTRATOR, not a financial authority. Its graph
state therefore carries only *context* and *references* to results produced by
deterministic services:

    AI investigates/proposes.
    Policy authorizes.
    Deterministic services execute.
    Reconciliation verifies truth.

Hard rules encoded here:

* No ORM sessions, provider handles, credentials or secrets live in state.
* Every field is JSON-serializable domain data (dicts/lists/plain scalars), so
  the graph can be persisted, logged and replayed without leaking internals.
* The state never carries a financial-truth *computation*: ``difference`` and
  ``match_status`` only ever appear copied from the deterministic reconciliation
  engine's output.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrchestrationStage(str, Enum):
    """Ordered stages of the controlled orchestration graph."""

    START = "START"
    LOAD_EXCEPTION = "LOAD_EXCEPTION"
    COLLECT_EVIDENCE = "COLLECT_EVIDENCE"
    GET_INTELLIGENCE = "GET_INTELLIGENCE"
    GET_HISTORICAL_CONTEXT = "GET_HISTORICAL_CONTEXT"
    GENERATE_RESOLUTION = "GENERATE_RESOLUTION"
    POLICY_EVALUATION = "POLICY_EVALUATION"
    AUTHORIZATION_DECISION = "AUTHORIZATION_DECISION"
    EXECUTE_IF_AUTHORIZED = "EXECUTE_IF_AUTHORIZED"
    POST_EXECUTION_VERIFICATION = "POST_EXECUTION_VERIFICATION"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ESCALATION = "ESCALATION"
    FINALIZE = "FINALIZE"


class OrchestrationStatus(str, Enum):
    """Terminal/lifecycle status of an orchestration run.

    ``WAITING_FOR_HUMAN_APPROVAL`` is a *stop* state: the graph halts before any
    financial execution and never reinterprets it as approval.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN_APPROVAL = "WAITING_FOR_HUMAN_APPROVAL"
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


#: Hard upper bound on graph node executions (defense against uncontrolled loops).
DEFAULT_MAX_STEPS = 25


class OrchestrationState(BaseModel):
    """Typed state for the Phase 11 controlled orchestration graph.

    Nodes perform work by invoking narrow tools; this model only holds the
    resulting context. Each node returns the fields it changes.
    """

    # ── Identity ──
    workflow_id: str = Field(..., description="Unique workflow identifier")
    exception_id: str = Field(..., description="Exception under investigation")
    case_id: Optional[str] = Field(default=None, description="Case identifier")
    merchant_id: Optional[str] = Field(default=None, description="Merchant identifier")
    payment_id: Optional[str] = Field(default=None, description="Payment identifier")

    # ── Progress ──
    stage: OrchestrationStage = Field(
        default=OrchestrationStage.START, description="Current graph stage"
    )
    status: OrchestrationStatus = Field(
        default=OrchestrationStatus.PENDING, description="Lifecycle status"
    )
    steps: int = Field(default=0, ge=0, description="Node executions so far")
    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS, ge=1, description="Hard node-execution bound"
    )

    # ── Deterministic context ──
    exception_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Exception context (deterministic read)"
    )
    evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="Read-only evidence snapshot"
    )
    reconciliation: Optional[Dict[str, Any]] = Field(
        default=None, description="Latest deterministic reconciliation result"
    )

    # ── Advisory context (never authorizing) ──
    intelligence: Optional[Dict[str, Any]] = Field(
        default=None, description="Advisory classification / ML intelligence"
    )
    historical_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Advisory historical similar cases"
    )

    # ── Resolution / policy / authorization ──
    proposals: List[Dict[str, Any]] = Field(
        default_factory=list, description="Structured resolution proposals"
    )
    selected_proposal: Optional[Dict[str, Any]] = Field(
        default=None, description="Selected structured proposal"
    )
    authorization: Optional[Dict[str, Any]] = Field(
        default=None, description="Authorization decision (policy output)"
    )
    approval_record: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Human approval artifact. Only consulted when policy returns "
            "HUMAN_REVIEW; a valid APPROVED record lets a later invocation "
            "continue through the existing authorization/execution path."
        ),
    )

    # ── Execution / verification ──
    execution: Optional[Dict[str, Any]] = Field(
        default=None, description="Phase 9 execution result summary"
    )
    verification: Optional[Dict[str, Any]] = Field(
        default=None, description="Phase 10 closed-loop verification result"
    )

    # ── Outcome ──
    final_status: Optional[str] = Field(default=None, description="Terminal status")
    tool_invocations: List[str] = Field(
        default_factory=list, description="Ordered MCP tool invocations"
    )
    audit_refs: List[str] = Field(
        default_factory=list, description="Audit identifiers referenced"
    )
    errors: List[str] = Field(default_factory=list, description="Accumulated errors")
    warnings: List[str] = Field(default_factory=list, description="Accumulated warnings")

    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        return (
            f"Orchestration {self.workflow_id} | exception={self.exception_id} | "
            f"stage={self.stage.value} | status={self.status.value}"
        )


__all__ = [
    "OrchestrationStage",
    "OrchestrationStatus",
    "OrchestrationState",
    "DEFAULT_MAX_STEPS",
]
