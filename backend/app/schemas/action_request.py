"""
Action Request schema for Razorpay CloseLoop Phase 7I.

Defines the action request object produced at the resolve/action boundary.

This is NOT an execution request.
It is a proposal that a future execution service can pick up.

CRITICAL:
- This schema does not execute financial actions
- It only produces an idempotent, authorized request
- Real execution belongs to a future guarded agent/action layer
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Action Request Enums
# ─────────────────────────────────────────────────────────────────────────────


class ActionStatus(str, Enum):
    """Status of the action request."""
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AuthorizationSource(str, Enum):
    """Source of authorization."""
    AUTO_GUARDRAIL = "AUTO_GUARDRAIL"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    NONE = "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# Action Request
# ─────────────────────────────────────────────────────────────────────────────


class ActionRequest(BaseModel):
    """An idempotent action request for future financial execution.

    This is the BOUNDARY between the recommendation/verification pipeline
    and the future execution layer.

    A future execution service would:
    1. Receive this request
    2. Validate idempotency
    3. Execute the financial action
    4. Return execution result

    This request must NEVER be executed by the workflow itself.
    """

    # Identity
    action_id: str = Field(
        ..., description="Unique action request ID (idempotency key)"
    )
    idempotency_key: str = Field(
        ..., description="Idempotency key — duplicate requests are deduplicated"
    )
    workflow_id: str = Field(..., description="Source workflow ID")
    exception_id: str = Field(..., description="Exception being resolved")
    case_id: Optional[str] = Field(default=None, description="Case being resolved")
    candidate_id: Optional[str] = Field(default=None, description="Selected candidate")

    # Resolution
    resolution_type: str = Field(..., description="Proposed resolution type")
    financial_adjustment_paise: int = Field(
        ..., description="Financial adjustment in integer paise"
    )
    financial_adjustment_description: Optional[str] = Field(
        default=None, description="Human-readable description of adjustment"
    )

    # Authorization
    authorization_source: AuthorizationSource = Field(
        ..., description="Who authorized this action"
    )
    authorized_by: Optional[str] = Field(
        default=None, description="Human reviewer ID or 'auto_guardrail'"
    )
    authorization_timestamp: Optional[datetime] = Field(
        default=None, description="When authorization was granted"
    )

    # Verification
    verification_passed: bool = Field(
        ..., description="Whether verification passed before this request"
    )
    verification_action: Optional[str] = Field(
        default=None, description="Verification action result"
    )

    # Guardrail
    guardrail_decision: str = Field(
        ..., description="Guardrail decision at time of request"
    )
    guardrail_confidence: Optional[float] = Field(
        default=None, description="Guardrail confidence"
    )

    # Status
    status: ActionStatus = Field(
        default=ActionStatus.PENDING, description="Current status"
    )

    # Context
    evidence_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Evidence context at time of request"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When request was created"
    )

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"ActionRequest: {self.action_id} | "
            f"Resolution: {self.resolution_type} | "
            f"Amount: {self.financial_adjustment_paise} paise | "
            f"Authorization: {self.authorization_source.value} | "
            f"Status: {self.status.value}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Action Request Result
# ─────────────────────────────────────────────────────────────────────────────


class ActionRequestResult(BaseModel):
    """Result of attempting to create an action request."""
    success: bool = Field(..., description="Whether request was created")
    action_request: Optional[ActionRequest] = Field(
        default=None, description="Created request (if success)"
    )
    rejection_reasons: List[str] = Field(
        default_factory=list, description="Reasons request was rejected"
    )
    blocked: bool = Field(
        default=False, description="Whether the request was blocked by safety"
    )
