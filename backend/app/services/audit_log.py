"""
Audit Log Service for Razorpay CloseLoop Phase 8F.

Immutable-style audit event storage.
Every automated resolution must be explainable after the fact.

Audit history must not be silently overwritten.
If a correction is necessary, create another audit event.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.audit import (
    AuditEvent,
    AuditEventType,
    ActorType,
    FinalOutcome,
    GuardrailMetadata,
    ModelMetadata,
    ActionMetadata,
    RollbackMetadata,
    VerificationMetadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Audit Log Service
# ─────────────────────────────────────────────────────────────────────────────


class AuditLogService:
    """Immutable audit log service.

    Stores audit events in append-only fashion.
    Never modifies or deletes existing events.
    """

    def __init__(self):
        # In-memory store (simulates database)
        self._events: List[AuditEvent] = []
        self._events_by_id: Dict[str, AuditEvent] = {}
        self._events_by_workflow: Dict[str, List[str]] = {}

    def record(self, event: AuditEvent) -> AuditEvent:
        """Record an audit event.

        Events are append-only. Once recorded, they cannot be modified.
        """
        # Ensure unique ID
        if event.event_id in self._events_by_id:
            # Deduplicate — return existing
            return self._events_by_id[event.event_id]

        self._events.append(event)
        self._events_by_id[event.event_id] = event

        # Index by workflow
        wf_id = event.workflow_id
        if wf_id not in self._events_by_workflow:
            self._events_by_workflow[wf_id] = []
        self._events_by_workflow[wf_id].append(event.event_id)

        return event

    def create_event(
        self,
        event_type: AuditEventType,
        workflow_id: str,
        exception_id: str,
        **kwargs,
    ) -> AuditEvent:
        """Create and record an audit event."""
        event = AuditEvent(
            event_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            event_type=event_type,
            workflow_id=workflow_id,
            exception_id=exception_id,
            **kwargs,
        )
        return self.record(event)

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get an audit event by ID."""
        return self._events_by_id.get(event_id)

    def get_workflow_events(self, workflow_id: str) -> List[AuditEvent]:
        """Get all events for a workflow, in order."""
        event_ids = self._events_by_workflow.get(workflow_id, [])
        return [self._events_by_id[eid] for eid in event_ids if eid in self._events_by_id]

    def get_all_events(self) -> List[AuditEvent]:
        """Get all audit events."""
        return list(self._events)

    def get_event_count(self) -> int:
        """Get total number of events."""
        return len(self._events)

    def correct_event(
        self,
        original_event_id: str,
        correction_type: AuditEventType,
        correction_reason: str,
        **kwargs,
    ) -> AuditEvent:
        """Create a correction event.

        Does NOT modify the original event.
        Creates a new event that references the original.
        """
        original = self._events_by_id.get(original_event_id)
        if not original:
            raise ValueError(f"Original event {original_event_id} not found")

        # Create correction event with same context
        correction = AuditEvent(
            event_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            event_type=correction_type,
            workflow_id=original.workflow_id,
            exception_id=original.exception_id,
            case_id=original.case_id,
            candidate_id=original.candidate_id,
            correction_of=original_event_id,
            correction_reason=correction_reason,
            **kwargs,
        )
        return self.record(correction)

    def get_corrections(self, event_id: str) -> List[AuditEvent]:
        """Get all correction events for a given event."""
        return [
            e for e in self._events
            if e.correction_of == event_id
        ]

    def get_events_by_type(
        self,
        event_type: AuditEventType,
        workflow_id: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Get events filtered by type."""
        events = self._events
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        return [e for e in events if e.event_type == event_type]

    def get_events_by_actor(
        self,
        actor_type: ActorType,
        workflow_id: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Get events filtered by actor type."""
        events = self._events
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        return [e for e in events if e.actor_type == actor_type]

    def get_final_outcomes(self, workflow_id: Optional[str] = None) -> List[AuditEvent]:
        """Get events that have a final outcome."""
        events = self._events
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        return [e for e in events if e.final_outcome is not None]
