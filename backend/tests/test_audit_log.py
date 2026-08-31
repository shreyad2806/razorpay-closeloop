"""
Tests for Razorpay CloseLoop Phase 8F — Complete Audit Logging.

Tests audit event creation, immutability, actor model, metadata,
and query capabilities.
"""

import pytest
from app.schemas.audit import (
    ActionMetadata,
    AuditEvent,
    AuditEventType,
    ActorType,
    FinalOutcome,
    GuardrailMetadata,
    ModelMetadata,
    RollbackMetadata,
    VerificationMetadata,
)
from app.services.audit_log import AuditLogService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_event(**overrides) -> AuditEvent:
    """Build a valid audit event."""
    defaults = dict(
        event_id="AUD-001",
        event_type=AuditEventType.WORKFLOW_STARTED,
        workflow_id="WF-001",
        exception_id="EXC-001",
    )
    defaults.update(overrides)
    return AuditEvent(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_event_type_values(self):
        assert AuditEventType.WORKFLOW_STARTED.value == "WORKFLOW_STARTED"
        assert AuditEventType.GUARDRAIL_EVALUATED.value == "GUARDRAIL_EVALUATED"
        assert AuditEventType.EXECUTION_COMPLETED.value == "EXECUTION_COMPLETED"
        assert AuditEventType.ROLLBACK_INITIATED.value == "ROLLBACK_INITIATED"
        assert AuditEventType.CORRECTION_APPLIED.value == "CORRECTION_APPLIED"

    def test_actor_type_values(self):
        assert ActorType.SYSTEM.value == "SYSTEM"
        assert ActorType.AGENT.value == "AGENT"
        assert ActorType.HUMAN.value == "HUMAN"
        assert ActorType.SERVICE.value == "SERVICE"

    def test_final_outcome_values(self):
        assert FinalOutcome.VERIFIED_SUCCESS.value == "VERIFIED_SUCCESS"
        assert FinalOutcome.HUMAN_REJECTED.value == "HUMAN_REJECTED"
        assert FinalOutcome.ROLLED_BACK.value == "ROLLED_BACK"
        assert FinalOutcome.ESCALATED.value == "ESCALATED"
        assert FinalOutcome.UNRESOLVED.value == "UNRESOLVED"

    def test_event_summary(self):
        event = _make_event(
            actor_type=ActorType.SYSTEM,
            final_outcome=FinalOutcome.VERIFIED_SUCCESS,
        )
        s = event.summary()
        assert "WORKFLOW_STARTED" in s
        assert "WF-001" in s
        assert "VERIFIED_SUCCESS" in s


# ─────────────────────────────────────────────────────────────────────────────
# Complete Audit Trail Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCompleteAuditTrail:
    def test_full_workflow_audit_trail(self):
        """Record a complete workflow audit trail."""
        service = AuditLogService()

        # 1. Workflow started
        service.create_event(
            AuditEventType.WORKFLOW_STARTED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
        )

        # 2. Evidence gathered
        service.create_event(
            AuditEventType.EVIDENCE_GATHERED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            evidence_references=["EV-001", "EV-002"],
        )

        # 3. Classification complete
        service.create_event(
            AuditEventType.CLASSIFICATION_COMPLETE, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            decision="FEE_DIFFERENCE",
            model_metadata=ModelMetadata(model_name="xgboost", model_version="1.0"),
        )

        # 4. Guardrail evaluated
        service.create_event(
            AuditEventType.GUARDRAIL_EVALUATED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            decision="AUTO",
            confidence=0.85,
            risk="LOW",
            guardrail_metadata=GuardrailMetadata(
                decision="AUTO",
                confidence=0.85,
                passed_checks=["confidence", "exposure", "evidence"],
                failed_checks=[],
                reason_codes=["ALL_GATES_PASSED"],
            ),
        )

        # 5. Execution completed
        service.create_event(
            AuditEventType.EXECUTION_COMPLETED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SERVICE,
            action_metadata=ActionMetadata(
                resolution_type="APPLY_FEE_CORRECTION",
                requested_adjustment_paise=3000,
                actual_adjustment_paise=3000,
                execution_status="EXECUTED",
                idempotency_key="key-001",
            ),
        )

        # 6. Verification performed
        service.create_event(
            AuditEventType.VERIFICATION_PERFORMED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            verification_metadata=VerificationMetadata(
                difference_before=3000,
                difference_after=0,
                discrepancy_eliminated=True,
                verification_status="PASSED",
            ),
        )

        # 7. Resolution verified
        service.create_event(
            AuditEventType.RESOLUTION_VERIFIED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            final_outcome=FinalOutcome.VERIFIED_SUCCESS,
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 7
        assert events[0].event_type == AuditEventType.WORKFLOW_STARTED
        assert events[-1].final_outcome == FinalOutcome.VERIFIED_SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# Failed Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFailedExecution:
    def test_execution_failure_audit(self):
        """Failed execution produces audit trail."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.EXECUTION_REQUESTED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
        )

        service.create_event(
            AuditEventType.EXECUTION_FAILED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SERVICE,
            error="Adjustment validation failed",
            final_outcome=FinalOutcome.EXECUTION_FAILED,
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 2
        assert events[1].error == "Adjustment validation failed"
        assert events[1].final_outcome == FinalOutcome.EXECUTION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Verification Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationFailure:
    def test_verification_failure_audit(self):
        """Verification failure produces audit trail."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.VERIFICATION_PERFORMED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            verification_metadata=VerificationMetadata(
                difference_before=3000,
                difference_after=1500,
                discrepancy_eliminated=False,
                verification_status="FAILED",
                verification_failure_reason="Discrepancy not eliminated",
            ),
            final_outcome=FinalOutcome.VERIFICATION_FAILED,
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 1
        assert events[0].verification_metadata.verification_status == "FAILED"
        assert events[0].final_outcome == FinalOutcome.VERIFICATION_FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackAudit:
    def test_rollback_audit_trail(self):
        """Rollback produces audit trail."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.ROLLBACK_INITIATED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            rollback_metadata=RollbackMetadata(
                rollback_id="RBK-001",
                rollback_status="ROLLING_BACK",
                reversal_amount_paise=3000,
                rollback_reason="Verification failed",
            ),
        )

        service.create_event(
            AuditEventType.ROLLBACK_COMPLETED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            rollback_metadata=RollbackMetadata(
                rollback_id="RBK-001",
                rollback_status="ROLLED_BACK",
                reversal_amount_paise=3000,
                rollback_verified=True,
            ),
            final_outcome=FinalOutcome.ROLLED_BACK,
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 2
        assert events[1].final_outcome == FinalOutcome.ROLLED_BACK


# ─────────────────────────────────────────────────────────────────────────────
# Human Approval Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanApproval:
    def test_human_approval_audit(self):
        """Human approval produces audit trail."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.HUMAN_REVIEW_REQUESTED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            decision="HUMAN_REVIEW",
            confidence=0.60,
        )

        service.create_event(
            AuditEventType.HUMAN_DECISION_RECEIVED, "WF-001", "EXC-001",
            actor="reviewer-001",
            actor_type=ActorType.HUMAN,
            decision="APPROVED",
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 2
        # First event is SYSTEM
        assert events[0].actor_type == ActorType.SYSTEM
        # Second event is HUMAN
        assert events[1].actor_type == ActorType.HUMAN
        assert events[1].actor == "reviewer-001"


# ─────────────────────────────────────────────────────────────────────────────
# Escalation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEscalation:
    def test_escalation_audit(self):
        """Escalation produces audit trail."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.CASE_ESCALATED, "WF-001", "EXC-001",
            actor="system", actor_type=ActorType.SYSTEM,
            final_outcome=FinalOutcome.ESCALATED,
            error="Conflicting evidence detected",
        )

        events = service.get_workflow_events("WF-001")
        assert len(events) == 1
        assert events[0].final_outcome == FinalOutcome.ESCALATED


# ─────────────────────────────────────────────────────────────────────────────
# Model Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelMetadata:
    def test_model_metadata_recorded(self):
        """Model metadata is preserved in audit event."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.CLASSIFICATION_COMPLETE, "WF-001", "EXC-001",
            model_metadata=ModelMetadata(
                model_name="xgboost",
                model_version="1.0",
                classifier_version="2.1",
                embedding_model_version="3.0",
                retrieval_config={"top_k": 5, "threshold": 0.7},
            ),
        )

        event = service.get_workflow_events("WF-001")[0]
        assert event.model_metadata.model_name == "xgboost"
        assert event.model_metadata.classifier_version == "2.1"
        assert event.model_metadata.retrieval_config["top_k"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Policy Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyMetadata:
    def test_guardrail_metadata_recorded(self):
        """Guardrail metadata is preserved."""
        service = AuditLogService()

        service.create_event(
            AuditEventType.GUARDRAIL_EVALUATED, "WF-001", "EXC-001",
            guardrail_metadata=GuardrailMetadata(
                decision="AUTO",
                confidence=0.85,
                exposure_paise=3000,
                passed_checks=["confidence", "exposure", "evidence"],
                failed_checks=[],
                reason_codes=["ALL_GATES_PASSED"],
                policy_version="1.0",
            ),
        )

        event = service.get_workflow_events("WF-001")[0]
        assert event.guardrail_metadata.decision == "AUTO"
        assert event.guardrail_metadata.exposure_paise == 3000
        assert event.guardrail_metadata.policy_version == "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Immutability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestImmutability:
    def test_events_are_append_only(self):
        """Events cannot be modified after recording."""
        service = AuditLogService()

        event = _make_event(event_id="AUD-001", decision="AUTO")
        service.record(event)

        # Try to record same event ID again — should deduplicate
        event2 = _make_event(event_id="AUD-001", decision="HUMAN_REVIEW")
        result = service.record(event2)

        # Should return original, not modified
        stored = service.get_event("AUD-001")
        assert stored.decision == "AUTO"
        assert result.decision == "AUTO"

    def test_correction_creates_new_event(self):
        """Correction creates a new event, does not modify original."""
        service = AuditLogService()

        original = service.create_event(
            AuditEventType.RESOLUTION_VERIFIED, "WF-001", "EXC-001",
            final_outcome=FinalOutcome.VERIFIED_SUCCESS,
        )

        # Correct the original
        correction = service.correct_event(
            original.event_id,
            AuditEventType.CORRECTION_APPLIED,
            correction_reason="Incorrect verification — discrepancy not actually eliminated",
            final_outcome=FinalOutcome.VERIFICATION_FAILED,
        )

        # Original is unchanged
        stored_original = service.get_event(original.event_id)
        assert stored_original.final_outcome == FinalOutcome.VERIFIED_SUCCESS

        # Correction exists separately
        assert correction.correction_of == original.event_id
        assert correction.final_outcome == FinalOutcome.VERIFICATION_FAILED

        # Both events exist
        assert service.get_event_count() == 2

    def test_corrections_queryable(self):
        """Corrections can be queried for an event."""
        service = AuditLogService()

        original = service.create_event(
            AuditEventType.RESOLUTION_VERIFIED, "WF-001", "EXC-001",
        )

        service.correct_event(
            original.event_id,
            AuditEventType.CORRECTION_APPLIED,
            correction_reason="Wrong outcome",
        )

        corrections = service.get_corrections(original.event_id)
        assert len(corrections) == 1
        assert corrections[0].correction_of == original.event_id

    def test_multiple_corrections(self):
        """Multiple corrections are all preserved."""
        service = AuditLogService()

        original = service.create_event(
            AuditEventType.RESOLUTION_VERIFIED, "WF-001", "EXC-001",
        )

        service.correct_event(original.event_id, AuditEventType.CORRECTION_APPLIED, "First correction")
        service.correct_event(original.event_id, AuditEventType.CORRECTION_APPLIED, "Second correction")

        corrections = service.get_corrections(original.event_id)
        assert len(corrections) == 2

    def test_correction_nonexistent_event(self):
        """Correction of nonexistent event raises error."""
        service = AuditLogService()
        with pytest.raises(ValueError, match="not found"):
            service.correct_event("AUD-NONEXISTENT", AuditEventType.CORRECTION_APPLIED, "reason")


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Event Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateEvents:
    def test_duplicate_event_id_deduplicated(self):
        """Duplicate event IDs are deduplicated when recorded directly."""
        service = AuditLogService()

        # Use record() with same ID to test deduplication
        event1 = _make_event(event_id="AUD-DUP-001")
        event2 = _make_event(event_id="AUD-DUP-001")
        service.record(event1)
        service.record(event2)

        assert service.get_event_count() == 1

    def test_different_workflows_independent(self):
        """Different workflows have independent audit trails."""
        service = AuditLogService()

        service.create_event(AuditEventType.WORKFLOW_STARTED, "WF-001", "EXC-001")
        service.create_event(AuditEventType.WORKFLOW_STARTED, "WF-002", "EXC-002")

        assert service.get_event_count() == 2
        assert len(service.get_workflow_events("WF-001")) == 1
        assert len(service.get_workflow_events("WF-002")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Query Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQueries:
    def test_get_events_by_type(self):
        """Events can be queried by type."""
        service = AuditLogService()

        service.create_event(AuditEventType.WORKFLOW_STARTED, "WF-001", "EXC-001")
        service.create_event(AuditEventType.GUARDRAIL_EVALUATED, "WF-001", "EXC-001")
        service.create_event(AuditEventType.EXECUTION_COMPLETED, "WF-001", "EXC-001")

        guardrail_events = service.get_events_by_type(AuditEventType.GUARDRAIL_EVALUATED)
        assert len(guardrail_events) == 1

    def test_get_events_by_actor(self):
        """Events can be queried by actor type."""
        service = AuditLogService()

        service.create_event(AuditEventType.WORKFLOW_STARTED, "WF-001", "EXC-001", actor_type=ActorType.SYSTEM)
        service.create_event(AuditEventType.HUMAN_DECISION_RECEIVED, "WF-001", "EXC-001", actor_type=ActorType.HUMAN)

        human_events = service.get_events_by_actor(ActorType.HUMAN)
        assert len(human_events) == 1
        assert human_events[0].event_type == AuditEventType.HUMAN_DECISION_RECEIVED

    def test_get_final_outcomes(self):
        """Events with final outcomes can be queried."""
        service = AuditLogService()

        service.create_event(AuditEventType.WORKFLOW_STARTED, "WF-001", "EXC-001")
        service.create_event(AuditEventType.RESOLUTION_VERIFIED, "WF-001", "EXC-001", final_outcome=FinalOutcome.VERIFIED_SUCCESS)

        outcomes = service.get_final_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].final_outcome == FinalOutcome.VERIFIED_SUCCESS

    def test_get_all_events(self):
        """All events can be retrieved."""
        service = AuditLogService()

        for i in range(5):
            service.record(_make_event(event_id=f"AUD-{i:03d}"))

        assert service.get_event_count() == 5
        assert len(service.get_all_events()) == 5
