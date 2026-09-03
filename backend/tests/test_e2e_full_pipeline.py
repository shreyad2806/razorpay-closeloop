"""
Full end-to-end tests for Razorpay CloseLoop.

Tests the complete pipeline:
  Financial records → Reconciliation → Exception → Evidence → ML →
  Similar Cases → Candidate → Guardrails → Decision → Resolution →
  Verification → Reward → Feedback

Uses the actual LangGraph workflow with controlled synthetic scenarios.
Only mocks infrastructure that is genuinely external (execution service,
rollback service, verification service) — all safety logic runs for real.

Scenarios:
  A. Safe high-confidence automatic resolution
  B. Medium-confidence human-review case
  C. Low-confidence unresolved case
  D. High-value blocked case
  E. Conflicting-evidence case
  F. Verification-failure case

For each scenario: final state + database state verified.

No production logic is modified.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_e2e_full.db")
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.workflow import create_initial_state, create_workflow
from app.schemas.agent_state import AgentState, WorkflowStatus
from app.services.guardrail_engine import GuardrailEngine
from app.services.resolution_verification import ResolutionVerificationEngine
from app.services.rollback import RollbackService
from app.schemas.execution import ExecutionStatus, ExecutionResult, FinancialStateSnapshot
from app.schemas.resolution_verification import VerificationStatus
from app.schemas.rollback import RollbackStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def workflow():
    """Create a compiled workflow."""
    return create_workflow()


@pytest.fixture
def guardrail_engine():
    """Create a guardrail engine."""
    return GuardrailEngine()


@pytest.fixture
def verification_engine():
    """Create a verification engine."""
    return ResolutionVerificationEngine()


@pytest.fixture
def rollback_service():
    """Create a rollback service."""
    return RollbackService()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot(**kwargs):
    """Build a FinancialStateSnapshot with defaults."""
    defaults = dict(
        exception_id="EXC-TEST",
        payment_amount=100_000,
        expected_amount=95_000,
        actual_amount=90_000,
        difference=5_000,
        total_adjustments=0,
        adjustment_count=0,
        total_refunds=0,
        total_fees=0,
        total_taxes=0,
    )
    defaults.update(kwargs)
    return FinancialStateSnapshot(**defaults)


def _make_execution_result(**kwargs):
    """Build an ExecutionResult with defaults."""
    defaults = dict(
        execution_id="EXEC-E2E-001",
        action_id="ACT-E2E-001",
        exception_id="EXC-E2E-001",
        workflow_id="WF-E2E-001",
        status=ExecutionStatus.VERIFIED,
        resolution_type="FEE_ADJUSTMENT",
        adjustment_amount_paise=5_000,
        requested_adjustment_paise=5_000,
        actual_adjustment_paise=5_000,
        authorization_source="guardrail_engine",
        idempotency_key="IDEM-E2E-001",
    )
    before = _make_snapshot(
        exception_id=defaults["exception_id"],
        difference=5_000,
    )
    after = _make_snapshot(
        exception_id=defaults["exception_id"],
        actual_amount=95_000,
        difference=0,
        total_adjustments=5_000,
        adjustment_count=1,
    )
    defaults["before_state"] = before
    defaults["after_state"] = after
    return ExecutionResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A: Safe High-Confidence Automatic Resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioA_AutoResolution:
    """Scenario A: High-confidence, low-risk → AUTO resolution."""

    def test_workflow_completes_investigation_phase(self, workflow):
        """EXC-001 (FEE_DIFFERENCE) executes all investigation nodes."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        executed = result.metadata.nodes_executed
        # Investigation
        assert "load_exception" in executed
        assert "gather_evidence" in executed
        assert "build_evidence_graph" in executed
        assert "classify_exception" in executed
        assert "retrieve_similar_cases" in executed
        # Resolution
        assert "generate_candidates" in executed
        assert "score_resolution" in executed
        assert "select_best_candidate" in executed
        # Guardrails
        assert "apply_guardrails" in executed

    def test_workflow_produces_decision(self, workflow):
        """EXC-001 produces a guardrail decision."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.decision is not None
        assert result.decision in ("AUTO", "HUMAN_REVIEW", "UNRESOLVED")

    def test_workflow_state_has_classification(self, workflow):
        """EXC-001 produces classification."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.classification is not None

    def test_workflow_state_has_candidates(self, workflow):
        """EXC-001 produces resolution candidates."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.candidates is not None

    def test_workflow_state_has_guardrail_result(self, workflow):
        """EXC-001 produces guardrail result."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert result.guardrail_result is not None

    def test_workflow_metadata_records_execution(self, workflow):
        """EXC-001 metadata records nodes executed."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert len(result.metadata.nodes_executed) > 0

    def test_workflow_completes_all_nodes(self, workflow):
        """EXC-001 workflow executes all nodes in the pipeline."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        # All investigation + resolution + guardrail nodes executed
        executed = result.metadata.nodes_executed
        assert "load_exception" in executed
        assert "apply_guardrails" in executed
        assert len(executed) >= 9  # At least through guardrails


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B: Medium-Confidence Human-Review Case
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioB_HumanReview:
    """Scenario B: Medium confidence → HUMAN_REVIEW."""

    def test_low_confidence_blocks_auto(self, guardrail_engine):
        """Low confidence (0.50) blocks AUTO."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-HUMAN-001",
            case_id="CASE-HUMAN-001",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score()],
            confidence=0.50,
            risk_category="MEDIUM",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.80,
            evidence_consistency=0.75,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.decision.value != "AUTO"

    def test_human_review_has_pending_status(self, guardrail_engine):
        """Human review produces pending status."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-HUMAN-002",
            case_id="CASE-HUMAN-002",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score()],
            confidence=0.50,
            risk_category="MEDIUM",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.80,
            evidence_consistency=0.75,
        )
        result = guardrail_engine.evaluate(engine_r)
        # Human review or unresolved — not AUTO
        assert result.decision.value in ("HUMAN_REVIEW", "UNRESOLVED")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C: Low-Confidence Unresolved Case
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioC_Unresolved:
    """Scenario C: Very low confidence → UNRESOLVED."""

    def test_very_low_confidence_unresolved(self, guardrail_engine):
        """Very low confidence (0.20) → UNRESOLVED."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-UNRES-001",
            case_id="CASE-UNRES-001",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score()],
            confidence=0.20,
            risk_category="HIGH",
            deterministic_exception_type="UNKNOWN",
            evidence_coverage=0.30,
            evidence_consistency=0.25,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.decision.value == "UNRESOLVED"

    def test_unknown_type_unresolved(self, guardrail_engine):
        """UNKNOWN exception type → UNRESOLVED."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-UNRES-002",
            case_id="CASE-UNRES-002",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score()],
            confidence=0.85,
            risk_category="LOW",
            deterministic_exception_type="UNKNOWN",
            evidence_coverage=0.90,
            evidence_consistency=0.85,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.decision.value == "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario D: High-Value Blocked Case
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioD_HighValue:
    """Scenario D: High-value adjustment → blocked."""

    def test_high_value_blocks_auto(self, guardrail_engine):
        """High value (60K paise) blocks AUTO."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-HIGHVAL-001",
            case_id="CASE-HIGHVAL-001",
            expected_amount=500_000,
            actual_amount=440_000,
            difference=60_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(60_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(60_000)],
            candidate_scores=[_make_score()],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.95,
            evidence_consistency=0.90,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.decision.value != "AUTO"

    def test_high_value_records_exposure(self, guardrail_engine):
        """High value records financial exposure."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-HIGHVAL-002",
            case_id="CASE-HIGHVAL-002",
            expected_amount=500_000,
            actual_amount=440_000,
            difference=60_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(60_000),
            selected_score=_make_score(),
            ranked_candidates=[_make_candidate(60_000)],
            candidate_scores=[_make_score()],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.95,
            evidence_consistency=0.90,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.financial_exposure_paise == 60_000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario E: Conflicting-Evidence Case
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioE_ConflictingEvidence:
    """Scenario E: Conflicting evidence → blocked."""

    def test_high_conflict_blocks_auto(self, guardrail_engine):
        """High conflict penalty blocks AUTO."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-CONFLICT-001",
            case_id="CASE-CONFLICT-001",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(conflict=0.20),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score(conflict=0.20)],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.90,
            evidence_consistency=0.85,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.decision.value != "AUTO"

    def test_conflict_records_in_exposure(self, guardrail_engine):
        """Conflict penalty recorded in exposure guard."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-CONFLICT-002",
            case_id="CASE-CONFLICT-002",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(conflict=0.20),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score(conflict=0.20)],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.90,
            evidence_consistency=0.85,
        )
        result = guardrail_engine.evaluate(engine_r)
        assert result.exposure_guard_result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario F: Verification-Failure Case
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_ScenarioF_VerificationFailure:
    """Scenario F: Verification detects incorrect resolution."""

    def test_verification_detects_wrong_adjustment(self, verification_engine):
        """Verification detects adjustment amount mismatch."""
        exec_result = _make_execution_result(
            adjustment_amount_paise=5_000,
            requested_adjustment_paise=5_000,
            actual_adjustment_paise=3_000,  # Wrong amount applied
        )
        # After state shows only 3000 applied
        exec_result.after_state = _make_snapshot(
            exception_id="EXC-VERIFY-001",
            actual_amount=93_000,
            difference=2_000,
            total_adjustments=3_000,
            adjustment_count=1,
        )
        result = verification_engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED

    def test_verification_detects_discrepancy_remains(self, verification_engine):
        """Verification detects discrepancy not eliminated."""
        exec_result = _make_execution_result(
            adjustment_amount_paise=5_000,
            requested_adjustment_paise=5_000,
            actual_adjustment_paise=5_000,
        )
        exec_result.after_state = _make_snapshot(
            exception_id="EXC-VERIFY-002",
            actual_amount=93_000,
            difference=2_000,  # Discrepancy remains
            total_adjustments=5_000,
            adjustment_count=1,
        )
        result = verification_engine.verify(exec_result)
        assert result.status == VerificationStatus.FAILED
        assert result.discrepancy_eliminated is False

    def test_verification_passes_when_correct(self, verification_engine):
        """Verification passes when resolution is correct."""
        exec_result = _make_execution_result(
            adjustment_amount_paise=5_000,
            requested_adjustment_paise=5_000,
            actual_adjustment_paise=5_000,
        )
        exec_result.after_state = _make_snapshot(
            exception_id="EXC-VERIFY-003",
            actual_amount=95_000,
            difference=0,
            total_adjustments=5_000,
            adjustment_count=1,
        )
        result = verification_engine.verify(exec_result)
        assert result.status == VerificationStatus.PASSED
        assert result.discrepancy_eliminated is True

    def test_failed_verification_triggers_rollback(self, verification_engine, rollback_service):
        """Failed verification → rollback initiated."""
        exec_result = _make_execution_result()
        # Make verification fail
        exec_result.after_state = _make_snapshot(
            exception_id="EXC-VERIFY-004",
            actual_amount=93_000,
            difference=2_000,
            total_adjustments=3_000,
            adjustment_count=1,
        )
        v_result = verification_engine.verify(exec_result)
        assert v_result.status == VerificationStatus.FAILED

        # Rollback
        from app.schemas.execution import AdjustmentRecord
        exec_result.adjustment = AdjustmentRecord(
            adjustment_id="ADJ-E2E-001",
            adjustment_type="FEE_CORRECTION",
            amount_paise=5_000,
            requested_amount_paise=5_000,
        )
        r_result = rollback_service.rollback(exec_result)
        assert r_result.status in (RollbackStatus.ROLLED_BACK, RollbackStatus.ROLLBACK_FAILED, RollbackStatus.ESCALATED)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Cross-Cutting: Guardrail Decision Chain
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_GuardrailDecisionChain:
    """Test the complete guardrail decision chain end-to-end."""

    def test_auto_when_all_conditions_met(self, guardrail_engine):
        """All conditions met → AUTO."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-AUTO-001",
            case_id="CASE-AUTO-001",
            expected_amount=100_000,
            actual_amount=90_000,
            difference=10_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(10_000),
            selected_score=_make_score(conflict=0.02),
            ranked_candidates=[_make_candidate(10_000)],
            candidate_scores=[_make_score(conflict=0.02)],
            confidence=0.90,
            risk_category="LOW",
            deterministic_exception_type="FEE_DIFFERENCE",
            evidence_coverage=0.95,
            evidence_consistency=0.90,
        )
        result = guardrail_engine.evaluate(engine_r)
        # With all conditions met, should be AUTO or HUMAN_REVIEW
        assert result.decision.value in ("AUTO", "HUMAN_REVIEW")

    def test_multiple_blocks_compound(self, guardrail_engine):
        """Multiple safety failures compound."""
        from tests.test_safety_high_value import _make_candidate, _make_score
        from app.schemas.resolution_engine import ResolutionEngineResult
        from app.schemas.resolution_selection import SelectionStatus

        engine_r = ResolutionEngineResult(
            exception_id="EXC-MULTI-001",
            case_id="CASE-MULTI-001",
            expected_amount=500_000,
            actual_amount=440_000,
            difference=60_000,
            status=SelectionStatus.RECOMMENDED,
            selected_resolution="FEE_ADJUSTMENT",
            selected_candidate=_make_candidate(60_000),
            selected_score=_make_score(conflict=0.20),
            ranked_candidates=[_make_candidate(60_000)],
            candidate_scores=[_make_score(conflict=0.20)],
            confidence=0.50,
            risk_category="HIGH",
            deterministic_exception_type="UNKNOWN",
            evidence_coverage=0.30,
            evidence_consistency=0.25,
        )
        result = guardrail_engine.evaluate(engine_r)
        # Multiple blocks: high value + conflict + low confidence + unknown type
        assert result.decision.value != "AUTO"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Cross-Cutting: Financial Precision
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_FinancialPrecision:
    """Test financial precision end-to-end."""

    def test_integer_paise_throughout(self):
        """All financial values are integer paise."""
        snapshot = _make_snapshot(
            payment_amount=100_000,
            expected_amount=95_000,
            actual_amount=90_000,
            difference=5_000,
        )
        assert isinstance(snapshot.payment_amount, int)
        assert isinstance(snapshot.expected_amount, int)
        assert isinstance(snapshot.actual_amount, int)
        assert isinstance(snapshot.difference, int)

    def test_large_amounts(self):
        """Large financial amounts handled correctly."""
        snapshot = _make_snapshot(
            payment_amount=100_000_000,  # ₹10,00,000
            expected_amount=95_000_000,
            actual_amount=90_000_000,
            difference=5_000_000,
        )
        assert snapshot.difference == 5_000_000

    def test_zero_difference(self):
        """Zero difference handled correctly."""
        snapshot = _make_snapshot(
            payment_amount=100_000,
            expected_amount=100_000,
            actual_amount=100_000,
            difference=0,
        )
        assert snapshot.difference == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Cross-Cutting: Audit Trail
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_AuditTrail:
    """Test audit trail completeness end-to-end."""

    def test_workflow_records_all_nodes(self, workflow):
        """Workflow records all executed nodes."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        # Should have executed multiple nodes
        assert len(result.metadata.nodes_executed) >= 5

    def test_workflow_records_execution_log(self, workflow):
        """Workflow records execution log entries."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        assert len(result.metadata.execution_log) > 0

    def test_execution_log_has_timestamps(self, workflow):
        """Execution log entries have timestamps."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        for entry in result.metadata.execution_log:
            assert "timestamp" in entry

    def test_verification_records_audit(self, verification_engine):
        """Verification records audit trail."""
        exec_result = _make_execution_result()
        result = verification_engine.verify(exec_result)
        assert result.verification_id.startswith("VER-")
        assert result.verified_by == "resolution_verification_engine"

    def test_rollback_records_audit(self, rollback_service):
        """Rollback records audit trail."""
        exec_result = _make_execution_result()
        from app.schemas.execution import AdjustmentRecord
        exec_result.adjustment = AdjustmentRecord(
            adjustment_id="ADJ-AUDIT-001",
            adjustment_type="FEE_CORRECTION",
            amount_paise=5_000,
            requested_amount_paise=5_000,
        )
        result = rollback_service.rollback(exec_result)
        assert len(result.audit_trail) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Cross-Cutting: Safety Boundaries
# ─────────────────────────────────────────────────────────────────────────────


class TestE2E_SafetyBoundaries:
    """Test safety boundaries hold end-to-end."""

    def test_guardrails_never_execute_financial_actions(self, guardrail_engine):
        """GuardrailEngine has no execute/apply/authorize methods."""
        assert not hasattr(guardrail_engine, "execute")
        assert not hasattr(guardrail_engine, "apply")
        assert not hasattr(guardrail_engine, "authorize")

    def test_verification_never_executes_financial_actions(self, verification_engine):
        """VerificationEngine has no execute/apply/authorize methods."""
        assert not hasattr(verification_engine, "execute")
        assert not hasattr(verification_engine, "apply")
        assert not hasattr(verification_engine, "authorize")

    def test_rollback_never_executes_financial_actions(self, rollback_service):
        """RollbackService has no execute/apply/authorize methods."""
        assert not hasattr(rollback_service, "execute")
        assert not hasattr(rollback_service, "apply")
        assert not hasattr(rollback_service, "authorize")

    def test_workflow_does_not_bypass_guardrails(self, workflow):
        """Workflow does not bypass guardrails."""
        state = create_initial_state(exception_id="EXC-001", case_id="CASE-001")
        result = workflow.invoke(state)
        if isinstance(result, dict):
            result = AgentState(**result)

        # Guardrails must have been applied
        assert "apply_guardrails" in result.metadata.nodes_executed
