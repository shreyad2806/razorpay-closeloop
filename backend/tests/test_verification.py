"""
Tests for Razorpay CloseLoop Phase 7H — Verification Service.

Tests verification checks, stale-state detection, and failure behavior.
"""

import pytest
from app.services.verification import VerificationService
from app.schemas.verification import (
    CheckStatus,
    VerificationAction,
    VerificationCheck,
    VerificationConfig,
    VerificationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot(**overrides) -> dict:
    """Build a verification snapshot."""
    base = {
        "exception_id": "EXC-001",
        "candidate_id": "CAND-001",
        "exception_exists": True,
        "candidate_exists": True,
        "evidence_records": ["EV-001", "EV-002"],
        "expected_amount": 50000,
        "difference": 3000,
        "decision": "AUTO",
        "state_version": 1,
        "reconciliation_hash": None,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = VerificationConfig()
        assert config.check_exception_exists is True
        assert config.check_candidate_exists is True
        assert config.check_evidence_exists is True
        assert config.check_financial_consistency is True
        assert config.check_guardrail_valid is True
        assert config.check_no_conflicting_update is True
        assert config.require_all_passed is True

    def test_custom_config(self):
        config = VerificationConfig(
            check_exception_exists=False,
            check_financial_consistency=False,
        )
        assert config.check_exception_exists is False
        assert config.check_financial_consistency is False
        assert config.check_candidate_exists is True


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_check_status_values(self):
        assert CheckStatus.PASSED.value == "PASSED"
        assert CheckStatus.FAILED.value == "FAILED"
        assert CheckStatus.SKIPPED.value == "SKIPPED"

    def test_verification_action_values(self):
        assert VerificationAction.VERIFIED.value == "VERIFIED"
        assert VerificationAction.STALE_STATE.value == "STALE_STATE"
        assert VerificationAction.VERIFICATION_FAILED.value == "VERIFICATION_FAILED"

    def test_verification_check(self):
        check = VerificationCheck(
            check_name="test_check",
            status=CheckStatus.PASSED,
            expected=True,
            actual=True,
        )
        assert check.check_name == "test_check"
        assert check.status == CheckStatus.PASSED

    def test_verification_result_summary(self):
        result = VerificationResult(
            exception_id="EXC-001",
            action=VerificationAction.VERIFIED,
            passed=True,
            checks=[
                VerificationCheck(check_name="c1", status=CheckStatus.PASSED),
                VerificationCheck(check_name="c2", status=CheckStatus.PASSED),
            ],
            amount_consistent=True,
            evidence_exists=True,
        )
        summary = result.summary()
        assert "VERIFIED" in summary
        assert "2 passed" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Core Verification Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerification:
    def test_all_checks_pass(self):
        """Happy path: all checks pass when snapshot == current."""
        service = VerificationService()
        snapshot = _make_snapshot()
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        assert result.passed is True
        assert result.action == VerificationAction.VERIFIED
        assert len(result.checks) > 0
        assert all(c.status == CheckStatus.PASSED for c in result.checks)
        assert result.stale_checks == []
        assert result.amount_consistent is True
        assert result.evidence_exists is True
        assert result.candidate_exists is True

    def test_no_exception(self):
        """Exception no longer exists → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(exception_exists=True)
        current = _make_snapshot(exception_exists=False)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.STALE_STATE
        assert "exception_exists" in result.stale_checks

    def test_no_candidate(self):
        """Candidate no longer exists → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(candidate_exists=True)
        current = _make_snapshot(candidate_exists=False)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.STALE_STATE
        assert "candidate_exists" in result.stale_checks
        assert result.candidate_exists is False

    def test_evidence_removed(self):
        """Evidence record removed → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(evidence_records=["EV-001", "EV-002", "EV-003"])
        current = _make_snapshot(evidence_records=["EV-001", "EV-002"])  # EV-003 removed
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.STALE_STATE
        assert "evidence_exists" in result.stale_checks
        assert result.evidence_exists is False
        # Changed records should show the removal
        evidence_changes = [r for r in result.changed_records if r["type"] == "evidence"]
        assert len(evidence_changes) == 1
        assert evidence_changes[0]["id"] == "EV-003"

    def test_evidence_all_removed(self):
        """All evidence removed → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(evidence_records=["EV-001"])
        current = _make_snapshot(evidence_records=[])
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert "evidence_exists" in result.stale_checks

    def test_financial_amount_changed(self):
        """Expected amount changed → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(expected_amount=50000, difference=3000)
        current = _make_snapshot(expected_amount=52000, difference=5000)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.STALE_STATE
        assert "financial_consistent" in result.stale_checks
        assert result.amount_consistent is False
        assert result.expected_amount_at_recommendation == 50000
        assert result.expected_amount_now == 52000
        # Should have a financial change record
        fin_changes = [r for r in result.changed_records if r["type"] == "financial"]
        assert len(fin_changes) == 1

    def test_financial_difference_changed(self):
        """Difference changed but expected_amount same → STALE_STATE."""
        service = VerificationService()
        snapshot = _make_snapshot(expected_amount=50000, difference=3000)
        current = _make_snapshot(expected_amount=50000, difference=1500)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert "financial_consistent" in result.stale_checks

    def test_guardrail_decision_changed(self):
        """Guardrail decision changed → VERIFICATION_FAILED."""
        service = VerificationService()
        snapshot = _make_snapshot(decision="AUTO")
        current = _make_snapshot(decision="HUMAN_REVIEW")
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.VERIFICATION_FAILED
        assert "guardrail_valid" in result.stale_checks
        guardrail_changes = [r for r in result.changed_records if r["type"] == "guardrail"]
        assert len(guardrail_changes) == 1

    def test_conflicting_update(self):
        """State version changed with reconciliation change → VERIFICATION_FAILED."""
        service = VerificationService()
        snapshot = _make_snapshot(state_version=1, reconciliation_hash="hash_a")
        current = _make_snapshot(state_version=2, reconciliation_hash="hash_b")
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.VERIFICATION_FAILED
        assert "no_conflicting_update" in result.stale_checks


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleFailures:
    def test_multiple_stale_checks(self):
        """Multiple staleness checks fail simultaneously."""
        service = VerificationService()
        snapshot = _make_snapshot(
            exception_exists=True,
            evidence_records=["EV-001"],
            expected_amount=50000,
        )
        current = _make_snapshot(
            exception_exists=False,
            evidence_records=[],
            expected_amount=0,
        )
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is False
        assert result.action == VerificationAction.STALE_STATE
        assert len(result.stale_checks) >= 2
        assert "exception_exists" in result.stale_checks
        assert "evidence_exists" in result.stale_checks
        assert "financial_consistent" in result.stale_checks

    def test_staleness_overrides_guardrail_change(self):
        """Staleness checks take priority over guardrail changes."""
        service = VerificationService()
        snapshot = _make_snapshot(
            exception_exists=True,
            expected_amount=50000,
            decision="AUTO",
        )
        current = _make_snapshot(
            exception_exists=False,
            expected_amount=0,
            decision="HUMAN_REVIEW",
        )
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        # Should be STALE_STATE, not VERIFICATION_FAILED
        assert result.action == VerificationAction.STALE_STATE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Override Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigOverrides:
    def test_skip_exception_check(self):
        """Disabled exception check → not evaluated."""
        config = VerificationConfig(check_exception_exists=False)
        service = VerificationService(config=config)
        snapshot = _make_snapshot(exception_exists=True)
        current = _make_snapshot(exception_exists=False)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        check_names = [c.check_name for c in result.checks]
        assert "exception_exists" not in check_names

    def test_skip_financial_check(self):
        """Disabled financial check → amount change not detected."""
        config = VerificationConfig(check_financial_consistency=False)
        service = VerificationService(config=config)
        snapshot = _make_snapshot(expected_amount=50000)
        current = _make_snapshot(expected_amount=99999)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        check_names = [c.check_name for c in result.checks]
        assert "financial_consistent" not in check_names
        assert result.amount_consistent is True  # not checked

    def test_skip_all_checks(self):
        """All checks disabled → always VERIFIED."""
        config = VerificationConfig(
            check_exception_exists=False,
            check_candidate_exists=False,
            check_evidence_exists=False,
            check_financial_consistency=False,
            check_guardrail_valid=False,
            check_no_conflicting_update=False,
        )
        service = VerificationService(config=config)
        snapshot = _make_snapshot()
        current = _make_snapshot(exception_exists=False, expected_amount=0)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is True
        assert result.action == VerificationAction.VERIFIED


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_current_state_uses_snapshot(self):
        """When current_state is None, uses snapshot (no change)."""
        service = VerificationService()
        snapshot = _make_snapshot()
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        assert result.passed is True

    def test_empty_evidence_records(self):
        """Empty evidence records on both sides → passes."""
        service = VerificationService()
        snapshot = _make_snapshot(evidence_records=[])
        current = _make_snapshot(evidence_records=[])
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is True

    def test_missing_candidate_in_snapshot(self):
        """No candidate in snapshot → candidate check passes (exists=True default)."""
        service = VerificationService()
        snapshot = _make_snapshot(candidate_id=None, candidate_exists=True)
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        assert result.passed is True
        assert result.candidate_exists is True

    def test_zero_financial_amounts(self):
        """Zero amounts are consistent."""
        service = VerificationService()
        snapshot = _make_snapshot(expected_amount=0, difference=0)
        current = _make_snapshot(expected_amount=0, difference=0)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        assert result.passed is True
        assert result.amount_consistent is True

    def test_verification_result_has_metadata(self):
        """Verification result contains timestamp and metadata."""
        service = VerificationService()
        snapshot = _make_snapshot()
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        assert result.exception_id == "EXC-001"
        assert result.verified_at is not None
        assert result.verified_by == "verification_service"
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0

    def test_candidate_id_preserved(self):
        """Candidate ID is preserved in result."""
        service = VerificationService()
        snapshot = _make_snapshot(candidate_id="CAND-042")
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        assert result.candidate_id == "CAND-042"

    def test_summary_format(self):
        """Summary contains key information."""
        service = VerificationService()
        snapshot = _make_snapshot()
        result = service.verify(exception_id="EXC-001", state_snapshot=snapshot)

        summary = result.summary()
        assert "VERIFIED" in summary
        assert "Amount consistent" in summary
        assert "Evidence exists" in summary

    def test_summary_with_failure(self):
        """Summary shows failure information."""
        service = VerificationService()
        snapshot = _make_snapshot(exception_exists=True, expected_amount=50000)
        current = _make_snapshot(exception_exists=False, expected_amount=0)
        result = service.verify(
            exception_id="EXC-001",
            state_snapshot=snapshot,
            current_state=current,
        )

        summary = result.summary()
        assert "STALE_STATE" in summary
        assert "failed" in summary.lower()
