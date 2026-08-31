"""
Verification Service for Razorpay CloseLoop Phase 7H.

Independently verifies that a proposed resolution is still valid
before any future financial action is considered successful.

Checks:
- Exception still exists
- Candidate still exists
- Evidence still exists
- Financial adjustment remains consistent
- Guardrail decision remains valid
- No conflicting update occurred
- State has not become stale

IMPORTANT:
This service does NOT execute financial actions.
It only verifies whether the recommendation is still valid.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.schemas.verification import (
    CheckStatus,
    VerificationAction,
    VerificationCheck,
    VerificationConfig,
    VerificationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Verification Service
# ─────────────────────────────────────────────────────────────────────────────


class VerificationService:
    """Deterministic verification of proposed resolutions.

    Verifies that nothing has changed since the recommendation
    that would invalidate it.
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()

    def verify(
        self,
        exception_id: str,
        state_snapshot: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify a proposed resolution is still valid.

        Args:
            exception_id: Exception being verified
            state_snapshot: State at time of recommendation
            current_state: Current state (if None, uses snapshot — no change)

        Returns:
            VerificationResult with all checks
        """
        start = time.perf_counter()
        checks: List[VerificationCheck] = []
        stale_checks: List[str] = []
        changed_records: List[Dict[str, Any]] = []

        # Use current_state if provided, otherwise snapshot (no change)
        if current_state is None:
            current_state = state_snapshot

        # ── Check 1: Exception exists ──
        if self.config.check_exception_exists:
            check = self._check_exception_exists(exception_id, state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)

        # ── Check 2: Candidate exists ──
        candidate_id = state_snapshot.get("candidate_id")
        if self.config.check_candidate_exists:
            check = self._check_candidate_exists(candidate_id, state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)

        # ── Check 3: Evidence exists ──
        if self.config.check_evidence_exists:
            check = self._check_evidence_exists(state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)
                # Record what evidence changed
                snap_ev = set((state_snapshot.get("evidence_records") or []))
                cur_ev = set((current_state.get("evidence_records") or []))
                for ev_id in snap_ev - cur_ev:
                    changed_records.append({"type": "evidence", "id": ev_id, "change": "removed"})

        # ── Check 4: Financial consistency ──
        amount_consistent = True
        expected_at_rec = state_snapshot.get("expected_amount")
        expected_now = current_state.get("expected_amount")
        if self.config.check_financial_consistency:
            check = self._check_financial_consistency(state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)
                amount_consistent = False
                changed_records.append({
                    "type": "financial",
                    "field": "expected_amount",
                    "old": expected_at_rec,
                    "new": expected_now,
                })

        # ── Check 5: Guardrail decision valid ──
        if self.config.check_guardrail_valid:
            check = self._check_guardrail_valid(state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)
                changed_records.append({
                    "type": "guardrail",
                    "old_decision": state_snapshot.get("decision"),
                    "new_decision": current_state.get("decision"),
                })

        # ── Check 6: No conflicting update ──
        if self.config.check_no_conflicting_update:
            check = self._check_no_conflicting_update(state_snapshot, current_state)
            checks.append(check)
            if check.status == CheckStatus.FAILED:
                stale_checks.append(check.check_name)

        # ── Determine overall result ──
        failed_checks = [c for c in checks if c.status == CheckStatus.FAILED]
        candidate_id = state_snapshot.get("candidate_id")

        if failed_checks:
            has_staleness = any(
                c.check_name in ("exception_exists", "candidate_exists", "evidence_exists", "financial_consistent")
                for c in failed_checks
            )
            action = VerificationAction.STALE_STATE if has_staleness else VerificationAction.VERIFICATION_FAILED
            passed = False
        else:
            action = VerificationAction.VERIFIED
            passed = True

        elapsed = (time.perf_counter() - start) * 1000

        return VerificationResult(
            exception_id=exception_id,
            candidate_id=candidate_id,
            action=action,
            passed=passed,
            checks=checks,
            stale_checks=stale_checks,
            changed_records=changed_records,
            expected_amount_at_recommendation=expected_at_rec,
            expected_amount_now=expected_now,
            amount_consistent=amount_consistent,
            evidence_exists=not any(c.check_name == "evidence_exists" and c.status == CheckStatus.FAILED for c in checks),
            candidate_exists=not any(c.check_name == "candidate_exists" and c.status == CheckStatus.FAILED for c in checks),
            verified_at=datetime.utcnow(),
            verified_by="verification_service",
            elapsed_ms=round(elapsed, 2),
        )

    # ── Individual checks ──

    def _check_exception_exists(
        self,
        exception_id: str,
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Verify the exception still exists in the current state.

        If snapshot says exception existed but current says it doesn't → FAIL.
        If both say it doesn't exist → PASS (was never there).
        """
        snap_has = snapshot.get("exception_exists", True)
        cur_has = current.get("exception_exists", True)

        if snap_has and not cur_has:
            return VerificationCheck(
                check_name="exception_exists",
                status=CheckStatus.FAILED,
                expected=True,
                actual=cur_has,
                message=f"Exception {exception_id} was present at recommendation but no longer exists",
            )

        return VerificationCheck(
            check_name="exception_exists",
            status=CheckStatus.PASSED,
            expected=snap_has,
            actual=cur_has,
        )

    def _check_candidate_exists(
        self,
        candidate_id: Optional[str],
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Verify the candidate still exists in the current state.

        If the snapshot says candidate existed but current says it doesn't → FAIL.
        If both say it doesn't exist → PASS (it was never there).
        """
        snap_has = snapshot.get("candidate_exists", True)
        cur_has = current.get("candidate_exists", True)

        # Candidate was there at recommendation time but is gone now
        if snap_has and not cur_has:
            return VerificationCheck(
                check_name="candidate_exists",
                status=CheckStatus.FAILED,
                expected=True,
                actual=cur_has,
                message=f"Candidate {candidate_id} was present at recommendation but no longer exists",
            )

        # Candidate exists (or never existed in either snapshot/current)
        return VerificationCheck(
            check_name="candidate_exists",
            status=CheckStatus.PASSED,
            expected=snap_has,
            actual=cur_has,
        )

    def _check_evidence_exists(
        self,
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Verify evidence records still exist."""
        snap_ev = set(snapshot.get("evidence_records") or [])
        cur_ev = set(current.get("evidence_records") or [])

        missing = snap_ev - cur_ev
        if missing:
            return VerificationCheck(
                check_name="evidence_exists",
                status=CheckStatus.FAILED,
                expected=len(snap_ev),
                actual=len(cur_ev),
                message=f"Evidence records removed: {sorted(missing)}",
            )

        return VerificationCheck(
            check_name="evidence_exists",
            status=CheckStatus.PASSED,
            expected=len(snap_ev),
            actual=len(cur_ev),
        )

    def _check_financial_consistency(
        self,
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Verify financial amounts haven't changed since recommendation."""
        snap_amount = snapshot.get("expected_amount")
        cur_amount = current.get("expected_amount")
        snap_diff = snapshot.get("difference")
        cur_diff = current.get("difference")

        if snap_amount != cur_amount or snap_diff != cur_diff:
            return VerificationCheck(
                check_name="financial_consistent",
                status=CheckStatus.FAILED,
                expected={"expected_amount": snap_amount, "difference": snap_diff},
                actual={"expected_amount": cur_amount, "difference": cur_diff},
                message=f"Financial amounts changed: expected {snap_amount}→{cur_amount}, diff {snap_diff}→{cur_diff}",
            )

        return VerificationCheck(
            check_name="financial_consistent",
            status=CheckStatus.PASSED,
            expected=snap_amount,
            actual=cur_amount,
        )

    def _check_guardrail_valid(
        self,
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Verify the guardrail decision is still valid."""
        snap_decision = snapshot.get("decision")
        cur_decision = current.get("decision")

        if snap_decision != cur_decision:
            return VerificationCheck(
                check_name="guardrail_valid",
                status=CheckStatus.FAILED,
                expected=snap_decision,
                actual=cur_decision,
                message=f"Guardrail decision changed: {snap_decision} → {cur_decision}",
            )

        return VerificationCheck(
            check_name="guardrail_valid",
            status=CheckStatus.PASSED,
            expected=snap_decision,
            actual=cur_decision,
        )

    def _check_no_conflicting_update(
        self,
        snapshot: Dict[str, Any],
        current: Dict[str, Any],
    ) -> VerificationCheck:
        """Check for any conflicting updates since recommendation."""
        snap_version = snapshot.get("state_version", 0)
        cur_version = current.get("state_version", 0)

        # If both have version tracking and version changed
        if snap_version and cur_version and snap_version != cur_version:
            # Check if the reconciliation result also changed
            snap_recon = snapshot.get("reconciliation_hash")
            cur_recon = current.get("reconciliation_hash")
            if snap_recon and cur_recon and snap_recon != cur_recon:
                return VerificationCheck(
                    check_name="no_conflicting_update",
                    status=CheckStatus.FAILED,
                    expected=snap_version,
                    actual=cur_version,
                    message=f"Conflicting update: state version {snap_version}→{cur_version}, reconciliation changed",
                )

        return VerificationCheck(
            check_name="no_conflicting_update",
            status=CheckStatus.PASSED,
            expected=snap_version,
            actual=cur_version,
        )
