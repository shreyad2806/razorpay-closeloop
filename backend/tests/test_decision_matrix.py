"""
Tests for AutomationDecisionMatrix (Phase 6E).

Tests the full decision-matrix test suite:
1. high confidence + low exposure + known + consistent → AUTO
2. medium confidence → HUMAN_REVIEW
3. moderate exposure → HUMAN_REVIEW
4. high exposure → NEVER AUTO
5. unknown pattern → NEVER AUTO
6. conflicting evidence → HUMAN_REVIEW
7. low confidence → HUMAN_REVIEW
8. missing critical evidence → HUMAN_REVIEW / UNRESOLVED
9. novel pattern → HUMAN_REVIEW
10. ML unavailable → NEVER AUTO (via fallback)
11. database failure → NEVER AUTO
12. verification unavailable → NEVER AUTO
"""

import pytest

from app.schemas.confidence_gate import ConfidenceGateResult, GateAction
from app.schemas.decision_matrix import (
    AutomationDecision,
    AutomationDecisionResult,
    DecisionConfig,
    GateResult,
    GateStatus,
    ReasonCode,
)
from app.schemas.evidence_guard import EvidenceAction, EvidenceGuardResult
from app.schemas.exposure_guard import ExposureAction, ExposureGuardResult
from app.schemas.failure_fallback import (
    DependencyFailure,
    ErrorCategory,
    FailureFallbackResult,
    FailureSeverity,
    FallbackAction,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import (
    ExplainabilityDetail,
    ExplainabilityLevel,
    SelectionStatus,
)
from app.services.decision_matrix import AutomationDecisionMatrix


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_engine_result(
    status=SelectionStatus.RECOMMENDED,
    confidence=0.85,
    risk="LOW",
    evidence_coverage=0.90,
    evidence_consistency=0.85,
    exception_type="FEE_DIFFERENCE",
    resolution="FEE_ADJUSTMENT",
):
    return ResolutionEngineResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        merchant_id="MER-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        status=status,
        selected_resolution=resolution,
        confidence=confidence,
        risk_category=risk,
        explainability=ExplainabilityDetail(
            level=ExplainabilityLevel.FULLY_EXPLAINABLE,
            has_evidence_trace=True,
            has_financial_trace=True,
            source_count=2,
        ),
        deterministic_exception_type=exception_type,
        classification_agreement=True,
        evidence_explanation_status="FULLY_EXPLAINED",
        evidence_coverage=evidence_coverage,
        evidence_consistency=evidence_consistency,
    )


def _make_gate(confidence=0.85, passed=True):
    action = GateAction.CONTINUE if passed else GateAction.HUMAN_REVIEW
    return ConfidenceGateResult(
        passed=passed,
        action=action,
        confidence=confidence,
        threshold=0.70,
        reason="test",
    )


def _make_exposure(amount=3000, passed=True):
    action = ExposureAction.PASS if passed else ExposureAction.BLOCK
    return ExposureGuardResult(
        passed=passed,
        action=action,
        adjustment_amount_paise=amount,
        max_auto_resolution_paise=50000,
        reason="test",
    )


def _make_evidence(
    coverage=0.90,
    consistency=0.85,
    conflict=False,
    novel=False,
    passed=True,
):
    action = EvidenceAction.PASS if passed else EvidenceAction.BLOCK
    return EvidenceGuardResult(
        passed=passed,
        action=action,
        evidence_coverage=coverage,
        evidence_consistency=consistency,
        has_conflict=conflict,
        is_novel=novel,
        reason="test",
    )


def _make_fallback(can_proceed=True, critical=None, ml_ok=True):
    failures = []
    critical_failures = []
    if critical:
        for dep in critical:
            f = DependencyFailure(
                dependency_name=dep,
                error_category=ErrorCategory.DATABASE_UNAVAILABLE,
                severity=FailureSeverity.CRITICAL,
                fallback_action=FallbackAction.FAIL_CLOSED,
                fallback_status="HUMAN_REVIEW",
            )
            failures.append(f)
            critical_failures.append(f)

    ml_failures = []
    if not ml_ok:
        for dep in ["ml_classifier", "ml_resolution_predictor", "similarity_service"]:
            ml_failures.append(DependencyFailure(
                dependency_name=dep,
                error_category=ErrorCategory.ML_UNAVAILABLE,
                severity=FailureSeverity.DEGRADED,
                fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
                fallback_status="",
            ))
        failures.extend(ml_failures)

    return FailureFallbackResult(
        can_proceed=can_proceed,
        action=(
            FallbackAction.FAIL_CLOSED if not can_proceed
            else FallbackAction.USE_DETERMINISTIC_ONLY if not ml_ok
            else FallbackAction.CONTINUE_WITHOUT
        ),
        fallback_status="HUMAN_REVIEW" if not can_proceed else "",
        failures=failures,
        critical_failures=critical_failures,
        failed_categories=[],
        has_critical_failure=len(critical_failures) > 0,
        can_use_deterministic_only=not ml_ok and can_proceed,
        reason="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAutomationDecision:
    def test_values(self):
        assert AutomationDecision.AUTO.value == "AUTO"
        assert AutomationDecision.HUMAN_REVIEW.value == "HUMAN_REVIEW"
        assert AutomationDecision.UNRESOLVED.value == "UNRESOLVED"


class TestReasonCode:
    def test_all_codes_exist(self):
        assert ReasonCode.ALL_GATES_PASSED.value == "ALL_GATES_PASSED"
        assert ReasonCode.UNKNOWN_PATTERN.value == "UNKNOWN_PATTERN"
        assert ReasonCode.HIGH_EXPOSURE.value == "HIGH_EXPOSURE"
        assert ReasonCode.CRITICAL_DEP_FAILURE.value == "CRITICAL_DEP_FAILURE"
        assert ReasonCode.CONFLICTING_EVIDENCE.value == "CONFLICTING_EVIDENCE"


class TestGateStatus:
    def test_values(self):
        assert GateStatus.PASSED.value == "PASSED"
        assert GateStatus.FAILED.value == "FAILED"
        assert GateStatus.SKIPPED.value == "SKIPPED"


class TestDecisionConfig:
    def test_defaults(self):
        config = DecisionConfig()
        assert config.min_confidence_for_auto == 0.75
        assert config.max_exposure_for_auto == 25000
        assert config.min_evidence_coverage_for_auto == 0.60
        assert "LOW" in config.allowed_risk_for_auto


class TestAutomationDecisionResult:
    def test_auto_result(self):
        result = AutomationDecisionResult(
            decision=AutomationDecision.AUTO,
            reason_codes=[ReasonCode.ALL_GATES_PASSED],
            primary_reason="All gates passed",
            confidence=0.85,
            risk_category="LOW",
        )
        assert result.decision == AutomationDecision.AUTO
        assert result.decision_version == "1.0.0"

    def test_summary(self):
        result = AutomationDecisionResult(
            decision=AutomationDecision.AUTO,
            primary_reason="OK",
            confidence=0.85,
            risk_category="LOW",
        )
        s = result.summary()
        assert "AUTO" in s
        assert "85.0%" in s


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: AUTO — high confidence + low exposure + known + consistent
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoDecision:
    """All conditions pass → AUTO."""

    def test_all_conditions_pass(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.85,
            risk="LOW",
            evidence_coverage=0.90,
            evidence_consistency=0.85,
        )
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.AUTO
        assert ReasonCode.ALL_GATES_PASSED in result.reason_codes
        assert result.confidence == 0.85
        assert result.system_healthy is True

    def test_auto_with_all_gates_recorded(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.90, risk="LOW")
        gate = _make_gate(confidence=0.90, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.95, consistency=0.90, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.AUTO
        all_gate_names = [g.gate_name for g in result.passed_gates]
        assert "confidence_gate" in all_gate_names
        assert "exposure_guard" in all_gate_names
        assert "evidence_guard" in all_gate_names


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: medium confidence → HUMAN_REVIEW
# ─────────────────────────────────────────────────────────────────────────────


class TestMediumConfidence:
    def test_medium_confidence(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.60, risk="LOW")
        gate = _make_gate(confidence=0.60, passed=False)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.HUMAN_REVIEW
        assert ReasonCode.MEDIUM_CONFIDENCE in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: moderate exposure → HUMAN_REVIEW
# ─────────────────────────────────────────────────────────────────────────────


class TestModerateExposure:
    def test_moderate_exposure(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=50000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        # 50000 > max_exposure_for_auto (25000) → HUMAN_REVIEW
        assert result.decision == AutomationDecision.HUMAN_REVIEW
        assert ReasonCode.MODERATE_EXPOSURE in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: high exposure → NEVER AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestHighExposure:
    def test_high_exposure_unresolved(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.95, risk="LOW")
        gate = _make_gate(confidence=0.95, passed=True)
        exposure = _make_exposure(amount=200000, passed=True)
        evidence = _make_evidence(coverage=0.95, consistency=0.95, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.HIGH_EXPOSURE in result.reason_codes
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: unknown pattern → NEVER AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownPattern:
    def test_unknown_exception_type(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.99,
            risk="LOW",
            exception_type="UNKNOWN",
            resolution="UNKNOWN_UNRESOLVED",
        )
        gate = _make_gate(confidence=0.99, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.95, consistency=0.95, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.UNKNOWN_PATTERN in result.reason_codes
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: conflicting evidence → HUMAN_REVIEW
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictingEvidence:
    def test_conflict_blocks_auto(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(
            coverage=0.90, consistency=0.85, conflict=True, passed=True
        )
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        # Conflict makes evidence guard block → HUMAN_REVIEW
        assert result.decision == AutomationDecision.HUMAN_REVIEW
        assert result.has_conflict is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: low confidence → HUMAN_REVIEW
# ─────────────────────────────────────────────────────────────────────────────


class TestLowConfidence:
    def test_low_confidence_human_review(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.45, risk="LOW")
        gate = _make_gate(confidence=0.45, passed=False)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.HUMAN_REVIEW
        assert ReasonCode.MEDIUM_CONFIDENCE in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: missing critical evidence → HUMAN_REVIEW / UNRESOLVED
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingEvidence:
    def test_missing_evidence_human_review(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.85,
            risk="LOW",
            evidence_coverage=0.20,
            evidence_consistency=0.30,
        )
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(
            coverage=0.20, consistency=0.30, passed=True
        )
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        # Low coverage/consistency → HUMAN_REVIEW
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: novel pattern → HUMAN_REVIEW
# ─────────────────────────────────────────────────────────────────────────────


class TestNovelPattern:
    def test_novel_blocks_auto(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(
            coverage=0.90, consistency=0.85, novel=True, passed=True
        )
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.HUMAN_REVIEW
        assert result.is_novel is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: ML unavailable → NEVER AUTO (via fallback)
# ─────────────────────────────────────────────────────────────────────────────


class TestMLUnavailable:
    def test_ml_down_still_auto_if_deterministic_ok(self):
        """ML down but all deterministic gates pass → AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True, ml_ok=False)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        # ML is optional — deterministic pipeline can proceed
        assert result.decision == AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: database failure → NEVER AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestDatabaseFailure:
    def test_database_down_unresolved(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.95, risk="LOW")
        gate = _make_gate(confidence=0.95, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.95, consistency=0.95, passed=True)
        fallback = _make_fallback(can_proceed=False, critical=["database"])

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.CRITICAL_DEP_FAILURE in result.reason_codes
        assert result.system_healthy is False
        assert "database" in result.critical_failures


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: verification unavailable → NEVER AUTO
# ─────────────────────────────────────────────────────────────────────────────


class TestVerificationUnavailable:
    def test_evidence_retrieval_down(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.95, risk="LOW")
        gate = _make_gate(confidence=0.95, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.95, consistency=0.95, passed=True)
        fallback = _make_fallback(can_proceed=False, critical=["evidence_retrieval"])

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.CRITICAL_DEP_FAILURE in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Priority Order Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorityOrder:
    """Verify safety rules override positive signals."""

    def test_high_confidence_cannot_override_critical_failure(self):
        """99% confidence + database down → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.99, risk="LOW")
        gate = _make_gate(confidence=0.99, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.99, consistency=0.99, passed=True)
        fallback = _make_fallback(can_proceed=False, critical=["database"])

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert result.confidence == 0.99  # Confidence was high, but overridden

    def test_high_confidence_cannot_override_unknown(self):
        """99% confidence + UNKNOWN → UNRESOLVED."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.99, risk="LOW", exception_type="UNKNOWN"
        )
        gate = _make_gate(confidence=0.99, passed=True)
        exposure = _make_exposure(amount=1000, passed=True)
        evidence = _make_evidence(coverage=0.99, consistency=0.99, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_strong_evidence_cannot_override_high_exposure(self):
        """99% coverage + high exposure → not AUTO."""
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.90, risk="LOW")
        gate = _make_gate(confidence=0.90, passed=True)
        exposure = _make_exposure(amount=150000, passed=True)
        evidence = _make_evidence(coverage=0.99, consistency=0.99, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Engine Deferred Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineDeferred:
    def test_engine_unresolved(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(status=SelectionStatus.UNRESOLVED)
        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.ENGINE_DEFERRED in result.reason_codes

    def test_engine_human_review(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(status=SelectionStatus.HUMAN_REVIEW)
        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED
        assert ReasonCode.ENGINE_DEFERRED in result.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Blocked Type Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockedTypes:
    def test_complex_multi_adjustment(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.90,
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            resolution="MULTI_ADJUSTMENT",
        )
        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED

    def test_missing_record(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.90,
            exception_type="MISSING_RECORD",
            resolution="MISSING_RECORD_ESCALATION",
        )
        result = matrix.evaluate(engine)

        assert result.decision == AutomationDecision.UNRESOLVED


# ─────────────────────────────────────────────────────────────────────────────
# Risk Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskLevel:
    def test_high_risk_human_review(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="HIGH")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        # HIGH risk not in allowed_risk_for_auto or allowed_risk_for_human
        assert result.decision != AutomationDecision.AUTO


# ─────────────────────────────────────────────────────────────────────────────
# Gate Tracking Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGateTracking:
    def test_all_gates_recorded(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        all_names = {g.gate_name for g in result.passed_gates} | {
            g.gate_name for g in result.failed_gates
        }
        assert "confidence_gate" in all_names
        assert "exposure_guard" in all_names
        assert "evidence_guard" in all_names
        assert "fallback_guard" in all_names

    def test_failed_gates_recorded(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.60, risk="LOW")
        gate = _make_gate(confidence=0.60, passed=False)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        failed_names = {g.gate_name for g in result.failed_gates}
        assert "confidence_gate" in failed_names


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionMetadata:
    def test_exception_id_preserved(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert result.exception_id == "EXC-001"
        assert result.case_id == "CASE-001"

    def test_decision_version(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        result = matrix.evaluate(engine)

        assert result.decision_version == "1.0.0"

    def test_reason_codes_non_empty(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert len(result.reason_codes) > 0

    def test_primary_reason_non_empty(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)

        assert len(result.primary_reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionSummary:
    def test_auto_summary(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(confidence=0.85, risk="LOW")
        gate = _make_gate(confidence=0.85, passed=True)
        exposure = _make_exposure(amount=3000, passed=True)
        evidence = _make_evidence(coverage=0.90, consistency=0.85, passed=True)
        fallback = _make_fallback(can_proceed=True)

        result = matrix.evaluate(engine, gate, exposure, evidence, fallback)
        s = result.summary()

        assert "AUTO" in s
        assert "85.0%" in s

    def test_unresolved_summary(self):
        matrix = AutomationDecisionMatrix()
        engine = _make_engine_result(
            confidence=0.95, risk="LOW", exception_type="UNKNOWN"
        )
        result = matrix.evaluate(engine)
        s = result.summary()

        assert "UNRESOLVED" in s
