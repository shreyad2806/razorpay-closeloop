"""
Tests for FallbackGuard (Phase 6D).

Tests:
- All dependencies healthy → proceed
- ML classifier failure → continue deterministic
- ML resolution predictor failure → continue deterministic
- Similarity service failure → continue deterministic
- Database failure → FAIL_CLOSED
- Evidence retrieval failure → FAIL_CLOSED
- LLM failure → continue
- MCP failure → FAIL_CLOSED
- Multiple ML failures → continue deterministic
- Multiple critical failures → FAIL_CLOSED
- Unknown dependency → FAIL_CLOSED
- Engine already deferred → respect
- Fail-closed verification
- Error categories
- Configuration policies
"""

import pytest

from app.schemas.failure_fallback import (
    DependencyFailure,
    DependencyPolicy,
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
from app.services.fallback_guard import FallbackGuard


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _all_healthy():
    return {
        "ml_classifier": True,
        "ml_resolution_predictor": True,
        "similarity_service": True,
        "database": True,
        "evidence_retrieval": True,
        "llm": True,
        "mcp": True,
    }


def _make_engine_result(status=SelectionStatus.RECOMMENDED):
    return ResolutionEngineResult(
        exception_id="EXC-001",
        case_id="CASE-001",
        payment_id="PAY-001",
        expected_amount=100000,
        actual_amount=97000,
        difference=3000,
        status=status,
        confidence=0.85,
        risk_category="LOW",
        deterministic_exception_type="FEE_DIFFERENCE",
        classification_agreement=True,
        evidence_coverage=0.9,
        evidence_consistency=0.85,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorCategory:
    def test_values(self):
        assert ErrorCategory.ML_UNAVAILABLE.value == "ML_UNAVAILABLE"
        assert ErrorCategory.DATABASE_UNAVAILABLE.value == "DATABASE_UNAVAILABLE"
        assert ErrorCategory.LLM_UNAVAILABLE.value == "LLM_UNAVAILABLE"
        assert ErrorCategory.MCP_UNAVAILABLE.value == "MCP_UNAVAILABLE"
        assert ErrorCategory.MISSING_REQUIRED_DATA.value == "MISSING_REQUIRED_DATA"

    def test_all_categories_exist(self):
        assert len(ErrorCategory) >= 13


class TestFailureSeverity:
    def test_values(self):
        assert FailureSeverity.CRITICAL.value == "CRITICAL"
        assert FailureSeverity.DEGRADED.value == "DEGRADED"
        assert FailureSeverity.INFO.value == "INFO"


class TestFallbackAction:
    def test_values(self):
        assert FallbackAction.FAIL_CLOSED.value == "FAIL_CLOSED"
        assert FallbackAction.CONTINUE_WITHOUT.value == "CONTINUE_WITHOUT"
        assert FallbackAction.USE_DETERMINISTIC_ONLY.value == "USE_DETERMINISTIC_ONLY"


class TestDependencyPolicy:
    def test_required_policy(self):
        policy = DependencyPolicy(
            dependency_name="database",
            is_required=True,
            fallback_action=FallbackAction.FAIL_CLOSED,
            fallback_status="HUMAN_REVIEW",
            error_category=ErrorCategory.DATABASE_UNAVAILABLE,
        )
        assert policy.is_required is True

    def test_optional_policy(self):
        policy = DependencyPolicy(
            dependency_name="ml_classifier",
            is_required=False,
            fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
            fallback_status="",
            error_category=ErrorCategory.ML_UNAVAILABLE,
        )
        assert policy.is_required is False


class TestDependencyFailure:
    def test_summary(self):
        failure = DependencyFailure(
            dependency_name="database",
            error_category=ErrorCategory.DATABASE_UNAVAILABLE,
            severity=FailureSeverity.CRITICAL,
            fallback_action=FallbackAction.FAIL_CLOSED,
            fallback_status="HUMAN_REVIEW",
        )
        s = failure.summary()
        assert "database" in s
        assert "DATABASE_UNAVAILABLE" in s
        assert "CRITICAL" in s


class TestFailureFallbackResult:
    def test_proceed_result(self):
        result = FailureFallbackResult(
            can_proceed=True,
            action=FallbackAction.CONTINUE_WITHOUT,
            fallback_status="",
            reason="All healthy",
        )
        assert result.can_proceed is True
        assert result.fallback_version == "1.0.0"

    def test_fail_closed_result(self):
        result = FailureFallbackResult(
            can_proceed=False,
            action=FallbackAction.FAIL_CLOSED,
            fallback_status="HUMAN_REVIEW",
            reason="Database down",
        )
        assert result.can_proceed is False

    def test_summary(self):
        result = FailureFallbackResult(
            can_proceed=True,
            action=FallbackAction.CONTINUE_WITHOUT,
            fallback_status="",
            reason="OK",
        )
        s = result.summary()
        assert "PROCEED" in s


# ─────────────────────────────────────────────────────────────────────────────
# Core Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAllHealthy:
    """Tests for all dependencies healthy."""

    def test_proceed_when_all_healthy(self):
        guard = FallbackGuard()
        status = _all_healthy()
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.CONTINUE_WITHOUT
        assert len(output.failures) == 0
        assert output.has_critical_failure is False

    def test_proceed_with_engine_result(self):
        guard = FallbackGuard()
        status = _all_healthy()
        engine = _make_engine_result()
        output = guard.evaluate(status, engine)

        assert output.can_proceed is True
        assert output.exception_id == "EXC-001"


# ─────────────────────────────────────────────────────────────────────────────
# ML Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMLFailure:
    """Tests for ML dependency failures — should continue with deterministic."""

    def test_ml_classifier_failure(self):
        """ML classifier fails → continue deterministic, not FAIL_CLOSED."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.USE_DETERMINISTIC_ONLY
        assert output.can_use_deterministic_only is True
        assert len(output.failures) == 1
        assert output.failures[0].severity == FailureSeverity.DEGRADED

    def test_ml_resolution_predictor_failure(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_resolution_predictor"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.USE_DETERMINISTIC_ONLY

    def test_similarity_service_failure(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["similarity_service"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.USE_DETERMINISTIC_ONLY

    def test_all_ml_failures(self):
        """All ML services fail → still proceed deterministic."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        status["ml_resolution_predictor"] = False
        status["similarity_service"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.USE_DETERMINISTIC_ONLY
        assert output.can_use_deterministic_only is True
        assert len(output.failures) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Database Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatabaseFailure:
    """Tests for database failures — must FAIL_CLOSED."""

    def test_database_unavailable(self):
        """Database down → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.action == FallbackAction.FAIL_CLOSED
        assert output.fallback_status == "HUMAN_REVIEW"
        assert output.has_critical_failure is True

    def test_database_with_ml(self):
        """Database down + ML works → still FAIL_CLOSED."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.has_critical_failure is True


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Retrieval Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceRetrievalFailure:
    """Tests for evidence retrieval failures — must FAIL_CLOSED."""

    def test_evidence_retrieval_unavailable(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["evidence_retrieval"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.action == FallbackAction.FAIL_CLOSED
        assert output.fallback_status == "HUMAN_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# LLM Failure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMFailure:
    """Tests for LLM failures — optional, continue without."""

    def test_llm_unavailable(self):
        """LLM down → continue deterministic pipeline."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["llm"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.action == FallbackAction.CONTINUE_WITHOUT


# ─────────────────────────────────────────────────────────────────────────────
# MCP Failure Tests
#────────────────────────────────────────────────────────────────────────────


class TestMCPFailure:
    """Tests for MCP failures — FAIL_CLOSED if workflow requires it."""

    def test_mcp_unavailable(self):
        """MCP down → FAIL_CLOSED (per default policy)."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["mcp"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.action == FallbackAction.FAIL_CLOSED
        assert output.fallback_status == "HUMAN_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# Engine Already Deferred Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineDeferred:
    """Tests for engine already returning HUMAN_REVIEW/UNRESOLVED."""

    def test_engine_human_review(self):
        """Engine returned HUMAN_REVIEW → respect that."""
        guard = FallbackGuard()
        status = _all_healthy()
        engine = _make_engine_result(status=SelectionStatus.HUMAN_REVIEW)
        output = guard.evaluate(status, engine)

        assert output.can_proceed is False
        assert output.fallback_status == "HUMAN_REVIEW"

    def test_engine_unresolved(self):
        """Engine returned UNRESOLVED → respect that."""
        guard = FallbackGuard()
        status = _all_healthy()
        engine = _make_engine_result(status=SelectionStatus.UNRESOLVED)
        output = guard.evaluate(status, engine)

        assert output.can_proceed is False
        assert output.fallback_status == "UNRESOLVED"


# ─────────────────────────────────────────────────────────────────────────────
# Unknown Dependency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownDependency:
    """Tests for unknown dependencies — fail closed."""

    def test_unknown_dependency_fails_closed(self):
        """Unknown dependency failure → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["unknown_service_xyz"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.has_critical_failure is True


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Critical Failures Tests
#────────────────────────────────────────────────────────────────────────────


class TestMultipleCriticalFailures:
    """Tests for multiple critical failures."""

    def test_database_and_evidence_fail(self):
        """Both database and evidence retrieval fail."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        status["evidence_retrieval"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert len(output.critical_failures) == 2
        assert output.has_critical_failure is True

    def test_all_critical_fail(self):
        """All required dependencies fail."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        status["evidence_retrieval"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert len(output.critical_failures) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Mixed Failure Tests
#────────────────────────────────────────────────────────────────────────────


class TestMixedFailures:
    """Tests for mixed critical and optional failures."""

    def test_ml_down_database_down(self):
        """ML down + database down → FAIL_CLOSED."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        status["database"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.has_critical_failure is True
        # ML failure also recorded
        assert len(output.failures) == 2

    def test_ml_down_llm_down(self):
        """ML down + LLM down → proceed deterministic."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        status["llm"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True
        assert output.can_use_deterministic_only is True


# ─────────────────────────────────────────────────────────────────────────────
# Fail-Closed Verification Tests
#────────────────────────────────────────────────────────────────────────────


class TestFailClosed:
    """Verify fail-closed principle."""

    def test_critical_never_auto(self):
        """Critical dependency failure must never produce AUTO."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.action == FallbackAction.FAIL_CLOSED

    def test_optional_ml_does_not_block(self):
        """ML failure alone does not block — deterministic continues."""
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True

    def test_unknown_dep_blocks(self):
        """Unknown dependency → fail closed."""
        guard = FallbackGuard()
        status = {"some_unknown_dep": False}
        output = guard.evaluate(status)

        assert output.can_proceed is False


# ─────────────────────────────────────────────────────────────────────────────
# Policy Configuration Tests
#────────────────────────────────────────────────────────────────────────────


class TestPolicyConfiguration:
    """Tests for custom dependency policies."""

    def test_custom_required_policy(self):
        """Make ML classifier required → fail closed on ML failure."""
        policies = dict(FallbackGuard().policies)
        policies["ml_classifier"] = DependencyPolicy(
            dependency_name="ml_classifier",
            is_required=True,
            fallback_action=FallbackAction.FAIL_CLOSED,
            fallback_status="HUMAN_REVIEW",
            error_category=ErrorCategory.ML_UNAVAILABLE,
        )
        guard = FallbackGuard(policies=policies)
        status = _all_healthy()
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is False
        assert output.has_critical_failure is True

    def test_custom_optional_database(self):
        """Make database optional (dangerous but testable)."""
        policies = dict(FallbackGuard().policies)
        policies["database"] = DependencyPolicy(
            dependency_name="database",
            is_required=False,
            fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
            fallback_status="",
            error_category=ErrorCategory.DATABASE_UNAVAILABLE,
        )
        guard = FallbackGuard(policies=policies)
        status = _all_healthy()
        status["database"] = False
        output = guard.evaluate(status)

        # Now database failure doesn't block
        assert output.can_proceed is True

    def test_custom_mcp_optional(self):
        """Make MCP optional."""
        policies = dict(FallbackGuard().policies)
        policies["mcp"] = DependencyPolicy(
            dependency_name="mcp",
            is_required=False,
            fallback_action=FallbackAction.CONTINUE_WITHOUT,
            fallback_status="",
            error_category=ErrorCategory.MCP_UNAVAILABLE,
        )
        guard = FallbackGuard(policies=policies)
        status = _all_healthy()
        status["mcp"] = False
        output = guard.evaluate(status)

        assert output.can_proceed is True


# ─────────────────────────────────────────────────────────────────────────────
# Error Category Tests
#────────────────────────────────────────────────────────────────────────────


class TestErrorCategories:
    """Verify correct error categories are assigned."""

    def test_ml_failure_category(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        assert ErrorCategory.ML_UNAVAILABLE in output.failed_categories

    def test_database_failure_category(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        output = guard.evaluate(status)

        assert ErrorCategory.DATABASE_UNAVAILABLE in output.failed_categories

    def test_llm_failure_category(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["llm"] = False
        output = guard.evaluate(status)

        assert ErrorCategory.LLM_UNAVAILABLE in output.failed_categories

    def test_mcp_failure_category(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["mcp"] = False
        output = guard.evaluate(status)

        assert ErrorCategory.MCP_UNAVAILABLE in output.failed_categories

    def test_unknown_category(self):
        guard = FallbackGuard()
        status = {"some_unknown_service": False}
        output = guard.evaluate(status)

        assert ErrorCategory.UNKNOWN_FAILURE in output.failed_categories


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Tests
#────────────────────────────────────────────────────────────────────────────


class TestGuardMetadata:
    """Tests for guard metadata and traceability."""

    def test_exception_id_preserved(self):
        guard = FallbackGuard()
        status = _all_healthy()
        engine = _make_engine_result()
        output = guard.evaluate(status, engine)

        assert output.exception_id == "EXC-001"
        assert output.case_id == "CASE-001"

    def test_fallback_version(self):
        guard = FallbackGuard()
        output = guard.evaluate(_all_healthy())

        assert output.fallback_version == "1.0.0"

    def test_failure_details_recorded(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        status["llm"] = False
        output = guard.evaluate(status)

        assert len(output.failures) == 2
        dep_names = {f.dependency_name for f in output.failures}
        assert "ml_classifier" in dep_names
        assert "llm" in dep_names

    def test_critical_failures_subset(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        status["database"] = False
        output = guard.evaluate(status)

        assert len(output.failures) == 2
        assert len(output.critical_failures) == 1
        assert output.critical_failures[0].dependency_name == "database"


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
#────────────────────────────────────────────────────────────────────────────


class TestGuardSummary:
    """Tests for fallback result summary."""

    def test_proceed_summary(self):
        guard = FallbackGuard()
        output = guard.evaluate(_all_healthy())

        s = output.summary()
        assert "PROCEED" in s

    def test_fail_closed_summary(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["database"] = False
        output = guard.evaluate(status)

        s = output.summary()
        assert "FAIL_CLOSED" in s
        assert "CRITICAL" in s

    def test_degraded_summary(self):
        guard = FallbackGuard()
        status = _all_healthy()
        status["ml_classifier"] = False
        output = guard.evaluate(status)

        s = output.summary()
        assert "PROCEED" in s
        assert "Failures: 1" in s
