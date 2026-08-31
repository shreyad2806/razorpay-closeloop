"""
System Failure Fallback Guard for Razorpay CloseLoop Phase 6D.

Implements fail-closed behavior for dependent system failures.

The system must fail CLOSED rather than accidentally auto-resolve
when a critical dependency is unavailable.

Fail-closed principle:
- Critical dependency failure → HUMAN_REVIEW or UNRESOLVED
- Never AUTO when a dependency is missing
- ML failure does not reduce safety requirements
- LLM failure does not reduce safety requirements
- MCP failure blocks any financial action
"""

from typing import Dict, List, Optional

from app.schemas.failure_fallback import (
    DependencyFailure,
    DependencyPolicy,
    ErrorCategory,
    FailureFallbackResult,
    FailureSeverity,
    FallbackAction,
)
from app.schemas.resolution_engine import ResolutionEngineResult
from app.schemas.resolution_selection import SelectionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Default Policies
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_POLICIES: Dict[str, DependencyPolicy] = {
    "ml_classifier": DependencyPolicy(
        dependency_name="ml_classifier",
        is_required=False,
        fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
        fallback_status="",
        error_category=ErrorCategory.ML_UNAVAILABLE,
        description="ML exception classifier — optional, deterministic fallback available",
    ),
    "ml_resolution_predictor": DependencyPolicy(
        dependency_name="ml_resolution_predictor",
        is_required=False,
        fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
        fallback_status="",
        error_category=ErrorCategory.ML_UNAVAILABLE,
        description="ML resolution predictor — optional, deterministic fallback available",
    ),
    "similarity_service": DependencyPolicy(
        dependency_name="similarity_service",
        is_required=False,
        fallback_action=FallbackAction.USE_DETERMINISTIC_ONLY,
        fallback_status="",
        error_category=ErrorCategory.ML_UNAVAILABLE,
        description="Historical similarity service — optional",
    ),
    "database": DependencyPolicy(
        dependency_name="database",
        is_required=True,
        fallback_action=FallbackAction.FAIL_CLOSED,
        fallback_status="HUMAN_REVIEW",
        error_category=ErrorCategory.DATABASE_UNAVAILABLE,
        description="PostgreSQL database — required for all operations",
    ),
    "evidence_retrieval": DependencyPolicy(
        dependency_name="evidence_retrieval",
        is_required=True,
        fallback_action=FallbackAction.FAIL_CLOSED,
        fallback_status="HUMAN_REVIEW",
        error_category=ErrorCategory.EVIDENCE_RETRIEVAL_FAILURE,
        description="Evidence retrieval service — required for resolution",
    ),
    "llm": DependencyPolicy(
        dependency_name="llm",
        is_required=False,
        fallback_action=FallbackAction.CONTINUE_WITHOUT,
        fallback_status="",
        error_category=ErrorCategory.LLM_UNAVAILABLE,
        description="LLM service — optional, deterministic pipeline continues",
    ),
    "mcp": DependencyPolicy(
        dependency_name="mcp",
        is_required=True,
        fallback_action=FallbackAction.FAIL_CLOSED,
        fallback_status="HUMAN_REVIEW",
        error_category=ErrorCategory.MCP_UNAVAILABLE,
        description="MCP tool service — block execution when unavailable",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Guard
# ─────────────────────────────────────────────────────────────────────────────


class FallbackGuard:
    """
    Evaluates dependency health and enforces fail-closed behavior.

    When a critical dependency is unavailable, the system must fail
    CLOSED (HUMAN_REVIEW or UNRESOLVED) rather than accidentally
    auto-resolve.
    """

    def __init__(
        self,
        policies: Optional[Dict[str, DependencyPolicy]] = None,
    ):
        """Initialize the fallback guard.

        Args:
            policies: Dependency policies. Uses defaults if not provided.
        """
        self.policies = policies or DEFAULT_POLICIES

    def evaluate(
        self,
        dependency_status: Dict[str, bool],
        engine_result: Optional[ResolutionEngineResult] = None,
    ) -> FailureFallbackResult:
        """Evaluate dependency health and determine fallback behavior.

        Args:
            dependency_status: Map of dependency_name → is_healthy (True/False)
            engine_result: Optional engine result for context

        Returns:
            FailureFallbackResult with fail-closed decision
        """
        failures: List[DependencyFailure] = []
        critical_failures: List[DependencyFailure] = []
        failed_categories: List[ErrorCategory] = []
        can_proceed = True
        fallback_status = ""
        primary_reason = ""

        for dep_name, is_healthy in dependency_status.items():
            if is_healthy:
                continue

            # Get policy for this dependency
            policy = self.policies.get(dep_name)
            if policy is None:
                # Unknown dependency — fail closed
                policy = DependencyPolicy(
                    dependency_name=dep_name,
                    is_required=True,
                    fallback_action=FallbackAction.FAIL_CLOSED,
                    fallback_status="HUMAN_REVIEW",
                    error_category=ErrorCategory.UNKNOWN_FAILURE,
                    description=f"Unknown dependency: {dep_name}",
                )

            failure = DependencyFailure(
                dependency_name=dep_name,
                error_category=policy.error_category,
                severity=(
                    FailureSeverity.CRITICAL
                    if policy.is_required
                    else FailureSeverity.DEGRADED
                ),
                error_message=f"{dep_name} is unavailable",
                fallback_action=policy.fallback_action,
                fallback_status=policy.fallback_status,
                is_recoverable=True,
            )

            failures.append(failure)
            if policy.error_category not in failed_categories:
                failed_categories.append(policy.error_category)

            if policy.is_required:
                critical_failures.append(failure)
                can_proceed = False
                if not fallback_status:
                    fallback_status = policy.fallback_status
                    primary_reason = (
                        f"Required dependency '{dep_name}' is unavailable. "
                        f"Failing closed to {policy.fallback_status}."
                    )
            else:
                # Optional dependency — can continue without it
                if not primary_reason:
                    primary_reason = (
                        f"Optional dependency '{dep_name}' is unavailable. "
                        f"Continuing with deterministic pipeline."
                    )

        # Determine if deterministic-only mode is possible
        ml_deps = ["ml_classifier", "ml_resolution_predictor", "similarity_service"]
        ml_unavailable = any(not dependency_status.get(d, True) for d in ml_deps)
        can_use_deterministic = ml_unavailable and can_proceed

        # If engine already deferred, respect that
        if engine_result and engine_result.status in (
            SelectionStatus.UNRESOLVED,
            SelectionStatus.HUMAN_REVIEW,
        ):
            can_proceed = False
            if not fallback_status:
                fallback_status = engine_result.status.value
                primary_reason = (
                    f"Engine already deferred to {engine_result.status.value}"
                )

        # Final fail-closed check
        if not can_proceed and not primary_reason:
            primary_reason = "Critical dependency failure — failing closed"

        if not primary_reason and can_proceed:
            primary_reason = "All dependencies healthy — proceeding"

        return FailureFallbackResult(
            can_proceed=can_proceed,
            action=(
                FallbackAction.FAIL_CLOSED
                if not can_proceed
                else (
                    FallbackAction.USE_DETERMINISTIC_ONLY
                    if can_use_deterministic
                    else FallbackAction.CONTINUE_WITHOUT
                )
            ),
            fallback_status=fallback_status,
            failures=failures,
            critical_failures=critical_failures,
            failed_categories=failed_categories,
            has_critical_failure=len(critical_failures) > 0,
            can_use_deterministic_only=can_use_deterministic,
            reason=primary_reason,
            exception_id=(
                engine_result.exception_id if engine_result else None
            ),
            case_id=engine_result.case_id if engine_result else None,
        )
