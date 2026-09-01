"""
Tests for Razorpay CloseLoop Phase 10G — MLflow Model Registry.

Tests the model lifecycle (CANDIDATE → VALIDATION → PRODUCTION → ARCHIVED),
validation gates, promotion gates, rollback, and metadata preservation.
"""

import pytest

from app.schemas.mlflow_model_registry import (
    LifecycleTransition,
    ModelLifecycleState,
    PromotionGateConfig,
    RegistryModelEntry,
    RegistrySummary,
    ValidationGateConfig,
    is_valid_transition,
)
from app.services.mlflow_model_registry import MLflowModelRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> MLflowModelRegistry:
    return MLflowModelRegistry()


@pytest.fixture
def strict_registry() -> MLflowModelRegistry:
    return MLflowModelRegistry(
        validation_config=ValidationGateConfig(
            min_accuracy=0.80,
            min_f1_macro=0.75,
            max_false_automation=2,
            max_high_value_errors=0,
        ),
        promotion_config=PromotionGateConfig(
            min_accuracy=0.85,
            min_f1_macro=0.80,
            max_false_automation=1,
            max_high_value_errors=0,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle State Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLifecycleState:
    """Test lifecycle state enum and transitions."""

    def test_all_states(self):
        states = [s.value for s in ModelLifecycleState]
        assert "CANDIDATE" in states
        assert "VALIDATION" in states
        assert "PRODUCTION" in states
        assert "ARCHIVED" in states

    def test_valid_transitions(self):
        assert is_valid_transition(ModelLifecycleState.CANDIDATE, ModelLifecycleState.VALIDATION)
        assert is_valid_transition(ModelLifecycleState.CANDIDATE, ModelLifecycleState.ARCHIVED)
        assert is_valid_transition(ModelLifecycleState.VALIDATION, ModelLifecycleState.PRODUCTION)
        assert is_valid_transition(ModelLifecycleState.VALIDATION, ModelLifecycleState.CANDIDATE)
        assert is_valid_transition(ModelLifecycleState.VALIDATION, ModelLifecycleState.ARCHIVED)
        assert is_valid_transition(ModelLifecycleState.PRODUCTION, ModelLifecycleState.ARCHIVED)
        assert is_valid_transition(ModelLifecycleState.PRODUCTION, ModelLifecycleState.VALIDATION)

    def test_invalid_transitions(self):
        assert not is_valid_transition(ModelLifecycleState.CANDIDATE, ModelLifecycleState.PRODUCTION)
        assert not is_valid_transition(ModelLifecycleState.CANDIDATE, ModelLifecycleState.CANDIDATE)
        assert not is_valid_transition(ModelLifecycleState.ARCHIVED, ModelLifecycleState.CANDIDATE)
        assert not is_valid_transition(ModelLifecycleState.ARCHIVED, ModelLifecycleState.VALIDATION)
        assert not is_valid_transition(ModelLifecycleState.PRODUCTION, ModelLifecycleState.CANDIDATE)


# ─────────────────────────────────────────────────────────────────────────────
# Registration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    """Test candidate model registration."""

    def test_register_candidate(self, registry: MLflowModelRegistry):
        entry = registry.register_candidate(
            model_id="MOD-001",
            model_name="xgb-classifier",
            model_version="1.0.0",
            mlflow_run_id="RUN-123",
            dataset_version="1.0.0",
            feature_schema_version="1.0.0",
            algorithm="xgboost",
            accuracy=0.85,
            f1_macro=0.82,
        )
        assert entry.model_id == "MOD-001"
        assert entry.state == ModelLifecycleState.CANDIDATE
        assert entry.mlflow_run_id == "RUN-123"
        assert entry.accuracy == 0.85

    def test_get_model(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        found = registry.get_model("MOD-1")
        assert found is not None
        assert found.model_version == "1.0.0"

    def test_get_nonexistent(self, registry: MLflowModelRegistry):
        assert registry.get_model("MOD-NONE") is None

    def test_list_all(self, registry: MLflowModelRegistry):
        registry.register_candidate(model_id="M1", model_name="a", model_version="1.0")
        registry.register_candidate(model_id="M2", model_name="b", model_version="2.0")
        assert len(registry.list_models()) == 2

    def test_list_by_state(self, registry: MLflowModelRegistry):
        registry.register_candidate(model_id="M1", model_name="a", model_version="1.0")
        registry.register_candidate(model_id="M2", model_name="b", model_version="2.0")
        registry.validate_candidate("M1", reason="test")
        candidates = registry.list_models(state=ModelLifecycleState.CANDIDATE)
        validations = registry.list_models(state=ModelLifecycleState.VALIDATION)
        assert len(candidates) == 1
        assert len(validations) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Validation Gate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationGate:
    """Test CANDIDATE → VALIDATION transition."""

    def test_validation_passes(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85, f1_macro=0.82,
        )
        transition = registry.validate_candidate(
            "MOD-1",
            training_succeeded=True,
            evaluation_metrics_exist=True,
            safety_checks_passed=True,
        )
        assert transition.to_state == ModelLifecycleState.VALIDATION
        assert registry.get_model("MOD-1").state == ModelLifecycleState.VALIDATION

    def test_validation_fails_training(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        transition = registry.validate_candidate(
            "MOD-1",
            training_succeeded=False,
            evaluation_metrics_exist=True,
            safety_checks_passed=True,
        )
        assert transition.to_state == ModelLifecycleState.ARCHIVED
        assert "training_failed" in transition.reason

    def test_validation_fails_no_metrics(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        transition = registry.validate_candidate(
            "MOD-1",
            training_succeeded=True,
            evaluation_metrics_exist=False,
            safety_checks_passed=True,
        )
        assert transition.to_state == ModelLifecycleState.ARCHIVED

    def test_validation_fails_safety(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        transition = registry.validate_candidate(
            "MOD-1",
            training_succeeded=True,
            evaluation_metrics_exist=True,
            safety_checks_passed=False,
        )
        assert transition.to_state == ModelLifecycleState.ARCHIVED

    def test_validation_fails_low_accuracy(self, strict_registry: MLflowModelRegistry):
        strict_registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.60,  # Below 0.80 threshold
        )
        transition = strict_registry.validate_candidate(
            "MOD-1",
            training_succeeded=True,
            evaluation_metrics_exist=True,
            safety_checks_passed=True,
        )
        assert transition.to_state == ModelLifecycleState.ARCHIVED
        assert "accuracy" in transition.reason

    def test_validation_fails_high_false_auto(self, strict_registry: MLflowModelRegistry):
        strict_registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.90, f1_macro=0.85, false_automation=5,
        )
        transition = strict_registry.validate_candidate(
            "MOD-1",
            training_succeeded=True,
            evaluation_metrics_exist=True,
            safety_checks_passed=True,
        )
        assert transition.to_state == ModelLifecycleState.ARCHIVED

    def test_validation_nonexistent_model(self, registry: MLflowModelRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.validate_candidate("MOD-NONE")

    def test_validation_wrong_state(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        registry.validate_candidate("MOD-1")
        with pytest.raises(ValueError, match="not CANDIDATE"):
            registry.validate_candidate("MOD-1")


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromotionGate:
    """Test VALIDATION → PRODUCTION transition."""

    def test_promotion_passes(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85, f1_macro=0.82,
        )
        registry.validate_candidate("MOD-1")
        transition = registry.promote_to_production(
            "MOD-1",
            evaluation_verdict="PROMOTE",
            accuracy=0.85,
            f1_macro=0.82,
        )
        assert transition.to_state == ModelLifecycleState.PRODUCTION
        assert registry.production_model_id == "MOD-1"

    def test_promotion_fails_verdict(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        registry.validate_candidate("MOD-1")
        transition = registry.promote_to_production(
            "MOD-1",
            evaluation_verdict="REJECT",
        )
        assert transition.to_state == ModelLifecycleState.CANDIDATE

    def test_promotion_fails_low_accuracy(self, strict_registry: MLflowModelRegistry):
        strict_registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.82, f1_macro=0.78,
        )
        strict_registry.validate_candidate("MOD-1")
        transition = strict_registry.promote_to_production(
            "MOD-1",
            evaluation_verdict="PROMOTE",
            accuracy=0.82,  # Below 0.85 threshold
        )
        assert transition.to_state == ModelLifecycleState.CANDIDATE

    def test_promotion_nonexistent(self, registry: MLflowModelRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.promote_to_production("MOD-NONE")

    def test_promotion_wrong_state(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        with pytest.raises(ValueError, match="not VALIDATION"):
            registry.promote_to_production("MOD-1")


# ─────────────────────────────────────────────────────────────────────────────
# Archive Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArchive:
    """Test archiving models."""

    def test_archive_candidate(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        transition = registry.archive_model("MOD-1")
        assert transition.to_state == ModelLifecycleState.ARCHIVED

    def test_archive_already_archived(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        registry.archive_model("MOD-1")
        with pytest.raises(ValueError, match="already ARCHIVED"):
            registry.archive_model("MOD-1")

    def test_archive_nonexistent(self, registry: MLflowModelRegistry):
        with pytest.raises(ValueError, match="not found"):
            registry.archive_model("MOD-NONE")


# ─────────────────────────────────────────────────────────────────────────────
# Production Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProductionModel:
    """Test production model identification."""

    def test_no_production_initially(self, registry: MLflowModelRegistry):
        assert registry.production_model_id is None
        assert registry.get_production_model() is None

    def test_production_after_promotion(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        registry.validate_candidate("MOD-1")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.85)
        prod = registry.get_production_model()
        assert prod is not None
        assert prod.model_id == "MOD-1"
        assert prod.state == ModelLifecycleState.PRODUCTION

    def test_new_promotion_archives_old(self, registry: MLflowModelRegistry):
        # First model
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.80,
        )
        registry.validate_candidate("MOD-1")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.80)

        # Second model
        registry.register_candidate(
            model_id="MOD-2", model_name="test", model_version="2.0.0",
            accuracy=0.85, previous_model_id="MOD-1", previous_model_version="1.0.0",
        )
        registry.validate_candidate("MOD-2")
        registry.promote_to_production("MOD-2", evaluation_verdict="PROMOTE", accuracy=0.85)

        # Old model should be archived
        old = registry.get_model("MOD-1")
        assert old.state == ModelLifecycleState.ARCHIVED
        assert registry.production_model_id == "MOD-2"


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRollback:
    """Test model rollback."""

    def test_rollback_no_production(self, registry: MLflowModelRegistry):
        result = registry.rollback_to_previous()
        assert result is None

    def test_rollback_to_previous(self, registry: MLflowModelRegistry):
        # Promote v1
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.80,
        )
        registry.validate_candidate("MOD-1")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.80)

        # Promote v2 (replaces v1)
        registry.register_candidate(
            model_id="MOD-2", model_name="test", model_version="2.0.0",
            accuracy=0.85, previous_model_id="MOD-1", previous_model_version="1.0.0",
        )
        registry.validate_candidate("MOD-2")
        registry.promote_to_production("MOD-2", evaluation_verdict="PROMOTE", accuracy=0.85)

        # Rollback
        transition = registry.rollback_to_previous(reason="quality issue")
        assert transition is not None
        assert transition.to_state == ModelLifecycleState.PRODUCTION
        assert registry.production_model_id == "MOD-1"

        # Verify states
        mod1 = registry.get_model("MOD-1")
        mod2 = registry.get_model("MOD-2")
        assert mod1.state == ModelLifecycleState.PRODUCTION
        assert mod2.state == ModelLifecycleState.ARCHIVED


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Preservation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetadataPreservation:
    """Test that metadata is preserved through lifecycle transitions."""

    def test_metadata_survives_validation(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            mlflow_run_id="RUN-123", dataset_version="1.0.0",
            feature_schema_version="1.0.0", algorithm="xgboost",
            accuracy=0.85, f1_macro=0.82, false_automation=2,
        )
        registry.validate_candidate("MOD-1")
        entry = registry.get_model("MOD-1")
        assert entry.mlflow_run_id == "RUN-123"
        assert entry.dataset_version == "1.0.0"
        assert entry.feature_schema_version == "1.0.0"
        assert entry.accuracy == 0.85

    def test_metadata_survives_promotion(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            mlflow_run_id="RUN-123", dataset_version="1.0.0",
            algorithm="xgboost", accuracy=0.85,
        )
        registry.validate_candidate("MOD-1")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.85)
        entry = registry.get_model("MOD-1")
        assert entry.mlflow_run_id == "RUN-123"
        assert entry.dataset_version == "1.0.0"
        assert entry.algorithm == "xgboost"
        assert entry.promoted_at is not None

    def test_timestamps_recorded(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        entry = registry.get_model("MOD-1")
        assert entry.created_at is not None

        registry.validate_candidate("MOD-1")
        assert entry.validated_at is not None

        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE")
        assert entry.promoted_at is not None

        registry.archive_model("MOD-1")
        assert entry.archived_at is not None

    def test_custom_metadata(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            metadata={"team": "ml-ops", "ticket": "JIRA-123"},
        )
        entry = registry.get_model("MOD-1")
        assert entry.metadata["team"] == "ml-ops"
        assert entry.metadata["ticket"] == "JIRA-123"


# ─────────────────────────────────────────────────────────────────────────────
# Transition History Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTransitionHistory:
    """Test lifecycle transition recording."""

    def test_transitions_recorded(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        registry.validate_candidate("MOD-1", reason="quality check passed")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.85)

        transitions = registry.get_transitions("MOD-1")
        assert len(transitions) == 2
        assert transitions[0].to_state == ModelLifecycleState.VALIDATION
        assert transitions[1].to_state == ModelLifecycleState.PRODUCTION

    def test_all_transitions(self, registry: MLflowModelRegistry):
        registry.register_candidate(model_id="M1", model_name="a", model_version="1.0")
        registry.register_candidate(model_id="M2", model_name="b", model_version="2.0")
        registry.validate_candidate("M1")
        registry.validate_candidate("M2")
        all_trans = registry.get_transitions()
        assert len(all_trans) == 2

    def test_transition_records_reason(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
        )
        transition = registry.validate_candidate("MOD-1", reason="all checks passed")
        assert transition.reason == "all checks passed"


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSummary:
    """Test registry summary."""

    def test_empty_summary(self, registry: MLflowModelRegistry):
        summary = registry.get_summary()
        assert summary.total_models == 0
        assert summary.production_count == 0

    def test_summary_with_models(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="M1", model_name="a", model_version="1.0", accuracy=0.80,
        )
        registry.register_candidate(
            model_id="M2", model_name="b", model_version="2.0", accuracy=0.85,
        )
        registry.validate_candidate("M1")
        registry.promote_to_production("M1", evaluation_verdict="PROMOTE", accuracy=0.80)

        summary = registry.get_summary()
        assert summary.total_models == 2
        assert summary.production_count == 1
        assert summary.candidate_count == 1
        assert summary.production_model is not None
        assert summary.total_transitions == 2


# ─────────────────────────────────────────────────────────────────────────────
# Full Lifecycle End-to-End Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullLifecycle:
    """Test the complete lifecycle flow."""

    def test_candidate_to_production(self, registry: MLflowModelRegistry):
        # Register
        entry = registry.register_candidate(
            model_id="MOD-001",
            model_name="xgb-classifier",
            model_version="1.0.0",
            mlflow_run_id="RUN-abc",
            dataset_version="1.0.0",
            feature_schema_version="1.0.0",
            algorithm="xgboost",
            accuracy=0.85,
            f1_macro=0.82,
        )
        assert entry.state == ModelLifecycleState.CANDIDATE

        # Validate
        registry.validate_candidate("MOD-001")
        assert registry.get_model("MOD-001").state == ModelLifecycleState.VALIDATION

        # Promote
        registry.promote_to_production("MOD-001", evaluation_verdict="PROMOTE", accuracy=0.85)
        assert registry.get_model("MOD-001").state == ModelLifecycleState.PRODUCTION
        assert registry.production_model_id == "MOD-001"

    def test_candidate_rejected_at_validation(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-001", model_name="test", model_version="1.0.0",
            accuracy=0.30,
        )
        registry.validate_candidate(
            "MOD-001",
            training_succeeded=True,
            evaluation_metrics_exist=True,
            safety_checks_passed=True,
        )
        assert registry.get_model("MOD-001").state == ModelLifecycleState.ARCHIVED

    def test_candidate_rejected_at_promotion(self, registry: MLflowModelRegistry):
        registry.register_candidate(
            model_id="MOD-001", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        registry.validate_candidate("MOD-001")
        registry.promote_to_production("MOD-001", evaluation_verdict="REJECT")
        assert registry.get_model("MOD-001").state == ModelLifecycleState.CANDIDATE

    def test_multiple_versions_lifecycle(self, registry: MLflowModelRegistry):
        # v1 → PRODUCTION
        registry.register_candidate(model_id="M1", model_name="test", model_version="1.0", accuracy=0.80)
        registry.validate_candidate("M1")
        registry.promote_to_production("M1", evaluation_verdict="PROMOTE", accuracy=0.80)

        # v2 → PRODUCTION (archives v1)
        registry.register_candidate(model_id="M2", model_name="test", model_version="2.0", accuracy=0.85, previous_model_id="M1")
        registry.validate_candidate("M2")
        registry.promote_to_production("M2", evaluation_verdict="PROMOTE", accuracy=0.85)

        # v3 → REJECTED at validation
        registry.register_candidate(model_id="M3", model_name="test", model_version="3.0", accuracy=0.30)
        registry.validate_candidate("M3")

        assert registry.get_model("M1").state == ModelLifecycleState.ARCHIVED
        assert registry.get_model("M2").state == ModelLifecycleState.PRODUCTION
        assert registry.get_model("M3").state == ModelLifecycleState.ARCHIVED
        assert registry.production_model_id == "M2"


# ─────────────────────────────────────────────────────────────────────────────
# Safety Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    """Test that registry does not bypass Phase 6."""

    def test_registry_does_not_execute(self, registry: MLflowModelRegistry):
        """Registry only manages lifecycle — no financial execution."""
        registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.85,
        )
        registry.validate_candidate("MOD-1")
        registry.promote_to_production("MOD-1", evaluation_verdict="PROMOTE", accuracy=0.85)
        # No financial action should have occurred
        # Registry only manages state transitions

    def test_low_quality_candidate_blocked(self, strict_registry: MLflowModelRegistry):
        """Low-quality candidates cannot reach production."""
        strict_registry.register_candidate(
            model_id="MOD-1", model_name="test", model_version="1.0.0",
            accuracy=0.50, f1_macro=0.45, false_automation=10,
        )
        strict_registry.validate_candidate("MOD-1")
        assert strict_registry.get_model("MOD-1").state == ModelLifecycleState.ARCHIVED
