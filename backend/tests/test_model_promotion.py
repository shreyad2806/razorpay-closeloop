"""
Tests for Phase 9F — Safe Model Promotion.

Tests cover:
- Promotion gate checks
- Candidate passes all checks
- Candidate fails precision
- Candidate fails false automation
- Candidate fails high-value safety
- Candidate fails verification safety
- Candidate regression
- Promotion flow
- Model rollback
- Edge cases
"""

import pytest
from datetime import datetime

from app.schemas.model_training import (
    EvaluationMetrics,
    ModelMetadata,
    ModelStatus,
    ModelType,
    TrainingConfig,
)
from app.schemas.model_promotion import (
    GateCheck,
    PromotionDecision,
    PromotionGateResult,
    PromotionGateStatus,
    PromotionRecord,
    PromotionThresholds,
    RollbackRecord,
)
from app.services.model_promotion import ModelRegistry, PromotionGate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_metadata(
    model_id: str = "MOD-001",
    version: str = "1.0.0",
    name: str = "test_model",
    status: ModelStatus = ModelStatus.CANDIDATE,
) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_name=name,
        version=version,
        model_type=ModelType.EXCEPTION_CLASSIFIER,
        status=status,
        dataset_version="1.0.0",
        feature_schema_version="1.0.0",
        training_examples=100,
        feature_count=10,
    )


def _make_metrics(
    model_id: str = "MOD-001",
    version: str = "1.0.0",
    accuracy: float = 0.80,
    precision: float = 0.78,
    recall: float = 0.75,
    f1: float = 0.76,
    false_auto: int = 3,
    hv_errors: int = 0,
    unknown_errors: int = 1,
    ver_fail_rate: float = 0.05,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        model_id=model_id,
        model_version=version,
        total_samples=100,
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        precision_weighted=precision,
        recall_weighted=recall,
        f1_weighted=f1,
        incorrect_auto_resolution=false_auto,
        false_automation=false_auto,
        high_value_errors=hv_errors,
        unknown_case_errors=unknown_errors,
        verification_failure_rate=ver_fail_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Gate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPromotionGate:
    """Test promotion gate checks."""

    def test_candidate_passes_all(self):
        """Candidate meeting all thresholds passes."""
        current = _make_metrics(model_id="MOD-CUR", version="1.0.0", accuracy=0.75)
        candidate = _make_metrics(model_id="MOD-CAND", version="2.0.0", accuracy=0.85)
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is True
        assert result.decision == PromotionDecision.PROMOTED
        assert len(result.failed_checks) == 0

    def test_candidate_fails_precision(self):
        """Candidate with low precision is rejected."""
        current = _make_metrics(precision=0.80)
        candidate = _make_metrics(model_id="MOD-CAND", version="2.0.0", precision=0.50)
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert result.decision == PromotionDecision.REJECTED
        assert "min_precision" in result.failed_checks

    def test_candidate_fails_false_automation(self):
        """Candidate with too many false automations is rejected."""
        current = _make_metrics(false_auto=2)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", false_auto=10
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "max_false_automation" in result.failed_checks

    def test_candidate_fails_high_value_safety(self):
        """Candidate with more high-value errors is rejected."""
        current = _make_metrics(hv_errors=0)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", hv_errors=2
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        failed_hv = [c for c in result.checks if "hv" in c.check_name.lower()]
        assert len(failed_hv) >= 1

    def test_candidate_fails_verification_safety(self):
        """Candidate with high verification failure rate is rejected."""
        current = _make_metrics(ver_fail_rate=0.03)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", ver_fail_rate=0.20
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "max_verification_failure_rate" in result.failed_checks

    def test_candidate_fails_accuracy(self):
        """Candidate below minimum accuracy is rejected."""
        current = _make_metrics(accuracy=0.80)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", accuracy=0.40
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "min_accuracy" in result.failed_checks

    def test_candidate_fails_f1(self):
        """Candidate below minimum F1 is rejected."""
        current = _make_metrics(f1=0.80)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", f1=0.30
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "min_f1" in result.failed_checks

    def test_candidate_fails_recall(self):
        """Candidate below minimum recall is rejected."""
        current = _make_metrics(recall=0.80)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", recall=0.40
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "min_recall" in result.failed_checks

    def test_candidate_fails_unknown_case_errors(self):
        """Candidate with too many unknown case errors is rejected."""
        current = _make_metrics(unknown_errors=1)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", unknown_errors=10
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "max_unknown_case_errors" in result.failed_checks

    def test_candidate_fails_false_auto_increase(self):
        """Candidate with too large false automation increase is rejected."""
        current = _make_metrics(false_auto=2)
        # 2 → 5 is a 150% increase, exceeds 20% default threshold
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", false_auto=5
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "false_auto_increase" in result.failed_checks

    def test_candidate_fails_false_auto_from_zero(self):
        """Candidate introducing false automation from zero is rejected."""
        current = _make_metrics(false_auto=0)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", false_auto=1
        )
        gate = PromotionGate()
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "false_auto_from_zero" in result.failed_checks

    def test_no_current_model(self):
        """Promotion works even without a current model (first model)."""
        candidate = _make_metrics(model_id="MOD-FIRST", version="1.0.0")
        gate = PromotionGate()
        result = gate.evaluate(None, candidate, "MOD-FIRST", "1.0.0")
        assert result.all_passed is True
        assert result.decision == PromotionDecision.PROMOTED

    def test_custom_thresholds(self):
        """Custom thresholds are applied."""
        thresholds = PromotionThresholds(
            min_precision=0.90,
            min_f1=0.90,
        )
        current = _make_metrics(precision=0.80, f1=0.80)
        candidate = _make_metrics(
            model_id="MOD-CAND", version="2.0.0",
            precision=0.85, f1=0.85,
        )
        gate = PromotionGate(thresholds=thresholds)
        result = gate.evaluate(current, candidate, "MOD-CAND", "2.0.0", "MOD-CUR", "1.0.0")
        assert result.all_passed is False
        assert "min_precision" in result.failed_checks
        assert "min_f1" in result.failed_checks

    def test_gate_has_all_checks(self):
        """Gate produces all expected checks."""
        candidate = _make_metrics()
        gate = PromotionGate()
        result = gate.evaluate(None, candidate, "MOD-001", "1.0.0")
        check_names = [c.check_name for c in result.checks]
        assert "min_precision" in check_names
        assert "min_recall" in check_names
        assert "min_f1" in check_names
        assert "min_accuracy" in check_names
        assert "max_false_automation" in check_names
        assert "max_unknown_case_errors" in check_names

    def test_gate_result_has_metadata(self):
        """Gate result has all metadata."""
        candidate = _make_metrics()
        gate = PromotionGate()
        result = gate.evaluate(None, candidate, "MOD-001", "1.0.0")
        assert result.gate_id.startswith("GATE-")
        assert result.candidate_model_id == "MOD-001"
        assert result.candidate_version == "1.0.0"
        assert result.thresholds is not None
        assert result.evaluated_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    """Test model registry promotion and rollback."""

    def test_register_and_get_model(self):
        """Register and retrieve a model."""
        registry = ModelRegistry()
        meta = _make_metadata()
        registry.register_model(meta)
        assert registry.get_model("MOD-001") is not None

    def test_promote_passes(self):
        """Candidate passing gate is promoted."""
        registry = ModelRegistry()
        current_meta = _make_metadata(model_id="MOD-CUR", version="1.0.0")
        current_meta.status = ModelStatus.ACTIVE
        registry.register_model(current_meta)
        registry._active_model_id = "MOD-CUR"

        cand_meta = _make_metadata(model_id="MOD-CAND", version="2.0.0")
        registry.register_model(cand_meta)

        current_metrics = _make_metrics(model_id="MOD-CUR", version="1.0.0")
        cand_metrics = _make_metrics(model_id="MOD-CAND", version="2.0.0", accuracy=0.90)

        record = registry.promote("MOD-CAND", current_metrics, cand_metrics)
        assert record.decision == PromotionDecision.PROMOTED
        assert registry.active_model_id == "MOD-CAND"
        assert registry.get_model("MOD-CAND").status == ModelStatus.ACTIVE
        assert registry.get_model("MOD-CUR").status == ModelStatus.RETIRED

    def test_promote_rejects(self):
        """Candidate failing gate is rejected."""
        registry = ModelRegistry()
        current_meta = _make_metadata(model_id="MOD-CUR", version="1.0.0")
        current_meta.status = ModelStatus.ACTIVE
        registry.register_model(current_meta)
        registry._active_model_id = "MOD-CUR"

        cand_meta = _make_metadata(model_id="MOD-CAND", version="2.0.0")
        registry.register_model(cand_meta)

        current_metrics = _make_metrics(model_id="MOD-CUR", version="1.0.0", precision=0.90)
        cand_metrics = _make_metrics(
            model_id="MOD-CAND", version="2.0.0", precision=0.50
        )

        record = registry.promote("MOD-CAND", current_metrics, cand_metrics)
        assert record.decision == PromotionDecision.REJECTED
        assert registry.active_model_id == "MOD-CUR"  # Unchanged
        assert registry.get_model("MOD-CAND").status == ModelStatus.REJECTED

    def test_promote_nonexistent_raises(self):
        """Promoting non-existent model raises error."""
        registry = ModelRegistry()
        with pytest.raises(ValueError):
            registry.promote("MOD-NONE", None, _make_metrics())

    def test_first_model_promotion(self):
        """First model is promoted without a current model."""
        registry = ModelRegistry()
        cand_meta = _make_metadata(model_id="MOD-FIRST", version="1.0.0")
        registry.register_model(cand_meta)

        cand_metrics = _make_metrics(model_id="MOD-FIRST", version="1.0.0")
        record = registry.promote("MOD-FIRST", None, cand_metrics)
        assert record.decision == PromotionDecision.PROMOTED
        assert registry.active_model_id == "MOD-FIRST"
        assert record.old_model_id is None

    def test_promotion_history(self):
        """Promotion history is recorded."""
        registry = ModelRegistry()
        meta = _make_metadata(model_id="MOD-001", version="1.0.0")
        meta.status = ModelStatus.ACTIVE
        registry.register_model(meta)
        registry._active_model_id = "MOD-001"

        cand = _make_metadata(model_id="MOD-002", version="2.0.0")
        registry.register_model(cand)

        registry.promote("MOD-002", _make_metrics(), _make_metrics(model_id="MOD-002"))
        history = registry.get_promotion_history()
        assert len(history) == 1
        assert history[0].decision == PromotionDecision.PROMOTED

    def test_rollback(self):
        """Rollback restores previous model."""
        registry = ModelRegistry()
        # Set up initial active
        meta1 = _make_metadata(model_id="MOD-V1", version="1.0.0")
        meta1.status = ModelStatus.ACTIVE
        registry.register_model(meta1)
        registry._active_model_id = "MOD-V1"

        # Promote v2
        meta2 = _make_metadata(model_id="MOD-V2", version="2.0.0")
        registry.register_model(meta2)
        registry.promote(
            "MOD-V2",
            _make_metrics(model_id="MOD-V1"),
            _make_metrics(model_id="MOD-V2"),
        )
        assert registry.active_model_id == "MOD-V2"

        # Rollback
        rb = registry.rollback("Regression detected")
        assert rb is not None
        assert rb.restored_to_version == "1.0.0"
        assert rb.rolled_back_from_version == "2.0.0"
        assert registry.active_model_id == "MOD-V1"
        assert registry.get_model("MOD-V1").status == ModelStatus.ACTIVE
        assert registry.get_model("MOD-V2").status == ModelStatus.RETIRED

    def test_rollback_no_previous(self):
        """Rollback with no previous model returns None."""
        registry = ModelRegistry()
        result = registry.rollback("No reason")
        assert result is None

    def test_rollback_history(self):
        """Rollback history is recorded."""
        registry = ModelRegistry()
        # Set up v1 as active
        m1 = _make_metadata(model_id="MOD-V1", version="1.0.0")
        m1.status = ModelStatus.ACTIVE
        registry.register_model(m1)
        registry._active_model_id = "MOD-V1"

        # Promote v2
        m2 = _make_metadata(model_id="MOD-V2", version="2.0.0")
        registry.register_model(m2)
        registry.promote("MOD-V2", _make_metrics(), _make_metrics(model_id="MOD-V2"))

        # Rollback
        registry.rollback("Testing rollback")
        history = registry.get_rollback_history()
        assert len(history) == 1
        assert history[0].reason == "Testing rollback"

    def test_get_models_by_status(self):
        """Filter models by status."""
        registry = ModelRegistry()
        m1 = _make_metadata(model_id="MOD-A", status=ModelStatus.ACTIVE)
        m2 = _make_metadata(model_id="MOD-B", status=ModelStatus.CANDIDATE)
        m3 = _make_metadata(model_id="MOD-C", status=ModelStatus.REJECTED)
        registry.register_model(m1)
        registry.register_model(m2)
        registry.register_model(m3)

        active = registry.get_models_by_status(ModelStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].model_id == "MOD-A"

    def test_multiple_promotions(self):
        """Multiple sequential promotions work correctly."""
        registry = ModelRegistry()

        # v1 → active
        m1 = _make_metadata(model_id="MOD-V1", version="1.0.0")
        m1.status = ModelStatus.ACTIVE
        registry.register_model(m1)
        registry._active_model_id = "MOD-V1"

        # v2
        m2 = _make_metadata(model_id="MOD-V2", version="2.0.0")
        registry.register_model(m2)
        registry.promote("MOD-V2", _make_metrics(), _make_metrics(model_id="MOD-V2"))
        assert registry.active_model_id == "MOD-V2"

        # v3
        m3 = _make_metadata(model_id="MOD-V3", version="3.0.0")
        registry.register_model(m3)
        registry.promote("MOD-V3", _make_metrics(), _make_metrics(model_id="MOD-V3"))
        assert registry.active_model_id == "MOD-V3"
        assert registry.get_model("MOD-V1").status == ModelStatus.RETIRED
        assert registry.get_model("MOD-V2").status == ModelStatus.RETIRED
        assert registry.get_model("MOD-V3").status == ModelStatus.ACTIVE

    def test_record_has_thresholds(self):
        """Promotion record captures thresholds used."""
        registry = ModelRegistry()
        meta = _make_metadata(model_id="MOD-001", version="1.0.0")
        registry.register_model(meta)
        record = registry.promote("MOD-001", None, _make_metrics())
        assert record.thresholds is not None
        assert record.thresholds.min_precision == 0.70

    def test_record_has_metrics_summary(self):
        """Promotion record captures metrics summary."""
        registry = ModelRegistry()
        meta = _make_metadata(model_id="MOD-001", version="1.0.0")
        registry.register_model(meta)
        current_m = _make_metrics(accuracy=0.75)
        cand_m = _make_metrics(model_id="MOD-001", accuracy=0.85)
        record = registry.promote("MOD-001", current_m, cand_m)
        assert record.old_metrics_summary is not None
        assert record.new_metrics_summary is not None
        assert record.old_metrics_summary["accuracy"] == 0.75
        assert record.new_metrics_summary["accuracy"] == 0.85
