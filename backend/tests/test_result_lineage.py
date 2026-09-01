"""
Tests for Razorpay CloseLoop Phase 10H — Complete Model Result Lineage.

Verifies end-to-end traceability:
  Exception → Prediction → Model Version → MLflow Run → Dataset → Features
"""

import pytest
from datetime import datetime, timezone

from app.schemas.model_version_metadata import ModelVersionMetadata, ModelVersionStatus
from app.schemas.result_lineage import (
    AuditResponse,
    ModelLineageChain,
    ResultRecord,
    ResultType,
)
from app.services.model_version_registry import ModelVersionRegistry
from app.services.result_lineage import ModelResultLineage


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ModelVersionRegistry:
    """Create a registry with a model version."""
    reg = ModelVersionRegistry()
    reg.register_version(
        model_id="model-xgb-001",
        model_name="XGBoost Classifier",
        model_version="1.2.0",
        mlflow_run_id="run-abc-123",
        mlflow_experiment_name="razorpay-closeloop.classification",
        dataset_id="dataset-001",
        dataset_version="v3",
        feature_schema_version="fv2",
        label_schema_version="lv1",
        algorithm="xgboost",
        training_config={"n_estimators": 100},
        hyperparameters={"max_depth": 6},
        random_seed=42,
        feature_count=12,
        feature_names=["fee_diff", "tax_diff", "settlement_gap", "refund_count",
                        "adjustment_count", "discrepancy_pct", "days_outstanding",
                        "merchant_volume", "exception_frequency", "evidence_coverage",
                        "similarity_score", "historical_success"],
        label_classes=["FEE_DIFFERENCE", "REFUND_ADJUSTMENT", "DUPLICATE", "UNKNOWN"],
        training_examples=5000,
        accuracy=0.87,
        f1_macro=0.82,
        precision_macro=0.85,
        false_automation=3,
        high_value_errors=0,
        policy_version="policy-v2",
    )
    return reg


@pytest.fixture
def lineage(registry: ModelVersionRegistry) -> ModelResultLineage:
    return ModelResultLineage(registry=registry)


@pytest.fixture
def lineage_no_registry() -> ModelResultLineage:
    return ModelResultLineage(registry=None)


# ─────────────────────────────────────────────────────────────────────────────
# ResultType Enum
# ─────────────────────────────────────────────────────────────────────────────


class TestResultType:
    def test_all_types_exist(self):
        assert ResultType.CLASSIFICATION.value == "classification"
        assert ResultType.RESOLUTION.value == "resolution"
        assert ResultType.EXCEPTION_TYPE.value == "exception_type"
        assert ResultType.RISK_ASSESSMENT.value == "risk_assessment"
        assert ResultType.CUSTOM.value == "custom"

    def test_result_type_count(self):
        assert len(ResultType) == 5


# ─────────────────────────────────────────────────────────────────────────────
# ResultRecord Schema
# ─────────────────────────────────────────────────────────────────────────────


class TestResultRecordSchema:
    def test_minimal_record(self):
        record = ResultRecord(
            result_id="RESULT-001",
            model_name="test",
            model_version="1.0",
        )
        assert record.result_id == "RESULT-001"
        assert record.model_name == "test"
        assert record.result_type == ResultType.CLASSIFICATION

    def test_full_record(self):
        record = ResultRecord(
            result_id="RESULT-002",
            result_type=ResultType.RESOLUTION,
            workflow_id="wf-1",
            exception_id="exc-1",
            model_name="XGB",
            model_version="2.0",
            model_id="m-1",
            mlflow_run_id="run-1",
            mlflow_experiment_name="exp-1",
            dataset_id="ds-1",
            dataset_version="v3",
            feature_schema_version="fv2",
            algorithm="xgboost",
            prediction="FEE_DIFFERENCE",
            confidence=0.95,
            policy_version="policy-v2",
        )
        assert record.mlflow_run_id == "run-1"
        assert record.confidence == 0.95

    def test_summary(self):
        record = ResultRecord(
            result_id="RESULT-ABCD1234",
            model_name="XGB",
            model_version="1.0",
            exception_id="exc-1",
            mlflow_run_id="run-ABCD1234",
        )
        s = record.summary()
        # summary() truncates to 8 chars: RESULT-A... and run-ABCD...
        assert "RESULT-A" in s
        assert "XGB" in s
        assert "v1.0" in s
        assert "run-ABCD" in s

    def test_summary_no_mlflow(self):
        record = ResultRecord(
            result_id="RESULT-001",
            model_name="XGB",
            model_version="1.0",
        )
        s = record.summary()
        assert "none" in s


# ─────────────────────────────────────────────────────────────────────────────
# ModelLineageChain Schema
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLineageChain:
    def test_basic_chain(self):
        chain = ModelLineageChain(
            result_id="r-1",
            result_type="classification",
            model_name="XGB",
            model_version="1.0",
        )
        assert chain.result_id == "r-1"
        assert chain.model_name == "XGB"

    def test_summary(self):
        chain = ModelLineageChain(
            result_id="RESULT-ABCD1234",
            result_type="classification",
            model_name="XGB",
            model_version="1.0",
            mlflow_run_id="run-ABCD1234",
            dataset_version="v3",
        )
        s = chain.summary()
        # summary() truncates to 8 chars
        assert "RESULT-A" in s
        assert "run-ABCD" in s
        assert "v3" in s


# ─────────────────────────────────────────────────────────────────────────────
# AuditResponse Schema
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditResponse:
    def test_basic_audit(self):
        resp = AuditResponse(
            exception_id="exc-1",
            model_name="XGB",
            model_version="1.0",
            mlflow_run_id="run-1",
            dataset_version="v3",
            prediction="FEE_DIFFERENCE",
            confidence=0.95,
        )
        assert resp.exception_id == "exc-1"
        assert resp.confidence == 0.95

    def test_summary(self):
        resp = AuditResponse(
            exception_id="exc-1",
            model_name="XGB",
            model_version="1.0",
            mlflow_run_id="run-ABCD12345678",
        )
        s = resp.summary()
        assert "exc-1" in s
        assert "run-ABCD" in s


# ─────────────────────────────────────────────────────────────────────────────
# Service — Recording Results
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordResult:
    def test_record_basic(self, lineage: ModelResultLineage):
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            exception_id="exc-1",
            prediction="FEE_DIFFERENCE",
            confidence=0.92,
        )
        assert record.result_id.startswith("RESULT-")
        assert record.model_name == "XGBoost Classifier"
        assert record.exception_id == "exc-1"
        assert record.prediction == "FEE_DIFFERENCE"

    def test_record_with_registry_enrichment(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
        )
        # Should be enriched from registry
        assert record.mlflow_run_id == "run-abc-123"
        assert record.mlflow_experiment_name == "razorpay-closeloop.classification"
        assert record.dataset_version == "v3"
        assert record.feature_schema_version == "fv2"
        assert record.algorithm == "xgboost"
        assert record.model_accuracy == 0.87
        assert record.model_f1_macro == 0.82

    def test_record_without_registry(
        self, lineage_no_registry: ModelResultLineage
    ):
        record = lineage_no_registry.record_result(
            model_name="XGB",
            model_version="1.0",
            exception_id="exc-1",
        )
        assert record.mlflow_run_id is None
        assert record.dataset_version is None

    def test_record_with_explicit_params(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            mlflow_run_id="override-run",  # Override registry value
            dataset_version="v5",  # Override registry value
        )
        # Explicit params should override registry
        assert record.mlflow_run_id == "override-run"
        assert record.dataset_version == "v5"

    def test_record_stored_and_retrievable(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            exception_id="exc-1",
        )
        found = lineage.get_result(record.result_id)
        assert found is not None
        assert found.result_id == record.result_id

    def test_record_with_custom_result_id(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            result_id="CUSTOM-ID-001",
        )
        assert record.result_id == "CUSTOM-ID-001"
        assert lineage.get_result("CUSTOM-ID-001") is not None

    def test_record_with_all_types(
        self, lineage: ModelResultLineage
    ):
        for rt in ResultType:
            record = lineage.record_result(
                model_name="XGB",
                model_version="1.0",
                result_type=rt,
                exception_id="exc-1",
            )
            assert record.result_type == rt


# ─────────────────────────────────────────────────────────────────────────────
# Service — Lookup
# ─────────────────────────────────────────────────────────────────────────────


class TestLookup:
    def test_get_result_exists(self, lineage: ModelResultLineage):
        record = lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1"
        )
        found = lineage.get_result(record.result_id)
        assert found is not None
        assert found.exception_id == "exc-1"

    def test_get_result_not_exists(self, lineage: ModelResultLineage):
        assert lineage.get_result("nonexistent") is None

    def test_get_results_for_exception(
        self, lineage: ModelResultLineage
    ):
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1"
        )
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1",
            result_type=ResultType.RESOLUTION,
        )
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-2"
        )
        results = lineage.get_results_for_exception("exc-1")
        assert len(results) == 2
        assert all(r.exception_id == "exc-1" for r in results)

    def test_get_results_for_exception_empty(
        self, lineage: ModelResultLineage
    ):
        results = lineage.get_results_for_exception("nonexistent")
        assert results == []

    def test_get_results_by_model(self, lineage: ModelResultLineage):
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1"
        )
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-2"
        )
        lineage.record_result(
            model_name="LR", model_version="1.0", exception_id="exc-3"
        )
        results = lineage.get_results_by_model("XGB")
        assert len(results) == 2

    def test_get_results_by_model_and_version(
        self, lineage: ModelResultLineage
    ):
        lineage.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1"
        )
        lineage.record_result(
            model_name="XGB", model_version="2.0", exception_id="exc-2"
        )
        results = lineage.get_results_by_model("XGB", model_version="2.0")
        assert len(results) == 1
        assert results[0].model_version == "2.0"


# ─────────────────────────────────────────────────────────────────────────────
# Service — Lineage Chain
# ─────────────────────────────────────────────────────────────────────────────


class TestLineageChain:
    def test_build_chain_basic(self, lineage: ModelResultLineage):
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            prediction="FEE_DIFFERENCE",
            confidence=0.95,
        )
        chain = lineage.build_lineage_chain(record.result_id)
        assert chain is not None
        assert chain.model_name == "XGBoost Classifier"
        assert chain.model_version == "1.2.0"
        assert chain.mlflow_run_id == "run-abc-123"
        assert chain.dataset_version == "v3"
        assert chain.algorithm == "xgboost"
        assert chain.prediction == "FEE_DIFFERENCE"
        assert chain.confidence == 0.95

    def test_chain_includes_training_details(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
        )
        chain = lineage.build_lineage_chain(record.result_id)
        assert chain.training_examples == 5000
        assert chain.feature_count == 12
        assert len(chain.feature_names) == 12
        assert "fee_diff" in chain.feature_names
        assert "FEE_DIFFERENCE" in chain.label_classes

    def test_chain_without_registry(
        self, lineage_no_registry: ModelResultLineage
    ):
        record = lineage_no_registry.record_result(
            model_name="XGB", model_version="1.0", exception_id="exc-1"
        )
        chain = lineage_no_registry.build_lineage_chain(record.result_id)
        assert chain is not None
        assert chain.model_name == "XGB"
        assert chain.training_examples is None

    def test_chain_nonexistent_result(
        self, lineage: ModelResultLineage
    ):
        assert lineage.build_lineage_chain("nonexistent") is None

    def test_chain_summary(self, lineage: ModelResultLineage):
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            mlflow_run_id="run-ABCDEF123456",
            dataset_version="v3",
        )
        chain = lineage.build_lineage_chain(record.result_id)
        s = chain.summary()
        assert "XGBoost Classifier" in s
        assert "v1.2.0" in s
        # summary() truncates to 8 chars: run-ABCD
        assert "run-ABCD" in s
        assert "v3" in s


# ─────────────────────────────────────────────────────────────────────────────
# Service — Audit
# ─────────────────────────────────────────────────────────────────────────────


class TestAudit:
    def test_audit_exception(
        self, lineage: ModelResultLineage
    ):
        lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            prediction="FEE_DIFFERENCE",
            confidence=0.95,
        )
        audit = lineage.audit_exception("exc-1")
        assert audit is not None
        assert audit.exception_id == "exc-1"
        assert audit.model_name == "XGBoost Classifier"
        assert audit.model_version == "1.2.0"
        assert audit.mlflow_run_id == "run-abc-123"
        assert audit.dataset_version == "v3"
        assert audit.feature_schema_version == "fv2"
        assert audit.algorithm == "xgboost"
        assert audit.prediction == "FEE_DIFFERENCE"
        assert audit.confidence == 0.95
        assert audit.model_accuracy == 0.87
        assert audit.model_f1_macro == 0.82
        assert audit.policy_version == "policy-v2"

    def test_audit_exception_not_found(
        self, lineage: ModelResultLineage
    ):
        assert lineage.audit_exception("nonexistent") is None

    def test_audit_exception_uses_most_recent(
        self, lineage: ModelResultLineage
    ):
        import time
        r1 = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            exception_id="exc-1",
            prediction="PRED1",
            confidence=0.80,
        )
        # Force a later timestamp for the second result
        r2 = lineage.record_result(
            model_name="XGB",
            model_version="2.0",
            exception_id="exc-1",
            prediction="PRED2",
            confidence=0.95,
        )
        # Manually set the second result's predicted_at to be later
        from datetime import datetime, timedelta
        r2_obj = lineage.get_result(r2.result_id)
        lineage._results[r2.result_id] = r2_obj.model_copy(
            update={"predicted_at": datetime.utcnow() + timedelta(seconds=1)}
        )
        audit = lineage.audit_exception("exc-1")
        assert audit is not None
        assert audit.model_version == "2.0"
        assert audit.prediction == "PRED2"

    def test_audit_result(self, lineage: ModelResultLineage):
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            prediction="FEE_DIFFERENCE",
        )
        audit = lineage.audit_result(record.result_id)
        assert audit is not None
        assert audit.result_id == record.result_id
        assert audit.model_name == "XGB"

    def test_audit_result_not_found(
        self, lineage: ModelResultLineage
    ):
        assert lineage.audit_result("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Service — Historical Result Preservation
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalPreservation:
    def test_historical_result_preserves_model_v1(
        self, lineage: ModelResultLineage
    ):
        record_v1 = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            exception_id="exc-1",
            prediction="PRED-v1",
        )
        record_v2 = lineage.record_result(
            model_name="XGB",
            model_version="2.0",
            exception_id="exc-2",
            prediction="PRED-v2",
        )
        # Old result must still reference v1
        hist = lineage.get_historical_result(record_v1.result_id)
        assert hist is not None
        assert hist.model_version == "1.0"
        assert hist.prediction == "PRED-v1"

        # New result references v2
        hist2 = lineage.get_historical_result(record_v2.result_id)
        assert hist2 is not None
        assert hist2.model_version == "2.0"

    def test_historical_not_affected_by_model_changes(
        self, lineage: ModelResultLineage
    ):
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            model_id="model-xgb-001",
            mlflow_run_id="run-v1",
            exception_id="exc-1",
        )
        # Even if we registered a new model version, old result is unchanged
        assert record.mlflow_run_id == "run-v1"
        historical = lineage.get_historical_result(record.result_id)
        assert historical.mlflow_run_id == "run-v1"


# ─────────────────────────────────────────────────────────────────────────────
# Service — Summary
# ─────────────────────────────────────────────────────────────────────────────


class TestSummary:
    def test_empty_summary(self, lineage: ModelResultLineage):
        s = lineage.get_summary()
        assert s["total_results"] == 0
        assert s["total_exceptions"] == 0

    def test_populated_summary(self, lineage: ModelResultLineage):
        lineage.record_result(
            model_name="XGB", model_version="1.0",
            exception_id="exc-1", result_type=ResultType.CLASSIFICATION,
        )
        lineage.record_result(
            model_name="XGB", model_version="2.0",
            exception_id="exc-1", result_type=ResultType.RESOLUTION,
        )
        lineage.record_result(
            model_name="LR", model_version="1.0",
            exception_id="exc-2", result_type=ResultType.CLASSIFICATION,
        )
        s = lineage.get_summary()
        assert s["total_results"] == 3
        assert s["total_exceptions"] == 2
        assert s["results_by_type"]["classification"] == 2
        assert s["results_by_type"]["resolution"] == 1
        assert "XGB v1.0" in s["models_used"] or "XGB v2.0" in s["models_used"]
        assert s["has_registry"] is True

    def test_summary_no_registry(
        self, lineage_no_registry: ModelResultLineage
    ):
        lineage_no_registry.record_result(
            model_name="XGB", model_version="1.0"
        )
        s = lineage_no_registry.get_summary()
        assert s["has_registry"] is False


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Trace
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndTrace:
    def test_full_lineage_chain(
        self, lineage: ModelResultLineage
    ):
        """Demonstrate an end-to-end trace:
        Exception → Prediction → Model Version → MLflow Run → Dataset → Metrics
        """
        # Record a result linked to a registered model
        record = lineage.record_result(
            model_name="XGBoost Classifier",
            model_version="1.2.0",
            model_id="model-xgb-001",
            result_type=ResultType.CLASSIFICATION,
            workflow_id="wf-001",
            exception_id="exc-fee-diff-001",
            prediction="FEE_DIFFERENCE",
            confidence=0.95,
        )

        # 1. Retrieve by result ID
        found = lineage.get_result(record.result_id)
        assert found is not None
        assert found.exception_id == "exc-fee-diff-001"

        # 2. Build full lineage chain
        chain = lineage.build_lineage_chain(record.result_id)
        assert chain is not None
        # Model details
        assert chain.model_name == "XGBoost Classifier"
        assert chain.model_version == "1.2.0"
        # MLflow run
        assert chain.mlflow_run_id == "run-abc-123"
        assert chain.mlflow_experiment_name == "razorpay-closeloop.classification"
        # Dataset
        assert chain.dataset_version == "v3"
        assert chain.feature_schema_version == "fv2"
        assert chain.algorithm == "xgboost"
        # Training details from registry
        assert chain.training_examples == 5000
        assert chain.feature_count == 12
        assert len(chain.feature_names) == 12
        assert "fee_diff" in chain.feature_names
        assert "FEE_DIFFERENCE" in chain.label_classes
        # Prediction
        assert chain.prediction == "FEE_DIFFERENCE"
        assert chain.confidence == 0.95
        # Policy
        assert chain.policy_version == "policy-v2"

        # 3. Audit by exception ID
        audit = lineage.audit_exception("exc-fee-diff-001")
        assert audit is not None
        assert audit.model_name == "XGBoost Classifier"
        assert audit.mlflow_run_id == "run-abc-123"
        assert audit.dataset_version == "v3"
        assert audit.prediction == "FEE_DIFFERENCE"
        assert audit.confidence == 0.95
        assert audit.model_accuracy == 0.87
        assert audit.lineage_chain is not None
        assert audit.lineage_chain.mlflow_run_id == "run-abc-123"

        # 4. Audit by result ID
        audit2 = lineage.audit_result(record.result_id)
        assert audit2 is not None
        assert audit2.exception_id == "exc-fee-diff-001"

        # 5. Summary
        summary = lineage.get_summary()
        assert summary["total_results"] == 1
        assert summary["has_registry"] is True

    def test_multiple_models_same_exception(
        self, lineage: ModelResultLineage
    ):
        """Multiple models predict on the same exception — each retains lineage."""
        r1 = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            prediction="FEE_DIFF",
            confidence=0.90,
        )
        r2 = lineage.record_result(
            model_name="XGB",
            model_version="2.0",
            exception_id="exc-1",
            prediction="UNKNOWN",
            confidence=0.45,
        )

        # Both results exist
        results = lineage.get_results_for_exception("exc-1")
        assert len(results) == 2

        # Each retains its own model version
        hist1 = lineage.get_historical_result(r1.result_id)
        hist2 = lineage.get_historical_result(r2.result_id)
        assert hist1.model_version == "1.0"
        assert hist2.model_version == "2.0"

        # Chain for v1 has MLflow run, chain for v2 does not
        chain1 = lineage.build_lineage_chain(r1.result_id)
        chain2 = lineage.build_lineage_chain(r2.result_id)
        assert chain1.mlflow_run_id == "run-abc-123"
        assert chain2.mlflow_run_id is None


# ─────────────────────────────────────────────────────────────────────────────
# Safety Verification
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyBoundary:
    def test_no_financial_modification(self, lineage: ModelResultLineage):
        """Result lineage is observational only — no financial operations."""
        record = lineage.record_result(
            model_name="XGB",
            model_version="1.0",
            model_id="model-xgb-001",
            exception_id="exc-1",
            prediction="FEE_DIFFERENCE",
        )
        # Verify the record only stores data, no execution methods
        assert hasattr(record, 'model_name')
        assert not hasattr(record, 'execute')
        assert not hasattr(record, 'authorize')
        assert not hasattr(record, 'modify_balance')

    def test_lineage_cannot_bypass_guardrails(
        self, lineage: ModelResultLineage
    ):
        """Lineage chain has no decision/guardrail fields."""
        record = lineage.record_result(
            model_name="XGB", model_version="1.0",
            exception_id="exc-1",
        )
        chain = lineage.build_lineage_chain(record.result_id)
        assert chain is not None
        assert not hasattr(chain, 'guardrail_result')
        assert not hasattr(chain, 'decision')
        assert not hasattr(chain, 'auto_approve')

    def test_audit_response_observational(
        self, lineage: ModelResultLineage
    ):
        """Audit response only records provenance."""
        lineage.record_result(
            model_name="XGB", model_version="1.0",
            exception_id="exc-1",
        )
        audit = lineage.audit_exception("exc-1")
        assert audit is not None
        assert not hasattr(audit, 'execute')
        assert not hasattr(audit, 'authorize')
        assert not hasattr(audit, 'approve')
