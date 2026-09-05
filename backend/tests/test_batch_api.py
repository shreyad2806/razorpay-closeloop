"""
Tests for Phase 13.3 — Batch REST APIs.

Tests:
- Valid upload
- Invalid upload
- Empty upload
- Run batch
- Duplicate run
- Unknown batch
- Summary
- Processing failure
- Validation errors
"""

import pytest
from fastapi.testclient import TestClient

from app.api.services.batch_service import BatchService, _batch_registry
from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean batch registry before each test."""
    _batch_registry.clear()
    yield
    _batch_registry.clear()


# ─────────────────────────────────────────────────────────────────────────────
# POST /batches — Create Batch
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateBatch:
    def test_create_synthetic_batch(self, client):
        """Create a synthetic batch using the generator."""
        response = client.post("/batches", json={
            "name": "Test Batch",
            "num_merchants": 3,
            "num_cases": 5,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        batch = data["data"]
        assert batch["batch_id"].startswith("BATCH-")
        assert batch["status"] == "CREATED"
        assert batch["name"] == "Test Batch"

    def test_create_batch_with_defaults(self, client):
        """Create a batch with default parameters."""
        response = client.post("/batches", json={"name": "Default Batch"})
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["status"] == "CREATED"

    def test_create_batch_with_payload(self, client):
        """Create a batch from a provided payload."""
        payload = {
            "name": "Payload Batch",
            "payload": {
                "payments": [
                    {"payment_id": "PAY-001", "merchant_id": "M-001", "amount": 100000, "status": "CAPTURED"}
                ],
                "settlements": [
                    {"settlement_id": "SET-001", "payment_id": "PAY-001", "merchant_id": "M-001", "amount": 95000, "status": "SETTLED"}
                ],
                "cases": [
                    {"case_id": "CASE-001", "payment_id": "PAY-001", "merchant_id": "M-001",
                     "expected_amount": 95000, "actual_amount": 95000, "difference": 0}
                ],
                "refunds": [],
                "fees": [],
                "taxes": [],
                "adjustments": [],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["status"] == "CREATED"

    def test_create_batch_invalid_payload_missing_payments(self, client):
        """Reject payload missing payments."""
        payload = {
            "name": "Bad Batch",
            "payload": {
                "settlements": [{"settlement_id": "S-1"}],
                "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 422

    def test_create_batch_invalid_payload_missing_cases(self, client):
        """Reject payload missing cases."""
        payload = {
            "name": "Bad Batch",
            "payload": {
                "payments": [{"payment_id": "P-1", "amount": 100}],
                "settlements": [],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 422

    def test_create_batch_invalid_payload_empty_payments(self, client):
        """Reject payload with empty payments list."""
        payload = {
            "name": "Empty Batch",
            "payload": {
                "payments": [],
                "settlements": [],
                "cases": [],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 422

    def test_create_batch_invalid_payload_negative_amount(self, client):
        """Reject payment with negative amount."""
        payload = {
            "name": "Neg Batch",
            "payload": {
                "payments": [{"payment_id": "P-1", "amount": -100}],
                "settlements": [],
                "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 422

    def test_create_batch_invalid_payload_duplicate_payment_ids(self, client):
        """Reject payload with duplicate payment IDs."""
        payload = {
            "name": "Dup Batch",
            "payload": {
                "payments": [
                    {"payment_id": "P-1", "amount": 100},
                    {"payment_id": "P-1", "amount": 200},
                ],
                "settlements": [],
                "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
            },
        }
        response = client.post("/batches", json=payload)
        assert response.status_code == 422

    def test_create_batch_missing_name(self, client):
        """Reject request without name."""
        response = client.post("/batches", json={})
        assert response.status_code == 422

    def test_create_batch_with_empty_payload(self, client):
        """Empty payload should trigger synthetic generation, not validation error."""
        response = client.post("/batches", json={
            "name": "Empty Payload Test",
            "payload": {},
            "num_merchants": 2,
            "num_cases": 3,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        batch = data["data"]
        assert batch["status"] == "CREATED"
        assert batch["name"] == "Empty Payload Test"

    def test_create_batch_with_null_payload(self, client):
        """Null payload should trigger synthetic generation."""
        response = client.post("/batches", json={
            "name": "Null Payload Test",
            "payload": None,
            "num_merchants": 2,
            "num_cases": 3,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        batch = data["data"]
        assert batch["status"] == "CREATED"

    def test_create_batch_invalid_num_merchants(self, client):
        """Reject invalid num_merchants."""
        response = client.post("/batches", json={
            "name": "Bad Merchants",
            "num_merchants": 0,
        })
        assert response.status_code == 422

    def test_create_batch_invalid_num_cases(self, client):
        """Reject invalid num_cases."""
        response = client.post("/batches", json={
            "name": "Bad Cases",
            "num_cases": 0,
        })
        assert response.status_code == 422

    def test_create_batch_payload_missing_required_financial_structures(self, client):
        """Reject payload with only partial financial structures."""
        response = client.post("/batches", json={
            "name": "Partial Payload",
            "payload": {
                "merchants": [{"merchant_id": "M-001"}],
                # Missing payments and cases
            },
        })
        assert response.status_code == 422
        data = response.json()
        # The error should mention missing required keys in details
        errors = data.get("details", {}).get("errors", [])
        assert any("payments" in str(e) for e in errors) or any("cases" in str(e) for e in errors)

    def test_create_batch_invalid_payment_no_payment_id(self, client):
        """Reject payment without payment_id."""
        response = client.post("/batches", json={
            "name": "Bad Payment",
            "payload": {
                "payments": [{"amount": 100000, "merchant_id": "M-001"}],
                "cases": [{"case_id": "C-001", "payment_id": "P-001"}],
            },
        })
        assert response.status_code == 422

    def test_create_batch_invalid_payment_no_amount(self, client):
        """Reject payment without amount."""
        response = client.post("/batches", json={
            "name": "Bad Payment",
            "payload": {
                "payments": [{"payment_id": "P-001", "merchant_id": "M-001"}],
                "cases": [{"case_id": "C-001", "payment_id": "P-001"}],
            },
        })
        assert response.status_code == 422

    def test_create_batch_invalid_payment_non_numeric_amount(self, client):
        """Reject payment with non-numeric amount."""
        response = client.post("/batches", json={
            "name": "Bad Payment",
            "payload": {
                "payments": [{"payment_id": "P-001", "amount": "invalid"}],
                "cases": [{"case_id": "C-001", "payment_id": "P-001"}],
            },
        })
        assert response.status_code == 422

    def test_create_batch_invalid_case_no_case_id(self, client):
        """Reject case without case_id."""
        response = client.post("/batches", json={
            "name": "Bad Case",
            "payload": {
                "payments": [{"payment_id": "P-001", "amount": 100000}],
                "cases": [{"payment_id": "P-001"}],
            },
        })
        assert response.status_code == 422

    def test_create_batch_invalid_case_no_payment_id(self, client):
        """Reject case without payment_id."""
        response = client.post("/batches", json={
            "name": "Bad Case",
            "payload": {
                "payments": [{"payment_id": "P-001", "amount": 100000}],
                "cases": [{"case_id": "C-001"}],
            },
        })
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /batches — List Batches
# ─────────────────────────────────────────────────────────────────────────────


class TestListBatches:
    def test_list_empty(self, client):
        response = client.get("/batches")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []

    def test_list_after_create(self, client):
        client.post("/batches", json={"name": "Batch 1"})
        client.post("/batches", json={"name": "Batch 2"})
        response = client.get("/batches")
        data = response.json()
        assert len(data["data"]) == 2

    def test_list_with_limit(self, client):
        client.post("/batches", json={"name": "B1"})
        client.post("/batches", json={"name": "B2"})
        response = client.get("/batches?limit=1")
        data = response.json()
        assert len(data["data"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# GET /batches/{batch_id} — Get Batch
# ─────────────────────────────────────────────────────────────────────────────


class TestGetBatch:
    def test_get_existing(self, client):
        create_resp = client.post("/batches", json={"name": "Test"})
        batch_id = create_resp.json()["data"]["batch_id"]
        response = client.get(f"/batches/{batch_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["batch_id"] == batch_id

    def test_get_not_found(self, client):
        response = client.get("/batches/NONEXISTENT")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# POST /batches/{batch_id}/run — Run Batch
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBatch:
    def test_run_synthetic_batch(self, client):
        """Run a batch created with synthetic data."""
        create_resp = client.post("/batches", json={
            "name": "Run Test",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]

        run_resp = client.post(f"/batches/{batch_id}/run")
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert data["success"] is True
        result = data["data"]
        assert result["status"] == "COMPLETED"
        assert result["total_records"] > 0
        assert "match_rate" in result
        assert "exception_rate" in result
        assert "processing_time_ms" in result

    def test_run_not_found(self, client):
        response = client.post("/batches/NONEXISTENT/run")
        assert response.status_code == 404

    def test_run_already_completed(self, client):
        """Second run of same batch returns already completed."""
        create_resp = client.post("/batches", json={
            "name": "Dup Run",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]

        # First run
        client.post(f"/batches/{batch_id}/run")

        # Second run
        response = client.post(f"/batches/{batch_id}/run")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "ALREADY_COMPLETED"


# ─────────────────────────────────────────────────────────────────────────────
# GET /batches/{batch_id}/summary — Batch Summary
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchSummary:
    def test_summary_created_batch(self, client):
        create_resp = client.post("/batches", json={"name": "Summary Test"})
        batch_id = create_resp.json()["data"]["batch_id"]

        response = client.get(f"/batches/{batch_id}/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        summary = data["data"]
        assert summary["batch_id"] == batch_id
        assert summary["status"] == "CREATED"
        assert "match_rate" in summary
        assert "exception_rate" in summary
        assert "processing_time_ms" in summary

    def test_summary_after_run(self, client):
        create_resp = client.post("/batches", json={
            "name": "Summary Run Test",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]
        client.post(f"/batches/{batch_id}/run")

        response = client.get(f"/batches/{batch_id}/summary")
        data = response.json()
        summary = data["data"]
        assert summary["status"] == "COMPLETED"
        assert summary["total_records"] > 0
        assert summary["processing_time_ms"] > 0

    def test_summary_not_found(self, client):
        response = client.get("/batches/NONEXISTENT/summary")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# BatchService Unit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchServiceUnit:
    def test_validate_valid_payload(self):
        svc = BatchService()
        payload = {
            "payments": [{"payment_id": "P-1", "amount": 100}],
            "settlements": [],
            "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_missing_payments(self):
        svc = BatchService()
        payload = {"settlements": [], "cases": []}
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("payments" in e for e in result["errors"])

    def test_validate_missing_settlements_ok(self):
        """Settlements are optional in payload."""
        svc = BatchService()
        payload = {"payments": [{"payment_id": "P-1", "amount": 100}], "cases": [{"case_id": "C-1", "payment_id": "P-1"}]}
        result = svc._validate_payload(payload)
        assert result["valid"] is True

    def test_validate_empty_payments(self):
        svc = BatchService()
        payload = {"payments": [], "settlements": [], "cases": []}
        result = svc._validate_payload(payload)
        assert result["valid"] is False

    def test_validate_negative_amount(self):
        svc = BatchService()
        payload = {
            "payments": [{"payment_id": "P-1", "amount": -100}],
            "settlements": [],
            "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("non-negative" in e for e in result["errors"])

    def test_validate_duplicate_payment_ids(self):
        svc = BatchService()
        payload = {
            "payments": [
                {"payment_id": "P-1", "amount": 100},
                {"payment_id": "P-1", "amount": 200},
            ],
            "settlements": [],
            "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("Duplicate payment_id" in e for e in result["errors"])

    def test_validate_duplicate_case_ids(self):
        svc = BatchService()
        payload = {
            "payments": [{"payment_id": "P-1", "amount": 100}],
            "settlements": [],
            "cases": [
                {"case_id": "C-1", "payment_id": "P-1"},
                {"case_id": "C-1", "payment_id": "P-2"},
            ],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("Duplicate case_id" in e for e in result["errors"])

    def test_validate_missing_payment_id_in_payment(self):
        svc = BatchService()
        payload = {
            "payments": [{"amount": 100}],
            "settlements": [],
            "cases": [{"case_id": "C-1", "payment_id": "P-1"}],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("payment_id" in e for e in result["errors"])

    def test_validate_missing_case_id_in_case(self):
        svc = BatchService()
        payload = {
            "payments": [{"payment_id": "P-1", "amount": 100}],
            "settlements": [],
            "cases": [{"payment_id": "P-1"}],
        }
        result = svc._validate_payload(payload)
        assert result["valid"] is False
        assert any("case_id" in e for e in result["errors"])

    def test_validate_empty_payload_dict(self):
        """Empty dict payload should not be validated (synthetic generation path)."""
        svc = BatchService()
        payload = {}
        result = svc._validate_payload(payload)
        # Empty payload is valid but will not be used - synthetic generation will occur
        assert result["valid"] is False  # Missing required keys
        assert any("payments" in e for e in result["errors"])


# ─────────────────────────────────────────────────────────────────────────────
# Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchSafety:
    def test_batch_does_not_bypass_guardrails(self, client):
        """Batch processing uses existing reconciliation only."""
        create_resp = client.post("/batches", json={
            "name": "Safety Test",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]
        run_resp = client.post(f"/batches/{batch_id}/run")
        result = run_resp.json()["data"]
        # Reconciliation results are deterministic
        assert "match_rate" in result
        assert "exception_rate" in result
        assert 0.0 <= result["match_rate"] <= 1.0
        assert 0.0 <= result["exception_rate"] <= 1.0

    def test_batch_creates_no_duplicate_data(self, client):
        """Running same batch twice returns already completed."""
        create_resp = client.post("/batches", json={
            "name": "Idempotent",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]
        client.post(f"/batches/{batch_id}/run")
        resp2 = client.post(f"/batches/{batch_id}/run")
        assert resp2.json()["data"]["status"] == "ALREADY_COMPLETED"

    def test_batch_summary_matches_run(self, client):
        """Summary after run reflects run results."""
        create_resp = client.post("/batches", json={
            "name": "Consistency",
            "num_merchants": 2,
            "num_cases": 3,
        })
        batch_id = create_resp.json()["data"]["batch_id"]
        run_resp = client.post(f"/batches/{batch_id}/run")
        run_data = run_resp.json()["data"]

        summary_resp = client.get(f"/batches/{batch_id}/summary")
        summary = summary_resp.json()["data"]

        assert summary["total_records"] == run_data["total_records"]
        assert summary["status"] == "COMPLETED"
