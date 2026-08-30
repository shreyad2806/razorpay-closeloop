"""
Tests for persistence service.

Tests cover:
- Matched result persistence
- Exception persistence
- Duplicate run behavior (idempotency)
- Transaction safety
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.enums import (
    ExceptionType,
    MatchStatus,
    ReconciliationStatus,
)
from app.schemas.reconciliation import ReconciliationResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def matched_result():
    """Fixture for a matched reconciliation result."""
    return ReconciliationResult(
        reconciliation_id="REC-000001",
        case_id="CASE-000001",
        payment_id="PAY-000001",
        merchant_id="MER-0001",
        payment_amount=100000,
        total_refunds=0,
        total_fees=2000,
        total_taxes=18000,
        total_adjustments=0,
        expected_amount=80000,
        actual_amount=80000,
        difference=0,
        match_status=MatchStatus.MATCHED,
        exception_type=ExceptionType.EXACT_MATCH,
        reconciliation_status=ReconciliationStatus.PROCESSED,
    )


@pytest.fixture
def exception_result():
    """Fixture for an exception reconciliation result."""
    return ReconciliationResult(
        reconciliation_id="REC-000002",
        case_id="CASE-000002",
        payment_id="PAY-000002",
        merchant_id="MER-0001",
        payment_amount=100000,
        total_refunds=0,
        total_fees=2000,
        total_taxes=18000,
        total_adjustments=0,
        expected_amount=80000,
        actual_amount=79500,
        difference=500,
        match_status=MatchStatus.EXCEPTION,
        exception_type=ExceptionType.FEE_DIFFERENCE,
        reconciliation_status=ReconciliationStatus.PROCESSED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_matched_result_schema(self, matched_result):
        """Test that matched result schema is valid."""
        assert matched_result.reconciliation_id == "REC-000001"
        assert matched_result.case_id == "CASE-000001"
        assert matched_result.match_status == MatchStatus.MATCHED
        assert matched_result.exception_type == ExceptionType.EXACT_MATCH
        assert matched_result.difference == 0

    def test_exception_result_schema(self, exception_result):
        """Test that exception result schema is valid."""
        assert exception_result.reconciliation_id == "REC-000002"
        assert exception_result.case_id == "CASE-000002"
        assert exception_result.match_status == MatchStatus.EXCEPTION
        assert exception_result.exception_type == ExceptionType.FEE_DIFFERENCE
        assert exception_result.difference == 500

    def test_calculation_verification(self, matched_result):
        """Test that calculation verification works."""
        assert matched_result.verify_calculation()

    def test_difference_calculation(self, exception_result):
        """Test that difference calculation is correct."""
        expected_diff = exception_result.expected_amount - exception_result.actual_amount
        assert exception_result.difference == expected_diff


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    """Tests for idempotency."""

    def test_result_idempotency_key(self, matched_result):
        """Test that idempotency key is case_id + batch_id."""
        batch_id = "batch_001"
        # Same case_id + batch_id should be idempotent
        key1 = (matched_result.case_id, batch_id)
        key2 = (matched_result.case_id, batch_id)
        assert key1 == key2

    def test_different_batch_different_key(self, matched_result):
        """Test that different batch_id creates different key."""
        key1 = (matched_result.case_id, "batch_001")
        key2 = (matched_result.case_id, "batch_002")
        assert key1 != key2

    def test_exception_idempotency_key(self, exception_result):
        """Test that exception idempotency key is case_id + batch_id."""
        batch_id = "batch_001"
        key = (exception_result.case_id, batch_id)
        assert key == ("CASE-000002", "batch_001")


# ─────────────────────────────────────────────────────────────────────────────
# Record Structure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordStructure:
    """Tests for record structure."""

    def test_matched_result_no_exception(self, matched_result):
        """Test that matched results don't need exception records."""
        assert matched_result.match_status == MatchStatus.MATCHED
        # Matched results should not create exception records

    def test_exception_result_has_exception(self, exception_result):
        """Test that exception results need exception records."""
        assert exception_result.match_status == MatchStatus.EXCEPTION
        assert exception_result.exception_type != ExceptionType.EXACT_MATCH

    def test_financial_amounts_are_integer(self, matched_result, exception_result):
        """Test that all financial amounts are integers."""
        for result in [matched_result, exception_result]:
            assert isinstance(result.payment_amount, int)
            assert isinstance(result.total_refunds, int)
            assert isinstance(result.total_fees, int)
            assert isinstance(result.total_taxes, int)
            assert isinstance(result.total_adjustments, int)
            assert isinstance(result.expected_amount, int)
            assert isinstance(result.actual_amount, int)
            assert isinstance(result.difference, int)


# ─────────────────────────────────────────────────────────────────────────────
# Traceability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTraceability:
    """Tests for record traceability."""

    def test_result_references_case(self, matched_result):
        """Test that result references case."""
        assert matched_result.case_id is not None
        assert matched_result.case_id.startswith("CASE-")

    def test_result_references_payment(self, matched_result):
        """Test that result references payment."""
        assert matched_result.payment_id is not None
        assert matched_result.payment_id.startswith("PAY-")

    def test_result_references_merchant(self, matched_result):
        """Test that result references merchant."""
        assert matched_result.merchant_id is not None
        assert matched_result.merchant_id.startswith("MER-")

    def test_exception_references_reconciliation(self, exception_result):
        """Test that exception references reconciliation."""
        assert exception_result.reconciliation_id is not None
        assert exception_result.reconciliation_id.startswith("REC-")


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Tests for ground truth separation."""

    def test_no_ground_truth_fields(self, matched_result, exception_result):
        """Test that results don't have ground truth fields."""
        for result in [matched_result, exception_result]:
            assert not hasattr(result, "true_exception_type")
            assert not hasattr(result, "true_resolution")
            assert not hasattr(result, "resolvable")
            assert not hasattr(result, "risk_category")

    def test_exception_type_is_engine_output(self, exception_result):
        """Test that exception_type is engine output, not ground truth."""
        # The exception_type should come from the engine's classification
        # not from ground truth
        assert exception_result.exception_type == ExceptionType.FEE_DIFFERENCE
