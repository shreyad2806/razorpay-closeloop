"""
Centralized enumerations for the Razorpay CloseLoop financial data contract.

All string-based categories are defined here to avoid scattered string literals.
"""

from enum import Enum


class ExceptionType(str, Enum):
    """Taxonomy of financial reconciliation exceptions."""

    EXACT_MATCH = "EXACT_MATCH"
    FEE_DIFFERENCE = "FEE_DIFFERENCE"
    REFUND_ADJUSTMENT = "REFUND_ADJUSTMENT"
    TAX_ADJUSTMENT = "TAX_ADJUSTMENT"
    TIMING_DIFFERENCE = "TIMING_DIFFERENCE"
    PARTIAL_SETTLEMENT = "PARTIAL_SETTLEMENT"
    DUPLICATE = "DUPLICATE"
    MISSING_RECORD = "MISSING_RECORD"
    COMPLEX_MULTI_ADJUSTMENT = "COMPLEX_MULTI_ADJUSTMENT"
    UNKNOWN = "UNKNOWN"


class ResolutionType(str, Enum):
    """Controlled set of resolution labels for reconciled cases."""

    NO_ACTION = "NO_ACTION"
    FEE_ADJUSTMENT = "FEE_ADJUSTMENT"
    REFUND_ADJUSTMENT = "REFUND_ADJUSTMENT"
    TAX_ADJUSTMENT = "TAX_ADJUSTMENT"
    TIMING_RECONCILIATION = "TIMING_RECONCILIATION"
    PARTIAL_SETTLEMENT_RECONCILIATION = "PARTIAL_SETTLEMENT_RECONCILIATION"
    DUPLICATE_SETTLEMENT = "DUPLICATE_SETTLEMENT"
    MISSING_RECORD_ESCALATION = "MISSING_RECORD_ESCALATION"
    MULTI_ADJUSTMENT = "MULTI_ADJUSTMENT"
    UNKNOWN_UNRESOLVED = "UNKNOWN_UNRESOLVED"


class RiskCategory(str, Enum):
    """Risk levels for financial cases, independent of exception type."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PaymentStatus(str, Enum):
    """Status values for payment records."""

    CAPTURED = "CAPTURED"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"
    PENDING = "PENDING"


class SettlementStatus(str, Enum):
    """Status values for settlement records."""

    SETTLED = "SETTLED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    PROCESSING = "PROCESSING"


class RefundStatus(str, Enum):
    """Status values for refund records."""

    PROCESSED = "PROCESSED"
    PENDING = "PENDING"
    FAILED = "FAILED"


class FeeType(str, Enum):
    """Types of fees applied to payments."""

    TRANSACTION = "TRANSACTION"
    PLATFORM = "PLATFORM"
    TDR = "TDR"
    GST_ON_FEES = "GST_ON_FEES"
    REFUND_FEE = "REFUND_FEE"
    CHARGEBACK_FEE = "CHARGEBACK_FEE"


class TaxType(str, Enum):
    """Types of taxes applied to payments."""

    GST = "GST"
    TDS = "TDS"
    GST_ON_FEES = "GST_ON_FEES"
    SERVICE_TAX = "SERVICE_TAX"


class AdjustmentType(str, Enum):
    """Types of financial adjustments."""

    CREDIT = "CREDIT"
    DEBIT = "DEBIT"
    FEE_REVERSAL = "FEE_REVERSAL"
    PENALTY = "PENALTY"
    BONUS = "BONUS"
    CORRECTION = "CORRECTION"


class Currency(str, Enum):
    """Supported currencies using ISO 4217 codes."""

    INR = "INR"
    USD = "USD"


class MissingRecordSubtype(str, Enum):
    """Subtypes for MISSING_RECORD exception scenarios."""

    MISSING_SETTLEMENT = "MISSING_SETTLEMENT"
    MISSING_REFUND = "MISSING_REFUND"
    MISSING_FEE = "MISSING_FEE"
    MISSING_TAX = "MISSING_TAX"
    MISSING_ADJUSTMENT = "MISSING_ADJUSTMENT"


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation Enums
# ─────────────────────────────────────────────────────────────────────────────


class MatchStatus(str, Enum):
    """Deterministic match status for reconciliation results."""

    MATCHED = "MATCHED"
    EXCEPTION = "EXCEPTION"
    MISSING = "MISSING"
    DUPLICATE = "DUPLICATE"


class ReconciliationStatus(str, Enum):
    """Processing status for reconciliation results."""

    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
