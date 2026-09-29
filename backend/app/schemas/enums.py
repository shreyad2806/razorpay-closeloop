"""
Centralized enumerations for the Razorpay CloseLoop financial data contract.

All string-based categories are defined here to avoid scattered string literals.
"""

from enum import Enum


class ExceptionType(str, Enum):
    """Taxonomy of financial reconciliation exceptions."""

    EXACT_MATCH = "EXACT_MATCH"
    FEE_DIFFERENCE = "FEE_DIFFERENCE"
    FEE_MISMATCH = "FEE_MISMATCH"
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
    """Status values for payment records.

    Mirrors the provider's payment lifecycle (see CLOSELOOP_2.0_ARCHITECTURE.md
    section 7.1). CloseLoop never sets these itself - they are updated only from
    provider payloads during ingestion.
    """

    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    SETTLED = "SETTLED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"
    REFUNDED = "REFUNDED"
    CHARGEBACK = "CHARGEBACK"
    FAILED = "FAILED"
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


# =============================================================================
# CloseLoop 2.0 - Phase 1 domain enumerations
#
# Added by the Phase 1 (domain + database foundation) work. Everything below is
# additive: no pre-existing enum member is renamed or removed, so existing
# services, APIs and tests keep working unchanged.
# =============================================================================


class MerchantStatus(str, Enum):
    """Lifecycle status for a merchant record (soft delete is the only delete)."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class PaymentAttemptKind(str, Enum):
    """Whether a payment attempt authorised or captured funds."""

    AUTH = "AUTH"
    CAPTURE = "CAPTURE"


class PaymentAttemptStatus(str, Enum):
    """Status values for an individual payment attempt."""

    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ChargebackStatus(str, Enum):
    """Chargeback / dispute lifecycle status."""

    OPEN = "OPEN"
    PRE_ARBITRATION = "PRE_ARBITRATION"
    WON = "WON"
    LOST = "LOST"


class SettlementComponentType(str, Enum):
    """Component breakdown of a single settlement line."""

    GROSS = "GROSS"
    FEE = "FEE"
    TAX = "TAX"
    REFUND = "REFUND"
    ADJUSTMENT = "ADJUSTMENT"
    CHARGEBACK = "CHARGEBACK"


class AdjustmentOrigin(str, Enum):
    """Where an adjustment came from.

    PROVIDER            - observed in provider data (read-only mirror).
    CLOSELOOP_EXECUTION - created by CloseLoop as a resolution action.
    """

    PROVIDER = "PROVIDER"
    CLOSELOOP_EXECUTION = "CLOSELOOP_EXECUTION"


class LedgerEntryType(str, Enum):
    """Semantic kind of a ledger entry."""

    PAYMENT = "PAYMENT"
    SETTLEMENT = "SETTLEMENT"
    REFUND = "REFUND"
    CHARGEBACK = "CHARGEBACK"
    FEE = "FEE"
    TAX = "TAX"
    ADJUSTMENT = "ADJUSTMENT"
    REVERSAL = "REVERSAL"


class LedgerDirection(str, Enum):
    """Double-entry direction of a ledger entry."""

    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


class LedgerSource(str, Enum):
    """How the ledger entry entered CloseLoop."""

    INGEST = "INGEST"
    EXECUTION = "EXECUTION"
    ROLLBACK = "ROLLBACK"


class ReconciliationScopeType(str, Enum):
    """Scope a reconciliation run was executed over."""

    BATCH = "BATCH"
    PAYMENT = "PAYMENT"
    EXCEPTION = "EXCEPTION"


class ReconciliationTrigger(str, Enum):
    """What caused a reconciliation run to start."""

    SCHEDULED = "SCHEDULED"
    ON_INGEST = "ON_INGEST"
    POST_RESOLUTION = "POST_RESOLUTION"
    MANUAL = "MANUAL"


class ReconciliationRunStatus(str, Enum):
    """Lifecycle of a single reconciliation run (section 7.2).

    PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExceptionLifecycleStatus(str, Enum):
    """Exception lifecycle (section 7.3).

    ``OPEN`` is retained from the pre-2.0 codebase and is the persistence-time
    synonym for ``DETECTED``: rows created by the existing reconciliation and
    evidence code are written as OPEN and enter the machine there. Nothing
    depends on the distinction, and no existing value is removed.
    """

    OPEN = "OPEN"
    DETECTED = "DETECTED"
    INVESTIGATING = "INVESTIGATING"
    ANALYZED = "ANALYZED"
    RESOLUTION_PROPOSED = "RESOLUTION_PROPOSED"
    AUTO_APPROVED = "AUTO_APPROVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECONCILING = "RECONCILING"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    UNRESOLVED = "UNRESOLVED"
    MATCHED = "MATCHED"
    RESOLVED = "RESOLVED"


class EvidenceRelationship(str, Enum):
    """How a piece of evidence relates to its exception."""

    PRIMARY = "PRIMARY"
    CALCULATION_COMPONENT = "CALCULATION_COMPONENT"
    SUPPORTING = "SUPPORTING"
    CONFLICTING = "CONFLICTING"
    MISSING = "MISSING"


class EvidenceRecordedBy(str, Enum):
    """Provenance of a persisted evidence row."""

    RECONCILIATION = "RECONCILIATION"
    AGENT = "AGENT"
    HUMAN = "HUMAN"


class ResolutionStatus(str, Enum):
    """Resolution lifecycle (section 7.4).

    DRAFT -> PROPOSED -> POLICY_EVALUATED -> (AUTO_APPROVED | HUMAN_REVIEW | BLOCKED)
    AUTO_APPROVED / HUMAN_REVIEW+APPROVED -> EXECUTING -> EXECUTED -> VERIFIED | FAILED
    FAILED -> (ROLLBACK_PENDING -> ROLLED_BACK | ROLLBACK_FAILED)
    BLOCKED -> WITHDRAWN
    """

    DRAFT = "DRAFT"
    PROPOSED = "PROPOSED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    AUTO_APPROVED = "AUTO_APPROVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    WITHDRAWN = "WITHDRAWN"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


class PolicyDecision(str, Enum):
    """Outcome of policy / guardrail evaluation for a resolution."""

    AUTO = "AUTO"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    UNRESOLVED = "UNRESOLVED"
    BLOCKED = "BLOCKED"


class ResolutionActionType(str, Enum):
    """Executable unit kinds."""

    ADJUSTMENT = "ADJUSTMENT"
    REVERSAL = "REVERSAL"


class ResolutionActionStatus(str, Enum):
    """ResolutionAction lifecycle (section 7.5)."""

    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


class ApprovalDecision(str, Enum):
    """Approval lifecycle (section 7.6)."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class ActorType(str, Enum):
    """Who/what produced an audit event, approval or feedback record."""

    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"
    AGENT = "AGENT"


class ApprovalRequester(str, Enum):
    """Who requested an approval."""

    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"


class FeedbackType(str, Enum):
    """Kind of feedback recorded against a resolution outcome."""

    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    CORRECTION = "CORRECTION"
    OUTCOME = "OUTCOME"
    REVIEW = "REVIEW"

