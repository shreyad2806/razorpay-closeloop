"""
CloseLoop database models.

Importing this package registers every model with ``Base.metadata``. Anything
that needs the complete schema (Alembic autogenerate, ``create_all``, the test
harness) should import ``app.models`` rather than individual modules, because a
model that is never imported never becomes a table.

CloseLoop 2.0 naming note: ``app.models.merchant.Merchant`` is the SQLAlchemy
table for a merchant. ``app.schemas.financial.Merchant`` is a separate Pydantic
contract used by the synthetic generator; the two are unrelated and both are
intentional.
"""

# --- Reference / financial layer --------------------------------------------
from app.models.adjustment import Adjustment
from app.models.chargeback import Chargeback
from app.models.fee import Fee
from app.models.ledger_entry import LedgerEntry
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.payment_attempt import PaymentAttempt
from app.models.refund import Refund
from app.models.settlement import Settlement
from app.models.settlement_line import SettlementLine
from app.models.tax import Tax

# --- Reconciliation layer ---------------------------------------------------
from app.models.reconciliation import (
    ReconciliationEvidence,
    ReconciliationResult,
)
from app.models.reconciliation_run import ReconciliationRun

# --- Exception & investigation layer ----------------------------------------
from app.models.audit_event import AuditEvent
from app.models.evidence import Evidence
from app.models.evidence_link import EvidenceLink
from app.models.exception import ExceptionStatus, FinancialException
from app.models.feedback import Feedback
from app.models.model_prediction import ModelPrediction
from app.models.root_cause import RootCause

# --- Resolution layer -------------------------------------------------------
from app.models.approval import Approval
from app.models.resolution import Resolution
from app.models.resolution_action import ResolutionAction

# --- Historical memory ------------------------------------------------------
#
# ``historical_cases`` and ``case_embeddings`` are declared in the service
# modules that own them, following the pre-existing repository convention. They
# are re-exported here so that ``import app.models`` really does register the
# complete schema, as this package promises. Importing them creates no cycle:
# neither service module imports ``app.models``.
#
# Consolidating these two models into ``app/models/`` is a Phase 6 concern (the
# phase that owns historical memory) and is listed in the Phase 1 report.
from app.models.historical_resolution import HistoricalResolution
from app.services.historical_case_store import HistoricalCaseRecord
from app.services.similarity_service import CaseEmbedding

# --- Infrastructure ---------------------------------------------------------
from app.models.immutability import ImmutableRecordError

__all__ = [
    # Financial entities
    "Merchant",
    "Payment",
    "PaymentAttempt",
    "Settlement",
    "SettlementLine",
    "Refund",
    "Chargeback",
    "Fee",
    "Tax",
    "Adjustment",
    "LedgerEntry",
    # Reconciliation
    "ReconciliationRun",
    "ReconciliationResult",
    "ReconciliationEvidence",
    # Exception & investigation
    "FinancialException",
    "ExceptionStatus",
    "Evidence",
    "EvidenceLink",
    "RootCause",
    "ModelPrediction",
    "AuditEvent",
    "Feedback",
    # Resolution
    "Resolution",
    "ResolutionAction",
    "Approval",
    # Historical
    "HistoricalCaseRecord",
    "CaseEmbedding",
    "HistoricalResolution",
    # Infrastructure
    "ImmutableRecordError",
]
