"""
Phase 1 baseline: the frozen CloseLoop 2.0 domain + database foundation.

Revision ID: 0001_phase1_domain
Revises: (none - this is the initial revision)

This revision materialises the complete CloseLoop 2.0 section 5 schema: the ten
pre-existing financial/reconciliation tables (now carrying real foreign keys,
explicit currency, BIGINT paise money and CHECK constraints) plus the Phase 1
additions (merchants, payment_attempts, settlement_lines, chargebacks,
ledger_entries, reconciliation_runs, evidence, root_causes, resolutions,
resolution_actions, approvals, audit_events, model_predictions, feedback).

Why these definitions are frozen text
------------------------------------
The tables below are written out explicitly rather than delegating to
``Base.metadata.create_all``. ``alembic revision --autogenerate`` was not used
because it does not capture CHECK constraints or partial-index WHERE clauses,
and this schema depends on both (architecture section 6). Freezing the
definitions here also keeps revision 0001 immutable: changing ``app/models/*``
later cannot retroactively change what this revision does. Phase 2 onward must
land new forward-only revisions.

Databases built by the old ``init_db.py``
----------------------------------------
``upgrade()`` is safe on a database previously created with
``Base.metadata.create_all``:

* a table that does not exist is created in full - columns, foreign keys,
  CHECK constraints, unique constraints and indexes;
* a table that already exists only receives the columns it is missing, via
  ALTER TABLE. Every Phase 1 column added to a pre-existing table is nullable or
  carries a server default, so no existing row is invalidated and no data is
  dropped.

Known limitation of the in-place path (documented in the Phase 1 report):
adding a *missing constraint* to a pre-existing table is not attempted. SQLite
cannot ALTER a table's constraints at all, and this project has no pre-existing
production database - before this revision the only schema source was
`init_db.py`'s `create_all` against throwaway local databases. A fresh
`alembic upgrade head` produces the complete, fully constrained schema.

Downgrading returns the database to an empty schema because this is the baseline
revision.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001_phase1_domain"
down_revision = None
branch_labels = None
depends_on = None


def _frozen_schema() -> sa.MetaData:
    """The Phase 1 schema as explicit definitions.

    Mirrors ``app.models`` exactly; ``tests/test_phase1_migration.py`` asserts
    that the schema this migration produces matches the models.
    """
    metadata = sa.MetaData()

    sa.Table(
        "adjustments",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("adjustment_type", sa.String(length=32), nullable=False),
        sa.Column("origin", sa.String(length=24), nullable=False, server_default='PROVIDER'),
        sa.Column("resolution_action_id", sa.String()),
        sa.Column("reversed_by_adjustment_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("adjustment_type IN ('CREDIT', 'DEBIT', 'FEE_REVERSAL', 'PENALTY', 'BONUS', 'CORRECTION')", name="ck_adjustments_adjustment_type"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint("origin IN ('PROVIDER', 'CLOSELOOP_EXECUTION')", name="ck_adjustments_origin"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["resolution_action_id"], ["resolution_actions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["reversed_by_adjustment_id"], ["adjustments.id"], ondelete="RESTRICT"),
        sa.Index("ix_adjustments_case_id", "case_id"),
        sa.Index("ix_adjustments_merchant_id", "merchant_id"),
        sa.Index("ix_adjustments_origin", "origin"),
        sa.Index("ix_adjustments_payment_case", "payment_id", "case_id"),
        sa.Index("ix_adjustments_payment_id", "payment_id"),
        sa.Index("ix_adjustments_resolution_action", "resolution_action_id"),
    )

    sa.Table(
        "approvals",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("resolution_id", sa.String(), nullable=False),
        sa.Column("requested_by", sa.String(length=16), nullable=False),
        sa.Column("required_role", sa.String(length=32)),
        sa.Column("decision", sa.String(length=16), nullable=False, server_default='PENDING'),
        sa.Column("decided_by", sa.String(length=128)),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("comments", sa.Text()),
        sa.Column("evidence_digest", sa.String(length=128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("decision IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED', 'REVOKED')", name="ck_approvals_decision"),
        sa.CheckConstraint("requested_by IN ('SYSTEM', 'HUMAN')", name="ck_approvals_requested_by"),
        sa.ForeignKeyConstraint(["resolution_id"], ["resolutions.id"], ondelete="RESTRICT"),
        sa.Index("ix_approvals_decision_expires", "decision", "expires_at"),
        sa.Index("ix_approvals_resolution", "resolution_id"),
        sa.Index("uq_approvals_one_pending_per_resolution", "resolution_id", unique=True, postgresql_where=sa.text("decision = 'PENDING'"), sqlite_where=sa.text("decision = 'PENDING'")),
    )

    sa.Table(
        "audit_events",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String()),
        sa.Column("workflow_id", sa.String()),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("actor_type", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("correlation_id", sa.String()),
        sa.Column("evidence_ids", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("policy_decision", sa.String(length=16)),
        sa.Column("policy_version", sa.String(length=32)),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("before_state", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("after_state", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("model_version", sa.String(length=64)),
        sa.Column("llm_provider", sa.String(length=64)),
        sa.Column("correction_of", sa.String()),
        sa.Column("detail", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("policy_decision IN ('AUTO', 'HUMAN_REVIEW', 'UNRESOLVED', 'BLOCKED')", name="ck_audit_events_policy_decision"),
        sa.CheckConstraint("actor_type IN ('SYSTEM', 'HUMAN', 'AGENT')", name="ck_audit_events_actor_type"),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["correction_of"], ["audit_events.id"], ondelete="RESTRICT"),
        sa.Index("ix_audit_events_actor_action", "actor_type", "action"),
        sa.Index("ix_audit_events_correlation_id", "correlation_id"),
        sa.Index("ix_audit_events_exception_created", "exception_id", "created_at"),
        sa.Index("ix_audit_events_occurred_at", "occurred_at"),
    )

    sa.Table(
        "case_embeddings",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("exception_type", sa.String(), nullable=False),
        sa.Column("resolution_type", sa.String(), nullable=False),
        sa.Column("resolution_outcome", sa.String(), nullable=False),
        sa.Column("payment_amount", sa.BigInteger(), nullable=False),
        sa.Column("difference", sa.BigInteger(), nullable=False),
        sa.Column("supporting_evidence_count", sa.Integer()),
        sa.Column("tags_json", sa.Text()),
        sa.Column("case_text", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("embedding_template_version", sa.String()),
        sa.Column("created_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["id"], ["historical_cases.id"], ondelete="CASCADE"),
        sa.Index("ix_case_embeddings_exception_type", "exception_type"),
        sa.Index("ix_case_embeddings_model_template", "embedding_model", "embedding_template_version"),
    )

    sa.Table(
        "chargebacks",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("status", sa.String(length=32), nullable=False, server_default='OPEN'),
        sa.Column("reason_code", sa.String(length=64)),
        sa.Column("evidence_due_at", sa.DateTime(timezone=True)),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('amount >= 0', name="ck_amount_non_negative"),
        sa.CheckConstraint("status IN ('OPEN', 'PRE_ARBITRATION', 'WON', 'LOST')", name="ck_chargebacks_status"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_chargebacks_payment_status", "payment_id", "status"),
    )

    sa.Table(
        "evidence",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String()),
        sa.Column("entity_type", sa.String(length=48), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("relationship", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_by", sa.String(length=16), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("relationship IN ('PRIMARY', 'CALCULATION_COMPONENT', 'SUPPORTING', 'CONFLICTING', 'MISSING')", name="ck_evidence_relationship"),
        sa.CheckConstraint("recorded_by IN ('RECONCILIATION', 'AGENT', 'HUMAN')", name="ck_evidence_recorded_by"),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("exception_id", "entity_type", "entity_id", "relationship", name="uq_evidence_exception_entity_relationship"),
        sa.Index("ix_evidence_case_entity", "case_id", "entity_type"),
        sa.Index("ix_evidence_exception_entity", "exception_id", "entity_type"),
        sa.Index("ix_evidence_exception_relationship", "exception_id", "relationship"),
    )

    sa.Table(
        "evidence_links",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("relationship", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.Index("ix_evidence_links_case_entity", "case_id", "entity_type"),
        sa.Index("ix_evidence_links_case_id", "case_id"),
        sa.Index("ix_evidence_links_entity_id", "entity_id"),
        sa.Index("ix_evidence_links_exception_entity", "exception_id", "entity_type"),
        sa.Index("ix_evidence_links_exception_id", "exception_id"),
    )

    sa.Table(
        "exceptions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("merchant_id", sa.String()),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("expected_amount", sa.BigInteger(), nullable=False),
        sa.Column("actual_amount", sa.BigInteger(), nullable=False),
        sa.Column("difference", sa.BigInteger(), nullable=False),
        sa.Column("exception_type", sa.String(length=48), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("status_reason", sa.Text()),
        sa.Column("risk_category", sa.String(length=16)),
        sa.Column("assigned_to", sa.String(length=128)),
        sa.Column("sla_due_at", sa.DateTime(timezone=True)),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("close_reason", sa.String(length=64)),
        sa.Column("reopen_count", sa.Integer(), nullable=False, server_default='0'),
        sa.Column("reconciliation_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('OPEN', 'MATCHED', 'RESOLVED', 'DETECTED', 'INVESTIGATING', 'ANALYZED', 'RESOLUTION_PROPOSED', 'AUTO_APPROVED', 'HUMAN_REVIEW', 'APPROVED', 'REJECTED', 'EXECUTING', 'VERIFYING', 'RECONCILING', 'CLOSED', 'ESCALATED', 'FAILED', 'ROLLED_BACK', 'UNRESOLVED')", name="ck_exceptions_status"),
        sa.CheckConstraint('version >= 0', name="ck_exceptions_version"),
        sa.CheckConstraint('reopen_count >= 0', name="ck_exceptions_reopen_count"),
        sa.CheckConstraint("risk_category IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_exceptions_risk"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.Index("ix_exceptions_batch_id", "batch_id"),
        sa.Index("ix_exceptions_case_batch", "case_id", "batch_id", unique=True),
        sa.Index("ix_exceptions_case_id", "case_id"),
        sa.Index("ix_exceptions_merchant_status", "merchant_id", "status"),
        sa.Index("ix_exceptions_payment_id", "payment_id"),
        sa.Index("ix_exceptions_reconciliation_id", "reconciliation_id"),
        sa.Index("ix_exceptions_status_created", "status", "created_at"),
        sa.Index("ix_exceptions_type_status", "exception_type", "status"),
    )

    sa.Table(
        "feedback",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String()),
        sa.Column("resolution_id", sa.String()),
        sa.Column("feedback_type", sa.String(length=16), nullable=False),
        sa.Column("reviewer", sa.String(length=128)),
        sa.Column("reviewer_role", sa.String(length=32)),
        sa.Column("actor_type", sa.String(length=16), nullable=False),
        sa.Column("system_prediction", sa.String(length=64)),
        sa.Column("system_confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("correction_details", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("evidence_reviewed", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("model_version", sa.String(length=64)),
        sa.Column("policy_version", sa.String(length=32)),
        sa.Column("notes", sa.Text()),
        sa.Column("correction_of", sa.String()),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("actor_type IN ('SYSTEM', 'HUMAN', 'AGENT')", name="ck_feedback_actor_type"),
        sa.CheckConstraint("feedback_type IN ('APPROVAL', 'REJECTION', 'CORRECTION', 'OUTCOME', 'REVIEW')", name="ck_feedback_feedback_type"),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["correction_of"], ["feedback.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["resolution_id"], ["resolutions.id"], ondelete="RESTRICT"),
        sa.Index("ix_feedback_exception", "exception_id"),
        sa.Index("ix_feedback_resolution", "resolution_id"),
        sa.Index("ix_feedback_type_occurred", "feedback_type", "occurred_at"),
    )

    sa.Table(
        "fees",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("fee_type", sa.String(length=32), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint('amount >= 0', name="ck_fees_amount_non_negative"),
        sa.CheckConstraint("fee_type IN ('TRANSACTION', 'PLATFORM', 'TDR', 'GST_ON_FEES', 'REFUND_FEE', 'CHARGEBACK_FEE')", name="ck_fees_fee_type"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_fees_case_id", "case_id"),
        sa.Index("ix_fees_merchant_id", "merchant_id"),
        sa.Index("ix_fees_payment_case", "payment_id", "case_id"),
        sa.Index("ix_fees_payment_id", "payment_id"),
    )

    sa.Table(
        "historical_cases",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("merchant_id", sa.String()),
        sa.Column("exception_type", sa.String(), nullable=False),
        sa.Column("payment_amount", sa.BigInteger(), nullable=False),
        sa.Column("expected_amount", sa.BigInteger(), nullable=False),
        sa.Column("actual_amount", sa.BigInteger(), nullable=False),
        sa.Column("difference", sa.BigInteger(), nullable=False),
        sa.Column("total_refunds", sa.BigInteger()),
        sa.Column("total_fees", sa.BigInteger()),
        sa.Column("total_taxes", sa.BigInteger()),
        sa.Column("total_adjustments", sa.BigInteger()),
        sa.Column("financial_exposure_paise", sa.BigInteger()),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("resolution_type", sa.String(), nullable=False),
        sa.Column("resolution_outcome", sa.String(), nullable=False),
        sa.Column("resolution_origin", sa.String(), nullable=False),
        sa.Column("resolved_amount", sa.BigInteger()),
        sa.Column("resolution_notes", sa.Text()),
        sa.Column("resolution_action_id", sa.String()),
        sa.Column("reconciliation_verified", sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column("promoted_at", sa.DateTime(timezone=True)),
        sa.Column("evidence_refs_json", sa.Text()),
        sa.Column("supporting_evidence_count", sa.Integer()),
        sa.Column("exception_type_confidence", sa.Numeric(precision=None, scale=None)),
        sa.Column("evidence_coverage", sa.Numeric(precision=None, scale=None)),
        sa.Column("tags_json", sa.Text()),
        sa.Column("resolution_metadata_json", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("resolved_at", sa.DateTime()),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint('expected_amount >= 0', name="ck_historical_cases_expected_amount"),
        sa.CheckConstraint('actual_amount >= 0', name="ck_historical_cases_actual_amount"),
        sa.CheckConstraint('payment_amount >= 0', name="ck_historical_cases_payment_amount"),
        sa.ForeignKeyConstraint(["resolution_action_id"], ["resolution_actions.id"], ondelete="RESTRICT"),
        sa.Index("ix_historical_cases_created_at", "created_at"),
        sa.Index("ix_historical_cases_exception_type", "exception_type"),
        sa.Index("ix_historical_cases_resolution_outcome", "resolution_outcome"),
        sa.Index("ix_historical_cases_resolution_type", "resolution_type"),
        sa.Index("ix_historical_cases_verified", "reconciliation_verified"),
    )

    sa.Table(
        "historical_resolutions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String()),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("resolution_type", sa.String(), nullable=False),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("resolved_amount", sa.BigInteger()),
        sa.Column("difference_at_resolution", sa.BigInteger()),
        sa.Column("exception_type", sa.String()),
        sa.Column("resolvable", sa.Boolean()),
        sa.Column("notes", sa.String()),
        sa.Column("resolution_metadata", sa.String()),
        sa.Column("source", sa.String()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("resolved_at", sa.DateTime()),
        sa.Index("ix_historical_resolutions_case_id", "case_id"),
        sa.Index("ix_historical_resolutions_exception", "exception_id"),
        sa.Index("ix_historical_resolutions_exception_id", "exception_id"),
        sa.Index("ix_historical_resolutions_outcome", "outcome"),
        sa.Index("ix_historical_resolutions_resolution_type", "resolution_type"),
    )

    sa.Table(
        "ledger_entries",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("entry_type", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("source_entity_type", sa.String(length=64)),
        sa.Column("source_entity_id", sa.String()),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("correlation_id", sa.String()),
        sa.Column("metadata", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("direction IN ('DEBIT', 'CREDIT')", name="ck_ledger_entries_direction"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint("source IN ('INGEST', 'EXECUTION', 'ROLLBACK')", name="ck_ledger_entries_source"),
        sa.CheckConstraint("entry_type IN ('PAYMENT', 'SETTLEMENT', 'REFUND', 'CHARGEBACK', 'FEE', 'TAX', 'ADJUSTMENT', 'REVERSAL')", name="ck_ledger_entries_type"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_ledger_entries_correlation_id", "correlation_id"),
        sa.Index("ix_ledger_entries_payment_occurred", "payment_id", "occurred_at"),
        sa.Index("ix_ledger_entries_source_entity", "source_entity_type", "source_entity_id"),
    )

    sa.Table(
        "merchants",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default='ACTIVE'),
        sa.Column("category_code", sa.String(length=64)),
        sa.Column("onboarded_at", sa.DateTime(timezone=True)),
        sa.Column("deactivated_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED')", name="ck_merchants_status"),
        sa.Index("ix_merchants_status", "status"),
    )

    sa.Table(
        "model_predictions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=32), nullable=False),
        sa.Column("predicted_type", sa.String(length=48), nullable=False),
        sa.Column("probabilities", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("features_hash", sa.String(length=128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.Index("ix_model_predictions_exception", "exception_id"),
        sa.Index("ix_model_predictions_features_hash", "features_hash"),
        sa.Index("ix_model_predictions_model", "model_name", "model_version"),
    )

    sa.Table(
        "payment_attempts",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("status", sa.String(length=32), nullable=False, server_default='CREATED'),
        sa.Column("provider_attempt_id", sa.String()),
        sa.Column("occurred_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('amount >= 0', name="ck_amount_non_negative"),
        sa.CheckConstraint("status IN ('CREATED', 'AUTHORIZED', 'CAPTURED', 'FAILED', 'CANCELLED')", name="ck_payment_attempts_status"),
        sa.CheckConstraint("kind IN ('AUTH', 'CAPTURE')", name="ck_payment_attempts_kind"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_payment_attempts_payment", "payment_id"),
        sa.Index("ix_payment_attempts_payment_attempt_no", "payment_id", "attempt_no"),
    )

    sa.Table(
        "payments",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("merchant_id", sa.String()),
        sa.Column("order_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("status", sa.String(length=32), nullable=False, server_default='CREATED'),
        sa.Column("method", sa.String(length=32)),
        sa.Column("captured_at", sa.DateTime(timezone=True)),
        sa.Column("provider_created_at", sa.DateTime(timezone=True)),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint('amount >= 0', name="ck_payments_amount_non_negative"),
        sa.CheckConstraint("status IN ('PENDING', 'CREATED', 'AUTHORIZED', 'CAPTURED', 'SETTLED', 'PARTIALLY_REFUNDED', 'REFUNDED', 'CHARGEBACK', 'FAILED')", name="ck_payments_status"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint('version >= 0', name="ck_payments_version"),
        sa.ForeignKeyConstraint(["merchant_id"], ["merchants.id"], ondelete="RESTRICT"),
        sa.Index("ix_payments_merchant_provider_created", "merchant_id", "provider_created_at"),
        sa.Index("ix_payments_order_id", "order_id"),
        sa.Index("ix_payments_status", "status"),
    )

    sa.Table(
        "reconciliation_evidence",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("reconciliation_id", sa.String(), nullable=False),
        sa.Column("evidence_type", sa.String(), nullable=False),
        sa.Column("evidence_data", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.Index("ix_reconciliation_evidence_reconciliation_id", "reconciliation_id"),
    )

    sa.Table(
        "reconciliation_results",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("merchant_id", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("reconciliation_run_id", sa.String()),
        sa.Column("payment_amount", sa.BigInteger(), nullable=False),
        sa.Column("total_refunds", sa.BigInteger()),
        sa.Column("total_fees", sa.BigInteger()),
        sa.Column("total_taxes", sa.BigInteger()),
        sa.Column("total_adjustments", sa.BigInteger()),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("expected_amount", sa.BigInteger(), nullable=False),
        sa.Column("actual_amount", sa.BigInteger(), nullable=False),
        sa.Column("difference", sa.BigInteger(), nullable=False),
        sa.Column("match_status", sa.String(length=16), nullable=False),
        sa.Column("exception_type", sa.String(length=48), nullable=False),
        sa.Column("reconciliation_status", sa.String(length=24)),
        sa.Column("reconciliation_timestamp", sa.DateTime(timezone=True)),
        sa.Column("processing_notes", sa.String()),
        sa.CheckConstraint('expected_amount >= 0', name="ck_reconciliation_results_expected"),
        sa.CheckConstraint("exception_type IN ('EXACT_MATCH', 'FEE_DIFFERENCE', 'REFUND_ADJUSTMENT', 'TAX_ADJUSTMENT', 'TIMING_DIFFERENCE', 'PARTIAL_SETTLEMENT', 'DUPLICATE', 'MISSING_RECORD', 'COMPLEX_MULTI_ADJUSTMENT', 'UNKNOWN')", name="ck_reconciliation_results_exception_type"),
        sa.CheckConstraint('total_refunds >= 0', name="ck_reconciliation_results_total_refunds"),
        sa.CheckConstraint('actual_amount >= 0', name="ck_reconciliation_results_actual"),
        sa.CheckConstraint("reconciliation_status IN ('PENDING', 'PROCESSED', 'FAILED', 'REVIEW_REQUIRED')", name="ck_reconciliation_results_status"),
        sa.CheckConstraint('total_fees >= 0', name="ck_reconciliation_results_total_fees"),
        sa.CheckConstraint('payment_amount >= 0', name="ck_reconciliation_results_payment_amount"),
        sa.CheckConstraint('total_taxes >= 0', name="ck_reconciliation_results_total_taxes"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint("match_status IN ('MATCHED', 'EXCEPTION', 'MISSING', 'DUPLICATE')", name="ck_reconciliation_results_match_status"),
        sa.ForeignKeyConstraint(["reconciliation_run_id"], ["reconciliation_runs.id"], ondelete="RESTRICT"),
        sa.Index("ix_reconciliation_case_batch", "case_id", "batch_id", unique=True),
        sa.Index("ix_reconciliation_results_batch_id", "batch_id"),
        sa.Index("ix_reconciliation_results_case_id", "case_id"),
        sa.Index("ix_reconciliation_results_payment_id", "payment_id"),
        sa.Index("ix_reconciliation_results_reconciliation_run_id", "reconciliation_run_id"),
    )

    sa.Table(
        "reconciliation_runs",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("scope_type", sa.String(length=16), nullable=False),
        sa.Column("scope_ref", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default='PENDING'),
        sa.Column("trigger", sa.String(length=32), nullable=False),
        sa.Column("run_key", sa.String()),
        sa.Column("counts", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("matched_count", sa.Integer(), nullable=False, server_default='0'),
        sa.Column("exception_count", sa.Integer(), nullable=False, server_default='0'),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("engine_version", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=128)),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("scope_type IN ('BATCH', 'PAYMENT', 'EXCEPTION')", name="ck_reconciliation_runs_scope_type"),
        sa.CheckConstraint('exception_count >= 0', name="ck_reconciliation_runs_exception_count"),
        sa.CheckConstraint('version >= 0', name="ck_reconciliation_runs_version"),
        sa.CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')", name="ck_reconciliation_runs_status"),
        sa.CheckConstraint("trigger IN ('SCHEDULED', 'ON_INGEST', 'POST_RESOLUTION', 'MANUAL')", name="ck_reconciliation_runs_trigger"),
        sa.CheckConstraint('matched_count >= 0', name="ck_reconciliation_runs_matched_count"),
        sa.Index("ix_reconciliation_runs_scope", "scope_type", "scope_ref"),
        sa.Index("ix_reconciliation_runs_status_started", "status", "started_at"),
        sa.Index("uq_reconciliation_runs_one_running_per_scope", "scope_type", "scope_ref", unique=True, postgresql_where=sa.text("status = 'RUNNING'"), sqlite_where=sa.text("status = 'RUNNING'")),
        sa.Index("uq_reconciliation_runs_run_key", "scope_type", "scope_ref", "trigger", "run_key", unique=True),
    )

    sa.Table(
        "refunds",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=64)),
        sa.Column("refund_timestamp", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('PROCESSED', 'PENDING', 'FAILED')", name="ck_refunds_status"),
        sa.CheckConstraint('amount >= 0', name="ck_refunds_amount_non_negative"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_refunds_case_id", "case_id"),
        sa.Index("ix_refunds_merchant_id", "merchant_id"),
        sa.Index("ix_refunds_payment_case", "payment_id", "case_id"),
        sa.Index("ix_refunds_payment_id", "payment_id"),
    )

    sa.Table(
        "resolution_actions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("resolution_id", sa.String(), nullable=False),
        sa.Column("action_type", sa.String(length=16), nullable=False),
        sa.Column("provider_operation", sa.String(length=64)),
        sa.Column("amount_paise", sa.BigInteger()),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default='PROPOSED'),
        sa.Column("provider_reference", sa.String(length=128)),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default='0'),
        sa.Column("last_error", sa.Text()),
        sa.Column("executed_at", sa.DateTime(timezone=True)),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("rollback_action_id", sa.String()),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('attempts >= 0', name="ck_resolution_actions_attempts"),
        sa.CheckConstraint('amount_paise >= 0', name="ck_resolution_actions_amount_paise"),
        sa.CheckConstraint('version >= 0', name="ck_resolution_actions_version"),
        sa.CheckConstraint("status IN ('PROPOSED', 'VALIDATED', 'APPROVED', 'EXECUTING', 'EXECUTED', 'VERIFIED', 'FAILED', 'ROLLBACK_PENDING', 'ROLLED_BACK', 'ROLLBACK_FAILED')", name="ck_resolution_actions_status"),
        sa.CheckConstraint("action_type IN ('ADJUSTMENT', 'REVERSAL')", name="ck_resolution_actions_action_type"),
        sa.ForeignKeyConstraint(["rollback_action_id"], ["resolution_actions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["resolution_id"], ["resolutions.id"], ondelete="RESTRICT"),
        sa.Index("ix_resolution_actions_resolution_status", "resolution_id", "status"),
        sa.Index("uq_resolution_actions_idempotency_key", "idempotency_key", unique=True),
        sa.Index("uq_resolution_actions_provider_reference", "provider_reference", unique=True, postgresql_where=sa.text('provider_reference IS NOT NULL'), sqlite_where=sa.text('provider_reference IS NOT NULL')),
    )

    sa.Table(
        "resolutions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("candidate_rank", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("resolution_type", sa.String(length=64), nullable=False),
        sa.Column("amount_paise", sa.BigInteger()),
        sa.Column("direction", sa.String(length=8)),
        sa.Column("financial_exposure_paise", sa.BigInteger()),
        sa.Column("calculation_basis", sa.Text()),
        sa.Column("expected_effect", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("risk_category", sa.String(length=16)),
        sa.Column("evidence_ids", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("historical_support", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("ml_support", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("status", sa.String(length=24), nullable=False, server_default='DRAFT'),
        sa.Column("policy_decision", sa.String(length=16)),
        sa.Column("policy_version", sa.String(length=32)),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('candidate_rank >= 0', name="ck_resolutions_candidate_rank"),
        sa.CheckConstraint('amount_paise >= 0', name="ck_resolutions_amount_paise"),
        sa.CheckConstraint("policy_decision IN ('AUTO', 'HUMAN_REVIEW', 'UNRESOLVED', 'BLOCKED')", name="ck_resolutions_policy_decision"),
        sa.CheckConstraint('version >= 0', name="ck_resolutions_version"),
        sa.CheckConstraint("risk_category IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_resolutions_risk"),
        sa.CheckConstraint("direction IN ('CREDIT', 'DEBIT')", name="ck_resolutions_direction"),
        sa.CheckConstraint('financial_exposure_paise >= 0', name="ck_resolutions_financial_exposure"),
        sa.CheckConstraint("status IN ('DRAFT', 'PROPOSED', 'POLICY_EVALUATED', 'AUTO_APPROVED', 'HUMAN_REVIEW', 'APPROVED', 'REJECTED', 'BLOCKED', 'EXECUTING', 'EXECUTED', 'VERIFIED', 'FAILED', 'WITHDRAWN', 'ROLLBACK_PENDING', 'ROLLED_BACK', 'ROLLBACK_FAILED')", name="ck_resolutions_status"),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("exception_id", "candidate_rank", name="uq_resolutions_exception_rank"),
        sa.Index("ix_resolutions_exception_status", "exception_id", "status"),
        sa.Index("ix_resolutions_status", "status"),
    )

    sa.Table(
        "root_causes",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("exception_id", sa.String(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("cause_type", sa.String(length=48), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("deterministic_basis", sa.String(length=128)),
        sa.Column("ml_prediction_id", sa.String()),
        sa.Column("llm_explanation", sa.Text()),
        sa.Column("evidence_ids", sa.JSON().with_variant(postgresql.JSONB(), "postgresql")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('rank >= 0', name="ck_root_causes_rank"),
        sa.ForeignKeyConstraint(["exception_id"], ["exceptions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["ml_prediction_id"], ["model_predictions.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("exception_id", "rank", name="uq_root_causes_exception_rank"),
        sa.Index("ix_root_causes_exception", "exception_id"),
        sa.Index("ix_root_causes_exception_cause", "exception_id", "cause_type"),
    )

    sa.Table(
        "settlement_lines",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("settlement_id", sa.String(), nullable=False),
        sa.Column("component_type", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("source_entity_type", sa.String(length=64)),
        sa.Column("source_entity_id", sa.String()),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.CheckConstraint("component_type IN ('GROSS', 'FEE', 'TAX', 'REFUND', 'ADJUSTMENT', 'CHARGEBACK')", name="ck_settlement_lines_component_type"),
        sa.ForeignKeyConstraint(["settlement_id"], ["settlements.id"], ondelete="RESTRICT"),
        sa.Index("ix_settlement_lines_settlement", "settlement_id"),
        sa.Index("ix_settlement_lines_source_entity", "source_entity_type", "source_entity_id"),
    )

    sa.Table(
        "settlements",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("status", sa.String(length=16)),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.Column("provider_batch_id", sa.String()),
        sa.Column("version", sa.Integer(), nullable=False, server_default='1'),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint('version >= 0', name="ck_settlements_version"),
        sa.CheckConstraint('amount >= 0', name="ck_settlements_amount_non_negative"),
        sa.CheckConstraint("status IN ('PENDING', 'PROCESSING', 'SETTLED', 'FAILED')", name="ck_settlements_status"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["merchant_id"], ["merchants.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_settlements_merchant_settled_at", "merchant_id", "settled_at"),
        sa.Index("ix_settlements_payment", "payment_id"),
        sa.Index("ix_settlements_settled_at", "settled_at"),
    )

    sa.Table(
        "taxes",
        metadata,
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("payment_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String()),
        sa.Column("merchant_id", sa.String()),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default='INR'),
        sa.Column("tax_type", sa.String(length=32), nullable=False),
        sa.Column("jurisdiction", sa.String(length=64)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint('amount >= 0', name="ck_taxes_amount_non_negative"),
        sa.CheckConstraint("tax_type IN ('GST', 'TDS', 'GST_ON_FEES', 'SERVICE_TAX')", name="ck_taxes_tax_type"),
        sa.CheckConstraint("currency IN ('INR', 'USD')", name="ck_currency_currency"),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="RESTRICT"),
        sa.Index("ix_taxes_case_id", "case_id"),
        sa.Index("ix_taxes_merchant_id", "merchant_id"),
        sa.Index("ix_taxes_payment_case", "payment_id", "case_id"),
        sa.Index("ix_taxes_payment_id", "payment_id"),
    )

    return metadata


def _add_missing_columns(bind) -> None:
    """ALTER TABLE ... ADD COLUMN for any column an existing table lacks.

    This is the path taken when the database was previously built by the old
    ``init_db.py`` `create_all` rather than by a migration. It is additive only:
    columns are added, never dropped or rewritten, so no existing row is lost.

    SQLite limitation: SQLite refuses ``ADD COLUMN`` for a column whose default
    is an expression (``CURRENT_TIMESTAMP`` included). PostgreSQL - the
    production database - accepts it. Rather than silently dropping the NOT NULL
    constraint to make the statement pass, this raises and says what to do.
    """
    inspector = sa.inspect(bind)
    dialect = bind.dialect.name

    for table in _frozen_schema().sorted_tables:
        if not inspector.has_table(table.name):
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue

            default = None
            if column.server_default is not None:
                arg = column.server_default.arg
                # A string literal is constant and always safe. A SQL function
                # such as now() is not portable to SQLite's ALTER TABLE.
                if isinstance(arg, str) or dialect == "postgresql":
                    default = arg
                elif not column.nullable:
                    raise RuntimeError(
                        "Cannot add NOT NULL column "
                        f"{table.name}.{column.name} with a non-constant server "
                        f"default on '{dialect}'. Recreate the database with "
                        "`alembic upgrade head` (the databases this path applies "
                        "to hold only local sample data), or run the migration "
                        "against PostgreSQL, which supports this ALTER."
                    )

            op.add_column(
                table.name,
                sa.Column(
                    column.name,
                    column.type,
                    nullable=column.nullable,
                    server_default=default,
                ),
            )


def upgrade() -> None:
    """Create the Phase 1 schema, extending pre-existing tables in place."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    metadata = _frozen_schema()

    for table in metadata.sorted_tables:
        if not inspector.has_table(table.name):
            table.create(bind=bind, checkfirst=True)

    _add_missing_columns(bind)

    # create() above emits a new table's indexes; this pass covers indexes on
    # tables that pre-existed and were only extended. checkfirst keeps it
    # idempotent, and it preserves each index's dialect options (partial WHERE
    # clauses in particular).
    for table in metadata.sorted_tables:
        for index in table.indexes:
            index.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    """Drop the Phase 1 schema (baseline revision - returns to empty)."""
    bind = op.get_bind()
    for table in reversed(_frozen_schema().sorted_tables):
        table.drop(bind=bind, checkfirst=True)
