"""
Deterministic evidence retrieval service for Razorpay CloseLoop.

Given an exception_id or case_id, retrieves all relevant financial records
and constructs a structured EvidencePackage.

All logic is deterministic. No ground truth is read. No ML is used.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.exception import FinancialException
from app.models.refund import Refund
from app.models.fee import Fee
from app.models.tax import Tax
from app.models.adjustment import Adjustment
from app.models.evidence_link import EvidenceLink
from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
    StructuralConflict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Link Relationship Constants
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIP_PRIMARY = "PRIMARY_RECORD"
RELATIONSHIP_CALCULATION = "CALCULATION_COMPONENT"
RELATIONSHIP_SUPPORTING = "SUPPORTING_EVIDENCE"
RELATIONSHIP_CONFLICTING = "CONFLICTING_EVIDENCE"


class EvidenceRetrievalService:
    """
    Deterministic evidence retrieval service.

    Given an exception, retrieves all relevant financial records and
    constructs a structured EvidencePackage.

    Features:
    - Explicit missing evidence representation
    - Structural conflict detection
    - Idempotent EvidenceLink persistence
    - No ground truth reading
    """

    def __init__(self, session: Session):
        self.session = session

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_by_exception_id(
        self, exception_id: str, persist_links: bool = True
    ) -> Optional[EvidencePackage]:
        """
        Retrieve evidence for a specific exception.

        Args:
            exception_id: The exception to investigate
            persist_links: Whether to persist EvidenceLink records

        Returns:
            EvidencePackage or None if exception not found
        """
        exception = (
            self.session.query(FinancialException)
            .filter_by(id=exception_id)
            .first()
        )

        if exception is None:
            return None

        return self._retrieve_for_exception(exception, persist_links)

    def retrieve_by_case_id(
        self, case_id: str, persist_links: bool = True
    ) -> Optional[EvidencePackage]:
        """
        Retrieve evidence for the exception associated with a case.

        Args:
            case_id: The case to investigate
            persist_links: Whether to persist EvidenceLink records

        Returns:
            EvidencePackage or None if no exception found for case
        """
        exception = (
            self.session.query(FinancialException)
            .filter_by(case_id=case_id)
            .first()
        )

        if exception is None:
            return None

        return self._retrieve_for_exception(exception, persist_links)

    def retrieve_batch(
        self, exception_ids: List[str], persist_links: bool = True
    ) -> List[EvidencePackage]:
        """
        Retrieve evidence for multiple exceptions.

        Args:
            exception_ids: List of exception IDs to investigate
            persist_links: Whether to persist EvidenceLink records

        Returns:
            List of EvidencePackages (skips missing exceptions)
        """
        packages = []
        for eid in exception_ids:
            pkg = self.retrieve_by_exception_id(eid, persist_links=persist_links)
            if pkg is not None:
                packages.append(pkg)
        return packages

    # ─────────────────────────────────────────────────────────────────────────
    # Core Retrieval Logic
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_for_exception(
        self, exception: FinancialException, persist_links: bool
    ) -> EvidencePackage:
        """
        Build an EvidencePackage for a given exception.

        This is the core retrieval logic. It:
        1. Retrieves all related financial records
        2. Detects missing evidence
        3. Detects structural conflicts
        4. Calculates financial summary
        5. Optionally persists evidence links
        """
        payment_id = exception.payment_id
        case_id = exception.case_id

        # 1. Retrieve all financial records
        payment_record = self._retrieve_payment(exception)
        settlements = self._retrieve_settlements(payment_id, case_id)
        refunds = self._retrieve_refunds(payment_id, case_id)
        fees = self._retrieve_fees(payment_id, case_id)
        taxes = self._retrieve_taxes(payment_id, case_id)
        adjustments = self._retrieve_adjustments(payment_id, case_id)

        # 2. Detect missing evidence
        missing = self._detect_missing(
            payment_id=payment_id,
            settlements=settlements,
            exception_type=exception.exception_type,
        )

        # 3. Detect structural conflicts
        conflicts = self._detect_conflicts(settlements=settlements)

        # 4. Calculate financial summary
        total_settlement = sum(s.amount for s in settlements)
        total_refunds = sum(r.amount for r in refunds)
        total_fees = sum(f.amount for f in fees)
        total_taxes = sum(t.amount for t in taxes)
        total_adjustments = sum(a.amount for a in adjustments)

        # 5. Build evidence records
        settlement_records = [
            self._to_evidence_record(s, "SETTLEMENT", RELATIONSHIP_CALCULATION)
            for s in settlements
        ]
        refund_records = [
            self._to_evidence_record(r, "REFUND", RELATIONSHIP_CALCULATION)
            for r in refunds
        ]
        fee_records = [
            self._to_evidence_record(f, "FEE", RELATIONSHIP_CALCULATION)
            for f in fees
        ]
        tax_records = [
            self._to_evidence_record(t, "TAX", RELATIONSHIP_CALCULATION)
            for t in taxes
        ]
        adjustment_records = [
            self._to_evidence_record(a, "ADJUSTMENT", RELATIONSHIP_CALCULATION)
            for a in adjustments
        ]

        # Mark conflicting settlements
        if conflicts:
            for sr in settlement_records:
                sr.relationship = RELATIONSHIP_CONFLICTING

        # 6. Persist evidence links
        link_count = 0
        if persist_links:
            link_count = self._persist_evidence_links(
                exception_id=exception.id,
                case_id=case_id,
                payment_record=payment_record,
                settlements=settlements,
                refunds=refunds,
                fees=fees,
                taxes=taxes,
                adjustments=adjustments,
            )

        # 7. Build package
        return EvidencePackage(
            exception_id=exception.id,
            case_id=case_id,
            payment_id=payment_id,
            merchant_id=None,  # Payment model doesn't store merchant_id in DB yet
            expected_amount=exception.expected_amount,
            actual_amount=exception.actual_amount,
            difference=exception.difference,
            exception_type=exception.exception_type,
            payment=payment_record,
            settlements=settlement_records,
            refunds=refund_records,
            fees=fee_records,
            taxes=tax_records,
            adjustments=adjustment_records,
            total_settlement_amount=total_settlement,
            total_refund_amount=total_refunds,
            total_fee_amount=total_fees,
            total_tax_amount=total_taxes,
            total_adjustment_amount=total_adjustments,
            missing_evidence=missing,
            conflicts=conflicts,
            evidence_link_count=link_count,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Individual Record Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_payment(self, exception: FinancialException) -> EvidenceRecord:
        """Retrieve the payment record as evidence."""
        return EvidenceRecord(
            record_id=exception.payment_id,
            entity_type="PAYMENT",
            relationship=RELATIONSHIP_PRIMARY,
            amount=exception.expected_amount,  # Approximate from exception
            status=None,
            timestamp=None,
            metadata={
                "note": "Payment record referenced by exception",
            },
        )

    def _retrieve_settlements(
        self, payment_id: str, case_id: str
    ) -> list:
        """Retrieve all settlement records for a payment."""
        # Settlements are not stored in a dedicated model in the current schema.
        # The Payment and Settlement SQLAlchemy models only have id, payment_id/merchant_id, amount.
        # We query the Settlement model by payment_id.
        from app.models.settlement import Settlement as SettlementModel

        settlements = (
            self.session.query(SettlementModel)
            .filter_by(payment_id=payment_id)
            .all()
        )
        return settlements

    def _retrieve_refunds(self, payment_id: str, case_id: str) -> list:
        """Retrieve all refund records for a payment."""
        return (
            self.session.query(Refund)
            .filter_by(payment_id=payment_id)
            .all()
        )

    def _retrieve_fees(self, payment_id: str, case_id: str) -> list:
        """Retrieve all fee records for a payment."""
        return (
            self.session.query(Fee)
            .filter_by(payment_id=payment_id)
            .all()
        )

    def _retrieve_taxes(self, payment_id: str, case_id: str) -> list:
        """Retrieve all tax records for a payment."""
        return (
            self.session.query(Tax)
            .filter_by(payment_id=payment_id)
            .all()
        )

    def _retrieve_adjustments(self, payment_id: str, case_id: str) -> list:
        """Retrieve all adjustment records for a payment."""
        return (
            self.session.query(Adjustment)
            .filter_by(payment_id=payment_id)
            .all()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Missing Evidence Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_missing(
        self,
        payment_id: str,
        settlements: list,
        exception_type: str,
    ) -> List[MissingEvidence]:
        """
        Detect missing financial records.

        An empty list means no missing records detected.
        Explicitly distinguishes "record does not exist" from "was not queried".
        """
        missing = []

        # Check for missing settlement (always expected for a payment)
        if len(settlements) == 0:
            missing.append(
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement record found for payment "
                    + payment_id,
                )
            )

        # For MISSING_RECORD exception type, the exception itself tells us
        # something is missing — but we detect it from the records, not the label.
        # The settlement check above already covers the most common case.

        return missing

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Conflict Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_conflicts(
        self, settlements: list
    ) -> List[StructuralConflict]:
        """
        Detect structural conflicts in evidence.

        Does NOT resolve conflicts — only represents them.
        """
        conflicts = []

        # Multiple settlements
        if len(settlements) > 1:
            # Check for duplicate settlement IDs
            ids = [s.id for s in settlements]
            unique_ids = set(ids)
            if len(unique_ids) < len(ids):
                duplicate_ids = [sid for sid in ids if ids.count(sid) > 1]
                conflicts.append(
                    StructuralConflict(
                        conflict_type="DUPLICATE_SETTLEMENT_ID",
                        description=f"Duplicate settlement IDs found: {set(list(duplicate_ids))}",
                        affected_records=list(unique_ids),
                    )
                )

            # Multiple different settlements (not just duplicates)
            amounts = [s.amount for s in settlements]
            if len(set(amounts)) > 1:
                conflicts.append(
                    StructuralConflict(
                        conflict_type="MULTIPLE_SETTLEMENTS",
                        description=f"Multiple settlements with different amounts: {amounts}",
                        affected_records=ids,
                    )
                )
            elif len(set(amounts)) == 1 and len(settlements) > 1:
                # Same amount repeated — likely duplicate
                conflicts.append(
                    StructuralConflict(
                        conflict_type="DUPLICATE_SETTLEMENT_ID",
                        description=f"Multiple settlements with identical amount {amounts[0]}: possible duplicate",
                        affected_records=ids,
                    )
                )

        return conflicts

    # ─────────────────────────────────────────────────────────────────────────
    # Evidence Record Conversion
    # ─────────────────────────────────────────────────────────────────────────

    def _to_evidence_record(
        self,
        record,
        entity_type: str,
        relationship: str,
    ) -> EvidenceRecord:
        """Convert a database model to an EvidenceRecord."""
        # Extract common fields
        record_id = record.id
        amount = record.amount
        status = getattr(record, "status", None)
        timestamp = None

        # Try to extract timestamp from various field names
        for ts_field in [
            "refund_timestamp",
            "settlement_timestamp",
            "payment_timestamp",
            "created_at",
        ]:
            ts = getattr(record, ts_field, None)
            if ts is not None:
                timestamp = ts
                break

        # Build metadata
        metadata = {}
        if entity_type == "REFUND":
            metadata["fee_type"] = getattr(record, "fee_type", None)
        elif entity_type == "TAX":
            metadata["tax_type"] = getattr(record, "tax_type", None)
        elif entity_type == "ADJUSTMENT":
            metadata["adjustment_type"] = getattr(record, "adjustment_type", None)
        elif entity_type == "FEE":
            metadata["fee_type"] = getattr(record, "fee_type", None)
        elif entity_type == "SETTLEMENT":
            metadata["merchant_id"] = getattr(record, "merchant_id", None)

        # Filter out None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return EvidenceRecord(
            record_id=record_id,
            entity_type=entity_type,
            relationship=relationship,
            amount=amount,
            status=status,
            timestamp=timestamp,
            metadata=metadata if metadata else None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EvidenceLink Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_evidence_links(
        self,
        exception_id: str,
        case_id: str,
        payment_record: EvidenceRecord,
        settlements: list,
        refunds: list,
        fees: list,
        taxes: list,
        adjustments: list,
    ) -> int:
        """
        Persist evidence links for all retrieved records.

        Idempotent: does not create duplicate links for the same
        exception_id + entity_id combination.

        Returns the number of links created/verified.
        """
        link_count = 0

        # Get existing links for this exception to avoid duplicates
        existing_links = (
            self.session.query(EvidenceLink)
            .filter_by(exception_id=exception_id)
            .all()
        )
        existing_entity_ids = {
            (link.entity_type, link.entity_id) for link in existing_links
        }

        # Payment link
        link_count += self._persist_single_link(
            exception_id=exception_id,
            case_id=case_id,
            entity_type="PAYMENT",
            entity_id=payment_record.record_id,
            relationship=RELATIONSHIP_PRIMARY,
            existing=existing_entity_ids,
        )

        # Settlement links
        for s in settlements:
            link_count += self._persist_single_link(
                exception_id=exception_id,
                case_id=case_id,
                entity_type="SETTLEMENT",
                entity_id=s.id,
                relationship=RELATIONSHIP_CALCULATION,
                existing=existing_entity_ids,
            )

        # Refund links
        for r in refunds:
            link_count += self._persist_single_link(
                exception_id=exception_id,
                case_id=case_id,
                entity_type="REFUND",
                entity_id=r.id,
                relationship=RELATIONSHIP_CALCULATION,
                existing=existing_entity_ids,
            )

        # Fee links
        for f in fees:
            link_count += self._persist_single_link(
                exception_id=exception_id,
                case_id=case_id,
                entity_type="FEE",
                entity_id=f.id,
                relationship=RELATIONSHIP_CALCULATION,
                existing=existing_entity_ids,
            )

        # Tax links
        for t in taxes:
            link_count += self._persist_single_link(
                exception_id=exception_id,
                case_id=case_id,
                entity_type="TAX",
                entity_id=t.id,
                relationship=RELATIONSHIP_CALCULATION,
                existing=existing_entity_ids,
            )

        # Adjustment links
        for a in adjustments:
            link_count += self._persist_single_link(
                exception_id=exception_id,
                case_id=case_id,
                entity_type="ADJUSTMENT",
                entity_id=a.id,
                relationship=RELATIONSHIP_CALCULATION,
                existing=existing_entity_ids,
            )

        return link_count

    def _persist_single_link(
        self,
        exception_id: str,
        case_id: str,
        entity_type: str,
        entity_id: str,
        relationship: str,
        existing: set,
    ) -> int:
        """
        Persist a single evidence link if it doesn't already exist.

        Returns 1 if created, 0 if already exists.
        """
        if (entity_type, entity_id) in existing:
            return 0

        # Generate deterministic ID
        link_id = f"EL-{exception_id}-{entity_type}-{entity_id}"

        link = EvidenceLink(
            id=link_id,
            exception_id=exception_id,
            case_id=case_id,
            entity_type=entity_type,
            entity_id=entity_id,
            relationship=relationship,
        )

        self.session.add(link)
        existing.add((entity_type, entity_id))
        return 1
