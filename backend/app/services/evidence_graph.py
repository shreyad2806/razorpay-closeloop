"""
Financial Evidence Graph builder using NetworkX.

Constructs a deterministic directed graph representing financial relationships
behind an exception. The graph makes the financial explanation discoverable
through graph traversal.

All construction is deterministic — no ML, no LLM, no probabilistic reasoning.

Input: EvidencePackage
Output: networkx.DiGraph
"""

from typing import Optional

import networkx as nx

from app.schemas.evidence import EvidencePackage, EvidenceRecord


# ─────────────────────────────────────────────────────────────────────────────
# Node Type Constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_MERCHANT = "MERCHANT"
NODE_PAYMENT = "PAYMENT"
NODE_SETTLEMENT = "SETTLEMENT"
NODE_REFUND = "REFUND"
NODE_FEE = "FEE"
NODE_TAX = "TAX"
NODE_ADJUSTMENT = "ADJUSTMENT"
NODE_EXCEPTION = "EXCEPTION"
NODE_MISSING = "MISSING"

# ─────────────────────────────────────────────────────────────────────────────
# Edge Type Constants
# ─────────────────────────────────────────────────────────────────────────────

EDGE_OWNS = "OWNS"                  # Merchant → Payment
EDGE_GENERATED = "GENERATED"        # Payment → Settlement/Refund/Fee/Tax/Adjustment
EDGE_EXPLAINS = "EXPLAINS"          # Settlement/Refund/Fee/Tax/Adjustment → Exception
EDGE_RELATES_TO = "RELATES_TO"      # Exception → Payment
EDGE_IS_MISSING = "IS_MISSING"      # Exception → Missing node
EDGE_CONTRIBUTES_TO = "CONTRIBUTES_TO"  # Financial event → expected settlement

# ─────────────────────────────────────────────────────────────────────────────
# Financial Contribution Sign Convention
# ─────────────────────────────────────────────────────────────────────────────
#
# expected_amount = payment
#                  - refunds
#                  - fees
#                  - taxes
#                  + adjustments
#
# contribution_to_expected:
#   payment:    +amount
#   refund:     -amount  (reduces expected settlement)
#   fee:        -amount  (reduces expected settlement)
#   tax:        -amount  (reduces expected settlement)
#   adjustment:  amount  (signed: credit positive, debit negative)


class EvidenceGraphBuilder:
    """
    Builds a NetworkX DiGraph from an EvidencePackage.

    The graph represents:
    - Financial entities as nodes with attributes
    - Relationships as directed edges with metadata
    - Missing evidence as placeholder nodes
    - Financial contributions to expected settlement

    The resulting graph is deterministic and traceable.
    """

    def build(self, package: EvidencePackage) -> nx.DiGraph:
        """
        Build a directed graph from an evidence package.

        Args:
            package: The EvidencePackage from evidence retrieval

        Returns:
            networkx.DiGraph with financial nodes and relationships
        """
        G = nx.DiGraph()

        # 1. Add exception node
        self._add_exception_node(G, package)

        # 2. Add merchant node if present
        if package.merchant_id:
            self._add_merchant_node(G, package)

        # 3. Add payment node (always — every exception has a payment_id)
        self._add_payment_node(G, package)

        # 4. Add settlement nodes
        for settlement in package.settlements:
            self._add_settlement_node(G, package, settlement)

        # 5. Add refund nodes
        for refund in package.refunds:
            self._add_refund_node(G, package, refund)

        # 6. Add fee nodes
        for fee in package.fees:
            self._add_fee_node(G, package, fee)

        # 7. Add tax nodes
        for tax in package.taxes:
            self._add_tax_node(G, package, tax)

        # 8. Add adjustment nodes
        for adj in package.adjustments:
            self._add_adjustment_node(G, package, adj)

        # 9. Add missing evidence nodes
        for missing in package.missing_evidence:
            self._add_missing_node(G, package, missing)

        # 10. Add conflict annotations to edges
        self._annotate_conflicts(G, package)

        return G

    # ─────────────────────────────────────────────────────────────────────────
    # Node Builders
    # ─────────────────────────────────────────────────────────────────────────

    def _add_exception_node(self, G: nx.DiGraph, pkg: EvidencePackage):
        """Add the exception node as the central node."""
        G.add_node(
            pkg.exception_id,
            node_type=NODE_EXCEPTION,
            entity_id=pkg.exception_id,
            case_id=pkg.case_id,
            payment_id=pkg.payment_id,
            exception_type=pkg.exception_type,
            expected_amount=pkg.expected_amount,
            actual_amount=pkg.actual_amount,
            difference=pkg.difference,
            amount=pkg.difference,
        )

    def _add_merchant_node(self, G: nx.DiGraph, pkg: EvidencePackage):
        """Add merchant node and edge to payment."""
        G.add_node(
            pkg.merchant_id,
            node_type=NODE_MERCHANT,
            entity_id=pkg.merchant_id,
            case_id=pkg.case_id,
        )
        G.add_edge(
            pkg.merchant_id,
            pkg.payment_id,
            edge_type=EDGE_OWNS,
            relationship="merchant_owns_payment",
        )

    def _add_payment_node(self, G: nx.DiGraph, pkg: EvidencePackage):
        """Add payment node and edges to exception and financial events."""
        payment_id = pkg.payment_id
        amount = pkg.payment.amount if pkg.payment else pkg.expected_amount
        status = pkg.payment.status if pkg.payment else None
        timestamp = pkg.payment.timestamp if pkg.payment else None

        G.add_node(
            payment_id,
            node_type=NODE_PAYMENT,
            entity_id=payment_id,
            case_id=pkg.case_id,
            amount=amount,
            status=status,
            timestamp=timestamp,
            contribution_to_expected=amount,
        )

        # Exception → Payment (relates_to)
        G.add_edge(
            pkg.exception_id,
            payment_id,
            edge_type=EDGE_RELATES_TO,
            relationship="exception_relates_to_payment",
        )

        # Payment → Settlement edges added when settlements are processed
        # Payment → Refund/Fee/Tax/Adjustment edges added below

    def _add_settlement_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, record: EvidenceRecord
    ):
        """Add settlement node with edges."""
        G.add_node(
            record.record_id,
            node_type=NODE_SETTLEMENT,
            entity_id=record.record_id,
            case_id=pkg.case_id,
            amount=record.amount,
            status=record.status,
            timestamp=record.timestamp,
            contribution_to_expected=0,  # Settlement is actual, not expected
            relationship_to_exception=record.relationship,
        )

        # Payment → Settlement
        G.add_edge(
            pkg.payment_id,
            record.record_id,
            edge_type=EDGE_GENERATED,
            relationship="payment_generates_settlement",
            amount=record.amount,
        )

        # Settlement → Exception (explains)
        G.add_edge(
            record.record_id,
            pkg.exception_id,
            edge_type=EDGE_EXPLAINS,
            relationship="settlement_explains_exception",
            amount=record.amount,
        )

    def _add_refund_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, record: EvidenceRecord
    ):
        """Add refund node with edges."""
        contribution = -record.amount  # Refund reduces expected settlement
        G.add_node(
            record.record_id,
            node_type=NODE_REFUND,
            entity_id=record.record_id,
            case_id=pkg.case_id,
            amount=record.amount,
            status=record.status,
            timestamp=record.timestamp,
            contribution_to_expected=contribution,
            relationship_to_exception=record.relationship,
        )

        # Payment → Refund
        G.add_edge(
            pkg.payment_id,
            record.record_id,
            edge_type=EDGE_GENERATED,
            relationship="payment_generates_refund",
            amount=record.amount,
        )

        # Refund → Exception (contributes to explanation)
        G.add_edge(
            record.record_id,
            pkg.exception_id,
            edge_type=EDGE_EXPLAINS,
            relationship="refund_contributes_to_exception",
            amount=record.amount,
            contribution=contribution,
        )

    def _add_fee_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, record: EvidenceRecord
    ):
        """Add fee node with edges."""
        contribution = -record.amount  # Fee reduces expected settlement
        fee_type = record.metadata.get("fee_type") if record.metadata else None
        G.add_node(
            record.record_id,
            node_type=NODE_FEE,
            entity_id=record.record_id,
            case_id=pkg.case_id,
            amount=record.amount,
            fee_type=fee_type,
            timestamp=record.timestamp,
            contribution_to_expected=contribution,
            relationship_to_exception=record.relationship,
        )

        # Payment → Fee
        G.add_edge(
            pkg.payment_id,
            record.record_id,
            edge_type=EDGE_GENERATED,
            relationship="payment_generates_fee",
            amount=record.amount,
        )

        # Fee → Exception (contributes to explanation)
        G.add_edge(
            record.record_id,
            pkg.exception_id,
            edge_type=EDGE_EXPLAINS,
            relationship="fee_contributes_to_exception",
            amount=record.amount,
            contribution=contribution,
        )

    def _add_tax_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, record: EvidenceRecord
    ):
        """Add tax node with edges."""
        contribution = -record.amount  # Tax reduces expected settlement
        tax_type = record.metadata.get("tax_type") if record.metadata else None
        G.add_node(
            record.record_id,
            node_type=NODE_TAX,
            entity_id=record.record_id,
            case_id=pkg.case_id,
            amount=record.amount,
            tax_type=tax_type,
            timestamp=record.timestamp,
            contribution_to_expected=contribution,
            relationship_to_exception=record.relationship,
        )

        # Payment → Tax
        G.add_edge(
            pkg.payment_id,
            record.record_id,
            edge_type=EDGE_GENERATED,
            relationship="payment_generates_tax",
            amount=record.amount,
        )

        # Tax → Exception (contributes to explanation)
        G.add_edge(
            record.record_id,
            pkg.exception_id,
            edge_type=EDGE_EXPLAINS,
            relationship="tax_contributes_to_exception",
            amount=record.amount,
            contribution=contribution,
        )

    def _add_adjustment_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, record: EvidenceRecord
    ):
        """Add adjustment node with edges."""
        contribution = record.amount  # Adjustment is signed (credit positive, debit negative)
        adj_type = (
            record.metadata.get("adjustment_type") if record.metadata else None
        )
        G.add_node(
            record.record_id,
            node_type=NODE_ADJUSTMENT,
            entity_id=record.record_id,
            case_id=pkg.case_id,
            amount=record.amount,
            adjustment_type=adj_type,
            timestamp=record.timestamp,
            contribution_to_expected=contribution,
            relationship_to_exception=record.relationship,
        )

        # Payment → Adjustment
        G.add_edge(
            pkg.payment_id,
            record.record_id,
            edge_type=EDGE_GENERATED,
            relationship="payment_generates_adjustment",
            amount=record.amount,
        )

        # Adjustment → Exception (contributes to explanation)
        G.add_edge(
            record.record_id,
            pkg.exception_id,
            edge_type=EDGE_EXPLAINS,
            relationship="adjustment_contributes_to_exception",
            amount=record.amount,
            contribution=contribution,
        )

    def _add_missing_node(
        self, G: nx.DiGraph, pkg: EvidencePackage, missing
    ):
        """
        Add a missing evidence placeholder node.

        Does NOT create a fake financial record.
        Creates a MISSING node that indicates what is absent.
        """
        missing_node_id = f"MISSING-{missing.entity_id if hasattr(missing, 'entity_id') else missing.entity_type}-{pkg.payment_id}"
        G.add_node(
            missing_node_id,
            node_type=NODE_MISSING,
            entity_type=missing.entity_type,
            entity_id=missing_node_id,
            case_id=pkg.case_id,
            expected=missing.expected,
            reason=missing.reason,
            amount=0,
            contribution_to_expected=0,
        )

        # Exception → Missing (is_missing)
        G.add_edge(
            pkg.exception_id,
            missing_node_id,
            edge_type=EDGE_IS_MISSING,
            relationship=f"exception_has_missing_{missing.entity_type.lower()}",
            entity_type=missing.entity_type,
        )

        # Link payment → missing
        G.add_edge(
            pkg.payment_id,
            missing_node_id,
            edge_type=EDGE_IS_MISSING,
            relationship=f"payment_missing_{missing.entity_type.lower()}",
            entity_type=missing.entity_type,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Conflict Annotation
    # ─────────────────────────────────────────────────────────────────────────

    def _annotate_conflicts(self, G: nx.DiGraph, pkg: EvidencePackage):
        """Add conflict metadata to edges where applicable."""
        for conflict in pkg.conflicts:
            for record_id in conflict.affected_records:
                if G.has_node(record_id):
                    # Add conflict attribute to the node
                    G.nodes[record_id]["has_conflict"] = True
                    G.nodes[record_id]["conflict_type"] = conflict.conflict_type
                    G.nodes[record_id]["conflict_description"] = conflict.description

    # ─────────────────────────────────────────────────────────────────────────
    # Graph Analysis Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_nodes_by_type(G: nx.DiGraph, node_type: str) -> list:
        """Get all nodes of a specific type."""
        return [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]

    @staticmethod
    def get_total_contribution(G: nx.DiGraph) -> int:
        """
        Calculate total financial contribution from all nodes.

        This represents the expected settlement calculation:
        payment - refunds - fees - taxes + adjustments
        """
        total = 0
        for node, attrs in G.nodes(data=True):
            contrib = attrs.get("contribution_to_expected", 0)
            if contrib != 0:
                total += contrib
        return total

    @staticmethod
    def get_exception_explainers(G: nx.DiGraph, exception_id: str) -> list:
        """
        Get all nodes that have an EXPLAINS edge to the exception.

        These are the financial events that contribute to explaining
        the exception.
        """
        explainers = []
        for predecessor in G.predecessors(exception_id):
            edge_data = G.edges[predecessor, exception_id]
            if edge_data.get("edge_type") == EDGE_EXPLAINS:
                node_data = G.nodes[predecessor]
                explainers.append({
                    "node_id": predecessor,
                    "node_type": node_data.get("node_type"),
                    "amount": node_data.get("amount", 0),
                    "contribution": edge_data.get("contribution", 0),
                })
        return explainers

    @staticmethod
    def get_missing_evidence(G: nx.DiGraph) -> list:
        """Get all missing evidence nodes."""
        return [
            {
                "node_id": node,
                "entity_type": attrs.get("entity_type"),
                "reason": attrs.get("reason"),
                "expected": attrs.get("expected"),
            }
            for node, attrs in G.nodes(data=True)
            if attrs.get("node_type") == NODE_MISSING
        ]

    @staticmethod
    def get_conflicts(G: nx.DiGraph) -> list:
        """Get all nodes with conflicts."""
        return [
            {
                "node_id": node,
                "conflict_type": attrs.get("conflict_type"),
                "description": attrs.get("conflict_description"),
            }
            for node, attrs in G.nodes(data=True)
            if attrs.get("has_conflict")
        ]
