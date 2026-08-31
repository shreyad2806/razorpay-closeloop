"""
Tests for evidence graph builder.

Covers:
- Single-event case
- Refund case
- Fee case
- Tax case
- Adjustment case
- Complex multi-adjustment case
- Missing settlement
- Duplicate settlement
- Unknown case
- Node and edge counts
- Node attributes
- Financial contribution calculation
- Traceability
- Ground truth separation
"""

import os
import sys
from pathlib import Path

import networkx as nx
import pytest

# Set env before importing database module
os.environ.setdefault("DATABASE_URL", "sqlite:///test_evidence_graph.db")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.evidence import (
    EvidencePackage,
    EvidenceRecord,
    MissingEvidence,
    StructuralConflict,
)
from app.services.evidence_graph import (
    EvidenceGraphBuilder,
    NODE_EXCEPTION,
    NODE_PAYMENT,
    NODE_SETTLEMENT,
    NODE_REFUND,
    NODE_FEE,
    NODE_TAX,
    NODE_ADJUSTMENT,
    NODE_MISSING,
    NODE_MERCHANT,
    EDGE_GENERATED,
    EDGE_EXPLAINS,
    EDGE_RELATES_TO,
    EDGE_IS_MISSING,
    EDGE_OWNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def builder():
    """Provide an evidence graph builder."""
    return EvidenceGraphBuilder()


def _make_package(**kwargs):
    """Helper to create an EvidencePackage with defaults."""
    defaults = {
        "exception_id": "EXC-001",
        "case_id": "CASE-001",
        "payment_id": "PAY-001",
        "expected_amount": 100000,
        "actual_amount": 100000,
        "difference": 0,
        "exception_type": "EXACT_MATCH",
    }
    defaults.update(kwargs)
    return EvidencePackage(**defaults)


def _make_record(record_id, entity_type, amount, **kwargs):
    """Helper to create an EvidenceRecord."""
    defaults = {
        "record_id": record_id,
        "entity_type": entity_type,
        "relationship": "CALCULATION_COMPONENT",
        "amount": amount,
    }
    defaults.update(kwargs)
    return EvidenceRecord(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Single Event Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleEventCase:
    """Tests for the simplest case — just a payment and exception."""

    def test_exception_node_created(self, builder):
        """Test that exception node is created."""
        pkg = _make_package()
        G = builder.build(pkg)

        assert G.has_node("EXC-001")
        assert G.nodes["EXC-001"]["node_type"] == NODE_EXCEPTION

    def test_payment_node_created(self, builder):
        """Test that payment node is created."""
        pkg = _make_package()
        G = builder.build(pkg)

        assert G.has_node("PAY-001")
        assert G.nodes["PAY-001"]["node_type"] == NODE_PAYMENT

    def test_exception_relates_to_payment(self, builder):
        """Test edge from exception to payment."""
        pkg = _make_package()
        G = builder.build(pkg)

        assert G.has_edge("EXC-001", "PAY-001")
        assert G.edges["EXC-001", "PAY-001"]["edge_type"] == EDGE_RELATES_TO

    def test_node_count_basic(self, builder):
        """Test node count for basic case."""
        pkg = _make_package()
        G = builder.build(pkg)

        assert G.number_of_nodes() == 2  # exception + payment
        assert G.number_of_edges() == 1  # exception → payment

    def test_payment_attributes(self, builder):
        """Test that payment node has correct attributes."""
        pkg = _make_package(expected_amount=100000)
        G = builder.build(pkg)

        attrs = G.nodes["PAY-001"]
        assert attrs["amount"] == 100000
        assert attrs["contribution_to_expected"] == 100000


# ─────────────────────────────────────────────────────────────────────────────
# Refund Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRefundCase:
    """Tests for refund evidence in graph."""

    def test_refund_node_created(self, builder):
        """Test that refund node is created."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_node("REF-001")
        assert G.nodes["REF-001"]["node_type"] == NODE_REFUND

    def test_refund_negative_contribution(self, builder):
        """Test that refund has negative contribution to expected."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["REF-001"]["contribution_to_expected"] == -5000

    def test_payment_to_refund_edge(self, builder):
        """Test edge from payment to refund."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_edge("PAY-001", "REF-001")
        assert G.edges["PAY-001", "REF-001"]["edge_type"] == EDGE_GENERATED

    def test_refund_to_exception_edge(self, builder):
        """Test edge from refund to exception."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_edge("REF-001", "EXC-001")
        assert G.edges["REF-001", "EXC-001"]["edge_type"] == EDGE_EXPLAINS

    def test_multiple_refunds(self, builder):
        """Test multiple refund nodes."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 3000),
                _make_record("REF-002", "REFUND", 2000),
            ]
        )
        G = builder.build(pkg)

        refund_nodes = EvidenceGraphBuilder.get_nodes_by_type(G, NODE_REFUND)
        assert len(refund_nodes) == 2

    def test_refund_amount_attribute(self, builder):
        """Test refund amount attribute."""
        pkg = _make_package(
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["REF-001"]["amount"] == 5000


# ─────────────────────────────────────────────────────────────────────────────
# Fee Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeeCase:
    """Tests for fee evidence in graph."""

    def test_fee_node_created(self, builder):
        """Test that fee node is created."""
        pkg = _make_package(
            fees=[
                _make_record("FEE-001", "FEE", 2000, metadata={"fee_type": "TDR"}),
            ]
        )
        G = builder.build(pkg)

        assert G.has_node("FEE-001")
        assert G.nodes["FEE-001"]["node_type"] == NODE_FEE

    def test_fee_negative_contribution(self, builder):
        """Test that fee has negative contribution."""
        pkg = _make_package(
            fees=[
                _make_record("FEE-001", "FEE", 2000),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["FEE-001"]["contribution_to_expected"] == -2000

    def test_fee_type_attribute(self, builder):
        """Test that fee type is captured."""
        pkg = _make_package(
            fees=[
                _make_record("FEE-001", "FEE", 2000, metadata={"fee_type": "TDR"}),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["FEE-001"]["fee_type"] == "TDR"

    def test_payment_to_fee_edge(self, builder):
        """Test edge from payment to fee."""
        pkg = _make_package(
            fees=[
                _make_record("FEE-001", "FEE", 2000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_edge("PAY-001", "FEE-001")
        assert G.edges["PAY-001", "FEE-001"]["edge_type"] == EDGE_GENERATED


# ─────────────────────────────────────────────────────────────────────────────
# Tax Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaxCase:
    """Tests for tax evidence in graph."""

    def test_tax_node_created(self, builder):
        """Test that tax node is created."""
        pkg = _make_package(
            taxes=[
                _make_record("TAX-001", "TAX", 1800, metadata={"tax_type": "GST"}),
            ]
        )
        G = builder.build(pkg)

        assert G.has_node("TAX-001")
        assert G.nodes["TAX-001"]["node_type"] == NODE_TAX

    def test_tax_negative_contribution(self, builder):
        """Test that tax has negative contribution."""
        pkg = _make_package(
            taxes=[
                _make_record("TAX-001", "TAX", 1800),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["TAX-001"]["contribution_to_expected"] == -1800

    def test_tax_type_attribute(self, builder):
        """Test that tax type is captured."""
        pkg = _make_package(
            taxes=[
                _make_record("TAX-001", "TAX", 1800, metadata={"tax_type": "GST"}),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["TAX-001"]["tax_type"] == "GST"


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjustmentCase:
    """Tests for adjustment evidence in graph."""

    def test_adjustment_node_created(self, builder):
        """Test that adjustment node is created."""
        pkg = _make_package(
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 3000, metadata={"adjustment_type": "CREDIT"}),
            ]
        )
        G = builder.build(pkg)

        assert G.has_node("ADJ-001")
        assert G.nodes["ADJ-001"]["node_type"] == NODE_ADJUSTMENT

    def test_positive_adjustment_contribution(self, builder):
        """Test that credit adjustment has positive contribution."""
        pkg = _make_package(
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 3000, metadata={"adjustment_type": "CREDIT"}),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["ADJ-001"]["contribution_to_expected"] == 3000

    def test_negative_adjustment_contribution(self, builder):
        """Test that debit adjustment has negative contribution."""
        pkg = _make_package(
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", -5000, metadata={"adjustment_type": "DEBIT"}),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["ADJ-001"]["contribution_to_expected"] == -5000

    def test_adjustment_type_attribute(self, builder):
        """Test that adjustment type is captured."""
        pkg = _make_package(
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 3000, metadata={"adjustment_type": "CREDIT"}),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["ADJ-001"]["adjustment_type"] == "CREDIT"


# ─────────────────────────────────────────────────────────────────────────────
# Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSettlementCase:
    """Tests for settlement evidence in graph."""

    def test_settlement_node_created(self, builder):
        """Test that settlement node is created."""
        pkg = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 95000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_node("SET-001")
        assert G.nodes["SET-001"]["node_type"] == NODE_SETTLEMENT

    def test_settlement_zero_contribution(self, builder):
        """Test that settlement has zero contribution (it's actual, not expected)."""
        pkg = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 95000),
            ]
        )
        G = builder.build(pkg)

        assert G.nodes["SET-001"]["contribution_to_expected"] == 0

    def test_settlement_to_exception_edge(self, builder):
        """Test edge from settlement to exception."""
        pkg = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 95000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_edge("SET-001", "EXC-001")
        assert G.edges["SET-001", "EXC-001"]["edge_type"] == EDGE_EXPLAINS

    def test_payment_to_settlement_edge(self, builder):
        """Test edge from payment to settlement."""
        pkg = _make_package(
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 95000),
            ]
        )
        G = builder.build(pkg)

        assert G.has_edge("PAY-001", "SET-001")
        assert G.edges["PAY-001", "SET-001"]["edge_type"] == EDGE_GENERATED


# ─────────────────────────────────────────────────────────────────────────────
# Complex Multi-Adjustment Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexMultiAdjustment:
    """Tests for complex case with multiple financial events."""

    def test_complex_graph_structure(self, builder):
        """Test graph for case with refund, fee, tax, adjustment, settlement."""
        pkg = _make_package(
            expected_amount=80000,
            actual_amount=75000,
            difference=5000,
            exception_type="COMPLEX_MULTI_ADJUSTMENT",
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 75000),
            ],
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 2000),
                _make_record("FEE-002", "FEE", 1000),
            ],
            taxes=[
                _make_record("TAX-001", "TAX", 1800),
            ],
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 3000),
            ],
        )
        G = builder.build(pkg)

        # Nodes: exception, payment, 1 settlement, 1 refund, 2 fees, 1 tax, 1 adjustment = 8
        assert G.number_of_nodes() == 8

        # Verify all node types present
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_EXCEPTION)) == 1
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_PAYMENT)) == 1
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_SETTLEMENT)) == 1
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_REFUND)) == 1
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_FEE)) == 2
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_TAX)) == 1
        assert len(EvidenceGraphBuilder.get_nodes_by_type(G, NODE_ADJUSTMENT)) == 1

    def test_complex_contribution_calculation(self, builder):
        """Test that contribution calculation is correct."""
        pkg = _make_package(
            payment=_make_record("PAY-001", "PAYMENT", 100000),
            expected_amount=80000,
            actual_amount=75000,
            difference=5000,
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 2000),
            ],
            taxes=[
                _make_record("TAX-001", "TAX", 1800),
            ],
            adjustments=[
                _make_record("ADJ-001", "ADJUSTMENT", 3000),
            ],
        )
        G = builder.build(pkg)

        total = EvidenceGraphBuilder.get_total_contribution(G)
        # Payment(100000) + Refund(-5000) + Fee(-2000) + Tax(-1800) + Adj(+3000) = 94200
        assert total == 94200

    def test_complex_exception_explainers(self, builder):
        """Test that explainers are correctly identified."""
        pkg = _make_package(
            expected_amount=80000,
            actual_amount=75000,
            difference=5000,
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 75000),
            ],
            refunds=[
                _make_record("REF-001", "REFUND", 5000),
            ],
            fees=[
                _make_record("FEE-001", "FEE", 2000),
            ],
        )
        G = builder.build(pkg)

        explainers = EvidenceGraphBuilder.get_exception_explainers(G, "EXC-001")
        assert len(explainers) == 3  # settlement + refund + fee
        types = {e["node_type"] for e in explainers}
        assert NODE_SETTLEMENT in types
        assert NODE_REFUND in types
        assert NODE_FEE in types

    def test_complex_node_count(self, builder):
        """Test exact node count for complex case."""
        pkg = _make_package(
            settlements=[_make_record("SET-001", "SETTLEMENT", 75000)],
            refunds=[_make_record("REF-001", "REFUND", 5000)],
            fees=[_make_record("FEE-001", "FEE", 2000)],
            taxes=[_make_record("TAX-001", "TAX", 1800)],
            adjustments=[_make_record("ADJ-001", "ADJUSTMENT", 3000)],
        )
        G = builder.build(pkg)

        # exception + payment + settlement + refund + fee + tax + adjustment = 7
        assert G.number_of_nodes() == 7

    def test_complex_edge_count(self, builder):
        """Test edge count for complex case."""
        pkg = _make_package(
            settlements=[_make_record("SET-001", "SETTLEMENT", 75000)],
            refunds=[_make_record("REF-001", "REFUND", 5000)],
            fees=[_make_record("FEE-001", "FEE", 2000)],
            taxes=[_make_record("TAX-001", "TAX", 1800)],
            adjustments=[_make_record("ADJ-001", "ADJUSTMENT", 3000)],
        )
        G = builder.build(pkg)

        # Edges:
        # exception → payment (1)
        # payment → settlement, refund, fee, tax, adjustment (5)
        # settlement, refund, fee, tax, adjustment → exception (5)
        # Total: 11
        assert G.number_of_edges() == 11


# ─────────────────────────────────────────────────────────────────────────────
# Missing Record Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingRecord:
    """Tests for missing evidence graph representation."""

    def test_missing_settlement_node(self, builder):
        """Test that missing settlement creates a MISSING node."""
        pkg = _make_package(
            actual_amount=0,
            difference=100000,
            exception_type="MISSING_RECORD",
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement record found for payment PAY-001",
                ),
            ],
        )
        G = builder.build(pkg)

        missing_nodes = EvidenceGraphBuilder.get_missing_evidence(G)
        assert len(missing_nodes) == 1
        assert missing_nodes[0]["entity_type"] == "SETTLEMENT"
        assert missing_nodes[0]["expected"] is True

    def test_missing_node_no_fake_record(self, builder):
        """Test that missing evidence does not create a fake financial record."""
        pkg = _make_package(
            exception_type="MISSING_RECORD",
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        G = builder.build(pkg)

        # No settlement node should exist
        settlement_nodes = EvidenceGraphBuilder.get_nodes_by_type(G, NODE_SETTLEMENT)
        assert len(settlement_nodes) == 0

    def test_missing_evidence_edge_from_exception(self, builder):
        """Test that exception has IS_MISSING edge to missing node."""
        pkg = _make_package(
            exception_type="MISSING_RECORD",
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        G = builder.build(pkg)

        missing_nodes = EvidenceGraphBuilder.get_missing_evidence(G)
        missing_node_id = missing_nodes[0]["node_id"]
        assert G.has_edge("EXC-001", missing_node_id)
        assert G.edges["EXC-001", missing_node_id]["edge_type"] == EDGE_IS_MISSING

    def test_missing_evidence_edge_from_payment(self, builder):
        """Test that payment has IS_MISSING edge to missing node."""
        pkg = _make_package(
            exception_type="MISSING_RECORD",
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        G = builder.build(pkg)

        missing_nodes = EvidenceGraphBuilder.get_missing_evidence(G)
        missing_node_id = missing_nodes[0]["node_id"]
        assert G.has_edge("PAY-001", missing_node_id)

    def test_missing_node_has_zero_contribution(self, builder):
        """Test that missing node has zero financial contribution."""
        pkg = _make_package(
            exception_type="MISSING_RECORD",
            missing_evidence=[
                MissingEvidence(
                    entity_type="SETTLEMENT",
                    expected=True,
                    reason="No settlement found",
                ),
            ],
        )
        G = builder.build(pkg)

        missing_nodes = EvidenceGraphBuilder.get_missing_evidence(G)
        missing_node_id = missing_nodes[0]["node_id"]
        assert G.nodes[missing_node_id]["contribution_to_expected"] == 0
        assert G.nodes[missing_node_id]["amount"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Settlement Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateSettlement:
    """Tests for duplicate settlement graph representation."""

    def test_duplicate_settlements_in_graph(self, builder):
        """Test that multiple settlements are represented."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=200000,
            difference=-100000,
            exception_type="DUPLICATE",
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 100000),
                _make_record("SET-002", "SETTLEMENT", 100000),
            ],
        )
        G = builder.build(pkg)

        settlement_nodes = EvidenceGraphBuilder.get_nodes_by_type(G, NODE_SETTLEMENT)
        assert len(settlement_nodes) == 2

    def test_duplicate_conflict_annotation(self, builder):
        """Test that duplicate conflict is annotated on nodes."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=200000,
            difference=-100000,
            exception_type="DUPLICATE",
            settlements=[
                _make_record("SET-001", "SETTLEMENT", 100000),
                _make_record("SET-002", "SETTLEMENT", 100000),
            ],
            conflicts=[
                StructuralConflict(
                    conflict_type="DUPLICATE_SETTLEMENT_ID",
                    description="Duplicate settlement with identical amount",
                    affected_records=["SET-001", "SET-002"],
                ),
            ],
        )
        G = builder.build(pkg)

        conflicts = EvidenceGraphBuilder.get_conflicts(G)
        assert len(conflicts) == 2
        assert all(c["conflict_type"] == "DUPLICATE_SETTLEMENT_ID" for c in conflicts)


# ─────────────────────────────────────────────────────────────────────────────
# Unknown Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownCase:
    """Tests for unknown exception graph representation."""

    def test_unknown_case_graph(self, builder):
        """Test that unknown case still produces a valid graph."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=92000,
            difference=8000,
            exception_type="UNKNOWN",
            fees=[
                _make_record("FEE-001", "FEE", 2000),
            ],
        )
        G = builder.build(pkg)

        assert G.has_node("EXC-001")
        assert G.has_node("PAY-001")
        assert G.has_node("FEE-001")
        assert G.nodes["EXC-001"]["exception_type"] == "UNKNOWN"

    def test_unknown_does_not_invent_explanation(self, builder):
        """Test that unknown case graph doesn't claim to explain the discrepancy."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=92000,
            difference=8000,
            exception_type="UNKNOWN",
        )
        G = builder.build(pkg)

        explainers = EvidenceGraphBuilder.get_exception_explainers(G, "EXC-001")
        # No explainers for unknown case — graph doesn't invent explanation
        assert len(explainers) == 0

    def test_unknown_with_records_has_explainers(self, builder):
        """Test that unknown case with records has explainers."""
        pkg = _make_package(
            expected_amount=100000,
            actual_amount=92000,
            difference=8000,
            exception_type="UNKNOWN",
            fees=[_make_record("FEE-001", "FEE", 2000)],
        )
        G = builder.build(pkg)

        explainers = EvidenceGraphBuilder.get_exception_explainers(G, "EXC-001")
        assert len(explainers) == 1
        assert explainers[0]["node_type"] == NODE_FEE


# ─────────────────────────────────────────────────────────────────────────────
# Merchant Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMerchantNode:
    """Tests for merchant node in graph."""

    def test_merchant_node_created(self, builder):
        """Test that merchant node is created when merchant_id present."""
        pkg = _make_package(merchant_id="MER-001")
        G = builder.build(pkg)

        assert G.has_node("MER-001")
        assert G.nodes["MER-001"]["node_type"] == NODE_MERCHANT

    def test_merchant_to_payment_edge(self, builder):
        """Test edge from merchant to payment."""
        pkg = _make_package(merchant_id="MER-001")
        G = builder.build(pkg)

        assert G.has_edge("MER-001", "PAY-001")
        assert G.edges["MER-001", "PAY-001"]["edge_type"] == EDGE_OWNS

    def test_no_merchant_when_none(self, builder):
        """Test that no merchant node when merchant_id is None."""
        pkg = _make_package(merchant_id=None)
        G = builder.build(pkg)

        merchant_nodes = EvidenceGraphBuilder.get_nodes_by_type(G, NODE_MERCHANT)
        assert len(merchant_nodes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Traceability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTraceability:
    """Tests for graph traceability."""

    def test_node_has_entity_id(self, builder):
        """Test that all financial nodes have entity_id."""
        pkg = _make_package(
            refunds=[_make_record("REF-001", "REFUND", 5000)],
            fees=[_make_record("FEE-001", "FEE", 2000)],
        )
        G = builder.build(pkg)

        for node, attrs in G.nodes(data=True):
            assert "entity_id" in attrs
            assert "node_type" in attrs

    def test_edge_has_relationship(self, builder):
        """Test that all edges have relationship metadata."""
        pkg = _make_package(
            refunds=[_make_record("REF-001", "REFUND", 5000)],
        )
        G = builder.build(pkg)

        for u, v, attrs in G.edges(data=True):
            assert "edge_type" in attrs
            assert "relationship" in attrs

    def test_exception_has_case_id(self, builder):
        """Test that exception node has case_id."""
        pkg = _make_package(case_id="CASE-999")
        G = builder.build(pkg)

        assert G.nodes["EXC-001"]["case_id"] == "CASE-999"

    def test_financial_nodes_have_case_id(self, builder):
        """Test that financial nodes have case_id."""
        pkg = _make_package(
            refunds=[_make_record("REF-001", "REFUND", 5000)],
        )
        G = builder.build(pkg)

        assert G.nodes["REF-001"]["case_id"] == "CASE-001"


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Separation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruthSeparation:
    """Verify that graph builder does not reference ground truth."""

    GROUND_TRUTH_TERMS = [
        "ground_truth",
        "true_exception_type",
        "true_resolution",
    ]

    def test_builder_code_no_ground_truth(self):
        """Test that graph builder source has no ground truth references."""
        import inspect
        from app.services.evidence_graph import EvidenceGraphBuilder

        source = inspect.getsource(EvidenceGraphBuilder)
        for term in self.GROUND_TRUTH_TERMS:
            assert term not in source, f"Ground truth reference found: {term}"

    def test_graph_nodes_no_ground_truth(self, builder):
        """Test that graph nodes have no ground truth attributes."""
        pkg = _make_package(
            exception_type="FEE_DIFFERENCE",
            refunds=[_make_record("REF-001", "REFUND", 5000)],
        )
        G = builder.build(pkg)

        for node, attrs in G.nodes(data=True):
            assert "true_exception_type" not in attrs
            assert "true_resolution" not in attrs
            assert "resolvable" not in attrs
            assert "risk_category" not in attrs


# ─────────────────────────────────────────────────────────────────────────────
# Financial Contribution Audit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialContributionAudit:
    """Tests for financial contribution calculation."""

    def test_contribution_matches_formula(self, builder):
        """Test that contribution calculation matches expected formula."""
        pkg = _make_package(
            payment=_make_record("PAY-001", "PAYMENT", 100000),
            expected_amount=80000,
            settlements=[_make_record("SET-001", "SETTLEMENT", 75000)],
            refunds=[_make_record("REF-001", "REFUND", 5000)],
            fees=[_make_record("FEE-001", "FEE", 2000)],
            taxes=[_make_record("TAX-001", "TAX", 1800)],
            adjustments=[_make_record("ADJ-001", "ADJUSTMENT", 3000)],
        )
        G = builder.build(pkg)

        # Payment: +100000
        # Refund: -5000
        # Fee: -2000
        # Tax: -1800
        # Adjustment: +3000
        # Settlement: 0 (actual, not expected)
        # Total: 100000 - 5000 - 2000 - 1800 + 3000 = 94200
        total = EvidenceGraphBuilder.get_total_contribution(G)
        assert total == 94200

    def test_empty_graph_zero_contribution(self, builder):
        """Test that empty graph has zero contribution."""
        pkg = _make_package()
        G = builder.build(pkg)

        total = EvidenceGraphBuilder.get_total_contribution(G)
        assert total == 100000  # Just the payment

    def test_all_deductions(self, builder):
        """Test graph with only deductions."""
        pkg = _make_package(
            refunds=[_make_record("REF-001", "REFUND", 10000)],
            fees=[_make_record("FEE-001", "FEE", 5000)],
            taxes=[_make_record("TAX-001", "TAX", 3000)],
        )
        G = builder.build(pkg)

        total = EvidenceGraphBuilder.get_total_contribution(G)
        # Payment(100000) - Refund(10000) - Fee(5000) - Tax(3000) = 82000
        assert total == 82000
