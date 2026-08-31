"""
Investigation nodes for Razorpay CloseLoop Phase 7C.

Nodes that orchestrate the investigation pipeline:
- Gather Evidence (Phase 3)
- Build Evidence Graph (Phase 3)
- Classify Exception (Phase 4)
- Retrieve Similar Cases (Phase 4)

Each node delegates to existing services.
Each node has structured failure handling.
"""

import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.agent_state import AgentState, WorkflowStatus


# ─────────────────────────────────────────────────────────────────────────────
# Node Result Helper
# ─────────────────────────────────────────────────────────────────────────────


def _record_node(
    state: AgentState,
    node_name: str,
    success: bool,
    error: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Record node execution and return state updates."""
    elapsed_ms = None
    if start_time:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    log_entry = {
        "node": node_name,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms else None,
        "error": error,
    }

    metadata = state.metadata.model_dump()
    metadata["last_updated_at"] = datetime.utcnow().isoformat()
    metadata["nodes_executed"] = list(state.metadata.nodes_executed) + [node_name]
    metadata["execution_log"] = list(state.metadata.execution_log) + [log_entry]

    if error:
        metadata["errors"] = list(state.metadata.errors) + [error]

    return {"metadata": metadata}


def _fail_node(
    state: AgentState,
    node_name: str,
    error_msg: str,
    start_time: float,
) -> Dict[str, Any]:
    """Create failure state update."""
    updates = _record_node(state, node_name, success=False, error=error_msg, start_time=start_time)
    updates["metadata"]["workflow_status"] = WorkflowStatus.FAILED.value
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Gather Evidence Node
# ─────────────────────────────────────────────────────────────────────────────


def gather_evidence(state: AgentState) -> Dict[str, Any]:
    """Gather financial evidence for the exception.

    Delegates to Phase 3 EvidenceRetrievalService.

    Stores:
    - evidence package
    - evidence coverage
    - evidence consistency
    """
    start_time = time.perf_counter()
    node_name = "gather_evidence"

    exception_id = state.metadata.exception_id
    if not exception_id:
        return _fail_node(state, node_name, "No exception ID", start_time)

    try:
        # Simulate evidence retrieval
        # In production: EvidenceRetrievalService.retrieve_by_exception_id(exception_id)
        evidence = _simulate_evidence_retrieval(exception_id)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["evidence_package"] = evidence
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Evidence retrieval failed: {str(e)}", start_time)


def _simulate_evidence_retrieval(exception_id: str) -> Dict[str, Any]:
    """Simulate evidence retrieval."""
    return {
        "exception_id": exception_id,
        "payment": {"payment_id": "PAY-001", "amount": 100000, "status": "CAPTURED"},
        "settlements": [{"settlement_id": "SET-001", "amount": 97000, "status": "SETTLED"}],
        "refunds": [],
        "fees": [{"fee_id": "FEE-001", "amount": 3000, "fee_type": "TDR"}],
        "taxes": [],
        "adjustments": [],
        "evidence_coverage": 0.95,
        "evidence_consistency": 0.90,
        "supporting_evidence_count": 2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build Evidence Graph Node
# ─────────────────────────────────────────────────────────────────────────────


def build_evidence_graph(state: AgentState) -> Dict[str, Any]:
    """Build NetworkX evidence graph from evidence package.

    Delegates to Phase 3 EvidenceGraphBuilder.

    Stores:
    - serialized evidence graph
    """
    start_time = time.perf_counter()
    node_name = "build_evidence_graph"

    evidence_package = state.evidence_package
    if not evidence_package:
        return _fail_node(state, node_name, "No evidence package available", start_time)

    try:
        # Simulate graph building
        # In production: EvidenceGraphBuilder().build(evidence_package)
        graph_data = _simulate_graph_building(evidence_package)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["evidence_graph"] = graph_data
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Graph building failed: {str(e)}", start_time)


def _simulate_graph_building(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate evidence graph building."""
    nodes = []
    edges = []

    # Add payment node
    if evidence.get("payment"):
        nodes.append({"id": evidence["payment"]["payment_id"], "type": "PAYMENT", "amount": evidence["payment"]["amount"]})

    # Add settlement nodes
    for s in evidence.get("settlements", []):
        nodes.append({"id": s["settlement_id"], "type": "SETTLEMENT", "amount": s["amount"]})
        edges.append({"source": evidence["payment"]["payment_id"], "target": s["settlement_id"], "relationship": "HAS_SETTLEMENT"})

    # Add fee nodes
    for f in evidence.get("fees", []):
        nodes.append({"id": f["fee_id"], "type": "FEE", "amount": f["amount"]})
        edges.append({"source": evidence["payment"]["payment_id"], "target": f["fee_id"], "relationship": "HAS_FEE"})

    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}


# ─────────────────────────────────────────────────────────────────────────────
# Classify Exception Node
# ─────────────────────────────────────────────────────────────────────────────


def classify_exception(state: AgentState) -> Dict[str, Any]:
    """Classify the exception type.

    Delegates to Phase 4 classification service.

    Stores:
    - classification result
    - classification confidence
    """
    start_time = time.perf_counter()
    node_name = "classify_exception"

    exception_id = state.metadata.exception_id
    evidence_package = state.evidence_package

    if not evidence_package:
        return _fail_node(state, node_name, "No evidence package for classification", start_time)

    try:
        # Simulate classification
        # In production: ExceptionClassifierService.classify(features)
        classification = _simulate_classification(exception_id, evidence_package)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["classification"] = classification
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Classification failed: {str(e)}", start_time)


def _simulate_classification(exception_id: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate exception classification."""
    fees = evidence.get("fees", [])
    refunds = evidence.get("refunds", [])

    if fees:
        exc_type = "FEE_DIFFERENCE"
        confidence = 0.95
    elif refunds:
        exc_type = "REFUND_ADJUSTMENT"
        confidence = 0.90
    else:
        exc_type = "EXACT_MATCH"
        confidence = 0.85

    return {
        "exception_id": exception_id,
        "exception_type": exc_type,
        "confidence": confidence,
        "classification_source": "deterministic",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve Similar Cases Node
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_similar_cases(state: AgentState) -> Dict[str, Any]:
    """Retrieve historically similar cases.

    Delegates to Phase 4 similarity service.

    Stores:
    - similar cases
    - similarity scores
    - historical resolutions
    """
    start_time = time.perf_counter()
    node_name = "retrieve_similar_cases"

    if not state.classification:
        return _fail_node(state, node_name, "No classification available for similarity search", start_time)

    try:
        # Simulate similarity search
        # In production: SimilarityService.search(query_case, top_k=5)
        similar = _simulate_similarity_search(state.classification)

        updates = _record_node(state, node_name, success=True, start_time=start_time)
        updates["similar_cases"] = similar
        updates["metadata"]["current_node"] = node_name
        return updates

    except Exception as e:
        return _fail_node(state, node_name, f"Similarity search failed: {str(e)}", start_time)


def _simulate_similarity_search(classification: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate similarity search."""
    return {
        "query_exception_type": classification.get("exception_type"),
        "total_indexed": 150,
        "top_k": 5,
        "similar_cases": [
            {"case_id": "CASE-042", "similarity_score": 0.92, "resolution": "FEE_ADJUSTMENT", "outcome": "SUCCESSFUL"},
            {"case_id": "CASE-017", "similarity_score": 0.88, "resolution": "FEE_ADJUSTMENT", "outcome": "SUCCESSFUL"},
            {"case_id": "CASE-098", "similarity_score": 0.85, "resolution": "NO_ACTION", "outcome": "SUCCESSFUL"},
        ],
        "best_similarity_score": 0.92,
        "embedding_model": "all-MiniLM-L6-v2",
    }
